"""Dev server for the verifier UI.

Serves the repo's static files (verifier/, data/, tests/) and provides:

  POST /api/lookup    — same-origin proxy to request-o-matic /request
                        (request-o-matic doesn't emit CORS headers, and
                        a proxy is simpler than configuring CORS on a
                        third-party service).
  POST /api/save      — persist a verifier UI session: writes
                        <stem>.verified.json and <stem>.corrections.json
                        into data/verifier/, and (when the bundle carries
                        a `pdf_path`/`page_number` pair) updates
                        `jobs.db` via `JobStore.mark_verified`. Honors
                        an optional `status` field with preservation
                        semantics — see the handler docstring.
  GET  /api/bundles   — list every bundle in data/verifier/ with its
                        verification state (incomplete / partial /
                        complete) for the index page and Prev/Next nav.

Auth (when WXYC_OIDC_CLIENT_ID is set):

  GET  /auth/login     — kick off the OIDC code+PKCE dance against
                         api.wxyc.org/auth.
  GET  /auth/callback  — exchange the code, seal the reviewer into a
                         12-hour signed session cookie, redirect home.
  POST /auth/logout    — clear the session cookie (local logout only —
                         api.wxyc.org session is unaffected).
  GET  /auth/me        — return the currently logged-in ReviewerSession.

The middleware gate is tri-modal:
  * OIDC session cookie when WXYC_OIDC_CLIENT_ID is set.
  * HTTP Basic Auth when VERIFIER_PASSWORD is set.
  * No gate otherwise (local-dev default).
Modes are mutually exclusive; OIDC wins when both are configured.

Run:

    .venv/bin/python verifier/serve.py

Then open http://localhost:8765/verifier/?bundle=/data/verifier/<stem>.bundle.json
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import urllib.parse
from base64 import b64decode
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import uvicorn
from authlib.jose.errors import JoseError
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import core.auth as auth_mod
from core.auth import (
    COOKIE_NAME,
    ONE_SHOT_TTL,
    SESSION_TTL,
    ReviewerSession,
)
from core.jobs import JobStore
from core.schema import PageResult, VerifiedBy

REQUEST_O_MATIC_URL = os.environ.get(
    "REQUEST_O_MATIC_URL",
    "https://request-o-matic-production.up.railway.app/api/v1/request",
)
# Railway sets $PORT; honor it first, then VERIFIER_PORT, then 8765 for local.
PORT = int(os.environ.get("PORT") or os.environ.get("VERIFIER_PORT") or "8765")
# Default to loopback locally; Railway sets VERIFIER_HOST=0.0.0.0.
HOST = os.environ.get("VERIFIER_HOST", "127.0.0.1")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))
VERIFIER_DIR = DATA_ROOT / "verifier"
JOBS_DB_PATH = DATA_ROOT / "jobs.db"

# When VERIFIER_PASSWORD is set, every request goes through HTTP Basic Auth.
# Unset → no auth, matching the local-dev default the README documents.
# VERIFIER_USER defaults to a placeholder so the volunteer only has to
# remember a password.
VERIFIER_PASSWORD = os.environ.get("VERIFIER_PASSWORD")
VERIFIER_USER = os.environ.get("VERIFIER_USER", "verifier")

# OIDC takes precedence over BasicAuth when both are configured. A
# WXYC_OIDC_CLIENT_ID with an empty value (e.g. someone unsetting OIDC
# by clearing the value rather than removing the line) is treated as
# unset so the env-var-gate semantics match the BasicAuth path.
OIDC_ENABLED = bool(os.environ.get("WXYC_OIDC_CLIENT_ID"))
# `FLOWSHEET_PUBLIC_URL` decides whether session + one-shot cookies
# carry `Secure`. Local dev runs plain HTTP; production runs HTTPS; the
# flag tracks the URL scheme without a separate `IS_PRODUCTION` env.
_COOKIE_SECURE = os.environ.get("FLOWSHEET_PUBLIC_URL", "").startswith("https")


class _BasicAuthMiddleware(BaseHTTPMiddleware):
    """Gates every request with HTTP Basic Auth when a password is set.

    Both static mounts (/verifier/, /data/) and the /api/* endpoints are
    behind the gate — the verifier UI loads the bundle JSON via the same
    static mount, so any leak of /data/ leaks the corpus.

    `secrets.compare_digest` runs the credential check in constant time
    relative to length, which is the only reasonable thing to do with a
    shared bearer-style password.
    """

    def __init__(self, app: ASGIApp, *, user: str, password: str) -> None:
        super().__init__(app)
        self._user = user
        self._password = password

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                decoded = b64decode(auth[6:]).decode("utf-8")
            except Exception:  # noqa: BLE001
                decoded = ""
            user, _, password = decoded.partition(":")
            if secrets.compare_digest(user, self._user) and secrets.compare_digest(
                password, self._password
            ):
                return await call_next(request)
        return JSONResponse(
            {"detail": "authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="WXYC Verifier"'},
        )


# Routes that must answer unauthenticated so the auth dance + Railway's
# healthcheck can complete. `PUBLIC_PATHS` lists specific public routes;
# do NOT widen the `/auth/` prefix to mean anything else — `/auth/me` is
# deliberately gated, and a future `/auth/admin` would be too. Add new
# public endpoints to PUBLIC_PATHS by name.
#
# `/api/version` is load-bearing: Railway uses it as the healthcheck. If
# this set ever drops it, every deploy will roll back because the gate
# redirects the healthcheck request to /auth/login.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {"/auth/login", "/auth/callback", "/auth/logout", "/api/version"}
)


class _SessionCookieMiddleware(BaseHTTPMiddleware):
    """Gate all non-public paths on the signed session cookie.

    Reads `flowsheet_session`, verifies it via `core.auth.decode_session`,
    and stashes the resulting `ReviewerSession` on `request.state.reviewer`
    so downstream handlers (and the `get_reviewer` dependency) can read
    it without re-parsing the cookie.

    A missing or invalid session redirects to `/auth/login?return_to=<path>`
    for HTML requests, and returns a 401 JSON for AJAX requests (the SPA
    inspects the response to know to redirect client-side).
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS:
            request.state.reviewer = None
            return await call_next(request)

        raw = request.cookies.get(COOKIE_NAME, "")
        reviewer = auth_mod.decode_session(raw)
        if reviewer is None:
            # AJAX / fetch — return 401 JSON so the SPA can redirect via
            # location.href. Lets the verifier UI continue running its
            # JS error handlers instead of getting an opaque 302.
            if "application/json" in request.headers.get("accept", "").lower():
                return JSONResponse({"detail": "authentication required"}, status_code=401)
            # HTML / browser nav — server-side 302 to /auth/login with
            # a signed return_to. The login handler will validate it
            # against the /verifier prefix allowlist.
            return_to = path
            if request.url.query:
                return_to += f"?{request.url.query}"
            login_url = f"/auth/login?return_to={urllib.parse.quote(return_to)}"
            return RedirectResponse(login_url, status_code=302)

        request.state.reviewer = reviewer
        return await call_next(request)


app = FastAPI(docs_url=None, redoc_url=None)
# Tri-modal install. OIDC wins over BasicAuth so that the BasicAuth env
# var can stay set during the transition period without competing with
# the session gate.
if OIDC_ENABLED:
    app.add_middleware(_SessionCookieMiddleware)
elif VERIFIER_PASSWORD:
    app.add_middleware(_BasicAuthMiddleware, user=VERIFIER_USER, password=VERIFIER_PASSWORD)


def get_reviewer(request: Request) -> ReviewerSession | None:
    """Read the authenticated reviewer off `request.state`, set by the
    session middleware.

    Returns None in two cases:
      * Non-OIDC deployments (BasicAuth or no-auth) — there is no
        ReviewerSession to read.
      * Public routes the middleware bypassed — they don't need an
        identity to answer.

    Handlers that *require* identity (`/auth/me`, `/api/save` when we
    want to credit the reviewer) raise 401 themselves on None.
    """
    return getattr(request.state, "reviewer", None)


def _set_cookie(
    response: Response,
    key: str,
    value: str,
    *,
    max_age: int,
    path: str = "/",
) -> None:
    """Set a cookie with the project's standard flags.

    HttpOnly + SameSite=lax + (Secure when serving over HTTPS) so the
    OIDC redirect-back can carry one-shot cookies but no other site can
    read the session cookie. SameSite=strict would drop the cookies on
    the cross-site redirect from api.wxyc.org/auth.
    """
    response.set_cookie(
        key=key,
        value=value,
        max_age=max_age,
        httponly=True,
        secure=_COOKIE_SECURE,
        samesite="lax",
        path=path,
    )


def _validate_return_to(raw: str) -> str:
    """Allowlist `return_to` to a relative path under `/verifier`.

    Defuses the open-redirect risk on `/auth/login?return_to=https://evil`
    — an attacker who tricks a user into clicking that link would
    otherwise see the user land on their site post-login.

    The allowlist requires a path-separator-or-end boundary after the
    prefix so `/verifier` itself, `/verifier/?bundle=…`, and `/verifier/…`
    are accepted while a future sibling route like `/verifierx-admin`
    or `/verifierdata` is not silently treated as in-scope. A plain
    `startswith("/verifier")` would accept any of those.
    """
    if raw == "/verifier" or raw.startswith("/verifier/") or raw.startswith("/verifier?"):
        return raw
    return "/verifier/"


# Cache of jobs.db paths whose `init()` migrations have already been
# applied this process. Avoids re-running PRAGMAs + ALTER-TABLE checks
# on every /api/save when jobs.db is unchanged across saves. Tests that
# swap DATA_ROOT via importlib.reload get a fresh empty set.
_initialized_jobs_dbs: set[Path] = set()


async def _open_jobs_store() -> JobStore | None:
    """Return an initialized `JobStore` if `jobs.db` is on disk, else None.

    Runs `JobStore.init()` once per (db_path, process) pair — subsequent
    calls hit the in-memory cache and skip the schema-migration round
    trip. Re-checks `is_file()` every call so a DB created after the
    server starts (e.g., user runs the pipeline mid-session) is picked
    up without a server restart.
    """
    if not JOBS_DB_PATH.is_file():
        return None
    store = JobStore(JOBS_DB_PATH)
    if JOBS_DB_PATH not in _initialized_jobs_dbs:
        await store.init()
        _initialized_jobs_dbs.add(JOBS_DB_PATH)
    return store


def _existing_verified_by(path: Path) -> VerifiedBy | None:
    """Read `verified_by` from an existing on-disk verified.json, or
    None if the file doesn't exist / isn't parseable / has no block.

    Used by `/api/save` in BasicAuth and no-auth deployments to preserve
    a previously-recorded reviewer identity across a re-save where the
    server has no authenticated reviewer to assert — the data-safety
    rule says we don't clobber successfully collected data without
    explicit user direction.
    """
    if not path.is_file():
        return None
    try:
        return PageResult.model_validate_json(path.read_text()).verified_by
    except Exception:  # noqa: BLE001
        # Malformed JSON or a shape that doesn't validate isn't fatal
        # — the save proceeds, just without preservation. The save's
        # downstream Pydantic validation will produce a clean file.
        return None


def _safe_stem(stem: str) -> str:
    """Reject anything that could escape `data/verifier/` via path traversal.

    Bundle stems come from image filenames (e.g. `1990-04apr0106-page25`)
    and are unlikely to contain `/`, but a hostile or malformed POST
    shouldn't let the verifier server write outside the verifier dir.
    Whitespace-only stems are also refused — they'd produce files named
    ` .verified.json` which are confusing and almost certainly a bug.
    """
    if not stem or not stem.strip() or "/" in stem or "\\" in stem or stem.startswith(".."):
        raise HTTPException(status_code=400, detail=f"invalid stem: {stem!r}")
    return stem


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via a `.tmp` sibling and `os.replace`.

    Two writes on the save path (verified + corrections) — atomic
    individual writes mean a partially-failed save leaves either both
    files at their pre-save state OR both at the new state, never a
    half-updated state where verified.json reflects the edit but
    corrections.json doesn't.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


# -- auth routes -----------------------------------------------------------


@app.get("/auth/login")
async def auth_login(return_to: str = "/verifier/") -> Response:
    """Kick off the OIDC code+PKCE dance.

    Validates `return_to`, builds the authorize URL, stashes
    `state` / `code_verifier` / `return_to` in three signed one-shot
    cookies (so the callback can validate them) and 302s to the auth
    server. A transient auth-server failure surfaces as 503.
    """
    return_to = _validate_return_to(return_to)
    try:
        url, state, code_verifier = await auth_mod.build_authorize_url()
    except httpx.HTTPError as exc:
        raise HTTPException(503, detail="auth server unavailable") from exc
    except RuntimeError as exc:
        # `RuntimeError` propagates from the env-var accessors in
        # `core.auth` when a required var (WXYC_AUTH_ISSUER, etc.) is
        # unset. `core.auth.decode_session` already catches this for
        # the request-gating path; mirror the protection here so a
        # misconfigured deploy with WXYC_OIDC_CLIENT_ID set but
        # WXYC_AUTH_ISSUER missing returns 503 (matching the auth-
        # server-unavailable shape) rather than 500 with the env-var
        # name in the traceback.
        raise HTTPException(503, detail="auth not configured") from exc

    response = RedirectResponse(url, status_code=302)
    one_shot_age = int(ONE_SHOT_TTL.total_seconds())
    # path="/auth" limits the one-shot cookies to the OIDC dance —
    # they aren't sent on /verifier or /api/* requests.
    _set_cookie(
        response, "oidc_state", auth_mod.sign_one_shot(state), max_age=one_shot_age, path="/auth"
    )
    _set_cookie(
        response,
        "oidc_verifier",
        auth_mod.sign_one_shot(code_verifier),
        max_age=one_shot_age,
        path="/auth",
    )
    _set_cookie(
        response,
        "oidc_return_to",
        auth_mod.sign_one_shot(return_to),
        max_age=one_shot_age,
        path="/auth",
    )
    return response


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str = "", state: str = "") -> Response:
    """Validate the redirect, exchange the code, seal the session.

    Failure paths:
      * Missing / tampered / expired one-shot cookies → 400.
      * `state` mismatch → 400 (CSRF defense).
      * Auth server unreachable → 503.
      * id_token signature or aud/iss mismatch → 400.

    On success, sets `flowsheet_session` (12h) and 302s to the
    validated `return_to`.
    """
    if not code or not state:
        raise HTTPException(400, detail="missing code or state")

    state_cookie = request.cookies.get("oidc_state", "")
    verifier_cookie = request.cookies.get("oidc_verifier", "")
    return_to_cookie = request.cookies.get("oidc_return_to", "")

    one_shot_age = int(ONE_SHOT_TTL.total_seconds())
    try:
        signed_state = auth_mod.verify_one_shot(state_cookie, max_age=one_shot_age)
        code_verifier = auth_mod.verify_one_shot(verifier_cookie, max_age=one_shot_age)
    except BadSignature as exc:
        raise HTTPException(400, detail="invalid auth state") from exc
    except RuntimeError as exc:
        # See the matching catch in `auth_login` — env-var unset
        # surfaces as RuntimeError from `core.auth._session_secret()`
        # via the one-shot signer. Map to 503 ("auth not configured")
        # instead of letting it 500 with the env-var name in the
        # traceback.
        raise HTTPException(503, detail="auth not configured") from exc

    # `secrets.compare_digest` mirrors the precedent in
    # `_BasicAuthMiddleware` — constant-time string comparison so a
    # timing attack can't unmask the expected state value.
    if not secrets.compare_digest(state, signed_state):
        raise HTTPException(400, detail="invalid auth state")

    try:
        return_to = _validate_return_to(
            auth_mod.verify_one_shot(return_to_cookie, max_age=one_shot_age)
        )
    except BadSignature:
        # A missing/tampered return_to cookie isn't fatal — fall back
        # to the verifier index. The user gets logged in either way.
        return_to = "/verifier/"

    try:
        reviewer = await auth_mod.exchange_code(code=code, code_verifier=code_verifier)
    except httpx.HTTPError as exc:
        raise HTTPException(503, detail="auth server unavailable") from exc
    except RuntimeError as exc:
        # Same protection as the verify_one_shot block — env var unset
        # surfaces from core.auth's accessor functions as RuntimeError.
        raise HTTPException(503, detail="auth not configured") from exc
    except (JoseError, KeyError, ValueError) as exc:
        # `KeyError` covers a token response missing `id_token`;
        # `ValueError` covers any downstream parse error not caught
        # by the JOSE layer.
        raise HTTPException(400, detail="invalid id token") from exc

    response = RedirectResponse(return_to, status_code=302)
    _set_cookie(
        response,
        COOKIE_NAME,
        auth_mod.encode_session(reviewer),
        max_age=int(SESSION_TTL.total_seconds()),
        path="/",
    )
    # Clear the one-shots so a back-button retry can't replay the code
    # (which Better Auth would reject anyway, but defense in depth).
    for k in ("oidc_state", "oidc_verifier", "oidc_return_to"):
        response.delete_cookie(k, path="/auth")
    return response


@app.post("/auth/logout")
async def auth_logout() -> Response:
    """Clear the local session cookie.

    The Better Auth session at api.wxyc.org is unaffected — same
    behavior Wiki.js has today. Single-sign-out is out of scope until
    we have a per-session server-side store to revoke.
    """
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@app.get("/auth/me")
async def auth_me(
    reviewer: Annotated[ReviewerSession | None, Depends(get_reviewer)] = None,
) -> JSONResponse:
    """Return the currently logged-in reviewer.

    Used by the SPA on page load to render the reviewer's name in the
    header. 401 when no session — the SPA reads that as "redirect to
    /auth/login" (the middleware would have done this for HTML
    requests, but /auth/me is fetched as JSON).
    """
    if reviewer is None:
        raise HTTPException(401, detail="not authenticated")
    return JSONResponse(dataclasses.asdict(reviewer))


# -- existing API routes ---------------------------------------------------


@app.get("/api/version")
async def api_version() -> JSONResponse:
    """Version tag the JS can poll to detect a new deploy.

    The volunteer might keep the verifier open for hours. Without a way
    to notice a deploy mid-session, browser-cached JS keeps running stale
    code (cache headers force revalidation on reload, but not while a
    page is sitting still). The JS polls this endpoint every minute; if
    the tag changes from what it captured at load, it surfaces a
    "reload to get the latest" banner.

    Uses `app.js`'s mtime as the tag — cheap, accurate, no build step.
    """
    js_path = REPO_ROOT / "verifier" / "app.js"
    try:
        mtime = int(js_path.stat().st_mtime)
    except FileNotFoundError:
        mtime = 0
    return JSONResponse({"version": str(mtime)})


@app.post("/api/lookup")
async def lookup(request: Request) -> JSONResponse:
    """Same-origin proxy to request-o-matic's /request endpoint.

    Accepts the same shape request-o-matic does (`{"message": "..."}`) and
    forwards verbatim. Adds skip_slack=True so the lookup doesn't post to
    Slack. Returns request-o-matic's response unchanged.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    message = (payload or {}).get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="missing 'message' field")

    forward_body = {"message": message, "skip_slack": True}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(REQUEST_O_MATIC_URL, json=forward_body)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"upstream timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(r.json())


_VALID_STATUSES = frozenset({"draft", "complete"})


def _resolve_status(incoming: str | None, corrections_path: Path) -> str:
    """Compute the corrections.json `status` field for this save.

    Rules (matching the UI's Save + toggleable Mark-complete design):
      - Any explicit valid status from the client (`"complete"` or
        `"draft"`) wins outright. `"draft"` means the user clicked the
        toggle on an already-complete page to revert it.
      - When the client omits the status field (plain Save), preserve
        an existing `"complete"` on disk — a refine-in-place edit, not
        a status change.
      - Otherwise the page is a draft.

    Returns one of `"draft"` or `"complete"`.
    """
    if incoming in _VALID_STATUSES:
        return incoming
    if corrections_path.is_file():
        try:
            existing = json.loads(corrections_path.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
        if existing.get("status") == "complete":
            return "complete"
    return "draft"


@app.post("/api/save")
async def save(
    request: Request,
    reviewer: Annotated[ReviewerSession | None, Depends(get_reviewer)] = None,
) -> JSONResponse:
    """Persist a verifier UI session to disk and (optionally) `jobs.db`.

    Expected body shape:

        {
          "stem": "<page stem>",
          "pdf_path": "<rel path to PDF>" | null,
          "page_number": <int> | null,
          "status": "draft" | "complete" | null,
          "verified": { ...PageResult... },
          "corrections": { ...corrections... }
        }

    Writes:
      - `data/verifier/<stem>.verified.json`  (validated as PageResult)
      - `data/verifier/<stem>.corrections.json` (verbatim JSON +
        server-resolved `status` field)

    Status resolution: `"complete"` wins; otherwise an existing
    on-disk `"complete"` is preserved across plain Saves. See
    `_resolve_status` for the table.

    Server-as-authority on `verified_by`: a client-supplied
    `verified_by` block inside the `verified` payload is discarded —
    the server writes its own block from the authenticated reviewer's
    `ReviewerSession`. Without this rule, a malicious or buggy client
    could spoof another reviewer's identity into the saved file.

    If `pdf_path` and `page_number` are present, also calls
    `JobStore.mark_verified` to record the verification in `jobs.db`,
    including the reviewer's user_id as `reviewer_id`. For bundles
    without a job key (test fixtures), only the files are written and
    `db_updated` is False in the response.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    stem = _safe_stem(str(payload.get("stem", "")))
    verified = payload.get("verified")
    corrections = payload.get("corrections")
    pdf_path = payload.get("pdf_path")
    page_number = payload.get("page_number")
    incoming_status = payload.get("status")
    if verified is None or corrections is None:
        raise HTTPException(
            status_code=400,
            detail="body must include `verified` and `corrections` objects",
        )
    if incoming_status is not None and incoming_status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"status must be one of {sorted(_VALID_STATUSES)} or omitted, "
                f"got {incoming_status!r}"
            ),
        )

    # Validate verified against PageResult and keep the parsed model so
    # the on-disk JSON is the Pydantic-normalized round-trip rather than
    # whatever the client happened to send. This makes the verified file
    # a canonical representation that bit-matches what the pipeline
    # writes, regardless of any extra fields or non-canonical datetime
    # formats the client may have included.
    try:
        validated = PageResult.model_validate(verified)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"verified payload not a valid PageResult: {exc}"
        ) from exc

    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    verified_path = VERIFIER_DIR / f"{stem}.verified.json"
    corrections_path = VERIFIER_DIR / f"{stem}.corrections.json"

    # Server-authority overwrite, with prior-state preservation when
    # there's no reviewer.
    #
    # OIDC mode (reviewer present): discard whatever the client sent
    # for `verified_by` and write the authenticated reviewer's
    # identity. The client cannot spoof another reviewer.
    #
    # BasicAuth / no-auth mode (reviewer is None): there's no identity
    # to assert. Reading the prior on-disk `verified_by` and writing it
    # back preserves any historical OIDC-recorded reviewer record (data-
    # safety rule: never overwrite successfully collected data) while
    # still discarding any value the client tried to forge — the client
    # has no role in deciding `verified_by` in either mode.
    #
    # Pydantic's validation above runs first, so a malformed client-
    # supplied block returns 400 before this overwrite ever runs.
    if reviewer is not None:
        validated.verified_by = VerifiedBy(
            user_id=reviewer.user_id,
            username=reviewer.username,
            real_name=reviewer.real_name,
            dj_name=reviewer.dj_name,
            verified_at=datetime.now(UTC),
        )
    else:
        validated.verified_by = _existing_verified_by(verified_path)
    # Resolve status BEFORE we overwrite the existing corrections file,
    # so the preservation rule can see the prior on-disk value.
    resolved_status = _resolve_status(incoming_status, corrections_path)
    corrections_with_status = {"status": resolved_status, **(corrections or {})}
    _atomic_write_text(verified_path, validated.model_dump_json(indent=2))
    _atomic_write_text(corrections_path, json.dumps(corrections_with_status, indent=2))

    db_updated = False
    # `isinstance(x, int)` is True for `bool` in Python — explicitly reject
    # so a malformed `page_number: true` doesn't coerce to 1 and lookup
    # the wrong row.
    if pdf_path and isinstance(page_number, int) and not isinstance(page_number, bool):
        store = await _open_jobs_store()
        if store is not None:
            # Pair with the verified_by preservation rule above: when
            # there's no reviewer to credit, send the prior `verified_by`'s
            # user_id (read off disk) so the denormalized jobs.reviewer_id
            # column doesn't get clobbered to NULL on a non-OIDC re-save.
            reviewer_id_for_db: str | None
            if reviewer is not None:
                reviewer_id_for_db = reviewer.user_id
            else:
                preserved = validated.verified_by
                reviewer_id_for_db = preserved.user_id if preserved is not None else None
            db_updated = await store.mark_verified(
                pdf_path=pdf_path,
                page_number=page_number,
                verified_path=verified_path,
                corrections_path=corrections_path,
                reviewer_id=reviewer_id_for_db,
            )

    # Report paths relative to DATA_ROOT.parent so the UI displays
    # `data/verifier/<stem>.verified.json` whether `data/` lives under
    # the repo root (production) or a tmp dir (tests). Always succeeds —
    # both written paths are under DATA_ROOT, which is a child of
    # DATA_ROOT.parent by construction.
    return JSONResponse(
        {
            "verified_path": str(verified_path.relative_to(DATA_ROOT.parent)),
            "corrections_path": str(corrections_path.relative_to(DATA_ROOT.parent)),
            "db_updated": db_updated,
            "status": resolved_status,
        }
    )


@app.get("/api/bundles")
async def list_bundles() -> JSONResponse:
    """Enumerate every bundle in data/verifier/ with its verification state.

    The state machine: `incomplete` (no corrections.json yet), `partial`
    (corrections.json with `status="draft"`), `complete` (corrections.json
    with `status="complete"`). Legacy corrections.json files without a
    `status` field default to `partial` — they were saved before status
    tracking landed, so they're at least in-progress.

    Returns:
        {
          "bundles": [
            {
              "stem": "...",
              "page_date_raw": "..." | null,
              "pdf_path": "..." | null,
              "page_number": int | null,
              "status": "incomplete" | "partial" | "complete",
              "verified_at": "<ISO timestamp>" | null,
              "url": "/verifier/?bundle=/data/verifier/<stem>.bundle.json"
            },
            ...
          ]
        }

    Sorted by stem so navigation is deterministic and matches alphanumeric
    page order (which the WXYC stem format respects: `page05 < page25`).
    """
    if not VERIFIER_DIR.is_dir():
        return JSONResponse({"bundles": []})

    bundles_out: list[dict[str, object]] = []
    for bundle_path in sorted(VERIFIER_DIR.glob("*.bundle.json")):
        stem = bundle_path.name.removesuffix(".bundle.json")
        try:
            bundle = json.loads(bundle_path.read_text())
        except Exception:  # noqa: BLE001
            # A malformed bundle file shouldn't break the whole index —
            # surface it as incomplete with null metadata so it still
            # appears and the user can see something's off.
            bundle = {}

        corrections_path = VERIFIER_DIR / f"{stem}.corrections.json"
        verified_path = VERIFIER_DIR / f"{stem}.verified.json"
        status, verified_at = _bundle_state(corrections_path, verified_path)

        bundles_out.append(
            {
                "stem": stem,
                "page_date_raw": bundle.get("page_date_raw"),
                "pdf_path": bundle.get("pdf_path"),
                "page_number": bundle.get("page_number"),
                "status": status,
                "verified_at": verified_at,
                "url": f"/verifier/?bundle=/data/verifier/{stem}.bundle.json",
            }
        )

    return JSONResponse({"bundles": bundles_out})


def _bundle_state(corrections_path: Path, verified_path: Path) -> tuple[str, str | None]:
    """Derive (status, verified_at) for one bundle from disk.

    `verified_at` is the mtime of `<stem>.verified.json` if it exists;
    None otherwise. Reflects when the last Save / Mark-complete happened.
    """
    if not corrections_path.is_file():
        return ("incomplete", None)
    try:
        corrections = json.loads(corrections_path.read_text())
    except Exception:  # noqa: BLE001
        corrections = {}
    raw_status = corrections.get("status")
    if raw_status == "complete":
        status = "complete"
    else:
        # Anything else (`draft`, missing, malformed) is `partial`. Legacy
        # corrections.json files written before status tracking landed end
        # up here, which is the right semantic — they're saved, not done.
        status = "partial"

    verified_at: str | None = None
    if verified_path.is_file():
        verified_at = datetime.fromtimestamp(verified_path.stat().st_mtime, tz=UTC).isoformat()
    return (status, verified_at)


# Static mounts. Each top-level dir we need to serve gets its own mount so
# the URL structure mirrors the repo layout — `image_path` in bundles is
# relative (`../pages/.../page-NN.png`), and the UI fetches relative to
# the bundle URL. `/data` honors the same DATA_ROOT override that writes
# use so the read and write sides stay in sync when DATA_ROOT is moved.
class _NoCacheStaticFiles(StaticFiles):
    """`StaticFiles` that forces revalidation on every request.

    Default StaticFiles sends ETag + Last-Modified but no Cache-Control, so
    browsers can hold a heuristically-fresh stale copy of the SPA's HTML
    or JS for tens of minutes after a deploy. A volunteer who's mid-session
    keeps running the old code and looks like the new fix didn't ship.
    With `no-cache` the browser still gets to use its cached copy, but it
    MUST revalidate against the ETag, which always returns the latest
    after a deploy.
    """

    async def get_response(self, path: str, scope):  # type: ignore[no-untyped-def]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount(
    "/verifier",
    _NoCacheStaticFiles(directory=REPO_ROOT / "verifier", html=True),
    name="verifier",
)
app.mount("/data", StaticFiles(directory=DATA_ROOT, check_dir=False), name="data")
# tests/ is dev-only (golden fixtures). Mount it conditionally so the
# Docker runtime image — which excludes tests/ — doesn't crash at boot.
if (REPO_ROOT / "tests").is_dir():
    app.mount("/tests", StaticFiles(directory=REPO_ROOT / "tests"), name="tests")


def main() -> None:
    # Pass the app object directly rather than an import string — the script
    # is invoked as a file (verifier/serve.py), not as a package import.
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
