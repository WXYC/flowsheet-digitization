"""Tests for the OIDC auth routes + session middleware in verifier/serve.py.

Covers:
  * `/auth/login` builds the authorize URL, sets 3 signed one-shot cookies.
  * `/auth/callback` validates state + return_to, runs `exchange_code`,
    seals the session cookie, clears one-shots. Transport failures
    translate to 503; claim/signature failures translate to 400.
  * `/auth/logout` clears the session cookie.
  * `/auth/me` returns 401 without a session, 200 with one.
  * `/api/save` requires a session, overwrites client-supplied
    `verified_by` from the authenticated reviewer (server is authority),
    and threads `reviewer_id` into `jobs.db`.
  * Session middleware bypass for `_PUBLIC_PATHS` + 302 redirect for
    HTML + 401 JSON for AJAX.
  * Tri-modal middleware install: OIDC > BasicAuth > none.
  * Cookie `Secure` flag gated on `FLOWSHEET_PUBLIC_URL` scheme.

These tests use the existing `httpx.ASGITransport` pattern from
`test_verifier_serve.py` to exercise the FastAPI app in-process. The
auth-server calls are mocked at the `core.auth.build_authorize_url`
/ `core.auth.exchange_code` seam — no live OIDC needed.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import core.auth as auth_mod
from core.auth import ReviewerSession
from core.schema import QUADRANT_ORDER, PageResult, Quadrant
from tests.unit.conftest import _reset_serve_module_state

ISSUER = "https://auth.example/auth"
CLIENT_ID = "flowsheet-test"
CLIENT_SECRET = "test-client-secret"
PUBLIC_URL = "https://flowsheet.example"
SESSION_SECRET = "x" * 64


def _set_oidc_env(monkeypatch: pytest.MonkeyPatch, *, public_url: str = PUBLIC_URL) -> None:
    monkeypatch.setenv("WXYC_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("WXYC_SESSION_SECRET", SESSION_SECRET)
    monkeypatch.setenv("FLOWSHEET_PUBLIC_URL", public_url)


def _page_result_dict() -> dict[str, Any]:
    """Minimal valid PageResult payload, matching test_verifier_serve.py."""
    return PageResult(
        page_date_raw="Mon 1 Jan 90",
        quadrants=[
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[], oddities=[])
            for p in QUADRANT_ORDER
        ],
        comments_raw=None,
        oddities=[],
        model_version="test-model",
        extracted_at=datetime(2026, 5, 12, tzinfo=UTC),
    ).model_dump(mode="json")


def _corrections_dict() -> dict[str, Any]:
    return {
        "stem": "test",
        "model_version": "test-model",
        "extracted_at": "2026-05-12T00:00:00Z",
        "exported_at": "2026-05-12T00:00:01Z",
        "page_corrections": [],
        "quadrant_corrections": [],
        "row_corrections": [],
        "added_rows": [],
        "deleted_rows": [],
    }


def _make_reviewer() -> ReviewerSession:
    return ReviewerSession(
        user_id="reviewer-real-id",
        email="dj@wxyc.org",
        username="dj_radio",
        real_name="Real Name",
        dj_name="Stage Name",
        role="dj",
    )


@pytest.fixture
def serve_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fresh OIDC-enabled FastAPI app rooted at `tmp_path`.

    Reloads `verifier.serve` after env is set so `OIDC_ENABLED`
    + the middleware install pick up our env. Cleans up module-level
    state on teardown so subsequent tests start clean.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    _set_oidc_env(monkeypatch)

    import verifier.serve as serve_mod

    importlib.reload(serve_mod)
    _reset_serve_module_state(serve_mod)
    yield serve_mod
    monkeypatch.undo()
    importlib.reload(serve_mod)
    _reset_serve_module_state(serve_mod)


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _session_cookie_for(reviewer: ReviewerSession) -> str:
    """Mint a session cookie value the way the callback handler does.

    Tests that want to skip the OIDC dance and exercise a protected
    endpoint directly use this to plant a session.
    """
    return auth_mod.encode_session(reviewer)


# -- /auth/login -----------------------------------------------------------


async def test_auth_login_redirects_with_signed_oneshots(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/auth/login 302s to the auth server with code_challenge + state,
    and sets `oidc_state` / `oidc_verifier` / `oidc_return_to` as signed
    one-shot cookies. Without those cookies, `/auth/callback` can't
    validate the round-trip — pin the contract so a refactor doesn't
    drop any of the three."""

    async def fake_authorize() -> tuple[str, str, str]:
        return ("https://auth.example/oauth2/authorize?fake=1", "state-value", "verifier-value")

    monkeypatch.setattr(auth_mod, "build_authorize_url", fake_authorize)

    async with await _client(serve_app.app) as c:
        r = await c.get("/auth/login?return_to=/verifier/?bundle=x", follow_redirects=False)

    assert r.status_code == 302
    assert r.headers["location"].startswith("https://auth.example/oauth2/authorize")
    cookies = r.headers.get_list("set-cookie")
    cookie_names = {c.split("=", 1)[0] for c in cookies}
    assert "oidc_state" in cookie_names
    assert "oidc_verifier" in cookie_names
    assert "oidc_return_to" in cookie_names


async def test_auth_login_503_on_auth_server_transport_failure(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If discovery doc fetch fails, /auth/login surfaces 503 (not 500)
    so the SPA can show a "try again" instead of a generic crash."""

    async def boom() -> tuple[str, str, str]:
        raise httpx.ConnectError("discovery unreachable")

    monkeypatch.setattr(auth_mod, "build_authorize_url", boom)

    async with await _client(serve_app.app) as c:
        r = await c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 503


# -- /auth/callback --------------------------------------------------------


async def _seed_one_shots(c: AsyncClient, state: str, verifier: str, return_to: str) -> None:
    """Plant the three signed one-shot cookies a callback round-trip
    would have set during /auth/login.

    Httpx's cookie jar would need cookie attributes; setting raw via the
    `Cookie` header on subsequent requests keeps the test simple.
    """
    signed_state = auth_mod.sign_one_shot(state)
    signed_verifier = auth_mod.sign_one_shot(verifier)
    signed_return_to = auth_mod.sign_one_shot(return_to)
    c.cookies.set("oidc_state", signed_state)
    c.cookies.set("oidc_verifier", signed_verifier)
    c.cookies.set("oidc_return_to", signed_return_to)


async def test_auth_callback_happy_path(serve_app, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid state + valid exchange → session cookie set, 302 to return_to."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        assert code == "auth-code"
        assert code_verifier == "the-verifier"
        return _make_reviewer()

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "the-state", "the-verifier", "/verifier/?bundle=x")
        r = await c.get("/auth/callback?code=auth-code&state=the-state", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/verifier/?bundle=x"
    cookie_names = {c.split("=", 1)[0] for c in r.headers.get_list("set-cookie")}
    assert "flowsheet_session" in cookie_names


async def test_auth_callback_400_on_state_mismatch(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `state` query param that doesn't match the signed cookie value
    is a CSRF attempt (or a botched cookie jar) — return 400, do not
    proceed to token exchange."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise AssertionError("exchange_code must not be called when state mismatches")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "expected-state", "v", "/verifier/")
        r = await c.get("/auth/callback?code=auth-code&state=wrong-state", follow_redirects=False)
    assert r.status_code == 400
    # Distinct detail from the missing/expired-cookie path so the two
    # 400s are separable in logs and by the SPA.
    assert r.json()["detail"] == "auth state mismatch"


async def test_auth_callback_400_on_non_ascii_state(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-ASCII `state` query param must be rejected as a mismatch
    (400), not crash the constant-time compare with a TypeError (500).

    `secrets.compare_digest` raises TypeError on non-ASCII str operands;
    the handler must compare bytes so a crafted `state=caf%C3%A9` (with
    the attacker's own valid one-shot cookies) is a clean 400."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise AssertionError("exchange_code must not be called when state mismatches")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "expected-state", "v", "/verifier/")
        r = await c.get("/auth/callback?code=auth-code&state=caf%C3%A9", follow_redirects=False)
    assert r.status_code == 400
    assert r.json()["detail"] == "auth state mismatch"


async def test_auth_callback_400_on_tampered_state_cookie(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered `oidc_state` cookie produces 400 — the signed
    one-shot's tamper resistance is load-bearing for the CSRF check."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise AssertionError("exchange must not run when state cookie is bad")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        c.cookies.set("oidc_state", "not-a-real-signed-value")
        c.cookies.set("oidc_verifier", auth_mod.sign_one_shot("v"))
        c.cookies.set("oidc_return_to", auth_mod.sign_one_shot("/verifier/"))
        r = await c.get("/auth/callback?code=auth-code&state=anything", follow_redirects=False)
    assert r.status_code == 400
    assert r.json()["detail"] == "missing or expired auth cookie"


async def test_auth_callback_400_on_missing_state_cookie(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `oidc_state` cookie at all (e.g. expired past ONE_SHOT_TTL, or
    dropped by the browser) yields the missing/expired-cookie 400 —
    distinct detail from a genuine state mismatch."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise AssertionError("exchange must not run when the state cookie is absent")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        # A code+state on the URL but no one-shot cookies in the jar.
        r = await c.get("/auth/callback?code=auth-code&state=anything", follow_redirects=False)
    assert r.status_code == 400
    assert r.json()["detail"] == "missing or expired auth cookie"


@pytest.mark.parametrize(
    "return_to",
    [
        "https://evil.example/",
        "//evil.example/verifier",
        # Word-boundary defense: '/verifierx-admin' looks like /verifier
        # to a naive startswith check; the allowlist must reject it so
        # an attacker can't smuggle a hostile route through a future
        # mount whose name shares the prefix.
        "/verifierx-admin/login",
        "/verifierdata/leak",
    ],
)
async def test_auth_callback_falls_back_to_verifier_for_off_allowlist_return_to(
    return_to: str, serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off-allowlist return_to → fallback to `/verifier/`. Open-redirect
    defense AND a word-boundary check so `/verifier*` siblings can't
    smuggle through a naive `startswith("/verifier")` check."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        return _make_reviewer()

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "s", "v", return_to)
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/verifier/"


async def test_auth_login_503_when_env_var_missing(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial OIDC config — WXYC_OIDC_CLIENT_ID set but
    WXYC_AUTH_ISSUER unset — surfaces as 503 ('auth not configured')
    rather than 500 with the env-var name in the traceback.
    Mirrors `core.auth.decode_session`'s RuntimeError catch for the
    middleware path.
    """

    async def boom() -> tuple[str, str, str]:
        raise RuntimeError("WXYC_AUTH_ISSUER is not set")

    monkeypatch.setattr(auth_mod, "build_authorize_url", boom)
    async with await _client(serve_app.app) as c:
        r = await c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 503


async def test_auth_login_503_when_session_secret_unset(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_authorize_url` succeeds but the subsequent `sign_one_shot`
    calls reach `_session_secret()`, which raises RuntimeError if
    WXYC_SESSION_SECRET is unset. The /auth/login handler must wrap
    BOTH the URL build and the cookie-sealing steps so a config
    rollout that updates WXYC_OIDC_CLIENT_ID and forgets
    WXYC_SESSION_SECRET still produces 503 (not 500). Regression
    guard for the earlier shape that only wrapped build_authorize_url.
    """

    async def fake_authorize() -> tuple[str, str, str]:
        return ("https://auth.example/oauth2/authorize?fake=1", "state-x", "verifier-x")

    monkeypatch.setattr(auth_mod, "build_authorize_url", fake_authorize)
    # Now blank WXYC_SESSION_SECRET so sign_one_shot raises RuntimeError.
    monkeypatch.delenv("WXYC_SESSION_SECRET", raising=False)
    async with await _client(serve_app.app) as c:
        r = await c.get("/auth/login", follow_redirects=False)
    assert r.status_code == 503


async def test_auth_callback_503_when_env_var_missing(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same protection on the callback path — RuntimeError from
    exchange_code (e.g., env var blanked mid-process) is mapped to 503,
    not 500.
    """

    async def boom(*, code: str, code_verifier: str) -> ReviewerSession:
        raise RuntimeError("WXYC_OIDC_CLIENT_SECRET is not set")

    monkeypatch.setattr(auth_mod, "exchange_code", boom)
    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "s", "v", "/verifier/")
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    assert r.status_code == 503


async def test_auth_callback_503_on_token_endpoint_transport_failure(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An httpx error from `exchange_code` (e.g., auth server slow) is
    503 not 500 so the SPA can distinguish transport from claim failure."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise httpx.ConnectError("token endpoint unreachable")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "s", "v", "/verifier/")
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    assert r.status_code == 503


async def test_auth_callback_400_on_claim_validation_failure(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `JoseError` from `exchange_code` (e.g., bad signature, aud
    mismatch) is 400 — the user needs to redo the login, not a 500
    crash."""
    from authlib.jose.errors import JoseError

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        raise JoseError("invalid aud")

    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        await _seed_one_shots(c, "s", "v", "/verifier/")
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    assert r.status_code == 400


# -- /auth/logout ----------------------------------------------------------


async def test_auth_logout_clears_session_cookie(serve_app) -> None:
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(_make_reviewer()))
        r = await c.post("/auth/logout")
    assert r.status_code == 200
    # `Set-Cookie` for delete sends an immediate-expiry cookie value.
    set_cookies = r.headers.get_list("set-cookie")
    assert any("flowsheet_session" in sc and "Max-Age=0" in sc for sc in set_cookies), (
        f"expected a delete on flowsheet_session, got: {set_cookies}"
    )


# -- /auth/me --------------------------------------------------------------


async def test_auth_me_401_without_session(serve_app) -> None:
    async with await _client(serve_app.app) as c:
        # Accept: application/json so the middleware returns 401 JSON
        # rather than a 302 redirect.
        r = await c.get(
            "/auth/me",
            headers={"accept": "application/json"},
            follow_redirects=False,
        )
    assert r.status_code == 401


async def test_auth_me_returns_reviewer_with_valid_session(serve_app) -> None:
    """With a valid session cookie, /auth/me returns the ReviewerSession
    as JSON. The SPA reads this on load to render the reviewer's name."""
    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        r = await c.get("/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == reviewer.user_id
    assert body["real_name"] == reviewer.real_name
    assert body["dj_name"] == reviewer.dj_name


# -- /api/version unauthenticated -----------------------------------------


async def test_api_version_is_public(serve_app) -> None:
    """`/api/version` must stay public — it's the Railway healthcheck.
    A regression here breaks every deploy because the healthcheck would
    redirect to /auth/login. Load-bearing assertion."""
    async with await _client(serve_app.app) as c:
        r = await c.get("/api/version")
    assert r.status_code == 200
    assert "version" in r.json()


# -- /api/save -------------------------------------------------------------


async def test_api_save_returns_401_for_non_document_fetch(serve_app) -> None:
    """A fetch to /api/save without a session (and without an explicit
    `Accept: application/json`) gets a 401, NOT a 302 to /auth/login.

    /api/save is only ever called by the SPA via fetch(); it is never a
    top-level browser navigation. Redirecting a non-document request to
    /auth/login would mint a fresh OIDC `state` and clobber the one-shot
    cookie of an in-flight login (the state-mismatch bug). Only genuine
    document navigations redirect; everything else 401s."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={"stem": "x", "verified": _page_result_dict(), "corrections": _corrections_dict()},
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_api_save_returns_401_json_for_ajax_without_session(serve_app) -> None:
    """A fetch() request with `Accept: application/json` gets a JSON
    401 so the SPA can redirect via `location.href`."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            headers={"accept": "application/json"},
            json={"stem": "x", "verified": _page_result_dict(), "corrections": _corrections_dict()},
            follow_redirects=False,
        )
    assert r.status_code == 401


# -- session gate: document navigation vs subresource ----------------------
#
# Only a genuine top-level browser navigation is 302'd to /auth/login.
# Every other request type (subresource fetch, favicon, prefetch, XHR)
# gets a 401. This prevents a gated subresource from being funneled
# through /auth/login mid-login, which mints a fresh OIDC `state` and
# clobbers the one-shot cookie of the login the user is completing —
# the "invalid auth state" (state mismatch) bug.


async def test_gated_document_navigation_redirects_to_login(serve_app) -> None:
    """A top-level navigation (`Sec-Fetch-Dest: document`) to a gated
    path without a session 302s to /auth/login with `return_to` set."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/auth/me",
            headers={"sec-fetch-dest": "document"},
            follow_redirects=False,
        )
    assert r.status_code == 302
    assert r.headers["location"].startswith("/auth/login?return_to=")


async def test_gated_html_accept_without_fetch_metadata_redirects(serve_app) -> None:
    """Fallback for clients that omit `Sec-Fetch-Dest` (old browsers,
    some proxies): an `Accept: text/html` request is treated as a
    document navigation and redirected."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/auth/me",
            headers={"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
            follow_redirects=False,
        )
    assert r.status_code == 302
    assert r.headers["location"].startswith("/auth/login?return_to=")


async def test_gated_subresource_fetch_returns_401_not_redirect(serve_app) -> None:
    """A subresource fetch (favicon-style, `Sec-Fetch-Dest: image`)
    without a session gets a 401 and is NOT redirected to /auth/login.

    Regression test for the state-clobber: if this 302'd, the browser
    would follow it to /auth/login, mint a new OIDC state, and break the
    concurrent real login's callback with 'invalid auth state'."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/favicon.ico",
            headers={
                "sec-fetch-dest": "image",
                "accept": "image/avif,image/webp,image/png,*/*",
            },
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_gated_prefetch_document_returns_401(serve_app) -> None:
    """A speculative prefetch/prerender carries `Sec-Fetch-Dest: document`
    but also `Sec-Purpose: prefetch`. It must be treated as a subresource
    (401), NOT a real navigation — otherwise the prefetch 302s to
    /auth/login and mints a fresh OIDC state that clobbers the one-shot
    cookie of a concurrent real login (the exact bug this PR fixes)."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/verifier/",
            headers={"sec-fetch-dest": "document", "sec-purpose": "prefetch;prerender"},
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_gated_legacy_purpose_prefetch_returns_401(serve_app) -> None:
    """The legacy `Purpose: prefetch` header (older Chromium / some
    crawlers) must also demote a document-dest request to a subresource."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/verifier/",
            headers={"sec-fetch-dest": "document", "purpose": "prefetch"},
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_gated_both_accept_types_without_fetch_metadata_returns_401(serve_app) -> None:
    """When `Sec-Fetch-Dest` is absent, an `Accept` that advertises BOTH
    application/json and text/html must NOT be routed to the state-minting
    /auth/login. The pre-Fetch-Metadata gate excluded any request
    advertising application/json; the fallback preserves that so an XHR
    tolerant of html doesn't reach /auth/login."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/auth/me",
            headers={"accept": "application/json, text/html"},
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_gated_empty_dest_fetch_returns_401(serve_app) -> None:
    """A programmatic fetch/XHR (`Sec-Fetch-Dest: empty`) without a
    session gets a 401, not a redirect."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/auth/me",
            headers={"sec-fetch-dest": "empty"},
            follow_redirects=False,
        )
    assert r.status_code == 401
    assert "location" not in r.headers


async def test_document_dest_wins_over_non_html_accept(serve_app) -> None:
    """`Sec-Fetch-Dest` is authoritative when present: a document
    navigation redirects even if its `Accept` header lacks text/html."""
    async with await _client(serve_app.app) as c:
        r = await c.get(
            "/auth/me",
            headers={"sec-fetch-dest": "document", "accept": "*/*"},
            follow_redirects=False,
        )
    assert r.status_code == 302
    assert r.headers["location"].startswith("/auth/login?return_to=")


async def test_api_save_writes_verified_by_from_session(serve_app, tmp_path: Path) -> None:
    """With a session, /api/save writes a `verified_by` block populated
    from the reviewer's identity. The block includes user_id, name,
    dj_name."""
    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        r = await c.post(
            "/api/save",
            json={
                "stem": "page25",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200

    on_disk = json.loads((tmp_path / "data" / "verifier" / "page25.verified.json").read_text())
    vb = on_disk["verified_by"]
    assert vb is not None
    assert vb["user_id"] == reviewer.user_id
    assert vb["username"] == reviewer.username
    assert vb["real_name"] == reviewer.real_name
    assert vb["dj_name"] == reviewer.dj_name
    assert "verified_at" in vb


async def test_api_save_discards_client_supplied_verified_by(serve_app, tmp_path: Path) -> None:
    """Server-authority: a client that crafts a `verified_by` block in
    the payload gets it overwritten with the server's view of the
    authenticated reviewer. Protects against credential confusion via
    a buggy or hostile client."""
    reviewer = _make_reviewer()
    polluted = _page_result_dict()
    polluted["verified_by"] = {
        "user_id": "spoofed-id",
        "username": "attacker",
        "real_name": "Not Real",
        "dj_name": "Fake",
        "verified_at": "2020-01-01T00:00:00Z",
    }
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        r = await c.post(
            "/api/save",
            json={
                "stem": "spoof-test",
                "verified": polluted,
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    on_disk = json.loads((tmp_path / "data" / "verifier" / "spoof-test.verified.json").read_text())
    assert on_disk["verified_by"]["user_id"] == reviewer.user_id
    # And the spoofed verified_at was replaced with the server's clock.
    assert on_disk["verified_by"]["verified_at"] != "2020-01-01T00:00:00Z"


async def test_api_save_rejects_malformed_client_verified_by(serve_app) -> None:
    """A malformed `verified_by` (wrong type) fails Pydantic
    validation BEFORE the server-authority overwrite runs. The error
    surfaces as 400 (defense-in-depth — the overwrite would have
    discarded the value, but exposing a corrupt shape via 400 is better
    than silently accepting it)."""
    reviewer = _make_reviewer()
    cases = [
        {"verified_by": 123},
        {"verified_by": []},
        {"verified_by": "hello"},
        {"verified_by": {}},  # missing required user_id + verified_at
    ]
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        for i, override in enumerate(cases):
            verified = {**_page_result_dict(), **override}
            r = await c.post(
                "/api/save",
                json={
                    "stem": f"malformed-{i}",
                    "verified": verified,
                    "corrections": _corrections_dict(),
                },
            )
            assert r.status_code == 400, f"case {i}={override!r} expected 400 got {r.status_code}"


async def test_api_save_writes_reviewer_id_to_jobs_db(serve_app, tmp_path: Path) -> None:
    """With pdf_path + page_number + a session, mark_verified is called
    with the reviewer's user_id and the jobs.db row picks it up."""
    from core.jobs import JobStore

    db_path = tmp_path / "data" / "jobs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = JobStore(db_path)
    await store.init()
    await store.register("1990/x.pdf", 1)
    await store.mark_rendered("1990/x.pdf", 1, image_path=tmp_path / "x.png")
    await store.mark_completed("1990/x.pdf", 1, result_path=tmp_path / "x.json", model_version="m")

    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        r = await c.post(
            "/api/save",
            json={
                "stem": "x-page-01",
                "pdf_path": "1990/x.pdf",
                "page_number": 1,
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["db_updated"] is True

    job = await store.get("1990/x.pdf", 1)
    assert job is not None
    assert job.reviewer_id == reviewer.user_id


async def test_api_save_upgrades_pre_oidc_file_on_save(serve_app, tmp_path: Path) -> None:
    """An old verified.json on disk (no `verified_by` key) loads via
    Pydantic, gets the server-authoritative block added, and is written
    back. This is the round-trip the SPA performs on every save against
    files that existed before the OIDC PR.
    """
    # Plant an old verified.json directly so the test exercises the
    # full load → modify → save shape. `_page_result_dict()` always
    # serializes `verified_by: None` because the field has a default —
    # pop it explicitly to mirror a file written before the field
    # existed on the schema at all.
    verifier_dir = tmp_path / "data" / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    old = _page_result_dict()
    old.pop("verified_by", None)
    assert "verified_by" not in old  # baseline: shape is truly pre-OIDC
    (verifier_dir / "legacy.verified.json").write_text(json.dumps(old))

    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie_for(reviewer))
        r = await c.post(
            "/api/save",
            json={
                "stem": "legacy",
                # SPA round-trips whatever it read; we mirror that here.
                "verified": old,
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    saved = json.loads((verifier_dir / "legacy.verified.json").read_text())
    assert saved["verified_by"]["user_id"] == reviewer.user_id


# -- middleware precedence -------------------------------------------------


def _reload_with_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    oidc: bool,
    basic: bool,
    public_url: str = PUBLIC_URL,
):
    """Reload `verifier.serve` with a controlled middleware-install env.

    Tests that toggle (oidc, basic) parametrizations call this to
    rebuild the app. `_reset_serve_module_state` is invoked after the
    reload — see its docstring for the why.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    if oidc:
        _set_oidc_env(monkeypatch, public_url=public_url)
    else:
        for k in (
            "WXYC_OIDC_CLIENT_ID",
            "WXYC_OIDC_CLIENT_SECRET",
            "WXYC_AUTH_ISSUER",
            "WXYC_SESSION_SECRET",
        ):
            monkeypatch.delenv(k, raising=False)
    if basic:
        monkeypatch.setenv("VERIFIER_PASSWORD", "test-password")
    else:
        monkeypatch.delenv("VERIFIER_PASSWORD", raising=False)
    # FLOWSHEET_PUBLIC_URL controls the cookie Secure flag and is used
    # in the Secure-flag tests too.
    monkeypatch.setenv("FLOWSHEET_PUBLIC_URL", public_url)

    import verifier.serve as serve_mod

    importlib.reload(serve_mod)
    _reset_serve_module_state(serve_mod)
    return serve_mod


async def test_oidc_wins_over_basicauth_when_both_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both `WXYC_OIDC_CLIENT_ID` and `VERIFIER_PASSWORD` are
    set, the OIDC middleware installs and BasicAuth does not. A request
    with Basic credentials but no session cookie redirects to
    /auth/login (the BasicAuth gate would have returned 200 here)."""
    from base64 import b64encode

    serve_mod = _reload_with_env(monkeypatch, tmp_path, oidc=True, basic=True)
    creds = b64encode(b"verifier:test-password").decode()
    async with await _client(serve_mod.app) as c:
        r = await c.post(
            "/api/save",
            headers={
                "authorization": f"Basic {creds}",
                # A document navigation so the OIDC gate 302s to login —
                # the behavior that distinguishes it from BasicAuth (which
                # would answer 401 with WWW-Authenticate for any request).
                "sec-fetch-dest": "document",
            },
            json={
                "stem": "x",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
            follow_redirects=False,
        )
    # OIDC middleware ignores Basic and redirects the document nav to login.
    assert r.status_code == 302
    assert r.headers["location"].startswith("/auth/login")


async def test_basicauth_used_when_only_password_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without OIDC env, `VERIFIER_PASSWORD` alone installs BasicAuth.
    A request without credentials returns the existing 401 + WWW-
    Authenticate header. Mirrors the pre-OIDC deployment shape."""
    serve_mod = _reload_with_env(monkeypatch, tmp_path, oidc=False, basic=True)
    async with await _client(serve_mod.app) as c:
        r = await c.get("/api/version")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


async def test_no_gate_when_neither_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The local-dev default — neither OIDC nor BasicAuth — leaves
    every endpoint open."""
    serve_mod = _reload_with_env(monkeypatch, tmp_path, oidc=False, basic=False)
    async with await _client(serve_mod.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "open",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    saved = json.loads((tmp_path / "data" / "verifier" / "open.verified.json").read_text())
    # No reviewer → verified_by stays None.
    assert saved["verified_by"] is None


async def test_no_auth_save_preserves_existing_verified_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-save in BasicAuth / no-auth mode must NOT clobber a
    previously-recorded `verified_by` block on the on-disk file.

    Sequence: an earlier OIDC-mode save left Alice's identity on the
    file. The deployment is briefly flipped to no-auth (e.g. during an
    OIDC outage rollback) and a volunteer re-saves the same stem. Per
    the data-safety rule ('never overwrite successfully collected
    data'), the existing `verified_by` block must survive even though
    the current request has no authenticated reviewer to credit. The
    bug this guards against: an unconditional `validated.verified_by =
    None` silently destroys the provenance record on every re-save.
    """
    verifier_dir = tmp_path / "data" / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    # Plant a verified.json that already carries Alice's identity (as
    # an OIDC save would have written).
    prior_with_alice = {
        **_page_result_dict(),
        "verified_by": {
            "user_id": "alice-id",
            "username": "alice",
            "real_name": "Alice Real",
            "dj_name": "DJ Alice",
            "verified_at": "2026-05-01T12:00:00Z",
        },
    }
    (verifier_dir / "preserved.verified.json").write_text(json.dumps(prior_with_alice))

    serve_mod = _reload_with_env(monkeypatch, tmp_path, oidc=False, basic=False)
    async with await _client(serve_mod.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "preserved",
                # The SPA round-trips the file's shape; mirror that.
                "verified": prior_with_alice,
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    saved = json.loads((verifier_dir / "preserved.verified.json").read_text())
    # Alice's identity survives the no-auth re-save.
    assert saved["verified_by"] is not None
    assert saved["verified_by"]["user_id"] == "alice-id"
    assert saved["verified_by"]["real_name"] == "Alice Real"


async def test_no_auth_save_ignores_client_supplied_verified_by_when_no_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In no-auth mode with NO prior on-disk file, a client-supplied
    `verified_by` is still discarded — the server has no identity to
    assert and no prior data to preserve, so the saved file's
    `verified_by` is `None`. Pins the anti-spoof rule in non-OIDC
    mode (a client can't write provenance unilaterally).
    """
    serve_mod = _reload_with_env(monkeypatch, tmp_path, oidc=False, basic=False)
    spoofed = {
        **_page_result_dict(),
        "verified_by": {
            "user_id": "attacker",
            "username": "evil",
            "real_name": None,
            "dj_name": None,
            "verified_at": "2026-05-01T00:00:00Z",
        },
    }
    async with await _client(serve_mod.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "fresh",
                "verified": spoofed,
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    saved = json.loads((tmp_path / "data" / "verifier" / "fresh.verified.json").read_text())
    assert saved["verified_by"] is None


# -- cookie Secure flag ----------------------------------------------------


async def test_cookie_secure_on_https_public_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FLOWSHEET_PUBLIC_URL=https://..., the session cookie set on
    /auth/callback carries `Secure`. Prevents the cookie from being
    sent over a downgraded HTTP request."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        return _make_reviewer()

    serve_mod = _reload_with_env(
        monkeypatch, tmp_path, oidc=True, basic=False, public_url="https://flowsheet.example"
    )
    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_mod.app) as c:
        await _seed_one_shots(c, "s", "v", "/verifier/")
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    set_cookies = r.headers.get_list("set-cookie")
    session_sc = next(sc for sc in set_cookies if sc.startswith("flowsheet_session="))
    assert "Secure" in session_sc


async def test_cookie_not_secure_on_http_public_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With FLOWSHEET_PUBLIC_URL=http://localhost:8765, the session
    cookie does NOT carry `Secure` — local dev runs plain HTTP and a
    Secure cookie would never be sent at all."""

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        return _make_reviewer()

    serve_mod = _reload_with_env(
        monkeypatch, tmp_path, oidc=True, basic=False, public_url="http://localhost:8765"
    )
    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_mod.app) as c:
        await _seed_one_shots(c, "s", "v", "/verifier/")
        r = await c.get("/auth/callback?code=c&state=s", follow_redirects=False)
    set_cookies = r.headers.get_list("set-cookie")
    session_sc = next(sc for sc in set_cookies if sc.startswith("flowsheet_session="))
    assert "Secure" not in session_sc


# -- return_to round-trip --------------------------------------------------


async def test_login_return_to_propagates_through_callback(
    serve_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The /auth/login → /auth/callback round-trip preserves return_to
    when it's a valid /verifier-relative path. Pins the integration of
    the one-shot cookie + the callback's allowlist check.

    We bypass httpx's cookie jar (which struggles with the path=/auth
    attribute the login handler sets) and read the Set-Cookie headers
    directly, then plant them on the next request. Confirms the wire
    format end-to-end rather than relying on jar semantics.
    """
    captured: dict[str, Any] = {}

    async def fake_authorize() -> tuple[str, str, str]:
        return ("https://auth.example/oauth2/authorize?fake=1", "round-trip-state", "the-verifier")

    async def fake_exchange(*, code: str, code_verifier: str) -> ReviewerSession:
        captured["code"] = code
        captured["verifier"] = code_verifier
        return _make_reviewer()

    monkeypatch.setattr(auth_mod, "build_authorize_url", fake_authorize)
    monkeypatch.setattr(auth_mod, "exchange_code", fake_exchange)

    async with await _client(serve_app.app) as c:
        r1 = await c.get("/auth/login?return_to=/verifier/?bundle=abc", follow_redirects=False)
        assert r1.status_code == 302
        assert urlparse(r1.headers["location"]).netloc == "auth.example"
        # Extract the three one-shot cookie values from Set-Cookie
        # headers and re-plant them on the client so the callback
        # request carries them.
        for sc in r1.headers.get_list("set-cookie"):
            name, _, rest = sc.partition("=")
            if name in {"oidc_state", "oidc_verifier", "oidc_return_to"}:
                value = rest.partition(";")[0]
                c.cookies.set(name, value)
        r2 = await c.get(
            "/auth/callback?code=AUTH-CODE&state=round-trip-state",
            follow_redirects=False,
        )
    assert r2.status_code == 302
    assert r2.headers["location"] == "/verifier/?bundle=abc"
    assert captured["code"] == "AUTH-CODE"
    assert captured["verifier"] == "the-verifier"


# Smoke check: prevent the "I forgot to import parse_qs" surprise where
# the test file fails to parse if a transitive ref disappears.
def test_helpers_import() -> None:
    assert parse_qs("a=b")["a"] == ["b"]
