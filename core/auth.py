"""OIDC client + session cookie codec for the verifier UI.

`verifier/serve.py` consumes this module from two places:

  * The request-gating middleware reads the session cookie via
    `decode_session`. None → redirect to `/auth/login`. A
    `ReviewerSession` → stash on `request.state.reviewer` and proceed.
  * The `/auth/login` and `/auth/callback` route handlers run the OIDC
    code + PKCE dance via `build_authorize_url` + `exchange_code`, then
    seal the resulting `ReviewerSession` into a cookie with
    `encode_session`.

The OIDC counterparty is WXYC's Better Auth server at the issuer URL
in `WXYC_AUTH_ISSUER` (typically `https://api.wxyc.org/auth`). This
module is configured purely by env vars — the plan rules out injecting
the OIDC client at construction time, because the discovery doc + JWKS
are configuration, not per-call state, and re-loading them on every
auth request would add a round-trip we don't need.

## Module-level cached state (deviation from `core/gemini.py`)

`core/gemini.py` injects the SDK client at construction time so tests
can substitute mocks; this module deviates and caches the discovery
doc + JWKS at module scope. The trade-off is intentional:

  * Discovery + JWKS are configuration that changes on a key-rotation
    clock — once an hour at most, in practice not for weeks. They are
    not per-call client state.
  * The verifier serves thousands of `/auth/callback` requests against
    the same JWKS in a single process lifetime. Re-fetching per call
    means a wasted httpx round-trip every login.
  * The function-level boundary (`_load_metadata` / `_load_jwks`) is
    the test seam — patching those two functions directly is the
    intended way to inject fakes; `_reset_metadata_cache()` is the
    explicit reset for parametrized tests that flip `WXYC_AUTH_ISSUER`.

Considered and rejected: eager-loading the discovery doc + JWKS at
verifier process startup. Coupling boot to api.wxyc.org reachability
means a brief auth-server slowdown during a flowsheet deploy would
prevent the verifier from starting at all (including its `/api/version`
healthcheck, which Railway uses to decide whether to keep a new
revision alive). Lazy loading degrades more gracefully: a transient
auth-server failure surfaces as a 503 on the first `/auth/*` call, the
verifier process stays up, and the next request retries. Cold-start
cost is one httpx round-trip per process; if that ever becomes a real
complaint, the right fix is async pre-warming on app startup, not
making boot blocking.

Don't "fix" this back to the DI pattern by reflex.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import urllib.parse
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import httpx
from authlib.jose import JsonWebKey, JsonWebToken
from itsdangerous import BadSignature, TimestampSigner

# Cookie + TTL constants. Treat as public surface — the UI and middleware
# both depend on these names not changing silently.
COOKIE_NAME = "flowsheet_session"
SESSION_TTL = timedelta(hours=12)
ONE_SHOT_TTL = timedelta(minutes=10)

# Discovery doc + JWKS are cached at module scope. See the module
# docstring for why this deviates from `core/gemini.py`'s DI pattern.
_metadata: dict[str, Any] | None = None
_jwks: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReviewerSession:
    """Identity of an authenticated volunteer reviewer.

    Populated by `exchange_code` from the OIDC id_token claims. Survives
    requests via the signed session cookie (`encode_session` /
    `decode_session`). The dataclass is frozen so the request-gating
    middleware can stash it on `request.state.reviewer` and trust that
    downstream handlers can't tamper with the fields.

    `role` is advisory only — nothing in this PR gates access on the
    claim. The plan threads it through anticipating per-reviewer
    feature gating in a future change.
    """

    user_id: str
    email: str
    username: str | None
    real_name: str | None
    dj_name: str | None
    role: str | None


# -- env-var accessors -----------------------------------------------------
#
# Every accessor reads `os.environ` directly so tests that flip env vars
# via `monkeypatch.setenv` are seen on the next call without any cache
# reset — the only thing cached is the discovery doc + JWKS.


def _issuer() -> str:
    value = os.environ.get("WXYC_AUTH_ISSUER", "")
    if not value:
        raise RuntimeError("WXYC_AUTH_ISSUER is not set")
    return value.rstrip("/")


def _client_id() -> str:
    value = os.environ.get("WXYC_OIDC_CLIENT_ID", "")
    if not value:
        raise RuntimeError("WXYC_OIDC_CLIENT_ID is not set")
    return value


def _client_secret() -> str:
    value = os.environ.get("WXYC_OIDC_CLIENT_SECRET", "")
    if not value:
        raise RuntimeError("WXYC_OIDC_CLIENT_SECRET is not set")
    return value


def _session_secret() -> str:
    value = os.environ.get("WXYC_SESSION_SECRET", "")
    if not value:
        raise RuntimeError("WXYC_SESSION_SECRET is not set")
    return value


def _public_url() -> str:
    value = os.environ.get("FLOWSHEET_PUBLIC_URL", "")
    if not value:
        raise RuntimeError("FLOWSHEET_PUBLIC_URL is not set")
    return value.rstrip("/")


def _redirect_uri() -> str:
    return f"{_public_url()}/auth/callback"


# -- discovery doc + JWKS --------------------------------------------------


async def _load_metadata() -> dict[str, Any]:
    """Return the OIDC discovery doc, fetching + caching on first call.

    Test seam: tests patch this function directly (see test_auth.py).
    """
    global _metadata
    if _metadata is None:
        url = f"{_issuer()}/.well-known/openid-configuration"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            _metadata = r.json()
    return _metadata


async def _load_jwks() -> dict[str, Any]:
    """Return the JWKS, fetching + caching on first call.

    Test seam: tests patch this function directly (see test_auth.py).
    """
    global _jwks
    if _jwks is None:
        metadata = await _load_metadata()
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(metadata["jwks_uri"])
            r.raise_for_status()
            _jwks = r.json()
    return _jwks


def _reset_metadata_cache() -> None:
    """Test-only: clear the module-level discovery + JWKS cache.

    Production callers never need this — the cache is intentionally
    long-lived. Parametrized tests that flip `WXYC_AUTH_ISSUER` need
    to call this between cases or they observe the previous test's
    stale metadata.
    """
    global _metadata, _jwks
    _metadata = None
    _jwks = None


# -- session cookie codec --------------------------------------------------
#
# itsdangerous's `TimestampSigner` produces `value.timestamp.signature`.
# `unsign(value, max_age=...)` verifies the signature AND rejects values
# older than `max_age` seconds. We serialize the dataclass to compact
# JSON and let the signer wrap it.


def _session_signer() -> TimestampSigner:
    # `salt` namespaces the signature so a one-shot cookie value cannot
    # be re-used as a session cookie (and vice versa) even though both
    # share `WXYC_SESSION_SECRET`.
    return TimestampSigner(_session_secret(), salt="flowsheet-session")


def _one_shot_signer() -> TimestampSigner:
    return TimestampSigner(_session_secret(), salt="flowsheet-one-shot")


def encode_session(s: ReviewerSession) -> str:
    """Sign a `ReviewerSession` into a cookie value with `SESSION_TTL`
    enforcement baked in (the freshness check is part of `decode_session`).
    """
    payload = json.dumps(asdict(s), separators=(",", ":"))
    return _session_signer().sign(payload).decode("utf-8")


def decode_session(raw: str) -> ReviewerSession | None:
    """Return the signed `ReviewerSession`, or None if the cookie is
    missing, tampered, expired, or otherwise unparseable.

    Returns None (rather than raising) so the request-gating middleware
    has one branch — "no valid session, redirect to /auth/login" —
    instead of needing a try/except on every protected request.
    """
    if not raw:
        return None
    try:
        unsigned = _session_signer().unsign(raw, max_age=int(SESSION_TTL.total_seconds()))
        payload = json.loads(unsigned)
        return ReviewerSession(**payload)
    except (BadSignature, ValueError, TypeError, KeyError):
        # BadSignature covers both tamper and expiry. The rest catch
        # malformed JSON or a payload shape that doesn't match the
        # ReviewerSession contract — both treated as "no session".
        return None


def sign_one_shot(value: str) -> str:
    """Sign a one-shot value (state, code_verifier, return_to) for
    transit in a redirect-survival cookie.

    Counterpart `verify_one_shot` MUST be called with
    `max_age=int(ONE_SHOT_TTL.total_seconds())` or shorter; anything
    older is considered a forged/expired cookie.
    """
    return _one_shot_signer().sign(value).decode("utf-8")


def verify_one_shot(raw: str, *, max_age: int) -> str:
    """Counterpart to `sign_one_shot`; raises `BadSignature` on tamper
    or expiry.

    Callers in `verifier/serve.py` always pass
    `max_age=int(ONE_SHOT_TTL.total_seconds())`. The parameter is
    explicit (not baked in) so tests can drive the expiry branch with
    `max_age=0` and a stubbed wall-clock without sleeping.
    """
    return _one_shot_signer().unsign(raw, max_age=max_age).decode("utf-8")


# -- OIDC dance ------------------------------------------------------------


def _generate_pkce_pair() -> tuple[str, str]:
    """Return `(code_verifier, code_challenge)` for an S256 PKCE flow.

    `code_verifier` is opaque to the auth server; `code_challenge` is
    `BASE64URL(SHA256(code_verifier))` with trailing `=` stripped per
    RFC 7636. The auth server is configured with `requirePKCE: true`,
    so a missing or malformed challenge fails the flow at authorize-
    time, not at token-exchange-time.
    """
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


async def build_authorize_url() -> tuple[str, str, str]:
    """Build the OIDC authorize URL and return `(url, state, code_verifier)`.

    The caller is responsible for stashing `state` and `code_verifier`
    in one-shot signed cookies and validating them on the callback
    round-trip. The `return_to` cookie is set by the route handler too,
    but is NOT part of this URL — it's a same-origin redirect target,
    not an OIDC parameter.
    """
    metadata = await _load_metadata()
    state = secrets.token_urlsafe(32)
    code_verifier, code_challenge = _generate_pkce_pair()

    params = {
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        # `openid` is required for an id_token; `profile email` get the
        # name / email / preferred_username claims we read in
        # `exchange_code`.
        "scope": "openid profile email",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    url = f"{metadata['authorization_endpoint']}?{urllib.parse.urlencode(params)}"
    return url, state, code_verifier


async def exchange_code(*, code: str, code_verifier: str) -> ReviewerSession:
    """Exchange an authorization code for an id_token, verify it, return
    the reviewer.

    Failure modes the caller must handle:

      * `httpx.HTTPError` — token endpoint or JWKS endpoint unreachable
        / non-2xx. Route layer translates to 503.
      * `authlib.jose.errors.JoseError` (and subclasses, including
        `InvalidClaimError` for `aud` / `iss` mismatch, and
        `BadSignatureError` for an id_token signed by an unknown key)
        — the id_token is structurally valid but its claims or
        signature are wrong. Route layer translates to 400.
    """
    metadata = await _load_metadata()
    jwks = await _load_jwks()

    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            metadata["token_endpoint"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _redirect_uri(),
                "code_verifier": code_verifier,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
            },
        )
        token_response.raise_for_status()
        body = token_response.json()

    id_token_raw = body["id_token"]

    # `aud` and `iss` are checked here, not somewhere downstream. Skipping
    # would let a Wiki.js-issued token (also signed by api.wxyc.org's JWKS)
    # authenticate to the verifier — that's a real cross-app confusion bug
    # the plan calls out explicitly.
    claims_options = {
        "iss": {"essential": True, "value": _issuer()},
        "aud": {"essential": True, "value": _client_id()},
    }
    jwt = JsonWebToken(["RS256", "ES256"])
    claims = jwt.decode(
        id_token_raw,
        key=JsonWebKey.import_key_set(jwks),
        claims_options=claims_options,
    )
    claims.validate()

    # Better Auth's OIDC provider includes `preferred_username`, `email`,
    # `name`, and any extra claims `getAdditionalUserInfoClaim` returns
    # (the auth.definition.ts setup pulls `dj_name` and `role` from the
    # WXYC user record). Treat each as optional — a user with no DJ
    # name is fine, we just don't render one.
    return ReviewerSession(
        user_id=str(claims["sub"]),
        email=str(claims.get("email", "")),
        username=claims.get("preferred_username") or claims.get("username") or None,
        real_name=claims.get("name") or claims.get("real_name") or None,
        dj_name=claims.get("dj_name"),
        role=claims.get("role"),
    )
