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
  * `asyncio.Lock`s around the cache fills serialize concurrent first
    callers (avoids two parallel `/auth/callback` requests both
    fetching the discovery doc when one would do).
  * `exchange_code` refreshes the JWKS cache once on signature failure
    so an auth-server key rotation recovers without a process restart.

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

## Why no `nonce`

OIDC nonce mitigates id_token replay across the redirect bounce. With
PKCE on a confidential code-flow client (this), the `code_verifier`
already binds the token exchange to the originating browser session,
and the resulting id_token is consumed locally — it's never presented
to downstream services as a bearer credential. Adding nonce would be
defense in depth, but the plan keeps the API surface tight; revisit
if the id_token starts flowing further than the verifier process.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any

import httpx
from authlib.common.urls import add_params_to_uri
from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import BadSignatureError
from authlib.oauth2.rfc7636 import create_s256_code_challenge
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
# Locks serialize concurrent first-callers; without them two parallel
# `/auth/callback` requests on a cold process can both pass the
# `if _metadata is None` check and both fetch, contradicting the
# 'one round-trip per process' rationale above. asyncio.Lock at module
# scope is safe in 3.12+ (binds lazily to the running loop on first use).
_metadata_lock = asyncio.Lock()
_jwks_lock = asyncio.Lock()

# Allowed signature algorithms. RS256 covers Better Auth's default; ES256
# is here so a future migration to elliptic-curve keys doesn't require a
# code change. Algorithms NOT in this set (HS256, none) are refused at
# decode time — defense against alg-confusion attacks.
_ALLOWED_SIG_ALGS: tuple[str, ...] = ("RS256", "ES256")
# `JsonWebToken` is stateless and immutable once configured with an
# algorithm list — hoist to module scope so we don't rebuild the
# algorithm registry on every login.
_JWT = JsonWebToken(list(_ALLOWED_SIG_ALGS))


@dataclass(frozen=True)
class ReviewerSession:
    """Identity of an authenticated volunteer reviewer.

    Populated by `exchange_code` from the OIDC id_token claims. Survives
    requests via the signed session cookie (`encode_session` /
    `decode_session`). The dataclass is frozen so the request-gating
    middleware can stash it on `request.state.reviewer` and trust that
    downstream handlers can't tamper with the fields.

    `email` is Optional because Better Auth users without an email scope
    or with no email on record have no email claim to surface — the
    earlier draft defaulted to `""` and silently produced an invalid
    record. `None` is the honest signal.

    `role` is advisory only — nothing in this PR gates access on the
    claim. The plan threads it through anticipating per-reviewer
    feature gating in a future change.
    """

    user_id: str
    email: str | None
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
    if _metadata is not None:
        return _metadata
    async with _metadata_lock:
        # Double-checked: a coroutine that lost the race for the lock
        # sees the cache filled by the winner.
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
    if _jwks is not None:
        return _jwks
    async with _jwks_lock:
        if _jwks is None:
            metadata = await _load_metadata()
            jwks_uri = metadata.get("jwks_uri")
            if not jwks_uri:
                # A discovery doc without `jwks_uri` is unusable; surface
                # this as a ValueError so the route layer translates to
                # 400 rather than the docstring-violating KeyError.
                raise ValueError("discovery doc missing jwks_uri")
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(jwks_uri)
                r.raise_for_status()
                _jwks = r.json()
        return _jwks


def _invalidate_jwks_cache() -> None:
    """Drop the cached JWKS without touching discovery.

    Called by `exchange_code` after a signature failure that might
    indicate the auth server rotated its signing key — the next
    `_load_jwks()` call refetches.
    """
    global _jwks
    _jwks = None


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


def _signing_keys_only(jwks: dict[str, Any]) -> dict[str, Any]:
    """Return a JWKS subset containing only keys usable for signature
    verification — keys whose `use` is `sig` or unspecified, and whose
    `alg` (if present) is in `_ALLOWED_SIG_ALGS`.

    Defense against an auth server publishing an encryption key (or a
    key for an unsupported alg) in the same JWKS that the verifier
    would otherwise trust during signature verification.
    """
    keep = []
    for k in jwks.get("keys", []):
        if k.get("use", "sig") != "sig":
            continue
        if "alg" in k and k["alg"] not in _ALLOWED_SIG_ALGS:
            continue
        keep.append(k)
    return {"keys": keep}


# -- session cookie codec --------------------------------------------------
#
# itsdangerous's `TimestampSigner` produces `value.timestamp.signature`.
# `unsign(value, max_age=...)` verifies the signature AND rejects values
# older than `max_age` seconds. We serialize the dataclass to compact
# JSON and let the signer wrap it.


# Signers are stateless functions of (secret, salt). Cache the
# constructed `TimestampSigner` keyed on those inputs so the
# request-gating middleware (which calls `decode_session` on every
# request) doesn't re-derive HMAC key material per call. The cache is
# tiny (one entry per active (secret, salt) pair) and self-invalidates
# when the secret rotates (new key → new cache entry).
@lru_cache(maxsize=8)
def _signer_for(secret: str, salt: str) -> TimestampSigner:
    return TimestampSigner(secret, salt=salt)


def _session_signer() -> TimestampSigner:
    # `salt` namespaces the signature so a one-shot cookie value cannot
    # be re-used as a session cookie (and vice versa) even though both
    # share `WXYC_SESSION_SECRET`.
    return _signer_for(_session_secret(), "flowsheet-session")


def _one_shot_signer() -> TimestampSigner:
    return _signer_for(_session_secret(), "flowsheet-one-shot")


def encode_session(s: ReviewerSession) -> str:
    """Sign a `ReviewerSession` into a cookie value with `SESSION_TTL`
    enforcement baked in (the freshness check is part of `decode_session`).

    Security note: the payload is SIGNED, not encrypted. The cookie's
    base64-ish wrapping decodes to plaintext JSON containing email +
    name + dj_name + role. This is by design (matches itsdangerous's
    intent for session cookies and the plan's stance that reviewer
    identity is not station-confidential PII), but a future change
    that adds genuinely sensitive fields to ReviewerSession should
    switch to an encrypted serializer.
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
    except (BadSignature, ValueError, TypeError, RuntimeError):
        # BadSignature covers both tamper and expiry (SignatureExpired is
        # a subclass). ValueError covers malformed JSON. TypeError covers
        # a payload shape that doesn't match ReviewerSession's fields
        # (missing-required or unknown-keyword on __init__). RuntimeError
        # catches the env-unset case from `_session_secret()` — without
        # it, a config-reload bug that unsets WXYC_SESSION_SECRET
        # mid-process would 500 every request instead of redirecting to
        # /auth/login.
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

    Uses authlib's RFC 7636 helper — we already depend on authlib for
    JWT verification and previous hand-rolled code drifted from the
    library's accepted alphabet/padding conventions. The auth server
    is configured with `requirePKCE: true`, so a missing or malformed
    challenge fails the flow at authorize-time, not at token-exchange-time.
    """
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = create_s256_code_challenge(code_verifier)
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

    params = [
        ("response_type", "code"),
        ("client_id", _client_id()),
        ("redirect_uri", _redirect_uri()),
        # `openid` is required for an id_token; `profile email` get the
        # name / email / preferred_username claims we read in
        # `exchange_code`.
        ("scope", "openid profile email"),
        ("state", state),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
    ]
    authorize_endpoint = metadata.get("authorization_endpoint")
    if not authorize_endpoint:
        raise ValueError("discovery doc missing authorization_endpoint")
    # `add_params_to_uri` parses an existing query string on the endpoint
    # and merges params correctly — `f"{endpoint}?{urlencode}"` would
    # produce two `?`s when the endpoint already has a query (e.g., a
    # multi-tenant IdP using `?tenant=...`).
    url = add_params_to_uri(authorize_endpoint, params)
    return url, state, code_verifier


async def _fetch_token(metadata: dict[str, Any], code: str, code_verifier: str) -> dict[str, Any]:
    """POST to the token endpoint and return the JSON body.

    Surfaces transport failures as `httpx.HTTPError` and malformed
    responses as `ValueError`. The split is so the route layer can map
    transport → 503 and parse failure → 400 without a try/except on
    `json.JSONDecodeError` of its own.
    """
    token_endpoint = metadata.get("token_endpoint")
    if not token_endpoint:
        raise ValueError("discovery doc missing token_endpoint")
    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _redirect_uri(),
                "code_verifier": code_verifier,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
            },
            # Accept JSON so a content-negotiating server cannot return
            # form-urlencoded by default (which would parse-fail below).
            headers={"Accept": "application/json"},
        )
        token_response.raise_for_status()
        try:
            return token_response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(f"token endpoint returned non-JSON body: {exc}") from exc


def _verify_id_token(id_token_raw: str, jwks: dict[str, Any]) -> Any:
    """Verify the id_token signature + claims and return the claims object.

    Raises authlib's `JoseError` subclasses on signature, alg, or
    aud/iss claim failures. The `_signing_keys_only` filter prevents
    an encryption key in the JWKS from joining the trust pool used
    for signature verification.
    """
    # `aud` and `iss` are checked here, not somewhere downstream. Skipping
    # would let a Wiki.js-issued token (also signed by api.wxyc.org's JWKS)
    # authenticate to the verifier — that's a real cross-app confusion bug
    # the plan calls out explicitly.
    claims_options = {
        "iss": {"essential": True, "value": _issuer()},
        "aud": {"essential": True, "value": _client_id()},
        # `sub` is mandatory per OIDC Core §2; mark essential so authlib
        # raises a JoseError on a token missing it, instead of letting a
        # bare `KeyError` propagate from the field access below.
        "sub": {"essential": True},
    }
    keyset = JsonWebKey.import_key_set(_signing_keys_only(jwks))
    claims = _JWT.decode(
        id_token_raw,
        key=keyset,
        claims_options=claims_options,
    )
    claims.validate()
    return claims


async def exchange_code(*, code: str, code_verifier: str) -> ReviewerSession:
    """Exchange an authorization code for an id_token, verify it, return
    the reviewer.

    Failure modes the caller must handle:

      * `httpx.HTTPError` — token endpoint or JWKS endpoint unreachable
        / non-2xx (includes `httpx.HTTPStatusError` for 4xx from the
        token endpoint, e.g., replayed code → `invalid_grant`). Route
        layer translates to 503.
      * `authlib.jose.errors.JoseError` (and subclasses, including
        `InvalidClaimError` for `aud` / `iss` / `sub` failures, and
        `BadSignatureError` for an id_token signed by an unknown key
        AFTER a JWKS refresh has already been attempted) — the
        id_token is structurally valid but its claims or signature
        are wrong. Route layer translates to 400.
      * `ValueError` — token response was 2xx but malformed (missing
        `id_token`, non-JSON body) or the discovery doc lacks a
        required endpoint URL. Route layer translates to 400.

    On a single `BadSignatureError` the JWKS cache is invalidated and
    one re-verification attempt is made against the fresh JWKS — this
    recovers automatically from an auth-server key rotation without a
    process restart.
    """
    metadata = await _load_metadata()
    body = await _fetch_token(metadata, code, code_verifier)
    id_token_raw = body.get("id_token")
    if not id_token_raw:
        raise ValueError("token response missing id_token field")

    jwks = await _load_jwks()
    try:
        claims = _verify_id_token(id_token_raw, jwks)
    except BadSignatureError:
        # The cached JWKS may pre-date a signing-key rotation on the
        # auth server. Drop it, refetch, and retry exactly once. If the
        # second attempt also fails the BadSignatureError propagates as
        # the documented JoseError path.
        _invalidate_jwks_cache()
        jwks = await _load_jwks()
        claims = _verify_id_token(id_token_raw, jwks)

    # Better Auth's OIDC provider includes `preferred_username`, `email`,
    # `name`, and any extra claims `getAdditionalUserInfoClaim` returns
    # (the auth.definition.ts setup pulls `dj_name` and `role` from the
    # WXYC user record). Each is optional — a user with no DJ name is
    # fine, we just don't render one. `_first_present` distinguishes
    # 'claim missing' (treated as None) from 'claim emitted as empty
    # string' (preserved as ''), so a misconfigured IdP that emits an
    # empty preferred_username doesn't silently fall through to a
    # different field.
    return ReviewerSession(
        user_id=str(claims["sub"]),
        email=_optional_str(claims, "email"),
        username=_first_present(claims, "preferred_username", "username"),
        real_name=_first_present(claims, "name", "real_name"),
        dj_name=_optional_str(claims, "dj_name"),
        role=_optional_str(claims, "role"),
    )


def _optional_str(claims: Any, name: str) -> str | None:
    """Return the claim value as a str, or None if the claim is absent
    or explicitly null.

    Unlike `claims.get(name) or None`, this preserves an empty-string
    value — useful when distinguishing 'claim absent' from 'claim
    explicitly empty', which the falsy-coalesce form collapses.

    JSON null is treated identically to 'absent' (both → None). A
    naive `str(claims[name])` would coerce Python None to the literal
    string `"None"`, which is exactly the silent corruption the
    Optional[str] field was introduced to prevent.
    """
    value = claims.get(name)
    if value is None:
        return None
    return str(value)


def _first_present(claims: Any, *names: str) -> str | None:
    """Return the first claim from `names` whose value is not absent
    and not null, or None if none qualify.

    Distinguishes 'present-with-content' (preserved, including empty
    string) from 'absent' and 'null' (both treated as 'not really
    there, try the next field'). Without the null-is-absent rule, a
    `null` first claim would short-circuit and return the literal
    string `"None"` rather than falling through to the second name.
    """
    for name in names:
        value = claims.get(name)
        if value is not None:
            return str(value)
    return None
