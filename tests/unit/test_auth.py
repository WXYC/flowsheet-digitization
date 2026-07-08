"""Tests for `core/auth.py` — OIDC client + session-cookie codec.

`core/auth.py` is the pure module behind the verifier's OIDC dance. The
HTTP routes (login/callback/logout/me) and the request-gating middleware
live in `verifier/serve.py` and are tested in `test_verifier_auth_routes.py`.

What we cover here:

  * `encode_session` / `decode_session` round-trip the `ReviewerSession`
    payload through a signed cookie value. Tampering, malformed input,
    or a wrong secret each produce `None` (never an exception leaked to
    the request path).
  * `sign_one_shot` / `verify_one_shot` round-trip arbitrary short
    strings (used for `oidc_state`, `oidc_verifier`, `oidc_return_to`
    cookies during the OIDC redirect dance). Past `max_age` or any
    tamper raises.
  * `build_authorize_url` produces a URL with PKCE + state and the
    matching `code_verifier` returned alongside.
  * `exchange_code` happy path returns a `ReviewerSession` populated
    from the id_token claims. Mismatched `aud`, mismatched `iss`, and a
    signature signed by an unknown key each raise. A transport-level
    error on the token endpoint propagates `httpx.HTTPError` (the route
    layer translates this to 503).
  * `_reset_metadata_cache()` clears module-level discovery + JWKS so
    test cases don't bleed metadata into each other.

All OIDC tests stub the discovery doc + JWKS at the `_load_metadata`
/ `_load_jwks` seam — the plan calls this out as the deliberate test
boundary (no DI on `core/auth.py`'s httpx client, by design).
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest
from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import (
    BadSignatureError,
    InvalidClaimError,
    JoseError,
    MissingClaimError,
)
from itsdangerous import BadSignature

import core.auth as auth_mod
from core.auth import (
    COOKIE_NAME,
    ONE_SHOT_TTL,
    SESSION_TTL,
    ReviewerSession,
    build_authorize_url,
    decode_session,
    encode_session,
    exchange_code,
    sign_one_shot,
    verify_one_shot,
)

ISSUER = "https://auth.example/auth"
CLIENT_ID = "flowsheet-test"
CLIENT_SECRET = "test-client-secret"
PUBLIC_URL = "https://flowsheet.example"
SESSION_SECRET = "x" * 64


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the 5 OIDC env vars and clear the module-level metadata cache.

    Autouse so every test in this file runs against a known config.
    """
    monkeypatch.setenv("WXYC_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("WXYC_SESSION_SECRET", SESSION_SECRET)
    monkeypatch.setenv("FLOWSHEET_PUBLIC_URL", PUBLIC_URL)
    auth_mod._reset_metadata_cache()
    yield
    auth_mod._reset_metadata_cache()


def _b64url(data: bytes) -> str:
    """Base64url-encode `data` without padding, matching the JWT compact-
    serialization convention. Used by tests that hand-craft raw JWTs
    (`alg: none`, unsupported-alg probes, invalid-UTF-8 headers) —
    authlib's `encode()` refuses to mint these by design."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_reviewer(**overrides: Any) -> ReviewerSession:
    base = {
        "user_id": "u-123",
        "email": "dj@wxyc.org",
        "username": "dj_radio",
        "real_name": "DJ Radio",
        "dj_name": "Radio Free",
        "role": "dj",
    }
    base.update(overrides)
    return ReviewerSession(**base)


# -- session cookie round-trip --------------------------------------------


def test_encode_decode_session_round_trips_all_fields() -> None:
    """Every field on `ReviewerSession` survives encode → decode without loss.

    Pins the contract that the cookie is a faithful transport for the
    reviewer record — the route layer can trust the decoded value to be
    equal to what was signed.
    """
    original = _make_reviewer()
    signed = encode_session(original)
    decoded = decode_session(signed)
    assert decoded == original


def test_decode_session_returns_none_on_tamper() -> None:
    """A flipped byte in the cookie value verifies as None rather than
    raising — callers treat None as "no session, send to login" and
    don't need to catch a tamper exception.

    Replace the last 4 chars (always inside the HMAC signature segment,
    since itsdangerous signatures are ~27 chars). A 1-char tamper is
    technically valid but intermittently flakes — base64url's 27-char
    signature has 2 padding bits in the last char, so 4-of-64 random
    last-char replacements decode to the same HMAC bytes; replacing 4
    chars touches multiple HMAC bytes so verification deterministically
    fails.
    """
    signed = encode_session(_make_reviewer())
    tampered = signed[:-4] + "----"
    assert decode_session(tampered) is None


def test_decode_session_returns_none_on_garbage() -> None:
    """Empty string and obvious garbage produce None, not an exception."""
    assert decode_session("") is None
    assert decode_session("not-a-real-cookie") is None
    assert decode_session("a.b.c") is None


def test_decode_session_returns_none_when_session_secret_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config-reload bug that unsets WXYC_SESSION_SECRET mid-process
    must not 500 every gated request — the middleware contract is
    'None → redirect to /auth/login'. Pins the RuntimeError catch in
    decode_session so a future refactor that drops it is caught by a
    test rather than by users.
    """
    signed = encode_session(_make_reviewer())
    monkeypatch.delenv("WXYC_SESSION_SECRET", raising=False)
    assert decode_session(signed) is None


def test_invalidate_jwks_cache_clears_only_jwks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_invalidate_jwks_cache` must drop `_jwks` without touching
    `_metadata`. Pins the cache-invalidation mechanism the rotation
    retry path relies on — the rotation integration test monkeypatches
    `_load_jwks` so it doesn't actually exercise the cache state; this
    test does.
    """
    # Seed the module-level cache directly so we don't depend on the
    # network or on _load_*.
    auth_mod._metadata = {"jwks_uri": "https://example/jwks"}
    auth_mod._jwks = {"keys": [{"kid": "old"}]}
    auth_mod._invalidate_jwks_cache()
    assert auth_mod._jwks is None
    assert auth_mod._metadata == {"jwks_uri": "https://example/jwks"}
    # The autouse `_env` fixture's teardown clears both caches; no
    # explicit reset needed here.


def test_decode_session_returns_none_when_signed_with_different_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cookie signed with one secret cannot be decoded after the secret
    rotates. Operationally this is the "all sessions invalid after
    deploy with a new secret" property the plan calls out.
    """
    signed_old = encode_session(_make_reviewer())
    # Rotate to a new secret. The next `_session_signer()` call passes a
    # different cache key to `_signer_for` and gets a different
    # TimestampSigner instance — no manual cache reset needed since the
    # lru_cache is keyed on the secret itself.
    monkeypatch.setenv("WXYC_SESSION_SECRET", "y" * 64)
    assert decode_session(signed_old) is None


def test_decode_session_returns_none_when_past_session_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cookie older than SESSION_TTL must decode as None. Pins the
    12-hour freshness check at the codec boundary so a refactor that
    drops `max_age` from the `unsign` call would surface here rather
    than silently extending session lifetime.

    Implementation: encode now, hold the wall clock SESSION_TTL + 60s
    forward, then decode. itsdangerous reads `time.time()` at unsign
    time so the patched clock makes the cookie appear stale.
    """
    signed = encode_session(_make_reviewer())
    future = time.time() + SESSION_TTL.total_seconds() + 60
    monkeypatch.setattr(time, "time", lambda: future)
    assert decode_session(signed) is None


# -- one-shot cookies (state / verifier / return_to) -----------------------


def test_sign_verify_one_shot_round_trips() -> None:
    signed = sign_one_shot("state-token-abc123")
    assert (
        verify_one_shot(signed, max_age=int(ONE_SHOT_TTL.total_seconds())) == "state-token-abc123"
    )


def test_verify_one_shot_raises_on_tamper() -> None:
    """A tampered one-shot must raise — the callback handler relies on
    this to refuse a forged `oidc_state` even if the attacker knows the
    payload shape.

    See `test_decode_session_returns_none_on_tamper` for why we replace
    4 chars rather than 1 (deterministic vs probabilistic tamper).
    """
    signed = sign_one_shot("verifier-token")
    tampered = signed[:-4] + "----"
    with pytest.raises(BadSignature):
        verify_one_shot(tampered, max_age=int(ONE_SHOT_TTL.total_seconds()))


def test_verify_one_shot_raises_when_past_max_age(monkeypatch: pytest.MonkeyPatch) -> None:
    """A signature older than `max_age` seconds must raise. Verified by
    holding the clock 1s forward of the signature time and passing
    max_age=0 — the rejection branch is exercised without sleeping.
    """
    signed = sign_one_shot("expired-token")
    # itsdangerous reads the wall clock at unsign time; with max_age=0,
    # any nonzero elapsed time triggers expiry. The signature has a
    # ~1s-old timestamp by the time we unsign it; max_age=0 forces the
    # expiry branch.
    fixed_now = time.time() + 1
    monkeypatch.setattr(time, "time", lambda: fixed_now)
    with pytest.raises(BadSignature):
        verify_one_shot(signed, max_age=0)


# -- OIDC test fixtures ----------------------------------------------------


@pytest.fixture(scope="module")
def signing_key() -> Any:
    """Generate an RSA keypair used to sign the test id_token.

    Module-scoped because RSA-2048 keygen is ~200-500ms per call and the
    keypair is content-static (same kid, same modulus) — every test in
    this file that needs a 'valid signing key' wants the same one. The
    plan's tests aren't asserting key-content so sharing is safe.

    The public half is exposed via the stubbed `_load_jwks`; the private
    half signs the id_token in `_mint_id_token`. Mirrors what the auth
    server's JWKS endpoint would publish.
    """
    return JsonWebKey.generate_key("RSA", 2048, is_private=True)


@pytest.fixture(scope="module")
def foreign_signing_key() -> Any:
    """A second RSA keypair NOT in the JWKS, used to mint id_tokens that
    must fail signature verification.

    Module-scoped for the same reason as `signing_key` — keygen is
    expensive and the test only cares that this key is NOT the trusted
    one.
    """
    return JsonWebKey.generate_key("RSA", 2048, is_private=True)


@pytest.fixture
def jwks_from_key(signing_key: Any) -> dict[str, Any]:
    """The JWKS the stubbed `_load_jwks` returns — just the public half."""
    pub = json.loads(signing_key.as_json(is_private=False))
    pub["kid"] = signing_key.kid
    return {"keys": [pub]}


@pytest.fixture
def stub_metadata(monkeypatch: pytest.MonkeyPatch, jwks_from_key: dict[str, Any]) -> dict[str, Any]:
    """Install the discovery doc + JWKS in the module-level cache.

    Tests that want to exercise the unhappy path (e.g., JWKS fetch
    failure) skip this fixture and set the cache themselves.
    """
    metadata = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/oauth2/authorize",
        "token_endpoint": f"{ISSUER}/oauth2/token",
        "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
    }

    async def _meta() -> dict[str, Any]:
        return metadata

    async def _jwks() -> dict[str, Any]:
        return jwks_from_key

    monkeypatch.setattr(auth_mod, "_load_metadata", _meta)
    monkeypatch.setattr(auth_mod, "_load_jwks", _jwks)
    return metadata


def _mint_id_token(signing_key: Any, **claim_overrides: Any) -> str:
    """Mint an RS256 id_token signed by `signing_key`.

    Defaults are a valid id_token for our `CLIENT_ID` and `ISSUER`;
    callers override `aud` / `iss` to exercise mismatch branches.
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "sub": "user-abc",
        "aud": CLIENT_ID,
        "iat": now,
        "exp": now + 600,
        "email": "reviewer@wxyc.org",
        "preferred_username": "reviewer",
        "name": "Real Name",
        "dj_name": "Stage Name",
        "role": "dj",
    }
    payload.update(claim_overrides)
    header = {"alg": "RS256", "kid": signing_key.kid}
    return JsonWebToken(["RS256"]).encode(header, payload, signing_key).decode("utf-8")


def _install_token_endpoint(monkeypatch: pytest.MonkeyPatch, id_token: str | Exception) -> None:
    """Stub `httpx.AsyncClient.post` so the token endpoint returns
    `{"id_token": ...}` (or raises, for the transport-failure test)."""

    class _Resp:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    async def fake_post(self, url: str, **kwargs: Any) -> _Resp:  # type: ignore[no-untyped-def]
        if isinstance(id_token, Exception):
            raise id_token
        return _Resp({"id_token": id_token, "token_type": "Bearer"})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)


# -- build_authorize_url ---------------------------------------------------


async def test_build_authorize_url_contains_pkce_and_state(
    stub_metadata: dict[str, Any],
) -> None:
    """The URL returned must include `code_challenge`, `code_challenge_method=S256`,
    a `state` matching the returned tuple element, the configured
    `client_id`, and the redirect_uri derived from `FLOWSHEET_PUBLIC_URL`.

    PKCE is required by the auth provider (`requirePKCE: true`); without
    these params the authorize call would be rejected at the server.
    """
    url, state, code_verifier = await build_authorize_url()

    assert url.startswith(stub_metadata["authorization_endpoint"])
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert f"state={state}" in url
    assert f"client_id={CLIENT_ID}" in url
    assert "response_type=code" in url
    # Redirect URI is FLOWSHEET_PUBLIC_URL + /auth/callback.
    assert "redirect_uri=https%3A%2F%2Fflowsheet.example%2Fauth%2Fcallback" in url
    # state and verifier are unguessable strings of meaningful length.
    assert len(state) >= 32
    assert len(code_verifier) >= 32


async def test_build_authorize_url_state_and_verifier_unique_per_call(
    stub_metadata: dict[str, Any],
) -> None:
    """Two back-to-back calls produce different state + code_verifier
    pairs. A reused state would defeat the CSRF check."""
    _, state1, verifier1 = await build_authorize_url()
    _, state2, verifier2 = await build_authorize_url()
    assert state1 != state2
    assert verifier1 != verifier2


# -- exchange_code ---------------------------------------------------------


async def test_exchange_code_happy_path_returns_reviewer(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """A valid signed id_token with matching `iss` + `aud` resolves to
    a `ReviewerSession` with all claim-derived fields populated."""
    _install_token_endpoint(monkeypatch, _mint_id_token(signing_key))
    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.user_id == "user-abc"
    assert reviewer.email == "reviewer@wxyc.org"
    assert reviewer.username == "reviewer"
    assert reviewer.real_name == "Real Name"
    assert reviewer.dj_name == "Stage Name"
    assert reviewer.role == "dj"


async def test_exchange_code_raises_on_aud_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """An id_token whose `aud` is another OIDC client (e.g. Wiki.js)
    must be rejected. Without this check, the flowsheet would accept
    sessions minted for a different relying party.
    """
    _install_token_endpoint(monkeypatch, _mint_id_token(signing_key, aud="wikijs-client-id"))
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_raises_on_iss_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """An id_token from a different issuer must be rejected even if it
    happens to address our `aud`."""
    _install_token_endpoint(
        monkeypatch, _mint_id_token(signing_key, iss="https://evil.example/auth")
    )
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_raises_on_unknown_signing_key(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    foreign_signing_key: Any,
) -> None:
    """An id_token signed by a key NOT in the JWKS must fail signature
    verification — even after the single JWKS-refresh retry, since the
    stubbed `_load_jwks` keeps returning the same trusted-key set.
    """
    _install_token_endpoint(monkeypatch, _mint_id_token(foreign_signing_key))
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_propagates_token_endpoint_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """If the auth server is unreachable, the HTTP error reaches the
    caller (the route layer translates it to 503)."""
    _install_token_endpoint(monkeypatch, httpx.ConnectError("auth server unreachable"))
    with pytest.raises(httpx.HTTPError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_propagates_4xx_from_token_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A 4xx from the token endpoint (e.g., `invalid_grant` on a replayed
    authorization code) surfaces as `httpx.HTTPStatusError`, a subclass
    of `HTTPError` the route layer's 503-mapping catches. Pins the
    raise_for_status branch — previously only ConnectError was exercised.
    """

    class _Resp:
        status_code = 400

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "400",
                request=httpx.Request("POST", "https://auth.example/oauth2/token"),
                response=httpx.Response(400),
            )

        def json(self) -> dict[str, Any]:
            return {"error": "invalid_grant"}

    async def fake_post(self, url: str, **kwargs: Any) -> _Resp:  # type: ignore[no-untyped-def]
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(httpx.HTTPError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_raises_value_error_when_id_token_missing(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A 2xx token response without an `id_token` field is a contract
    violation by the auth server — surface it as `ValueError` so the
    route layer's 400-mapping catches it rather than the prior raw
    `KeyError` that bypassed the documented exception contract.
    """

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            # No id_token — e.g., the openid scope was dropped or an
            # error response was returned with a 200 status.
            return {"access_token": "x", "token_type": "Bearer"}

    async def fake_post(self, url: str, **kwargs: Any) -> _Resp:  # type: ignore[no-untyped-def]
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ValueError, match="id_token"):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_raises_value_error_on_non_json_body(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A 2xx token response with a non-JSON body (e.g., HTML challenge
    page from a CDN) surfaces as `ValueError`, not the prior raw
    `json.JSONDecodeError` that the route layer never anticipated.
    """

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            raise json.JSONDecodeError("Expecting value", "<!DOCTYPE html>", 0)

    async def fake_post(self, url: str, **kwargs: Any) -> _Resp:  # type: ignore[no-untyped-def]
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    with pytest.raises(ValueError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_raises_missing_claim_when_sub_absent(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """An id_token missing the `sub` claim is rejected by authlib's
    essential-claim check (we declared `sub` essential in claims_options)
    rather than escaping as a raw `KeyError` at field-access time.

    Assert on `MissingClaimError` specifically rather than the broader
    `JoseError`, so a regression that flips `sub` out of `claims_options`
    (which would let the token reach the field access and raise a
    bare KeyError → would not be a JoseError at all) is distinguishable
    from a signature/aud/iss failure.
    """
    # Mint a token with `sub` removed. `_mint_id_token` overrides only
    # accept replacements, so build the payload manually.
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "iat": now,
        "exp": now + 600,
        "email": "x@y",
        # no `sub`
    }
    header = {"alg": "RS256", "kid": signing_key.kid}
    no_sub_token = JsonWebToken(["RS256"]).encode(header, payload, signing_key).decode("utf-8")
    _install_token_endpoint(monkeypatch, no_sub_token)
    with pytest.raises(MissingClaimError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_preserves_empty_string_username(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """A claim that's explicitly an empty string is preserved verbatim
    (distinguishes 'present-but-empty' from 'absent' / 'null'). Pins
    the docstring contract that the falsy-coalesce form was changed to
    avoid."""
    token = _mint_id_token(signing_key, preferred_username="")
    _install_token_endpoint(monkeypatch, token)
    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.username == ""


async def test_exchange_code_treats_null_claim_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """A claim emitted as JSON null becomes Python None on ReviewerSession,
    not the literal string 'None'. Pins the iter-2 fix to `_optional_str`
    / `_first_present` — without this, a `null` value would short-circuit
    the fallback chain and inject the four-character string 'None' into
    a field that's meant to be Optional[str]."""
    token = _mint_id_token(
        signing_key,
        email=None,
        preferred_username=None,
        username="djradio",  # fallback should be used
        dj_name=None,
        role=None,
    )
    _install_token_endpoint(monkeypatch, token)
    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.email is None
    # _first_present should have skipped the null preferred_username and
    # picked up username — the prior buggy version returned the string
    # 'None' here.
    assert reviewer.username == "djradio"
    assert reviewer.dj_name is None
    assert reviewer.role is None


async def test_exchange_code_handles_jwks_rotation(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    jwks_from_key: dict[str, Any],
    signing_key: Any,
) -> None:
    """After a single `BadSignatureError`, `exchange_code` invalidates
    the JWKS cache, refetches via `_load_jwks`, and retries decode.
    Simulated by having the first `_load_jwks` call return an empty key
    set (signature fails) and the second return the trusted set
    (signature passes) — mirrors what happens during a key rotation
    where the cached JWKS is stale but a refetch picks up the new key.
    """
    # Toggle: first call returns the wrong JWKS, second call returns
    # the right one.
    call_count = {"n": 0}

    async def _flipping_jwks() -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: an unrelated key the trust pool doesn't know.
            unrelated = JsonWebKey.generate_key("RSA", 2048, is_private=False)
            pub = json.loads(unrelated.as_json(is_private=False))
            pub["kid"] = "rotated-out"
            return {"keys": [pub]}
        return jwks_from_key

    monkeypatch.setattr(auth_mod, "_load_jwks", _flipping_jwks)
    _install_token_endpoint(monkeypatch, _mint_id_token(signing_key))

    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.user_id == "user-abc"
    assert call_count["n"] == 2, "_load_jwks should have been called twice"


# -- HS256 signing (Better Auth's oidcProvider default) -------------------
#
# api.wxyc.org/auth advertises id_token_signing_alg_values_supported=["HS256"].
# Better Auth's oidcProvider signs id_tokens with the client's client_secret
# as an HMAC key; the JWKS endpoint is still exposed (used by the JWT plugin
# for Bearer tokens) but is not the id_token trust root. `core/auth.py` must
# route HS256 tokens to `client_secret.encode()` and NEVER to a JWKS public
# key, or an attacker holding the public key could sign an HS256 token that
# the verifier accepts (classic alg-confusion attack).


def _mint_hs256_id_token(secret: str, **claim_overrides: Any) -> str:
    """Mint an HS256 id_token signed with `secret` as the HMAC key.

    Mirrors `_mint_id_token` but uses symmetric HMAC — matches how Better
    Auth's oidcProvider signs id_tokens in production.
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "sub": "user-abc",
        "aud": CLIENT_ID,
        "iat": now,
        "exp": now + 600,
        "email": "reviewer@wxyc.org",
        "preferred_username": "reviewer",
        "name": "Real Name",
        "dj_name": "Stage Name",
        "role": "dj",
    }
    payload.update(claim_overrides)
    header = {"alg": "HS256"}
    return JsonWebToken(["HS256"]).encode(header, payload, secret.encode("utf-8")).decode("utf-8")


async def test_exchange_code_hs256_happy_path_returns_reviewer(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A valid HS256 id_token signed with the configured `client_secret`
    verifies and returns a `ReviewerSession`. Pins the shape prod actually
    ships (Better Auth's oidcProvider default) — without HS256 support,
    turning on OIDC in Railway breaks every login at token-exchange."""
    _install_token_endpoint(monkeypatch, _mint_hs256_id_token(CLIENT_SECRET))
    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.user_id == "user-abc"
    assert reviewer.email == "reviewer@wxyc.org"
    assert reviewer.role == "dj"


async def test_exchange_code_rejects_hs256_signed_with_wrong_secret(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """An HS256 id_token signed with any secret OTHER than our configured
    `client_secret` fails signature verification. Confirms we're actually
    verifying — not just accepting anything with `alg: HS256`."""
    _install_token_endpoint(monkeypatch, _mint_hs256_id_token("not-our-secret"))
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rejects_hs256_signed_with_jwks_public_key(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    jwks_from_key: dict[str, Any],
) -> None:
    """Alg-confusion defense: an attacker who has read the auth server's
    JWKS (published, public data) cannot use one of the RSA public key's
    modulus bytes as an HMAC key and get an HS256 token accepted.

    This works ONLY if the code paths for HS256 and RS256 use disjoint
    key material: HS256 must key off `client_secret`, never the JWKS.
    A naïve implementation that just extended the alg list to
    `["HS256", "RS256"]` and passed both the client_secret and the JWKS
    to `decode()` would fall to this attack — that's the whole point of
    the CVE-2016-10555 family."""
    public_key = jwks_from_key["keys"][0]
    # `n` is the RSA modulus, the largest chunk of public bytes attackers
    # can lift straight from the JWKS. Use it as the HMAC key exactly as
    # the classic alg-confusion PoC does.
    attacker_secret = public_key["n"]
    _install_token_endpoint(monkeypatch, _mint_hs256_id_token(attacker_secret))
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rejects_alg_none(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A token with `alg: none` and no signature must be refused even
    though it structurally looks like a JWT. Not restricted by any prior
    test; asserts the defense stays intact after the HS256 change."""
    # Hand-craft the `alg: none` token — authlib refuses to mint one via
    # the standard `encode()` path (which is the correct default).
    header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode("utf-8"))
    now = int(time.time())
    payload = _b64url(
        json.dumps(
            {
                "iss": ISSUER,
                "sub": "user-abc",
                "aud": CLIENT_ID,
                "iat": now,
                "exp": now + 600,
            }
        ).encode("utf-8")
    )
    alg_none_token = f"{header}.{payload}."
    _install_token_endpoint(monkeypatch, alg_none_token)
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_hs256_does_not_load_jwks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HS256 verification is keyed off `client_secret`, not the JWKS — so
    an HS256 login must NOT reach `_load_jwks()`. Prod (`api.wxyc.org`)
    advertises HS256 only, and a JWKS-endpoint outage would otherwise
    take every login offline for no gain. This pins the invariant that
    the asymmetric-only fetch stays in the asymmetric branch."""

    metadata = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/oauth2/authorize",
        "token_endpoint": f"{ISSUER}/oauth2/token",
        "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
    }

    async def _meta() -> dict[str, Any]:
        return metadata

    async def _jwks_that_fails() -> dict[str, Any]:
        raise AssertionError("_load_jwks must not be called for HS256")

    monkeypatch.setattr(auth_mod, "_load_metadata", _meta)
    monkeypatch.setattr(auth_mod, "_load_jwks", _jwks_that_fails)
    _install_token_endpoint(monkeypatch, _mint_hs256_id_token(CLIENT_SECRET))

    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.user_id == "user-abc"


async def test_exchange_code_hs256_bad_signature_does_not_invalidate_jwks(
    monkeypatch: pytest.MonkeyPatch,
    jwks_from_key: dict[str, Any],
) -> None:
    """A failed HS256 verification must NOT invalidate the JWKS cache.
    The JWKS is irrelevant to HS256 (client_secret is the key material),
    and a bad-secret HS256 login that flushed the cache would (a) waste
    an httpx round-trip on the pointless refetch, and (b) amplify
    auth-server load on a brute-force by evicting the cache for every
    other in-flight asymmetric login on the process. Pins the invariant
    that JWKS-rotation-recovery is scoped to the branch that actually
    consumes the JWKS."""

    metadata = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/oauth2/authorize",
        "token_endpoint": f"{ISSUER}/oauth2/token",
        "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
    }

    async def _meta() -> dict[str, Any]:
        return metadata

    jwks_calls = {"n": 0}

    async def _jwks_counting() -> dict[str, Any]:
        jwks_calls["n"] += 1
        return jwks_from_key

    invalidate_calls = {"n": 0}

    def _invalidate_counting() -> None:
        invalidate_calls["n"] += 1

    monkeypatch.setattr(auth_mod, "_load_metadata", _meta)
    monkeypatch.setattr(auth_mod, "_load_jwks", _jwks_counting)
    monkeypatch.setattr(auth_mod, "_invalidate_jwks_cache", _invalidate_counting)
    _install_token_endpoint(monkeypatch, _mint_hs256_id_token("not-the-secret"))

    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")
    assert jwks_calls["n"] == 0, "HS256 failure must not touch JWKS"
    # Also pin that the cache-eviction function itself isn't called — a
    # bug that speculatively evicted without refetching would leave the
    # `_load_jwks` counter at 0 and pass the assertion above, while still
    # clobbering the cached JWKS used by legitimate asymmetric traffic.
    assert invalidate_calls["n"] == 0, "HS256 failure must not invalidate JWKS cache"


async def test_exchange_code_hs256_rejects_wrong_aud(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """An HS256 id_token whose `aud` targets a different OIDC client
    (e.g. Wiki.js) must be rejected. Pins cross-app isolation for the
    symmetric branch — the `claims_options` mechanism that closed this
    class of confusion bug for RS256 also applies for HS256, and this
    test fires loudly if a refactor bypasses claims_options on the
    symmetric path."""
    _install_token_endpoint(
        monkeypatch, _mint_hs256_id_token(CLIENT_SECRET, aud="wikijs-client-id")
    )
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_hs256_rejects_wrong_iss(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """An HS256 id_token minted by a different issuer must be rejected
    even if it addresses our `aud`. Mirrors the RS256 iss-check test for
    the symmetric verify path so an issuer-forgery attempt against a
    known client_secret still hits the essential-claim check."""
    _install_token_endpoint(
        monkeypatch,
        _mint_hs256_id_token(CLIENT_SECRET, iss="https://evil.example/auth"),
    )
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rejects_non_utf8_header(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """A token whose header segment base64url-decodes to bytes that are
    not valid UTF-8 must surface as `DecodeError`, not the raw
    `UnicodeDecodeError` that `json.loads(bytes)` produces on invalid
    UTF-8 input. Pins the regression fixed in round 2: an earlier
    revision narrowed the except clause to `json.JSONDecodeError` alone
    (which does NOT catch UnicodeDecodeError), letting attacker-crafted
    non-UTF-8 headers escape the DecodeError contract and 500 the route
    layer instead of the documented 400."""
    # Header segment whose bytes are not valid UTF-8. \x80 is a
    # continuation byte with no lead byte; \xc0\x28 is an over-long
    # 2-byte sequence encoding an ASCII char, which is also invalid.
    invalid_utf8 = base64.urlsafe_b64encode(b"\x80\xc0\x28").rstrip(b"=").decode("ascii")
    token = f"{invalid_utf8}.eyJzdWIiOiJ4In0.deadbeef"
    _install_token_endpoint(monkeypatch, token)
    with pytest.raises(JoseError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rotation_refetch_failure_reraises_signature_error(
    monkeypatch: pytest.MonkeyPatch,
    foreign_signing_key: Any,
    jwks_from_key: dict[str, Any],
) -> None:
    """When the initial JWKS-based verify fails with `BadSignatureError`
    and the rotation-retry refetch itself raises `httpx.HTTPError`, the
    ORIGINAL signature failure must be re-raised (route: 400 'invalid
    token'), NOT the transport error (route: 503 'auth unavailable').

    Otherwise an attacker probing with a forged token during a brief
    JWKS-endpoint outage sees their 400-worthy forgery obscured as a
    503 auth incident — misleads on-call, dilutes attack telemetry."""
    metadata = {
        "issuer": ISSUER,
        "authorization_endpoint": f"{ISSUER}/oauth2/authorize",
        "token_endpoint": f"{ISSUER}/oauth2/token",
        "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
    }

    async def _meta() -> dict[str, Any]:
        return metadata

    calls = {"n": 0}

    async def _jwks_first_ok_then_500(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            # First fetch succeeds — signature will fail against foreign key
            return jwks_from_key
        # Refetch (rotation-retry): JWKS endpoint blip
        raise httpx.ConnectError("jwks endpoint unavailable")

    monkeypatch.setattr(auth_mod, "_load_metadata", _meta)
    monkeypatch.setattr(auth_mod, "_load_jwks", _jwks_first_ok_then_500)
    _install_token_endpoint(monkeypatch, _mint_id_token(foreign_signing_key))

    with pytest.raises(BadSignatureError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rejects_token_missing_exp(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """A token without an `exp` claim must be rejected. authlib's
    `validate_exp` silently no-ops when the claim is absent — verified
    against authlib 1.6.x — so without an essential-claim declaration,
    a token issued with no expiration would verify forever. This test
    pins the round-3 hardening: any accepted id_token has a real
    expiration authlib is checking against wall clock."""
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "sub": "user-abc",
        "aud": CLIENT_ID,
        "iat": now,
        # no `exp`
    }
    header = {"alg": "RS256", "kid": signing_key.kid}
    no_exp_token = JsonWebToken(["RS256"]).encode(header, payload, signing_key).decode("utf-8")
    _install_token_endpoint(monkeypatch, no_exp_token)
    with pytest.raises(MissingClaimError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_rejects_multi_aud_with_extra_audience(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """A token whose `aud` is a list containing our client_id AND another
    audience must be rejected — even though authlib's default
    `validate_aud` accepts it (any-of-set-membership semantics). Per OIDC
    Core 3.1.3.7 §4-5, extra audiences must be explicitly trusted, and
    multi-valued `aud` requires an `azp` claim we verify. We trust
    no extra audience by default; require `aud == [client_id]`."""
    token = _mint_id_token(signing_key, aud=[CLIENT_ID, "wikijs-client-id"])
    _install_token_endpoint(monkeypatch, token)
    with pytest.raises(InvalidClaimError):
        await exchange_code(code="auth-code", code_verifier="verifier")


async def test_exchange_code_accepts_single_element_aud_list(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
    signing_key: Any,
) -> None:
    """An id_token whose `aud` is a single-element list containing our
    client_id must be accepted (equivalent to the scalar form per OIDC
    Core). Pins that the round-3 multi-aud defense doesn't over-reach
    into rejecting a spec-legal shape."""
    token = _mint_id_token(signing_key, aud=[CLIENT_ID])
    _install_token_endpoint(monkeypatch, token)
    reviewer = await exchange_code(code="auth-code", code_verifier="verifier")
    assert reviewer.user_id == "user-abc"


# -- JWKS filter -----------------------------------------------------------


def test_signing_keys_only_drops_hs256_jwk_entry() -> None:
    """`_signing_keys_only` filters JWK entries whose `alg` isn't in the
    asymmetric allowlist. An `alg=HS256` JWK entry — nonsensical but
    publishable — must never join the asymmetric trust pool: if it did,
    an RS256/ES256 verify call could receive HMAC-shaped key material
    and reopen the alg-confusion attack the header dispatch closes.
    Pins the invariant so a future refactor that merges the symmetric
    and asymmetric constant sets fails loudly here rather than silently
    regressing verifier security."""
    jwks = {
        "keys": [
            {"kid": "hmac-1", "kty": "oct", "alg": "HS256", "k": "AAAA", "use": "sig"},
            {"kid": "rsa-1", "kty": "RSA", "alg": "RS256", "n": "n-value", "e": "AQAB"},
        ]
    }
    filtered = auth_mod._signing_keys_only(jwks)
    kids = [k.get("kid") for k in filtered["keys"]]
    assert kids == ["rsa-1"]


async def test_unsupported_algorithm_error_names_offending_alg(
    monkeypatch: pytest.MonkeyPatch,
    stub_metadata: dict[str, Any],
) -> None:
    """When `_verify_id_token` rejects an unknown alg, the raised
    exception must (a) carry authlib's canonical
    `error='unsupported_algorithm'` short code (observability filters
    group on this) and (b) name the offending alg in `description` so
    on-call can diagnose 'why is my token being rejected' from prod
    logs without reproducing the raw token. Passing the human message
    positionally clobbers the `error` slot — pin both invariants."""
    header = _b64url(json.dumps({"alg": "HS384", "typ": "JWT"}).encode("utf-8"))
    now = int(time.time())
    payload = _b64url(
        json.dumps(
            {"iss": ISSUER, "sub": "u", "aud": CLIENT_ID, "iat": now, "exp": now + 600}
        ).encode("utf-8")
    )
    hs384_token = f"{header}.{payload}.deadbeef"
    _install_token_endpoint(monkeypatch, hs384_token)

    with pytest.raises(JoseError) as exc_info:
        await exchange_code(code="auth-code", code_verifier="verifier")
    exc = exc_info.value
    assert exc.error == "unsupported_algorithm", (
        f"authlib short code clobbered: error={exc.error!r}"
    )
    assert "HS384" in (exc.description or "")


# -- _reset_metadata_cache -------------------------------------------------


async def test_reset_metadata_cache_forces_re_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a reset, the next `_load_metadata()` call hits httpx
    again. Without this property, tests that change `WXYC_AUTH_ISSUER`
    would silently observe the previous test's cached metadata.
    """
    calls: list[str] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "issuer": ISSUER,
                "authorization_endpoint": f"{ISSUER}/oauth2/authorize",
                "token_endpoint": f"{ISSUER}/oauth2/token",
                "jwks_uri": f"{ISSUER}/.well-known/jwks.json",
            }

    async def fake_get(self, url: str, **kwargs: Any) -> _Resp:  # type: ignore[no-untyped-def]
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    # First call: cache cold, makes a real request.
    await auth_mod._load_metadata()
    assert len(calls) == 1
    # Second call: cache warm, no new request.
    await auth_mod._load_metadata()
    assert len(calls) == 1
    # Reset → next call refetches.
    auth_mod._reset_metadata_cache()
    await auth_mod._load_metadata()
    assert len(calls) == 2


# -- module-level constants ------------------------------------------------


def test_module_constants_match_plan() -> None:
    """SESSION_TTL is 12 hours; ONE_SHOT_TTL is 10 minutes; cookie name
    is `flowsheet_session`. These are stable contract surface — a UI or
    middleware that hard-codes them shouldn't break silently on a typo."""
    assert SESSION_TTL.total_seconds() == 12 * 60 * 60
    assert ONE_SHOT_TTL.total_seconds() == 10 * 60
    assert COOKIE_NAME == "flowsheet_session"
