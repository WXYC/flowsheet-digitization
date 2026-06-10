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

import json
import time
from typing import Any

import httpx
import pytest
from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import JoseError
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
    don't need to catch a tamper exception."""
    signed = encode_session(_make_reviewer())
    # Flip the last char; signature won't verify.
    tampered = signed[:-1] + ("X" if signed[-1] != "X" else "Y")
    assert decode_session(tampered) is None


def test_decode_session_returns_none_on_garbage() -> None:
    """Empty string and obvious garbage produce None, not an exception."""
    assert decode_session("") is None
    assert decode_session("not-a-real-cookie") is None
    assert decode_session("a.b.c") is None


def test_decode_session_returns_none_when_signed_with_different_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cookie signed with one secret cannot be decoded after the secret
    rotates. Operationally this is the "all sessions invalid after
    deploy with a new secret" property the plan calls out.
    """
    signed_old = encode_session(_make_reviewer())
    # Rotate to a new secret. Cache reset is needed because the signer
    # picks up the env var at instantiation time.
    monkeypatch.setenv("WXYC_SESSION_SECRET", "y" * 64)
    auth_mod._reset_metadata_cache()
    assert decode_session(signed_old) is None


# -- one-shot cookies (state / verifier / return_to) -----------------------


def test_sign_verify_one_shot_round_trips() -> None:
    signed = sign_one_shot("state-token-abc123")
    assert (
        verify_one_shot(signed, max_age=int(ONE_SHOT_TTL.total_seconds())) == "state-token-abc123"
    )


def test_verify_one_shot_raises_on_tamper() -> None:
    """A tampered one-shot must raise — the callback handler relies on
    this to refuse a forged `oidc_state` even if the attacker knows the
    payload shape."""
    signed = sign_one_shot("verifier-token")
    tampered = signed[:-1] + ("X" if signed[-1] != "X" else "Y")
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


@pytest.fixture
def signing_key() -> Any:
    """Generate an RSA keypair used to sign the test id_token.

    The public half is exposed via the stubbed `_load_jwks`; the private
    half signs the id_token in `_mint_id_token`. Mirrors what the auth
    server's JWKS endpoint would publish.
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
) -> None:
    """An id_token signed by a key NOT in the JWKS must fail signature
    verification. Tests with a fresh keypair whose public half was never
    added to the stubbed JWKS."""
    foreign_key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
    _install_token_endpoint(monkeypatch, _mint_id_token(foreign_key))
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
