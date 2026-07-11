# Plan: Reviewer identity via OIDC against api.wxyc.org/auth

## Motivation

The verifier currently writes `verified.json` and updates `jobs.db` without any record of which volunteer reviewed the page. As we scale volunteer reviewers, we need to track who reviewed what — both to enable per-reviewer leaderboards / queues and to flag quality issues back to the reviewer who introduced them. The minimum capability is "every `verified.json` and every `mark_verified` call carries a stable `user_id` for a station member."

The wider context is that WXYC already runs a Better Auth service at `api.wxyc.org/auth` (Backend-Service `apps/auth/`), with users, organizations, roles, and a JWKS endpoint. Three frontends already authenticate against it: dj-site (browser session + cookie), Wiki.js (OIDC), and any Express service that imports `@wxyc/authentication` (JWKS-verified Bearer JWT). Adding a fourth — the flowsheet verifier — should mean adding *one* `trustedClient` entry on the auth server, not standing up a parallel auth surface.

This plan implements the OIDC code-flow path from a previous design exploration (option B in the conversation that produced this plan): flowsheet-digitization becomes an OIDC client of `api.wxyc.org/auth`, indistinguishable from how Wiki.js does it today. The HTTP Basic Auth gate currently in `verifier/serve.py` (`_BasicAuthMiddleware`) is replaced with a session-cookie gate backed by an OIDC login round-trip.

## Why OIDC (option B) over the alternatives

Three shapes were considered:

| Option | Shape | Outcome |
|---|---|---|
| A | Redirect to `dj.wxyc.org/login?returnTo=...`, dj-site sets the api.wxyc.org session cookie, redirects back | Works, but conflates dj-site with "the auth provider" — gets awkward when a fifth app needs the same treatment. |
| **B** | **Add flowsheet as a `trustedClient` of the existing `oidcProvider()` plugin; run standard OIDC code + PKCE flow in `verifier/serve.py`** | **Wiki.js already does exactly this. Zero new patterns. Portable to app #5, #6.** |
| C | Build an email-OTP login form in the verifier UI that POSTs directly to `api.wxyc.org/auth/sign-in/email-otp/*` | Cheapest first cut, but commits us to maintaining a parallel login UI forever. |

B is chosen because the infrastructure is already in place (`oidcProvider({ trustedClients: [...] })` in `shared/authentication/src/auth.definition.ts:175-207`), the precedent (Wiki.js) is the working reference implementation, and the boundary it draws — "flowsheet is a confidential OIDC client of api.wxyc.org/auth" — is the same boundary any future internal app will draw.

## Scope

### In scope

- Adding flowsheet as a `trustedClient` on Backend-Service's `oidcProvider`.
- Replacing `verifier/serve.py`'s `_BasicAuthMiddleware` with an OIDC session-cookie middleware.
- New auth routes on the verifier FastAPI app: `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/me`.
- A new module `verifier/auth.py` holding the OIDC client + session cookie codec + reviewer dataclass.
- Threading `ReviewerSession` into `POST /api/save` so `verified.json` and `jobs.db` capture `reviewer_id`.
- A new `reviewer_id` column on the `jobs` table, written by `JobStore.mark_verified`.
- A small UI change in `verifier/app.js` to render the reviewer's name from `/auth/me` and to redirect to `/auth/login` on `401`.
- A local-dev story that doesn't require touching Backend-Service most of the time.

### Deliberately out of scope

- **Refresh tokens.** 12-hour signed session cookie; re-login on expiry. Additive to revisit later.
- **Calling Backend-Service APIs *as the user*.** The verifier's existing endpoints (`/api/save`, `/api/lookup`, `/api/bundles`) are server-side work that the FastAPI does on its own behalf; they don't need to forward the user's JWT. If a future endpoint needs to act on behalf of the user against a Backend-Service API, we add access-token storage and forwarding then.
- **Logout-everywhere / single logout.** `POST /auth/logout` clears the flowsheet session cookie only. The Better Auth session cookie on `api.wxyc.org` is unaffected. This matches Wiki.js's behavior.
- **Per-reviewer queues, leaderboards, quality flagging.** This plan delivers the *identity*. The features built on top of it are separate work.
- **A Python port of `@wxyc/authentication`'s `requirePermissions`.** We don't gate any verifier endpoint on a station role today; everyone who can sign in can review. Role enforcement is additive when we need it.
- **Migrating the existing BasicAuth deployment immediately.** The middleware stays available as a fallback (gated on env vars) so we can ship the OIDC code without an atomic flip in production.

## Constraints driving the design

- **`requirePKCE: true` is already set on the OIDC provider** (`auth.definition.ts:178`). The client must implement PKCE; `authlib`'s `AsyncOAuth2Client` does this with `code_challenge_method="S256"`.
- **Cookie scope must be the verifier's own host**, not `.wxyc.org`. The session cookie carries reviewer identity; leaking it to dj-site is a credential-confusion bug waiting to happen. `httponly=True`, `secure=True` in prod, `samesite="lax"` (required for the OIDC redirect-back to carry the cookie).
- **`secrets.compare_digest` for state comparison.** Already the precedent in `_BasicAuthMiddleware`; the OIDC `state` parameter check must follow the same rule.
- **`return_to` must be a relative path** to defuse the open-redirect risk on `/auth/login?return_to=...`. Allowlist: must start with `/verifier`. Anything else falls back to `/verifier/`.
- **`id_token.aud` must equal the flowsheet client_id.** Skipping the audience check would let a Wiki.js-issued token authenticate to the verifier.
- **Backend-Service is a separate repo with its own CI / deploy cadence.** Coordinating across two repos means the rollout has to handle the transition where the OIDC client is configured on the auth server but the verifier hasn't shipped yet (no-op) *and* where the verifier ships before the auth server is updated (env-var-gated fallback to BasicAuth).

## File-by-file diff

### Backend-Service

#### `shared/authentication/src/auth.definition.ts`

The current `trustedClients` is an inline literal inside a module-top-level `betterAuth({...})` call, which makes it impossible to unit-test in isolation. Extract the array construction into a small helper that takes env vars and returns the array. The helper is the test surface; the `betterAuth({...})` site just calls it.

```ts
// new export, lives in auth.definition.ts above the betterAuth call
export type TrustedClient = {
  clientId: string;
  clientSecret: string;
  redirectUrls: string[];
  name: string;
  type: 'web';
  disabled: boolean;
  icon: undefined;
  metadata: null;
  skipConsent: boolean;
};

export function buildTrustedClients(env: NodeJS.ProcessEnv = process.env): TrustedClient[] {
  const clients: TrustedClient[] = [];

  if (env.WIKIJS_OIDC_CLIENT_ID && env.WIKIJS_OIDC_CLIENT_SECRET && env.WIKIJS_URL) {
    clients.push({
      clientId: env.WIKIJS_OIDC_CLIENT_ID,
      clientSecret: env.WIKIJS_OIDC_CLIENT_SECRET,
      redirectUrls: [`${env.WIKIJS_URL}/login/oidc/callback`],
      name: 'Wiki.js',
      type: 'web',
      disabled: false,
      icon: undefined,
      metadata: null,
      skipConsent: true,
    });
  }

  if (env.FLOWSHEET_OIDC_CLIENT_ID && env.FLOWSHEET_OIDC_CLIENT_SECRET) {
    clients.push({
      clientId: env.FLOWSHEET_OIDC_CLIENT_ID,
      clientSecret: env.FLOWSHEET_OIDC_CLIENT_SECRET,
      redirectUrls: (env.FLOWSHEET_OIDC_REDIRECT_URLS || '')
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
      name: 'Flowsheet Verifier',
      type: 'web',
      disabled: false,
      icon: undefined,
      metadata: null,
      skipConsent: true,
    });
  }

  return clients;
}
```

Then the `oidcProvider` block (currently at lines 175-207) becomes:

```ts
oidcProvider({
  loginPage: '/sign-in',
  allowDynamicClientRegistration: false,
  requirePKCE: true,
  trustedClients: buildTrustedClients(),
  getAdditionalUserInfoClaim: async (userRecord) => { /* unchanged */ },
}),
```

Note the Wiki.js entry moves from inline-literal into the helper; this is a behavior-preserving refactor *if and only if* the same env vars are set. Backend-Service tests catch any inadvertent drop. The inline literal at line 183 builds `redirectUrls: [\`${process.env.WIKIJS_URL}/login/oidc/callback\`]` without a non-null check, so a missing `WIKIJS_URL` silently produces a malformed `undefined/login/oidc/callback` redirect (not a crash — silent misconfiguration, which is worse for ops). The helper's `if (env.WIKIJS_OIDC_CLIENT_ID && env.WIKIJS_OIDC_CLIENT_SECRET && env.WIKIJS_URL)` gate turns that into "Wiki.js trustedClient absent from the array", which surfaces visibly the first time someone tries to log into Wiki.js. Small ops win.

**Line numbers are pinned to the file's current `main` state.** The refactor itself shifts the `oidcProvider({...})` call by a few dozen lines downward, so post-merge line references will not match. Reviewers should diff against `main` to confirm equivalence: the helper output for `(WIKIJS_OIDC_CLIENT_ID, WIKIJS_OIDC_CLIENT_SECRET, WIKIJS_URL)` set to the production values must produce a `TrustedClient[]` whose Wiki.js entry is byte-equal to the current inline literal (modulo the `as const` quirk on `type`). The first `buildTrustedClients` test case captures this explicitly.

`FLOWSHEET_OIDC_REDIRECT_URLS` is comma-separated so we can list `https://flowsheet.wxyc.org/auth/callback` for prod and `http://localhost:8765/auth/callback` for dev in a single env var, mirroring the way `BETTER_AUTH_TRUSTED_ORIGINS` is parsed at line 60.

#### `.env.example` (Backend-Service)

Three new lines:

```
FLOWSHEET_OIDC_CLIENT_ID=flowsheet
FLOWSHEET_OIDC_CLIENT_SECRET=
FLOWSHEET_OIDC_REDIRECT_URLS=http://localhost:8765/auth/callback
```

Production gets `FLOWSHEET_OIDC_REDIRECT_URLS=https://flowsheet.wxyc.org/auth/callback,http://localhost:8765/auth/callback` so the same secret works both places (the localhost callback is harmless in prod because the redirect_uri is allowlisted on the *provider* side, not relaxed).

#### Tests

New file `tests/unit/auth/build-trusted-clients.test.ts` exercising the helper directly:

- With `FLOWSHEET_OIDC_CLIENT_ID` + `FLOWSHEET_OIDC_CLIENT_SECRET` + `FLOWSHEET_OIDC_REDIRECT_URLS=https://a/cb,https://b/cb` set, the returned array contains a flowsheet entry with `redirectUrls = ['https://a/cb', 'https://b/cb']`.
- With surrounding whitespace on the URL list (`"https://a/cb , https://b/cb "`), the helper trims and still produces a clean two-element array.
- With an empty `FLOWSHEET_OIDC_REDIRECT_URLS`, the flowsheet entry's `redirectUrls` is `[]` and the entry is still present (Better Auth will reject the resulting flow, but that's the auth server's job, not the helper's).
- With `FLOWSHEET_OIDC_CLIENT_ID` unset, no flowsheet entry is produced — the array is just the Wiki.js entry.
- With both Wiki.js and Flowsheet env vars set, both entries are present in stable order (Wiki.js first; tested explicitly so a reorder is a visible diff).
- With `WIKIJS_OIDC_CLIENT_ID` and `WIKIJS_OIDC_CLIENT_SECRET` set but `WIKIJS_URL` missing, no Wiki.js entry is produced — this is the defensive gate that prevents the current silent `undefined/login/oidc/callback` misconfiguration. The flowsheet entry is still produced if its env vars are set.
- With neither set, the helper returns `[]` (current behavior produces a malformed `undefined/login/oidc/callback` redirect URL silently; the refactor turns that into "Wiki.js trustedClient absent", a visible failure mode).

Pattern matches the existing `tests/unit/auth/` suite — vitest/jest-style table tests against a pure function, no `betterAuth({...})` instantiation, no DB mock needed.

No integration test for the OIDC flow itself — Better Auth's own test suite covers that.

### flowsheet-digitization

#### `core/schema.py`

Add a `VerifiedBy` model and an optional `verified_by` field on `PageResult`. The field belongs on `PageResult` (caller-owned, never on `GeminiPageResult`) for the same reason `model_version` and `extracted_at` live there: Gemini doesn't fill it; the verifier server does, at save time.

```python
class VerifiedBy(BaseModel):
    """Identity of the reviewer who saved this verified page.

    Populated only by `verifier/serve.py`'s POST /api/save handler when an
    authenticated reviewer session is present. Pipeline-written PageResults
    leave it as None.

    The `user_id` field is deliberately denormalized to `jobs.reviewer_id`
    (see `core/jobs.py`) so per-reviewer queries don't have to parse every
    JSON file on disk. The two values are always equal for the same save
    call; both are written inside `/api/save` under server authority — the
    client never sets either.
    """

    user_id: str = Field(description="Better Auth user.id (the OIDC `sub` claim).")
    username: str | None = Field(default=None, description="Better Auth username, if set.")
    real_name: str | None = Field(default=None, description="Reviewer's real name from the WXYC user record.")
    dj_name: str | None = Field(default=None, description="Reviewer's on-air DJ name, if set.")
    verified_at: datetime = Field(description="When the verifier UI saved this page (UTC).")


class PageResult(GeminiPageResult):
    """On-disk shape: `GeminiPageResult` plus the fields the caller owns."""

    model_version: str = Field(description="Model id that produced this result.")
    extracted_at: datetime = Field(description="When the extraction completed (UTC).")
    verified_by: VerifiedBy | None = Field(
        default=None,
        description=(
            "Reviewer who saved this corrected page via the verifier UI. "
            "None for results that have only been through the automatic pipeline."
        ),
    )
```

`verified_by` is `Optional[VerifiedBy]` with `default=None` so every `verified.json` written before this PR continues to parse — the field is silently absent on old files. New saves write the block; old files load with `verified_by=None` and are upgraded on next save. No migration script needed.

**Server-as-authority on `verified_by`** (security-relevant): the client never sets `verified_by`. The SPA sends the corrected `PageResult` shape as before; on the server, `/api/save` validates the client payload, then *overwrites* `verified_by` from the authenticated `ReviewerSession` before persistence. The ordering matters: Pydantic's `model_validate` runs first and returns 400 on any malformed `verified_by` the client sent (wrong type, missing required field on a non-null block, etc.) before the overwrite ever runs. On valid input, the overwrite happens unconditionally — what the client sent is discarded.

```python
validated = PageResult.model_validate(body)
validated.verified_by = VerifiedBy(
    user_id=reviewer.user_id,
    username=reviewer.username,
    real_name=reviewer.real_name,
    dj_name=reviewer.dj_name,
    verified_at=datetime.now(UTC),
)
verified_path.write_text(validated.model_dump_json())
```

Any `verified_by` block the client tries to send is silently discarded. The test for `/api/save` asserts this: a request that includes `verified_by={"user_id": "spoofed-id"}` in the body and a valid session cookie for user `real-id` produces a saved file whose `verified_by.user_id == "real-id"`.

**Save-call semantics, in full:**

- `verified_at` is stamped from `datetime.now(UTC)` on every save. There is no "preserve the original verified_at across overwrites" behavior in this PR — a reviewer who edits and re-saves the same page gets a new timestamp, and the previous one is gone from `verified.json`. The git history of `data/verifier/<stem>.verified.json` (if checked in) and the `jobs.db` audit trail are the only places the prior timestamp survives. If the future history view ever wants "first verified at" vs "last verified at", it's an additive column on `jobs` plus a small server-side merge — not a redesign.
- The client can read the persisted `verified_by` back. `verified.json` is served as a static file from `/data/verifier/<stem>.verified.json` (existing static mount in `verifier/serve.py`). When the SPA reloads a previously-verified page, it parses the JSON and *will* see the `verified_by` block. This PR does **not** add a UI surface that renders "last verified by X" — the SPA continues to read `verified.json` for the page data and ignores the block. A follow-up that shows a reviewer's history is out of scope here but unblocked by this change. The `verified_by` block is not personal-identifying in the secret-management sense (it's the reviewer's WXYC username/dj name, which are already shown elsewhere in the dj-site UI), so leaving it readable from the static mount is consistent with the rest of the corpus.
- `/auth/me` returns the *currently logged-in* reviewer (from the session cookie), not the page's last-saved reviewer. Those are two distinct queries and the SPA should not conflate them in the future-UI work.

**Relationship to `jobs.reviewer_id`** (two storage layers, one identity): the `verified.json` file carries the full provenance record (`VerifiedBy` block: id + name + dj_name + timestamp), while the `jobs` table carries only `reviewer_id TEXT` — denormalized for cheap "which jobs has reviewer X verified" queries without parsing every JSON file. The round-trip is:

```
ReviewerSession.user_id
   │
   ├──> VerifiedBy.user_id  (persisted in verified.json's verified_by block)
   │
   └──> JobStore.mark_verified(..., reviewer_id=...)  (persisted as jobs.reviewer_id)
```

Both writes happen in `/api/save`, in the same transaction-shaped block. `VerifiedBy.user_id` and `jobs.reviewer_id` are always equal for the same save call; they are deliberate denormalizations of the same field for two different access patterns. The `core/schema.py` docstring on `VerifiedBy` should state this relationship explicitly.

#### Modified: `pyproject.toml`

Append three entries to the existing `dependencies` list (currently lines 10-29 of `pyproject.toml`). These are *new* dependencies — the OIDC PR does not work until they are added.

```toml
"authlib>=1.3",                    # OIDC client (discovery + PKCE + token exchange + id_token verify)
"itsdangerous>=2.2",               # signed session cookie
"python-jose[cryptography]>=3.3",  # used by authlib for JWKS verification
```

`authlib` is the choice over rolling the dance by hand with `httpx` + `python-jose` because it handles the discovery doc + token endpoint + id_token claim validation as one library and gets ~140 lines off our diff. The dependency cost is real (it pulls cryptography) but it's already in the Python ecosystem's standard tier for OIDC.

The three deps land in PR #2 (the `core/auth.py` PR) so the new module compiles in CI; the routes that consume them land in PR #3.

#### `.env.example`

Append this block to the existing `.env.example`. All five vars must be set together for OIDC to engage; with any unset, the verifier falls back to BasicAuth (if `VERIFIER_PASSWORD` is set) or no auth (current local-dev default).

```
## OIDC Authentication (verifier reviewer identity)

# When unset, the verifier falls back to either VERIFIER_PASSWORD-gated
# BasicAuth or no auth at all (current local-dev default). Set all five
# vars together to enable OIDC against api.wxyc.org/auth.

# Issuer URL of the WXYC Better Auth server. Production:
# https://api.wxyc.org/auth. Local: http://localhost:8082/auth.
WXYC_AUTH_ISSUER=

# OIDC client id (must match Backend-Service's FLOWSHEET_OIDC_CLIENT_ID).
WXYC_OIDC_CLIENT_ID=

# OIDC client secret (must match Backend-Service's
# FLOWSHEET_OIDC_CLIENT_SECRET exactly). Treat as production secret;
# never commit a real value.
WXYC_OIDC_CLIENT_SECRET=

# Signing key for the verifier's own session cookie. 32+ random bytes;
# rotating it invalidates every active reviewer session. Generate with
# `python -c "import secrets; print(secrets.token_urlsafe(48))"`. The
# cookie's TTL is hard-coded to 12 hours in core/auth.py; expect a
# re-login at that boundary mid-session (no refresh-token plumbing in
# this PR — by design, see plans/oidc-auth.md "Deliberately out of
# scope").
WXYC_SESSION_SECRET=

# Public origin where this verifier is reachable. Determines the
# `redirect_uri` sent to the auth server and gates the `secure` flag
# on session cookies (Secure only when the value starts with `https`).
# Production: https://flowsheet.wxyc.org. Local: http://localhost:8765.
FLOWSHEET_PUBLIC_URL=
```

#### New: `core/auth.py`

The OIDC client + session model + cookie codec. ~120 lines. Public surface:

```python
@dataclass(frozen=True)
class ReviewerSession:
    user_id: str
    email: str
    username: str | None
    real_name: str | None
    dj_name: str | None
    # Advisory only — populated from the id_token's `role` claim (which
    # Backend-Service's JWT plugin sets from the user's organization
    # membership). Nothing in this PR gates access on `role`; it's
    # threaded through for future per-reviewer feature gating.
    role: str | None  # "stationManager" | "musicDirector" | "dj" | "member" | None

# One-shot cookies (state, PKCE verifier, return_to) live for the duration
# of the OIDC redirect bounce. Originally 10 minutes; raised to 30 after a
# first-time interactive login (find credentials / provision account on the
# dj.wxyc.org sign-in) exceeded 10 and failed the callback with an expired
# one-shot cookie. Kept as a `timedelta` to mirror SESSION_TTL below; call
# sites use `.total_seconds()` for itsdangerous's max_age and
# `int(.total_seconds())` for cookie max-age headers.
ONE_SHOT_TTL = timedelta(minutes=30)

async def build_authorize_url(return_to: str) -> tuple[str, str, str]:
    """Return (authorize_url, state, code_verifier). Caller stashes state+verifier."""

async def exchange_code(code: str, code_verifier: str) -> ReviewerSession:
    """Exchange the authorization code for an id_token; verify it; return reviewer.

    Raises if the auth server is unreachable, if the id_token's signature
    doesn't verify, or if `aud` / `iss` / `sub` / `exp` don't match.
    The verifier serve.py middleware translates these into a 503 (transport
    failure) or 400 (claim failure) — see `/auth/callback` handler.
    """

def encode_session(s: ReviewerSession) -> str:
    """Sign the reviewer payload into a session-cookie value. TTL = SESSION_TTL."""

def decode_session(raw: str) -> ReviewerSession | None:
    """Verify signature + freshness; return None if bad/expired."""

def sign_one_shot(value: str) -> str:
    """Sign `value` for transit in a redirect-survival cookie.

    Used for `oidc_state`, `oidc_verifier`, and `oidc_return_to`. The
    matching `verify_one_shot` call MUST pass max_age=ONE_SHOT_TTL;
    the value is otherwise considered tampered/expired and rejected.
    """

def verify_one_shot(raw: str, *, max_age: int) -> str:
    """Counterpart to sign_one_shot; raises on tamper or expiry.

    Callers in serve.py always pass max_age=ONE_SHOT_TTL. The
    parameter is kept explicit (rather than baked in) so tests can drive
    the expiry branch with a short max_age and a stub time.
    """

# For tests: lets the test suite force a fresh discovery+JWKS fetch
# between cases that monkeypatch the auth-server URL or stub httpx.
def _reset_metadata_cache() -> None:
    """Test-only: clear the module-level discovery/JWKS cache."""

COOKIE_NAME = "flowsheet_session"
SESSION_TTL = timedelta(hours=12)
```

**Why module-level cached state here, when `core/gemini.py` uses dependency injection.** The project convention (see `core/gemini.py`'s "Design note: we accept the SDK client at construction time rather than constructing it inside the class") is to inject SDK clients so tests can substitute mocks. `core/auth.py` deviates: the OIDC discovery doc and JWKS are fetched once on first auth request and held in module-level globals. The trade-off is intentional:

- The discovery doc + JWKS are *configuration* (issuer URL, signing keys), not *clients with per-call state*. They change on a much slower clock (key rotation) than a single request.
- A long-running Railway process serves thousands of `/auth/callback` requests against the same JWKS. Re-fetching per call would mean a `httpx.get` round-trip on every login — measurably worse latency for no testability gain we can't get another way.
- The function-level boundary (`_load_metadata`) is mockable directly; tests don't need DI. The `_reset_metadata_cache()` test helper is the explicit seam.

**Considered and rejected: eager load at app startup.** Fetching discovery + JWKS during `verifier/serve.py`'s module load would avoid the "first user pays a ~50-150ms cold-start" cost and would let tests treat the cache as immutable. We rejected it because it couples verifier process boot to the auth server's reachability — a brief api.wxyc.org slowdown during a flowsheet deploy would prevent the verifier from starting at all, including its `/api/version` healthcheck, which Railway uses to decide whether to keep the new revision alive. The current lazy approach degrades more gracefully: a transient auth-server failure surfaces as a 503 on the first `/auth/*` call, the verifier process stays up, and the next request retries. If the cold-start latency becomes a real complaint, the right fix is async pre-warming on app startup (fire-and-forget `_load_metadata()` in a startup event, with the cache populated by the time the first user clicks login), not making boot blocking.

The module docstring should state this trade-off in those terms so a future reader doesn't "fix" it back to the DI pattern by reflex.

**Signing-algorithm dispatch (added post-merge, see PR #79 series; updated for EdDSA).** This plan originally assumed the auth server would sign id_tokens with RS256/ES256 keyed off the JWKS. As of 2026-07, `api.wxyc.org/auth/.well-known/openid-configuration` advertises `["RS256", "EdDSA"]` and publishes a single OKP/Ed25519 JWKS key, so production signs id_tokens with **EdDSA** (verified live in Safari against the real IdP). HS256 support is retained because Better Auth's `oidcProvider` plugin *can* be configured to HMAC-sign with the client's `client_secret`, and other WXYC deployments may do so — but it is not what api.wxyc.org uses. `_verify_id_token` parses the token's `alg` header first and dispatches to disjoint key material: HS256 → `client_secret.encode("utf-8")`; RS256/ES256/EdDSA → filtered JWKS via `_signing_keys_only`. Two disjoint constants (`_SYMMETRIC_SIG_ALGS`, `_ASYMMETRIC_SIG_ALGS`) name the split, and `_ALLOWED_SIG_ALGS` is their union so a new alg lands in exactly one set on purpose. Consequences: HS256 logins do NOT consume the JWKS (cached or otherwise), the JWKS rotation-retry lives strictly inside the asymmetric branch, and a bad-secret HS256 login can never invalidate the cache used by legitimate asymmetric traffic. Multi-audience `aud` claims require exact `[client_id]` match (OIDC Core 3.1.3.7 §4-5); `exp` is essential to prevent tokens without expiration from verifying forever. The alg-confusion attack class (CVE-2016-10555 family — attacker signs an HS256 token using JWKS public-key bytes as HMAC secret) is closed by construction because the HS256 verify path never receives JWKS material.

**Issuer-claim source (updated for EdDSA rollout).** The required `iss` claim is validated against the discovery document's `issuer` field — the OIDC-spec-authoritative provider identity — not `WXYC_AUTH_ISSUER`. Better Auth serves discovery under the endpoint base (`https://api.wxyc.org/auth/.well-known/openid-configuration`) but sets `issuer` to the bare origin (`https://api.wxyc.org`), so the id_token's `iss` legitimately differs from `WXYC_AUTH_ISSUER`. Validating against the env var rejects every real token with an `iss` mismatch (400 "invalid id token"). `_verify_id_token` reads `issuer` from the cached discovery doc and threads it into `_decode_with_key`; a discovery doc missing `issuer` raises `ValueError` (route → 400), mirroring the `jwks_uri` guard.

Why this lives under `core/` and not `verifier/`: `core/` is the project's home for "pure modules with no FastAPI deps" per the existing layout. The dataclass and codec are pure; only the OIDC HTTP calls reach out, and they use `httpx` (already a project dep). Tests live under `tests/unit/test_auth.py` and can run without spinning up the FastAPI app.

#### Modified: `CLAUDE.md`

Insert a new entry under the existing `core/` block in the "Architecture (one-liner per module)" section. The existing entries are not strictly alphabetical — they're loosely ordered by how often a reader hits them (`schema.py` first because every module reads or writes through it). Insert `auth.py` between `schema.py` and `prompts.py`. Rationale: `auth.py` is, like `schema.py`, a small support module that a reader needs to understand before reading the pipeline steps (`render`, `gemini`, `jobs`, `pipeline`) — not a step itself. The exact slot is a judgment call; any position above the pipeline-step modules is acceptable. The literal text to add:

```
  auth.py                        OIDC client (against Better Auth at
                                 api.wxyc.org/auth), signed session-cookie codec
                                 (itsdangerous), and the ReviewerSession
                                 dataclass. Consumed by verifier/serve.py
                                 (middleware + /api/save). Module-level cached
                                 discovery doc + JWKS — configuration, not
                                 per-request state; _reset_metadata_cache()
                                 is the test seam. Lazy-loaded (not eager at
                                 boot) so a transient auth-server slowdown
                                 doesn't take the verifier offline; first
                                 caller after a deploy pays ~50-150ms cold
                                 start. Deviation from the DI pattern in
                                 core/gemini.py is intentional; don't "fix"
                                 it back — see the module docstring.
```

This is the only CLAUDE.md change in scope; per-PR design choices (PKCE, the 12h TTL, server-as-authority on `verified_by`) stay in this plan and the code itself.

#### Modified: `verifier/serve.py`

Three deltas:

1. **Replace the BasicAuth-only install block with a tri-modal gate.** Reading order of preference: OIDC (if `WXYC_OIDC_CLIENT_ID` set), BasicAuth (if `VERIFIER_PASSWORD` set), neither (current local-dev default). The two middlewares are mutually exclusive — never both at once.

2. **Add four auth routes**: `GET /auth/login`, `GET /auth/callback`, `POST /auth/logout`, `GET /auth/me`. The flow follows the standard OIDC code + PKCE pattern. State, code verifier, and `return_to` ride along in three signed one-shot cookies (`oidc_state`, `oidc_verifier`, `oidc_return_to`) that the callback validates and clears.

   `/auth/login` and `/auth/callback` are public routes (not behind the session middleware) — so they must translate transport-level failures from `core.auth` to HTTP themselves. Specifically: `httpx.HTTPError` from `build_authorize_url` (discovery doc unreachable) or `exchange_code` (token endpoint unreachable / JWKS unreachable) becomes `503 { "detail": "auth server unavailable" }`. Claim validation failures from `exchange_code` (`aud` / `iss` mismatch, bad signature) become `400 { "detail": "invalid id token" }`. The `try/except` wraps the entire handler body, not just the OIDC call; the test suite parametrizes the failure injection at the `core.auth` boundary.

3. **Add a `Depends(get_reviewer)` to `/api/save`**, and write a `verified_by` block into the saved JSON. The middleware and the dependency play distinct roles — the *middleware* gates "are you allowed past `/api/*`" (request rejected with 302/401 if no session); the *`Depends(get_reviewer)`* extracts the typed `ReviewerSession` object out of `request.state.reviewer` so the handler can call `reviewer.user_id` etc. without dotting through framework internals. The middleware runs first and is the security boundary; the Depends is just typed access. They are not redundant. `/api/bundles` and `/api/lookup` are gated by the middleware (neither is in `PUBLIC_PATHS`) but don't need the Depends because they don't read the reviewer — `/api/bundles` because the bundle list is corpus data we don't want anonymous traffic seeing, `/api/lookup` because the proxy adds an authentication-free hop to request-o-matic's station-wide rate-limit budget that we don't want unauthenticated visitors burning. See the "Open questions" section below for the prior framing of the lookup decision.

The middleware itself is ~30 lines: extract cookie → `decode_session` → if missing/invalid, 302 to `/auth/login?return_to=<current path>` for a genuine top-level document navigation, else 401 JSON → otherwise stash `reviewer` on `request.state`. (Document navigation is detected via `Sec-Fetch-Dest: document`, falling back to an `Accept: text/html` when the header is absent. The earlier design 302'd everything that wasn't `Accept: application/json`, but that funneled gated subresource fetches — a favicon, a prefetch — through `/auth/login`, and each such call mints a fresh OIDC `state` that overwrites the single one-shot cookie, breaking the concurrent real login's callback with a state mismatch. Only document navigations may reach the state-minting `/auth/login`; everything else 401s.) The bypass rule is two-part: `PUBLIC_PATHS = {"/auth/login", "/auth/callback", "/auth/logout", "/api/version"}` is the exact-match set, and *additionally* any path starting with `/auth/` bypasses the gate so the redirect dance can complete. A comment in the middleware body should call this out explicitly — "`PUBLIC_PATHS` lists specific public routes; `path.startswith('/auth/')` is the second condition for the OIDC dance. Add new public endpoints to `PUBLIC_PATHS`; don't widen the `/auth/` prefix to mean anything else."

`/api/version` stays open so the Railway healthcheck doesn't need a credential. This is load-bearing for the deploy pipeline — Railway uses `/api/version` to decide whether a new revision is healthy; if the OIDC middleware ever drops it from `PUBLIC_PATHS`, every deploy will roll back because the gate redirects the healthcheck to `/auth/login`. The PR description for PR #3 should call this out so a future refactor doesn't break it silently. `/auth/me` stays gated — calling it unauthenticated should 401 so the SPA knows to redirect.

#### Modified: `core/jobs.py`

Add a `reviewer_id TEXT` column to the `jobs` table via a small migration in `JobStore.__init__`'s schema-bootstrap path (the existing `CREATE TABLE IF NOT EXISTS` block, plus an `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` for already-deployed databases — SQLite supports `ALTER TABLE ADD COLUMN` since 3.25). Extend `mark_verified(...)` to accept an optional `reviewer_id: str | None = None` and persist it. Default `None` keeps existing call sites compiling.

#### Modified: `verifier/app.js`

Two small additions:

1. On page load, `fetch("/auth/me", { credentials: "include" })`. If 200, render the reviewer's `real_name || dj_name || username || email` in the header. If 401, the middleware already redirected; this branch should rarely fire.
2. Wrap `/api/save` and `/api/bundles` calls in a helper that, on `401`, sets `location.href = "/auth/login?return_to=" + encodeURIComponent(location.pathname + location.search)`.

No build step; no new JS dependency.

### Tests

New files:

- `tests/unit/test_auth.py` — exercises `encode_session` / `decode_session` round-trip, expiry, signature tampering. Exercises `build_authorize_url` against a mocked discovery doc. Exercises `exchange_code` against a mocked token endpoint + JWKS, including the `aud` mismatch case (must raise), the `iss` mismatch case (must raise), and a JWKS-fetch transport failure (httpx raises → `exchange_code` propagates → middleware translates to 503). Calls `_reset_metadata_cache()` between cases so the module-level cache doesn't carry a previous test's mock metadata into the next.
- `tests/unit/test_verifier_auth_routes.py` — uses the existing `httpx.ASGITransport` pattern from `tests/unit/test_verifier_serve.py`. Monkeypatches `core.auth.build_authorize_url` and `core.auth.exchange_code` to return canned values. Asserts:
  - `/auth/login` 302s and sets the three one-shot cookies (signed).
  - `/auth/callback` with a tampered `state` cookie returns 400.
  - `/auth/callback` happy path sets `flowsheet_session` and 302s to the validated `return_to`.
  - `/auth/callback` with an absolute or off-allowlist `return_to` falls back to `/verifier/`.
  - `/api/save` returns 401 with no cookie, 200 with a valid cookie, and writes `verified_by.user_id` into the saved JSON.
  - `/api/save` ignores any client-supplied `verified_by`: a request whose body includes `verified_by={"user_id": "spoofed-id"}` and a session cookie for `real-id` produces a saved file whose `verified_by.user_id == "real-id"` and whose `verified_by.verified_at` is the server's clock, not anything the client sent.
  - `/api/save` of a payload representing a pre-`verified_by` file (no `verified_by` key in the request body) succeeds and adds the server-authoritative block. This is the "upgrade-on-save" case for old `verified.json` files.
  - **Full disk round-trip for backwards compat**: write a `verified.json` to disk without the `verified_by` key (mirroring a pre-PR file), POST it through `/api/save` (the SPA's normal save path round-trips the on-disk shape), confirm the response is 200 and the rewritten file now contains a `verified_by` block with the authenticated reviewer's id. This test exercises Pydantic's `model_validate_json` against an old shape, the server-side overwrite, and the disk write — covering the load → modify-in-UI → save path end-to-end.
  - **Malformed client-supplied `verified_by`** (defense-in-depth, not security-load-bearing because we overwrite anyway): a request whose body includes `verified_by={"user_id": 123}` (int instead of str), `verified_by=[]` (array instead of object), or `verified_by="hello"` (scalar) returns 400 from Pydantic's `model_validate` step *before* the overwrite runs. A request whose body includes `verified_by={"user_id": "x", "extra_field": "y"}` succeeds (Pydantic ignores unknown keys by default) and produces a saved file whose `verified_by` is the server-authoritative block — no `extra_field`. A request whose body includes `verified_by={}` (empty object missing the required `user_id`/`verified_at`) returns 400. A request that omits `verified_by` entirely succeeds and writes the server-authoritative block.
  - `/auth/me` returns 401 with no cookie, 200 with a valid cookie.
  - `/api/version` is unauthenticated.
  - **Middleware precedence**: with both `WXYC_OIDC_CLIENT_ID` and `VERIFIER_PASSWORD` set, the OIDC gate is installed and BasicAuth is *not* (asserted by checking that a request with Basic credentials but no session cookie 302s to `/auth/login` instead of returning 200). With only `VERIFIER_PASSWORD` set, BasicAuth is installed. With neither set, no gate is installed and protected paths return 200 without credentials. Parametrization mechanics: each case calls `monkeypatch.setenv()` for the relevant env vars *before* `importlib.reload(serve_mod)`, then calls `_reset_serve_module_state(serve_mod)` immediately after the reload. The env-var set order matters — `verifier/serve.py` reads them at module import — so the test that flips a var without re-reading would observe stale config.
  - **Cookie `Secure` flag**: with `FLOWSHEET_PUBLIC_URL=https://...`, the `Set-Cookie` header on `/auth/login` and `/auth/callback` contains `Secure`. With `FLOWSHEET_PUBLIC_URL=http://localhost:8765`, it does not. (The middleware-install path reads `FLOWSHEET_PUBLIC_URL` at app construction, so the test parametrizes over a fresh app instance for each value.)
  - **`/auth/callback` translates transport failures to 503**: monkeypatch `core.auth.exchange_code` to raise an `httpx.ConnectError`; assert the route returns 503 with a stable error envelope, not 500. Same for `core.auth.build_authorize_url` from `/auth/login`.

**Test-isolation handling for app-rebuilding tests.** The middleware-precedence and `Secure`-flag tests rebuild the FastAPI app per parametrization. Two pieces of module-level state need clearing between cases:

  - `verifier.serve._initialized_jobs_dbs` (the `set[Path]` at `verifier/serve.py:112` that records which jobs.db files have had their schema bootstrap run). Tests that rebuild the app via `importlib.reload(serve_mod)` reset the *binding*, but a process-wide fixture that touches a real jobs.db across parametrizations needs to call `serve_mod._initialized_jobs_dbs.clear()` after the reload — or before the next case sets `DATA_ROOT`.
  - `core.auth._metadata` / `core.auth._jwks` — cleared via `core.auth._reset_metadata_cache()` (see above). Always call after the reload, even when the test doesn't seem to touch metadata: a parametrize tuple that flips `WXYC_OIDC_CLIENT_ID` between cases will rebuild the OIDC client against a different issuer URL and pick up stale metadata otherwise.

Wrap both into a single `_reset_serve_module_state(serve_mod)` test helper in `tests/unit/conftest.py`. **`tests/unit/conftest.py` does not exist yet** — this PR creates it (the existing `test_verifier_serve.py` does its own `importlib.reload` inline). Adding the helper now establishes the convention before more callers proliferate. Existing `test_verifier_serve.py` should be migrated to use the helper too, but that's a follow-up commit, not a blocker for the auth PR.

A worth-considering alternative the reviewer raised: extract the middleware-selection logic into a pure function (`_select_middleware(*, oidc_enabled, basicauth_enabled) -> list[tuple[Type, dict]]`) and unit-test the function directly without `importlib.reload`. That eliminates the reload tax and makes precedence testable in isolation. Plan defers this to a follow-up because (a) the reload-based test catches real middleware-order bugs that a pure-function unit test would miss (e.g., a `app.add_middleware` call accidentally outside the env-var gate), and (b) introducing the pure-function indirection ahead of a second caller violates the project's "no speculative abstractions" convention. If the reload tests prove flaky in CI, refactoring to the pure function is the right escape hatch.

The helper itself is not separately TDD-tested — it's ~4 lines of `set.clear()` and `delattr`-style calls with no branches. The middleware-precedence and Secure-flag tests *exercise* the helper as a side effect (without it, the second case would inherit the first case's metadata and fail), so any breakage in the helper surfaces as concrete test failures rather than abstract "the reset didn't work" assertions. That's the TDD signal we need; a dedicated `test_reset_serve_module_state` would test `set.clear()` and add noise without catching a real failure mode. The helper's docstring must be explicit so a future test author doesn't skip the call:

```python
def _reset_serve_module_state(serve_mod) -> None:
    """Clear module-level caches that survive importlib.reload().

    `importlib.reload(serve_mod)` rebinds the *module object*, but two
    pieces of process-wide state can still leak from a previous test:

      * `serve_mod._initialized_jobs_dbs` — set[Path] tracking which
        jobs.db files have had their schema bootstrap run. A test that
        flips DATA_ROOT between parametrizations needs this cleared, or
        the new path inherits the old "already bootstrapped" assumption
        and the schema migration is silently skipped.
      * `core.auth._metadata` and `core.auth._jwks` — cached OIDC
        discovery doc and JWKS. A test that flips WXYC_AUTH_ISSUER
        between parametrizations needs these cleared, or the new
        issuer's auth calls run against stale metadata.

    REQUIRED: call this unconditionally after every
    `importlib.reload(serve_mod)` and every `importlib.reload(auth_mod)`
    — including when the test under construction looks orthogonal to
    these caches. Whether a future parametrization touches DATA_ROOT or
    WXYC_AUTH_ISSUER is not predictable from inside one test case;
    skipping the reset produces flakes that depend on test execution
    order, which is the failure mode you do not want. The cost of
    calling it when unneeded is two `set.clear()` calls.
    """
```
- Extend `tests/unit/test_jobs.py` — `mark_verified` round-trips `reviewer_id`. The column accepts NULL.

The `external_api` marker is *not* used; everything is mocked at the `core.auth` boundary. CI runs the new tests by default.

## Rollout

Five steps, each independently mergeable and reversible:

1. **Backend-Service: add the `trustedClient` entry and ship.** Inactive until the env vars are set — there's no breaking change. PR #1.
2. **flowsheet-digitization: add `core/auth.py` and `tests/unit/test_auth.py`. No routes wired yet.** Pure module addition. PR #2.
3. **flowsheet-digitization: add the auth routes + middleware + `/api/save` reviewer threading + `reviewer_id` column on `jobs.db` + `mark_verified` signature change, all gated on `WXYC_OIDC_CLIENT_ID`.** Also adds the `_reset_serve_module_state` helper to `tests/unit/conftest.py` (new file or extension of existing) as a precondition for the new parametrized middleware-precedence and Secure-flag tests. If the env var is unset, behavior is unchanged. PR #3.
4. **flowsheet-digitization: enable OIDC in the Railway env, set redirect URL, retire `VERIFIER_PASSWORD`.** Operational change, no code. Reversible by unsetting `WXYC_OIDC_CLIENT_ID`.

Step 2 can ship before step 3 (the `core/auth.py` module compiles without any caller). Steps 3 and 4 are sequenced (3 lands code, 4 turns it on). Step 4 also depends on step 1 having landed in the auth deployment.

**Why the `reviewer_id`/`mark_verified` change merges with the routes PR rather than its own PR.** An earlier draft sketched these as separate PRs and claimed "Steps 2–4 can ship in either order." That was wrong: PR #3's `/api/save` handler calls `store.mark_verified(..., reviewer_id=...)`, which raises `TypeError: unexpected keyword argument 'reviewer_id'` against an unmodified `JobStore`. Either the kwarg lands first (making the routes PR a no-op revert risk if something goes wrong on the SQLite side) or they land together. We chose together — the kwarg addition and column ALTER are ~10 lines of code with their own targeted test, and bundling them with the caller eliminates the forward-incompatibility window.

## Local dev

Two processes:

- **Backend-Service auth** on `localhost:8082/auth`. `.env`: `FLOWSHEET_OIDC_CLIENT_ID=flowsheet`, `FLOWSHEET_OIDC_CLIENT_SECRET=<dev-secret>`, `FLOWSHEET_OIDC_REDIRECT_URLS=http://localhost:8765/auth/callback`, `COOKIE_SAME_SITE=lax`, `NODE_ENV=development`.
- **flowsheet verifier** on `localhost:8765`. `.env`: `WXYC_AUTH_ISSUER=http://localhost:8082/auth`, `WXYC_OIDC_CLIENT_ID=flowsheet`, `WXYC_OIDC_CLIENT_SECRET=<dev-secret>`, `WXYC_SESSION_SECRET=<dev random>`, `FLOWSHEET_PUBLIC_URL=http://localhost:8765`.

The session cookie's `secure` flag is gated on `FLOWSHEET_PUBLIC_URL.startswith("https")`, so dev works on plain HTTP without per-env conditionals scattered through the code.

A developer who doesn't want to run Backend-Service locally can leave `WXYC_OIDC_CLIENT_ID` unset and (optionally) set `VERIFIER_PASSWORD` to keep the existing BasicAuth gate. This is the default flow we ship for fresh checkouts.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Open redirect via `return_to` | Allowlist: relative path starting with `/verifier`; else fall back to `/verifier/`. |
| State CSRF | `secrets.compare_digest` between cookie state and query state. |
| id_token from a different OIDC client (e.g. Wiki.js) accepted | Strict `aud` check against our `client_id` in `JsonWebToken.decode(claims_options=...)`. |
| Session cookie replay after logout | 12h TTL by signature; `/auth/logout` clears the cookie client-side. Server-side revocation is out of scope until we have a per-session store. |
| OIDC discovery doc / JWKS fetch fails on first request and brings down the gate | Discovery + JWKS are fetched lazily and cached. First call after a process restart can fail; we surface that as a 503 with a clear log line. A Railway redeploy refreshes the cache. |
| Cross-repo coordination failure (auth deployed without flowsheet, or vice versa) | Both sides gate on env vars. Auth side: no client without `FLOWSHEET_OIDC_CLIENT_ID`. Flowsheet side: BasicAuth fallback if `WXYC_OIDC_CLIENT_ID` unset. |
| `authlib` adds a meaningful dependency surface | Accepted. The alternative (hand-rolling discovery + PKCE + JWKS + claim validation) is more code we own and a worse review target. |
| jobs.db migration on a corpus with many in-flight rows | `ALTER TABLE ADD COLUMN` is O(1) in SQLite (metadata-only); no row rewrite. Tested in `test_jobs.py`. |

## Open questions for review

1. Is `core/auth.py` the right home, or do we want a `verifier/auth.py` because it's tied to the FastAPI app? My read: the dataclass + codec + OIDC client are pure modules; routes are the FastAPI-coupled piece and stay in `verifier/serve.py`.
2. Do we want to ship the `reviewer_id` column change as a separate PR with its own migration test, or fold it into the auth PR? I've sketched it as a separate PR; the migration is small enough that bundling is also defensible.
3. ~~Should `/api/lookup` be gated?~~ **Decided: gate it.** The verifier's `/api/lookup` is the same-origin proxy to request-o-matic. Even though request-o-matic is open upstream, the proxy adds an authentication-free hop the rest of our infrastructure now lacks, and the rate-limit budget on request-o-matic is shared station-wide — an unauthenticated visitor hammering the proxy burns the budget for actual reviewers. Gating costs nothing (the same `Depends(get_reviewer)` already added to `/api/save` / `/api/bundles`) and aligns the protected surface with the rule "anything that touches the corpus or station infra goes through the session." Treat this as a corrected decision, not an open question — the test plan above already gates lookup behind the middleware via `PUBLIC_PATHS` exclusion.
4. Should we cache the OIDC discovery doc to a file rather than just module-level memory, so a process restart on Railway doesn't have a brief window of "first user gets an error"? My read: not worth the complexity for what amounts to one cold request per deploy.
