"""Shared fixtures + helpers for `tests/unit/`.

Created in the OIDC PR for `_reset_serve_module_state`, which the
parametrized middleware-precedence and Secure-flag tests need to keep
test cases from leaking module-level state into each other.
"""

from __future__ import annotations

from types import ModuleType


def _reset_serve_module_state(serve_mod: ModuleType) -> None:
    """Clear module-level caches that survive `importlib.reload()`.

    `importlib.reload(serve_mod)` rebinds the *module object*, but two
    pieces of process-wide state can still leak from a previous test:

      * `serve_mod._initialized_jobs_dbs` — set[Path] tracking which
        jobs.db files have had their schema bootstrap run. A test that
        flips DATA_ROOT between parametrizations needs this cleared,
        or the new path inherits the old "already bootstrapped"
        assumption and the schema migration is silently skipped.
      * `core.auth._metadata` and `core.auth._jwks` — cached OIDC
        discovery doc and JWKS. A test that flips `WXYC_AUTH_ISSUER`
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
    import core.auth as auth_mod

    if hasattr(serve_mod, "_initialized_jobs_dbs"):
        serve_mod._initialized_jobs_dbs.clear()
    auth_mod._reset_metadata_cache()
