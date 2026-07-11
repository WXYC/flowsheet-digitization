"""Shared fixtures + helpers for `tests/unit/`.

Two pieces of shared machinery live here:

  * `_restore_environ` (autouse) — snapshots and restores `os.environ`
    around every test so a durable env mutation (e.g. a CLI's
    `load_dotenv` leaking into the shared process) can't poison a later
    test regardless of collection order. This is the #91 order-pollution
    guard; see the fixture docstring.
  * `_reset_serve_module_state` — clears the module-level caches that
    survive `importlib.reload(serve_mod)`, so tests that flip DATA_ROOT
    or WXYC_AUTH_ISSUER between reloads don't inherit stale state.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """Snapshot `os.environ` before each test and restore it after.

    Some tests mutate the process environment durably, outside pytest's
    `monkeypatch` (which self-restores). The concrete case that motivated
    this: `test_cli` invokes the Typer CLI, whose `@app.callback()` runs
    `load_dotenv(override=False)`, which reads the developer's real `.env`
    and injects `WXYC_OIDC_CLIENT_ID=flowsheet` (and secrets) into
    `os.environ` for the *rest of the process*. Every later test that
    rebuilds the verifier app then sees OIDC enabled and 401s where it
    expected an open gate — the #91 order-dependent failure.

    Restoring the full environment after every test makes the suite
    order-independent regardless of what mutated it, and is the airtight
    counterpart to the per-request env reads in `verifier/serve.py`. CI
    doesn't hit the failure (its `.env` has no real values), which is
    exactly why the leak went unnoticed; this guards it everywhere.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


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
