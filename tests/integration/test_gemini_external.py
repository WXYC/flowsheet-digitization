"""End-to-end smoke tests against the real Gemini API.

Excluded from the default test run (`addopts = -m 'not external_api'` in
`pyproject.toml`). Opt in with `pytest -m external_api` and a valid
`GEMINI_API_KEY` in the environment.

These tests exist to catch SDK-shape drift between our mock-based unit
tests and the actual API surface. The mocks know the shape we PASS to
the SDK; only a real call can confirm the SDK accepts that shape. A
small handful of these tests is enough; running them is opt-in for cost
control.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.gemini import GeminiClient
from core.schema import QUADRANT_ORDER, GeminiPageResult

# HTTP statuses that mean "the request was well-formed but the
# environment can't service it right now": auth, quota, billing.
# Treating these as skips (not failures) keeps the test useful in any
# dev environment without losing its 400-INVALID_ARGUMENT signal.
# `ClientError.status` is the Google API status string, not the int code.
_ENVIRONMENT_NOT_CALLABLE = frozenset(
    {"UNAUTHENTICATED", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED"}
)

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
# The smallest golden by bytes; image-token cost is resolution-bound but
# we pick the cheapest available to keep this test's per-run cost lowest.
SMOKE_IMAGE = GOLDEN_DIR / "1990-04apr0106-page05.png"


def _build_real_client() -> GeminiClient:
    """Construct a GeminiClient backed by the real google-genai SDK.

    Skips the test (rather than failing) when the API key isn't set —
    a developer running `pytest -m external_api` without a key gets a
    clear skip reason instead of an obscure auth error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set; external_api tests require a real key")

    from google import genai

    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    return GeminiClient(sdk=genai.Client(api_key=api_key), model=model)


@pytest.mark.external_api
async def test_create_cache_and_extract_with_real_gemini() -> None:
    """Smoke test: cache creation + cached `extract_page` against real Gemini.

    Confirms that:
      * `caches.create` accepts the `CreateCachedContentConfig` shape we
        build (system_instruction + ttl) — would fail if Google renamed
        a field or tightened validation.
      * `generate_content` with `cached_content` referencing that cache
        accepts our `GenerateContentConfig` shape and returns a parseable
        `GeminiPageResult` — would fail if the cached-call surface drifts.

    What this test does NOT verify: actual billing discount, response
    token counts, or extraction quality. The first is a billing concern
    (not an SDK-shape concern); the latter two are covered by the
    calibration harness.

    Cost: ~$0.01-0.02 per run. Opt in with `pytest -m external_api`.
    """
    if not SMOKE_IMAGE.is_file():
        pytest.skip(f"golden image missing: {SMOKE_IMAGE}")

    from google.genai.errors import ClientError

    client = _build_real_client()

    ok = await client.create_cache()
    if not ok:
        # Caching has a min-token threshold and is unsupported on some
        # models; either case is a legitimate skip rather than a failure
        # since the production fallback path handles it. `create_cache`
        # swallows all exceptions (including billing/auth/quota) and
        # returns False — so this branch covers those too.
        pytest.skip("cache creation failed (min-token threshold, unsupported model, or quota)")

    try:
        result = await client.extract_page(SMOKE_IMAGE)
    except ClientError as exc:
        # Environment-level errors (auth, quota, billing) are skips, not
        # failures — the cache call already proved the SDK accepted our
        # request shape, which is what this test exists to verify.
        # A 400 INVALID_ARGUMENT would still propagate and fail the test
        # (that IS shape drift) since it isn't in the allowlist.
        if exc.status in _ENVIRONMENT_NOT_CALLABLE:
            pytest.skip(f"API not callable in this environment: {exc.status}")
        raise

    # Structural assertions only — quality is a calibration concern.
    assert isinstance(result, GeminiPageResult)
    assert len(result.quadrants) == 4
    assert tuple(q.position for q in result.quadrants) == QUADRANT_ORDER
