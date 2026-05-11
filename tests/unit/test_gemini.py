"""Tests for the Gemini extraction client.

Strategy: dependency-inject the google-genai Client so tests never touch
the network. The fake captures the SDK call arguments so we can assert on
model id, prompt, image mime type, and response_schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.gemini import GeminiClient, GeminiError, MediaResolution
from core.prompts import PAGE_EXTRACTION_PROMPT
from core.schema import GeminiPageResult, Quadrant


def _sample_page_result() -> GeminiPageResult:
    """The production shape of `extract_page`'s return value. The SDK fills
    only `GeminiPageResult` because that's the `response_schema` we pass —
    `model_version` and `extracted_at` are added by the pipeline."""
    return GeminiPageResult(
        page_date_raw="Monday 1 Jan '90",
        quadrants=[
            Quadrant(position="top_left", hour_raw="6AM", jock_raw="ALECIA", entries=[]),
            Quadrant(position="top_right", hour_raw=None, jock_raw=None, entries=[]),
            Quadrant(position="bottom_left", hour_raw=None, jock_raw=None, entries=[]),
            Quadrant(position="bottom_right", hour_raw=None, jock_raw=None, entries=[]),
        ],
    )


def _fake_sdk(parsed: GeminiPageResult | None) -> tuple[MagicMock, AsyncMock]:
    """Build a fake `google-genai` Client whose async generate_content returns `parsed`.

    Returns the sdk mock plus the AsyncMock for `generate_content` so tests
    can inspect call arguments.
    """
    response = MagicMock()
    response.parsed = parsed
    generate_content = AsyncMock(return_value=response)

    sdk = MagicMock()
    sdk.aio.models.generate_content = generate_content
    return sdk, generate_content


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    p = tmp_path / "page.png"
    # Real PNG signature (8 bytes) — enough for a "this is a PNG" check if we add one.
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-image-data")
    return p


class TestGeminiClient:
    async def test_extract_page_returns_parsed_gemini_page_result(self, png_file: Path) -> None:
        expected = _sample_page_result()
        sdk, _generate = _fake_sdk(parsed=expected)

        client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")
        result = await client.extract_page(png_file)

        # Production result type — no model_version / extracted_at on it.
        assert isinstance(result, GeminiPageResult)
        assert result.page_date_raw == "Monday 1 Jan '90"
        # Sanity: those caller-set fields really aren't on the model.
        assert "model_version" not in type(result).model_fields

    def test_client_exposes_configured_model_id(self) -> None:
        """The pipeline reads `client.model` to fill the truthful
        `model_version` on the on-disk `PageResult`. Without this property
        the wrap step would have to reach into `_model` (private)."""
        sdk = MagicMock()
        client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")
        assert client.model == "gemini-3.1-pro-preview"

    async def test_extract_page_passes_model_id_to_sdk(self, png_file: Path) -> None:
        sdk, generate = _fake_sdk(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")

        await client.extract_page(png_file)

        kwargs = generate.call_args.kwargs
        assert kwargs["model"] == "gemini-3.1-pro-preview"

    async def test_extract_page_sends_prompt_and_image(self, png_file: Path) -> None:
        sdk, generate = _fake_sdk(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m")

        await client.extract_page(png_file)

        contents = generate.call_args.kwargs["contents"]
        # First content element is the prompt text.
        assert contents[0] == PAGE_EXTRACTION_PROMPT
        # Second is a Part-from-bytes with image/png.
        image_part = contents[1]
        # `from_bytes` returns a Part with inline_data on the .inline_data attr.
        assert image_part.inline_data.mime_type == "image/png"
        # The PNG signature should be present in the encoded bytes.
        assert image_part.inline_data.data.startswith(b"\x89PNG")

    async def test_extract_page_uses_response_schema(self, png_file: Path) -> None:
        sdk, generate = _fake_sdk(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m")

        await client.extract_page(png_file)

        config = generate.call_args.kwargs["config"]
        assert config.response_mime_type == "application/json"
        # GeminiPageResult, NOT PageResult — caller-set fields stay out of
        # the schema so the model can't hallucinate them. See core.schema
        # module docstring.
        assert config.response_schema is GeminiPageResult

    async def test_extract_page_passes_media_resolution(self, png_file: Path) -> None:
        sdk, generate = _fake_sdk(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m", media_resolution=MediaResolution.HIGH)

        await client.extract_page(png_file)

        config = generate.call_args.kwargs["config"]
        # Don't pin the exact enum object (SDK may rename); just assert it was set.
        assert config.media_resolution is not None

    async def test_extract_page_raises_when_parsed_missing(self, png_file: Path) -> None:
        sdk, _generate = _fake_sdk(parsed=None)
        client = GeminiClient(sdk=sdk, model="m")
        with pytest.raises(GeminiError):
            await client.extract_page(png_file)

    async def test_extract_page_raises_for_wrong_parsed_type(self, png_file: Path) -> None:
        sdk, _generate = _fake_sdk(parsed=None)
        # Patch the response so .parsed is a non-GeminiPageResult object.
        response: Any = sdk.aio.models.generate_content.return_value
        response.parsed = {"not": "a-page-result"}
        client = GeminiClient(sdk=sdk, model="m")
        with pytest.raises(GeminiError):
            await client.extract_page(png_file)

    async def test_extract_page_rejects_nonexistent_image(self, tmp_path: Path) -> None:
        sdk, _ = _fake_sdk(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m")
        with pytest.raises(GeminiError):
            await client.extract_page(tmp_path / "missing.png")


def _fake_sdk_with_cache(
    *,
    parsed: GeminiPageResult | None,
    cache_create_error: Exception | None = None,
    cache_name: str = "cachedContents/abc123",
) -> tuple[MagicMock, AsyncMock, AsyncMock]:
    """SDK fake that also captures `caches.create` calls.

    Returns the sdk plus the `generate_content` and `caches.create` mocks
    so tests can inspect call arguments on either surface.
    """
    response = MagicMock()
    response.parsed = parsed
    generate_content = AsyncMock(return_value=response)

    if cache_create_error is not None:
        caches_create = AsyncMock(side_effect=cache_create_error)
    else:
        cache_response = MagicMock()
        cache_response.name = cache_name
        caches_create = AsyncMock(return_value=cache_response)

    sdk = MagicMock()
    sdk.aio.models.generate_content = generate_content
    sdk.aio.caches.create = caches_create
    return sdk, generate_content, caches_create


class TestGeminiClientCaching:
    """Context caching is opt-in via `create_cache()` and saves the input-
    token cost of the ~2-3K-token prompt + response schema on every page
    after the first. The pipeline calls it once at the start of a process
    run; failures degrade to the un-cached path rather than abort the run."""

    async def test_create_cache_caches_the_prompt_as_system_instruction(self) -> None:
        """The cache holds the page prompt as `system_instruction`. The
        SDK's `CreateCachedContentConfig` has no `response_schema` field
        so the schema cannot be cached — see the module docstring for
        why this is the right design given the SDK's constraints."""
        sdk, _gen, caches_create = _fake_sdk_with_cache(parsed=None)
        client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")

        ok = await client.create_cache()

        assert ok is True
        caches_create.assert_called_once()
        kwargs = caches_create.call_args.kwargs
        assert kwargs["model"] == "gemini-3.1-pro-preview"
        config = kwargs["config"]
        assert config.system_instruction == PAGE_EXTRACTION_PROMPT

    async def test_create_cache_is_idempotent(self) -> None:
        """Pipeline calls `create_cache` once at the start, but a future
        caller (a retry path, a second corpus run) might call it again.
        Second call no-ops rather than spinning up a redundant cache."""
        sdk, _gen, caches_create = _fake_sdk_with_cache(parsed=None)
        client = GeminiClient(sdk=sdk, model="m")

        ok1 = await client.create_cache()
        ok2 = await client.create_cache()

        assert ok1 is True and ok2 is True
        caches_create.assert_called_once()

    async def test_create_cache_returns_false_when_sdk_raises(self) -> None:
        """Caching has a min-content-size threshold (currently ~1024 tokens)
        and some models don't support it at all. Either condition raises
        from the SDK; we degrade to the un-cached path rather than fail
        the entire corpus run."""
        sdk, _gen, _create = _fake_sdk_with_cache(
            parsed=None,
            cache_create_error=RuntimeError("content too small for caching"),
        )
        client = GeminiClient(sdk=sdk, model="m")

        ok = await client.create_cache()

        assert ok is False

    async def test_extract_page_uses_cached_content_after_create_cache(
        self, png_file: Path
    ) -> None:
        """The load-bearing assertion of the whole change: after caching is
        set up, the per-call payload omits the prompt (now in the cache)
        so the ~2-3K prompt tokens aren't re-billed. The response schema
        still travels in the per-call config because the SDK's
        `CreateCachedContentConfig` has no schema field."""
        sdk, generate, _create = _fake_sdk_with_cache(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m")
        await client.create_cache()

        await client.extract_page(png_file)

        contents = generate.call_args.kwargs["contents"]
        # The prompt is in the cache; the per-call contents are image-only.
        assert PAGE_EXTRACTION_PROMPT not in contents
        config = generate.call_args.kwargs["config"]
        assert config.cached_content == "cachedContents/abc123"
        # Schema still travels in the per-call config — SDK limitation.
        assert config.response_schema is GeminiPageResult

    async def test_extract_page_falls_back_when_cache_creation_failed(self, png_file: Path) -> None:
        """`create_cache` returning False must not break `extract_page` —
        the un-cached call site stays the production-correct fallback so a
        cache-creation hiccup doesn't tank a corpus run."""
        sdk, generate, _create = _fake_sdk_with_cache(
            parsed=_sample_page_result(),
            cache_create_error=RuntimeError("nope"),
        )
        client = GeminiClient(sdk=sdk, model="m")
        ok = await client.create_cache()
        assert ok is False

        await client.extract_page(png_file)

        # Un-cached payload: prompt is in contents, schema is in config.
        contents = generate.call_args.kwargs["contents"]
        assert contents[0] == PAGE_EXTRACTION_PROMPT
        config = generate.call_args.kwargs["config"]
        assert config.response_schema is GeminiPageResult

    async def test_extract_page_without_create_cache_uses_uncached_path(
        self, png_file: Path
    ) -> None:
        """When `create_cache` is never called (the default), the client
        behaves exactly as it did before this PR — preserving existing
        test contracts and the un-cached dev-iteration path."""
        sdk, generate, caches_create = _fake_sdk_with_cache(parsed=_sample_page_result())
        client = GeminiClient(sdk=sdk, model="m")

        await client.extract_page(png_file)

        # `caches.create` must NOT have been invoked.
        caches_create.assert_not_called()
        contents = generate.call_args.kwargs["contents"]
        assert contents[0] == PAGE_EXTRACTION_PROMPT


class TestMediaResolution:
    def test_default_is_high(self) -> None:
        # HIGH is recommended for fine handwriting (1120 tokens/image).
        assert MediaResolution.HIGH.value.endswith("HIGH") or MediaResolution.HIGH.value == "high"

    def test_from_string_accepts_canonical_lowercase(self) -> None:
        assert MediaResolution.from_string("high") is MediaResolution.HIGH
        assert MediaResolution.from_string("medium") is MediaResolution.MEDIUM
        assert MediaResolution.from_string("low") is MediaResolution.LOW
        assert MediaResolution.from_string("ultra_high") is MediaResolution.ULTRA_HIGH

    def test_from_string_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            MediaResolution.from_string("epic")
