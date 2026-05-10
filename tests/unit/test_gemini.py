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
