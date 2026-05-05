"""Gemini extraction client.

Wraps the google-genai SDK with:
  * a clean async surface (`extract_page`),
  * dependency-injected SDK client for testability,
  * a typed `MediaResolution` enum that maps to the SDK's enum,
  * structured-output validation via Pydantic (`response_schema=PageResult`).

Design note: we accept the SDK client at construction time rather than
constructing it inside the class. That makes tests trivial to write
(inject a Mock) and keeps env/credential loading at the call site.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Self

from google.genai import types

from core.prompts import PAGE_EXTRACTION_PROMPT
from core.schema import PageResult


class GeminiError(RuntimeError):
    """Raised for any extraction failure: missing file, bad response, schema mismatch."""


class MediaResolution(Enum):
    """Maps to the SDK's MediaResolution enum.

    The SDK's enum values shift between releases; we wrap them so the rest of
    the code uses stable identifiers and we have one place to update if Google
    renames things.
    """

    LOW = types.MediaResolution.MEDIA_RESOLUTION_LOW
    MEDIUM = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    HIGH = types.MediaResolution.MEDIA_RESOLUTION_HIGH
    # ULTRA_HIGH was added later; tolerate older SDKs lacking it.
    ULTRA_HIGH = getattr(
        types.MediaResolution,
        "MEDIA_RESOLUTION_ULTRA_HIGH",
        types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )

    @classmethod
    def from_string(cls, value: str) -> Self:
        try:
            return cls[value.upper()]
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(
                f"unknown media_resolution {value!r}; expected one of: {valid}"
            ) from exc


class GeminiClient:
    """Async wrapper around the google-genai Client.

    Pass a constructed `genai.Client` (or a mock) as `sdk`. Use the `from_env`
    classmethod for the production path that reads `GEMINI_API_KEY` and builds
    the real client.
    """

    def __init__(
        self,
        *,
        sdk: Any,  # google.genai.Client; typed loosely to keep tests light
        model: str,
        media_resolution: MediaResolution = MediaResolution.HIGH,
    ) -> None:
        self._sdk = sdk
        self._model = model
        self._config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PageResult,
            media_resolution=media_resolution.value,
        )

    async def extract_page(self, image_path: Path) -> PageResult:
        if not image_path.is_file():
            raise GeminiError(f"image not found: {image_path}")

        image_bytes = image_path.read_bytes()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        response = await self._sdk.aio.models.generate_content(
            model=self._model,
            contents=[PAGE_EXTRACTION_PROMPT, image_part],
            config=self._config,
        )

        parsed = response.parsed
        if parsed is None:
            raise GeminiError(
                "Gemini returned no parsed result; the model may have refused or "
                "produced output that did not match the response schema."
            )
        if not isinstance(parsed, PageResult):
            raise GeminiError(
                f"Gemini returned parsed type {type(parsed).__name__}; expected PageResult."
            )
        return parsed
