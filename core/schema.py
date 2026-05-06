"""Pydantic models for the Gemini structured-output contract.

These models are the single source of truth for both:
  * the response_schema sent to Gemini, and
  * the validated shape stored to disk.

Phase 1 captures only the per-row text and the four-quadrant frame. The
left-margin type column (H/M/L/Std/O/R/R⇒), continuation/double-height
handling, the comments field, and reconciliation against the WXYC library
are all phase 2.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

Confidence = Literal["high", "medium", "low"]
QuadrantPosition = Literal["top_left", "top_right", "bottom_left", "bottom_right"]

QUADRANT_ORDER: tuple[QuadrantPosition, ...] = get_args(QuadrantPosition)


class Entry(BaseModel):
    """A single handwritten row inside a quadrant."""

    row_index: NonNegativeInt = Field(description="0-based row position within the quadrant.")
    raw_text: str = Field(
        description=(
            "Verbatim transcription of the line. Do not expand abbreviations or normalize "
            "spacing. If unreadable, give a best-effort partial transcription."
        )
    )
    artist_guess: str | None = Field(
        default=None,
        description="Best-effort parse of the artist portion (left of the dash).",
    )
    track_guess: str | None = Field(
        default=None,
        description="Best-effort parse of the track portion (right of the dash).",
    )
    confidence: Confidence = Field(
        description="high if the row is clearly legible; low if mostly illegible.",
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Free-text marker for special cases deferred to phase 2. "
            "Use one of: continuation, double_height, crossed_out, illegible, other."
        ),
    )


class Quadrant(BaseModel):
    """One of the four hour-blocks on a flowsheet page."""

    position: QuadrantPosition = Field(
        description="Which quadrant this is (top_left, top_right, bottom_left, bottom_right)."
    )
    hour_raw: str | None = Field(
        default=None,
        description="Verbatim hour label (e.g. '6AM', '7PM', '10°'). None if blank.",
    )
    jock_raw: str | None = Field(
        default=None,
        description="Verbatim DJ name. None if blank.",
    )
    entries: list[Entry] = Field(
        default_factory=list,
        description="Rows in the quadrant, in the order they appear on the page.",
    )


class PageResult(BaseModel):
    """The full extraction for one flowsheet page."""

    page_date_raw: str | None = Field(
        default=None,
        description=(
            'Verbatim date as written at the top of the page (e.g. "Monday 1 Jan \'90"). '
            "None if blank or unreadable. Date normalization happens downstream."
        ),
    )
    quadrants: list[Quadrant] = Field(
        description=(
            "Exactly four quadrants in fixed order: top_left, top_right, bottom_left, "
            "bottom_right. Always return all four even if a quadrant is blank."
        )
    )
    model_version: str = Field(description="Gemini model id that produced this result.")
    extracted_at: datetime = Field(description="When the extraction completed.")

    @model_validator(mode="after")
    def _check_quadrant_order(self) -> PageResult:
        if len(self.quadrants) != 4:
            raise ValueError(
                f"expected exactly 4 quadrants in fixed order {QUADRANT_ORDER}, "
                f"got {len(self.quadrants)}"
            )
        actual = tuple(q.position for q in self.quadrants)
        if actual != QUADRANT_ORDER:
            raise ValueError(f"quadrants must be in order {QUADRANT_ORDER}, got {actual}")
        return self
