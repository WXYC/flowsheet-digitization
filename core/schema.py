"""Pydantic models for the Gemini structured-output contract.

The response_schema sent to Gemini and the on-disk shape are *almost*
the same model — they share `page_date_raw`, `quadrants`, `comments_raw`,
and page-level `oddities`. They differ in two fields the caller owns,
not Gemini:

  * `model_version` — the SDK arg, set by the pipeline at write-time.
  * `extracted_at`  — wall-clock UTC at the call site.

If those two fields are part of the response_schema, Gemini fills them
with hallucinated plausible values (real run with `gemini-3.1-pro-preview`
produced 4 distinct fake model ids and timestamps off by 14+ months).
The split avoids that:

  * `GeminiPageResult` is what Gemini returns. Used as `response_schema`.
  * `PageResult` is the on-disk shape — `GeminiPageResult` plus the two
    caller-set fields, populated by `pipeline._process_one_job`.

Phase 1 captures the per-row text and the four-quadrant frame. Phase 2
adds the left-margin type column (H/M/L/Std/O/R/R⇒, in `Entry.type_raw`),
the bottom-of-page comments field (`GeminiPageResult.comments_raw`), and
is iteratively rolling out continuation/double-height handling and
reconciliation against the WXYC library.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, get_args

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
    type_raw: str | None = Field(
        default=None,
        description=(
            "Verbatim character(s) from the printed type-column circle to the LEFT "
            "of this row. Common values: 'H' (heavy rotation), 'M' (medium), "
            "'L' (light), 'Std' (standards), 'O' (oldies), 'R' (request, sometimes "
            "written 'R⇒' for handoff). Keep verbatim — do NOT normalize 'Std' to "
            "'std' or expand abbreviations. If the circle contains a doodle (e.g. "
            "a face) instead of a letter, set type_raw to a short description "
            "('hand-drawn smiley with tongue'); the rest of the row is still a "
            "normal entry. Null if the circle is blank."
        ),
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
    oddities: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text descriptions of anything specific to THIS row that the rest of "
            "the schema doesn't capture (e.g. a hand-drawn arrow next to it, an "
            "asterisk in the right margin). Empty list if nothing unusual. Each item "
            "is one short sentence."
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
    oddities: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text descriptions of multi-row visual structure within this "
            "quadrant that the schema doesn't capture (e.g. a curly brace "
            "grouping rows 4-8 with a label, an arrow drawn from row 3 to row 6). "
            "Empty list if nothing unusual."
        ),
    )


class GeminiPageResult(BaseModel):
    """The page-level subset that Gemini actually fills.

    Used directly as `response_schema` on the SDK call. Has no
    caller-set metadata — see the module docstring for why.
    """

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
    comments_raw: str | None = Field(
        default=None,
        description=(
            "Verbatim contents of the printed 'Comments' field at the bottom of "
            "the page (free-text DJ commentary about the broadcast — e.g. "
            '"declared today anti-Valentines Day"). Null when the field is '
            "blank, unreadable, or absent from the form. Keep verbatim: do not "
            "normalize spelling, fix grammar, expand abbreviations, or truncate. "
            "Multi-line entries are joined with a single newline. This field "
            "replaces capturing the comments field as a page-level oddity — "
            "do NOT also list the comments contents under `oddities`."
        ),
    )
    oddities: list[str] = Field(
        default_factory=list,
        description=(
            "Free-text descriptions of anything on the page OUTSIDE the four "
            "quadrants and the comments field — content the schema doesn't have "
            "a place for. Examples: the page is rotated, there is a header note "
            "above the date, the right column has a DJ-handoff message, "
            "marginal notes appear next to the grid. Empty list if nothing "
            "unusual. Each item is one short sentence. The bottom comments "
            "field has its own `comments_raw` slot — do not repeat its "
            "contents here."
        ),
    )

    @model_validator(mode="after")
    def _check_quadrant_order(self) -> Self:
        if len(self.quadrants) != 4:
            raise ValueError(
                f"expected exactly 4 quadrants in fixed order {QUADRANT_ORDER}, "
                f"got {len(self.quadrants)}"
            )
        actual = tuple(q.position for q in self.quadrants)
        if actual != QUADRANT_ORDER:
            raise ValueError(f"quadrants must be in order {QUADRANT_ORDER}, got {actual}")
        return self


class VerifiedBy(BaseModel):
    """Identity of the reviewer who saved this verified page.

    Populated only by `verifier/serve.py`'s POST /api/save handler when
    an authenticated reviewer session is present. Pipeline-written
    PageResults leave `PageResult.verified_by` as None.

    `user_id` is deliberately denormalized to `jobs.reviewer_id` (see
    `core/jobs.py`) so per-reviewer queries don't have to parse every
    JSON file on disk. The two values are always equal for the same
    save call; both are written inside `/api/save` under server
    authority — the client never sets either. A client-supplied
    `verified_by` block on a save POST is overwritten with the
    authenticated reviewer's values before persistence (see the
    handler).
    """

    user_id: str = Field(description="Better Auth user.id (the OIDC `sub` claim).")
    username: str | None = Field(
        default=None,
        description="Better Auth username, if set.",
    )
    real_name: str | None = Field(
        default=None,
        description="Reviewer's real name from the WXYC user record.",
    )
    dj_name: str | None = Field(
        default=None,
        description="Reviewer's on-air DJ name, if set.",
    )
    verified_at: datetime = Field(
        description="When the verifier UI saved this page (UTC).",
    )


class PageResult(GeminiPageResult):
    """On-disk shape: `GeminiPageResult` plus the fields the caller owns.

    `model_version` and `extracted_at` are filled by the pipeline (or by
    each calibration adapter) at write-time. They are NOT part of the
    Gemini response_schema — see the module docstring.

    `verified_by` is populated only by `verifier/serve.py` when a
    reviewer saves a page through the verifier UI. Defaulting to None
    keeps every pre-OIDC `verified.json` parseable (the field is
    silently absent on old files); new saves write the block and old
    files are upgraded on the next save.
    """

    model_version: str = Field(description="Model id that produced this result.")
    extracted_at: datetime = Field(description="When the extraction completed (UTC).")
    verified_by: VerifiedBy | None = Field(
        default=None,
        description=(
            "Reviewer who saved this corrected page via the verifier UI. "
            "None for results that have only been through the automatic pipeline."
        ),
    )
