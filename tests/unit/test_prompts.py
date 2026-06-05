"""Contract tests for the extraction prompt.

The prompt is a string but it carries load-bearing instructions: dropping
any of these clauses silently weakens extraction quality. These tests fail
loudly if a future edit removes a constraint we depend on.
"""

from __future__ import annotations

import pytest

from core.prompts import (
    FOOTER_EXTRACTION_PROMPT,
    HEADER_EXTRACTION_PROMPT,
    PAGE_EXTRACTION_PROMPT,
    QUADRANT_EXTRACTION_PROMPT_TEMPLATE,
)


@pytest.mark.parametrize("position", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_prompt_names_each_quadrant_position(position: str) -> None:
    assert position in PAGE_EXTRACTION_PROMPT


@pytest.mark.parametrize(
    "field",
    [
        "row_index",
        "raw_text",
        "type_raw",
        "confidence",
        "notes",
    ],
)
def test_prompt_names_every_entry_field(field: str) -> None:
    assert field in PAGE_EXTRACTION_PROMPT


@pytest.mark.parametrize("tag", ["continuation", "double_height", "crossed_out", "illegible"])
def test_prompt_lists_every_phase1_notes_tag(tag: str) -> None:
    assert tag in PAGE_EXTRACTION_PROMPT


def test_prompt_double_height_emits_single_entry_on_upper_row() -> None:
    """Drift observed 2026-06-04: fresh Gemini was splitting a 2-grid-row
    handwritten entry into two Entries and tagging the second one
    `continuation`, instead of emitting one Entry tagged `double_height`.
    The prompt must explicitly direct against the split shape — a single
    `double_height` definition without the negation reproduces the drift."""
    text = PAGE_EXTRACTION_PROMPT
    assert "SINGLE Entry" in text
    # The negation: do not also emit a row on the lower line. Allow either
    # "separate" or "second" wording — both are valid phrasings of the rule.
    # Normalise whitespace because the prompt is wrapped.
    normalised = " ".join(text.lower().split())
    assert "do not also emit a separate entry on the lower" in normalised or (
        "do not also emit a second entry on the lower" in normalised
    )


def test_prompt_continuation_definition_excludes_tall_handwriting() -> None:
    """`continuation` is for the SEPARATE-fragment case (visible arrow /
    re-write below). The wording must steer the model away from using
    `continuation` for tall handwriting that spans two grid rows — that
    case belongs to `double_height`."""
    # The directive: tall handwriting routes to double_height, not continuation.
    assert 'use "double_height" instead' in PAGE_EXTRACTION_PROMPT


def test_prompt_crossed_out_excludes_margin_marks() -> None:
    """`crossed_out` was at 27% precision; the prompt must enumerate the
    common false-positive shapes (margin doodles, asterisks, arrows,
    underlines, type-column-only marks) so the model recognises them as
    non-crossed-out."""
    text = PAGE_EXTRACTION_PROMPT.lower()
    assert "through the artist/track text" in text
    false_positives = ["doodle", "asterisk", "arrow", "underline", "type column"]
    found = sum(1 for fp in false_positives if fp in text)
    assert found >= 3, (
        f"expected at least 3 of {false_positives} in the crossed_out clause, found {found}"
    )


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_prompt_lists_every_confidence_value(confidence: str) -> None:
    assert f'"{confidence}"' in PAGE_EXTRACTION_PROMPT


def test_prompt_forbids_abbreviation_expansion() -> None:
    # A future edit that drops the "do not expand" clause is a regression.
    assert "Do not expand abbreviations" in PAGE_EXTRACTION_PROMPT


def test_prompt_forbids_invented_content() -> None:
    assert "Never invent content" in PAGE_EXTRACTION_PROMPT


def test_prompt_demands_exactly_four_quadrants() -> None:
    assert "EXACTLY FOUR quadrants" in PAGE_EXTRACTION_PROMPT


def test_prompt_captures_left_margin_type_column() -> None:
    """Phase 2: the H/M/L/Std/O/R column is captured into `type_raw`. The
    prompt must (a) name the field and (b) enumerate the canonical letters
    so the model knows what shape to look for."""
    assert "type_raw" in PAGE_EXTRACTION_PROMPT
    assert "H, M, L, Std, O, R" in PAGE_EXTRACTION_PROMPT


def test_prompt_keeps_type_letters_out_of_raw_text() -> None:
    """`type_raw` belongs in its own field. The CRITICAL RULE forbidding the
    type letter from leaking into `raw_text` must persist — without it a
    future prompt edit could re-introduce duplicated content."""
    assert "Do not include the left-margin type column" in PAGE_EXTRACTION_PROMPT


def test_prompt_marks_doodles_as_rare_and_forbids_fabrication() -> None:
    """An earlier prompt revision listed concrete doodle examples without
    qualifying frequency; the calibration spot-check showed the model
    parroting the example string ("hand-drawn smiley with tongue") onto
    rows where the type column was simply unreadable. The fix is two
    instructions that must both stay in the prompt: doodles are RARE,
    and example strings must NOT be invented onto unclear rows."""
    text = PAGE_EXTRACTION_PROMPT
    assert "RARE" in text, "doodle frequency qualifier must persist"
    # The instruction must be a NEGATION of fabrication, not just a mention.
    assert "do not invent" in text.lower() or "not invent" in text.lower()


def test_prompt_specifies_json_null_for_blank_type_column() -> None:
    """The model emitted the literal string "null" and "" for blank circles
    in the spot-check; the prompt must explicitly say JSON null, not a
    string."""
    assert "JSON null" in PAGE_EXTRACTION_PROMPT


def test_prompt_documents_oddities_field() -> None:
    """Prompt must explain the oddities list — otherwise Gemini won't fill it."""
    assert "oddities" in PAGE_EXTRACTION_PROMPT


@pytest.mark.parametrize("level_keyword", ["entry", "quadrant", "page"])
def test_prompt_distinguishes_three_oddity_levels(level_keyword: str) -> None:
    # The three lists serve different purposes; the prompt must call out each.
    assert level_keyword in PAGE_EXTRACTION_PROMPT.lower()


def test_prompt_warns_against_duplicating_existing_fields() -> None:
    """Oddities must not repeat what `notes` / `crossed_out` etc. already capture."""
    text = PAGE_EXTRACTION_PROMPT.lower()
    assert "do not repeat" in text or "don't repeat" in text


def test_prompt_captures_bottom_comments_field() -> None:
    """Phase 2: the bottom-of-page Comments band lands in `comments_raw`. The
    prompt must (a) name the field, (b) locate it (bottom of the page so the
    model knows what to look at), and (c) say verbatim — otherwise it'll get
    cleaned up like an editor."""
    text = PAGE_EXTRACTION_PROMPT
    assert "comments_raw" in text
    assert "bottom" in text.lower()
    assert "verbatim" in text.lower()


def test_prompt_specifies_json_null_for_blank_comments_field() -> None:
    """Blank Comments band must be null, not "" — same convention as
    `type_raw` / `hour_raw` / `jock_raw`. Otherwise consumers can't
    distinguish "blank" from "they wrote an empty string"."""
    # The model must be told what to emit when the field is blank.
    assert "comments_raw" in PAGE_EXTRACTION_PROMPT
    # Either a dedicated "blank -> null" sentence near comments_raw, or the
    # global JSON-null rule must be in force. Check the prompt explicitly
    # tells the model to use null for a blank comments field.
    lowered = PAGE_EXTRACTION_PROMPT.lower()
    # Look for "null" near "comments" — anything that gives the model the
    # signal. Cheap proximity check: same sentence-ish window.
    idx = lowered.find("comments_raw")
    window = lowered[idx : idx + 500]
    assert "null" in window, "expected the comments_raw section to specify null for blank fields"


def test_prompt_keeps_comments_out_of_page_oddities() -> None:
    """Before Phase 2, the prompt nudged the model to stash the Comments
    contents inside `oddities` (as a page-level oddity). With `comments_raw`
    in place, double-capturing would dilute oddities and produce duplicated
    text downstream. The prompt must explicitly tell the model NOT to do
    that, and the old illustrative example must be gone from the oddities
    section."""
    text = PAGE_EXTRACTION_PROMPT
    # The old illustrative example contained the literal "Comments field at
    # the bottom contains" — that must be removed.
    assert "Comments field at the bottom contains" not in text, (
        "old page-oddities example for the comments field must be removed — "
        "the contents now belong in `comments_raw` instead"
    )
    # A clear negation must remain so a future prompt edit can't re-introduce
    # the duplication by accident.
    lowered = text.lower()
    assert "do not" in lowered and "comments" in lowered, (
        "expected a 'do not …  comments' clause anchoring the negation"
    )


# -- QUADRANT_EXTRACTION_PROMPT_TEMPLATE -----------------------------------


@pytest.mark.parametrize("position", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_quadrant_template_substitutes_position(position: str) -> None:
    """Templating with each canonical position must produce a string that
    names the position — that's what tells the model which cell it's looking at."""
    rendered = QUADRANT_EXTRACTION_PROMPT_TEMPLATE.format(position=position)
    assert position in rendered


@pytest.mark.parametrize(
    "field",
    [
        "row_index",
        "raw_text",
        "type_raw",
        "confidence",
        "notes",
    ],
)
def test_quadrant_template_names_every_entry_field(field: str) -> None:
    """Per-row guidance must stay in lock-step with PAGE_EXTRACTION_PROMPT —
    the row-level rules apply identically whether the model sees one
    quadrant or all four."""
    assert field in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


@pytest.mark.parametrize("tag", ["continuation", "double_height", "crossed_out", "illegible"])
def test_quadrant_template_lists_every_phase1_notes_tag(tag: str) -> None:
    assert tag in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_double_height_emits_single_entry_on_upper_row() -> None:
    text = QUADRANT_EXTRACTION_PROMPT_TEMPLATE
    assert "SINGLE Entry" in text
    normalised = " ".join(text.lower().split())
    assert "do not also emit a separate entry on the lower" in normalised or (
        "do not also emit a second entry on the lower" in normalised
    )


def test_quadrant_template_continuation_excludes_tall_handwriting() -> None:
    assert 'use "double_height" instead' in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_crossed_out_excludes_margin_marks() -> None:
    text = QUADRANT_EXTRACTION_PROMPT_TEMPLATE.lower()
    assert "through the artist/track text" in text
    false_positives = ["doodle", "asterisk", "arrow", "underline", "type column"]
    found = sum(1 for fp in false_positives if fp in text)
    assert found >= 3


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_quadrant_template_lists_every_confidence_value(confidence: str) -> None:
    assert f'"{confidence}"' in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_forbids_abbreviation_expansion() -> None:
    assert "Do not expand abbreviations" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_forbids_invented_content() -> None:
    assert "Never invent content" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_captures_left_margin_type_column() -> None:
    assert "type_raw" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE
    assert "H, M, L, Std, O, R" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_keeps_type_letters_out_of_raw_text() -> None:
    assert "Do not include the left-margin type column" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_marks_doodles_as_rare_and_forbids_fabrication() -> None:
    text = QUADRANT_EXTRACTION_PROMPT_TEMPLATE
    assert "RARE" in text
    assert "do not invent" in text.lower() or "not invent" in text.lower()


def test_quadrant_template_specifies_json_null_for_blank_type_column() -> None:
    assert "JSON null" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_explicitly_excludes_page_level_oddities() -> None:
    """Page-level oddities are captured by HEADER_EXTRACTION_PROMPT — the
    quadrant prompt must tell the model not to duplicate them."""
    text = QUADRANT_EXTRACTION_PROMPT_TEMPLATE.lower()
    assert "page-level" in text
    # The instruction must be a NEGATION (do not / not) of page-level oddities,
    # not just a mention.
    negation_phrases = ["do not emit page-level", "not include", "must not"]
    assert any(p in text for p in negation_phrases), (
        "expected the quadrant prompt to negate page-level oddities, "
        f"none of {negation_phrases!r} appeared"
    )


def test_quadrant_template_warns_about_bleed_band() -> None:
    """The crop overlaps neighbors slightly; the prompt must tell the
    model to ignore content that bleeds in from outside the cell."""
    assert "bleed" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE.lower()


# -- HEADER_EXTRACTION_PROMPT ----------------------------------------------


def test_header_prompt_captures_page_date_raw() -> None:
    assert "page_date_raw" in HEADER_EXTRACTION_PROMPT


def test_header_prompt_scopes_oddities_to_page_level() -> None:
    """The header prompt must restrict the model to page-level notes —
    otherwise it would happily transcribe row content from the strip if
    any bled into the crop."""
    text = HEADER_EXTRACTION_PROMPT.lower()
    assert "page-level" in text
    assert "row-level" in text or "quadrant-level" in text
    assert "do not" in text


def test_header_prompt_forbids_invented_content() -> None:
    assert "Never invent content" in HEADER_EXTRACTION_PROMPT


# -- FOOTER_EXTRACTION_PROMPT ----------------------------------------------


def test_footer_prompt_captures_comments_raw() -> None:
    assert "comments_raw" in FOOTER_EXTRACTION_PROMPT


def test_footer_prompt_demands_verbatim_transcription() -> None:
    """The Comments band is free-text DJ commentary — the model must not
    clean it up like an editor."""
    assert "verbatim" in FOOTER_EXTRACTION_PROMPT.lower()


def test_footer_prompt_specifies_json_null_for_blank() -> None:
    """Blank comments band must round-trip as null, not "" — same convention
    as page_date_raw / hour_raw / jock_raw."""
    assert "JSON null" in FOOTER_EXTRACTION_PROMPT


def test_footer_prompt_scopes_to_footer_band() -> None:
    """The footer crop slightly overlaps the bottom-quadrant baseline; the
    prompt must tell the model to ignore content above the Comments line —
    otherwise the model will helpfully transcribe the last row of the
    bottom quadrants into comments_raw."""
    text = FOOTER_EXTRACTION_PROMPT.lower()
    assert "comments" in text
    # Negate transcribing content from above the Comments line.
    assert "do not" in text
    assert "above" in text


def test_footer_prompt_forbids_invented_content() -> None:
    assert "Never invent content" in FOOTER_EXTRACTION_PROMPT
