"""Contract tests for the extraction prompt.

The prompt is a string but it carries load-bearing instructions: dropping
any of these clauses silently weakens extraction quality. These tests fail
loudly if a future edit removes a constraint we depend on.
"""

from __future__ import annotations

import pytest

from core.prompts import (
    HEADER_EXTRACTION_PROMPT,
    PAGE_EXTRACTION_PROMPT,
    QUADRANT_EXTRACTION_PROMPT_TEMPLATE,
)


@pytest.mark.parametrize("position", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_prompt_names_each_quadrant_position(position: str) -> None:
    assert position in PAGE_EXTRACTION_PROMPT


@pytest.mark.parametrize(
    "field", ["row_index", "raw_text", "artist_guess", "track_guess", "confidence", "notes"]
)
def test_prompt_names_every_entry_field(field: str) -> None:
    assert field in PAGE_EXTRACTION_PROMPT


@pytest.mark.parametrize("tag", ["continuation", "double_height", "crossed_out", "illegible"])
def test_prompt_lists_every_phase1_notes_tag(tag: str) -> None:
    assert tag in PAGE_EXTRACTION_PROMPT


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


def test_prompt_excludes_left_margin_type_column() -> None:
    # The H/M/L/Std/O/R column is phase 2; the prompt must say so.
    assert "IGNORE this column" in PAGE_EXTRACTION_PROMPT


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


# -- QUADRANT_EXTRACTION_PROMPT_TEMPLATE -----------------------------------


@pytest.mark.parametrize("position", ["top_left", "top_right", "bottom_left", "bottom_right"])
def test_quadrant_template_substitutes_position(position: str) -> None:
    """Templating with each canonical position must produce a string that
    names the position — that's what tells the model which cell it's looking at."""
    rendered = QUADRANT_EXTRACTION_PROMPT_TEMPLATE.format(position=position)
    assert position in rendered


@pytest.mark.parametrize(
    "field", ["row_index", "raw_text", "artist_guess", "track_guess", "confidence", "notes"]
)
def test_quadrant_template_names_every_entry_field(field: str) -> None:
    """Per-row guidance must stay in lock-step with PAGE_EXTRACTION_PROMPT —
    the row-level rules apply identically whether the model sees one
    quadrant or all four."""
    assert field in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


@pytest.mark.parametrize("tag", ["continuation", "double_height", "crossed_out", "illegible"])
def test_quadrant_template_lists_every_phase1_notes_tag(tag: str) -> None:
    assert tag in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_quadrant_template_lists_every_confidence_value(confidence: str) -> None:
    assert f'"{confidence}"' in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_forbids_abbreviation_expansion() -> None:
    assert "Do not expand abbreviations" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_forbids_invented_content() -> None:
    assert "Never invent content" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


def test_quadrant_template_excludes_left_margin_type_column() -> None:
    assert "IGNORE this column" in QUADRANT_EXTRACTION_PROMPT_TEMPLATE


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
