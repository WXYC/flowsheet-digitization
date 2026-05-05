"""Contract tests for the extraction prompt.

The prompt is a string but it carries load-bearing instructions: dropping
any of these clauses silently weakens extraction quality. These tests fail
loudly if a future edit removes a constraint we depend on.
"""

from __future__ import annotations

import pytest

from core.prompts import PAGE_EXTRACTION_PROMPT


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
