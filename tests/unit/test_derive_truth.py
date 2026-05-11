"""Tests for `scripts/derive_truth.py`.

The truth-derivation tool consumes a `<stem>.verified.json` (PageResult-
shaped) and emits `<stem>.truth.json` (GoldenTruth-shaped) by extracting
short substrings from the user-corrected raw_text. Tests pin the
substring rules and the end-to-end CLI flow.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.golden import GoldenTruth
from core.schema import QUADRANT_ORDER, Entry, PageResult, Quadrant
from scripts.derive_truth import (
    _date_substrings,
    _jock_substring,
    _row_substring,
    derive_truth,
    main,
)

# -- _date_substrings -------------------------------------------------------


@pytest.mark.parametrize(
    ("page_date_raw", "expected"),
    [
        ("Tues 4/3 90", ["Tues", "4/3", "90"]),
        ("Monday 1 Jan '90", ["Monday", "1", "Jan", "'90"]),
        ("", []),
        (None, []),
        ("   ", []),  # whitespace-only
    ],
)
def test_date_substrings(page_date_raw: str | None, expected: list[str]) -> None:
    assert _date_substrings(page_date_raw) == expected


# -- _jock_substring --------------------------------------------------------


@pytest.mark.parametrize(
    ("jock_raw", "expected"),
    [
        ("Andrew", "ANDR"),
        ("ANDREW", "ANDR"),
        ("Andy J", "ANDY"),  # first token only
        ("Sam", "SAM"),  # shorter than 4 chars passes through
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_jock_substring(jock_raw: str | None, expected: str | None) -> None:
    assert _jock_substring(jock_raw) == expected


# -- _row_substring ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        # Examples from the plan body, matching existing golden truth convention.
        ("Beastie Boys - Sabotage", "BEASTIE BOYS"),
        ("Primal Scream - Loaded", "PRIMAL SCREAM"),
        ("Bo Diddley - Hey Bo", "BO DIDDLEY"),
        ("Elizabeth Cotten - Shake", "ELIZABETH COTTEN"),
        ("JUANA MOLINA - la paradoja", "JUANA MOLINA"),
        # No separator: full text uppercased, truncated at 24 chars (snap to ws).
        ("standalone continuation text here", "STANDALONE CONTINUATION"),
        ("short text", "SHORT TEXT"),
        # Exactly 24 chars: unchanged.
        ("a" * 24, "A" * 24),
        # 25 chars no whitespace: hard-cut at 24.
        ("a" * 25, "A" * 24),
        # Em-dash separator (handled by parse_artist_track).
        ("Hermanos Gutiérrez — Aguas Ardientes", "HERMANOS GUTIÉRREZ"),
    ],
)
def test_row_substring(raw_text: str, expected: str) -> None:
    assert _row_substring(raw_text) == expected


# -- derive_truth -----------------------------------------------------------


def _entry(text: str, idx: int = 0) -> Entry:
    return Entry(row_index=idx, raw_text=text, confidence="high")


def _quad(position: str, jock: str | None, hour: str | None, entries: list[Entry]) -> Quadrant:
    return Quadrant(
        position=position,  # type: ignore[arg-type]
        hour_raw=hour,
        jock_raw=jock,
        entries=entries,
    )


def _page(date: str | None, quads: list[Quadrant]) -> PageResult:
    return PageResult(
        page_date_raw=date,
        quadrants=quads,
        oddities=[],
        model_version="test-verified",
        extracted_at=datetime(2026, 5, 10, tzinfo=UTC),
    )


def test_derive_truth_returns_golden_truth_with_all_quadrants() -> None:
    page = _page(
        "Tues 4/3 90",
        [
            _quad("top_left", "Andrew", "6AM", [_entry("Primal Scream - Loaded")]),
            _quad("top_right", None, "7AM", [_entry("Beastie Boys - Sabotage")]),
            _quad("bottom_left", "Andrew", "8AM", [_entry("Bo Diddley - Hey Bo")]),
            _quad("bottom_right", None, "9AM", [_entry("Juana Molina - la paradoja")]),
        ],
    )
    truth = derive_truth(page)
    assert isinstance(truth, GoldenTruth)
    assert [q.position for q in truth.quadrants] == list(QUADRANT_ORDER)


def test_derive_truth_page_date_split_into_tokens() -> None:
    page = _page("Tues 4/3 90", [_quad(p, None, None, []) for p in QUADRANT_ORDER])
    truth = derive_truth(page)
    assert truth.page_date_substrings == ["Tues", "4/3", "90"]


def test_derive_truth_quadrant_substrings_match_rules() -> None:
    page = _page(
        None,
        [
            _quad("top_left", "Andrew", "6AM", [_entry("Primal Scream - Loaded")]),
            _quad("top_right", None, None, [_entry("Beastie Boys - Sabotage")]),
            _quad("bottom_left", None, None, []),
            _quad("bottom_right", None, None, [_entry("Bo Diddley - Hey Bo")]),
        ],
    )
    truth = derive_truth(page)
    by_pos = {q.position: q for q in truth.quadrants}
    assert by_pos["top_left"].jock_substring == "ANDR"
    assert by_pos["top_left"].hour_raw == "6AM"
    assert [r.raw_substring for r in by_pos["top_left"].rows] == ["PRIMAL SCREAM"]
    assert by_pos["top_right"].jock_substring is None
    assert [r.raw_substring for r in by_pos["top_right"].rows] == ["BEASTIE BOYS"]
    assert by_pos["bottom_left"].rows == []
    assert [r.raw_substring for r in by_pos["bottom_right"].rows] == ["BO DIDDLEY"]


def test_derive_truth_skips_empty_raw_text_rows() -> None:
    """An entry with empty raw_text shouldn't produce a truth row — there's
    nothing to match against."""
    page = _page(
        None,
        [
            _quad("top_left", None, None, [_entry(""), _entry("Primal Scream")]),
            _quad("top_right", None, None, []),
            _quad("bottom_left", None, None, []),
            _quad("bottom_right", None, None, []),
        ],
    )
    truth = derive_truth(page)
    by_pos = {q.position: q for q in truth.quadrants}
    assert [r.raw_substring for r in by_pos["top_left"].rows] == ["PRIMAL SCREAM"]


# -- main CLI ---------------------------------------------------------------


def test_main_writes_truth_file(tmp_path: Path) -> None:
    page = _page(
        "Tues 4/3 90",
        [
            _quad("top_left", "Andrew", "6AM", [_entry("Primal Scream - Loaded")]),
            _quad("top_right", None, None, [_entry("Beastie Boys - Sabotage")]),
            _quad("bottom_left", None, None, []),
            _quad("bottom_right", None, None, []),
        ],
    )
    verified_path = tmp_path / "verified.json"
    verified_path.write_text(page.model_dump_json(indent=2))

    out_path = tmp_path / "out" / "truth.json"
    rc = main([str(verified_path), "--out", str(out_path)])
    assert rc == 0

    truth = GoldenTruth.load(out_path)
    assert truth.page_date_substrings == ["Tues", "4/3", "90"]
    by_pos = {q.position: q for q in truth.quadrants}
    assert [r.raw_substring for r in by_pos["top_left"].rows] == ["PRIMAL SCREAM"]


def test_main_returns_one_when_input_missing(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "missing.json"), "--out", str(tmp_path / "out.json")])
    assert rc == 1


def test_main_round_trips_through_pydantic(tmp_path: Path) -> None:
    """End-to-end: PageResult on disk → derive_truth main → GoldenTruth on
    disk → GoldenTruth.load. Pins the export schema."""
    page = _page("Mon 5 May", [_quad(p, None, None, []) for p in QUADRANT_ORDER])
    verified_path = tmp_path / "verified.json"
    verified_path.write_text(page.model_dump_json(indent=2))

    out_path = tmp_path / "truth.json"
    main([str(verified_path), "--out", str(out_path)])

    # Both load the same data.
    loaded_from_disk = GoldenTruth.load(out_path)
    assert loaded_from_disk.page_date_substrings == ["Mon", "5", "May"]
    # Round-trip a raw dict too — extra fields would be caught by extra=forbid.
    raw = json.loads(out_path.read_text())
    GoldenTruth.model_validate(raw)
