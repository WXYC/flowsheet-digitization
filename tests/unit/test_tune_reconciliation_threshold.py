"""Tests for `scripts/tune_reconciliation_threshold.py`.

Cover the pure pieces (pair extraction, curve computation, threshold
picker). The async LML scoring path is exercised through `LMLClient`'s
own tests; here we just check the math.
"""

from __future__ import annotations

from pathlib import Path

from scripts import tune_reconciliation_threshold as tune


def test_extract_pairs_from_corrections_picks_up_raw_text_edits() -> None:
    corrections = {
        "status": "complete",
        "row_corrections": [
            {
                "position": "top_right",
                "row_index": 1,
                "field": "raw_text",
                "original": "Mothers - Absolutely Free",
                "corrected": "The Mothers of Invention - Absolutely Free",
            },
            # Different field — should be ignored.
            {
                "position": "top_right",
                "row_index": 1,
                "field": "type_raw",
                "original": None,
                "corrected": "O",
            },
        ],
    }
    pairs = tune._extract_pairs_from_corrections("page-x", corrections)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.stem == "page-x"
    assert p.quadrant == "top_right"
    assert p.row_index == 1
    assert p.gemini_artist == "Mothers"
    assert p.alex_artist == "The Mothers of Invention"


def test_extract_pairs_skips_artist_unchanged() -> None:
    """If the artist text matches modulo case+whitespace, the edit was
    on the track side and reconciliation can't speak to it."""
    corrections = {
        "status": "complete",
        "row_corrections": [
            {
                "position": "top_left",
                "row_index": 0,
                "field": "raw_text",
                "original": "STEREOLAB - Cybeles",
                "corrected": "STEREOLAB - Cybele's Reverie",
            },
        ],
    }
    pairs = tune._extract_pairs_from_corrections("page-x", corrections)
    assert pairs == []


def test_extract_pairs_skips_rows_with_no_artist() -> None:
    """Pure track-side or blank rows produce no pair."""
    corrections = {
        "status": "complete",
        "row_corrections": [
            {
                "position": "top_left",
                "row_index": 0,
                "field": "raw_text",
                "original": "",
                "corrected": "blank",
            },
        ],
    }
    pairs = tune._extract_pairs_from_corrections("page-x", corrections)
    assert pairs == []


def test_extract_pairs_skips_blank_placeholder_pairs() -> None:
    """Mirror `core.reconciliation`'s `_is_blank_placeholder` skip.

    Alex's `blank` placeholder for physically-empty rows is the dominant
    "Alex changed the artist" pair in the corpus (n=65), but reconciliation
    won't send `blank` to LML in production. Including these pairs in the
    tuning curve falsely inflates the denominator (recall) and forces
    LML lookups on bogus inputs that can't fire. The tuning script must
    skip them — both directions: blank in original, blank in corrected.
    """
    corrections = {
        "status": "complete",
        "row_corrections": [
            # original is a real artist but Alex marked the row blank.
            {
                "position": "top_left",
                "row_index": 0,
                "field": "raw_text",
                "original": "time change",
                "corrected": "blank",
            },
            # Gemini emitted "blank" (impossible in practice but covers
            # the symmetric case).
            {
                "position": "top_left",
                "row_index": 1,
                "field": "raw_text",
                "original": "blank",
                "corrected": "Real Artist - Track",
            },
        ],
    }
    pairs = tune._extract_pairs_from_corrections("page-x", corrections)
    assert pairs == []


def test_load_pairs_skips_draft_pages(tmp_path: Path) -> None:
    """Draft pages are not ground truth — `status: draft` excludes them."""
    import json

    (tmp_path / "p1.corrections.json").write_text(
        json.dumps(
            {
                "status": "draft",
                "row_corrections": [
                    {
                        "position": "top_left",
                        "row_index": 0,
                        "field": "raw_text",
                        "original": "Sterolab - x",
                        "corrected": "Stereolab - x",
                    }
                ],
            }
        )
    )
    (tmp_path / "p2.corrections.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "row_corrections": [
                    {
                        "position": "top_left",
                        "row_index": 0,
                        "field": "raw_text",
                        "original": "Mothers - x",
                        "corrected": "The Mothers of Invention - x",
                    }
                ],
            }
        )
    )
    pairs = tune._load_pairs(tmp_path)
    assert {p.stem for p in pairs} == {"p2"}


def test_compute_curve_precision_recall_basic() -> None:
    """One TP, one FP, one no-correction.

    Score >= 80 should fire on both the TP and the FP -> precision = 0.5,
    recall = 1/3 (TP over the universe of 3 pairs).
    """
    pairs = [
        tune.Pair(
            stem="x",
            quadrant="top_left",
            row_index=0,
            gemini_artist="MZRG",
            alex_artist="Beethoven",
        ),
        tune.Pair(
            stem="x",
            quadrant="top_left",
            row_index=1,
            gemini_artist="Stereolab",
            alex_artist="Stereolab Live",
        ),
        tune.Pair(
            stem="x", quadrant="top_left", row_index=2, gemini_artist="weird", alex_artist="Other"
        ),
    ]
    scored = [
        tune.Scored(pair=pairs[0], corrected_artist="Beethoven", score=100),  # TP at any T
        tune.Scored(pair=pairs[1], corrected_artist="Wrong", score=85),  # FP at 80
        tune.Scored(pair=pairs[2], corrected_artist=None, score=0),  # not eligible
    ]
    curve = tune._compute_curve(scored, [80, 90, 95, 100])
    by_t = {p.threshold: p for p in curve}
    # T=80: both auto-accept, 1 TP -> precision 0.5
    assert by_t[80].auto_accepts == 2
    assert by_t[80].true_positives == 1
    assert by_t[80].precision == 0.5
    assert by_t[80].recall == 1 / 3
    # T=95: only the Beethoven row qualifies; it's the TP.
    assert by_t[95].auto_accepts == 1
    assert by_t[95].true_positives == 1
    assert by_t[95].precision == 1.0


def test_pick_threshold_chooses_smallest_meeting_precision() -> None:
    pairs = [
        tune.Pair(
            stem="x", quadrant="top_left", row_index=i, gemini_artist=f"g{i}", alex_artist=f"a{i}"
        )
        for i in range(5)
    ]
    # At T=80, 2 TP + 2 FP -> precision 0.5
    # At T=90, 2 TP + 0 FP -> precision 1.0
    scored = [
        tune.Scored(pair=pairs[0], corrected_artist="a0", score=100),  # TP
        tune.Scored(pair=pairs[1], corrected_artist="a1", score=95),  # TP
        tune.Scored(pair=pairs[2], corrected_artist="wrong", score=85),  # FP
        tune.Scored(pair=pairs[3], corrected_artist="wrong", score=80),  # FP
        tune.Scored(pair=pairs[4], corrected_artist=None, score=0),
    ]
    curve = tune._compute_curve(scored, [80, 85, 90, 95, 100])
    pick = tune._pick_threshold(curve, min_precision=0.95)
    assert pick is not None
    assert pick.threshold == 90


def test_pick_threshold_returns_none_when_nothing_qualifies() -> None:
    """No T should be picked if precision never hits the floor."""
    pairs = [
        tune.Pair(stem="x", quadrant="top_left", row_index=0, gemini_artist="g", alex_artist="a")
    ]
    scored = [
        tune.Scored(pair=pairs[0], corrected_artist="wrong", score=100),
    ]
    curve = tune._compute_curve(scored, [50, 100])
    assert tune._pick_threshold(curve, min_precision=0.95) is None


def test_non_latin_fraction_counts_non_ascii() -> None:
    pairs = [
        tune.Pair(
            stem="x", quadrant="top_left", row_index=0, gemini_artist="a", alex_artist="Sigur Rós"
        ),
        tune.Pair(
            stem="x", quadrant="top_left", row_index=1, gemini_artist="a", alex_artist="Stereolab"
        ),
    ]
    assert tune._non_latin_fraction(pairs) == 0.5
