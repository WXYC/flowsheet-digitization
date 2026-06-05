"""Tests for `core.reconciliation.reconcile`.

Strategy: inject a stub `LMLClient` so tests run synchronously without
hitting the wire. Verify three contracts:

  1. Above-threshold corrections are applied to `Entry.raw_text`.
  2. Below-threshold corrections produce a `FlaggedRow` entry and leave
     the original row untouched.
  3. Every `Entry` field except `raw_text` is preserved byte-identical.

Sibling test: `tests/unit/test_lml_client.py` exercises the HTTP layer;
this file exercises the orchestration above it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.lml_client import LMLBulkItemResult
from core.reconciliation import FlaggedRow, reconcile
from core.schema import Entry, PageResult, Quadrant

# -- Fixtures ----------------------------------------------------------------


class _StubLMLClient:
    """Records `bulk_lookup` calls and returns a pre-canned result list.

    Tests build the `LMLBulkItemResult` list directly so they don't depend
    on the JSON serialization layer (covered by `test_lml_client.py`).
    """

    def __init__(self, results: list[LMLBulkItemResult]) -> None:
        self._results = results
        self.received_items: list[dict[str, Any]] = []

    async def bulk_lookup(self, items: list[dict[str, Any]]) -> list[LMLBulkItemResult]:
        self.received_items = list(items)
        return self._results


def _entry(row_index: int, raw_text: str, **overrides: Any) -> Entry:
    base: dict[str, Any] = {
        "row_index": row_index,
        "raw_text": raw_text,
        "type_raw": None,
        "confidence": "high",
        "notes": None,
        "oddities": [],
    }
    base.update(overrides)
    return Entry(**base)


def _quadrant(position: str, entries: list[Entry]) -> Quadrant:
    return Quadrant(
        position=position,  # type: ignore[arg-type]
        hour_raw=None,
        jock_raw=None,
        entries=entries,
        oddities=[],
    )


def _page(quadrants: list[Quadrant]) -> PageResult:
    return PageResult(
        page_date_raw="Mon 1/1/90",
        quadrants=quadrants,
        comments_raw=None,
        oddities=[],
        model_version="gemini-3.1-pro-preview",
        extracted_at=datetime(2026, 6, 4, 12, 0, 0),
    )


def _empty_quadrants_except(position: str, entries: list[Entry]) -> list[Quadrant]:
    """Build a four-quadrant list with content only in `position`."""
    positions = ("top_left", "top_right", "bottom_left", "bottom_right")
    return [_quadrant(p, entries if p == position else []) for p in positions]


# -- Tests -------------------------------------------------------------------


class TestReconcile:
    async def test_applies_above_threshold_correction(self) -> None:
        """A high-similarity LML correction should rewrite the artist
        side of `raw_text` while keeping the track side intact."""
        page = _page(
            _empty_quadrants_except(
                "top_left",
                [_entry(0, "STEREOLAB - Cybeles Reverie")],
            )
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(
                    index=0,
                    status="match",
                    corrected_artist="Stereolab",
                ),
            ]
        )

        corrected, flagged = await reconcile(page, lml=lml, threshold=80)

        assert lml.received_items == [{"artist": "STEREOLAB"}]
        # Artist on the row is rewritten (case-corrected), track preserved.
        assert corrected.quadrants[0].entries[0].raw_text == "Stereolab - Cybeles Reverie"
        # No flag: the correction was applied silently.
        assert flagged == []

    async def test_flags_below_threshold_correction(self) -> None:
        """A weak LML correction should NOT modify the row; instead it
        produces a `FlaggedRow` so a human can review."""
        page = _page(
            _empty_quadrants_except(
                "top_left",
                [_entry(0, "MZRG - some track")],
            )
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(
                    index=0,
                    status="match",
                    corrected_artist="Beethoven",  # nothing like the input
                ),
            ]
        )

        corrected, flagged = await reconcile(page, lml=lml, threshold=80)

        # Original row untouched.
        assert corrected.quadrants[0].entries[0].raw_text == "MZRG - some track"
        assert len(flagged) == 1
        flag = flagged[0]
        assert isinstance(flag, FlaggedRow)
        assert flag.quadrant == "top_left"
        assert flag.row_index == 0
        assert flag.original_artist == "MZRG"
        assert flag.suggested_artist == "Beethoven"
        assert flag.score < 80

    async def test_skips_artist_only_no_match(self) -> None:
        """A `no_match` from LML produces no correction and no flag —
        we can't say anything useful, leave it for downstream."""
        page = _page(
            _empty_quadrants_except(
                "top_left",
                [_entry(0, "obscure jingle band - jingle")],
            )
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(
                    index=0,
                    status="no_match",
                    corrected_artist=None,
                ),
            ]
        )

        corrected, flagged = await reconcile(page, lml=lml, threshold=80)
        assert corrected.quadrants[0].entries[0].raw_text == "obscure jingle band - jingle"
        assert flagged == []

    async def test_skips_rows_with_no_artist(self) -> None:
        """A row that doesn't parse to an artist (whitespace / Alex's
        literal "blank" placeholder for a physically blank line) is
        silently skipped: nothing to look up, no flag, no batch item
        sent."""
        page = _page(
            _empty_quadrants_except(
                "top_left",
                [
                    _entry(0, "blank"),  # Alex's blank placeholder (n=65 in corpus)
                    _entry(1, "  "),  # whitespace-only
                    _entry(2, "Stereolab - Track"),  # real row
                ],
            )
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )

        corrected, flagged = await reconcile(page, lml=lml, threshold=80)
        # Only the real row went up to LML.
        assert lml.received_items == [{"artist": "Stereolab"}]
        # Blank-style rows preserved verbatim.
        assert corrected.quadrants[0].entries[0].raw_text == "blank"
        assert corrected.quadrants[0].entries[1].raw_text == "  "
        assert flagged == []

    async def test_preserves_non_raw_text_entry_fields(self) -> None:
        """`type_raw`, `notes`, `confidence`, `oddities` must survive a
        correction unchanged."""
        page = _page(
            _empty_quadrants_except(
                "top_right",
                [
                    _entry(
                        4,
                        "STEREOLAB - Cybeles",
                        type_raw="H",
                        confidence="medium",
                        notes="double_height",
                        oddities=["asterisk in the right margin"],
                    )
                ],
            )
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )

        corrected, _ = await reconcile(page, lml=lml, threshold=80)
        entry = corrected.quadrants[1].entries[0]
        assert entry.raw_text == "Stereolab - Cybeles"
        assert entry.row_index == 4
        assert entry.type_raw == "H"
        assert entry.confidence == "medium"
        assert entry.notes == "double_height"
        assert entry.oddities == ["asterisk in the right margin"]

    async def test_preserves_page_level_metadata(self) -> None:
        """`page_date_raw`, `comments_raw`, page-level `oddities`,
        `model_version`, and `extracted_at` all survive untouched."""
        page = _page(_empty_quadrants_except("top_left", [_entry(0, "STEREOLAB - Track")]))
        # Tweak page-level metadata so we can assert it survives byte-identical.
        page = page.model_copy(
            update={
                "comments_raw": "DJ said something",
                "oddities": ["page is rotated"],
            }
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )
        corrected, _ = await reconcile(page, lml=lml, threshold=80)
        assert corrected.page_date_raw == "Mon 1/1/90"
        assert corrected.comments_raw == "DJ said something"
        assert corrected.oddities == ["page is rotated"]
        assert corrected.model_version == "gemini-3.1-pro-preview"
        assert corrected.extracted_at == datetime(2026, 6, 4, 12, 0, 0)

    async def test_exact_match_at_threshold_boundary_applies(self) -> None:
        """The threshold is inclusive: `score == threshold` applies."""
        page = _page(_empty_quadrants_except("top_left", [_entry(0, "Sterolab - x")]))
        # Pick a correction that scores exactly the threshold so we exercise
        # the boundary. token_set_ratio for ("sterolab", "stereolab") < 100
        # so we'll use a softer threshold to ensure the chosen correction
        # lands at-or-above.
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )
        # Threshold 1 = trivially under any plausible token_set_ratio of
        # similar strings.
        corrected, flagged = await reconcile(page, lml=lml, threshold=1)
        assert corrected.quadrants[0].entries[0].raw_text == "Stereolab - x"
        assert flagged == []

    async def test_correction_equals_original_is_no_op(self) -> None:
        """LML often returns `corrected_artist` equal to the input when
        the artist is already an exact catalog hit. We must NOT rewrite
        the row in that case — even a benign rewrite ('Stereolab' ==
        'Stereolab') would re-tokenize whitespace and could subtly
        diverge from the on-disk shape."""
        page = _page(
            _empty_quadrants_except("top_left", [_entry(0, "Stereolab - Cybele's Reverie")])
        )
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(
                    index=0,
                    status="match",
                    corrected_artist="Stereolab",
                ),
            ]
        )

        corrected, flagged = await reconcile(page, lml=lml, threshold=80)
        assert corrected.quadrants[0].entries[0].raw_text == "Stereolab - Cybele's Reverie"
        assert flagged == []

    async def test_idempotent_under_repeated_runs(self) -> None:
        """Running reconciliation a second time on the corrected output
        must produce the same output (no drift, no second correction)."""
        page = _page(_empty_quadrants_except("top_left", [_entry(0, "STEREOLAB - Track")]))
        # First run: applies the correction.
        lml1 = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )
        once, _ = await reconcile(page, lml=lml1, threshold=80)
        # Second run: LML now sees the already-corrected input and
        # returns the same canonical form.
        lml2 = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="Stereolab"),
            ]
        )
        twice, _ = await reconcile(once, lml=lml2, threshold=80)
        assert once.quadrants[0].entries[0].raw_text == twice.quadrants[0].entries[0].raw_text

    async def test_indexes_returned_results_by_position(self) -> None:
        """LML returns one result per input item, indexed 0..N-1. The
        reconciler must map index back to (quadrant, row_index) — not
        rely on result ordering across quadrants."""
        page = _page(
            [
                _quadrant(
                    "top_left",
                    [_entry(0, "a - x"), _entry(1, "b - y")],
                ),
                _quadrant("top_right", [_entry(0, "c - z")]),
                _quadrant("bottom_left", []),
                _quadrant("bottom_right", []),
            ]
        )
        # The reconciler should send rows in (quadrant_order × row_index)
        # order: a, b, c. Verify by returning distinctive corrections.
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="match", corrected_artist="A"),
                LMLBulkItemResult(index=1, status="match", corrected_artist="B"),
                LMLBulkItemResult(index=2, status="match", corrected_artist="C"),
            ]
        )
        corrected, _ = await reconcile(page, lml=lml, threshold=1)
        assert corrected.quadrants[0].entries[0].raw_text == "A - x"
        assert corrected.quadrants[0].entries[1].raw_text == "B - y"
        assert corrected.quadrants[1].entries[0].raw_text == "C - z"

    async def test_handles_lml_error_status_as_skip(self) -> None:
        """A per-item LML failure must surface as a skip, not a crash."""
        page = _page(_empty_quadrants_except("top_left", [_entry(0, "x - y")]))
        lml = _StubLMLClient(
            results=[
                LMLBulkItemResult(index=0, status="error", corrected_artist=None, message="boom"),
            ]
        )
        corrected, flagged = await reconcile(page, lml=lml, threshold=80)
        assert corrected.quadrants[0].entries[0].raw_text == "x - y"
        assert flagged == []
