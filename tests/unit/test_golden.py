"""Tests for the golden-truth comparison logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.golden import (
    AccuracyReport,
    GoldenTruth,
    QuadrantTruth,
    RowCountDiscrepancy,
    RowTruth,
    compare,
    compare_row_counts,
)
from core.schema import Entry, PageResult, Quadrant


def _page(*, date: str = "Monday 1 Jan '90", tl_entries: list[Entry] | None = None) -> PageResult:
    return PageResult(
        page_date_raw=date,
        quadrants=[
            Quadrant(
                position="top_left",
                hour_raw="6AM",
                jock_raw="ALECIA",
                entries=tl_entries or [],
            ),
            Quadrant(position="top_right", hour_raw="10AM", jock_raw="Brian L.", entries=[]),
            Quadrant(position="bottom_left", hour_raw="7PM", jock_raw="HOLLAND", entries=[]),
            Quadrant(position="bottom_right", hour_raw=None, jock_raw="Brian L.", entries=[]),
        ],
        model_version="m",
        extracted_at=datetime.now(UTC),
    )


class TestCompare:
    def test_perfect_match_yields_zero_misses(self) -> None:
        actual = _page(
            tl_entries=[Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high")]
        )
        truth = GoldenTruth(
            page_date_substrings=["Monday", "Jan", "90"],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="ALECIA",
                    rows=[RowTruth(raw_substring="LED ZEP - TRAMPLED")],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert report.matched_rows == 1
        assert report.missing_rows == []
        assert report.header_misses == []
        assert report.passed

    def test_missing_row_is_reported(self) -> None:
        actual = _page(tl_entries=[])  # no entries at all
        truth = GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="ALECIA",
                    rows=[RowTruth(raw_substring="LED ZEP - TRAMPLED")],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert report.matched_rows == 0
        assert report.missing_rows == [("top_left", "LED ZEP - TRAMPLED")]
        assert not report.passed

    def test_substring_match_is_case_insensitive(self) -> None:
        actual = _page(
            tl_entries=[Entry(row_index=0, raw_text="led zep - trampled", confidence="medium")]
        )
        truth = GoldenTruth(
            page_date_substrings=["MONDAY"],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="alecia",
                    rows=[RowTruth(raw_substring="LED ZEP")],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert report.passed

    def test_wrong_hour_is_a_header_miss(self) -> None:
        actual = _page()  # tl hour is "6AM"
        truth = GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="9AM",  # mismatch
                    jock_substring="ALECIA",
                    rows=[],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert any("hour" in m for m in report.header_misses)
        assert not report.passed

    def test_wrong_jock_is_a_header_miss(self) -> None:
        actual = _page()
        truth = GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="MARGE",  # mismatch
                    rows=[],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert any("jock" in m for m in report.header_misses)
        assert not report.passed

    def test_missing_date_substring_is_a_header_miss(self) -> None:
        actual = _page(date="Tuesday 2 Jan '90")
        truth = GoldenTruth(page_date_substrings=["Monday"], quadrants=[])
        report = compare(actual=actual, truth=truth)
        assert any("date" in m for m in report.header_misses)
        assert not report.passed

    def test_truth_substring_matches_across_continuation_rows(self) -> None:
        """A handwritten line that wraps onto the next grid row gets
        emitted by the model as two `Entry`s — one normal, one tagged
        `notes="continuation"`. The truth substring is the *whole*
        line. Without `merge_continuations`, neither raw_text alone
        contains the full substring and the row would be reported as
        missing. With it, the merged view contains the joined text and
        the match succeeds — the load-bearing assertion of this PR."""
        actual = _page(
            tl_entries=[
                Entry(
                    row_index=0,
                    raw_text="DUKE ELLINGTON & JOHN COLTRANE - IN A",
                    confidence="high",
                ),
                Entry(
                    row_index=1,
                    raw_text="SENTIMENTAL MOOD",
                    confidence="high",
                    notes="continuation",
                ),
            ]
        )
        truth = GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="ALECIA",
                    rows=[RowTruth(raw_substring="IN A SENTIMENTAL MOOD")],
                )
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert report.matched_rows == 1
        assert report.missing_rows == []

    def test_extra_actual_rows_are_allowed(self) -> None:
        # Subset semantics: model can transcribe more than truth specifies.
        actual = _page(
            tl_entries=[
                Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high"),
                Entry(row_index=1, raw_text="STONES - LITTLE RED", confidence="high"),
                Entry(row_index=2, raw_text="EXTRA - SOMETHING", confidence="high"),
            ]
        )
        truth = GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position="top_left",
                    hour_raw="6AM",
                    jock_substring="ALECIA",
                    rows=[RowTruth(raw_substring="LED ZEP")],
                ),
            ],
        )
        report = compare(actual=actual, truth=truth)
        assert report.passed


class TestCompareRowCounts:
    """Asymmetric row-count check: only NEGATIVE deltas (predicted < truth)
    are flagged as discrepancies, because truth is a subset (positive
    deltas are normal). The threshold is `delta < -tolerance`."""

    def _page_with_quadrant_entry_counts(
        self,
        counts: dict[str, int],
    ) -> PageResult:
        quads = []
        for pos in ("top_left", "top_right", "bottom_left", "bottom_right"):
            quads.append(
                Quadrant(
                    position=pos,  # type: ignore[arg-type]
                    hour_raw=None,
                    jock_raw=None,
                    entries=[
                        Entry(row_index=i, raw_text=f"row {i}", confidence="high")
                        for i in range(counts.get(pos, 0))
                    ],
                )
            )
        return PageResult(
            page_date_raw=None,
            quadrants=quads,
            model_version="m",
            extracted_at=datetime.now(UTC),
        )

    def _truth_with_row_counts(self, counts: dict[str, int]) -> GoldenTruth:
        return GoldenTruth(
            page_date_substrings=[],
            quadrants=[
                QuadrantTruth(
                    position=pos,  # type: ignore[arg-type]
                    rows=[RowTruth(raw_substring=f"r{i}") for i in range(n)],
                )
                for pos, n in counts.items()
            ],
        )

    def test_predicted_matches_truth_yields_no_discrepancies(self) -> None:
        actual = self._page_with_quadrant_entry_counts({"top_left": 6})
        truth = self._truth_with_row_counts({"top_left": 6})
        assert compare_row_counts(actual=actual, truth=truth) == []

    def test_predicted_far_below_truth_is_a_discrepancy(self) -> None:
        """Page 25's regression: top_right predicted=3 vs truth=9, delta=-6."""
        actual = self._page_with_quadrant_entry_counts({"top_right": 3})
        truth = self._truth_with_row_counts({"top_right": 9})
        diffs = compare_row_counts(actual=actual, truth=truth, tolerance=2)
        assert len(diffs) == 1
        assert diffs[0].position == "top_right"
        assert diffs[0].predicted_count == 3
        assert diffs[0].truth_count == 9
        assert diffs[0].delta == -6

    def test_predicted_above_truth_is_not_a_discrepancy(self) -> None:
        """Subset semantics: predicted=20 vs truth=6 is fine — truth is a
        subset of actual rows. Positive delta must NOT trigger."""
        actual = self._page_with_quadrant_entry_counts({"top_left": 20})
        truth = self._truth_with_row_counts({"top_left": 6})
        assert compare_row_counts(actual=actual, truth=truth, tolerance=2) == []

    def test_within_tolerance_is_not_a_discrepancy(self) -> None:
        """delta=-2 with tolerance=2 is on the boundary; tolerance accommodates
        a single-row scribe disagreement so this must NOT fail."""
        actual = self._page_with_quadrant_entry_counts({"top_left": 4})
        truth = self._truth_with_row_counts({"top_left": 6})
        assert compare_row_counts(actual=actual, truth=truth, tolerance=2) == []

    def test_delta_just_outside_tolerance_is_a_discrepancy(self) -> None:
        """delta=-3 with tolerance=2 should fail (one past the boundary)."""
        actual = self._page_with_quadrant_entry_counts({"top_left": 3})
        truth = self._truth_with_row_counts({"top_left": 6})
        diffs = compare_row_counts(actual=actual, truth=truth, tolerance=2)
        assert len(diffs) == 1
        assert diffs[0].delta == -3

    def test_predicted_zero_against_truth_count_is_a_discrepancy(self) -> None:
        """Quadrant present but empty (e.g., model returned no entries) vs
        truth specifying rows yields delta=-truth_count — a discrepancy."""
        actual = self._page_with_quadrant_entry_counts({"top_left": 0})
        truth = self._truth_with_row_counts({"top_left": 5})
        diffs = compare_row_counts(actual=actual, truth=truth, tolerance=2)
        assert len(diffs) == 1
        assert diffs[0].position == "top_left"
        assert diffs[0].predicted_count == 0
        assert diffs[0].truth_count == 5

    def test_quadrants_truth_omits_are_skipped(self) -> None:
        """If truth doesn't have a quadrant entry for `bottom_right`,
        we don't check it — it's not a 'truth says zero' case, it's
        a 'truth has nothing to say'. Substring scoring already does the
        same thing."""
        actual = self._page_with_quadrant_entry_counts({"top_left": 0, "bottom_right": 12})
        truth = self._truth_with_row_counts({"top_left": 0})  # no bottom_right
        assert compare_row_counts(actual=actual, truth=truth, tolerance=2) == []

    def test_multiple_discrepancies_returned_in_canonical_order(self) -> None:
        actual = self._page_with_quadrant_entry_counts(
            {"top_left": 1, "top_right": 1, "bottom_left": 1, "bottom_right": 1}
        )
        truth = self._truth_with_row_counts(
            {"top_left": 5, "top_right": 5, "bottom_left": 5, "bottom_right": 5}
        )
        diffs = compare_row_counts(actual=actual, truth=truth, tolerance=2)
        assert [d.position for d in diffs] == [
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]

    def test_discrepancy_is_frozen(self) -> None:
        d = RowCountDiscrepancy(position="top_left", predicted_count=3, truth_count=9)
        # dataclass(frozen=True) raises FrozenInstanceError, an AttributeError subclass.
        with pytest.raises(AttributeError):
            d.predicted_count = 0  # type: ignore[misc]


def test_truth_round_trips_through_json() -> None:
    truth = GoldenTruth(
        page_date_substrings=["Monday"],
        quadrants=[
            QuadrantTruth(
                position="top_left",
                hour_raw="6AM",
                jock_substring="ALECIA",
                rows=[RowTruth(raw_substring="LED ZEP")],
            )
        ],
    )
    assert GoldenTruth.model_validate_json(truth.model_dump_json()) == truth


def test_accuracy_report_passed_requires_no_misses() -> None:
    r = AccuracyReport(
        matched_rows=2,
        missing_rows=[],
        header_misses=[],
    )
    assert r.passed
    r2 = AccuracyReport(matched_rows=2, missing_rows=[("top_left", "x")], header_misses=[])
    assert not r2.passed


# -- discover_truths --------------------------------------------------------


def _minimal_truth_json() -> str:
    """A GoldenTruth JSON with the smallest valid shape."""
    return GoldenTruth(page_date_substrings=[], quadrants=[]).model_dump_json()


def test_discover_truths_finds_flat_layout(tmp_path) -> None:
    from core.golden import discover_truths

    (tmp_path / "1990-01jan-page01.truth.json").write_text(_minimal_truth_json())
    (tmp_path / "1990-01jan-page02.truth.json").write_text(_minimal_truth_json())
    found = discover_truths(tmp_path)
    assert [p.name for p in found] == [
        "1990-01jan-page01.truth.json",
        "1990-01jan-page02.truth.json",
    ]


def test_discover_truths_finds_nested_calibration_layout(tmp_path) -> None:
    """Calibration-derived truths live under
    `tests/golden/calibration/<year>/<stem>.truth.json`. rglob picks them
    up alongside flat files without any config change."""
    from core.golden import discover_truths

    (tmp_path / "1990-flat-page01.truth.json").write_text(_minimal_truth_json())
    nested = tmp_path / "calibration" / "1990"
    nested.mkdir(parents=True)
    (nested / "1990-nested-page02.truth.json").write_text(_minimal_truth_json())
    found = discover_truths(tmp_path)
    names = [p.name for p in found]
    assert "1990-flat-page01.truth.json" in names
    assert "1990-nested-page02.truth.json" in names
    # Loadable via GoldenTruth.load.
    for path in found:
        GoldenTruth.load(path)


def test_discover_truths_missing_dir_returns_empty(tmp_path) -> None:
    from core.golden import discover_truths

    assert discover_truths(tmp_path / "does-not-exist") == []
