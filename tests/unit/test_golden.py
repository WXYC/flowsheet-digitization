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
