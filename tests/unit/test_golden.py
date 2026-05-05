"""Tests for the golden-truth comparison logic."""

from __future__ import annotations

from datetime import UTC, datetime

from core.golden import (
    AccuracyReport,
    GoldenTruth,
    QuadrantTruth,
    RowTruth,
    compare,
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
