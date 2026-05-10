"""Hand-transcribed truth files and accuracy reports.

A golden truth file does NOT have to be exhaustive. It's a *subset* check:
"the model output must contain (at least) these things." Extra rows in the
model output are fine — handwriting is hard, and a partial truth is still
useful as regression bait.

Truth files live alongside their image at
`tests/golden/<stem>.png` + `tests/golden/<stem>.truth.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from core.continuations import merge_continuations
from core.schema import PageResult, QuadrantPosition


class RowTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_substring: str
    """A case-insensitive substring that must appear in some entry's raw_text."""


class QuadrantTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: QuadrantPosition
    hour_raw: str | None = None
    jock_substring: str | None = None
    rows: list[RowTruth] = []


class GoldenTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_date_substrings: list[str] = []
    quadrants: list[QuadrantTruth] = []

    @classmethod
    def load(cls, path: Path) -> GoldenTruth:
        return cls.model_validate_json(path.read_text())


@dataclass(slots=True)
class AccuracyReport:
    matched_rows: int
    missing_rows: list[tuple[str, str]] = field(default_factory=list)
    """List of (quadrant_position, expected_substring) for rows that were not found."""
    header_misses: list[str] = field(default_factory=list)
    """List of human-readable descriptions of header-level misses."""

    @property
    def passed(self) -> bool:
        return not self.missing_rows and not self.header_misses

    def summary(self) -> str:
        lines = [f"matched_rows: {self.matched_rows}"]
        if self.header_misses:
            lines.append("header_misses:")
            lines.extend(f"  - {m}" for m in self.header_misses)
        if self.missing_rows:
            lines.append("missing_rows:")
            lines.extend(f"  - [{pos}] {sub}" for pos, sub in self.missing_rows)
        if self.passed:
            lines.append("PASS")
        else:
            lines.append("FAIL")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RowCountDiscrepancy:
    """One quadrant where the model predicted fewer rows than truth.

    `delta = predicted_count - truth_count`. The check is asymmetric:
    truth files are subsets of actual rows (see `GoldenTruth`'s docstring),
    so a positive delta is normal — the model captured rows the human
    didn't transcribe. A NEGATIVE delta past the tolerance is the
    regression-grade signal: the model dropped rows the human DID
    transcribe.
    """

    position: QuadrantPosition
    predicted_count: int
    truth_count: int

    @property
    def delta(self) -> int:
        return self.predicted_count - self.truth_count


def compare_row_counts(
    *,
    actual: PageResult,
    truth: GoldenTruth,
    tolerance: int = 2,
) -> list[RowCountDiscrepancy]:
    """Flag quadrants where the model dropped truth-listed rows.

    Asymmetric: only quadrants with `predicted_count < truth_count - tolerance`
    are returned. Positive deltas (predicted > truth) are always normal in
    subset mode and never reported. Discrepancies are returned in canonical
    quadrant order (the order truth lists them).

    Quadrants that truth omits entirely are not checked — same convention
    `compare()` uses; truth says nothing, we say nothing.
    """
    actual_by_position = {q.position: q for q in actual.quadrants}
    discrepancies: list[RowCountDiscrepancy] = []
    for qt in truth.quadrants:
        truth_count = len(qt.rows)
        actual_q = actual_by_position.get(qt.position)
        predicted_count = len(actual_q.entries) if actual_q is not None else 0
        if predicted_count < truth_count - tolerance:
            discrepancies.append(
                RowCountDiscrepancy(
                    position=qt.position,
                    predicted_count=predicted_count,
                    truth_count=truth_count,
                )
            )
    return discrepancies


def _icontains(haystack: str | None, needle: str) -> bool:
    if haystack is None:
        return False
    return needle.casefold() in haystack.casefold()


def compare(*, actual: PageResult, truth: GoldenTruth) -> AccuracyReport:
    """Compare a model-produced PageResult against a hand-written truth.

    The semantics are:
      * For each substring in `page_date_substrings`, the actual page_date_raw
        must contain it (case-insensitive).
      * For each `QuadrantTruth`, find the actual quadrant by `position`, then
        check `hour_raw` (exact, case-insensitive) and `jock_substring`
        (case-insensitive substring), then for each `RowTruth` confirm that
        SOME entry in the actual quadrant has `raw_substring` in its raw_text.

    Row matching runs against the *continuation-merged* view of each
    quadrant — see `core.continuations`. A truth substring spanning a
    handwritten line that the model emitted as a primary row plus one
    or more `notes="continuation"` rows still matches.
    """
    header_misses: list[str] = []
    missing_rows: list[tuple[str, str]] = []
    matched_rows = 0

    for sub in truth.page_date_substrings:
        if not _icontains(actual.page_date_raw, sub):
            header_misses.append(f"date does not contain {sub!r} (got {actual.page_date_raw!r})")

    actual_by_position = {q.position: q for q in actual.quadrants}

    for qt in truth.quadrants:
        actual_q = actual_by_position.get(qt.position)
        if actual_q is None:
            header_misses.append(f"missing quadrant {qt.position}")
            for row in qt.rows:
                missing_rows.append((qt.position, row.raw_substring))
            continue

        if qt.hour_raw is not None and not _icontains(actual_q.hour_raw, qt.hour_raw):
            header_misses.append(
                f"{qt.position}: hour does not contain {qt.hour_raw!r} (got {actual_q.hour_raw!r})"
            )
        if qt.jock_substring is not None and not _icontains(actual_q.jock_raw, qt.jock_substring):
            header_misses.append(
                f"{qt.position}: jock does not contain {qt.jock_substring!r} "
                f"(got {actual_q.jock_raw!r})"
            )

        merged = merge_continuations(actual_q.entries)
        for row in qt.rows:
            found = any(_icontains(e.raw_text, row.raw_substring) for e in merged)
            if found:
                matched_rows += 1
            else:
                missing_rows.append((qt.position, row.raw_substring))

    return AccuracyReport(
        matched_rows=matched_rows,
        missing_rows=missing_rows,
        header_misses=header_misses,
    )
