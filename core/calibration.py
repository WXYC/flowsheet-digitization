"""Calibration harness for comparing extraction models on golden pages.

Runs an arbitrary `transcribe(image_path) -> PageResult` callable against
every `<stem>.png` + `<stem>.truth.json` pair in a golden directory, scores
each result with `core.golden.compare`, and rolls per-model outcomes into a
side-by-side summary.

The harness is dependency-free: it doesn't know about Gemini, transformers,
torch, or any specific model. Concrete adapters live with the CLI runner
(`scripts/calibrate_models.py`) and lazy-import their heavy deps so this
module — and its tests — can run on a bare checkout.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.golden import AccuracyReport, GoldenTruth, compare
from core.schema import PageResult

TranscribeFn = Callable[[Path], PageResult]
"""Adapter contract: take a rendered page image, return a parsed PageResult."""

ProgressCallback = Callable[["CalibrationCase", "CalibrationOutcome"], None]
"""Called after each case completes; lets the CLI drive a live progress line."""


@dataclass(frozen=True)
class CalibrationCase:
    """One image + truth pair to score against."""

    stem: str
    image_path: Path
    truth_path: Path


@dataclass
class CalibrationOutcome:
    """The result of running one model on one case.

    Exactly one of `report` and `error` is populated. `error` is a string
    rather than the exception itself so a summary table doesn't have to
    juggle exception types from heterogeneous adapters.

    `actual` carries the raw PageResult the adapter produced on success.
    The default `compare()` only needs the report, but downstream checks
    (e.g. row-count divergence) want the original prediction.
    """

    case: CalibrationCase
    report: AccuracyReport | None
    elapsed_seconds: float
    error: str | None = None
    actual: PageResult | None = None


def discover_cases(golden_dir: Path) -> list[CalibrationCase]:
    """Pair every `*.png` in `golden_dir` with its sibling `*.truth.json`.

    PNGs without a sibling truth file are silently skipped — the golden
    dir doubles as a render cache, and partial transcriptions are added
    incrementally.
    """
    if not golden_dir.is_dir():
        raise FileNotFoundError(f"golden_dir does not exist: {golden_dir}")

    cases: list[CalibrationCase] = []
    for png in sorted(golden_dir.glob("*.png")):
        truth = png.with_suffix("").with_suffix(".truth.json")
        if not truth.is_file():
            continue
        cases.append(CalibrationCase(stem=png.stem, image_path=png, truth_path=truth))
    return cases


def run_calibration(
    cases: list[CalibrationCase],
    transcribe: TranscribeFn,
    *,
    on_complete: ProgressCallback | None = None,
) -> list[CalibrationOutcome]:
    """Run `transcribe` on every case and score against its truth file.

    A failing adapter (any exception) is recorded as `error` on that case's
    outcome and does not abort the rest of the batch. Local models OOM,
    drop weights, or return malformed JSON often enough that one bad page
    must not throw away N-1 successful scorings.
    """
    outcomes: list[CalibrationOutcome] = []
    for case in cases:
        truth = GoldenTruth.load(case.truth_path)
        start = time.monotonic()
        try:
            actual = transcribe(case.image_path)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            outcome = CalibrationOutcome(
                case=case, report=None, elapsed_seconds=elapsed, error=str(exc)
            )
        else:
            elapsed = time.monotonic() - start
            outcome = CalibrationOutcome(
                case=case,
                report=compare(actual=actual, truth=truth),
                elapsed_seconds=elapsed,
                actual=actual,
            )
        outcomes.append(outcome)
        if on_complete is not None:
            on_complete(case, outcome)
    return outcomes


def summarize(outcomes_by_model: dict[str, list[CalibrationOutcome]]) -> str:
    """Render a side-by-side comparison of per-model results.

    Prints one section per model with overall counts and a per-page row
    showing matched/expected, header misses, and elapsed seconds. Designed
    to be eyeballed in a terminal, not parsed.
    """
    if not outcomes_by_model:
        return "No models run."

    lines: list[str] = []
    for model, outcomes in outcomes_by_model.items():
        scored = [o for o in outcomes if o.report is not None]
        errored = [o for o in outcomes if o.error is not None]

        total_matched = sum(o.report.matched_rows for o in scored if o.report)
        total_expected = sum(
            o.report.matched_rows + len(o.report.missing_rows) for o in scored if o.report
        )
        total_header_misses = sum(len(o.report.header_misses) for o in scored if o.report)
        total_elapsed = sum(o.elapsed_seconds for o in outcomes)
        passed = sum(1 for o in scored if o.report and o.report.passed)

        lines.append("=" * 78)
        lines.append(f"model: {model}")
        lines.append("-" * 78)
        lines.append(
            f"  pages:        {len(outcomes)}  (scored: {len(scored)}, errored: {len(errored)})"
        )
        lines.append(f"  passed:       {passed}/{len(scored)}")
        lines.append(f"  matched_rows: {total_matched}/{total_expected}")
        lines.append(f"  header miss:  {total_header_misses}")
        lines.append(f"  elapsed:      {total_elapsed:.1f}s total")
        lines.append("")
        lines.append(f"  {'page':<40}  {'rows':>8}  {'hdr':>3}  {'time':>7}  status")
        for o in outcomes:
            stem = o.case.stem
            if o.error is not None:
                lines.append(
                    f"  {stem:<40}  {'-':>8}  {'-':>3}  {o.elapsed_seconds:>6.1f}s  "
                    f"ERROR: {o.error[:30]}"
                )
                continue
            assert o.report is not None
            expected = o.report.matched_rows + len(o.report.missing_rows)
            rows = f"{o.report.matched_rows}/{expected}"
            status = "PASS" if o.report.passed else "FAIL"
            lines.append(
                f"  {stem:<40}  {rows:>8}  "
                f"{len(o.report.header_misses):>3}  "
                f"{o.elapsed_seconds:>6.1f}s  {status}"
            )
        lines.append("")
    return "\n".join(lines)
