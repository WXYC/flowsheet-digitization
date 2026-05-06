"""Tests for core.calibration — the local-model calibration harness.

The harness itself is dependency-free: callers inject a `transcribe`
callable (the "adapter"), so these tests don't need transformers or a
real model. Adapter implementations live alongside the CLI runner and
are tested separately if at all.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from core.calibration import (
    CalibrationCase,
    CalibrationOutcome,
    discover_cases,
    run_calibration,
    summarize,
)
from core.golden import GoldenTruth
from core.schema import PageResult, Quadrant


def _truth_payload() -> dict:
    return {
        "page_date_substrings": ["Jan", "90"],
        "quadrants": [
            {
                "position": "top_left",
                "hour_raw": "6AM",
                "jock_substring": "ALECIA",
                "rows": [
                    {"raw_substring": "LED ZEP"},
                    {"raw_substring": "STONES"},
                ],
            }
        ],
    }


def _make_case(tmp_path: Path, stem: str, *, with_truth: bool = True) -> Path:
    img = tmp_path / f"{stem}.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    if with_truth:
        truth = tmp_path / f"{stem}.truth.json"
        truth.write_text(json.dumps(_truth_payload()))
    return img


def _empty_quadrants() -> list[Quadrant]:
    return [
        Quadrant(position="top_left", entries=[]),
        Quadrant(position="top_right", entries=[]),
        Quadrant(position="bottom_left", entries=[]),
        Quadrant(position="bottom_right", entries=[]),
    ]


def _passing_page() -> PageResult:
    quads = _empty_quadrants()
    quads[0] = Quadrant(
        position="top_left",
        hour_raw="6AM",
        jock_raw="ALECIA TWO",
        entries=[
            {  # type: ignore[list-item]
                "row_index": 0,
                "raw_text": "LED ZEP - WHOLE LOTTA LOVE",
                "confidence": "high",
            },
            {  # type: ignore[list-item]
                "row_index": 1,
                "raw_text": "STONES - GIMME SHELTER",
                "confidence": "high",
            },
        ],
    )
    return PageResult(
        page_date_raw="Mon Jan 1 '90",
        quadrants=quads,
        model_version="test",
        extracted_at=datetime(2026, 5, 6),
    )


def _failing_page() -> PageResult:
    return PageResult(
        page_date_raw=None,
        quadrants=_empty_quadrants(),
        model_version="test",
        extracted_at=datetime(2026, 5, 6),
    )


# -- discover_cases --------------------------------------------------------


def test_discover_cases_pairs_png_with_truth(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    _make_case(tmp_path, "p2")
    cases = discover_cases(tmp_path)
    assert [c.stem for c in cases] == ["p1", "p2"]
    for c in cases:
        assert c.image_path.exists()
        assert c.truth_path.exists()


def test_discover_cases_skips_png_without_truth(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    _make_case(tmp_path, "p2", with_truth=False)
    cases = discover_cases(tmp_path)
    assert [c.stem for c in cases] == ["p1"]


def test_discover_cases_returns_empty_for_empty_dir(tmp_path: Path) -> None:
    assert discover_cases(tmp_path) == []


def test_discover_cases_raises_for_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_cases(tmp_path / "does-not-exist")


def test_discover_cases_sorts_by_stem(tmp_path: Path) -> None:
    _make_case(tmp_path, "z-page03")
    _make_case(tmp_path, "a-page01")
    _make_case(tmp_path, "m-page02")
    assert [c.stem for c in discover_cases(tmp_path)] == [
        "a-page01",
        "m-page02",
        "z-page03",
    ]


# -- run_calibration -------------------------------------------------------


def test_run_calibration_passes_image_path_to_adapter(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    seen: list[Path] = []

    def adapter(image_path: Path) -> PageResult:
        seen.append(image_path)
        return _passing_page()

    run_calibration(cases, adapter)
    assert seen == [cases[0].image_path]


def test_run_calibration_records_passing_outcome(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    outcomes = run_calibration(cases, lambda _p: _passing_page())
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.error is None
    assert o.report is not None
    assert o.report.passed
    assert o.report.matched_rows == 2
    assert o.elapsed_seconds >= 0


def test_run_calibration_records_missing_rows(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    outcomes = run_calibration(cases, lambda _p: _failing_page())
    o = outcomes[0]
    assert o.error is None
    assert o.report is not None
    assert not o.report.passed
    # Both expected rows missing + header miss for the date and the missing hour/jock.
    assert len(o.report.missing_rows) == 2


def test_run_calibration_captures_adapter_exception(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)

    def boom(_p: Path) -> PageResult:
        raise RuntimeError("model OOM'd")

    outcomes = run_calibration(cases, boom)
    o = outcomes[0]
    assert o.report is None
    assert o.error is not None
    assert "OOM" in o.error


def test_run_calibration_one_failure_does_not_abort_batch(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    _make_case(tmp_path, "p2")
    cases = discover_cases(tmp_path)
    calls: list[Path] = []

    def flaky(image_path: Path) -> PageResult:
        calls.append(image_path)
        if image_path.stem == "p1":
            raise RuntimeError("transient")
        return _passing_page()

    outcomes = run_calibration(cases, flaky)
    assert calls == [cases[0].image_path, cases[1].image_path]
    assert outcomes[0].error is not None
    assert outcomes[0].report is None
    assert outcomes[1].error is None
    assert outcomes[1].report is not None


def test_run_calibration_invokes_progress_callback(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    _make_case(tmp_path, "p2")
    cases = discover_cases(tmp_path)
    progress: list[tuple[str, bool]] = []

    def on_done(case: CalibrationCase, outcome: CalibrationOutcome) -> None:
        progress.append((case.stem, outcome.error is None))

    run_calibration(cases, lambda _p: _passing_page(), on_complete=on_done)
    assert progress == [("p1", True), ("p2", True)]


# -- summarize -------------------------------------------------------------


def test_summarize_with_one_model(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    outcomes = run_calibration(cases, lambda _p: _passing_page())

    text = summarize({"churro-3B": outcomes})
    assert "churro-3B" in text
    assert "matched_rows" in text
    assert "p1" in text


def test_summarize_compares_two_models(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    pass_outcomes = run_calibration(cases, lambda _p: _passing_page())
    fail_outcomes = run_calibration(cases, lambda _p: _failing_page())

    text = summarize({"good-model": pass_outcomes, "bad-model": fail_outcomes})
    assert "good-model" in text
    assert "bad-model" in text


def test_summarize_includes_errors_in_output(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)

    def boom(_p: Path) -> PageResult:
        raise RuntimeError("nope")

    outcomes = run_calibration(cases, boom)
    text = summarize({"broken": outcomes})
    assert "ERROR" in text or "error" in text


# -- CalibrationCase wiring ------------------------------------------------


def test_calibration_case_loads_truth(tmp_path: Path) -> None:
    _make_case(tmp_path, "p1")
    cases = discover_cases(tmp_path)
    truth = GoldenTruth.load(cases[0].truth_path)
    assert truth.page_date_substrings == ["Jan", "90"]
