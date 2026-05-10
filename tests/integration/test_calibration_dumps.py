"""Integration test: row-count check on saved calibration dumps.

When `scripts/calibrate_models.py --dump-dir <dir>` was run for a model,
each scored page produced a `<dir>/<model>/<stem>.json` file containing
the predicted `PageResult`. This test pairs every dump file with the
matching `tests/golden/<stem>.truth.json` and asserts that the model
didn't drop truth-listed rows beyond a tolerance.

The test is parametrised over (model_dir, stem) pairs found at runtime;
if no dumps exist (a fresh checkout, or `/tmp` cleared since the last
calibration), the parametrisation set is empty and pytest reports zero
collected — not a failure. This keeps the test useful in CI (where dumps
exist when re-running, after re-running paid calibration) without making
absent dumps a blocker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.golden import GoldenTruth, compare_row_counts
from core.schema import PageResult

DUMP_ROOT = Path("/tmp/modal-dump")
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"

# Tolerance for row-count drops. ±2 absorbs scribe-level disagreement
# (a row the human transcribed but the model legitimately couldn't read).
ROW_COUNT_TOLERANCE = 2


def _discover_pairs() -> list[tuple[str, Path, Path]]:
    """Yield (model_name, dump_file, truth_file) for every dump that has
    a matching truth file. Empty list if `/tmp/modal-dump/` is absent."""
    if not DUMP_ROOT.is_dir():
        return []
    pairs: list[tuple[str, Path, Path]] = []
    for model_dir in sorted(DUMP_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        for dump in sorted(model_dir.glob("*.json")):
            truth = GOLDEN_DIR / f"{dump.stem}.truth.json"
            if not truth.is_file():
                continue
            pairs.append((model_dir.name, dump, truth))
    return pairs


@pytest.mark.calibration_dump
@pytest.mark.parametrize(
    ("model", "dump_path", "truth_path"),
    _discover_pairs(),
    ids=lambda v: v.name if isinstance(v, Path) else str(v),
)
def test_dump_does_not_drop_truth_rows(model: str, dump_path: Path, truth_path: Path) -> None:
    """A saved dump must return at least `truth_count - tolerance` rows
    per quadrant truth specifies. Negative deltas past the tolerance are
    the page-25-style regression we want to catch automatically."""
    actual = PageResult.model_validate_json(dump_path.read_text())
    truth = GoldenTruth.load(truth_path)
    discrepancies = compare_row_counts(actual=actual, truth=truth, tolerance=ROW_COUNT_TOLERANCE)
    if discrepancies:
        formatted = "\n".join(
            f"  {d.position}: predicted={d.predicted_count}, "
            f"truth={d.truth_count}, delta={d.delta:+d}  "
            f"(FAIL, tolerance=±{ROW_COUNT_TOLERANCE})"
            for d in discrepancies
        )
        pytest.fail(f"{model} / {dump_path.stem}\n{formatted}")
