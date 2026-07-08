"""Real-bundle merge test.

Loads the smallest of the 5 anomaly-bucket seed bundles
(`1990-04apr0106-page28.bundle.json`), constructs three hand-built
submissions that introduce a mix of agreement and disagreement, and
asserts the merge produces the expected canonical + agreement outputs
against the real bundle shape (46 flat rows across 4 quadrants).

The point of this test is to exercise round-tripping through real
bundle-derived row counts and pair-concordance denominators, not to
re-cover the per-scenario decision matrix (that's in the unit test).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from core.calibration_consensus import merge
from core.schema import (
    CalibrationRowSubmission,
    CalibrationSubmission,
    VerifiedBy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_BUNDLE = REPO_ROOT / ".seed" / "verifier" / "1990-04apr0106-page28.bundle.json"


def _short(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _flat_bundle_rows(bundle_json: dict) -> list[str]:
    """Return the flat row-index → raw_text list from a bundle.

    Rows are ordered by (quadrant.position order, entry.row_index).
    """
    rows: list[str] = []
    for quad in bundle_json.get("quadrants", []):
        for entry in quad.get("entries", []):
            rows.append(entry.get("raw_text", ""))
    return rows


def _submission(
    user_id: str,
    rows: list[CalibrationRowSubmission],
    stem: str,
) -> CalibrationSubmission:
    return CalibrationSubmission(
        schema_version=1,
        stem=stem,
        reviewer=VerifiedBy(
            user_id=user_id,
            username=user_id,
            real_name=None,
            dj_name=None,
            verified_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
        ),
        submitted_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
        rows=rows,
        missing_row_markers=[],
    )


def _row(idx: int, text: str, type_raw: str | None = None) -> CalibrationRowSubmission:
    return CalibrationRowSubmission(
        bundle_row_index=idx,
        edited_text=text,
        type_raw=type_raw,
        notes=None,
        spurious_flag=False,
    )


def test_real_bundle_merge_produces_expected_settled_output() -> None:
    bundle_json = json.loads(SEED_BUNDLE.read_text())
    stem = bundle_json["stem"]
    flat_texts = _flat_bundle_rows(bundle_json)
    n = len(flat_texts)
    assert n == 46, f"expected page28 bundle to have 46 rows, got {n}"

    # Reviewer A: verbatim.
    rows_a = [_row(i, flat_texts[i], type_raw="H") for i in range(n)]
    # Reviewer B: verbatim except row 0 has a punctuation difference (still agrees under normalization).
    rows_b = [_row(i, flat_texts[i], type_raw="H") for i in range(n)]
    if flat_texts[0]:
        rows_b[0] = _row(0, flat_texts[0].replace(",", "."), type_raw="H")
    # Reviewer C: agrees with A/B on all but row 5 (a real dissent).
    rows_c = [_row(i, flat_texts[i], type_raw="H") for i in range(n)]
    if len(flat_texts) > 5:
        rows_c[5] = _row(5, "COMPLETELY DIFFERENT - READING", type_raw="H")

    submissions = [
        _submission("sub-a", rows_a, stem),
        _submission("sub-b", rows_b, stem),
        _submission("sub-c", rows_c, stem),
    ]

    canonical, agreement, target = merge(
        stem=stem,
        year="1990",
        bucket="anomaly",
        bundle_row_count=n,
        submissions=submissions,
        settled_at=datetime(2026, 6, 12, 14, 35, 11, tzinfo=UTC),
    )

    # A row-5 dissent means target = 3.
    assert target == 3
    assert canonical is not None
    assert agreement is not None
    assert canonical.stem == stem
    assert len(canonical.rows) == n

    # Row 0: A vs B differ only by punctuation → agree under normalization.
    assert canonical.rows[0].verification.status == "unanimous"
    # Row 5: C dissents but A + B agree → majority.
    assert canonical.rows[5].verification.status == "majority"
    assert canonical.rows[5].raw_text == flat_texts[5]
    assert any(
        d.value == "COMPLETELY DIFFERENT - READING"
        for d in canonical.rows[5].verification.raw_text_dissents
    )

    # Histogram totals to n.
    assert sum(agreement.row_status_histogram.values()) == n

    # Pair concordance: A-B and A-C have known relationships.
    rates = {(p.a, p.b): p.raw_text_agree_rate for p in agreement.pair_concordance}
    assert rates[(_short("sub-a"), _short("sub-b"))] == 1.0
    # A-C disagree on row 5 only → 45/46.
    assert rates[(_short("sub-a"), _short("sub-c"))] == 45 / 46
