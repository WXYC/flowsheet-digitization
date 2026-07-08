"""Tests for the pure-function consensus/merge over reviewer submissions.

See plans/multi-reviewer-calibration.md §Merge algorithm.

Each `Test<Scenario>` class covers one row of the auto-illegible /
majority / escalation decision matrix. Test IDs match the scenario names
in the plan's Testing section (unanimous_all_agree, majority_n3, etc.)
so a failing case reads cleanly against the plan.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from core.calibration_consensus import merge
from core.schema import (
    CalibrationCanonical,
    CalibrationRowSubmission,
    CalibrationSubmission,
    MissingRowMarker,
    VerifiedBy,
)

STEM = "1990-04apr0106-page28"
YEAR = "1990"
BUCKET = "anomaly"
SETTLED_AT = datetime(2026, 6, 12, 14, 35, 11, tzinfo=UTC)


def _short(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _reviewer(user_id: str) -> VerifiedBy:
    return VerifiedBy(
        user_id=user_id,
        username=user_id,
        real_name=None,
        dj_name=None,
        verified_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
    )


def _submission(
    user_id: str,
    rows: list[CalibrationRowSubmission],
    missing_row_markers: list[MissingRowMarker] | None = None,
    stem: str = STEM,
) -> CalibrationSubmission:
    return CalibrationSubmission(
        schema_version=1,
        stem=stem,
        reviewer=_reviewer(user_id),
        submitted_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
        rows=rows,
        missing_row_markers=missing_row_markers or [],
    )


def _row(
    idx: int,
    text: str | None = "BEATLES - HELP",
    type_raw: str | None = "H",
    notes: str | None = None,
    spurious: bool = False,
) -> CalibrationRowSubmission:
    return CalibrationRowSubmission(
        bundle_row_index=idx,
        edited_text=text,
        type_raw=type_raw,
        notes=notes,
        spurious_flag=spurious,
    )


def _run(subs: list[CalibrationSubmission], bundle_row_count: int = 1):
    return merge(
        stem=STEM,
        year=YEAR,
        bucket=BUCKET,
        bundle_row_count=bundle_row_count,
        submissions=subs,
        settled_at=SETTLED_AT,
    )


class TestUnanimousAllAgree:
    def test_two_reviewers_all_agree_settles_at_n2(self) -> None:
        subs = [
            _submission("sub-a", [_row(0)]),
            _submission("sub-b", [_row(0)]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 2
        assert canonical is not None
        assert agreement is not None
        assert len(canonical.rows) == 1
        row = canonical.rows[0]
        assert row.raw_text == "BEATLES - HELP"
        assert row.type_raw == "H"
        assert row.verification.status == "unanimous"
        assert row.verification.raw_text_status == "unanimous"
        assert row.verification.type_raw_status == "unanimous"
        assert row.verification.spurious_flag_status == "unanimous_keep"
        assert row.verification.spurious_flag_votes == {"keep": 2, "spurious": 0}
        assert set(row.verification.reviewer_shorts) == {_short("sub-a"), _short("sub-b")}
        assert agreement.target_reviewers == 2
        assert agreement.row_status_histogram == {"unanimous": 1}


class TestMajorityN3:
    def test_two_agree_one_dissents_at_n3(self) -> None:
        subs = [
            _submission("sub-a", [_row(0, text="BEATLES - HELP")]),
            _submission("sub-b", [_row(0, text="BEATLES - HELP")]),
            _submission("sub-c", [_row(0, text="BEATLES - HELO")]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 3
        assert canonical is not None
        row = canonical.rows[0]
        assert row.raw_text == "BEATLES - HELP"
        assert row.verification.raw_text_status == "majority"
        assert row.verification.status == "majority"
        assert len(row.verification.raw_text_dissents) == 1
        assert row.verification.raw_text_dissents[0].reviewer_short == _short("sub-c")
        assert row.verification.raw_text_dissents[0].value == "BEATLES - HELO"
        assert agreement is not None
        assert agreement.row_status_histogram == {"majority": 1}


class TestIllegibleTripleDisagree:
    def test_three_different_readings_flag_illegible(self) -> None:
        subs = [
            _submission("sub-a", [_row(0, text="BEATLES - HELP")]),
            _submission("sub-b", [_row(0, text="BEATLES - HELO")]),
            _submission("sub-c", [_row(0, text="BEATLES - HELF")]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 3
        assert canonical is not None
        row = canonical.rows[0]
        assert row.raw_text == "__illegible__"
        assert row.verification.raw_text_status == "illegible"
        assert row.verification.status == "illegible"
        assert len(row.verification.raw_text_dissents) == 3
        assert agreement is not None
        assert agreement.row_status_histogram == {"illegible": 1}


class TestN2DisagreesBumpsToThree:
    def test_two_disagreeing_reviewers_returns_no_canonical_target_3(self) -> None:
        subs = [
            _submission("sub-a", [_row(0, text="BEATLES - HELP")]),
            _submission("sub-b", [_row(0, text="BEATLES - HELO")]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 3
        assert canonical is None
        assert agreement is None


class TestSpuriousMajority:
    def test_majority_spurious_sets_text_null(self) -> None:
        # Bundle row 0: two reviewers vote spurious, one keeps.
        subs = [
            _submission("sub-a", [_row(0, text=None, type_raw=None, spurious=True)]),
            _submission("sub-b", [_row(0, text=None, type_raw=None, spurious=True)]),
            _submission("sub-c", [_row(0, text="BEATLES - HELP")]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 3
        assert canonical is not None
        row = canonical.rows[0]
        assert row.raw_text is None
        assert row.verification.spurious_flag_status == "majority_spurious"
        assert row.verification.spurious_flag_votes == {"keep": 1, "spurious": 2}
        assert row.verification.status == "majority_spurious"
        # Keeper's text recorded as dissent.
        assert any(
            d.reviewer_short == _short("sub-c") and d.value == "BEATLES - HELP"
            for d in row.verification.raw_text_dissents
        )
        assert agreement is not None
        assert agreement.row_status_histogram == {"majority_spurious": 1}


class TestUnanimousSpurious:
    def test_all_spurious_settles_row_null_no_dissent(self) -> None:
        subs = [
            _submission("sub-a", [_row(0, text=None, type_raw=None, spurious=True)]),
            _submission("sub-b", [_row(0, text=None, type_raw=None, spurious=True)]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 2
        assert canonical is not None
        row = canonical.rows[0]
        assert row.raw_text is None
        assert row.verification.spurious_flag_status == "unanimous_spurious"
        assert row.verification.status == "unanimous_spurious"
        assert row.verification.raw_text_dissents == []


class TestSpuriousSplitEscalates:
    def test_n2_1_1_spurious_split_bumps_target(self) -> None:
        subs = [
            _submission("sub-a", [_row(0)]),
            _submission("sub-b", [_row(0, text=None, type_raw=None, spurious=True)]),
        ]
        canonical, agreement, target = _run(subs)
        assert target == 3
        assert canonical is None


class TestMissingRowMajority:
    def test_two_reviewers_report_same_gap_and_text_injects(self) -> None:
        # Bundle has 2 rows (indices 0 and 1); reviewers report a missing
        # row between them (indices 0 and 1).
        subs = [
            _submission(
                "sub-a",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="LOU REED - PERFECT DAY",
                        type_raw="M",
                        notes=None,
                    )
                ],
            ),
            _submission(
                "sub-b",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="LOU REED - PERFECT DAY",
                        type_raw="M",
                        notes=None,
                    )
                ],
            ),
        ]
        canonical, agreement, target = _run(subs, bundle_row_count=2)
        assert target == 2
        assert canonical is not None
        # Two bundle rows + one injection = 3 canonical rows.
        assert len(canonical.rows) == 3
        # Injection sits between bundle rows 0 and 1 → canonical row 1.
        injected = canonical.rows[1]
        assert injected.bundle_row_index is None
        assert injected.inserted_between_bundle_rows == (0, 1)
        assert injected.raw_text == "LOU REED - PERFECT DAY"
        assert injected.verification.status == "under_emit_majority"
        # canonical_row_index is contiguous 0..M-1
        assert [r.canonical_row_index for r in canonical.rows] == [0, 1, 2]


class TestMissingRowMinority:
    def test_lone_reporter_no_injection_recorded_in_reports(self) -> None:
        subs = [
            _submission(
                "sub-a",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="LOU REED - PERFECT DAY",
                        type_raw="M",
                        notes=None,
                    )
                ],
            ),
            _submission("sub-b", [_row(0), _row(1)]),
            _submission("sub-c", [_row(0), _row(1)]),
        ]
        canonical, agreement, target = _run(subs, bundle_row_count=2)
        # target bumped to 3 because gap presence was 1-2 → not needed;
        # actually 1 out of 2 initial reviewers reports, escalates. Third
        # reviewer doesn't report — minority holds.
        assert target == 3
        assert canonical is not None
        assert len(canonical.rows) == 2  # no injection
        assert len(canonical.missing_row_reports) == 1
        report = canonical.missing_row_reports[0]
        assert report.between_bundle_rows == (0, 1)
        assert report.reporting_reviewer_shorts == [_short("sub-a")]
        assert report.suggested_texts == ["LOU REED - PERFECT DAY"]


class TestMissingRowGapSplitEscalates:
    def test_n2_one_reports_one_doesnt_bumps_target(self) -> None:
        subs = [
            _submission(
                "sub-a",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="LOU REED",
                        type_raw=None,
                        notes=None,
                    )
                ],
            ),
            _submission("sub-b", [_row(0), _row(1)]),
        ]
        canonical, agreement, target = _run(subs, bundle_row_count=2)
        assert target == 3
        assert canonical is None


class TestMissingRowGapMajorityNoTextConsensus:
    def test_majority_reports_gap_but_no_text_agreement_marks_illegible(self) -> None:
        subs = [
            _submission(
                "sub-a",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="LOU REED - PERFECT DAY",
                        type_raw=None,
                        notes=None,
                    )
                ],
            ),
            _submission(
                "sub-b",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="DAVID BOWIE - HEROES",
                        type_raw=None,
                        notes=None,
                    )
                ],
            ),
            _submission(
                "sub-c",
                [_row(0), _row(1)],
                missing_row_markers=[
                    MissingRowMarker(
                        between_bundle_rows=(0, 1),
                        suggested_text="PATTI SMITH - GLORIA",
                        type_raw=None,
                        notes=None,
                    )
                ],
            ),
        ]
        canonical, agreement, target = _run(subs, bundle_row_count=2)
        assert target == 3
        assert canonical is not None
        assert len(canonical.rows) == 3
        injected = canonical.rows[1]
        assert injected.raw_text == "__illegible__"
        assert injected.verification.status == "under_emit_no_text_agreement"
        assert len(injected.verification.raw_text_dissents) == 3


class TestTypeRawDoodleCluster:
    def test_doodle_cluster_folds_to_unknown(self) -> None:
        subs = [
            _submission("sub-a", [_row(0, type_raw="")]),
            _submission("sub-b", [_row(0, type_raw="?")]),
            _submission("sub-c", [_row(0, type_raw="doodle")]),
        ]
        canonical, agreement, target = _run(subs)
        # All 3 fold to `_unknown` → agree.
        assert target == 2 or target == 3  # depends on whether text agrees; test uses default text agreement
        assert canonical is not None
        row = canonical.rows[0]
        assert row.verification.type_raw_status == "unanimous"
        assert row.type_raw == "_unknown"


class TestPairConcordance:
    def test_pairwise_agreement_reflects_per_row_agreement(self) -> None:
        # 3 rows: rows 0 and 1 unanimous; row 2 has sub-c dissenting.
        subs = [
            _submission(
                "sub-a",
                [_row(0), _row(1), _row(2, text="X - Y")],
            ),
            _submission(
                "sub-b",
                [_row(0), _row(1), _row(2, text="X - Y")],
            ),
            _submission(
                "sub-c",
                [_row(0), _row(1), _row(2, text="Q - Z")],
            ),
        ]
        canonical, agreement, target = _run(subs, bundle_row_count=3)
        assert target == 3
        assert canonical is not None
        assert agreement is not None
        # Three pairs: a-b, a-c, b-c.
        assert len(agreement.pair_concordance) == 3
        rates = {(p.a, p.b): p.raw_text_agree_rate for p in agreement.pair_concordance}
        # a-b agree on all 3 rows.
        assert rates[(_short("sub-a"), _short("sub-b"))] == pytest.approx(1.0)
        # a-c agree on 2/3.
        assert rates[(_short("sub-a"), _short("sub-c"))] == pytest.approx(2 / 3)


class TestReturnsNoneWhenBelowTarget:
    def test_single_submission_returns_none(self) -> None:
        subs = [_submission("sub-a", [_row(0)])]
        canonical, agreement, target = _run(subs)
        assert target == 2
        assert canonical is None
        assert agreement is None


class TestReturnTypes:
    def test_settled_returns_pydantic_models(self) -> None:
        subs = [
            _submission("sub-a", [_row(0)]),
            _submission("sub-b", [_row(0)]),
        ]
        canonical, agreement, _ = _run(subs)
        assert isinstance(canonical, CalibrationCanonical)
        assert canonical.settled_at == SETTLED_AT
        assert canonical.stem == STEM
        assert agreement is not None
        assert agreement.stem == STEM
        assert agreement.year == YEAR
        assert agreement.bucket == BUCKET
