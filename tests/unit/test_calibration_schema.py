"""Tests for the calibration Pydantic models added to core/schema.py.

These cover the four on-disk shapes for the multi-reviewer calibration flow:

  * `CalibrationSubmission` — the `verified.<short>.json` / `draft.<short>.json` body.
  * `CalibrationCanonical` — the settled `canonical.json`.
  * `CalibrationAgreement` — the settled `agreement.json`.
  * Supporting types (`CalibrationRowSubmission`, `MissingRowMarker`,
    `CanonicalRow`, `RowVerification`, `RowDissent`, `MissingRowReport`,
    `SubmissionRecord`, `PairConcordance`).

The module-level `CALIBRATION_SCHEMA_VERSION` constant pins the shape family.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from core.schema import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationAgreement,
    CalibrationCanonical,
    CalibrationRowSubmission,
    CalibrationSubmission,
    CanonicalRow,
    MissingRowMarker,
    MissingRowReport,
    PairConcordance,
    RowDissent,
    RowVerification,
    SubmissionRecord,
    VerifiedBy,
)


def _reviewer(user_id: str = "sub-1") -> VerifiedBy:
    return VerifiedBy(
        user_id=user_id,
        username="jbromberg",
        real_name="Jake B.",
        dj_name="dj skid",
        verified_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
    )


class TestCalibrationSchemaVersion:
    def test_constant_is_one(self) -> None:
        assert CALIBRATION_SCHEMA_VERSION == 1


class TestCalibrationRowSubmission:
    def test_minimal_row(self) -> None:
        row = CalibrationRowSubmission(
            bundle_row_index=0,
            edited_text="BEATLES - HELP",
            type_raw="H",
            notes=None,
            spurious_flag=False,
        )
        assert row.bundle_row_index == 0
        assert row.edited_text == "BEATLES - HELP"
        assert row.spurious_flag is False

    def test_spurious_row_may_have_null_text(self) -> None:
        row = CalibrationRowSubmission(
            bundle_row_index=3,
            edited_text=None,
            type_raw=None,
            notes=None,
            spurious_flag=True,
        )
        assert row.edited_text is None
        assert row.spurious_flag is True

    def test_negative_bundle_row_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CalibrationRowSubmission(
                bundle_row_index=-1,
                edited_text="x",
                type_raw=None,
                notes=None,
                spurious_flag=False,
            )


class TestMissingRowMarker:
    def test_valid_adjacent_indices(self) -> None:
        marker = MissingRowMarker(
            between_bundle_rows=(5, 6),
            suggested_text="PIXIES - DEBASER",
            type_raw="M",
            notes=None,
        )
        assert marker.between_bundle_rows == (5, 6)

    def test_missing_row_marker_requires_adjacent_indices(self) -> None:
        with pytest.raises(ValidationError):
            MissingRowMarker(
                between_bundle_rows=(5, 7),
                suggested_text="PIXIES - DEBASER",
                type_raw=None,
                notes=None,
            )

    def test_missing_row_marker_rejects_reversed_pair(self) -> None:
        with pytest.raises(ValidationError):
            MissingRowMarker(
                between_bundle_rows=(6, 5),
                suggested_text="PIXIES - DEBASER",
                type_raw=None,
                notes=None,
            )


class TestCalibrationSubmission:
    def test_round_trip(self) -> None:
        submission = CalibrationSubmission(
            schema_version=1,
            stem="1990-04apr0106-page14",
            reviewer=_reviewer(),
            submitted_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
            rows=[
                CalibrationRowSubmission(
                    bundle_row_index=0,
                    edited_text="BEATLES - HELP",
                    type_raw="H",
                    notes=None,
                    spurious_flag=False,
                ),
            ],
            missing_row_markers=[],
        )
        dumped = submission.model_dump(mode="json")
        rebuilt = CalibrationSubmission.model_validate(dumped)
        assert rebuilt.stem == "1990-04apr0106-page14"
        assert rebuilt.schema_version == 1
        assert rebuilt.reviewer.user_id == "sub-1"
        assert len(rebuilt.rows) == 1

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CalibrationSubmission(
                schema_version=2,  # type: ignore[arg-type]
                stem="x",
                reviewer=_reviewer(),
                submitted_at=datetime.now(UTC),
                rows=[],
                missing_row_markers=[],
            )

    def test_empty_rows_and_markers_accepted(self) -> None:
        submission = CalibrationSubmission(
            schema_version=1,
            stem="x",
            reviewer=_reviewer(),
            submitted_at=datetime.now(UTC),
            rows=[],
            missing_row_markers=[],
        )
        assert submission.rows == []
        assert submission.missing_row_markers == []


class TestRowDissent:
    def test_dissent_shape(self) -> None:
        d = RowDissent(reviewer_short="a3b9e7c41f02", value="BEATLES - HELO")
        assert d.reviewer_short == "a3b9e7c41f02"
        assert d.value == "BEATLES - HELO"


class TestRowVerification:
    def test_full_verification_shape(self) -> None:
        v = RowVerification(
            status="majority",
            raw_text_status="majority",
            raw_text_dissents=[RowDissent(reviewer_short="a3b9e7c41f02", value="BEATLES - HELO")],
            type_raw_status="unanimous",
            type_raw_dissents=[],
            spurious_flag_status="unanimous_keep",
            spurious_flag_votes={"keep": 3, "spurious": 0},
            notes_values={"null": 3},
            reviewer_shorts=["a3b9e7c41f02", "9d27e6b5fa18", "f00b421ace30"],
        )
        assert v.status == "majority"
        assert v.spurious_flag_votes == {"keep": 3, "spurious": 0}


class TestCanonicalRow:
    def test_bundle_row_variant(self) -> None:
        row = CanonicalRow(
            canonical_row_index=0,
            bundle_row_index=0,
            inserted_between_bundle_rows=None,
            raw_text="BEATLES - HELP",
            type_raw="H",
            notes=None,
            confidence="high",
            verification=RowVerification(
                status="unanimous",
                raw_text_status="unanimous",
                raw_text_dissents=[],
                type_raw_status="unanimous",
                type_raw_dissents=[],
                spurious_flag_status="unanimous_keep",
                spurious_flag_votes={"keep": 2, "spurious": 0},
                notes_values={"null": 2},
                reviewer_shorts=["a", "b"],
            ),
        )
        assert row.bundle_row_index == 0
        assert row.inserted_between_bundle_rows is None

    def test_injected_row_variant(self) -> None:
        row = CanonicalRow(
            canonical_row_index=4,
            bundle_row_index=None,
            inserted_between_bundle_rows=(3, 4),
            raw_text="LOU REED - PERFECT DAY",
            type_raw=None,
            notes=None,
            confidence="medium",
            verification=RowVerification(
                status="under_emit_majority",
                raw_text_status="majority",
                raw_text_dissents=[],
                type_raw_status="unanimous",
                type_raw_dissents=[],
                spurious_flag_status="unanimous_keep",
                spurious_flag_votes={"keep": 2, "spurious": 0},
                notes_values={"null": 2},
                reviewer_shorts=["a", "b"],
            ),
        )
        assert row.bundle_row_index is None
        assert row.inserted_between_bundle_rows == (3, 4)


class TestCalibrationCanonical:
    def test_round_trip(self) -> None:
        canonical = CalibrationCanonical(
            schema_version=1,
            stem="1990-04apr0106-page14",
            settled_at=datetime(2026, 6, 12, 14, 35, 11, tzinfo=UTC),
            target_reviewers=3,
            rows=[],
            missing_row_reports=[
                MissingRowReport(
                    between_bundle_rows=(12, 13),
                    reporting_reviewer_shorts=["a3b9e7c41f02"],
                    suggested_texts=["LOU REED - PERFECT DAY"],
                )
            ],
        )
        dumped = canonical.model_dump(mode="json")
        rebuilt = CalibrationCanonical.model_validate(dumped)
        assert rebuilt.stem == "1990-04apr0106-page14"
        assert rebuilt.target_reviewers == 3
        assert len(rebuilt.missing_row_reports) == 1


class TestCalibrationAgreement:
    def test_round_trip(self) -> None:
        agreement = CalibrationAgreement(
            schema_version=1,
            stem="1990-04apr0106-page14",
            year="1990",
            bucket="anomaly",
            target_reviewers=3,
            submissions=[
                SubmissionRecord(
                    reviewer_short="a3b9e7c41f02",
                    submitted_at=datetime(2026, 6, 12, 14, 33, 1, tzinfo=UTC),
                ),
                SubmissionRecord(
                    reviewer_short="9d27e6b5fa18",
                    submitted_at=datetime(2026, 6, 12, 14, 34, 2, tzinfo=UTC),
                ),
            ],
            row_status_histogram={"unanimous": 18, "majority": 4, "illegible": 1},
            pair_concordance=[
                PairConcordance(
                    a="a3b9e7c41f02",
                    b="9d27e6b5fa18",
                    raw_text_agree_rate=0.92,
                    type_raw_agree_rate=1.0,
                ),
            ],
        )
        dumped = agreement.model_dump(mode="json")
        rebuilt = CalibrationAgreement.model_validate(dumped)
        assert rebuilt.year == "1990"
        assert rebuilt.bucket == "anomaly"
        assert rebuilt.pair_concordance[0].raw_text_agree_rate == pytest.approx(0.92)


class TestReuseVerifiedBy:
    """The plan requires the submission's `reviewer` field to reuse the
    existing `VerifiedBy` model unchanged. Ensures a `VerifiedBy` instance
    round-trips through a submission without any field renames or dropouts."""

    def test_reviewer_field_uses_verified_by_shape(self) -> None:
        r = _reviewer("sub-2")
        submission = CalibrationSubmission(
            schema_version=1,
            stem="x",
            reviewer=r,
            submitted_at=datetime.now(UTC),
            rows=[],
            missing_row_markers=[],
        )
        dumped = submission.model_dump(mode="json")
        assert dumped["reviewer"]["user_id"] == "sub-2"
        assert dumped["reviewer"]["username"] == "jbromberg"
        assert dumped["reviewer"]["real_name"] == "Jake B."
        assert dumped["reviewer"]["dj_name"] == "dj skid"
        # `verified_at` is a datetime; ensure it round-trips as ISO string.
        assert dumped["reviewer"]["verified_at"].startswith("2026-06-12T")
