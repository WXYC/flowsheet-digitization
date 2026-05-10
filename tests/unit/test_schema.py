"""Tests for the Pydantic models that form the Gemini response contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from core.schema import Confidence, Entry, PageResult, Quadrant, QuadrantPosition


class TestEntry:
    def test_minimal_entry(self) -> None:
        e = Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high")
        assert e.row_index == 0
        assert e.artist_guess is None
        assert e.track_guess is None
        assert e.notes is None

    def test_full_entry(self) -> None:
        e = Entry(
            row_index=2,
            raw_text="LED ZEP - TRAMPLED",
            artist_guess="Led Zeppelin",
            track_guess="Trampled Under Foot",
            confidence="medium",
            notes="continuation",
        )
        assert e.artist_guess == "Led Zeppelin"
        assert e.confidence == "medium"

    def test_invalid_confidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Entry(row_index=0, raw_text="x", confidence="very-high")  # type: ignore[arg-type]

    def test_negative_row_index_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Entry(row_index=-1, raw_text="x", confidence="high")

    def test_type_raw_defaults_to_none(self) -> None:
        e = Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high")
        assert e.type_raw is None

    @pytest.mark.parametrize(
        "value",
        [
            "H",
            "M",
            "L",
            "Std",
            "O",
            "R",
            "R⇒",
            "hand-drawn smiley with tongue",
        ],
    )
    def test_type_raw_round_trips_verbatim(self, value: str) -> None:
        """`type_raw` is a verbatim string; the schema must NOT normalize
        case, strip whitespace, or coerce the value into an enum.

        Tested values cover: the six canonical letters (H/M/L/Std/O/R), the
        `R⇒` handoff variant, and a doodle description (the `Phase 2 type-
        column` plan documents these as the value distribution we expect)."""
        e = Entry(row_index=0, raw_text="x", confidence="high", type_raw=value)
        rebuilt = Entry.model_validate_json(e.model_dump_json())
        assert rebuilt.type_raw == value

    def test_type_raw_omitted_in_old_extraction_json_validates(self) -> None:
        """Phase-1 extractions have no `type_raw` field anywhere. The new
        schema must keep validating those old extractions; otherwise we'd
        invalidate the existing `data/results/**/*.json` corpus on the day
        we land Phase 2.

        Hermetic: builds an old-shape dict in memory rather than reading
        from `data/` (which is gitignored / not present in CI).
        """
        old_extraction = {
            "page_date_raw": "Monday 1 Jan '90",
            "model_version": "gemini-3.1-pro-preview",
            "extracted_at": datetime.now(UTC).isoformat(),
            "oddities": [],
            "quadrants": [
                {
                    "position": p,
                    "hour_raw": None,
                    "jock_raw": None,
                    "oddities": [],
                    "entries": [
                        {
                            "row_index": 0,
                            "raw_text": "LED ZEP - TRAMPLED",
                            "artist_guess": None,
                            "track_guess": None,
                            "confidence": "high",
                            "notes": None,
                            "oddities": [],
                        }
                    ]
                    if p == "top_left"
                    else [],
                }
                for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
        }
        page = PageResult.model_validate_json(json.dumps(old_extraction))
        assert all(e.type_raw is None for q in page.quadrants for e in q.entries)


class TestQuadrant:
    def test_quadrant_with_entries(self) -> None:
        q = Quadrant(
            position="top_left",
            hour_raw="6AM",
            jock_raw="ALECIA",
            entries=[
                Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high"),
                Entry(row_index=1, raw_text="STONES - LITTLE RED", confidence="high"),
            ],
        )
        assert q.position == "top_left"
        assert len(q.entries) == 2

    def test_empty_entries_allowed(self) -> None:
        # An hour the DJ didn't fill in still has a quadrant placeholder.
        q = Quadrant(position="bottom_right", hour_raw=None, jock_raw=None, entries=[])
        assert q.entries == []

    def test_invalid_position_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Quadrant(position="middle", hour_raw=None, jock_raw=None, entries=[])  # type: ignore[arg-type]


class TestPageResult:
    def _quad(self, position: QuadrantPosition) -> Quadrant:
        return Quadrant(position=position, hour_raw=None, jock_raw=None, entries=[])

    def test_requires_four_quadrants_in_order(self) -> None:
        page = PageResult(
            page_date_raw="Monday 1 Jan '90",
            quadrants=[
                self._quad("top_left"),
                self._quad("top_right"),
                self._quad("bottom_left"),
                self._quad("bottom_right"),
            ],
            model_version="gemini-3.1-pro-preview",
            extracted_at=datetime.now(UTC),
        )
        assert [q.position for q in page.quadrants] == [
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]

    def test_rejects_wrong_quadrant_count(self) -> None:
        with pytest.raises(ValidationError):
            PageResult(
                page_date_raw=None,
                quadrants=[self._quad("top_left")],
                model_version="m",
                extracted_at=datetime.now(UTC),
            )

    def test_rejects_out_of_order_quadrants(self) -> None:
        with pytest.raises(ValidationError):
            PageResult(
                page_date_raw=None,
                quadrants=[
                    self._quad("top_right"),
                    self._quad("top_left"),
                    self._quad("bottom_left"),
                    self._quad("bottom_right"),
                ],
                model_version="m",
                extracted_at=datetime.now(UTC),
            )

    def test_rejects_duplicate_quadrant_positions(self) -> None:
        with pytest.raises(ValidationError):
            PageResult(
                page_date_raw=None,
                quadrants=[
                    self._quad("top_left"),
                    self._quad("top_left"),
                    self._quad("bottom_left"),
                    self._quad("bottom_right"),
                ],
                model_version="m",
                extracted_at=datetime.now(UTC),
            )


def test_confidence_values() -> None:
    # Sanity: documents the exact set the pipeline contracts on.
    assert set(Confidence.__args__) == {"high", "medium", "low"}  # type: ignore[attr-defined]


class TestOddities:
    """Free-text `oddities` lists at three schema levels.

    These are how we let Gemini surface anything unexpected on the page —
    things the rest of the schema doesn't have a place for. We aggregate
    them after a few hundred runs to discover phase-2 categories.
    """

    def test_entry_oddities_defaults_to_empty_list(self) -> None:
        e = Entry(row_index=0, raw_text="x", confidence="high")
        assert e.oddities == []

    def test_entry_accepts_oddities_list(self) -> None:
        e = Entry(
            row_index=0,
            raw_text="x",
            confidence="medium",
            oddities=["left margin has '*' next to this entry"],
        )
        assert e.oddities == ["left margin has '*' next to this entry"]

    def test_quadrant_oddities_defaults_to_empty_list(self) -> None:
        q = Quadrant(position="top_left", hour_raw=None, jock_raw=None, entries=[])
        assert q.oddities == []

    def test_quadrant_accepts_oddities_list(self) -> None:
        q = Quadrant(
            position="top_left",
            hour_raw="6AM",
            jock_raw="ALECIA",
            entries=[],
            oddities=["rows 4-8 bracketed with 'ALL-REQUEST XMAS'"],
        )
        assert q.oddities == ["rows 4-8 bracketed with 'ALL-REQUEST XMAS'"]

    def test_page_result_oddities_defaults_to_empty_list(self) -> None:
        page = PageResult(
            page_date_raw=None,
            quadrants=[
                Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
                for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
            model_version="m",
            extracted_at=datetime.now(UTC),
        )
        assert page.oddities == []

    def test_page_result_accepts_oddities_list(self) -> None:
        page = PageResult(
            page_date_raw=None,
            quadrants=[
                Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
                for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
            model_version="m",
            extracted_at=datetime.now(UTC),
            oddities=[
                "page is rotated 180 degrees",
                "comments field reads: 'declared today anti-valentines day'",
            ],
        )
        assert len(page.oddities) == 2

    def test_oddities_round_trip_through_json(self) -> None:
        page = PageResult(
            page_date_raw=None,
            quadrants=[
                Quadrant(
                    position="top_left",
                    hour_raw=None,
                    jock_raw=None,
                    entries=[
                        Entry(
                            row_index=0,
                            raw_text="x",
                            confidence="high",
                            oddities=["entry-level oddity"],
                        )
                    ],
                    oddities=["quadrant-level oddity"],
                ),
                *[
                    Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
                    for p in ("top_right", "bottom_left", "bottom_right")
                ],
            ],
            model_version="m",
            extracted_at=datetime.now(UTC),
            oddities=["page-level oddity"],
        )
        roundtripped = PageResult.model_validate_json(page.model_dump_json())
        assert roundtripped.oddities == ["page-level oddity"]
        assert roundtripped.quadrants[0].oddities == ["quadrant-level oddity"]
        assert roundtripped.quadrants[0].entries[0].oddities == ["entry-level oddity"]


def test_page_result_schema_has_no_additional_properties_key() -> None:
    """Google's response_schema validator rejects `additionalProperties`.

    Pydantic emits this key when a model has extra='forbid'; if any of our
    models uses that, Gemini returns 400 INVALID_ARGUMENT and every page
    fails. This test prevents the regression.
    """

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node, (
                "PageResult.model_json_schema() emits 'additionalProperties' — "
                "Google's response_schema validator rejects this. Remove "
                "extra='forbid' from the model_config that introduced it."
            )
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(PageResult.model_json_schema())
