"""Tests for the Pydantic models that form the Gemini response contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from core.schema import (
    Confidence,
    Entry,
    GeminiPageResult,
    PageResult,
    Quadrant,
    QuadrantPosition,
)


class TestEntry:
    def test_minimal_entry(self) -> None:
        e = Entry(row_index=0, raw_text="LED ZEP - TRAMPLED", confidence="high")
        assert e.row_index == 0
        assert e.notes is None

    def test_full_entry(self) -> None:
        e = Entry(
            row_index=2,
            raw_text="LED ZEP - TRAMPLED",
            confidence="medium",
            notes="continuation",
        )
        assert e.raw_text == "LED ZEP - TRAMPLED"
        assert e.confidence == "medium"

    def test_artist_guess_track_guess_keys_are_ignored(self) -> None:
        """The 34 pre-audit corpus JSONs carry `artist_guess` and `track_guess`.
        The new schema must accept the old shape (extra keys silently ignored)
        and round-trip the entry without those keys reappearing on dump. That's
        the load-bearing backward-compat contract this PR rests on."""
        e = Entry.model_validate(
            {
                "row_index": 0,
                "raw_text": "JUANA MOLINA - LA PARADOJA",
                "artist_guess": "JUANA MOLINA",
                "track_guess": "LA PARADOJA",
                "confidence": "high",
            }
        )
        assert e.raw_text == "JUANA MOLINA - LA PARADOJA"
        dumped = e.model_dump()
        assert "artist_guess" not in dumped
        assert "track_guess" not in dumped

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

    def test_verified_by_defaults_to_none(self) -> None:
        """A PageResult written by the pipeline has no reviewer. Default
        matters: every verified.json on disk that pre-dates the OIDC PR
        re-loads with `verified_by=None`, and no migration is needed."""
        page = PageResult(
            page_date_raw=None,
            quadrants=[
                self._quad("top_left"),
                self._quad("top_right"),
                self._quad("bottom_left"),
                self._quad("bottom_right"),
            ],
            model_version="m",
            extracted_at=datetime.now(UTC),
        )
        assert page.verified_by is None

    def test_old_verified_json_without_verified_by_loads(self) -> None:
        """Backwards compat: a verified.json saved before the OIDC PR
        omits `verified_by` and must still parse. Exercises the same
        load path the verifier server runs on every save (model_validate
        of the on-disk JSON)."""
        from core.schema import PageResult

        old = {
            "page_date_raw": None,
            "quadrants": [
                {
                    "position": p,
                    "hour_raw": None,
                    "jock_raw": None,
                    "entries": [],
                    "oddities": [],
                }
                for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
            "comments_raw": None,
            "oddities": [],
            "model_version": "m",
            "extracted_at": "2026-05-01T00:00:00Z",
            # No verified_by key — the pre-OIDC shape.
        }
        page = PageResult.model_validate(old)
        assert page.verified_by is None

    def test_verified_by_round_trips_through_pageresult(self) -> None:
        """A populated `verified_by` survives encode -> decode through
        `model_dump_json` -> `model_validate_json` so the verifier
        server can write the file and the SPA reload reads back the
        exact block it wrote."""
        from core.schema import PageResult, VerifiedBy

        page = PageResult(
            page_date_raw=None,
            quadrants=[
                self._quad(p) for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
            model_version="m",
            extracted_at=datetime.now(UTC),
            verified_by=VerifiedBy(
                user_id="u-1",
                username="reviewer",
                real_name="Real Name",
                dj_name="DJ Name",
                verified_at=datetime.now(UTC),
            ),
        )
        roundtripped = PageResult.model_validate_json(page.model_dump_json())
        assert roundtripped.verified_by is not None
        assert roundtripped.verified_by.user_id == "u-1"
        assert roundtripped.verified_by.username == "reviewer"
        assert roundtripped.verified_by.real_name == "Real Name"
        assert roundtripped.verified_by.dj_name == "DJ Name"


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


class TestGeminiPageResult:
    """`GeminiPageResult` is the subset of `PageResult` that Gemini actually
    fills. `model_version` and `extracted_at` belong to the caller — leaving
    them in `response_schema` makes Gemini hallucinate plausible values
    (real run with `gemini-3.1-pro-preview` produced 4 distinct fake model
    ids and timestamps off by 14+ months). The split closes that hole."""

    def _quads(self) -> list[Quadrant]:
        return [
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
            for p in ("top_left", "top_right", "bottom_left", "bottom_right")
        ]

    def test_minimal_construction(self) -> None:
        result = GeminiPageResult(page_date_raw=None, quadrants=self._quads())
        assert result.page_date_raw is None
        assert result.oddities == []

    def test_response_schema_omits_caller_set_fields(self) -> None:
        """The load-bearing assertion of this whole change: when
        `GeminiPageResult` is the `response_schema`, the model is never
        asked to fill `model_version` or `extracted_at`."""
        schema_json = json.dumps(GeminiPageResult.model_json_schema())
        assert "model_version" not in schema_json
        assert "extracted_at" not in schema_json

    def test_response_schema_has_no_additional_properties_key(self) -> None:
        """Same Gemini constraint as `PageResult` (see the regression test
        below) — `additionalProperties` triggers a 400 from Google's
        validator. Mirrored on the actual response-schema model so a
        future Pydantic-config change can't slip past us."""

        def walk(node: object) -> None:
            if isinstance(node, dict):
                assert "additionalProperties" not in node, (
                    "GeminiPageResult.model_json_schema() emits "
                    "'additionalProperties' — Google's response_schema "
                    "validator rejects this."
                )
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(GeminiPageResult.model_json_schema())

    def test_enforces_four_quadrants_in_order(self) -> None:
        with pytest.raises(ValidationError):
            GeminiPageResult(page_date_raw=None, quadrants=[])

    def test_can_be_promoted_to_page_result(self) -> None:
        """Pipeline pattern: take what Gemini produced + add the two
        caller-set fields to land a `PageResult` for disk."""
        gemini_result = GeminiPageResult(
            page_date_raw="Monday 1 Jan '90",
            quadrants=self._quads(),
            oddities=["page-level note"],
        )
        page = PageResult(
            **gemini_result.model_dump(),
            model_version="gemini-3.1-pro-preview",
            extracted_at=datetime.now(UTC),
        )
        assert page.page_date_raw == "Monday 1 Jan '90"
        assert page.oddities == ["page-level note"]
        assert page.model_version == "gemini-3.1-pro-preview"


class TestCommentsRaw:
    """The bottom-of-page Comments field is captured into `comments_raw` on
    `GeminiPageResult` (Phase 2). It's verbatim like the other `_raw` fields
    — no normalization, no truncation. Inheritance means `PageResult` gets
    the field for free and old extractions (no `comments_raw` key) still
    validate so we don't invalidate the existing corpus."""

    def _quads(self) -> list[Quadrant]:
        return [
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
            for p in ("top_left", "top_right", "bottom_left", "bottom_right")
        ]

    def test_defaults_to_none_on_gemini_page_result(self) -> None:
        result = GeminiPageResult(page_date_raw=None, quadrants=self._quads())
        assert result.comments_raw is None

    def test_defaults_to_none_on_page_result(self) -> None:
        page = PageResult(
            page_date_raw=None,
            quadrants=self._quads(),
            model_version="m",
            extracted_at=datetime.now(UTC),
        )
        assert page.comments_raw is None

    def test_accepts_verbatim_string(self) -> None:
        text = "declared today anti-Valentines Day"
        result = GeminiPageResult(
            page_date_raw=None,
            quadrants=self._quads(),
            comments_raw=text,
        )
        assert result.comments_raw == text

    def test_round_trips_through_json(self) -> None:
        text = "declared today anti-Valentines Day"
        page = PageResult(
            page_date_raw=None,
            quadrants=self._quads(),
            model_version="m",
            extracted_at=datetime.now(UTC),
            comments_raw=text,
        )
        rebuilt = PageResult.model_validate_json(page.model_dump_json())
        assert rebuilt.comments_raw == text

    def test_response_schema_names_comments_raw(self) -> None:
        """Gemini will only populate fields named in the response_schema."""
        schema_json = json.dumps(GeminiPageResult.model_json_schema())
        assert "comments_raw" in schema_json

    def test_old_extraction_json_without_comments_raw_validates(self) -> None:
        """The 34 existing corpus JSONs have no `comments_raw` key. Validation
        must accept that — the field defaults to None on missing input.
        Otherwise we'd invalidate every prior extraction the day we land this."""
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
                    "entries": [],
                }
                for p in ("top_left", "top_right", "bottom_left", "bottom_right")
            ],
        }
        page = PageResult.model_validate_json(json.dumps(old_extraction))
        assert page.comments_raw is None


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
