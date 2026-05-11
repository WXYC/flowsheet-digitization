"""Tests for `core.spot_check` — the pure data plumbing behind the spot-check tool.

Covers:
  * collect_entries: JSON-walking, skipping continuation rows (no artist),
    null vs empty-string track handling
  * shard: even/uneven splits, n > items, empty input
  * evaluate: per-page accumulation, hit/miss accounting, joint denominator
  * rank_by_joint_miss: ordering + tiebreak
  * joint_miss_pairs: corpus-wide frequency, ignores rows with no track

These never touch Postgres — the artist/joint hit dicts are inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.spot_check import (
    EntryRef,
    collect_entries,
    evaluate,
    joint_miss_pairs,
    rank_by_joint_miss,
    shard,
)

# -- fixtures --------------------------------------------------------------


def _make_page(
    entries: list[dict],
    *,
    page_date_raw: str | None = "Mon 1/1/90",
    quadrant_position: str = "top_left",
) -> dict:
    """Build a minimal PageResult-shaped dict for fixture writing."""
    return {
        "page_date_raw": page_date_raw,
        "quadrants": [
            {
                "position": quadrant_position,
                "hour_raw": "10AM",
                "jock_raw": "DJ",
                "entries": entries,
                "oddities": [],
            },
            # Three more empty quadrants to satisfy the canonical shape;
            # the script only iterates `data["quadrants"]` so emptiness is fine.
            {
                "position": "top_right",
                "hour_raw": None,
                "jock_raw": None,
                "entries": [],
                "oddities": [],
            },
            {
                "position": "bottom_left",
                "hour_raw": None,
                "jock_raw": None,
                "entries": [],
                "oddities": [],
            },
            {
                "position": "bottom_right",
                "hour_raw": None,
                "jock_raw": None,
                "entries": [],
                "oddities": [],
            },
        ],
        "model_version": "test",
        "extracted_at": "2026-01-01T00:00:00Z",
        "oddities": [],
    }


def _entry(
    row_index: int,
    *,
    artist: str | None = None,
    track: str | None = None,
    raw: str = "x",
) -> dict:
    """Legacy-shape entry: the 34 pre-audit corpus JSONs carry
    `artist_guess` / `track_guess` keys; `collect_entries` still honors
    them when present."""
    return {
        "row_index": row_index,
        "raw_text": raw,
        "artist_guess": artist,
        "track_guess": track,
        "confidence": "high",
        "notes": None,
        "oddities": [],
    }


def _entry_v2(row_index: int, *, raw: str) -> dict:
    """Post-audit entry: no `artist_guess` / `track_guess`. The artist
    and track come from `parse_artist_track(raw_text)` at read time."""
    return {
        "row_index": row_index,
        "raw_text": raw,
        "confidence": "high",
        "notes": None,
        "oddities": [],
    }


# -- collect_entries -------------------------------------------------------


class TestCollectEntries:
    def test_skips_entries_with_no_artist(self, tmp_path: Path) -> None:
        page = _make_page(
            [
                _entry(0, artist="Stereolab", track="Brakhage"),
                _entry(1, artist=None, track="continuation track"),
                _entry(2, artist="", track="empty-string artist"),
            ],
        )
        (tmp_path / "page-01.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert len(rows) == 1
        assert rows[0].artist == "Stereolab"
        assert rows[0].track == "Brakhage"

    def test_empty_track_becomes_none(self, tmp_path: Path) -> None:
        page = _make_page(
            [
                _entry(0, artist="Cat Power", track=""),
                _entry(1, artist="Stereolab", track=None),
            ],
        )
        (tmp_path / "p.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert {r.artist: r.track for r in rows} == {"Cat Power": None, "Stereolab": None}

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        page = _make_page([_entry(0, artist="  Juana Molina ", track="  la paradoja  ")])
        (tmp_path / "p.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert rows[0].artist == "Juana Molina"
        assert rows[0].track == "la paradoja"

    def test_walks_subdirectories_in_sorted_order(self, tmp_path: Path) -> None:
        (tmp_path / "1991").mkdir()
        (tmp_path / "1990").mkdir()
        (tmp_path / "1991" / "p.json").write_text(json.dumps(_make_page([_entry(0, artist="B")])))
        (tmp_path / "1990" / "p.json").write_text(json.dumps(_make_page([_entry(0, artist="A")])))

        rows = collect_entries(tmp_path)

        assert [r.artist for r in rows] == ["A", "B"]

    def test_returns_empty_list_for_empty_root(self, tmp_path: Path) -> None:
        assert collect_entries(tmp_path) == []

    def test_derives_artist_track_from_raw_text_for_post_audit_entries(
        self, tmp_path: Path
    ) -> None:
        """The post-#41 on-disk shape has no `artist_guess`/`track_guess`
        keys — both fall back to `parse_artist_track(raw_text)`. Without
        this fallback, `collect_entries` would silently return zero rows
        on every new corpus extraction."""
        page = _make_page(
            [
                _entry_v2(0, raw="JUANA MOLINA - LA PARADOJA"),
                _entry_v2(1, raw="STEREOLAB"),  # artist-only, no track
                _entry_v2(2, raw="   "),  # blank row, skipped
            ],
        )
        (tmp_path / "p.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert {(r.artist, r.track) for r in rows} == {
            ("JUANA MOLINA", "LA PARADOJA"),
            ("STEREOLAB", None),
        }

    def test_legacy_artist_guess_takes_precedence_over_parsing(self, tmp_path: Path) -> None:
        """On the 34 pre-audit JSONs `artist_guess` / `track_guess` exist
        and may have been hand-corrected by the original model in ways
        the deterministic parser can't reproduce. Honor the stored value
        on those rows so a re-run of the spot-check doesn't drift."""
        page = _make_page(
            [
                {
                    "row_index": 0,
                    "raw_text": "JUANA MOLINA - LA PARADOJA",
                    "artist_guess": "Juana Molina",
                    "track_guess": "la paradoja",
                    "confidence": "high",
                    "notes": None,
                    "oddities": [],
                }
            ],
        )
        (tmp_path / "p.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert (rows[0].artist, rows[0].track) == ("Juana Molina", "la paradoja")

    def test_carries_quadrant_position_and_row_index(self, tmp_path: Path) -> None:
        page = _make_page(
            [_entry(7, artist="Pavement", track="Cut Your Hair")],
            quadrant_position="bottom_right",
        )
        (tmp_path / "p.json").write_text(json.dumps(page))

        rows = collect_entries(tmp_path)

        assert rows[0].quadrant == "bottom_right"
        assert rows[0].row_index == 7


# -- shard -----------------------------------------------------------------


class TestShard:
    def test_even_split(self) -> None:
        assert shard([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split_extras_go_to_earliest_shards(self) -> None:
        # 5 items / 3 shards: extras of 5%3=2 go to shards 0 and 1
        assert shard([1, 2, 3, 4, 5], 3) == [[1, 2], [3, 4], [5]]

    def test_n_larger_than_items_drops_empty_shards(self) -> None:
        # 2 items into 5 shards → only 2 non-empty shards returned.
        out = shard(["a", "b"], 5)
        assert out == [["a"], ["b"]]
        assert all(len(s) > 0 for s in out)

    def test_empty_input_returns_empty(self) -> None:
        assert shard([], 4) == []

    def test_n_equals_one_returns_single_shard(self) -> None:
        assert shard([1, 2, 3], 1) == [[1, 2, 3]]

    def test_partition_is_complete_and_ordered(self) -> None:
        items = list(range(17))
        out = shard(items, 4)
        assert sum(out, []) == items


# -- evaluate --------------------------------------------------------------


def _ref(page: Path, artist: str, track: str | None, row_index: int = 0) -> EntryRef:
    return EntryRef(
        page_path=page, quadrant="top_left", row_index=row_index, artist=artist, track=track
    )


class TestEvaluate:
    def test_artist_hit_increments_hit_count(self) -> None:
        page = Path("p1.json")
        rows = [_ref(page, "Stereolab", "Brakhage")]
        artist_hits = {"Stereolab": True}
        joint_hits = {("Stereolab", "Brakhage"): True}

        pages = evaluate(rows, artist_hits, joint_hits)

        assert pages[page].total == 1
        assert pages[page].artist_hits == 1
        assert pages[page].artist_misses == []
        assert pages[page].joint_hits == 1
        assert pages[page].joint_misses == []

    def test_artist_miss_records_the_entry(self) -> None:
        page = Path("p1.json")
        rows = [_ref(page, "PE", "Revolution Generation")]
        artist_hits = {"PE": False}
        joint_hits = {("PE", "Revolution Generation"): False}

        pages = evaluate(rows, artist_hits, joint_hits)

        assert pages[page].artist_hits == 0
        assert [r.artist for r in pages[page].artist_misses] == ["PE"]
        assert [r.artist for r in pages[page].joint_misses] == ["PE"]

    def test_artist_hit_track_miss_is_a_real_signal(self) -> None:
        # Artist is in the library but joint pair isn't - the strict signal
        # of "we own this artist but not this track."
        page = Path("p1.json")
        rows = [_ref(page, "Misfits", "Mommy Can I Go Out")]
        artist_hits = {"Misfits": True}
        joint_hits = {("Misfits", "Mommy Can I Go Out"): False}

        pages = evaluate(rows, artist_hits, joint_hits)

        assert pages[page].artist_hits == 1
        assert pages[page].joint_hits == 0
        assert len(pages[page].joint_misses) == 1
        assert pages[page].artist_misses == []  # not a miss on the artist side

    def test_track_none_skips_joint_check(self) -> None:
        page = Path("p1.json")
        rows = [_ref(page, "Stereolab", None)]
        artist_hits = {"Stereolab": True}
        joint_hits: dict[tuple[str, str], bool] = {}  # no key needed

        pages = evaluate(rows, artist_hits, joint_hits)

        assert pages[page].with_track == 0
        assert pages[page].joint_hits == 0
        assert pages[page].joint_misses == []

    def test_groups_rows_by_page_path(self) -> None:
        p1, p2 = Path("p1.json"), Path("p2.json")
        rows = [
            _ref(p1, "Stereolab", "Brakhage"),
            _ref(p1, "Cat Power", "Cross Bones Style"),
            _ref(p2, "Pavement", "Cut Your Hair"),
        ]
        artist_hits = {"Stereolab": True, "Cat Power": True, "Pavement": False}
        joint_hits = {
            ("Stereolab", "Brakhage"): True,
            ("Cat Power", "Cross Bones Style"): False,
            ("Pavement", "Cut Your Hair"): False,
        }

        pages = evaluate(rows, artist_hits, joint_hits)

        assert set(pages.keys()) == {p1, p2}
        assert pages[p1].total == 2
        assert pages[p2].total == 1

    def test_missing_lookup_key_raises_keyerror(self) -> None:
        # A missing key indicates the dispatch layer skipped a query - that's
        # a bug we want surfaced loudly, not silently treated as a miss.
        rows = [_ref(Path("p.json"), "Stereolab", "Brakhage")]
        with pytest.raises(KeyError):
            evaluate(rows, artist_hits={}, joint_hits={})


# -- rank_by_joint_miss + joint_miss_pairs ---------------------------------


class TestRankAndCount:
    def test_rank_orders_by_joint_miss_count_descending(self) -> None:
        p1, p2, p3 = Path("p1.json"), Path("p2.json"), Path("p3.json")
        # p2 has 3 joint misses, p1 has 1, p3 has 2.
        rows = [
            _ref(p1, "A", "1"),
            _ref(p2, "B", "1"),
            _ref(p2, "B", "2"),
            _ref(p2, "B", "3"),
            _ref(p3, "C", "1"),
            _ref(p3, "C", "2"),
        ]
        artist_hits = {"A": True, "B": True, "C": True}
        joint_hits = {(r.artist, r.track): False for r in rows if r.track is not None}

        pages = evaluate(rows, artist_hits, joint_hits)
        ranked = rank_by_joint_miss(pages)

        assert [p.page_path for p in ranked] == [p2, p3, p1]

    def test_rank_tiebreak_is_artist_misses(self) -> None:
        p1, p2 = Path("p1.json"), Path("p2.json")
        # Both pages have 1 joint miss, but p1 also has the artist miss.
        rows = [_ref(p1, "A", "1"), _ref(p2, "B", "1")]
        artist_hits = {"A": False, "B": True}  # p1 misses artist, p2 hits it
        joint_hits = {("A", "1"): False, ("B", "1"): False}

        ranked = rank_by_joint_miss(evaluate(rows, artist_hits, joint_hits))

        assert ranked[0].page_path == p1

    def test_joint_miss_pairs_counts_across_pages(self) -> None:
        p1, p2 = Path("p1.json"), Path("p2.json")
        rows = [
            _ref(p1, "Loop", "The Nail Will Burn"),
            _ref(p2, "Loop", "The Nail Will Burn"),  # same pair, different page
            _ref(p1, "Misfits", "Mommy Can I Go Out"),
        ]
        artist_hits = {"Loop": True, "Misfits": True}
        joint_hits = {(r.artist, r.track): False for r in rows if r.track is not None}

        counts = joint_miss_pairs(evaluate(rows, artist_hits, joint_hits))

        assert counts[("Loop", "The Nail Will Burn")] == 2
        assert counts[("Misfits", "Mommy Can I Go Out")] == 1

    def test_joint_miss_pairs_ignores_rows_with_no_track(self) -> None:
        p = Path("p.json")
        rows = [_ref(p, "Stereolab", None), _ref(p, "Cat Power", "Sea of Love")]
        artist_hits = {"Stereolab": False, "Cat Power": False}
        joint_hits = {("Cat Power", "Sea of Love"): False}

        counts = joint_miss_pairs(evaluate(rows, artist_hits, joint_hits))

        assert list(counts.keys()) == [("Cat Power", "Sea of Love")]
