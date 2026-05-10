"""Tests for `core.comments`: read-time normalize + entry-overlap diagnostic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.comments import (
    CommentEntryOverlap,
    find_comment_entry_overlaps,
    normalize_comments,
)
from core.schema import Entry, PageResult, Quadrant, QuadrantPosition

# ---------------------------------------------------------------------------
# normalize_comments
# ---------------------------------------------------------------------------


def test_normalize_none_passes_through() -> None:
    assert normalize_comments(None) is None


def test_normalize_empty_string_returns_none() -> None:
    """The on-disk shape uses None to mean "blank"; an empty-after-trim
    string should normalize to None so consumers have one sentinel."""
    assert normalize_comments("") is None
    assert normalize_comments("   ") is None
    assert normalize_comments("\n\n\n") is None
    assert normalize_comments("  \n  \n  ") is None


def test_normalize_collapses_internal_whitespace_within_line() -> None:
    """Multiple spaces or tabs within a single line collapse to one space.
    Substring matching downstream needs whitespace-stable input."""
    assert normalize_comments("happy   birthday    mike") == "happy birthday mike"
    assert normalize_comments("happy\tbirthday\t\tmike") == "happy birthday mike"


def test_normalize_trims_leading_and_trailing_whitespace() -> None:
    assert normalize_comments("  hello  ") == "hello"


def test_normalize_trims_per_line() -> None:
    """Each line in a multi-line comment is independently trimmed."""
    raw = "  line one  \n  line two  "
    assert normalize_comments(raw) == "line one\nline two"


def test_normalize_preserves_line_breaks() -> None:
    """The on-disk shape promises multi-line entries are joined with a
    single newline (see GeminiPageResult.comments_raw). Normalize keeps
    those line breaks intact — they carry structural meaning."""
    raw = "happy bday mike\nrequest: stereolab - ping pong"
    assert normalize_comments(raw) == "happy bday mike\nrequest: stereolab - ping pong"


def test_normalize_drops_blank_interior_lines() -> None:
    """Blank lines between content lines collapse — they don't carry the
    same structural meaning as content-bearing lines, and downstream
    per-line iteration shouldn't have to filter empties."""
    raw = "happy bday mike\n\n\nrequest: stereolab"
    assert normalize_comments(raw) == "happy bday mike\nrequest: stereolab"


def test_normalize_is_idempotent() -> None:
    """`normalize(normalize(x)) == normalize(x)` for any input. A bug
    where we ever introduced a state-dependent transform would regress
    this. Parameterized across the inputs the other tests exercise."""
    inputs = [
        None,
        "",
        "   ",
        "happy   birthday",
        "  line one  \n  line two  ",
        "happy bday mike\n\n\nrequest: stereolab",
        "single",
    ]
    for raw in inputs:
        once = normalize_comments(raw)
        twice = normalize_comments(once)
        assert twice == once, f"not idempotent for {raw!r}: {once!r} -> {twice!r}"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("single", "single"),
        ("  hello  ", "hello"),
        ("a   b", "a b"),
        ("line one\nline two", "line one\nline two"),
        ("  line one  \n  line two  ", "line one\nline two"),
    ],
)
def test_normalize_parametrized(raw: str | None, expected: str | None) -> None:
    assert normalize_comments(raw) == expected


# ---------------------------------------------------------------------------
# find_comment_entry_overlaps
# ---------------------------------------------------------------------------


def _page(
    *,
    comments_raw: str | None,
    entries_by_position: dict[QuadrantPosition, list[Entry]] | None = None,
) -> PageResult:
    """Build a synthetic PageResult for overlap tests.

    Always emits all four quadrants in canonical order so the
    `GeminiPageResult` validator is happy. Entries default to empty per
    quadrant unless `entries_by_position` supplies them.
    """
    entries_by_position = entries_by_position or {}
    quadrants = [
        Quadrant(
            position=pos,
            entries=entries_by_position.get(pos, []),
        )
        for pos in ("top_left", "top_right", "bottom_left", "bottom_right")
    ]
    return PageResult(
        page_date_raw=None,
        quadrants=quadrants,
        comments_raw=comments_raw,
        oddities=[],
        model_version="test-model",
        extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _entry(row_index: int, raw_text: str) -> Entry:
    return Entry(row_index=row_index, raw_text=raw_text, confidence="high")


def test_overlaps_none_comments_returns_empty() -> None:
    page = _page(comments_raw=None)
    assert find_comment_entry_overlaps(page) == []


def test_overlaps_empty_after_normalize_returns_empty() -> None:
    """A whitespace-only `comments_raw` normalizes to None — no overlap
    work to do."""
    page = _page(comments_raw="   \n  \n  ")
    assert find_comment_entry_overlaps(page) == []


def test_overlaps_no_match_returns_empty() -> None:
    page = _page(
        comments_raw="happy bday mike",
        entries_by_position={
            "top_left": [_entry(0, "JUANA MOLINA - LA PARADOJA")],
        },
    )
    assert find_comment_entry_overlaps(page) == []


def test_overlaps_single_match_surfaces_position_and_row() -> None:
    """A single comment line that also appears in an entry's raw_text
    on the same page is the canonical duplication case."""
    page = _page(
        comments_raw="stereolab - ping pong",
        entries_by_position={
            "top_right": [
                _entry(0, "JUANA MOLINA - LA PARADOJA"),
                _entry(1, "STEREOLAB - PING PONG"),
            ],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    overlap = overlaps[0]
    assert isinstance(overlap, CommentEntryOverlap)
    assert overlap.position == "top_right"
    assert overlap.row_index == 1
    assert overlap.matched_text == "stereolab - ping pong"
    assert overlap.comment_line_raw == "stereolab - ping pong"
    assert overlap.entry_raw_text == "STEREOLAB - PING PONG"


def test_overlaps_carry_verbatim_comment_line() -> None:
    """`comment_line_raw` is the verbatim line from `page.comments_raw` —
    parallel to `entry_raw_text` on the entry side. A future audit CLI
    needs the original case AND any quirky whitespace the DJ wrote, so
    the matched record carries both the casefolded form used for matching
    AND the verbatim line that triggered it. Without this field, a
    consumer that wants to display the comment exactly as it appears on
    the page would have nothing to show."""
    page = _page(
        comments_raw="  STEREOLAB - Ping Pong  ",
        entries_by_position={
            "top_left": [_entry(0, "stereolab - ping pong")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    overlap = overlaps[0]
    # `matched_text` is the casefolded, whitespace-collapsed form used
    # for matching.
    assert overlap.matched_text == "stereolab - ping pong"
    # `comment_line_raw` is the verbatim line from `page.comments_raw`,
    # whitespace and case preserved.
    assert overlap.comment_line_raw == "  STEREOLAB - Ping Pong  "


def test_overlaps_carry_per_line_verbatim_for_multi_line_comments() -> None:
    """`comment_line_raw` is per-line, not the whole comments band. A
    multi-line comments band where only one line duplicates a grid entry
    should surface only that line's verbatim form — not the whole blob."""
    page = _page(
        comments_raw="HAPPY BDAY MIKE\nSTEREOLAB - PING PONG\nrequest line @ 962-7768",
        entries_by_position={
            "top_left": [_entry(2, "stereolab - ping pong")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    assert overlaps[0].comment_line_raw == "STEREOLAB - PING PONG"


def test_overlaps_case_insensitive() -> None:
    """Match is case-insensitive; DJs scribble in mixed case."""
    page = _page(
        comments_raw="STEREOLAB - PING PONG",
        entries_by_position={
            "top_left": [_entry(0, "stereolab - ping pong")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    assert overlaps[0].entry_raw_text == "stereolab - ping pong"


def test_overlaps_whitespace_insensitive() -> None:
    """The matcher collapses whitespace on both sides before comparing —
    a comment with weird inner spacing should still match an entry that
    doesn't, and vice versa."""
    page = _page(
        comments_raw="stereolab    -   ping  pong",
        entries_by_position={
            "bottom_left": [_entry(0, "STEREOLAB - PING PONG")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    assert overlaps[0].position == "bottom_left"


def test_overlaps_multi_line_one_matches() -> None:
    """Multi-line `comments_raw` is matched per-line — a comments band
    that has both a dedication and a song title should surface only
    the song-title line when that line duplicates a grid entry."""
    page = _page(
        comments_raw="happy bday mike\nstereolab - ping pong",
        entries_by_position={
            "top_left": [_entry(2, "STEREOLAB - PING PONG")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    assert overlaps[0].matched_text == "stereolab - ping pong"
    assert overlaps[0].row_index == 2


def test_overlaps_multiple_entries_match_same_line() -> None:
    """If the same comment line happens to substring-match more than one
    entry on the page, surface one record per (substring, entry) pair —
    callers want to see every duplicate, not just the first."""
    page = _page(
        comments_raw="stereolab",
        entries_by_position={
            "top_left": [_entry(0, "STEREOLAB - PING PONG")],
            "bottom_right": [_entry(0, "STEREOLAB - JENNY ONDIOLINE")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 2
    positions = {o.position for o in overlaps}
    assert positions == {"top_left", "bottom_right"}


def test_overlaps_substring_match_not_full_line() -> None:
    """The comment line is the needle; the entry's raw_text is the
    haystack. A short comment that's a substring of a longer entry row
    should match."""
    page = _page(
        comments_raw="ping pong",
        entries_by_position={
            "top_left": [_entry(0, "STEREOLAB - PING PONG")],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert len(overlaps) == 1
    assert overlaps[0].matched_text == "ping pong"


def test_overlaps_preserves_quadrant_canonical_order() -> None:
    """Multiple matches across quadrants are returned in canonical
    quadrant order, with row_index ascending within each quadrant.
    Stable ordering keeps diagnostic output diffable across runs."""
    page = _page(
        comments_raw="stereolab",
        entries_by_position={
            "bottom_right": [_entry(3, "STEREOLAB - LATE")],
            "top_left": [
                _entry(0, "STEREOLAB - EARLY"),
                _entry(2, "STEREOLAB - MIDDLE"),
            ],
        },
    )
    overlaps = find_comment_entry_overlaps(page)
    assert [(o.position, o.row_index) for o in overlaps] == [
        ("top_left", 0),
        ("top_left", 2),
        ("bottom_right", 3),
    ]
