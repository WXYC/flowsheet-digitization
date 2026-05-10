"""Tests for `core.continuations.merge_continuations`."""

from __future__ import annotations

from core.continuations import merge_continuations
from core.schema import Entry


def _entry(
    row_index: int,
    raw_text: str,
    *,
    notes: str | None = None,
    oddities: list[str] | None = None,
) -> Entry:
    return Entry(
        row_index=row_index,
        raw_text=raw_text,
        confidence="high",
        notes=notes,
        oddities=list(oddities) if oddities else [],
    )


def test_empty_list_returns_empty() -> None:
    assert merge_continuations([]) == []


def test_no_continuations_returns_input_unchanged() -> None:
    entries = [
        _entry(0, "JUANA MOLINA - LA PARADOJA"),
        _entry(1, "STEREOLAB - PING PONG"),
    ]
    out = merge_continuations(entries)
    assert [e.raw_text for e in out] == [
        "JUANA MOLINA - LA PARADOJA",
        "STEREOLAB - PING PONG",
    ]


def test_single_continuation_folds_into_prior() -> None:
    entries = [
        _entry(0, "DUKE ELLINGTON & JOHN COLTRANE - IN A"),
        _entry(1, "SENTIMENTAL MOOD", notes="continuation"),
    ]
    out = merge_continuations(entries)
    assert len(out) == 1
    assert out[0].raw_text == "DUKE ELLINGTON & JOHN COLTRANE - IN A SENTIMENTAL MOOD"
    # The merged row keeps the predecessor's row_index — the merge does
    # not invent a new position on the page.
    assert out[0].row_index == 0


def test_chained_continuations_fold_in_order() -> None:
    """Two-row wrap: the model emitted three rows for one handwritten
    entry. Merging produces a single row with all three texts joined."""
    entries = [
        _entry(0, "CHUQUIMAMANI-CONDORI - CALL"),
        _entry(1, "YOUR NAME (FROM EDITS,", notes="continuation"),
        _entry(2, "self-released)", notes="continuation"),
    ]
    out = merge_continuations(entries)
    assert len(out) == 1
    assert out[0].raw_text == "CHUQUIMAMANI-CONDORI - CALL YOUR NAME (FROM EDITS, self-released)"


def test_continuation_as_first_entry_is_preserved() -> None:
    """No prior entry to fold into — keep the row as-is so the matching
    layer can still see the text. Silently dropping it would lose the
    only OCR signal we have for that row."""
    entries = [
        _entry(0, "(continued from page above)", notes="continuation"),
        _entry(1, "JESSICA PRATT - BACK BABY"),
    ]
    out = merge_continuations(entries)
    assert len(out) == 2
    assert out[0].notes == "continuation"
    assert out[1].notes is None


def test_continuation_collapses_internal_whitespace() -> None:
    """The continuation row's leading whitespace and the predecessor's
    trailing whitespace must collapse into a single separating space.
    Without this, downstream substring matching would fail on rows where
    the model emitted ragged whitespace at the wrap boundary."""
    entries = [
        _entry(0, "ARTIST NAME -  "),
        _entry(1, "  TRACK", notes="continuation"),
    ]
    out = merge_continuations(entries)
    assert out[0].raw_text == "ARTIST NAME - TRACK"


def test_double_height_and_crossed_out_left_alone() -> None:
    """Only `notes="continuation"` triggers the merge. Other tags
    document the row in place and the row should remain its own entry."""
    entries = [
        _entry(0, "JUANA MOLINA - LA PARADOJA"),
        _entry(1, "(handwriting spans 2 grid rows)", notes="double_height"),
        _entry(2, "DROPPED TRACK", notes="crossed_out"),
        _entry(3, "STEREOLAB - PING PONG"),
    ]
    out = merge_continuations(entries)
    assert len(out) == 4
    assert [e.notes for e in out] == [None, "double_height", "crossed_out", None]


def test_oddities_preserved_through_merge() -> None:
    """Row-level oddities on the continuation row must be concatenated
    onto the predecessor — they describe a real annotation on the page
    and dropping them would lose data."""
    entries = [
        _entry(0, "ARTIST -", oddities=["asterisk in right margin"]),
        _entry(
            1,
            "TRACK",
            notes="continuation",
            oddities=["arrow drawn from this row to the row above"],
        ),
    ]
    out = merge_continuations(entries)
    assert out[0].oddities == [
        "asterisk in right margin",
        "arrow drawn from this row to the row above",
    ]


def test_input_entries_not_mutated() -> None:
    """Pure function: re-running on the same input gives the same result.
    A bug where we mutated `merged[-1].oddities` in place could regress
    this silently."""
    entries = [
        _entry(0, "ARTIST -"),
        _entry(1, "TRACK", notes="continuation"),
    ]
    snapshot = [e.model_copy(deep=True) for e in entries]
    merge_continuations(entries)
    assert entries == snapshot


def test_continuation_after_double_height_folds_into_double_height() -> None:
    """Defensive: a continuation following any non-continuation row is
    a wrap of THAT row, regardless of its notes tag. This includes
    double_height (a row that physically spans 2 grid rows but is one
    handwritten entry) — a wrap can still happen on the line below."""
    entries = [
        _entry(0, "VERY LONG ARTIST NAME WRITTEN BIG -", notes="double_height"),
        _entry(1, "A TRACK", notes="continuation"),
    ]
    out = merge_continuations(entries)
    assert len(out) == 1
    assert out[0].raw_text == "VERY LONG ARTIST NAME WRITTEN BIG - A TRACK"
    # The predecessor's `notes` is preserved — the entry is still a
    # double-height row, just with the wrapped text folded in.
    assert out[0].notes == "double_height"
