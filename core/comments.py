"""Phase-2 read-time normalization and duplication diagnostics for `comments_raw`.

The Phase-1 prompt captures the bottom-of-page Comments field verbatim
into `GeminiPageResult.comments_raw` (see `core/schema.py`). Verbatim is
right for ingestion but awkward to consume: whitespace is ragged, and
the same handwritten text sometimes appears in both the comments band
AND inside one of the four hour-quadrants (e.g. a DJ scribbles a song
title in the comments band that they also wrote into the grid).

This module mirrors the shape of `core.continuations`: pure read-time
functions, no I/O, no on-disk mutation. The on-disk `comments_raw`
stays verbatim per the org data-safety rule. Callers compute the
normalized form at read time.

Two primitives:

  * `normalize_comments(raw)` — collapses internal whitespace per line,
    trims each line, drops empty interior lines, returns None if the
    result is empty. Idempotent.
  * `find_comment_entry_overlaps(page)` — diagnostic: returns one record
    per (comment line, entry) pair where a normalized comment line is a
    substring of an entry's `raw_text` after case- and
    whitespace-insensitive matching. Empty list when there's nothing to
    flag.

No categorization here — the ticket explicitly defers that to a later
pass once we have a corpus of filled-in comments bands to look at.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, NonNegativeInt

from core.schema import PageResult, QuadrantPosition

# Multiple runs of any whitespace character on a single line collapse to
# one ASCII space. Newlines are handled separately (line breaks carry
# structural meaning — see the `comments_raw` field description).
_INTRA_LINE_WHITESPACE = re.compile(r"[^\S\n]+")


def normalize_comments(raw: str | None) -> str | None:
    """Return a read-time normalized form of `comments_raw`.

    Pure and idempotent. None passes through. The transform:

      * splits on newlines so multi-line entries stay multi-line,
      * collapses runs of intra-line whitespace (spaces, tabs, etc.) to
        a single space within each line,
      * trims leading/trailing whitespace on each line,
      * drops lines that are empty after trimming — blank interior lines
        are noise, and downstream per-line iteration shouldn't have to
        filter them,
      * returns None when the result is empty, so callers have one
        sentinel for "nothing useful here" instead of having to also
        check for ``""``.

    The verbatim on-disk shape is preserved; callers run this at read
    time when they want a cleaner form.
    """
    if raw is None:
        return None
    lines = [_INTRA_LINE_WHITESPACE.sub(" ", line).strip() for line in raw.split("\n")]
    kept = [line for line in lines if line]
    if not kept:
        return None
    return "\n".join(kept)


class CommentEntryOverlap(BaseModel):
    """One (comment line, entry) duplication record.

    Surfaces the case where text in the bottom comments band also
    appears inside a grid entry on the same page. The fields are the
    minimum a caller (calibration scorer, future audit CLI) needs to
    locate the duplicate on the page and inspect both sides.
    """

    matched_text: str
    """The normalized comments substring that matched. Lowercase,
    whitespace-collapsed — this is the form the matcher compared, not
    the verbatim on-disk text."""

    position: QuadrantPosition
    """Which quadrant the matched entry lives in."""

    row_index: NonNegativeInt
    """0-based row position of the matched entry within its quadrant."""

    entry_raw_text: str
    """The full `raw_text` of the matched entry, verbatim from disk —
    so a caller can show context around the substring match."""


def find_comment_entry_overlaps(page: PageResult) -> list[CommentEntryOverlap]:
    """Diagnose duplication between `comments_raw` and grid entries.

    Returns one record per (normalized comment line, matching entry)
    pair where the comment line is a substring of an entry's
    ``raw_text`` after both sides are normalized (lowercased and
    whitespace-collapsed). Empty list when:

      * ``page.comments_raw`` is None,
      * normalization yields None (whitespace-only comments),
      * no entry contains any comment line.

    Matching is **per-line**, not against the whole normalized comments
    blob. A two-line comments band with one dedication and one song
    title surfaces only the song-title line if just that line
    duplicates a grid row — the unrelated dedication doesn't pollute
    the diagnostic.

    Records come back in canonical quadrant order (top_left, top_right,
    bottom_left, bottom_right) with ascending ``row_index`` within each
    quadrant. Stable ordering keeps diagnostic output diffable across
    runs.

    Pure: no mutation of the input ``page``, no I/O.
    """
    normalized = normalize_comments(page.comments_raw)
    if normalized is None:
        return []

    # Case-insensitive matching is symmetric — fold both sides to the
    # same case so a comments band scribbled in caps still matches an
    # entry transcribed in lower case (and vice versa).
    needles = [line.casefold() for line in normalized.split("\n") if line]
    if not needles:
        return []

    overlaps: list[CommentEntryOverlap] = []
    for quadrant in page.quadrants:
        for entry in sorted(quadrant.entries, key=lambda e: e.row_index):
            entry_normalized = _normalize_for_match(entry.raw_text)
            if not entry_normalized:
                continue
            for needle in needles:
                if needle in entry_normalized:
                    overlaps.append(
                        CommentEntryOverlap(
                            matched_text=needle,
                            position=quadrant.position,
                            row_index=entry.row_index,
                            entry_raw_text=entry.raw_text,
                        )
                    )
    return overlaps


def _normalize_for_match(text: str) -> str:
    """Lowercased, whitespace-collapsed form used for substring matching.

    Mirrors what `normalize_comments` does to each line, plus lowercase.
    Kept separate from `normalize_comments` because the entry side
    doesn't need the empty-to-None convention — entries always have
    non-empty `raw_text` by schema.
    """
    return _INTRA_LINE_WHITESPACE.sub(" ", text).strip().casefold()
