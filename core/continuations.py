"""Phase-2 read-time merging of `notes="continuation"` rows.

The Phase-1 prompt asks Gemini to TAG continuation rows (a row whose
handwriting is a wrap of the previous line) without merging them — the
on-disk shape preserves every grid row verbatim. The matching layer
wants the cleaner list of distinct songs, so we merge at read-time.

Why not merge at write-time:

* Once a continuation tag is gone from the on-disk JSON, a future
  re-OCR queue (deferred to a later phase) cannot tell which rows had
  the wrap pattern without re-running the model.
* The org data-safety rule applies to the existing `data/results/`
  corpus: don't rewrite successful extractions. A read-time merge
  costs nothing and is reversible.

The function is pure. Pass it the per-quadrant `entries` list; get back
a new list with continuations folded into their predecessor's
`raw_text` and dropped from the result.
"""

from __future__ import annotations

from core.schema import Entry


def merge_continuations(entries: list[Entry]) -> list[Entry]:
    """Fold `notes="continuation"` rows into the prior entry's `raw_text`.

    Pure: returns a new list; original entries are not mutated. Each
    continuation's `raw_text` is appended to the predecessor with a
    single space separator (collapsing any trailing/leading whitespace).
    The continuation's `oddities` are concatenated onto the predecessor's
    so we don't lose row-level annotations on the dropped row.

    Edge case: a continuation that is the FIRST entry has nothing to
    fold into. It's preserved as-is (the consumer can still notice it
    via the `notes` tag) — silently dropping it would be worse, since
    the raw OCR text is the only signal we have for that row.
    """
    merged: list[Entry] = []
    for entry in entries:
        if entry.notes == "continuation" and merged:
            prior = merged[-1]
            joined = f"{prior.raw_text.rstrip()} {entry.raw_text.lstrip()}".strip()
            merged[-1] = prior.model_copy(
                update={
                    "raw_text": joined,
                    "oddities": [*prior.oddities, *entry.oddities],
                }
            )
        else:
            merged.append(entry)
    return merged
