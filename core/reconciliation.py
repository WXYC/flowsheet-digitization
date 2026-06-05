"""Phase-2 reconciliation: rewrite Gemini-emitted artists against the WXYC catalog.

Pure function `reconcile(page, *, lml, threshold)` walks `PageResult`,
parses each row's `raw_text` into `(artist, track)` via `core.parse`,
asks LML to fuzzy-match each artist, and rewrites the row's `raw_text`
when LML's correction agrees with the input above the chosen
similarity threshold. Below-threshold corrections produce a
`FlaggedRow` so a human can review.

Design notes
------------

* **Artist-only.** Track text stays as Gemini emitted it. Track
  verification would require Discogs (rate-limited, slow). v1 ships
  artist correction; v2 can extend without changing this module's
  public surface.

* **Non-Latin caveat.** LML's internal `to_match_form` strips
  diacritics and non-Latin characters. Inputs like "Sigur Rós" or
  "Stereolab" with diacritical marks will round-trip OK because the
  catalog stores them the same way, but artists with non-Latin scripts
  (Cyrillic, CJK) degrade silently. Measured non-Latin coverage on the
  Phase-2 corpus is small but non-zero — caller is responsible for
  treating low-recall pages with extra suspicion.

* **Idempotency.** The pure function takes no clock and no
  randomness. Running `reconcile(reconcile(p))` on stable input
  produces the same `raw_text` on every entry (a correction that's
  already canonical is a no-op; LML's `corrected_artist == input`
  branch leaves the row untouched).

* **Threshold semantics.** `score >= threshold` applies. The score is
  `rapidfuzz.fuzz.token_set_ratio((lower(corrected), lower(input)))`
  in 0..100. The default threshold is tuned offline by
  `scripts/tune_reconciliation_threshold.py` against Alex's verified
  corpus and baked into the CLI.

* **5xx policy.** This module does not catch `LMLError`; it bubbles up
  to the CLI, which logs and lets the caller decide. The bulk client
  itself wraps batch-level failures separately from per-item ones —
  per-item errors come back as `LMLBulkItemResult(status='error')`,
  which we treat as skip (no correction, no flag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from rapidfuzz import fuzz

from core.lml_client import LMLBulkItemResult
from core.parse import parse_artist_track
from core.schema import Entry, PageResult, Quadrant, QuadrantPosition

# Workflow artifact: Alex types the literal "blank" as a placeholder for
# rows that are physically blank on the paper (65 occurrences across n=20
# verified pages). Treat it as a non-artist sentinel so we don't fuzzy-
# match it against the catalog and waste a slot in the bulk batch.
_BLANK_PLACEHOLDERS = frozenset({"blank"})


def _is_blank_placeholder(artist: str) -> bool:
    return artist.strip().lower() in _BLANK_PLACEHOLDERS


class LMLClientProtocol(Protocol):
    """Subset of `core.lml_client.LMLClient` the reconciler uses.

    A Protocol (vs the concrete class) so tests can pass a stub without
    inheriting from `LMLClient`. Mirrors the pattern in
    `core.golden` and keeps the dependency arrow pointing inward.
    """

    async def bulk_lookup(self, items: list[dict[str, Any]]) -> list[LMLBulkItemResult]: ...


@dataclass(frozen=True)
class FlaggedRow:
    """A row where LML's suggested correction fell below the auto-accept
    threshold.

    Persisted alongside the corrected `PageResult` so a human can
    review the borderline cases. NOT shoved into a schema column —
    schema discipline keeps `PageResult.Entry` stable and
    correction-state lives next to the JSON, not inside it.
    """

    quadrant: QuadrantPosition
    row_index: int
    original_artist: str
    suggested_artist: str
    score: int
    raw_text: str


async def reconcile(
    page: PageResult,
    *,
    lml: LMLClientProtocol,
    threshold: int,
) -> tuple[PageResult, list[FlaggedRow]]:
    """Reconcile `page` against the WXYC library catalog via LML.

    Returns `(corrected_page, flagged_rows)`:

    * `corrected_page` has `Entry.raw_text` rewritten on rows where
      LML's `corrected_artist` matched the original at `score >=
      threshold`. Every other field is byte-identical to `page`.
    * `flagged_rows` collects below-threshold suggestions so the caller
      can write them out for human review.

    Pure-ish: depends only on `lml.bulk_lookup`, which is itself
    idempotent over its inputs. Same input → same output, with the
    caveat that LML's catalog grows over time.

    Non-Latin caveat: see module docstring.
    """
    # Walk in (quadrant_order × row_index) order so the index in LML's
    # response maps cleanly back to a (quadrant, row) coordinate.
    targets: list[tuple[QuadrantPosition, int, str]] = []
    for q in page.quadrants:
        for i, entry in enumerate(q.entries):
            artist, _track = parse_artist_track(entry.raw_text)
            if artist is None:
                continue
            if _is_blank_placeholder(artist):
                continue
            targets.append((q.position, i, artist))

    if not targets:
        return page, []

    items = [{"artist": artist} for _, _, artist in targets]
    results = await lml.bulk_lookup(items)

    # Build correction map keyed by (quadrant_position, entry-index-in-quadrant).
    corrections: dict[tuple[QuadrantPosition, int], str] = {}
    flagged: list[FlaggedRow] = []

    for (position, entry_idx, gemini_artist), result in zip(targets, results, strict=True):
        if result.status != "match":
            # `no_match` or `error` — leave the row alone, no flag.
            continue
        suggested = result.corrected_artist
        if suggested is None:
            continue
        if suggested == gemini_artist:
            # Already canonical; rewriting would be a no-op that risks
            # re-tokenizing whitespace.
            continue
        score = int(fuzz.token_set_ratio(suggested.lower(), gemini_artist.lower()))
        if score >= threshold:
            corrections[(position, entry_idx)] = suggested
        else:
            # Find the row's raw_text for the flag (cheap; we already
            # know the coordinates).
            raw_text = _find_raw_text(page, position, entry_idx)
            flagged.append(
                FlaggedRow(
                    quadrant=position,
                    row_index=_find_row_index(page, position, entry_idx),
                    original_artist=gemini_artist,
                    suggested_artist=suggested,
                    score=score,
                    raw_text=raw_text,
                )
            )

    if not corrections:
        return page, flagged

    # Apply corrections by rebuilding the affected quadrants / entries.
    new_quadrants: list[Quadrant] = []
    for q in page.quadrants:
        if not any((q.position, i) in corrections for i in range(len(q.entries))):
            new_quadrants.append(q)
            continue
        new_entries: list[Entry] = []
        for i, entry in enumerate(q.entries):
            suggested = corrections.get((q.position, i))
            if suggested is None:
                new_entries.append(entry)
                continue
            new_raw = _rewrite_artist(entry.raw_text, suggested)
            new_entries.append(entry.model_copy(update={"raw_text": new_raw}))
        new_quadrants.append(q.model_copy(update={"entries": new_entries}))

    corrected = page.model_copy(update={"quadrants": new_quadrants})
    return corrected, flagged


def _find_raw_text(page: PageResult, position: QuadrantPosition, entry_idx: int) -> str:
    for q in page.quadrants:
        if q.position == position:
            return q.entries[entry_idx].raw_text
    raise KeyError(f"no quadrant at position {position!r}")


def _find_row_index(page: PageResult, position: QuadrantPosition, entry_idx: int) -> int:
    for q in page.quadrants:
        if q.position == position:
            return q.entries[entry_idx].row_index
    raise KeyError(f"no quadrant at position {position!r}")


def _rewrite_artist(raw_text: str, new_artist: str) -> str:
    """Replace the artist portion of `raw_text` while keeping the track
    side (and the separator between them) byte-identical.

    Uses `parse_artist_track`'s own separator regex so the split here
    matches the split downstream consumers will run on the read path.
    """
    # Reuse the parse module's regex to find the exact separator span.
    from core.parse import _SEPARATOR

    match = _SEPARATOR.search(raw_text)
    if match is None:
        # No separator (artist-only row). Replace the whole thing,
        # preserving leading/trailing whitespace so the rewrite is a
        # surgical artist swap, not a strip.
        leading = raw_text[: len(raw_text) - len(raw_text.lstrip())]
        trailing = raw_text[len(raw_text.rstrip()) :]
        return f"{leading}{new_artist}{trailing}"
    # Keep everything from the separator onward verbatim.
    leading = raw_text[: len(raw_text) - len(raw_text.lstrip())]
    return f"{leading}{new_artist}{raw_text[match.start() :]}"
