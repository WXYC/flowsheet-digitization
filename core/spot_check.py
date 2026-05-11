"""Pure data plumbing for the spot-check tool.

Deterministic and dependency-free — no Postgres, no network, just JSON
reading and in-memory hit/miss accounting. The CLI + DB-IO half is a
separate module.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.parse import parse_artist_track


@dataclass(frozen=True)
class EntryRef:
    """One row from a PageResult, kept just enough to query and report."""

    page_path: Path
    quadrant: str
    row_index: int
    artist: str
    track: str | None


@dataclass
class PageReport:
    """Hit/miss accounting for a single result JSON.

    `total` counts entries we could derive an artist for. `with_track` is
    the joint-mode denominator: entries with both an artist and a track.
    The two miss lists hold the rows whose lookup returned False, so a
    caller can drill into specific transcriptions.
    """

    page_path: Path
    total: int = 0
    with_track: int = 0
    artist_hits: int = 0
    joint_hits: int = 0
    artist_misses: list[EntryRef] = field(default_factory=list)
    joint_misses: list[EntryRef] = field(default_factory=list)


def collect_entries(results_root: Path) -> list[EntryRef]:
    """Walk every result JSON under `results_root` and return queryable rows.

    Two on-disk shapes coexist:

      * Pre-audit (34 legacy corpus JSONs): `artist_guess` / `track_guess`
        are present on every entry. A null `artist_guess` is the explicit
        "continuation row" sentinel — skip those.
      * Post-audit (everything new): no `artist_guess` / `track_guess`
        keys. Artist and track are derived from `raw_text` via
        `core.parse.parse_artist_track`.

    The branch is on KEY PRESENCE, not value truthiness, so the legacy
    "explicit null means skip" contract is preserved on legacy rows
    while new rows get the deterministic parse. An empty track becomes
    `None` so callers can distinguish "no track to check" from "track
    is the empty string".
    """
    rows: list[EntryRef] = []
    for path in sorted(results_root.rglob("*.json")):
        data = json.loads(path.read_text())
        for q in data.get("quadrants", []):
            for e in q.get("entries", []):
                if "artist_guess" in e:
                    legacy = (e.get("artist_guess") or "").strip()
                    if not legacy:
                        continue
                    artist = legacy
                    track = (e.get("track_guess") or "").strip() or None
                else:
                    parsed_artist, track = parse_artist_track(e.get("raw_text"))
                    if parsed_artist is None:
                        continue
                    artist = parsed_artist
                rows.append(
                    EntryRef(
                        page_path=path,
                        quadrant=q["position"],
                        row_index=e["row_index"],
                        artist=artist,
                        track=track,
                    )
                )
    return rows


def shard(items: list, n: int) -> list[list]:
    """Split `items` into up to `n` roughly-equal shards.

    Drops empty shards: if `items` has fewer elements than `n`, the
    return has `len(items)` shards rather than `n`. The downstream
    ThreadPoolExecutor doesn't care, and dropping nulls keeps `for f
    in futures` from including no-op work.
    """
    if not items:
        return []
    size, extra = divmod(len(items), n)
    shards: list[list] = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        if end > start:
            shards.append(items[start:end])
        start = end
    return shards


def evaluate(
    rows: list[EntryRef],
    artist_hits: dict[str, bool],
    joint_hits: dict[tuple[str, str], bool],
) -> dict[Path, PageReport]:
    """Roll per-entry hit/miss decisions up into per-page reports.

    `artist_hits` and `joint_hits` are precomputed dicts keyed by the
    same artist / (artist, track) values that appear on `rows` — every
    row's keys must be present (KeyError is intentional: a missing key
    means the lookup-dispatching layer skipped a query, which is a bug).
    """
    pages: dict[Path, PageReport] = {}
    for row in rows:
        report = pages.setdefault(row.page_path, PageReport(page_path=row.page_path))
        report.total += 1

        if artist_hits[row.artist]:
            report.artist_hits += 1
        else:
            report.artist_misses.append(row)

        if row.track is not None:
            report.with_track += 1
            if joint_hits[(row.artist, row.track)]:
                report.joint_hits += 1
            else:
                report.joint_misses.append(row)
    return pages


def rank_by_joint_miss(pages: dict[Path, PageReport]) -> list[PageReport]:
    """Return pages sorted most-missed-first.

    Tiebreak on artist-miss count so two pages with identical joint
    misses don't reorder run-to-run on the same data.
    """
    return sorted(
        pages.values(),
        key=lambda p: (-len(p.joint_misses), -len(p.artist_misses)),
    )


def joint_miss_pairs(pages: dict[Path, PageReport]) -> Counter[tuple[str, str]]:
    """Frequency of `(artist, track)` joint misses across the whole corpus.

    A pair recurring on many pages is more likely to be a systemic
    transcription pattern than a one-off obscure release.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for p in pages.values():
        for row in p.joint_misses:
            if row.track is not None:
                counts[(row.artist, row.track)] += 1
    return counts
