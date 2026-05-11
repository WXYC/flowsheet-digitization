#!/usr/bin/env python3
"""spot_check_discogs.py — sanity-check Gemini-extracted entries against the
local Discogs PostgreSQL cache.

Walks every `data/results/**/*.json` and for each entry with a
derivable artist (from `artist_guess` on the 34 legacy JSONs, or from
`parse_artist_track(raw_text)` on post-schema-trim extractions) checks
two questions against the cache:

  * artist  — is this artist name in the WXYC library? (broad, cheap)
  * joint   — does any release for this WXYC-owned artist contain a
              track with this title?

Both queries run against pre-filtered tables that scope the work to
WXYC-owned artists:

    wxyc_library_artist     — 26K artist names (already cached)
    wxyc_track              — materialized view of (artist, track) pairs
                              for releases credited to WXYC-owned artists

If `wxyc_track` does not exist, build it once with:

    psql -h localhost -p 5432 -d discogs -f scripts/build_wxyc_track_mv.sql

Hits are positive signal that a transcription is plausibly real. Misses
are weak signal — DJ shorthand ("Dylan", "S Wonder") and personal
records WXYC never owned will both miss. Don't read this as accuracy.

The pure data plumbing (entry collection, sharding, hit/miss accounting,
ranking) lives in `core/spot_check.py` and is unit-tested. This file is
the CLI + DB-IO wrapper.

Install (one-time):
    pip install -e ".[analysis]"

Run:
    .venv/bin/python scripts/spot_check_discogs.py
    .venv/bin/python scripts/spot_check_discogs.py --workers 8 --top 20
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

# Allow `.../scripts/spot_check_discogs.py` to import the project's `core`
# package without requiring the script to be run as `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.spot_check import (  # noqa: E402
    PageReport,
    collect_entries,
    evaluate,
    joint_miss_pairs,
    rank_by_joint_miss,
    shard,
)

DEFAULT_DSN = "postgresql://localhost:5432/discogs"
DEFAULT_WORKERS = 8


# -- queries ---------------------------------------------------------------
#
# Both tables are pre-normalized (lower + f_unaccent) so the input is too.
# wxyc_library_artist.norm_name and wxyc_track.{artist_norm, track_norm}
# all carry the normalized form. Keeps the script's normalization logic
# in one place.


_ARTIST_SQL = """
SELECT 1 FROM wxyc_library_artist
WHERE norm_name = lower(f_unaccent(%s))
LIMIT 1
"""

_JOINT_SQL = """
SELECT 1 FROM wxyc_track
WHERE artist_norm = lower(f_unaccent(%s))
  AND track_norm  = lower(f_unaccent(%s))
LIMIT 1
"""


def _artist_lookup(dsn: str, artists: list[str]) -> dict[str, bool]:
    """Run artist queries on a single connection. Called once per worker."""
    out: dict[str, bool] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for a in artists:
            cur.execute(_ARTIST_SQL, (a,))
            out[a] = cur.fetchone() is not None
    return out


def _joint_lookup(dsn: str, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        for a, t in pairs:
            cur.execute(_JOINT_SQL, (a, t))
            out[(a, t)] = cur.fetchone() is not None
    return out


def parallel_lookups(
    dsn: str,
    artists: list[str],
    pairs: list[tuple[str, str]],
    *,
    workers: int,
) -> tuple[dict[str, bool], dict[tuple[str, str], bool]]:
    """Resolve every unique artist + (artist, track) once across `workers` connections.

    Each worker opens its own connection; psycopg connections are not
    thread-safe so we do not share. Sharding is round-robin sized; LIMIT 1
    + indexed equality means per-query work is uniform enough that we
    don't need work-stealing.
    """
    artist_results: dict[str, bool] = {}
    joint_results: dict[tuple[str, str], bool] = {}

    artist_shards = shard(artists, workers)
    joint_shards = shard(pairs, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        artist_futures = [pool.submit(_artist_lookup, dsn, s) for s in artist_shards]
        joint_futures = [pool.submit(_joint_lookup, dsn, s) for s in joint_shards]
        for af in artist_futures:
            artist_results.update(af.result())
        for jf in joint_futures:
            joint_results.update(jf.result())

    return artist_results, joint_results


# -- printing --------------------------------------------------------------


def print_report(
    pages: dict[Path, PageReport],
    *,
    results_root: Path,
    top_misses: int,
) -> None:
    if not pages:
        print("No completed result JSONs found.")
        return

    total = sum(p.total for p in pages.values())
    with_track = sum(p.with_track for p in pages.values())
    artist_hit_total = sum(p.artist_hits for p in pages.values())
    joint_hit_total = sum(p.joint_hits for p in pages.values())

    print()
    print("=" * 72)
    print(f"Pages:                          {len(pages)}")
    print(f"Entries with derivable artist:  {total}")
    print(f"  artist hits:                  {artist_hit_total}/{total}")
    print(f"Entries with artist + track:    {with_track}")
    print(f"  joint hits (artist+track):    {joint_hit_total}/{with_track}")
    print("=" * 72)

    print()
    print(f"Top {top_misses} pages by joint miss count:")
    print(f"  {'page':<58}  artist  joint")
    for p in rank_by_joint_miss(pages)[:top_misses]:
        rel = p.page_path.relative_to(results_root)
        artist_miss = p.total - p.artist_hits
        joint_miss = len(p.joint_misses)
        print(
            f"  {str(rel):<58}  {artist_miss:>2}/{p.total:<2}   {joint_miss:>2}/{p.with_track:<2}"
        )

    print()
    print(f"Top {top_misses} most-frequent joint-miss (artist, track) pairs:")
    for (a, t), n in joint_miss_pairs(pages).most_common(top_misses):
        print(f"  {n:>3}  {a} - {t}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "./data"),
        type=Path,
        help="Pipeline output root (default: $DATA_ROOT or ./data).",
    )
    parser.add_argument(
        "--dsn",
        default=os.environ.get("DISCOGS_DSN", DEFAULT_DSN),
        help=f"PostgreSQL DSN for the Discogs cache (default: {DEFAULT_DSN}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel DB connections (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many rows to print in the page-rank and miss-pair tables (default: 20).",
    )
    args = parser.parse_args(argv)

    results_root = (args.data_root / "results").expanduser().resolve()
    if not results_root.is_dir():
        print(f"results dir not found: {results_root}", file=sys.stderr)
        return 1

    rows = collect_entries(results_root)
    if not rows:
        print(f"No entries with a derivable artist under {results_root}.")
        return 0

    unique_artists = sorted({r.artist for r in rows})
    unique_pairs = sorted({(r.artist, r.track) for r in rows if r.track is not None})

    print(
        f"Loaded {len(rows)} entries → "
        f"{len(unique_artists)} unique artists, {len(unique_pairs)} unique pairs",
        file=sys.stderr,
    )
    print(f"Querying {args.dsn} with {args.workers} workers ...", file=sys.stderr)

    artist_hits, joint_hits = parallel_lookups(
        args.dsn, unique_artists, unique_pairs, workers=args.workers
    )

    pages = evaluate(rows, artist_hits, joint_hits)
    print_report(pages, results_root=results_root, top_misses=args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
