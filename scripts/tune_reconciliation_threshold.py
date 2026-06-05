"""Tune the artist-reconciliation threshold against Alex's verified corpus.

Background
----------

`core.reconciliation.reconcile` accepts a similarity threshold T (0-100,
default tuned here). At T, a row is auto-corrected only when

    rapidfuzz.fuzz.token_set_ratio(
        lower(LML.corrected_artist), lower(gemini_artist)
    ) >= T

This script computes the precision/recall curve of that decision against
Alex's verified.json, picks the smallest T where precision >= 95%, and
writes a CSV of the curve for plotting.

Inputs
------

The script reads each page's `<stem>.corrections.json`, which the
verifier UI writes alongside `<stem>.verified.json`. corrections.json
records *in-place* edits as `(original, corrected)` text pairs keyed by
`(position, row_index, field)`. We use the `field == "raw_text"` entries
only — those are rows Alex re-transcribed without inserting/deleting,
so the alignment is unambiguous.

We deliberately do NOT align bundle.json against verified.json by
row_index. Alex sometimes inserts blank rows or shifts the row_index
sequence; the bundle's row_index for a given visual row drifts from
verified's. corrections.json sidesteps that — it captures the edit at
the exact (row_index, original_text) before the shift happens.

For each `field == "raw_text"` correction:
  1. Parse Gemini's `original` -> (gemini_artist, _track)
  2. Parse Alex's `corrected` -> (alex_artist, _track)
  3. If gemini_artist == alex_artist (artist unchanged), skip — this
     row's change was on the track side and reconciliation can't speak
     to it.
  4. Call LML's bulk endpoint with `{"artist": gemini_artist}`.
  5. Score = token_set_ratio(LML.corrected_artist.lower(),
                              alex_artist.lower())
  6. Record (gemini_artist, alex_artist, corrected_artist, score).

Outputs
-------

  * Stdout: precision/recall table, threshold recommendation, non-Latin
    coverage fraction.
  * `<out-csv>`: machine-readable curve (T, precision, recall,
    auto_accepts, true_positives).

Running
-------

LML's bulk endpoint requires a bearer token in production. Set
`LML_URL` (default: production) and `LML_API_KEY` in env:

    LML_API_KEY=... .venv/bin/python scripts/tune_reconciliation_threshold.py \\
      --corpus data/verifier-pulled-refresh \\
      --out /tmp/reconciliation-curve.csv

Marker: this script touches live LML, so it's analogous to `external_api`
in the marker scheme. Not part of the default test run.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz

# Walk up the worktree to find core/ — the script lives under scripts/ and
# `python scripts/foo.py` doesn't put the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from core.lml_client import (  # noqa: E402
    DEFAULT_LML_URL,
    LMLClient,
    LMLError,
)
from core.parse import parse_artist_track  # noqa: E402
from core.reconciliation import _is_blank_placeholder  # noqa: E402


@dataclass(frozen=True)
class Pair:
    """One (gemini_artist, alex_artist) pair from a single row."""

    stem: str
    quadrant: str
    row_index: int
    gemini_artist: str
    alex_artist: str


@dataclass
class Scored:
    """A pair after LML correction + similarity scoring."""

    pair: Pair
    corrected_artist: str | None
    score: int  # 0..100; 0 when LML returned no correction


# -- Pair extraction ---------------------------------------------------------


def _extract_pairs_from_corrections(stem: str, corrections: dict[str, Any]) -> list[Pair]:
    """Walk a page's corrections.json, yield one Pair per `raw_text`
    edit where Alex's artist differs from Gemini's.

    Uses the corrections.json schema written by the verifier UI: each
    `row_corrections` entry carries `(position, row_index, field,
    original, corrected)`. We only care about `field == "raw_text"` —
    the in-place text edits whose alignment is unambiguous.
    """
    pairs: list[Pair] = []
    for entry in corrections.get("row_corrections", []) or []:
        if entry.get("field") != "raw_text":
            continue
        position = entry.get("position")
        if position not in ("top_left", "top_right", "bottom_left", "bottom_right"):
            continue
        try:
            row_index = int(entry["row_index"])
        except (KeyError, TypeError, ValueError):
            continue
        original = entry.get("original") or ""
        corrected = entry.get("corrected") or ""
        g_artist, _g_track = parse_artist_track(original)
        a_artist, _a_track = parse_artist_track(corrected)
        if g_artist is None or a_artist is None:
            continue
        # Mirror the production path: rows where Alex (or Gemini) used the
        # literal "blank" placeholder would never be sent to LML, so they
        # don't belong in the tuning denominator either. See
        # `core.reconciliation._is_blank_placeholder`.
        if _is_blank_placeholder(g_artist) or _is_blank_placeholder(a_artist):
            continue
        if g_artist.strip().lower() == a_artist.strip().lower():
            continue  # artist unchanged — not in scope for this curve
        pairs.append(
            Pair(
                stem=stem,
                quadrant=position,
                row_index=row_index,
                gemini_artist=g_artist,
                alex_artist=a_artist,
            )
        )
    return pairs


def _load_pairs(corpus_dir: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for corrections_path in sorted(corpus_dir.glob("*.corrections.json")):
        stem = corrections_path.stem.removesuffix(".corrections")
        try:
            corrections = json.loads(corrections_path.read_text())
        except json.JSONDecodeError:
            continue
        # Skip drafts — only Alex's signed-off pages are ground truth.
        if corrections.get("status") != "complete":
            continue
        pairs.extend(_extract_pairs_from_corrections(stem, corrections))
    return pairs


# -- Scoring -----------------------------------------------------------------


async def _score_pairs(pairs: list[Pair], lml: LMLClient) -> list[Scored]:
    """Look up each Gemini artist via LML, score the correction against Alex."""
    items = [{"artist": p.gemini_artist} for p in pairs]
    results = await lml.bulk_lookup(items)
    scored: list[Scored] = []
    for pair, result in zip(pairs, results, strict=True):
        corrected = result.corrected_artist
        if corrected is None:
            scored.append(Scored(pair=pair, corrected_artist=None, score=0))
            continue
        score = int(fuzz.token_set_ratio(corrected.lower(), pair.alex_artist.lower()))
        scored.append(Scored(pair=pair, corrected_artist=corrected, score=score))
    return scored


# -- Threshold sweep ---------------------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    threshold: int
    auto_accepts: int
    true_positives: int
    precision: float | None  # None when auto_accepts == 0
    recall: float


def _compute_curve(scored: list[Scored], thresholds: list[int]) -> list[CurvePoint]:
    """For each T, compute precision/recall.

    Precision  = TP / auto_accepts
    Recall     = TP / total (count of all rows Alex changed the artist on)
    TP         = rows where (a) score >= T and (b) lower(LML.corrected) == lower(alex_artist)

    'recall' here is the fraction of Alex-changed rows that this T would
    have caught. It plateaus at the lowest T because lowering T can never
    reduce TP.
    """
    total = len(scored)
    points: list[CurvePoint] = []
    for t in thresholds:
        auto = 0
        tp = 0
        for s in scored:
            if s.corrected_artist is None:
                continue
            if s.score < t:
                continue
            auto += 1
            if s.corrected_artist.lower() == s.pair.alex_artist.lower():
                tp += 1
        precision = (tp / auto) if auto else None
        recall = (tp / total) if total else 0.0
        points.append(
            CurvePoint(
                threshold=t,
                auto_accepts=auto,
                true_positives=tp,
                precision=precision,
                recall=recall,
            )
        )
    return points


def _pick_threshold(curve: list[CurvePoint], min_precision: float) -> CurvePoint | None:
    """Smallest threshold where precision >= min_precision and at least one
    auto-accept happened. Returns None if no T qualifies."""
    qualifying = [p for p in curve if p.precision is not None and p.precision >= min_precision]
    if not qualifying:
        return None
    return min(qualifying, key=lambda p: p.threshold)


def _non_latin_fraction(pairs: list[Pair]) -> float:
    """Fraction of pairs where `alex_artist` contains non-ASCII chars."""
    if not pairs:
        return 0.0
    non = sum(1 for p in pairs if any(ord(c) > 127 for c in p.alex_artist))
    return non / len(pairs)


# -- Reporting ---------------------------------------------------------------


def _print_table(curve: list[CurvePoint]) -> None:
    print(f"{'T':>4}  {'auto':>6}  {'TP':>6}  {'precision':>10}  {'recall':>8}")
    print("-" * 44)
    for p in curve:
        prec = f"{p.precision:.3f}" if p.precision is not None else "  ----"
        print(
            f"{p.threshold:>4}  {p.auto_accepts:>6}  {p.true_positives:>6}  "
            f"{prec:>10}  {p.recall:>8.3f}"
        )


def _write_csv(curve: list[CurvePoint], path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold", "auto_accepts", "true_positives", "precision", "recall"])
        for p in curve:
            writer.writerow(
                [
                    p.threshold,
                    p.auto_accepts,
                    p.true_positives,
                    f"{p.precision:.6f}" if p.precision is not None else "",
                    f"{p.recall:.6f}",
                ]
            )


# -- Entry point -------------------------------------------------------------


async def _run(corpus: Path, out_csv: Path, min_precision: float) -> int:
    pairs = _load_pairs(corpus)
    if not pairs:
        print(f"No (gemini, alex) pairs found under {corpus}", file=sys.stderr)
        return 1

    base_url = os.environ.get("LML_URL", DEFAULT_LML_URL)
    api_key = os.environ.get("LML_API_KEY") or None
    if api_key is None:
        print(
            "warning: LML_API_KEY not set; LML production requires auth and will 401.",
            file=sys.stderr,
        )

    # Per-page batches are tiny relative to LML's 100-item cap; the cap on
    # per-batch latency is the dominant constraint when LML's local Discogs
    # cache is cold (each artist forces a live Discogs lookup). Drop the
    # batch size to keep each roundtrip short.
    http = httpx.AsyncClient(base_url=base_url, timeout=300)
    async with LMLClient(
        http=http, api_key=api_key, batch_size=20, max_concurrent_batches=2
    ) as lml:
        try:
            scored = await _score_pairs(pairs, lml)
        except LMLError as exc:
            print(f"LML error: {exc}", file=sys.stderr)
            return 2

    thresholds = list(range(50, 101, 5))
    curve = _compute_curve(scored, thresholds)
    print(f"Corpus: {corpus} ({len(pairs)} (gemini, alex) pairs)")
    print(f"Non-Latin coverage: {_non_latin_fraction(pairs):.2%} of Alex artists")
    print()
    _print_table(curve)
    print()
    pick = _pick_threshold(curve, min_precision)
    if pick is None:
        print(
            f"No threshold achieves precision >= {min_precision:.0%}. "
            "Do NOT ship the default; investigate first."
        )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(curve, out_csv)
        print(f"Wrote curve to {out_csv}")
        return 3

    print(
        f"Recommended T = {pick.threshold} "
        f"(precision={pick.precision:.3f}, recall={pick.recall:.3f}, "
        f"auto_accepts={pick.auto_accepts}, true_positives={pick.true_positives})"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(curve, out_csv)
    print(f"Wrote curve to {out_csv}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_REPO_ROOT / "data" / "verifier-pulled-refresh",
        help="Directory containing <stem>.bundle.json + <stem>.verified.json pairs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/reconciliation-curve.csv"),
        help="Where to write the precision/recall CSV.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.95,
        help="Minimum precision required; pick the smallest T that hits it.",
    )
    args = parser.parse_args()
    code = asyncio.run(_run(args.corpus, args.out, args.min_precision))
    sys.exit(code)


if __name__ == "__main__":
    main()
