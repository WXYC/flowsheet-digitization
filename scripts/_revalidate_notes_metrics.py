"""Compute per-page and aggregate metrics for the notes revalidation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PAGES = [2, 6, 9, 14, 16, 19]
VERIFIED_DIR = Path("data/verifier-pulled-refresh")
FRESH_DIR = Path("data/notes-revalidation-2026-06-04")


def index_notes(page: dict) -> dict[tuple[str, int], str | None]:
    out: dict[tuple[str, int], str | None] = {}
    for q in page.get("quadrants", []):
        pos = q.get("position")
        for e in q.get("entries", []):
            ri = e.get("row_index")
            out[(pos, ri)] = e.get("notes")
    return out


def main() -> None:
    rows: list[dict] = []
    agg_emit = Counter()  # fresh-Gemini notes value -> count
    agg_truth = Counter()  # Alex notes value -> count
    agg_keep = Counter()  # for each fresh-emitted tag, # where Alex agrees
    agg_recall_hit = Counter()  # for each Alex-truth tag, # where fresh-Gemini matches
    agg_unmatched_fresh = 0
    agg_unmatched_truth = 0

    for p in PAGES:
        verified = json.loads(
            (VERIFIED_DIR / f"1990-04apr0106-page{p:02d}.verified.json").read_text()
        )
        fresh = json.loads((FRESH_DIR / f"1990-04apr0106-page{p:02d}.fresh.json").read_text())

        truth_notes = index_notes(verified)
        fresh_notes = index_notes(fresh)

        # Per-page counters
        emit = Counter()
        truth = Counter()
        keep = Counter()
        recall_hit = Counter()
        unmatched_fresh = 0
        unmatched_truth = 0

        all_keys = set(truth_notes) | set(fresh_notes)
        for k in all_keys:
            t = truth_notes.get(k)  # may be None or missing-entry
            f = fresh_notes.get(k)
            if k not in truth_notes:
                # fresh emitted an entry Alex doesn't have at that position
                if f is not None:
                    unmatched_fresh += 1
                continue
            if k not in fresh_notes:
                if t is not None:
                    unmatched_truth += 1
                continue
            if f is not None:
                emit[f] += 1
            if t is not None:
                truth[t] += 1
            if f is not None and t is not None and f == t:
                keep[f] += 1
                recall_hit[t] += 1

        rows.append(
            {
                "page": p,
                "emit": dict(emit),
                "truth": dict(truth),
                "keep": dict(keep),
                "recall_hit": dict(recall_hit),
                "unmatched_fresh_rows": unmatched_fresh,
                "unmatched_truth_rows": unmatched_truth,
            }
        )

        for k, v in emit.items():
            agg_emit[k] += v
        for k, v in truth.items():
            agg_truth[k] += v
        for k, v in keep.items():
            agg_keep[k] += v
        for k, v in recall_hit.items():
            agg_recall_hit[k] += v
        agg_unmatched_fresh += unmatched_fresh
        agg_unmatched_truth += unmatched_truth

    print("== Per-page (fresh emits / Alex truth) ==")
    tags = ["crossed_out", "continuation", "double_height", "illegible", "other"]
    header = f"{'page':>4}  " + "  ".join(f"{t[:5]:>11}" for t in tags) + "  unmatched(F/T)"
    print(header)
    for r in rows:
        cells = []
        for t in tags:
            e = r["emit"].get(t, 0)
            tr = r["truth"].get(t, 0)
            cells.append(f"{e:>4}/{tr:<5}")
        print(
            f"{r['page']:>4}  "
            + "  ".join(cells)
            + f"  {r['unmatched_fresh_rows']}/{r['unmatched_truth_rows']}"
        )

    print()
    print("== Aggregate ==")
    print(f"fresh-Gemini emit counts: {dict(agg_emit)}")
    print(f"Alex truth counts:        {dict(agg_truth)}")
    print(f"matched (fresh==truth):   {dict(agg_keep)}")
    print(f"truth rows recalled:      {dict(agg_recall_hit)}")
    print(f"unmatched fresh entries (row Alex doesn't have): {agg_unmatched_fresh}")
    print(f"unmatched truth entries (row fresh-Gemini lacks): {agg_unmatched_truth}")

    print()
    print("== Headline metrics ==")
    # crossed_out precision: of fresh-Gemini's crossed_out emits, how many does Alex keep as crossed_out
    co_emit = agg_emit.get("crossed_out", 0)
    co_keep = agg_keep.get("crossed_out", 0)
    co_prec = (co_keep / co_emit) if co_emit else None
    print(
        f"crossed_out precision  = {co_keep}/{co_emit} = {co_prec if co_prec is None else f'{co_prec * 100:.0f}%'}"
    )

    # continuation recall: of Alex's continuation, how many fresh-Gemini caught
    co_truth = agg_truth.get("continuation", 0)
    co_hit = agg_recall_hit.get("continuation", 0)
    print(
        f"continuation recall    = {co_hit}/{co_truth} = {(co_hit / co_truth * 100):.0f}%"
        if co_truth
        else "continuation recall    = N/A"
    )

    # double_height recall
    dh_truth = agg_truth.get("double_height", 0)
    dh_hit = agg_recall_hit.get("double_height", 0)
    print(
        f"double_height recall   = {dh_hit}/{dh_truth} = {(dh_hit / dh_truth * 100):.0f}%"
        if dh_truth
        else "double_height recall   = N/A"
    )

    # illegible recall (bonus)
    il_truth = agg_truth.get("illegible", 0)
    il_hit = agg_recall_hit.get("illegible", 0)
    if il_truth:
        print(f"illegible recall       = {il_hit}/{il_truth} = {(il_hit / il_truth * 100):.0f}%")


if __name__ == "__main__":
    main()
