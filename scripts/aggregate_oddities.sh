#!/usr/bin/env bash
# aggregate_oddities.sh — frequency-rank Gemini-flagged oddities across all
# extracted result JSONs.
#
# Reads every `data/results/**/*.json`, flattens the page-level, quadrant-level,
# and entry-level `oddities` lists, prefixes each with its level, then prints
# a frequency-ranked top-N. Intended for phase-2 discovery: after a few hundred
# pages have been processed, the most common oddity strings become candidates
# to formalize as proper schema fields.
#
# Usage:
#   scripts/aggregate_oddities.sh                # top 50, level-tagged (default)
#   scripts/aggregate_oddities.sh 200            # top 200
#   scripts/aggregate_oddities.sh 0              # no limit
#   MODE=flat   scripts/aggregate_oddities.sh    # all three levels mixed together
#   MODE=totals scripts/aggregate_oddities.sh    # just per-level counts
#
# Honors $DATA_ROOT (default ./data) so it picks up wherever the pipeline
# writes its results.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data}"
RESULTS="${DATA_ROOT}/results"
LIMIT="${1:-50}"
MODE="${MODE:-tagged}"

if [[ ! -d "$RESULTS" ]]; then
  echo "aggregate_oddities: no results directory at $RESULTS" >&2
  exit 1
fi

case "$MODE" in
  tagged)
    # Each oddity prefixed with its level (page / quadrant / entry).
    find "$RESULTS" -name '*.json' -type f -print0 \
      | xargs -0 jq -r '
          (.oddities[]?                       | "page\t"     + .),
          (.quadrants[].oddities[]?           | "quadrant\t" + .),
          (.quadrants[].entries[].oddities[]? | "entry\t"    + .)
        ' \
      | sort | uniq -c | sort -rn \
      | { [[ "$LIMIT" -gt 0 ]] && head -n "$LIMIT" || cat; }
    ;;
  flat)
    # All three levels mixed; useful when the level isn't informative.
    find "$RESULTS" -name '*.json' -type f -print0 \
      | xargs -0 jq -r '
          .oddities[]?,
          (.quadrants[].oddities[]?),
          (.quadrants[].entries[].oddities[]?)
        ' \
      | sort | uniq -c | sort -rn \
      | { [[ "$LIMIT" -gt 0 ]] && head -n "$LIMIT" || cat; }
    ;;
  totals)
    # Total oddity count per level across the corpus.
    find "$RESULTS" -name '*.json' -type f -print0 \
      | xargs -0 jq -r '
          [
            (.oddities | length),
            (.quadrants | map(.oddities | length) | add),
            (.quadrants | map(.entries | map(.oddities | length) | add) | add)
          ] | @tsv
        ' \
      | awk 'BEGIN { p=0; q=0; e=0; n=0 }
             { p += $1; q += $2; e += $3; n += 1 }
             END {
               printf "files:    %d\n", n
               printf "page:     %d\n", p
               printf "quadrant: %d\n", q
               printf "entry:    %d\n", e
               printf "total:    %d\n", p+q+e
             }'
    ;;
  *)
    echo "aggregate_oddities: unknown MODE='$MODE' (expected: tagged | flat | totals)" >&2
    exit 2
    ;;
esac
