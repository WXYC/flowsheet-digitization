"""Derive a `GoldenTruth` file from a hand-corrected `PageResult`.

The verifier UI exports `<stem>.verified.json` — a `PageResult` whose
`raw_text` fields have been hand-corrected. This tool extracts short
substrings from those fields and writes a `GoldenTruth`-shaped file
that plugs into the existing parity-test harness.

Substring rules (codified to match the convention in `tests/golden/*.truth.json`):

  * page_date_substrings: whitespace-delimited tokens of `page_date_raw`.
      e.g. "Tues 4/3 90" -> ["Tues", "4/3", "90"]
  * jock_substring: first whitespace-delimited token of `jock_raw`,
      uppercased and truncated to 4 chars.
      e.g. "Andrew" -> "ANDR"
  * raw_substring (per row): the artist portion of `raw_text`
      (`parse_artist_track`), uppercased, truncated to <=24 chars at
      the last whitespace boundary inside the cutoff. If no separator,
      use the full text.

The substrings are deliberately short — `core.golden._icontains` is a
case-insensitive substring match, so short tokens are forgiving of
small misspellings while remaining unambiguous within the WXYC corpus.

CLI:

    python -m scripts.derive_truth \\
        data/verifier/<stem>.verified.json \\
        --out tests/golden/<stem>.truth.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.golden import GoldenTruth, QuadrantTruth, RowTruth
from core.parse import parse_artist_track
from core.schema import PageResult

_MAX_ROW_SUBSTRING = 24
_MAX_JOCK_SUBSTRING = 4


def _date_substrings(page_date_raw: str | None) -> list[str]:
    """Split `page_date_raw` into whitespace-delimited tokens.

    Empty list when the field is None, empty, or whitespace-only.
    """
    if not page_date_raw:
        return []
    return page_date_raw.split()


def _jock_substring(jock_raw: str | None) -> str | None:
    """First whitespace-delimited token of `jock_raw`, uppercased,
    truncated to 4 chars. Returns None when the field is missing so the
    truth file omits the assertion entirely.
    """
    if not jock_raw or not jock_raw.strip():
        return None
    first_token = jock_raw.strip().split()[0]
    out = first_token.upper()[:_MAX_JOCK_SUBSTRING]
    return out or None


def _row_substring(raw_text: str) -> str:
    """Artist-portion of `raw_text`, uppercased, capped at 24 chars.

    Falls back to the full raw_text when `parse_artist_track` finds no
    separator (entries without "Artist - Track" structure, e.g. a
    continuation row that wasn't merged).

    The 24-char cap snaps to the last whitespace boundary inside the
    cutoff to avoid mid-word truncation. If the artist is one long
    word, hard-cut at 24.
    """
    artist, _track = parse_artist_track(raw_text)
    src = (artist or raw_text or "").strip().upper()
    if len(src) <= _MAX_ROW_SUBSTRING:
        return src
    cut = src.rfind(" ", 0, _MAX_ROW_SUBSTRING)
    return src[:cut] if cut > 0 else src[:_MAX_ROW_SUBSTRING]


def derive_truth(page: PageResult) -> GoldenTruth:
    """Build a `GoldenTruth` from a hand-corrected `PageResult`.

    Quadrants pass through in canonical order. Entries with empty
    `raw_text` are skipped (nothing to match against).
    """
    quadrants_out: list[QuadrantTruth] = []
    for quad in page.quadrants:
        rows = [
            RowTruth(raw_substring=_row_substring(entry.raw_text))
            for entry in quad.entries
            if entry.raw_text.strip()
        ]
        quadrants_out.append(
            QuadrantTruth(
                position=quad.position,
                hour_raw=quad.hour_raw,
                jock_substring=_jock_substring(quad.jock_raw),
                rows=rows,
            )
        )
    return GoldenTruth(
        page_date_substrings=_date_substrings(page.page_date_raw),
        quadrants=quadrants_out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive a GoldenTruth file from a verified PageResult.",
    )
    parser.add_argument("verified", type=Path, help="Path to the verified.json PageResult.")
    parser.add_argument("--out", type=Path, required=True, help="Output truth.json path.")
    args = parser.parse_args(argv)

    if not args.verified.is_file():
        print(f"verified file not found: {args.verified}", file=sys.stderr)
        return 1

    page = PageResult.model_validate_json(args.verified.read_text())
    truth = derive_truth(page)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(truth.model_dump_json(indent=2, exclude_defaults=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
