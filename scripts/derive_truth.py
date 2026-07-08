"""Derive a `GoldenTruth` file from a hand-corrected `PageResult` or a
settled multi-reviewer `CalibrationCanonical`.

Two source modes:

  * `--from verified` (default) — reads `<stem>.verified.json` produced
    by the single-reviewer verifier UI.
  * `--from canonical` — reads a settled `canonical.json` (plus the
    sibling `bundle.json`) produced by the multi-reviewer calibration
    flow (`plans/multi-reviewer-calibration.md`). The two files sit next
    to each other under `data/calibration/<year>/<bucket>/<stem>/`;
    the canonical carries the merged `raw_text` per row and the bundle
    carries the quadrant / `jock_raw` / `hour_raw` context that the
    canonical row list has flattened away.

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

    python -m scripts.derive_truth --from verified \\
        data/verifier/<stem>.verified.json \\
        --out tests/golden/<stem>.truth.json

    python -m scripts.derive_truth --from canonical \\
        data/calibration/1990/anomaly/<stem>/canonical.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from core.golden import GoldenTruth, QuadrantTruth, RowTruth
from core.parse import parse_artist_track
from core.schema import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationCanonical,
    PageResult,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_ILLEGIBLE_SENTINEL = "__illegible__"

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


class CanonicalReadError(RuntimeError):
    """Failure reading or interpreting a canonical.json file."""


class TruthWriteError(RuntimeError):
    """Failure writing a truth file to disk."""


def _year_from_stem(stem: str) -> str:
    return stem.split("-", 1)[0]


def _default_canonical_out_path(stem: str, out_root: Path | None) -> Path:
    """Default target for `--from canonical`:
    `tests/golden/calibration/<year>/<stem>.truth.json`."""
    if out_root is None:
        out_root = REPO_ROOT / "tests" / "golden" / "calibration"
    return out_root / _year_from_stem(stem) / f"{stem}.truth.json"


def derive_truth_from_canonical(
    canonical: CalibrationCanonical, bundle: dict
) -> GoldenTruth:
    """Merge canonical rows onto bundle quadrant structure to emit truth.

    Status handling (per plan §Truth derivation wiring):

      * `unanimous`, `majority` → emit row from `raw_text`.
      * `illegible` (`raw_text == __illegible__`) → skip.
      * `majority_spurious`, `unanimous_spurious` → skip
        (row asserted not to exist).
      * `under_emit_majority` → emit from injected `raw_text`.
      * `under_emit_no_text_agreement` → skip.
    """
    quadrants = bundle.get("quadrants") or []
    # Build flat index → (quadrant_idx, entry_index) map so we know where
    # to slot each canonical row.
    flat_bundle_rows: list[tuple[int, dict]] = []
    for quad_idx, quad in enumerate(quadrants):
        entries = quad.get("entries") if isinstance(quad, dict) else []
        if isinstance(entries, list):
            for entry in entries:
                flat_bundle_rows.append(
                    (quad_idx, entry if isinstance(entry, dict) else {})
                )

    # Injections (bundle_row_index is None) carry no quadrant assignment
    # in canonical, so they're skipped in the emit path. Under subset
    # semantics this is safe — the truth file's absence of a row means
    # "unassert" for that gap.
    per_quadrant_rows: dict[int, list[RowTruth]] = {i: [] for i in range(len(quadrants))}
    skip_statuses = {
        "majority_spurious",
        "unanimous_spurious",
        "illegible",
        "under_emit_no_text_agreement",
    }
    for row in canonical.rows:
        if row.bundle_row_index is None:
            continue
        if row.verification.status in skip_statuses:
            continue
        if row.raw_text is None or row.raw_text == _ILLEGIBLE_SENTINEL:
            continue
        if 0 <= row.bundle_row_index < len(flat_bundle_rows):
            quad_idx, _ = flat_bundle_rows[row.bundle_row_index]
            per_quadrant_rows[quad_idx].append(
                RowTruth(raw_substring=_row_substring(row.raw_text))
            )

    quadrants_out: list[QuadrantTruth] = []
    for quad_idx, quad in enumerate(quadrants):
        pos = quad.get("position") if isinstance(quad, dict) else None
        if not isinstance(pos, str):
            continue
        quadrants_out.append(
            QuadrantTruth(
                position=pos,  # type: ignore[arg-type]
                hour_raw=quad.get("hour_raw") if isinstance(quad, dict) else None,
                jock_substring=_jock_substring(
                    quad.get("jock_raw") if isinstance(quad, dict) else None
                ),
                rows=per_quadrant_rows.get(quad_idx, []),
            )
        )

    return GoldenTruth(
        page_date_substrings=_date_substrings(
            bundle.get("page_date_raw") if isinstance(bundle, dict) else None
        ),
        quadrants=quadrants_out,
    )


def from_canonical(
    canonical_path: Path, *, out_root: Path | None = None
) -> Path:
    """Read `canonical.json` at `canonical_path`, emit the corresponding
    `<stem>.truth.json`, and return the written path.

    `out_root` defaults to `<repo>/tests/golden/calibration/`. The file
    lands at `<out_root>/<year>/<stem>.truth.json`. Writes atomically
    (write-to-temp + rename) — an interrupt mid-write leaves the prior
    truth file intact.

    Raises:
        CanonicalReadError: canonical.json is missing, malformed, or its
            schema_version does not match CALIBRATION_SCHEMA_VERSION.
        TruthWriteError: the atomic rename failed.
    """
    if not canonical_path.is_file():
        raise CanonicalReadError(f"canonical.json not found: {canonical_path}")
    try:
        canonical = CalibrationCanonical.model_validate_json(canonical_path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise CanonicalReadError(
            f"canonical.json parse failed: {canonical_path}: {exc}"
        ) from exc
    if canonical.schema_version != CALIBRATION_SCHEMA_VERSION:
        raise CanonicalReadError(
            f"canonical schema_version {canonical.schema_version} != "
            f"CALIBRATION_SCHEMA_VERSION {CALIBRATION_SCHEMA_VERSION}"
        )
    bundle_path = canonical_path.parent / "bundle.json"
    if not bundle_path.exists():
        raise CanonicalReadError(f"sibling bundle.json missing: {bundle_path}")
    try:
        bundle = json.loads(bundle_path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise CanonicalReadError(
            f"bundle.json parse failed: {bundle_path}: {exc}"
        ) from exc

    truth = derive_truth_from_canonical(canonical, bundle)
    out_path = _default_canonical_out_path(canonical.stem, out_root)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(truth.model_dump_json(indent=2, exclude_defaults=False))
        os.replace(tmp, out_path)
    except Exception as exc:  # noqa: BLE001
        raise TruthWriteError(f"could not write truth {out_path}: {exc}") from exc
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Derive a GoldenTruth file from a verified PageResult or a settled canonical.",
    )
    parser.add_argument(
        "--from",
        dest="mode",
        choices=("verified", "canonical"),
        default="verified",
        help="Input mode: 'verified' (single-reviewer) or 'canonical' (multi-reviewer).",
    )
    parser.add_argument(
        "input", type=Path, help="Path to the verified.json or canonical.json input."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output truth.json path. Required for --from verified. "
            "For --from canonical, defaults to tests/golden/calibration/<year>/<stem>.truth.json; "
            "when supplied, its parent is used as the out_root."
        ),
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.mode == "canonical":
        try:
            out_root = args.out.parent if args.out else None
            written = from_canonical(args.input, out_root=out_root)
        except CanonicalReadError as exc:
            print(f"canonical read failed: {exc}", file=sys.stderr)
            return 2
        except TruthWriteError as exc:
            print(f"truth write failed: {exc}", file=sys.stderr)
            return 3
        print(f"wrote {written}")
        return 0

    # verified mode
    if args.out is None:
        print("--out is required for --from verified", file=sys.stderr)
        return 1
    page = PageResult.model_validate_json(args.input.read_text())
    truth = derive_truth(page)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(truth.model_dump_json(indent=2, exclude_defaults=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
