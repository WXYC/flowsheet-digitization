"""Pre-processor: turn a `PageResult` + page image into a verifier bundle.

The bundle is the input the static `verifier/` UI consumes. It contains the
extraction output (verbatim from the pipeline) plus geometry: a bbox per
quadrant, a bbox per row inside each quadrant, and a relative path to the
source image so the UI can canvas-crop each row in the browser.

CLI surface:

    python -m scripts.make_verifier_bundle \\
        data/results/<rel>/page-NN.json \\
        data/pages/<rel>/page-NN.png \\
        --out data/verifier/<stem>.bundle.json

If `--out` is omitted, the output is written to
`data/verifier/<stem>.bundle.json` next to the repo root.

The bundle is a derivation, not a long-running result — re-running
overwrites. The pre-processor creates the output's parent directory if
it doesn't exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from core.page_layout import PageLayout, detect_page_layout, partition_row_lines_by_quadrant
from core.schema import QUADRANT_ORDER, Entry, PageResult, QuadrantPosition

# Bump when the bundle JSON schema becomes incompatible.
# `verifier/README.md` documents the versioning strategy.
#   v1: initial schema.
#   v2: add `pdf_path` and `page_number` so the verifier UI can target the
#       corresponding `jobs.db` row when saving corrections back.
SCHEMA_VERSION = 2


BBox = tuple[int, int, int, int]


def _parse_job_key_from_result_path(result_path: Path) -> tuple[str, int] | None:
    """Recover `(pdf_path, page_number)` from a pipeline result-JSON path.

    The pipeline writes results at `<data_root>/results/<rel-pdf>/page-NN.json`
    (see `core.pipeline.result_path_for`). Reversing that gives us the
    `(pdf_path, page_number)` pair used as the primary key in `jobs.db`.

    Returns `None` when the path doesn't match this layout (e.g. test
    fixtures, `/tmp` spike outputs, ad-hoc files) — those bundles save
    files only, no DB update.
    """
    parts = result_path.parts
    if "results" not in parts:
        return None
    idx = parts.index("results")
    after = parts[idx + 1 :]
    if len(after) < 2:
        return None
    *pdf_dir_parts, page_file = after
    if not page_file.startswith("page-") or not page_file.endswith(".json"):
        return None
    try:
        page_number = int(page_file[len("page-") : -len(".json")])
    except ValueError:
        return None
    pdf_path = "/".join(pdf_dir_parts) + ".pdf"
    return (pdf_path, page_number)


def _quadrant_bboxes(layout: PageLayout, *, page_width: int) -> dict[QuadrantPosition, BBox]:
    """Bounding box of each quadrant's body region.

    Quadrants partition the body strip (header_bottom_y .. body_bottom_y)
    via `column_mid_x` (left/right) and `body_mid_y` (top/bottom).
    """
    return {
        "top_left": (0, layout.header_bottom_y, layout.column_mid_x, layout.body_mid_y),
        "top_right": (layout.column_mid_x, layout.header_bottom_y, page_width, layout.body_mid_y),
        "bottom_left": (0, layout.body_mid_y, layout.column_mid_x, layout.body_bottom_y),
        "bottom_right": (
            layout.column_mid_x,
            layout.body_mid_y,
            page_width,
            layout.body_bottom_y,
        ),
    }


def _merge_with_spans(entries: list[Entry]) -> list[tuple[Entry, int]]:
    """Apply continuation-row merging and compute each entry's physical-row span.

    This is the geometry-aware companion to `core.continuations.merge_continuations`:
    it produces the same merged entries, paired with the number of physical
    flowsheet rows each logical entry occupies on the page.

      - notes="continuation": folds into the previous logical entry's raw_text
        (verbatim with the existing merge rules) and adds 1 to its span.
      - notes="double_height": stays as a single logical entry but spans 2 rows.
      - All others: span 1.

    A leading "continuation" with nothing above it is preserved as-is with
    span 1, matching `merge_continuations`'s edge-case behavior.

    Why this lives here and not in `core/continuations.py`: span-tracking is
    a verifier-geometry concern. The on-disk pipeline doesn't need it.
    """
    result: list[tuple[Entry, int]] = []
    for entry in entries:
        if entry.notes == "continuation" and result:
            prior, prior_span = result[-1]
            joined = f"{prior.raw_text.rstrip()} {entry.raw_text.lstrip()}".strip()
            # Mark the merged entry as `double_height` so the verifier UI's
            # notes dropdown reflects the multi-row nature of the row. The
            # original schema enum doesn't distinguish "absorbed continuation"
            # from "model-tagged double_height" — both mean "this logical
            # entry occupies more than one physical row" for the verifier.
            merged = prior.model_copy(
                update={
                    "raw_text": joined,
                    "oddities": [*prior.oddities, *entry.oddities],
                    "notes": "double_height",
                }
            )
            result[-1] = (merged, prior_span + 1)
        elif entry.notes == "double_height":
            result.append((entry, 2))
        else:
            result.append((entry, 1))
    return result


def _assign_row_bboxes(
    quad_bbox: BBox,
    lines: list[int],
    spans: list[int],
) -> list[BBox]:
    """Pair logical entries to row strips inside a quadrant.

    `spans` is one int per logical entry: the number of physical row strips
    that entry occupies on the page (1 for normal entries; 2 for double_height
    or one continuation; 3 for two continuations; etc.).

    Heuristic:
      - When `len(lines) >= sum(spans) + 1`, slice consecutive line pairs
        according to each entry's span. Entry i's bbox spans from `lines[j]`
        to `lines[j + spans[i]]`, with `j` advancing by `spans[i]` between
        entries. Trailing lines (beyond what spans require) are ignored.
      - Otherwise, even-spacing fallback: divide the quadrant height into
        `len(spans)` equal strips, one per logical entry, ignoring the
        physical-row count.

    The fallback uses entry count, not physical row count, because uniform
    strips are better UX than partial pairing (which would leave the tail
    of the quadrant uncropped on entries with wider spans).
    """
    if not spans:
        return []
    x1, y1, x2, y2 = quad_bbox
    total_physical_rows = sum(spans)
    if len(lines) >= total_physical_rows + 1:
        rows: list[BBox] = []
        j = 0
        for span in spans:
            rows.append((x1, lines[j], x2, lines[j + span]))
            j += span
        return rows
    height = y2 - y1
    n_entries = len(spans)
    step = height / n_entries
    return [
        (x1, y1 + int(round(i * step)), x2, y1 + int(round((i + 1) * step)))
        for i in range(n_entries)
    ]


def make_bundle(
    page: PageResult,
    *,
    image_path: Path,
    bundle_path: Path,
    job_key: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Assemble the verifier bundle for one page.

    `bundle_path` is used only to compute the relative `image_path` field
    — the file isn't written here. The CLI's `main` writes the bundle to
    disk; this function is the pure construction step so tests can
    inspect the output without filesystem side effects.
    """
    image = Image.open(image_path)
    layout = detect_page_layout(image)
    width, _height = image.size

    quad_boxes = _quadrant_bboxes(layout, page_width=width)
    lines_by_quad = partition_row_lines_by_quadrant(image, layout)

    quadrants_out: list[dict[str, Any]] = []
    for position in QUADRANT_ORDER:
        # Continuations fold into the previous entry's raw_text; double_height
        # stays as one entry. `_merge_with_spans` does the merge and tracks
        # how many physical rows each resulting logical entry occupies, so
        # the bbox cropper can skip the right number of grid lines per entry.
        source_quad = next((q for q in page.quadrants if q.position == position), None)
        if source_quad is None:
            continue
        merged_with_spans = _merge_with_spans(source_quad.entries)
        bbox = quad_boxes[position]
        lines = lines_by_quad.get(position, [])
        spans = [s for _, s in merged_with_spans]
        row_boxes = _assign_row_bboxes(bbox, lines, spans=spans)

        entries_out: list[dict[str, Any]] = []
        merged_entries = [e for e, _ in merged_with_spans]
        for entry, row_bbox in zip(merged_entries, row_boxes, strict=True):
            entries_out.append(
                {
                    "row_index": entry.row_index,
                    "raw_text": entry.raw_text,
                    "confidence": entry.confidence,
                    "type_raw": entry.type_raw,
                    "notes": entry.notes,
                    "oddities": list(entry.oddities),
                    "row_bbox": list(row_bbox),
                }
            )
        quadrants_out.append(
            {
                "position": position,
                "bbox": list(bbox),
                "hour_raw": source_quad.hour_raw,
                "jock_raw": source_quad.jock_raw,
                "entries": entries_out,
                "oddities": list(source_quad.oddities),
            }
        )

    image_rel = os.path.relpath(image_path, bundle_path.parent)
    pdf_path: str | None = None
    page_number: int | None = None
    if job_key is not None:
        pdf_path, page_number = job_key
        # When we know the job key, derive a corpus-unique stem from it.
        # The pipeline's default image filename is just `page-NN.png`,
        # which would collide across PDFs.
        stem = f"{Path(pdf_path).stem}-page{page_number:02d}"
    else:
        stem = image_path.stem
    return {
        "schema_version": SCHEMA_VERSION,
        "stem": stem,
        "image_path": image_rel,
        "pdf_path": pdf_path,
        "page_number": page_number,
        "model_version": page.model_version,
        "extracted_at": page.extracted_at.isoformat(),
        "page_date_raw": page.page_date_raw,
        "comments_raw": page.comments_raw,
        "oddities": list(page.oddities),
        "quadrants": quadrants_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a verifier bundle from a PageResult JSON + page image.",
    )
    parser.add_argument("result", type=Path, help="Path to the extraction result JSON.")
    parser.add_argument("image", type=Path, help="Path to the page PNG.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output bundle path. Defaults to "
            "data/verifier/<image-stem>.bundle.json relative to the cwd."
        ),
    )
    args = parser.parse_args(argv)

    if not args.result.is_file():
        print(f"result not found: {args.result}", file=sys.stderr)
        return 1
    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1

    page = PageResult.model_validate_json(args.result.read_text())
    job_key = _parse_job_key_from_result_path(args.result)
    if args.out is not None:
        out_path = args.out
    elif job_key is not None:
        pdf_path, page_number = job_key
        out_stem = f"{Path(pdf_path).stem}-page{page_number:02d}"
        out_path = Path("data/verifier") / f"{out_stem}.bundle.json"
    else:
        out_path = Path("data/verifier") / f"{args.image.stem}.bundle.json"
    bundle = make_bundle(page, image_path=args.image, bundle_path=out_path, job_key=job_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
