#!/usr/bin/env python3
"""inspect_crops.py — visually verify `core.page_layout.detect_page_layout`.

For every PNG in `tests/golden/` (or the directory passed via `--golden-dir`),
runs the detector and writes:

  <out>/<stem>/overlay.png           full page with red lines on detected boundaries
  <out>/<stem>/header.png            header strip crop
  <out>/<stem>/top_left.png          four quadrant crops
  <out>/<stem>/top_right.png
  <out>/<stem>/bottom_left.png
  <out>/<stem>/bottom_right.png
  <out>/<stem>/footer.png            comments band (Phase 2 will use this)

Use this when iterating on the detector to confirm at-a-glance that the
crops cleanly separate top/bottom blocks, left/right columns, and the
comments band.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.page_layout import detect_page_layout  # noqa: E402


def _draw_overlay(image: Image.Image) -> Image.Image:
    """Render the detected grid lines on a copy of the page in red."""
    layout = detect_page_layout(image)
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    line_kwargs = {"fill": (255, 0, 0), "width": 6}
    draw.line([(0, layout.header_bottom_y), (w, layout.header_bottom_y)], **line_kwargs)
    draw.line([(0, layout.body_mid_y), (w, layout.body_mid_y)], **line_kwargs)
    draw.line([(0, layout.body_bottom_y), (w, layout.body_bottom_y)], **line_kwargs)
    draw.line([(layout.column_mid_x, 0), (layout.column_mid_x, h)], **line_kwargs)
    return overlay


def _write_crops(image: Image.Image, out_dir: Path) -> None:
    layout = detect_page_layout(image)
    w, h = image.size
    out_dir.mkdir(parents=True, exist_ok=True)
    image.crop((0, 0, w, layout.header_bottom_y)).save(out_dir / "header.png")
    image.crop((0, layout.header_bottom_y, layout.column_mid_x, layout.body_mid_y)).save(
        out_dir / "top_left.png"
    )
    image.crop((layout.column_mid_x, layout.header_bottom_y, w, layout.body_mid_y)).save(
        out_dir / "top_right.png"
    )
    image.crop((0, layout.body_mid_y, layout.column_mid_x, layout.body_bottom_y)).save(
        out_dir / "bottom_left.png"
    )
    image.crop((layout.column_mid_x, layout.body_mid_y, w, layout.body_bottom_y)).save(
        out_dir / "bottom_right.png"
    )
    image.crop((0, layout.body_bottom_y, w, h)).save(out_dir / "footer.png")
    _draw_overlay(image).save(out_dir / "overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Directory containing golden PNGs (default: tests/golden)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/crops"),
        help="Output directory (default: /tmp/crops)",
    )
    args = parser.parse_args()

    pngs = sorted(args.golden_dir.glob("*.png"))
    if not pngs:
        print(f"No PNGs found under {args.golden_dir}", file=sys.stderr)
        sys.exit(1)

    for png in pngs:
        image = Image.open(png)
        layout = detect_page_layout(image)
        out_dir = args.out / png.stem
        _write_crops(image, out_dir)
        print(
            f"{png.stem}: header_bottom_y={layout.header_bottom_y} "
            f"body_mid_y={layout.body_mid_y} body_bottom_y={layout.body_bottom_y} "
            f"column_mid_x={layout.column_mid_x} "
            f"-> {out_dir}/"
        )


if __name__ == "__main__":
    main()
