"""Detect the printed grid-line coordinates on a WXYC flowsheet page.

The 1990s flowsheet form is a printed grid: a header strip on top, a 2x2
body of "hour blocks" below, and a comments line at the bottom. The
dividers are continuous high-contrast lines at consistent
(per-form-revision) positions. They drift across pages by tens of pixels
because of paper alignment in the scanner — enough that fixed-fraction
cropping clips content or duplicates it across neighboring crops.

`detect_page_layout` returns the three coordinates the cropper needs:

  - `header_bottom_y` — y of the TOP edge of the Hour-Jock cell (the
    first detected row line minus one median row-spacing). Cropping at
    the row line itself would leave the Hour-Jock glyph ascenders inside
    the header crop; backing off one row puts them in the top-quadrant
    crops where they belong.
  - `body_mid_y`      — y in the gap between the top hour blocks' last
    row line and the bottom hour blocks' first row line.
  - `column_mid_x`    — x of the printed vertical divider between the
    left and right columns.

If any coordinate can't be detected, the function falls back to the
fixed-fraction values previously hard-coded in
`scripts/calibrate_models.py`. Detection is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.schema import QUADRANT_ORDER, QuadrantPosition

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Fallback fractions, matching the historical constants in
# `scripts/calibrate_models.py`. Used when detection fails on a page so
# the pipeline never crashes on a corner-case scan.
FALLBACK_HEADER_FRACTION = 0.12

# Fraction of page height for the body-bottom (Comments: line) fallback.
# Empirically the printed Comments: line lands at y/h ≈ 0.967-0.969 on
# clean goldens; 0.97 is the cleanest fixed approximation when detection
# fails on a degraded scan.
FALLBACK_BODY_BOTTOM_FRACTION = 0.97

# Row-line detection sweeps a few thresholds. Pages 2-5 (binary CCITT-G4
# scans) clear 0.5 easily; page 1 (RGB-mode degraded scan) needs 0.4. We
# keep going down to 0.3 in case future scans are even noisier.
_ROW_LINE_THRESHOLDS = (0.5, 0.4, 0.35, 0.3)

# A "row line" is detected if its left-half OR right-half (relative to
# the column midline) is more than this fraction filled with ink. Per-half
# detection is more robust than full-width because handwriting often
# obscures one column's printed line but spares the other.
_MIN_INKY_RUNS = 30  # below this, a threshold is too strict; try the next one

# Body-midline gap criterion: a gap between consecutive row lines counts
# as a candidate for the body midline if it's at least this multiple of
# the median row spacing. On clean scans the gap is between the top
# block's last row line (~y=2250) and the bottom block's "Hour ___ Jock
# ___" header line (~y=2351), so it's only ~1.35x the 75px row spacing.
# On handwriting-disrupted scans the bottom Hour line isn't detected and
# the gap extends to the first bottom-block row, giving ~2.7x. Threshold
# at 1.3x catches both, while staying above the 1.0-1.05x range that
# normal row-to-row spacing produces.
_MIN_GAP_RATIO = 1.3

# How close to the ends of the page the column-midline can be — a vertical
# line very near an edge would be the page binding, not the column divider.
_COLUMN_MID_SEARCH_BAND = (0.3, 0.7)

# Body-midline must be in the central band — a "biggest gap" near the
# top of the page would be the gap between the header strip and the
# first row line, which is a different kind of gap.
_BODY_MID_SEARCH_BAND = (0.3, 0.8)

# Anchor for picking among multiple qualifying gaps. The body midline
# sits below page center because the top-of-body header strip is taller
# than the bottom comments band. Empirically validated to ~y/h=0.54-0.55
# across the 5 goldens.
_BODY_MID_ANCHOR_FRACTION = 0.55

# Search band for the body-bottom (Comments: printed line). The line sits
# very near the bottom edge — pages 4 and 5 detect it at y/h ≈ 0.968. The
# bottom block's last content row line shows up around y/h ≈ 0.95 on
# pages with full row detection, so the lower bound 0.95 includes the
# Comments line and excludes the last body row.
_BODY_BOTTOM_SEARCH_BAND = (0.95, 0.99)

# When the top quadrant's last spacing exceeds this multiple of the
# global median row spacing, the trailing line is reattributed to the
# corresponding bottom quadrant. The anomaly signals that body_mid_y
# landed BELOW the bottom block's hour-jock-cell baseline, leaving that
# line in the top partition by mistake. See
# `partition_row_lines_by_quadrant`'s correction-pass comment.
_BOTTOM_BASELINE_REATTRIBUTION_RATIO = 1.3


@dataclass(frozen=True)
class PageLayout:
    """Pixel coordinates of the printed grid lines on a flowsheet page.

    The page splits into:
      - header strip:     image[:header_bottom_y]
      - top quadrants:    image[header_bottom_y:body_mid_y, :/column_mid_x:]
      - bottom quadrants: image[body_mid_y:body_bottom_y, :/column_mid_x:]
      - footer strip:     image[body_bottom_y:]  (Comments: line + marginalia)
    """

    header_bottom_y: int
    body_mid_y: int
    body_bottom_y: int
    column_mid_x: int


def detect_page_layout(image: PILImage) -> PageLayout:
    """Detect grid line positions via projection profiles.

    Falls back to fixed-fraction coordinates for any value that can't be
    detected so the caller never has to handle exceptions.
    """
    w, h = image.size
    grayscale = np.asarray(image.convert("L"))

    column_mid_x = _detect_column_mid_x(grayscale, w, h)
    row_lines = _detect_row_lines(grayscale, w, column_mid_x)
    header_bottom_y = _detect_header_bottom_y(row_lines, h)
    body_mid_y = _detect_body_mid_y(row_lines, h)
    body_bottom_y = _detect_body_bottom_y(row_lines, h)

    return PageLayout(
        header_bottom_y=header_bottom_y,
        body_mid_y=body_mid_y,
        body_bottom_y=body_bottom_y,
        column_mid_x=column_mid_x,
    )


def _detect_column_mid_x(grayscale: np.ndarray, w: int, h: int) -> int:
    """Strongest vertical line in the central band.

    The printed column divider runs nearly the full page height, so its
    column-sum is the unambiguous argmax in [0.3w, 0.7w] — every other
    vertical structure (handwriting, text columns) is shorter.
    """
    ink = 255 - grayscale
    col_profile = ink.sum(axis=0).astype(np.float64) / 255.0
    lo, hi = int(_COLUMN_MID_SEARCH_BAND[0] * w), int(_COLUMN_MID_SEARCH_BAND[1] * w)
    central = col_profile[lo:hi]
    if central.size == 0 or central.max() < 0.3 * h:
        return w // 2  # fallback
    return int(np.argmax(central) + lo)


def _detect_row_lines(grayscale: np.ndarray, w: int, column_mid_x: int) -> list[int]:
    """Y-coordinates of horizontal row lines.

    A row line is any y where the LEFT half (cols [0, column_mid_x)) OR
    the RIGHT half (cols [column_mid_x, w)) has at least
    `threshold * half_width` pixels of ink. Sweeping thresholds from
    strict to loose lets the same algorithm handle clean binary scans
    and noisy RGB scans without per-page tuning.
    """
    ink = 255 - grayscale
    left_half = ink[:, :column_mid_x].sum(axis=1).astype(np.float64) / 255.0
    right_half = ink[:, column_mid_x:].sum(axis=1).astype(np.float64) / 255.0
    left_w = float(column_mid_x)
    right_w = float(w - column_mid_x)

    for threshold in _ROW_LINE_THRESHOLDS:
        inky = (left_half > threshold * left_w) | (right_half > threshold * right_w)
        runs = _coalesce_runs(np.where(inky)[0])
        if len(runs) >= _MIN_INKY_RUNS:
            return runs
    return []


def _coalesce_runs(indices: np.ndarray, max_gap: int = 3) -> list[int]:
    """Collapse consecutive y-indices into a single line at their centroid.

    A horizontal line is a few pixels thick; raw `np.where` returns each
    pixel separately. Treat indices within `max_gap` of the previous as
    part of the same line.
    """
    if len(indices) == 0:
        return []
    lines: list[int] = []
    run: list[int] = [int(indices[0])]
    for idx in indices[1:]:
        if idx - run[-1] <= max_gap:
            run.append(int(idx))
        else:
            lines.append(int(np.mean(run)))
            run = [int(idx)]
    lines.append(int(np.mean(run)))
    return lines


def _estimate_row_spacing(row_lines: list[int]) -> float | None:
    """Median pixel distance between consecutive horizontal row lines.

    Returns None when fewer than 2 lines were detected (no diff to take)
    or when the median is non-positive (degenerate input). Both
    `_detect_header_bottom_y` and `_detect_body_mid_y` need this number,
    so it lives in one place.
    """
    if len(row_lines) < 2:
        return None
    spacings = np.diff(np.asarray(row_lines))
    median_spacing = float(np.median(spacings))
    if median_spacing <= 0:
        return None
    return median_spacing


def _detect_header_bottom_y(row_lines: list[int], h: int) -> int:
    """Y of the TOP edge of the top quadrants' Hour-Jock cell.

    The first detected horizontal row line is the BASELINE of the printed
    "Hour ___ Jock ___" underscore line. Character ascenders extend ~30-40px
    above that baseline, so cropping at ``image[0:first_row_y]`` includes
    the Hour-Jock glyphs in the header strip and the model transcribes
    them as page-level oddities. The structurally correct top-of-quadrant
    is one row up — i.e. ``first_row_y - median_row_spacing``. Fall back
    to the fixed-fraction value when row spacing can't be estimated.
    """
    if not row_lines:
        return int(h * FALLBACK_HEADER_FRACTION)
    # Defense: a row-line detected very low on the page is suspicious;
    # the first one should be in the upper third.
    first = row_lines[0]
    if first > 0.3 * h:
        return int(h * FALLBACK_HEADER_FRACTION)
    spacing = _estimate_row_spacing(row_lines)
    if spacing is None:
        return int(h * FALLBACK_HEADER_FRACTION)
    candidate = first - int(spacing)
    # Defense against degenerate spacings: only ~2 detected lines with one
    # straddling the body-midline gap can yield a spacing larger than
    # `first` itself, producing a negative y. Fall back rather than crop
    # outside the page.
    if candidate <= 0:
        return int(h * FALLBACK_HEADER_FRACTION)
    return candidate


def _detect_body_mid_y(row_lines: list[int], h: int) -> int:
    """Midpoint of the gap closest to the structural body midline.

    Picks the qualifying gap (size ≥ MIN_GAP_RATIO × median row spacing)
    whose midpoint is closest to ``h × BODY_MID_ANCHOR_FRACTION``. Picking
    by *size alone* is fragile: handwriting that obliterates row lines in
    the bottom block can produce spurious large gaps that beat the real
    body-midline gap on size. Anchoring picks the structurally-correct
    one even when the bottom block has noisier line detection than the
    top.
    """
    median_spacing = _estimate_row_spacing(row_lines)
    if median_spacing is None:
        return _fallback_body_mid_y(h)

    band_lo, band_hi = _BODY_MID_SEARCH_BAND[0] * h, _BODY_MID_SEARCH_BAND[1] * h
    central = [y for y in row_lines if band_lo <= y <= band_hi]
    anchor = h * _BODY_MID_ANCHOR_FRACTION

    best_gap_mid: int | None = None
    best_distance = float("inf")
    for i in range(len(central) - 1):
        gap = central[i + 1] - central[i]
        if gap < _MIN_GAP_RATIO * median_spacing:
            continue
        midpoint = (central[i] + central[i + 1]) // 2
        distance = abs(midpoint - anchor)
        if distance < best_distance:
            best_distance = distance
            best_gap_mid = midpoint

    if best_gap_mid is None:
        return _fallback_body_mid_y(h)
    return best_gap_mid


def _fallback_body_mid_y(h: int) -> int:
    """The historical body-midline fallback: midpoint of the body strip
    when the body starts at FALLBACK_HEADER_FRACTION."""
    body_top = int(h * FALLBACK_HEADER_FRACTION)
    return body_top + (h - body_top) // 2


def _detect_body_bottom_y(row_lines: list[int], h: int) -> int:
    """Y of the printed `Comments:` line — top edge of the footer strip.

    Picks the LAST detected horizontal line in [0.95h, 0.99h]. The Comments
    line is the lowest printed horizontal feature on the form; everything
    below it is handwritten marginalia that doesn't belong to either body
    or comments structurally. Falls back to int(h * 0.97) when no line is
    detected in that band (degraded scans where the printed Comments line
    is too disrupted to clear the row-line threshold).
    """
    band_lo, band_hi = _BODY_BOTTOM_SEARCH_BAND[0] * h, _BODY_BOTTOM_SEARCH_BAND[1] * h
    in_band = [y for y in row_lines if band_lo <= y <= band_hi]
    if not in_band:
        return int(h * FALLBACK_BODY_BOTTOM_FRACTION)
    return in_band[-1]


def partition_row_lines_by_quadrant(
    image: PILImage, layout: PageLayout
) -> dict[QuadrantPosition, list[int]]:
    """Detected row-line y-coords, partitioned by quadrant of the body grid.

    Reuses `_detect_row_lines` for the y-coordinates, then classifies each
    line by which page-column it spans (left, right, or both, based on ink
    density at that y) and which body band it sits in (top vs bottom, by
    `layout.body_mid_y`).

    A line spanning both columns is added to BOTH side quadrants — most
    printed flowsheet grid lines run full-width and bracket both hour-blocks
    of a row.

    Lines outside `[layout.header_bottom_y, layout.body_bottom_y)` are
    dropped (header or footer artifacts, not body rows).

    Returns a dict with all four `QUADRANT_ORDER` keys; empty list when
    no lines hit a quadrant (blank image, un-printed margin).
    """
    w, _h = image.size
    grayscale = np.asarray(image.convert("L"))
    col_mid = layout.column_mid_x

    all_lines = _detect_row_lines(grayscale, w, col_mid)

    ink = (255 - grayscale).astype(np.float64) / 255.0
    left_w = float(col_mid)
    right_w = float(w - col_mid)
    threshold = _ROW_LINE_THRESHOLDS[-1]

    out: dict[QuadrantPosition, list[int]] = {q: [] for q in QUADRANT_ORDER}
    for y in all_lines:
        if not (layout.header_bottom_y <= y < layout.body_bottom_y):
            continue
        left_ink = float(ink[y, :col_mid].sum())
        right_ink = float(ink[y, col_mid:].sum())
        on_left = left_ink > threshold * left_w
        on_right = right_ink > threshold * right_w
        if y < layout.body_mid_y:
            if on_left:
                out["top_left"].append(int(y))
            if on_right:
                out["top_right"].append(int(y))
        else:
            if on_left:
                out["bottom_left"].append(int(y))
            if on_right:
                out["bottom_right"].append(int(y))

    # Correction pass: on some pages `_detect_body_mid_y` lands BELOW the
    # bottom-block hour-jock-cell baseline (the anchor at 0.55h prefers the
    # gap below the cell over the true inter-block gap above it). The
    # baseline line then gets misattributed to the top quadrant, and the
    # bottom quadrant's first detected line is row 0's BOTTOM rather than
    # its top — shifting every row crop up by one.
    #
    # Signal: the top quadrant's last spacing is significantly larger than
    # the median row spacing across all detected lines (a normal sequence
    # has consistent spacing; an anomalous jump at the end means the last
    # line belongs to a different sequence — the bottom block).
    if len(all_lines) >= 2:
        median_spacing = float(np.median(np.diff(np.asarray(all_lines))))
        if median_spacing > 0:
            for top_pos, bottom_pos in (
                ("top_left", "bottom_left"),
                ("top_right", "bottom_right"),
            ):
                top_lines = out[top_pos]  # type: ignore[index]
                if len(top_lines) >= 2:
                    last_spacing = top_lines[-1] - top_lines[-2]
                    if last_spacing > _BOTTOM_BASELINE_REATTRIBUTION_RATIO * median_spacing:
                        moved = top_lines.pop()
                        out[bottom_pos].insert(0, moved)  # type: ignore[index]
    return out
