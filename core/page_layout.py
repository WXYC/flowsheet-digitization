"""Detect the printed grid-line coordinates on a WXYC flowsheet page.

The 1990s flowsheet form is a printed grid: a header strip on top, a 2x2
body of "hour blocks" below, and a comments line at the bottom. The
dividers are continuous high-contrast lines at consistent
(per-form-revision) positions. They drift across pages by tens of pixels
because of paper alignment in the scanner — enough that fixed-fraction
cropping clips content or duplicates it across neighboring crops.

`detect_page_layout` returns the three coordinates the cropper needs:

  - `header_bottom_y` — y of the first horizontal row line; everything
    above is the header strip.
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

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Fallback fractions, matching the historical constants in
# `scripts/calibrate_models.py`. Used when detection fails on a page so
# the pipeline never crashes on a corner-case scan.
FALLBACK_HEADER_FRACTION = 0.12

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


@dataclass(frozen=True)
class PageLayout:
    """Pixel coordinates of the printed grid lines on a flowsheet page.

    The page splits into:
      - header strip:   image[:header_bottom_y]
      - top quadrants:  image[header_bottom_y:body_mid_y, :/column_mid_x:]
      - bottom quadrants: image[body_mid_y:, :/column_mid_x:]
    """

    header_bottom_y: int
    body_mid_y: int
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

    return PageLayout(
        header_bottom_y=header_bottom_y,
        body_mid_y=body_mid_y,
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


def _detect_header_bottom_y(row_lines: list[int], h: int) -> int:
    """Y of the first row line, or fallback if no lines detected.

    The first horizontal row line is the bottom edge of the top
    quadrants' "Hour ___ Jock ___" header — equivalently, the start
    of the body grid. Everything above is the header strip.
    """
    if not row_lines:
        return int(h * FALLBACK_HEADER_FRACTION)
    # Defense: a row-line detected very low on the page is suspicious;
    # the first one should be in the upper third.
    first = row_lines[0]
    if first > 0.3 * h:
        return int(h * FALLBACK_HEADER_FRACTION)
    return first


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
    if len(row_lines) < 2:
        return _fallback_body_mid_y(h)

    spacings = np.diff(np.asarray(row_lines))
    median_spacing = float(np.median(spacings))
    if median_spacing <= 0:
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
