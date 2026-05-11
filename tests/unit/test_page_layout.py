"""Tests for `core.page_layout.detect_page_layout`.

Ground-truth grid-line coordinates were captured by inspecting
horizontal/vertical projection profiles of each golden image.
The detector must land within ±10px of these values; that's tight
enough to eliminate the bleed-band regression from #14 and loose enough
to absorb tiny per-page skew differences.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.page_layout import (
    FALLBACK_HEADER_FRACTION,
    PageLayout,
    _detect_header_bottom_y,
    _estimate_row_spacing,
    detect_page_layout,
    partition_row_lines_by_quadrant,
)
from core.schema import QUADRANT_ORDER

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"

# Hand-verified by reading projection profiles on each golden:
# - column_mid_x = argmax of vertical-line ink in the central 30-70% x-band
# - header_bottom_y ≈ first horizontal row line minus median row spacing
#                     (the TOP edge of the Hour-Jock cell, not its baseline —
#                     character ascenders extend above the baseline so the
#                     glyphs of "Hour ___ Jock ___" sit above the first row
#                     line). Derived as first_row_y - median(diff(row_lines)).
# - body_mid_y ≈ midpoint of the gap between top-block last row line and
#                bottom-block first row line (the "Hour Jock" band)
# - body_bottom_y ≈ last horizontal line in [0.95h, 0.99h] (the printed
#                   Comments: line). Pages 1 and 3 have no detected line in
#                   that band so they fall back to int(h * 0.97) = 4074.
GROUND_TRUTH: dict[str, dict[str, int]] = {
    "1990-01jan0106-page05": {
        "header_bottom_y": 471,
        "body_mid_y": 2256,
        "body_bottom_y": 4074,
        "column_mid_x": 1303,
    },
    "1990-04apr0106-page05": {
        "header_bottom_y": 473,
        "body_mid_y": 2301,
        "body_bottom_y": 4068,
        "column_mid_x": 1262,
    },
    "1990-04apr0106-page15": {
        "header_bottom_y": 471,
        "body_mid_y": 2254,
        "body_bottom_y": 4074,
        "column_mid_x": 1267,
    },
    "1990-04apr0106-page20": {
        "header_bottom_y": 471,
        "body_mid_y": 2300,
        "body_bottom_y": 4067,
        "column_mid_x": 1260,
    },
    "1990-04apr0106-page25": {
        "header_bottom_y": 472,
        "body_mid_y": 2301,
        "body_bottom_y": 4065,
        "column_mid_x": 1278,
    },
}

TOLERANCE_PX = 10


@pytest.fixture(params=list(GROUND_TRUTH.keys()))
def golden(request: pytest.FixtureRequest) -> tuple[str, Image.Image, dict[str, int]]:
    stem = request.param
    image = Image.open(GOLDEN_DIR / f"{stem}.png")
    return stem, image, GROUND_TRUTH[stem]


def test_detect_returns_page_layout(golden: tuple[str, Image.Image, dict[str, int]]) -> None:
    _, image, _ = golden
    layout = detect_page_layout(image)
    assert isinstance(layout, PageLayout)


def test_detect_column_mid_x_within_tolerance(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    stem, image, truth = golden
    layout = detect_page_layout(image)
    assert abs(layout.column_mid_x - truth["column_mid_x"]) <= TOLERANCE_PX, (
        f"{stem}: column_mid_x={layout.column_mid_x}, expected ≈{truth['column_mid_x']}"
    )


def test_detect_header_bottom_y_within_tolerance(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    stem, image, truth = golden
    layout = detect_page_layout(image)
    assert abs(layout.header_bottom_y - truth["header_bottom_y"]) <= TOLERANCE_PX, (
        f"{stem}: header_bottom_y={layout.header_bottom_y}, expected ≈{truth['header_bottom_y']}"
    )


def test_detect_body_mid_y_within_tolerance(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    stem, image, truth = golden
    layout = detect_page_layout(image)
    assert abs(layout.body_mid_y - truth["body_mid_y"]) <= TOLERANCE_PX, (
        f"{stem}: body_mid_y={layout.body_mid_y}, expected ≈{truth['body_mid_y']}"
    )


def test_detect_body_bottom_y_within_tolerance(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    stem, image, truth = golden
    layout = detect_page_layout(image)
    assert abs(layout.body_bottom_y - truth["body_bottom_y"]) <= TOLERANCE_PX, (
        f"{stem}: body_bottom_y={layout.body_bottom_y}, expected ≈{truth['body_bottom_y']}"
    )


def test_detect_returns_fallback_on_blank_image() -> None:
    """A pure-white image has no detectable grid lines; the detector must
    still return a usable PageLayout rather than crash. Fallback values
    match the fixed-fraction constants previously used in calibrate_models."""
    blank = Image.new("RGB", (2550, 4200), color="white")
    layout = detect_page_layout(blank)
    # Fallback: header at 12% (matches the old HEADER_STRIP_FRACTION).
    assert layout.header_bottom_y == int(4200 * 0.12)
    # Fallback: body mid is just below page center (matches the old
    # _crop_quadrants math: body_top + (h - body_top) // 2).
    body_top = int(4200 * 0.12)
    expected_body_mid = body_top + (4200 - body_top) // 2
    assert layout.body_mid_y == expected_body_mid
    # Fallback: body bottom at 97% (just above the absolute bottom edge —
    # tracks the y≈4068 observed across the 5 goldens at h=4200).
    assert layout.body_bottom_y == int(4200 * 0.97)
    # Fallback: column at exact horizontal center.
    assert layout.column_mid_x == 2550 // 2


def test_page_layout_is_frozen() -> None:
    """PageLayout is a value type; mutation should raise."""
    layout = PageLayout(
        header_bottom_y=100,
        body_mid_y=200,
        body_bottom_y=380,
        column_mid_x=300,
    )
    # dataclass(frozen=True) raises FrozenInstanceError, an AttributeError subclass.
    with pytest.raises(AttributeError):
        layout.header_bottom_y = 999  # type: ignore[misc]


def test_page_layout_orders_coordinates_consistently(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """Detected coords must obey the page-geometry invariant:
    header_bottom_y < body_mid_y < body_bottom_y < h, and column_mid_x
    is somewhere in the page."""
    stem, image, _ = golden
    layout = detect_page_layout(image)
    w, h = image.size
    assert 0 < layout.header_bottom_y < layout.body_mid_y < layout.body_bottom_y < h, (
        f"{stem}: invariants violated, layout={layout}, image={(w, h)}"
    )
    assert 0 < layout.column_mid_x < w, f"{stem}: column_mid_x out of range, layout={layout}"


# -- _estimate_row_spacing --------------------------------------------------


def test_estimate_row_spacing_returns_none_for_empty_list() -> None:
    assert _estimate_row_spacing([]) is None


def test_estimate_row_spacing_returns_none_for_single_line() -> None:
    """Need at least two lines to take a diff."""
    assert _estimate_row_spacing([100]) is None


def test_estimate_row_spacing_returns_median_diff() -> None:
    """Five lines at 75-px spacing -> median spacing 75."""
    assert _estimate_row_spacing([100, 175, 250, 325, 400]) == 75.0


def test_estimate_row_spacing_robust_to_outlier_gap() -> None:
    """A single 200-px outlier (e.g., the body-midline gap) must not move
    the median away from the regular row spacing."""
    # Eight rows at ~75px, with one 200px gap inserted in the middle.
    lines = [100, 175, 250, 325, 525, 600, 675, 750]
    assert _estimate_row_spacing(lines) == 75.0


def test_estimate_row_spacing_returns_none_for_non_positive_median() -> None:
    """Sorted input is the contract; if a caller passes an unsorted list
    that produces a non-positive median, return None rather than a
    surprising negative spacing."""
    # Strictly decreasing, so all diffs are negative.
    assert _estimate_row_spacing([400, 300, 200, 100]) is None


# -- _detect_header_bottom_y degenerate inputs -----------------------------


def test_detect_header_bottom_y_falls_back_when_subtraction_would_go_negative() -> None:
    """Pathological case: only two detected lines, one near the top and one
    near the body midline, yielding a spacing larger than the first row's
    y-coordinate. Without the clamp we'd return a negative y and crash
    image.crop downstream. Must return the fixed-fraction fallback instead."""
    h = 4200
    # spacing = 2200, first = 300, naive would be 300 - 2200 = -1900.
    result = _detect_header_bottom_y([300, 2500], h)
    assert result == int(h * FALLBACK_HEADER_FRACTION)


def test_detect_header_bottom_y_falls_back_on_empty_input() -> None:
    h = 4200
    assert _detect_header_bottom_y([], h) == int(h * FALLBACK_HEADER_FRACTION)


def test_detect_header_bottom_y_falls_back_when_first_line_too_low() -> None:
    """A first row line below 0.3h is suspicious — likely body content,
    not the body-grid top — fall back to the fixed fraction."""
    h = 4200
    # 0.3 * h = 1260; first at 1500 is too low to trust.
    assert _detect_header_bottom_y([1500, 1575, 1650], h) == int(h * FALLBACK_HEADER_FRACTION)


# -- partition_row_lines_by_quadrant ---------------------------------------


def test_partition_row_lines_returns_quadrant_keys(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """Returned dict has exactly the four quadrant keys in QUADRANT_ORDER."""
    _, image, _ = golden
    layout = detect_page_layout(image)
    partitions = partition_row_lines_by_quadrant(image, layout)
    assert set(partitions.keys()) == set(QUADRANT_ORDER)


def test_partition_row_lines_returns_y_coordinates_as_ints(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """Each list value is a pixel y-coordinate (int), matching the contract
    of `_detect_row_lines`. The verifier pre-processor consumes these as
    crop boundaries, so the integer type is load-bearing."""
    _, image, _ = golden
    layout = detect_page_layout(image)
    partitions = partition_row_lines_by_quadrant(image, layout)
    for ys in partitions.values():
        for y in ys:
            assert isinstance(y, int)


def test_partition_row_lines_within_correct_body_band(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """All returned y-coords fall within the body region and on the
    correct side of body_mid_y for their quadrant."""
    _, image, _ = golden
    layout = detect_page_layout(image)
    partitions = partition_row_lines_by_quadrant(image, layout)
    for pos in ("top_left", "top_right"):
        for y in partitions[pos]:
            assert layout.header_bottom_y <= y < layout.body_mid_y, (
                f"{pos}: y={y} outside top-band [{layout.header_bottom_y}, {layout.body_mid_y})"
            )
    for pos in ("bottom_left", "bottom_right"):
        for y in partitions[pos]:
            assert layout.body_mid_y <= y < layout.body_bottom_y, (
                f"{pos}: y={y} outside bottom-band [{layout.body_mid_y}, {layout.body_bottom_y})"
            )


def test_partition_row_lines_finds_content_in_top_band(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """All 5 goldens have detected lines somewhere in the top body band —
    the printed grid alone is ~9 lines per quadrant, so at least one side
    of the top band must come back populated."""
    stem, image, _ = golden
    layout = detect_page_layout(image)
    partitions = partition_row_lines_by_quadrant(image, layout)
    total_top = len(partitions["top_left"]) + len(partitions["top_right"])
    assert total_top > 0, f"{stem}: no row lines detected in top band"


def test_partition_row_lines_handles_blank_image() -> None:
    """A blank image returns four empty lists — no crash, no missing keys."""
    blank = Image.new("RGB", (1000, 1500), color="white")
    layout = detect_page_layout(blank)
    partitions = partition_row_lines_by_quadrant(blank, layout)
    assert set(partitions.keys()) == set(QUADRANT_ORDER)
    for ys in partitions.values():
        assert ys == []
