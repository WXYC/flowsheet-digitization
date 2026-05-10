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

from core.page_layout import PageLayout, detect_page_layout

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"

# Hand-verified by reading projection profiles on each golden:
# - column_mid_x = argmax of vertical-line ink in the central 30-70% x-band
# - header_bottom_y ≈ first horizontal row line of the top quadrants
# - body_mid_y ≈ midpoint of the gap between top-block last row line and
#                bottom-block first row line (the "Hour Jock" band)
GROUND_TRUTH: dict[str, dict[str, int]] = {
    "1990-01jan0106-page05": {"header_bottom_y": 546, "body_mid_y": 2256, "column_mid_x": 1303},
    "1990-04apr0106-page05": {"header_bottom_y": 548, "body_mid_y": 2301, "column_mid_x": 1262},
    "1990-04apr0106-page15": {"header_bottom_y": 545, "body_mid_y": 2254, "column_mid_x": 1267},
    "1990-04apr0106-page20": {"header_bottom_y": 546, "body_mid_y": 2300, "column_mid_x": 1260},
    "1990-04apr0106-page25": {"header_bottom_y": 547, "body_mid_y": 2301, "column_mid_x": 1278},
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
    # Fallback: column at exact horizontal center.
    assert layout.column_mid_x == 2550 // 2


def test_page_layout_is_frozen() -> None:
    """PageLayout is a value type; mutation should raise."""
    layout = PageLayout(header_bottom_y=100, body_mid_y=200, column_mid_x=300)
    # dataclass(frozen=True) raises FrozenInstanceError, an AttributeError subclass.
    with pytest.raises(AttributeError):
        layout.header_bottom_y = 999  # type: ignore[misc]


def test_page_layout_orders_coordinates_consistently(
    golden: tuple[str, Image.Image, dict[str, int]],
) -> None:
    """Detected coords must obey the page-geometry invariant:
    header_bottom_y < body_mid_y, and column_mid_x is somewhere in the page."""
    stem, image, _ = golden
    layout = detect_page_layout(image)
    w, h = image.size
    assert 0 < layout.header_bottom_y < layout.body_mid_y < h, (
        f"{stem}: invariants violated, layout={layout}, image={(w, h)}"
    )
    assert 0 < layout.column_mid_x < w, f"{stem}: column_mid_x out of range, layout={layout}"
