"""Tests for `scripts/make_verifier_bundle.py`.

The pre-processor turns a `PageResult` + page image into a `bundle.json`
the verifier UI consumes. Tests cover the geometry helpers, the bbox
assignment heuristic, the bundle assembly, and the CLI.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from core.page_layout import PageLayout
from core.schema import QUADRANT_ORDER, Entry, PageResult, Quadrant
from scripts.make_verifier_bundle import (
    SCHEMA_VERSION,
    _assign_row_bboxes,
    _merge_with_spans,
    _parse_job_key_from_result_path,
    _quadrant_bboxes,
    main,
    make_bundle,
)


def _layout(
    *,
    header_bottom_y: int = 100,
    body_mid_y: int = 600,
    body_bottom_y: int = 1100,
    column_mid_x: int = 500,
) -> PageLayout:
    return PageLayout(
        header_bottom_y=header_bottom_y,
        body_mid_y=body_mid_y,
        body_bottom_y=body_bottom_y,
        column_mid_x=column_mid_x,
    )


def _entry(row_index: int, text: str = "X - Y") -> Entry:
    return Entry(row_index=row_index, raw_text=text, confidence="high")


def _quad(position: str, n_entries: int) -> Quadrant:
    return Quadrant(
        position=position,  # type: ignore[arg-type]
        hour_raw=None,
        jock_raw=None,
        entries=[_entry(i) for i in range(n_entries)],
    )


def _page_result(*, comments: str | None = None) -> PageResult:
    return PageResult(
        page_date_raw="Mon 1 Jan 90",
        quadrants=[_quad(p, 3) for p in QUADRANT_ORDER],
        comments_raw=comments,
        oddities=[],
        model_version="test-model",
        extracted_at=datetime(2026, 5, 10, tzinfo=UTC),
    )


# -- _quadrant_bboxes -------------------------------------------------------


def test_quadrant_bboxes_returns_all_four_quadrants() -> None:
    boxes = _quadrant_bboxes(_layout(), page_width=1000)
    assert set(boxes.keys()) == set(QUADRANT_ORDER)


def test_quadrant_bboxes_match_layout_math() -> None:
    """Each quadrant's bbox is bounded by the corresponding layout
    coordinates: column_mid_x splits left/right; body_mid_y splits
    top/bottom; header_bottom_y is the top of the body; body_bottom_y
    is the bottom."""
    layout = _layout(
        header_bottom_y=100,
        body_mid_y=600,
        body_bottom_y=1100,
        column_mid_x=500,
    )
    boxes = _quadrant_bboxes(layout, page_width=1000)
    assert boxes["top_left"] == (0, 100, 500, 600)
    assert boxes["top_right"] == (500, 100, 1000, 600)
    assert boxes["bottom_left"] == (0, 600, 500, 1100)
    assert boxes["bottom_right"] == (500, 600, 1000, 1100)


# -- _assign_row_bboxes -----------------------------------------------------


def test_assign_row_bboxes_clean_pairing() -> None:
    """When all spans are 1 and n_lines == n_entries + 1, consecutive line
    pairs become row top/bottom for each entry."""
    quad_bbox = (0, 100, 500, 400)  # height 300
    lines = [100, 200, 300, 400]  # 4 lines -> 3 entries
    rows = _assign_row_bboxes(quad_bbox, lines, spans=[1, 1, 1])
    assert rows == [
        (0, 100, 500, 200),
        (0, 200, 500, 300),
        (0, 300, 500, 400),
    ]


def test_assign_row_bboxes_extra_lines_ignored() -> None:
    """When more lines exist than the entry-spans require, the trailing
    lines are ignored."""
    quad_bbox = (0, 100, 500, 700)
    lines = [100, 200, 300, 400, 500, 600, 700]  # 7 lines
    rows = _assign_row_bboxes(quad_bbox, lines, spans=[1, 1, 1])
    assert rows == [
        (0, 100, 500, 200),
        (0, 200, 500, 300),
        (0, 300, 500, 400),
    ]


def test_assign_row_bboxes_spans_skip_continuation_rows() -> None:
    """When an entry's span is 2 (it absorbed a continuation row or is
    double_height), its bbox spans two physical row lines, and the NEXT
    entry's bbox starts after the second line. This is the load-bearing
    behavior for the multiline-entry verifier case."""
    quad_bbox = (0, 800, 1000, 1100)
    # Three physical rows: y=800-900, 900-1000, 1000-1100.
    # Two logical entries: first spans rows 0-1 (continuation), second is row 2.
    lines = [800, 900, 1000, 1100]
    rows = _assign_row_bboxes(quad_bbox, lines, spans=[2, 1])
    assert rows == [
        (0, 800, 1000, 1000),  # entry 0: spans first TWO physical rows
        (0, 1000, 1000, 1100),  # entry 1: third physical row, not second
    ]


def test_assign_row_bboxes_falls_back_to_even_spacing_when_no_lines() -> None:
    quad_bbox = (10, 100, 510, 400)  # width 500, height 300
    rows = _assign_row_bboxes(quad_bbox, lines=[], spans=[1, 1, 1])
    assert rows == [
        (10, 100, 510, 200),
        (10, 200, 510, 300),
        (10, 300, 510, 400),
    ]


def test_assign_row_bboxes_falls_back_to_even_spacing_when_too_few_lines() -> None:
    """When detected lines don't cover the total physical row count, even-
    spacing fallback divides the quadrant by entry count (not physical
    count) — uniform strips are better UX than mis-paired pinned rows."""
    quad_bbox = (0, 100, 500, 700)
    rows = _assign_row_bboxes(quad_bbox, lines=[100, 300], spans=[1, 1, 1])
    assert rows == [
        (0, 100, 500, 300),
        (0, 300, 500, 500),
        (0, 500, 500, 700),
    ]


def test_assign_row_bboxes_returns_empty_for_zero_entries() -> None:
    rows = _assign_row_bboxes((0, 0, 100, 100), lines=[10, 20, 30], spans=[])
    assert rows == []


# -- _merge_with_spans ------------------------------------------------------


def test_merge_with_spans_collapses_continuation_into_span() -> None:
    """A continuation entry merges into the previous logical entry and
    increments its physical-row span by 1."""
    entries = [
        Entry(row_index=0, raw_text="The Standells - Sometimes Good Guys", confidence="high"),
        Entry(
            row_index=1,
            raw_text="Don't Wear White",
            confidence="medium",
            notes="continuation",
        ),
        Entry(row_index=2, raw_text="The Lovedolls - Pearls at Swine", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 2
    merged_first, span_first = result[0]
    assert merged_first.raw_text == "The Standells - Sometimes Good Guys Don't Wear White"
    assert span_first == 2
    # Merged entries inherit `double_height` notes so the verifier dropdown
    # reflects the multi-row nature.
    assert merged_first.notes == "double_height"
    merged_second, span_second = result[1]
    assert merged_second.raw_text == "The Lovedolls - Pearls at Swine"
    assert span_second == 1
    assert merged_second.notes is None


def test_merge_with_spans_double_height_counts_as_two() -> None:
    """`notes="double_height"` doesn't trigger a merge but spans 2 rows."""
    entries = [
        Entry(row_index=0, raw_text="X - Y", confidence="high", notes="double_height"),
        Entry(row_index=1, raw_text="A - B", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert [span for _, span in result] == [2, 1]
    assert result[0][0].raw_text == "X - Y"


def test_merge_with_spans_consecutive_continuations() -> None:
    """A single entry can absorb multiple continuation rows; span grows by
    one per continuation."""
    entries = [
        Entry(row_index=0, raw_text="Line A", confidence="high"),
        Entry(row_index=1, raw_text="Line B", confidence="high", notes="continuation"),
        Entry(row_index=2, raw_text="Line C", confidence="high", notes="continuation"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 1
    merged, span = result[0]
    assert merged.raw_text == "Line A Line B Line C"
    assert span == 3
    assert merged.notes == "double_height"


def test_merge_with_spans_leading_continuation_is_preserved() -> None:
    """A continuation as the first row has nothing to merge into — stays
    as its own entry with span 1, mirroring `merge_continuations`."""
    entries = [
        Entry(row_index=0, raw_text="orphan", confidence="low", notes="continuation"),
        Entry(row_index=1, raw_text="A - B", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 2
    assert [span for _, span in result] == [1, 1]
    assert result[0][0].raw_text == "orphan"
    assert result[0][0].notes == "continuation"


def test_merge_with_spans_empty_input() -> None:
    assert _merge_with_spans([]) == []


# -- make_bundle ------------------------------------------------------------


def _white_page(tmp_path: Path) -> Path:
    """A synthetic 1000x1500 white image with a black vertical column
    divider at x=500. Detection will land near-real coords, then we
    don't care about per-row exactness — the bundle just needs to
    assemble without crashing."""
    image = Image.new("RGB", (1000, 1500), color="white")
    # Paint the column divider so detect_column_mid_x finds it.
    for y in range(1500):
        image.putpixel((500, y), (0, 0, 0))
    path = tmp_path / "page.png"
    image.save(path)
    return path


def test_make_bundle_returns_schema_version(tmp_path: Path) -> None:
    image_path = _white_page(tmp_path)
    bundle_path = tmp_path / "out" / "verifier" / "page.bundle.json"
    bundle = make_bundle(_page_result(), image_path=image_path, bundle_path=bundle_path)
    assert bundle["schema_version"] == SCHEMA_VERSION == 2


def test_make_bundle_top_level_fields(tmp_path: Path) -> None:
    image_path = _white_page(tmp_path)
    bundle_path = tmp_path / "page.bundle.json"
    bundle = make_bundle(_page_result(), image_path=image_path, bundle_path=bundle_path)
    assert bundle["stem"] == "page"
    assert bundle["page_date_raw"] == "Mon 1 Jan 90"
    assert bundle["comments_raw"] is None
    assert bundle["model_version"] == "test-model"
    assert bundle["oddities"] == []
    assert len(bundle["quadrants"]) == 4
    # New in v2: job key fields default to null when no job_key is passed.
    assert bundle["pdf_path"] is None
    assert bundle["page_number"] is None


def test_make_bundle_carries_job_key_when_provided(tmp_path: Path) -> None:
    """When the bundle pre-processor can recover the (pdf_path, page_number)
    job key from the result path, it's preserved in the bundle so the
    verifier UI can target the right jobs.db row on save."""
    image_path = _white_page(tmp_path)
    bundle = make_bundle(
        _page_result(),
        image_path=image_path,
        bundle_path=tmp_path / "out.bundle.json",
        job_key=("1990/April 1990/1990-04apr0106.pdf", 25),
    )
    assert bundle["pdf_path"] == "1990/April 1990/1990-04apr0106.pdf"
    assert bundle["page_number"] == 25


def test_make_bundle_image_path_is_relative_to_bundle_dir(tmp_path: Path) -> None:
    """The bundle stays portable: image_path is computed via os.path.relpath
    from the bundle's parent directory to the source image. Tests nested
    subdirectories — the bundle in data/verifier/, image in data/pages/<rel>/."""
    data = tmp_path / "data"
    image_path = data / "pages" / "1990-04apr0106" / "page-05.png"
    image_path.parent.mkdir(parents=True)
    image = Image.new("RGB", (1000, 1500), color="white")
    for y in range(1500):
        image.putpixel((500, y), (0, 0, 0))
    image.save(image_path)

    bundle_path = data / "verifier" / "page-05.bundle.json"
    bundle = make_bundle(_page_result(), image_path=image_path, bundle_path=bundle_path)
    assert bundle["image_path"] == "../pages/1990-04apr0106/page-05.png"


def test_make_bundle_quadrants_in_canonical_order(tmp_path: Path) -> None:
    image_path = _white_page(tmp_path)
    bundle = make_bundle(
        _page_result(), image_path=image_path, bundle_path=tmp_path / "out.bundle.json"
    )
    positions = tuple(q["position"] for q in bundle["quadrants"])
    assert positions == QUADRANT_ORDER


def test_make_bundle_each_entry_has_row_bbox(tmp_path: Path) -> None:
    image_path = _white_page(tmp_path)
    bundle = make_bundle(
        _page_result(), image_path=image_path, bundle_path=tmp_path / "out.bundle.json"
    )
    for quad in bundle["quadrants"]:
        for entry in quad["entries"]:
            assert "row_bbox" in entry
            bbox = entry["row_bbox"]
            assert len(bbox) == 4
            x1, y1, x2, y2 = bbox
            assert x2 > x1 and y2 > y1, f"degenerate bbox: {bbox}"


def test_make_bundle_quadrant_has_bbox(tmp_path: Path) -> None:
    image_path = _white_page(tmp_path)
    bundle = make_bundle(
        _page_result(), image_path=image_path, bundle_path=tmp_path / "out.bundle.json"
    )
    for quad in bundle["quadrants"]:
        assert "bbox" in quad
        assert len(quad["bbox"]) == 4


# -- CLI --------------------------------------------------------------------


def _write_minimal_result(path: Path) -> None:
    page = _page_result()
    path.write_text(page.model_dump_json(indent=2))


def test_main_writes_bundle_to_out_path(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)

    out_path = tmp_path / "out" / "page.bundle.json"
    rc = main([str(result_path), str(image_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()
    bundle = json.loads(out_path.read_text())
    assert bundle["schema_version"] == SCHEMA_VERSION
    assert len(bundle["quadrants"]) == 4


def test_main_creates_output_parent_directory(tmp_path: Path) -> None:
    """Pre-processor creates output dirs that don't exist, matching the
    pattern in core/pipeline.py and core/jobs.py."""
    result_path = tmp_path / "result.json"
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)

    out_path = tmp_path / "deeply" / "nested" / "page.bundle.json"
    assert not out_path.parent.exists()

    rc = main([str(result_path), str(image_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.is_file()


def test_main_validates_bundle_against_page_result_shape(tmp_path: Path) -> None:
    """The bundle must round-trip through PageResult.model_validate_json
    after stripping bundle-only fields. This pins the export-schema
    contract end-to-end."""
    result_path = tmp_path / "result.json"
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)

    out_path = tmp_path / "page.bundle.json"
    main([str(result_path), str(image_path), "--out", str(out_path)])

    bundle = json.loads(out_path.read_text())
    # Strip bundle-only fields.
    for key in ("schema_version", "stem", "image_path"):
        bundle.pop(key, None)
    for quad in bundle["quadrants"]:
        quad.pop("bbox", None)
        for entry in quad["entries"]:
            entry.pop("row_bbox", None)
    PageResult.model_validate(bundle)


# -- _parse_job_key_from_result_path ----------------------------------------


def test_parse_job_key_from_pipeline_path() -> None:
    """The canonical pipeline-result path resolves to (pdf_path, page_number)."""
    p = Path("/var/data/results/1990/April 1990/1990-04apr0106/page-25.json")
    assert _parse_job_key_from_result_path(p) == (
        "1990/April 1990/1990-04apr0106.pdf",
        25,
    )


def test_parse_job_key_returns_none_for_non_pipeline_path() -> None:
    """Test fixtures (/tmp, /private, fixtures/) don't follow the layout."""
    assert _parse_job_key_from_result_path(Path("/tmp/flash-spike/pro/some.json")) is None
    assert _parse_job_key_from_result_path(Path("/Users/x/fixtures/result.json")) is None


def test_parse_job_key_returns_none_when_filename_not_page() -> None:
    """The trailing component must be `page-NN.json`."""
    p = Path("/var/data/results/1990/foo/notpage.json")
    assert _parse_job_key_from_result_path(p) is None


def test_parse_job_key_returns_none_when_page_index_not_numeric() -> None:
    p = Path("/var/data/results/1990/foo/page-abc.json")
    assert _parse_job_key_from_result_path(p) is None


def test_main_returns_nonzero_when_inputs_missing(tmp_path: Path) -> None:
    """Missing input file is a usage error, not a crash. Exit 1 lets
    shell scripts react cleanly."""
    rc = main(
        [
            str(tmp_path / "missing-result.json"),
            str(tmp_path / "missing-page.png"),
            "--out",
            str(tmp_path / "out.bundle.json"),
        ]
    )
    assert rc == 1


@pytest.mark.parametrize(
    ("entry_text", "expected_bbox_count"),
    [
        ("Juana Molina - la paradoja", 1),
        ("", 1),  # blank entries still get a bbox (UI shows them)
    ],
)
def test_make_bundle_handles_entry_text_variants(
    tmp_path: Path, entry_text: str, expected_bbox_count: int
) -> None:
    image_path = _white_page(tmp_path)
    result = PageResult(
        page_date_raw=None,
        quadrants=[
            Quadrant(
                position=p,
                hour_raw=None,
                jock_raw=None,
                entries=[Entry(row_index=0, raw_text=entry_text, confidence="high")],
            )
            for p in QUADRANT_ORDER
        ],
        oddities=[],
        model_version="t",
        extracted_at=datetime(2026, 5, 10, tzinfo=UTC),
    )
    bundle = make_bundle(result, image_path=image_path, bundle_path=tmp_path / "b.json")
    assert all(len(q["entries"]) == expected_bbox_count for q in bundle["quadrants"])
