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


def test_assign_row_bboxes_falls_back_to_line_anchored_spacing_when_too_few_lines() -> None:
    """When detected lines don't cover the total physical row count, anchor
    each entry's bbox to the first detected line and use the median gap
    between detected lines as the per-row height.

    Why this beats the old "even-space the whole quadrant" fallback: the
    quadrant body includes the Hour/Jock header band at the top, so even-
    spacing put entry 0's crop on top of the header instead of the first
    handwritten row. Anchoring to lines[0] skips that header band, and
    median-gap row heights stay aligned with the printed grid even when
    the model over-emits past the line count."""
    quad_bbox = (0, 100, 500, 700)
    rows = _assign_row_bboxes(quad_bbox, lines=[150, 250, 350], spans=[1, 1, 1, 1, 1])
    # First line is at 150 (skipping the 50px header band between 100 and 150).
    # Median gap is 100. Each entry is 100 tall. Entries past the last
    # detected line continue at the same cadence until they hit y2 = 700.
    assert rows == [
        (0, 150, 500, 250),
        (0, 250, 500, 350),
        (0, 350, 500, 450),
        (0, 450, 500, 550),
        (0, 550, 500, 650),
    ]


def test_assign_row_bboxes_fallback_respects_span_lengths() -> None:
    """A double_height (span=2) entry gets a row strip twice the median gap."""
    quad_bbox = (0, 100, 500, 800)
    rows = _assign_row_bboxes(quad_bbox, lines=[150, 250, 350], spans=[2, 1, 1, 1])
    # Median gap 100. Entry 0 covers 2 row heights from the first line.
    assert rows == [
        (0, 150, 500, 350),
        (0, 350, 500, 450),
        (0, 450, 500, 550),
        (0, 550, 500, 650),
    ]


def test_assign_row_bboxes_prepends_inferred_first_line_for_bottom_quadrant() -> None:
    """The bottom-quadrant line detector sometimes misses the line just
    below the Hour/Jock cell on the page where it's broken by handwriting
    or noise. That makes `lines[0]` row 0's BOTTOM rather than its TOP,
    shifting every crop up one row. When the gap from the quadrant body
    top to `lines[0]` is much larger than the median spacing between
    detected lines, infer the missing line and prepend it."""
    quad_bbox = (0, 2200, 500, 4070)
    # lines have gaps of 75 each, so median = 75.
    # Gap from y1=2200 to lines[0]=2350 is 150, ~2× the median.
    # Infer a missing line at 2350-75=2275 and treat it as the new lines[0].
    rows = _assign_row_bboxes(
        quad_bbox,
        lines=[2350, 2425, 2500],
        spans=[1, 1, 1, 1],
    )
    assert rows == [
        (0, 2275, 500, 2350),
        (0, 2350, 500, 2425),
        (0, 2425, 500, 2500),
        # Entry 3 extends past detected lines using median-gap cadence.
        (0, 2500, 500, 2575),
    ]


def test_assign_row_bboxes_does_not_prepend_when_first_line_close_to_top() -> None:
    """If `lines[0]` is already close to the quadrant body top (within
    ~1 median gap), the first row line was detected normally — no
    inference needed."""
    quad_bbox = (0, 475, 500, 2205)
    # Lines have gaps of 75 each (median = 75). Gap from y1=475 to
    # lines[0]=550 is 75 — exactly one median. Don't prepend.
    rows = _assign_row_bboxes(
        quad_bbox,
        lines=[550, 625, 700],
        spans=[1, 1, 1, 1],
    )
    # Median-gap fallback anchors entry 0 at lines[0]=550, no prepended line.
    assert rows[0] == (0, 550, 500, 625)


def test_assign_row_bboxes_drops_misattributed_lines_before_quad_top() -> None:
    """`partition_row_lines_by_quadrant`'s reattribution pass sometimes
    moves a line from the body-midline gap into the bottom quadrant. That
    line sits BEFORE the quadrant body top — using it as the row 0 anchor
    puts the printed line in the middle of the crop instead of at the row
    boundary. Drop such lines before computing row strips."""
    quad_bbox = (0, 2352, 500, 4070)
    # 2256 sits before y1=2352; drop it. The remaining lines [2357, 2433, 2505]
    # have gaps of 76 each. After the drop the gap from y1 to lines[0]=2357
    # is only 5px, which is too small for a real Hour/Jock cell — infer a
    # missing line at 2357-76=2281 so row 0 lands on the actual handwriting.
    rows = _assign_row_bboxes(
        quad_bbox,
        lines=[2256, 2357, 2433, 2505],
        spans=[1, 1, 1, 1],
    )
    assert rows[0] == (0, 2281, 500, 2357)
    assert rows[1] == (0, 2357, 500, 2433)
    assert rows[2] == (0, 2433, 500, 2509)
    assert rows[3] == (0, 2509, 500, 2585)


def test_assign_row_bboxes_fallback_squeezes_when_lines_undercover() -> None:
    """If detected row_height * n_entries would overflow the quadrant
    body, squeeze row_height so every entry gets a positive-height bbox.
    Zero-height bboxes are invisible in the verifier UI and defeat
    per-row verification, so any overflow is recomputed as an even
    division of the remaining space."""
    quad_bbox = (0, 100, 500, 400)
    rows = _assign_row_bboxes(quad_bbox, lines=[150, 250], spans=[1, 1, 1, 1])
    # Detected row_height = 100 but 4 entries × 100 = 400 > body height 250.
    # Squeezed row_height = 250 // 4 = 62. Each entry gets a 62px-tall strip.
    assert all(y2 > y1 for _, y1, _, y2 in rows), "no zero-height bboxes allowed"
    assert rows[0][1] == 150  # first row anchored at y_top
    # Final row should end at or before quad_bbox.y2.
    assert rows[-1][3] <= 400


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


def test_merge_with_spans_drops_crossed_out_tag() -> None:
    """Empirical precision of Gemini's `crossed_out` is ~22% (8 false
    positives per 11 emits, measured n=20 on 1990-04apr0106 and reproduced
    on 1990-04apr1318). Stripping the tag at bake time eliminates the
    false-positive review action; Alex marks the few true positives
    himself by toggling the dropdown. The raw_text is preserved verbatim —
    only the notes value is reset."""
    entries = [
        Entry(row_index=0, raw_text="Pixies - Debaser", confidence="high", notes="crossed_out"),
        Entry(row_index=1, raw_text="Sonic Youth - Sugar Kane", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 2
    merged_first, span_first = result[0]
    assert merged_first.notes is None, "expected crossed_out to be stripped from the bundle output"
    assert merged_first.raw_text == "Pixies - Debaser"
    assert span_first == 1


def test_merge_with_spans_drops_crossed_out_before_continuation_merge() -> None:
    """`crossed_out` stripping must happen before the continuation merge,
    so a crossed_out predecessor doesn't suppress the merge or carry the
    tag onto a logically-multi-row entry."""
    entries = [
        Entry(row_index=0, raw_text="Galaxie 500 -", confidence="high", notes="crossed_out"),
        Entry(row_index=1, raw_text="Tugboat", confidence="medium", notes="continuation"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 1
    merged, span = result[0]
    assert merged.raw_text == "Galaxie 500 - Tugboat"
    assert span == 2
    # The merged predecessor's tag should be `double_height` from the merge
    # rule, NOT `crossed_out` (which we just stripped).
    assert merged.notes == "double_height"


def test_merge_with_spans_reindexes_to_close_gaps_after_merge() -> None:
    """After absorbing continuation rows into their predecessors, the
    remaining entries must have contiguous row_index values 0..n-1. The
    verifier UI's add-row handler (verifier/app.js) assigns
    `row_index = quad.entries.length` and assumes per-quadrant uniqueness;
    a sparse row_index (e.g. [0,1,3,4]) lets the UI mint a new row whose
    row_index collides with an existing entry."""
    entries = [
        Entry(row_index=0, raw_text="The Standells - Sometimes Good Guys", confidence="high"),
        Entry(
            row_index=1,
            raw_text="Don't Wear White",
            confidence="medium",
            notes="continuation",
        ),
        Entry(row_index=2, raw_text="The Lovedolls - Pearls at Swine", confidence="high"),
        Entry(row_index=3, raw_text="Pavement - Box Elder", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 3
    assert [e.row_index for e, _ in result] == [0, 1, 2], (
        "row_index must be contiguous 0..n-1 after the merge so the verifier UI's "
        "add-row handler can mint a unique new row_index"
    )


def test_merge_with_spans_drops_empty_raw_text_with_null_notes() -> None:
    """Gemini occasionally emits a phantom row with raw_text='', confidence='low',
    and notes=None — a structurally-blank entry the model couldn't classify as
    illegible/continuation/double_height/crossed_out. Such rows render as empty
    input fields in the verifier UI with no visual cue, and `scripts.derive_truth`
    silently drops them, breaking row-count parity. Drop them at bake time."""
    entries = [
        Entry(row_index=0, raw_text="Pixies - Debaser", confidence="high"),
        Entry(row_index=1, raw_text="", confidence="low"),  # phantom blank
        Entry(row_index=2, raw_text="Sonic Youth - Sugar Kane", confidence="high"),
        Entry(row_index=3, raw_text="   ", confidence="low"),  # whitespace-only
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 2
    assert [e.raw_text for e, _ in result] == ["Pixies - Debaser", "Sonic Youth - Sugar Kane"]
    assert [e.row_index for e, _ in result] == [0, 1], "row_index must reindex after drop"


def test_maybe_prepend_missing_first_line_never_returns_negative_y() -> None:
    """The prepended `lines[0] - median_gap` can go negative when the
    detected median_gap is larger than `lines[0]` itself (page header
    layout mis-detection). A negative y in the first row's bbox makes
    `ctx.drawImage` in the verifier UI clip outside the image, showing
    whitespace instead of the row. Must clamp to image-space minimum 0."""
    from scripts.make_verifier_bundle import _maybe_prepend_missing_first_line

    # quad body at y1=10; lines[0]=20 (gap=10), median_gap=200 → prepended=-180.
    quad_bbox = (0, 10, 500, 1500)
    lines = [20, 220, 420, 620]
    out = _maybe_prepend_missing_first_line(quad_bbox, lines)
    assert out[0] >= 0, f"prepended line {out[0]} must be >= 0 (image-space)"


def test_assign_row_bboxes_clamps_y_cursor_at_y1_in_fallback() -> None:
    """The fallback path computes y_top from `lines[0]` (or the helper's
    prepended value); if the helper returns a value < y1, the resulting
    row_bbox's y1 must still be >= the quadrant's y1 so the verifier UI
    never crops outside the quadrant."""
    quad_bbox = (0, 200, 500, 1500)
    # Pathological: the inference would prepend lines[0]-median_gap=-200
    # without clamping; clamped, all row bboxes start at y >= 200.
    rows = _assign_row_bboxes(quad_bbox, lines=[100, 300, 500], spans=[1, 1, 1, 1])
    for _x1, y1, _x2, y2 in rows:
        assert y1 >= 200, f"row top y={y1} must be >= quad y1=200"
        assert y2 <= 1500, f"row bottom y={y2} must be <= quad y2=1500"


def test_assign_row_bboxes_fallback_overflow_rows_have_positive_height() -> None:
    """When the fallback row_height + n_entries puts the cursor past y2,
    all subsequent rows currently collapse to zero-height bboxes at y2
    (y_cursor == y_end == y2). The fix: a zero-height row is invisible in
    the verifier UI; better to either clamp the height down OR drop the
    overflow rows. This test asserts the new contract: no row in a
    returned bbox list has y_end == y_start (i.e., no zero-height bboxes)."""
    quad_bbox = (0, 200, 500, 600)  # height 400
    # row_height from lines = 200; 5 entries needs 5 * 200 = 1000px, but
    # the body is only 400 tall — overflow at entry 3.
    rows = _assign_row_bboxes(quad_bbox, lines=[200, 400, 600], spans=[1, 1, 1, 1, 1])
    for _x1, y1, _x2, y2 in rows:
        assert y2 > y1, f"row bbox must have positive height; got y1={y1}, y2={y2}"


def test_merge_with_spans_keeps_empty_raw_text_when_tagged_illegible() -> None:
    """An illegible-tagged row with raw_text='' is information: the model
    saw something but couldn't read it. The verifier UI surfaces it as a
    flagged row for the volunteer to attempt manually. Do not drop it."""
    entries = [
        Entry(row_index=0, raw_text="Pixies - Debaser", confidence="high"),
        Entry(row_index=1, raw_text="", confidence="low", notes="illegible"),
        Entry(row_index=2, raw_text="Sonic Youth - Sugar Kane", confidence="high"),
    ]
    result = _merge_with_spans(entries)
    assert len(result) == 3
    assert result[1][0].notes == "illegible"
    assert result[1][0].raw_text == ""


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


def test_make_bundle_stem_is_pdfstem_pageNN_when_job_key_present(tmp_path: Path) -> None:
    """The pipeline's default image filename is just `page-NN.png` — so an
    `image_path.stem`-based bundle stem collides across PDFs. When the
    job key is present we derive a corpus-unique stem instead:
    `<pdf-stem>-page<NN>` (matching the existing test-fixture convention)."""
    image_path = _white_page(tmp_path)
    bundle = make_bundle(
        _page_result(),
        image_path=image_path,
        bundle_path=tmp_path / "out.bundle.json",
        job_key=("1990/April 1990/1990-04apr0106.pdf", 7),
    )
    assert bundle["stem"] == "1990-04apr0106-page07"


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


def test_main_explicit_pdf_path_and_page_number_flags(tmp_path: Path) -> None:
    """When the result path doesn't follow the canonical
    `data/results/<rel>/page-NN.json` layout (one-shot rebake scripts,
    test fixtures, ad-hoc dumps), `--pdf-path` and `--page-number` flags
    let the caller supply the job key explicitly so the baked bundle
    still carries corpus-unique `stem`, `pdf_path`, and `page_number`."""
    result_path = tmp_path / "result.json"  # NOT a results/-layout path
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)

    out_path = tmp_path / "out.bundle.json"
    rc = main(
        [
            str(result_path),
            str(image_path),
            "--out",
            str(out_path),
            "--pdf-path",
            "1990/April 1990/1990-04apr0106.pdf",
            "--page-number",
            "25",
        ]
    )
    assert rc == 0
    bundle = json.loads(out_path.read_text())
    assert bundle["pdf_path"] == "1990/April 1990/1990-04apr0106.pdf"
    assert bundle["page_number"] == 25
    assert bundle["stem"] == "1990-04apr0106-page25"


def test_main_explicit_pdf_path_overrides_path_inference(tmp_path: Path) -> None:
    """If the result path happens to follow the canonical layout AND the
    caller passes explicit flags, the explicit values win — useful when
    re-keying an existing extraction."""
    results_dir = tmp_path / "results" / "1990" / "April 1990" / "1990-04apr0106"
    results_dir.mkdir(parents=True)
    result_path = results_dir / "page-25.json"
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)

    out_path = tmp_path / "out.bundle.json"
    rc = main(
        [
            str(result_path),
            str(image_path),
            "--out",
            str(out_path),
            "--pdf-path",
            "1990/April 1990/1990-04apr0712.pdf",
            "--page-number",
            "7",
        ]
    )
    assert rc == 0
    bundle = json.loads(out_path.read_text())
    assert bundle["pdf_path"] == "1990/April 1990/1990-04apr0712.pdf"
    assert bundle["page_number"] == 7
    assert bundle["stem"] == "1990-04apr0712-page07"


def test_main_requires_both_flags_or_neither(tmp_path: Path) -> None:
    """`--pdf-path` and `--page-number` form a job key together. Passing
    one without the other is ambiguous: silently treating it as `None`
    on the missing side would silently fall back to path-parsing and
    confuse the caller. argparse exits with code 2 on bad usage."""
    result_path = tmp_path / "result.json"
    image_path = _white_page(tmp_path)
    _write_minimal_result(result_path)
    out_path = tmp_path / "out.bundle.json"

    # Only --pdf-path: should fail
    with pytest.raises(SystemExit):
        main(
            [
                str(result_path),
                str(image_path),
                "--out",
                str(out_path),
                "--pdf-path",
                "1990/foo.pdf",
            ]
        )


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
    ("entry_text", "notes", "expected_bbox_count"),
    [
        ("Juana Molina - la paradoja", None, 1),
        # Untagged blank entries are phantom rows the model couldn't classify;
        # the baker drops them so they don't surface in the verifier UI.
        ("", None, 0),
        # An illegible-tagged blank IS information (the model saw something
        # it couldn't read). Kept so the volunteer can attempt the row.
        ("", "illegible", 1),
    ],
)
def test_make_bundle_handles_entry_text_variants(
    tmp_path: Path, entry_text: str, notes: str | None, expected_bbox_count: int
) -> None:
    image_path = _white_page(tmp_path)
    result = PageResult(
        page_date_raw=None,
        quadrants=[
            Quadrant(
                position=p,
                hour_raw=None,
                jock_raw=None,
                entries=[
                    Entry(
                        row_index=0,
                        raw_text=entry_text,
                        confidence="high",
                        notes=notes,  # type: ignore[arg-type]
                    )
                ],
            )
            for p in QUADRANT_ORDER
        ],
        oddities=[],
        model_version="t",
        extracted_at=datetime(2026, 5, 10, tzinfo=UTC),
    )
    bundle = make_bundle(result, image_path=image_path, bundle_path=tmp_path / "b.json")
    assert all(len(q["entries"]) == expected_bbox_count for q in bundle["quadrants"])
