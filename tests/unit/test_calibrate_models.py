"""Tests for the small helpers in scripts/calibrate_models.py.

The full adapter implementations are deliberately not tested here — they
import torch + transformers and pull multi-GB checkpoints. These tests
cover the file-mapping and JSON-extraction logic that has to be right
regardless of which model is wrapped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ isn't on sys.path; add it explicitly so we can import the module.
_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import calibrate_models as cm  # noqa: E402

from core.schema import QUADRANT_ORDER  # noqa: E402


def test_stored_result_path_finds_match_in_pdf_subdir(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "1990" / "April 1990" / "1990-04apr2430"
    pdf_dir.mkdir(parents=True)
    target = pdf_dir / "page-28.json"
    target.write_text("{}")
    image = tmp_path / "1990-04apr2430-page28.png"

    found = cm._stored_result_path(image, tmp_path)
    assert found == target


def test_stored_result_path_returns_none_when_pdf_dir_missing(tmp_path: Path) -> None:
    other = tmp_path / "1990" / "March 1990" / "1990-03mar0106"
    other.mkdir(parents=True)
    (other / "page-28.json").write_text("{}")
    image = tmp_path / "1990-04apr2430-page28.png"

    assert cm._stored_result_path(image, tmp_path) is None


def test_stored_result_path_returns_none_for_unparseable_stem(tmp_path: Path) -> None:
    image = tmp_path / "no-page-suffix.png"
    assert cm._stored_result_path(image, tmp_path) is None


def test_stored_result_path_handles_non_zero_padded_page(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "1990-04apr2430"
    pdf_dir.mkdir()
    target = pdf_dir / "page-3.json"
    target.write_text("{}")
    image = tmp_path / "1990-04apr2430-page3.png"

    found = cm._stored_result_path(image, tmp_path)
    assert found == target


def test_extract_json_block_strips_markdown_fence() -> None:
    raw = '```json\n{"a": 1, "b": [2, 3]}\n```\n'
    assert cm._extract_json_block(raw) == '{"a": 1, "b": [2, 3]}'


def test_extract_json_block_handles_nested_braces() -> None:
    raw = 'preamble {"outer": {"inner": "x"}} trailing chatter'
    assert cm._extract_json_block(raw) == '{"outer": {"inner": "x"}}'


def test_extract_json_block_raises_when_no_brace() -> None:
    with pytest.raises(ValueError, match="no '{'"):
        cm._extract_json_block("just prose, no JSON here")


def test_extract_json_block_raises_when_unbalanced() -> None:
    with pytest.raises(ValueError, match="unbalanced"):
        cm._extract_json_block('{"missing": "close"')


def test_select_models_rejects_unknown_model() -> None:
    with pytest.raises(SystemExit, match="unknown model"):
        cm._select_models(["gemini-stored", "not-a-model"])


def test_select_models_strips_whitespace_and_drops_empties() -> None:
    assert cm._select_models([" gemini-stored ", "", "churro"]) == ["gemini-stored", "churro"]


def test_select_models_accepts_modal_names() -> None:
    # The modal-* adapters lazily import `modal`, so the registration table
    # must list them by name even before that import is satisfied. This
    # test catches accidental drops of the modal entries.
    assert cm._select_models(["modal-churro", "modal-qwen-vl", "modal-qwen-vl-quad"]) == [
        "modal-churro",
        "modal-qwen-vl",
        "modal-qwen-vl-quad",
    ]


# -- _wrap_raw_text_as_page_result -----------------------------------------


def test_wrap_raw_text_broadcasts_to_all_four_quadrants() -> None:
    result = cm._wrap_raw_text_as_page_result("LED ZEP - WHOLE LOTTA", model_version="t")
    assert len(result.quadrants) == 4
    for quad in result.quadrants:
        assert len(quad.entries) == 1
        assert quad.entries[0].raw_text == "LED ZEP - WHOLE LOTTA"
    assert result.model_version == "t"


def test_wrap_raw_text_keeps_quadrant_order() -> None:
    result = cm._wrap_raw_text_as_page_result("text", model_version="t")
    positions = [q.position for q in result.quadrants]
    assert positions == ["top_left", "top_right", "bottom_left", "bottom_right"]


def test_wrap_raw_text_handles_empty_string() -> None:
    result = cm._wrap_raw_text_as_page_result("", model_version="t")
    assert result.page_date_raw is None
    for quad in result.quadrants:
        assert quad.entries[0].raw_text == ""


# -- _torch_dtype ----------------------------------------------------------


class _FakeTorch:
    """Stand-in for the torch module so this test stays import-light."""

    float16 = "FP16_SENTINEL"
    bfloat16 = "BF16_SENTINEL"
    float32 = "FP32_SENTINEL"


def test_torch_dtype_passes_auto_through() -> None:
    assert cm._torch_dtype(_FakeTorch, "auto") == "auto"


def test_torch_dtype_maps_named_dtypes() -> None:
    assert cm._torch_dtype(_FakeTorch, "fp16") == "FP16_SENTINEL"
    assert cm._torch_dtype(_FakeTorch, "bf16") == "BF16_SENTINEL"
    assert cm._torch_dtype(_FakeTorch, "fp32") == "FP32_SENTINEL"


def test_torch_dtype_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        cm._torch_dtype(_FakeTorch, "fp8")


# -- _wrap_with_dump -------------------------------------------------------


def test_wrap_with_dump_writes_result_and_returns_unchanged(tmp_path: Path) -> None:
    sentinel = cm._wrap_raw_text_as_page_result("hello world", model_version="t")

    def underlying(_image_path: Path) -> object:
        return sentinel

    decorated = cm._wrap_with_dump(underlying, tmp_path / "churro")  # type: ignore[arg-type]
    image = tmp_path / "1990-04apr0106-page05.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = decorated(image)
    assert result is sentinel

    dump = tmp_path / "churro" / "1990-04apr0106-page05.json"
    assert dump.is_file()
    text = dump.read_text()
    assert '"model_version": "t"' in text
    assert '"raw_text": "hello world"' in text


def test_wrap_with_dump_creates_target_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "dir"

    def underlying(_image_path: Path) -> object:
        return cm._wrap_raw_text_as_page_result("x", model_version="t")

    cm._wrap_with_dump(underlying, target)  # type: ignore[arg-type]
    assert target.is_dir()


def test_select_models_rejects_empty_list() -> None:
    with pytest.raises(SystemExit, match="empty"):
        cm._select_models([""])


# -- _qwen_vl_wire_schema --------------------------------------------------


def test_wire_schema_drops_server_set_fields() -> None:
    schema = cm._qwen_vl_wire_schema()
    properties = schema["properties"]
    assert "model_version" not in properties
    assert "extracted_at" not in properties
    required = schema.get("required", [])
    assert "model_version" not in required
    assert "extracted_at" not in required


def test_wire_schema_keeps_quadrants_required_array() -> None:
    schema = cm._qwen_vl_wire_schema()
    assert "quadrants" in schema["properties"]
    assert "quadrants" in schema["required"]
    quadrants_schema = schema["properties"]["quadrants"]
    assert quadrants_schema["type"] == "array"


def test_wire_schema_resolves_to_quadrant_definition() -> None:
    """The quadrants array must reference Quadrant; xgrammar walks $defs."""
    schema = cm._qwen_vl_wire_schema()
    items = schema["properties"]["quadrants"]["items"]
    ref = items.get("$ref", "")
    assert ref.endswith("/Quadrant"), f"expected quadrants items to $ref Quadrant, got {items!r}"
    defs = schema.get("$defs") or schema.get("definitions") or {}
    assert "Quadrant" in defs
    assert "Entry" in defs


def test_wire_schema_does_not_mutate_pageresult_schema() -> None:
    """Calling the helper should not mutate PageResult.model_json_schema()."""
    from core.schema import PageResult

    before = PageResult.model_json_schema()
    cm._qwen_vl_wire_schema()
    after = PageResult.model_json_schema()
    assert before == after


# -- _quadrant_wire_schema -------------------------------------------------


def test_quadrant_wire_schema_pins_position() -> None:
    """The position field is constrained to a singleton enum so the
    grammar mechanically forbids any wrong label for the cell we cropped."""
    schema = cm._quadrant_wire_schema("top_left")
    assert schema["properties"]["position"] == {"enum": ["top_left"]}


def test_quadrant_wire_schema_keeps_entries_array_and_defs() -> None:
    schema = cm._quadrant_wire_schema("bottom_right")
    entries = schema["properties"]["entries"]
    assert entries["type"] == "array"
    items = entries["items"]
    assert items.get("$ref", "").endswith("/Entry")
    defs = schema.get("$defs") or schema.get("definitions") or {}
    assert "Entry" in defs


def test_quadrant_wire_schema_does_not_mutate_quadrant_schema() -> None:
    from core.schema import Quadrant

    before = Quadrant.model_json_schema()
    cm._quadrant_wire_schema("top_right")
    cm._quadrant_wire_schema("bottom_left")
    after = Quadrant.model_json_schema()
    assert before == after


def test_quadrant_wire_schema_each_position_unique() -> None:
    """Calling once per position must yield distinct pinned schemas."""
    schemas = {pos: cm._quadrant_wire_schema(pos) for pos in QUADRANT_ORDER}
    for pos, schema in schemas.items():
        assert schema["properties"]["position"] == {"enum": [pos]}


# -- _crop_header_strip / _crop_quadrants ---------------------------------


def _painted_page(width: int, height: int) -> object:
    """A page-sized PIL image with each region painted a distinct color so
    crop assignments can be checked by sampling a pixel from the result."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    header_h = int(height * cm.HEADER_STRIP_FRACTION)
    body_top = header_h
    body_h = height - body_top
    mid_x = width // 2
    mid_y = body_top + body_h // 2
    # Paint each region with a distinct fill so the cropped sub-images
    # are identifiable by sampling a pixel near their inner-corner side.
    draw.rectangle((0, 0, width, header_h), fill=(10, 10, 10))  # header
    draw.rectangle((0, body_top, mid_x, mid_y), fill=(255, 0, 0))  # TL
    draw.rectangle((mid_x, body_top, width, mid_y), fill=(0, 255, 0))  # TR
    draw.rectangle((0, mid_y, mid_x, height), fill=(0, 0, 255))  # BL
    draw.rectangle((mid_x, mid_y, width, height), fill=(255, 255, 0))  # BR
    return image


def test_crop_header_strip_takes_top_fraction() -> None:
    image = _painted_page(800, 1000)
    strip = cm._crop_header_strip(image)
    expected_h = int(1000 * cm.HEADER_STRIP_FRACTION)
    assert strip.size == (800, expected_h)
    # The painted header is solid (10,10,10).
    assert strip.getpixel((400, expected_h // 2)) == (10, 10, 10)


def test_crop_quadrants_returns_canonical_keys() -> None:
    image = _painted_page(800, 1000)
    crops = cm._crop_quadrants(image)
    assert tuple(crops.keys()) == QUADRANT_ORDER


def test_crop_quadrants_each_region_carries_its_paint() -> None:
    """The center pixel of each cropped quadrant should be its paint
    color, confirming the crop pulled from the right region of the page."""
    image = _painted_page(800, 1000)
    crops = cm._crop_quadrants(image)
    expected = {
        "top_left": (255, 0, 0),
        "top_right": (0, 255, 0),
        "bottom_left": (0, 0, 255),
        "bottom_right": (255, 255, 0),
    }
    for pos, crop in crops.items():
        w, h = crop.size
        # Sample a point well inside the painted region (away from the
        # bleed band at the inner edges, which intrudes into the neighbor).
        sample_x = w // 4 if pos.endswith("right") else (3 * w) // 4
        sample_y = h // 4 if pos.startswith("bottom") else (3 * h) // 4
        assert crop.getpixel((sample_x, sample_y)) == expected[pos], (
            f"quadrant {pos} sample ({sample_x},{sample_y}) was "
            f"{crop.getpixel((sample_x, sample_y))!r}, expected {expected[pos]!r}"
        )


def test_crop_quadrants_includes_inner_bleed() -> None:
    """Each crop overlaps neighbors by QUADRANT_BLEED on its inner edge."""
    image = _painted_page(1000, 1200)
    crops = cm._crop_quadrants(image)
    body_h = 1200 - int(1200 * cm.HEADER_STRIP_FRACTION)
    bx = int(1000 * cm.QUADRANT_BLEED)
    by = int(body_h * cm.QUADRANT_BLEED)
    # top_left's width is mid_x + bx; mid_x is 500.
    assert crops["top_left"].size[0] == 500 + bx
    # top_left's height is half of body_h + by, measured from body_top.
    assert crops["top_left"].size[1] == body_h // 2 + by
    # bottom_right starts at (mid_x - bx, mid_y - by) and ends at (W, H).
    expected_w = 1000 - (500 - bx)
    expected_h = 1200 - (int(1200 * cm.HEADER_STRIP_FRACTION) + body_h // 2 - by)
    assert crops["bottom_right"].size == (expected_w, expected_h)


def test_crop_quadrants_handles_odd_dimensions() -> None:
    """Off-by-one safety: 1001x1003 should not raise and still produce 4 crops."""
    image = _painted_page(1001, 1003)
    crops = cm._crop_quadrants(image)
    assert tuple(crops.keys()) == QUADRANT_ORDER
    for crop in crops.values():
        assert crop.size[0] > 0 and crop.size[1] > 0


# -- _quadrant_fallback ----------------------------------------------------


def test_quadrant_fallback_builds_low_confidence_entry() -> None:
    """When a quadrant call returns malformed text, the page must still
    validate. The fallback packs the raw text into one Entry, tagged so a
    downstream scorer can tell parse-failed quadrants from real ones."""
    quad = cm._quadrant_fallback("garbled response", "top_right")
    assert quad.position == "top_right"
    assert len(quad.entries) == 1
    entry = quad.entries[0]
    assert entry.raw_text == "garbled response"
    assert entry.confidence == "low"
    assert entry.notes == "parse_failed"
    assert entry.row_index == 0


def test_quadrant_fallback_validates_against_pageresult() -> None:
    """A PageResult assembled from 4 fallback quadrants should validate —
    confirms the fallback satisfies every required field on Quadrant/Entry."""
    from datetime import UTC, datetime

    from core.schema import PageResult

    quads = [cm._quadrant_fallback("x", pos) for pos in QUADRANT_ORDER]
    page = PageResult(
        page_date_raw=None,
        quadrants=quads,
        model_version="test",
        extracted_at=datetime.now(UTC),
    )
    assert [q.position for q in page.quadrants] == list(QUADRANT_ORDER)


# -- _XGrammarLogitsProcessor ----------------------------------------------


class _FakeMatcher:
    """Records every accept_token / fill_next_token_bitmask call."""

    def __init__(self) -> None:
        self.accepted: list[int] = []
        self.fill_calls: list[tuple[object, int]] = []

    def accept_token(self, tok: int) -> bool:
        # Bindings reject anything that isn't a real Python int.
        assert isinstance(tok, int) and not isinstance(tok, bool), (
            f"matcher requires Python int, got {type(tok).__name__}"
        )
        self.accepted.append(tok)
        return True

    def fill_next_token_bitmask(self, bitmask: object, batch_idx: int) -> None:
        self.fill_calls.append((bitmask, batch_idx))


class _FakeBitmask:
    """Stand-in for the torch tensor xgrammar normally allocates."""

    def __init__(self) -> None:
        self.moved_to: list[object] = []

    def to(self, device: object) -> _FakeBitmask:
        self.moved_to.append(device)
        return self


class _FakeTensor:
    """1-D-or-2-D fake supporting shape, slicing, tolist(), and .device."""

    def __init__(self, rows: list[list[int]], device: str = "cpu") -> None:
        self._rows = rows
        self.device = device

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._rows), len(self._rows[0]) if self._rows else 0)

    def __getitem__(self, key: object) -> _FakeTensor:
        # Supports ndarray-like fancy indexing of (row_idx, slice).
        if isinstance(key, tuple) and len(key) == 2:
            row_idx, col_slice = key
            assert isinstance(row_idx, int)
            assert isinstance(col_slice, slice)
            return _FakeTensor([self._rows[row_idx][col_slice]], device=self.device)
        raise TypeError(f"unsupported index {key!r}")

    def tolist(self) -> list[int]:
        # `processor(input_ids[i, prev:])` collapses to a flat list of ints.
        assert len(self._rows) == 1
        return list(self._rows[0])


class _FakeXGr:
    """Drop-in for the `xgrammar` module that tracks every interaction."""

    def __init__(self) -> None:
        self.matchers_built: list[_FakeMatcher] = []
        self.applied_calls: list[tuple[object, object]] = []

    def GrammarMatcher(self, compiled_grammar: object) -> _FakeMatcher:
        m = _FakeMatcher()
        self.matchers_built.append(m)
        return m

    def allocate_token_bitmask(self, batch_size: int, vocab_size: int) -> _FakeBitmask:
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        return _FakeBitmask()

    def apply_token_bitmask_inplace(self, scores: object, bitmask: object) -> None:
        self.applied_calls.append((scores, bitmask))


def _modal_app():
    """Lazy import. `modal_app` raises at module-load when the modal SDK
    isn't installed, so we skip the test in environments (like CI's
    default test job) that don't have it. The class under test has no
    modal coupling — the skip is purely about the import barrier."""
    pytest.importorskip("modal", reason="modal_app requires the modal SDK to import")
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "scripts" / "modal_app.py"
    spec = importlib.util.spec_from_file_location("modal_app", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modal_app"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_logits_processor_skips_accept_on_first_call() -> None:
    """The first call carries only the prompt — there are no sampled
    tokens to accept yet, so accept_token must not fire."""
    modal_app = _modal_app()
    xgr = _FakeXGr()
    lp = modal_app._XGrammarLogitsProcessor(xgr, compiled_grammar=object(), vocab_size=128)

    input_ids = _FakeTensor([[10, 20, 30]])  # prompt only
    scores = _FakeTensor([[0] * 128])
    lp(input_ids, scores)

    assert xgr.matchers_built[0].accepted == []
    assert xgr.matchers_built[0].fill_calls == [(lp.bitmask, 0)]


def test_logits_processor_accepts_only_newly_sampled_tokens() -> None:
    modal_app = _modal_app()
    xgr = _FakeXGr()
    lp = modal_app._XGrammarLogitsProcessor(xgr, compiled_grammar=object(), vocab_size=128)

    # 1st call: prompt of length 3.
    lp(_FakeTensor([[10, 20, 30]]), _FakeTensor([[0] * 128]))
    # 2nd call: one new token sampled.
    lp(_FakeTensor([[10, 20, 30, 99]]), _FakeTensor([[0] * 128]))
    # 3rd call: another new token.
    lp(_FakeTensor([[10, 20, 30, 99, 7]]), _FakeTensor([[0] * 128]))

    assert xgr.matchers_built[0].accepted == [99, 7]


def test_logits_processor_moves_bitmask_to_scores_device() -> None:
    """Bitmask must follow the scores tensor onto its device, otherwise
    apply_token_bitmask_inplace mixes CPU and GPU memory."""
    modal_app = _modal_app()
    xgr = _FakeXGr()
    lp = modal_app._XGrammarLogitsProcessor(xgr, compiled_grammar=object(), vocab_size=128)

    scores = _FakeTensor([[0] * 128], device="cuda:0")
    lp(_FakeTensor([[1, 2]]), scores)

    assert lp.bitmask.moved_to == ["cuda:0"]
    assert xgr.applied_calls == [(scores, lp.bitmask)]


def test_logits_processor_one_matcher_per_batch_row() -> None:
    modal_app = _modal_app()
    xgr = _FakeXGr()
    modal_app._XGrammarLogitsProcessor(xgr, compiled_grammar=object(), vocab_size=64, batch_size=3)
    assert len(xgr.matchers_built) == 3


# -- make_modal_qwen_vl_quad_adapter ---------------------------------------


def _quadrant_json(position: str, hour: str, jock: str) -> str:
    """Minimal-but-valid Quadrant JSON string for adapter dispatch tests."""
    import json as _json

    return _json.dumps(
        {
            "position": position,
            "hour_raw": hour,
            "jock_raw": jock,
            "entries": [],
            "oddities": [],
        }
    )


def _save_fixture_page(path: Path, size: tuple[int, int] = (200, 300)) -> None:
    """Write a small PIL PNG to `path` so the adapter has something to crop."""
    from PIL import Image

    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _patch_modal_for_quadrant(
    monkeypatch: pytest.MonkeyPatch,
    side_effect: list[str | Exception],
) -> object:
    """Wire up the fakes the per-quadrant adapter needs:
      - `transcribe_qwen_vl.remote` returns the next entry from `side_effect`
      - `app.run()` is a no-op context manager.

    Returns the MagicMock so callers can inspect `call_args_list`.
    """
    pytest.importorskip("modal", reason="adapter test requires the modal SDK")
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    import scripts.modal_app as modal_app

    fake_remote = MagicMock(side_effect=side_effect)

    @contextmanager
    def fake_run() -> object:
        yield None

    monkeypatch.setattr(modal_app.transcribe_qwen_vl, "remote", fake_remote)
    monkeypatch.setattr(modal_app.app, "run", fake_run)
    return fake_remote


def test_modal_qwen_vl_quad_adapter_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 RPC calls in canonical order, each with the right schema and prompt;
    the assembled PageResult round-trips the header date and four quadrants."""
    image = tmp_path / "1990-04apr0106-page05.png"
    _save_fixture_page(image)
    fake_remote = _patch_modal_for_quadrant(
        monkeypatch,
        side_effect=[
            '{"page_date_raw": "Mon 1 Jan 90", "oddities": ["weather: snowy"]}',
            _quadrant_json("top_left", "6AM", "DJ A"),
            _quadrant_json("top_right", "7AM", "DJ B"),
            _quadrant_json("bottom_left", "8AM", "DJ C"),
            _quadrant_json("bottom_right", "9AM", "DJ D"),
        ],
    )

    transcribe = cm.make_modal_qwen_vl_quad_adapter("test-model")
    result = transcribe(image)

    # 5 calls: 1 header + 4 quadrants in canonical order.
    assert fake_remote.call_count == 5
    calls = fake_remote.call_args_list

    # Call 0: header.
    header_args = calls[0]
    from core.prompts import HEADER_EXTRACTION_PROMPT

    assert header_args.args[1] == HEADER_EXTRACTION_PROMPT
    assert header_args.kwargs["json_schema"] == cm.HEADER_WIRE_SCHEMA

    # Calls 1-4: quadrants in canonical order, each with pinned position.
    for i, position in enumerate(QUADRANT_ORDER, start=1):
        prompt_arg = calls[i].args[1]
        assert position in prompt_arg, f"call {i} prompt missing position {position!r}"
        schema = calls[i].kwargs["json_schema"]
        assert schema["properties"]["position"] == {"enum": [position]}

    # Assembled PageResult.
    assert result.page_date_raw == "Mon 1 Jan 90"
    assert result.oddities == ["weather: snowy"]
    assert [q.position for q in result.quadrants] == list(QUADRANT_ORDER)
    assert result.quadrants[0].hour_raw == "6AM"
    assert result.quadrants[3].jock_raw == "DJ D"
    assert result.model_version == "modal-qwen-vl-quad:test-model"


def test_modal_qwen_vl_quad_adapter_quadrant_fallback_on_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ONE quadrant call returns garbage, the page must still validate;
    the affected quadrant gets the parse_failed fallback. The other 3 are intact."""
    image = tmp_path / "1990-04apr0106-page15.png"
    _save_fixture_page(image)
    _patch_modal_for_quadrant(
        monkeypatch,
        side_effect=[
            '{"page_date_raw": null, "oddities": []}',
            _quadrant_json("top_left", "6AM", "A"),
            "not json {{",  # second quadrant returns garbage
            _quadrant_json("bottom_left", "8AM", "C"),
            _quadrant_json("bottom_right", "9AM", "D"),
        ],
    )

    transcribe = cm.make_modal_qwen_vl_quad_adapter("test-model")
    result = transcribe(image)

    # All 4 quadrants present, in canonical order.
    assert [q.position for q in result.quadrants] == list(QUADRANT_ORDER)

    # The failed quadrant carries the parse_failed sentinel.
    failed = result.quadrants[1]  # top_right
    assert failed.position == "top_right"
    assert len(failed.entries) == 1
    assert failed.entries[0].notes == "parse_failed"
    assert failed.entries[0].confidence == "low"
    assert failed.entries[0].raw_text == "not json {{"

    # The other three are NOT fallbacks — they have empty entries lists,
    # not a single parse_failed entry.
    for i in (0, 2, 3):
        for entry in result.quadrants[i].entries:
            assert entry.notes != "parse_failed"


def test_modal_qwen_vl_quad_adapter_header_failure_does_not_fail_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed header response leaves page_date_raw / oddities at their
    defaults; the page still validates with all four quadrants intact."""
    image = tmp_path / "1990-04apr0106-page20.png"
    _save_fixture_page(image)
    _patch_modal_for_quadrant(
        monkeypatch,
        side_effect=[
            "garbage response from header call",
            _quadrant_json("top_left", "6AM", "A"),
            _quadrant_json("top_right", "7AM", "B"),
            _quadrant_json("bottom_left", "8AM", "C"),
            _quadrant_json("bottom_right", "9AM", "D"),
        ],
    )

    transcribe = cm.make_modal_qwen_vl_quad_adapter("test-model")
    result = transcribe(image)

    assert result.page_date_raw is None
    assert result.oddities == []
    assert [q.position for q in result.quadrants] == list(QUADRANT_ORDER)
    # Quadrant data still flows through.
    assert result.quadrants[0].hour_raw == "6AM"
