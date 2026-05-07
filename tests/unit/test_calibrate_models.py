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
    assert cm._select_models(["modal-churro", "modal-qwen-vl"]) == [
        "modal-churro",
        "modal-qwen-vl",
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
    """Lazy import — the modal SDK has to be installed for this test file
    to load `modal_app`. We import inside the test fn so the wider test
    suite (which doesn't need modal) can still collect."""
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
