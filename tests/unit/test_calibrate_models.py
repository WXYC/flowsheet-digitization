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
