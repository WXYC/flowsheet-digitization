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


def test_select_models_rejects_empty_list() -> None:
    with pytest.raises(SystemExit, match="empty"):
        cm._select_models([""])
