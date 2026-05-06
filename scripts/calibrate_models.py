#!/usr/bin/env python3
"""calibrate_models.py — score one or more extraction models on the goldens.

Runs each requested model on every `<stem>.png` in `tests/golden/` (or
another directory you point it at) and prints a side-by-side comparison
of accuracy, header-miss count, and elapsed time per page.

The pure scoring logic lives in `core/calibration.py` and is unit-tested.
This file is the CLI + adapter wiring.

Models:

  gemini-stored    Read previously-saved Gemini results from `data/results/`.
                   No API call. Use this as the baseline to compare against.
                   Maps `<stem>.png` to `data/results/**/<stem>.json` when
                   the stem encodes `<pdf-stem>-page<NN>` (the convention
                   the golden README documents).

  churro           Run stanford-oval/churro-3B locally via transformers.
                   Requires: `pip install transformers torch pillow`.
                   ~3B params, fp16 ≈ 6-8GB RAM/VRAM.

  qwen-vl          Run Qwen2.5-VL-7B-Instruct locally via transformers.
                   Requires: `pip install transformers torch pillow`.
                   For 4-bit quantized variants, use `--qwen-model` to
                   point at an Unsloth checkpoint.

Both local-model adapters wrap the model output with the project's
`PageResult` schema. Churro's raw OCR string is run through a small
post-processor that splits lines into the four-quadrant layout; Qwen-VL
is prompted with the same `PAGE_EXTRACTION_PROMPT` used for Gemini and
asked to return JSON matching the schema.

Examples:

    .venv/bin/python scripts/calibrate_models.py --models gemini-stored
    .venv/bin/python scripts/calibrate_models.py --models gemini-stored,churro
    .venv/bin/python scripts/calibrate_models.py \\
        --models gemini-stored,qwen-vl \\
        --golden-dir tests/golden \\
        --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

# Allow `python scripts/calibrate_models.py` to import the project's
# `core` package without `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.calibration import (  # noqa: E402
    CalibrationCase,
    CalibrationOutcome,
    TranscribeFn,
    discover_cases,
    run_calibration,
    summarize,
)
from core.prompts import PAGE_EXTRACTION_PROMPT  # noqa: E402
from core.schema import QUADRANT_ORDER, PageResult, Quadrant  # noqa: E402

# -- gemini-stored adapter -------------------------------------------------


def _stored_result_path(image_path: Path, results_root: Path) -> Path | None:
    """Map `<stem>-page<NN>.png` to `data/results/**/page-NN.json`.

    The golden directory's images are renderings of pages already
    processed by the live pipeline, so the matching JSON is already on
    disk under `DATA_ROOT/results/<rel-pdf>/page-NN.json`. Match by the
    page number suffix in the stem, then disambiguate by the PDF stem.
    """
    stem = image_path.stem  # e.g. "1990-01jan0106-page05"
    if "-page" not in stem:
        return None
    pdf_stem, _, page_part = stem.rpartition("-page")
    try:
        page_number = int(page_part.lstrip("0") or "0")
    except ValueError:
        return None

    matches: list[Path] = []
    for candidate in results_root.rglob(f"page-*{page_number:02d}.json"):
        if candidate.parent.name == pdf_stem:
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    # Fall back: any page-NN.json with that page number, any pdf folder.
    for candidate in results_root.rglob("*.json"):
        if candidate.parent.name != pdf_stem:
            continue
        try:
            n = int(candidate.stem.split("-")[-1])
        except ValueError:
            continue
        if n == page_number:
            return candidate
    return None


def make_gemini_stored_adapter(results_root: Path) -> TranscribeFn:
    """Adapter that reads previously-saved Gemini results from disk."""

    def transcribe(image_path: Path) -> PageResult:
        path = _stored_result_path(image_path, results_root)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"no stored result for {image_path.name} under {results_root}")
        return PageResult.model_validate_json(path.read_text())

    return transcribe


# -- churro-3B adapter -----------------------------------------------------


def make_churro_adapter(model_id: str = "stanford-oval/churro-3B") -> TranscribeFn:
    """Adapter that calls Churro-3B for OCR, then wraps the text in PageResult.

    Churro is an OCR model: it returns a transcription, not structured
    JSON. We feed the full raw transcript into a single quadrant so the
    golden harness can substring-match against it. Producing a real
    quadrant-aware split is a follow-up; the goal of this calibration is
    "can the model READ the handwriting at all", not "does it match our
    schema today".
    """
    # Lazy import: torch + transformers is multi-GB and most callers
    # never touch this adapter.
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )

    def transcribe(image_path: Path) -> PageResult:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text="Transcribe this page.", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=2048)
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        return _wrap_raw_text_as_page_result(text, model_version=f"churro:{model_id}")

    return transcribe


# -- qwen-vl adapter -------------------------------------------------------


def make_qwen_vl_adapter(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> TranscribeFn:
    """Adapter that calls Qwen-VL with the production extraction prompt.

    Qwen-VL is a multimodal LLM and follows JSON-schema instructions.
    We send the same `PAGE_EXTRACTION_PROMPT` as Gemini and validate the
    JSON output against `PageResult`. Malformed JSON or schema mismatch
    raises and is recorded by the harness as an errored outcome.
    """
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )

    schema_hint = (
        PAGE_EXTRACTION_PROMPT + "\n\nReturn ONLY a JSON object matching the PageResult schema. "
        "No prose before or after the JSON."
    )

    def transcribe(image_path: Path) -> PageResult:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": schema_hint},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        output = model.generate(inputs, max_new_tokens=4096)
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        payload = _extract_json_block(text)
        data = json.loads(payload)
        data.setdefault("model_version", f"qwen-vl:{model_id}")
        data.setdefault("extracted_at", datetime.now(UTC).isoformat())
        return PageResult.model_validate(data)

    return transcribe


# -- helpers shared by adapters --------------------------------------------


def _wrap_raw_text_as_page_result(text: str, *, model_version: str) -> PageResult:
    """Build a PageResult that puts a whole-page transcript in one quadrant.

    The golden harness uses substring matching on `Entry.raw_text`, so a
    single Entry whose `raw_text` contains the entire transcript will
    pass any row-level expectation that exists somewhere on the page.
    Header expectations (date, hour, jock) will fail unless the OCR
    output happens to include them — which is exactly the signal we
    want: "OCR can read words" vs "model can find the schema".
    """
    quadrants = [
        Quadrant(position=pos, entries=[])
        for pos in QUADRANT_ORDER  # type: ignore[arg-type]
    ]
    quadrants[0] = Quadrant(
        position="top_left",
        hour_raw=None,
        jock_raw=None,
        entries=[
            {  # type: ignore[list-item]
                "row_index": 0,
                "raw_text": text,
                "confidence": "medium",
            }
        ],
    )
    return PageResult(
        page_date_raw=text[:200] if text else None,
        quadrants=quadrants,
        model_version=model_version,
        extracted_at=datetime.now(UTC),
    )


def _extract_json_block(text: str) -> str:
    """Pull the first {...} JSON object out of a chat-completion string.

    Local VLMs often wrap their JSON in markdown fences or chat tokens.
    We don't want to be lenient — we want a clear "the model produced
    something the schema can validate" signal — but we do want to look
    past obvious wrapping.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("model output contained no '{' — not JSON")
    # Find the matching brace by depth-tracking.
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("model output had unbalanced braces — JSON truncated?")


# -- CLI -------------------------------------------------------------------


_ADAPTER_BUILDERS: dict[str, Callable[[argparse.Namespace], TranscribeFn]] = {
    "gemini-stored": lambda a: make_gemini_stored_adapter(a.results_root),
    "churro": lambda a: make_churro_adapter(a.churro_model),
    "qwen-vl": lambda a: make_qwen_vl_adapter(a.qwen_model),
}


def _print_progress(case: CalibrationCase, outcome: CalibrationOutcome) -> None:
    if outcome.error is not None:
        marker = f"ERROR ({outcome.error[:40]})"
    elif outcome.report and outcome.report.passed:
        marker = "PASS"
    else:
        marker = "FAIL"
    print(
        f"  {case.stem:<40}  {outcome.elapsed_seconds:>6.1f}s  {marker}",
        file=sys.stderr,
    )


def _select_models(requested: Iterable[str]) -> list[str]:
    chosen: list[str] = []
    for m in requested:
        m = m.strip()
        if not m:
            continue
        if m not in _ADAPTER_BUILDERS:
            valid = ", ".join(_ADAPTER_BUILDERS)
            raise SystemExit(f"unknown model {m!r}; expected one of: {valid}")
        chosen.append(m)
    if not chosen:
        raise SystemExit("--models was empty")
    return chosen


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        default="gemini-stored",
        help=(
            f"Comma-separated model names: {', '.join(_ADAPTER_BUILDERS)}. Default: gemini-stored."
        ),
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path("tests/golden"),
        help="Directory of <stem>.png + <stem>.truth.json pairs (default: tests/golden).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "./data")) / "results",
        help="Where stored Gemini results live (default: $DATA_ROOT/results or ./data/results).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N pages (default: all).",
    )
    parser.add_argument(
        "--churro-model",
        default="stanford-oval/churro-3B",
        help="HuggingFace model id for the churro adapter.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model id for the qwen-vl adapter.",
    )
    args = parser.parse_args(argv)

    models = _select_models(args.models.split(","))

    cases = discover_cases(args.golden_dir.expanduser().resolve())
    if not cases:
        print(f"No <stem>.png + <stem>.truth.json pairs in {args.golden_dir}", file=sys.stderr)
        return 1
    if args.limit is not None:
        cases = cases[: args.limit]

    print(f"Scoring {len(cases)} page(s) on {len(models)} model(s)\n", file=sys.stderr)

    outcomes_by_model: dict[str, list[CalibrationOutcome]] = {}
    for model in models:
        print(f"--- {model} ---", file=sys.stderr)
        try:
            transcribe = _ADAPTER_BUILDERS[model](args)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed to build adapter: {exc}", file=sys.stderr)
            outcomes_by_model[model] = [
                CalibrationOutcome(case=c, report=None, elapsed_seconds=0.0, error=str(exc))
                for c in cases
            ]
            continue
        outcomes_by_model[model] = run_calibration(cases, transcribe, on_complete=_print_progress)

    print()
    print(summarize(outcomes_by_model))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
