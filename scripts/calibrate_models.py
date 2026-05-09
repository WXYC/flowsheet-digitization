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

  modal-churro     Same as `churro` but the inference runs on a remote
                   A100 via Modal. ~30-60s per page, no local RAM use.
                   Requires: `pip install modal && modal token new`.

  modal-qwen-vl    Same as `qwen-vl`, on Modal A100. Pricing reference:
                   one golden page ~$0.01-0.02; full 18K-page corpus
                   run ~$100-200.

  modal-qwen-vl-quad
                   Per-quadrant Qwen-VL on Modal: crops the page into
                   4 sub-images plus a header strip, calls the model 5x
                   per page, assembles a PageResult locally. Eliminates
                   cross-quadrant content placement errors that the
                   single-shot `modal-qwen-vl` adapter still suffers.
                   ~5x cost (~$0.05-0.10/page); full corpus ~$1000-1500.

Both local-model adapters wrap the model output with the project's
`PageResult` schema. Churro's raw OCR string is wrapped into the same
text in all four quadrants so substring scoring works regardless of
where the truth file located the row; Qwen-VL is prompted with the same
`PAGE_EXTRACTION_PROMPT` used for Gemini and asked to return JSON
matching the schema.

Examples:

    .venv/bin/python scripts/calibrate_models.py --models gemini-stored
    .venv/bin/python scripts/calibrate_models.py --models gemini-stored,churro
    .venv/bin/python scripts/calibrate_models.py --models modal-churro,modal-qwen-vl
    .venv/bin/python scripts/calibrate_models.py \\
        --models gemini-stored,qwen-vl \\
        --device cpu --dtype fp32 \\
        --golden-dir tests/golden --limit 3
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

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
from core.page_layout import PageLayout, detect_page_layout  # noqa: E402
from core.prompts import (  # noqa: E402
    HEADER_EXTRACTION_PROMPT,
    PAGE_EXTRACTION_PROMPT,
    QUADRANT_EXTRACTION_PROMPT_TEMPLATE,
)
from core.schema import QUADRANT_ORDER, Entry, PageResult, Quadrant, QuadrantPosition  # noqa: E402

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


def make_churro_adapter(
    model_id: str = "stanford-oval/churro-3B",
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> TranscribeFn:
    """Adapter that calls Churro-3B for OCR, then wraps the text in PageResult.

    Churro is an OCR model: it returns a transcription, not structured
    JSON. We feed the full raw transcript into a single quadrant so the
    golden harness can substring-match against it. Producing a real
    quadrant-aware split is a follow-up; the goal of this calibration is
    "can the model READ the handwriting at all", not "does it match our
    schema today".

    `device` is passed to from_pretrained as `device_map`; "cpu" forces
    everything onto host RAM (slow but bounded), "mps" uses Apple
    Silicon's GPU pool (fast but can blow past system RAM on large
    images), "auto" lets transformers choose. `dtype` is one of "auto",
    "fp16", "bf16", "fp32".
    """
    # Lazy import: torch + transformers is multi-GB and most callers
    # never touch this adapter.
    import torch
    from PIL import Image
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=_torch_dtype(torch, dtype),
    )

    def transcribe(image_path: Path) -> PageResult:
        image = Image.open(image_path).convert("RGB")
        # Churro is a Qwen2.5-VL fine-tune and requires the chat-template
        # path so the processor inserts the image-token sentinels the model
        # was trained on. Calling processor(images=..., text=...) directly
        # produces a token / feature mismatch at generate-time.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Transcribe this handwritten page verbatim."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=2048)
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        return _wrap_raw_text_as_page_result(text, model_version=f"churro:{model_id}")

    return transcribe


# -- qwen-vl adapter -------------------------------------------------------


def make_qwen_vl_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> TranscribeFn:
    """Adapter that calls Qwen-VL with the production extraction prompt.

    Qwen-VL is a multimodal LLM and follows JSON-schema instructions.
    We send the same `PAGE_EXTRACTION_PROMPT` as Gemini and validate the
    JSON output against `PageResult`. Malformed JSON or schema mismatch
    raises and is recorded by the harness as an errored outcome.
    """
    import torch
    from PIL import Image
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=_torch_dtype(torch, dtype),
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
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=4096)
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        payload = _extract_json_block(text)
        data = json.loads(payload)
        data.setdefault("model_version", f"qwen-vl:{model_id}")
        data.setdefault("extracted_at", datetime.now(UTC).isoformat())
        return PageResult.model_validate(data)

    return transcribe


# -- modal-* adapters ------------------------------------------------------


def make_modal_churro_adapter(
    model_id: str = "stanford-oval/churro-3B",
) -> TranscribeFn:
    """Adapter that runs Churro on a remote A100 via Modal.

    See `scripts/modal_app.py` for the deployed function definitions.
    Each call opens an ephemeral `app.run()` context so `modal token new`
    needs to have happened once on this machine.
    """
    # Lazy import: modal is only required if a modal-* model is selected.
    from scripts.modal_app import app, transcribe_churro

    def transcribe(image_path: Path) -> PageResult:
        image_bytes = image_path.read_bytes()
        with app.run():
            text: str = transcribe_churro.remote(image_bytes, model_id)
        return _wrap_raw_text_as_page_result(text, model_version=f"modal-churro:{model_id}")

    return transcribe


HEADER_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_date_raw": {"type": ["string", "null"]},
        "oddities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["page_date_raw", "oddities"],
    "additionalProperties": False,
}
"""JSON Schema for the header-strip call in `modal-qwen-vl-quad`.

Defined inline rather than as a Pydantic model: there's exactly one
caller, the schema is small, and a one-off model would just be a layer
of indirection.
"""


def make_modal_qwen_vl_quad_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> TranscribeFn:
    """Per-quadrant Qwen-VL on a remote A100 via Modal.

    Eliminates layout misplacement by construction: instead of asking
    the model to spatially attribute rows from one full-page image to
    the right quadrant slot in the JSON wrapper, we crop the page
    locally and call the model 5 times — once per quadrant + once on
    the header strip — and assemble the page server-side.

    Each call is grammar-constrained (xgrammar). The four quadrant
    schemas pin `position` to a singleton enum so the model literally
    cannot mislabel the cell. The header schema is small and ad-hoc,
    capturing only `page_date_raw` and page-level oddities (DJ-handoff
    notes, weather notes, marginal annotations above the grid).

    All 5 calls run inside one `with app.run():` block. Modal reuses
    the warm container across them — the first call pays whatever
    cold-start applies; calls 2-5 are warm. Per-page wall time is
    roughly `cold + 4 * warm`, not `5 * warm`.

    On per-quadrant JSON failure, the affected quadrant is replaced by
    a `_quadrant_fallback` carrying the raw text in one entry tagged
    `notes="parse_failed"`. Other quadrants still validate; the page
    is never lost wholesale.
    """
    from PIL import Image

    from scripts.modal_app import app, transcribe_qwen_vl

    quadrant_schemas: dict[QuadrantPosition, dict[str, Any]] = {
        pos: _quadrant_wire_schema(pos) for pos in QUADRANT_ORDER
    }
    quadrant_prompts: dict[QuadrantPosition, str] = {
        pos: QUADRANT_EXTRACTION_PROMPT_TEMPLATE.format(position=pos) for pos in QUADRANT_ORDER
    }

    def transcribe(image_path: Path) -> PageResult:
        image = Image.open(image_path).convert("RGB")
        layout = detect_page_layout(image)
        header_image = _crop_header_strip(image, layout)
        crops = _crop_quadrants(image, layout)

        page_date_raw: str | None = None
        page_oddities: list[str] = []
        quadrants: list[Quadrant] = []

        with app.run():
            # Header call — surfaces page-level fields the quadrant crops
            # don't see. Failure here drops the date but does NOT fail
            # the page; quadrant content is the load-bearing payload.
            try:
                header_text: str = transcribe_qwen_vl.remote(
                    _png_bytes(header_image),
                    HEADER_EXTRACTION_PROMPT,
                    model_id,
                    json_schema=HEADER_WIRE_SCHEMA,
                )
                header_data = json.loads(header_text)
                page_date_raw = header_data.get("page_date_raw")
                page_oddities = header_data.get("oddities") or []
            except Exception:
                pass  # leave defaults; not worth failing the page

            # Quadrant calls in canonical order. Each crop sees only its
            # own cell (plus a small bleed band) so cross-quadrant
            # placement is structurally impossible.
            for position in QUADRANT_ORDER:
                text: str = transcribe_qwen_vl.remote(
                    _png_bytes(crops[position]),
                    quadrant_prompts[position],
                    model_id,
                    json_schema=quadrant_schemas[position],
                )
                try:
                    data = json.loads(text)
                    # Defense in depth: grammar already pins position via
                    # singleton enum, but we overwrite anyway so a future
                    # xgrammar regression on enum handling can't sneak a
                    # mislabel through.
                    data["position"] = position
                    quadrants.append(Quadrant.model_validate(data))
                except Exception:
                    quadrants.append(_quadrant_fallback(text, position))

        return PageResult(
            page_date_raw=page_date_raw,
            quadrants=quadrants,
            model_version=f"modal-qwen-vl-quad:{model_id}",
            extracted_at=datetime.now(UTC),
            oddities=page_oddities,
        )

    return transcribe


def make_modal_qwen_vl_adapter(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> TranscribeFn:
    """Adapter that runs Qwen-VL on a remote A100 via Modal.

    Decoding is grammar-constrained: we derive a JSON Schema from
    `PageResult` (minus the server-set `model_version`/`extracted_at`
    fields) and pass it through to the Modal function, which feeds it
    into xgrammar's logits processor. The model is then mechanically
    forbidden from emitting anything outside the schema, which fixes
    the "wrong wrapper shape" failure we saw with prompt-only steering.

    The trimmed suffix is grammar-aware: with xgrammar enforcing the
    output, a long "return ONLY JSON, no prose before or after" prompt
    is redundant — the model literally cannot emit prose.
    """
    from scripts.modal_app import app, transcribe_qwen_vl

    schema_prompt = PAGE_EXTRACTION_PROMPT + "\n\nReturn the page transcript as JSON."
    wire_schema = _qwen_vl_wire_schema()

    def transcribe(image_path: Path) -> PageResult:
        image_bytes = image_path.read_bytes()
        with app.run():
            text: str = transcribe_qwen_vl.remote(
                image_bytes, schema_prompt, model_id, json_schema=wire_schema
            )
        try:
            data = json.loads(text)
            data["model_version"] = f"modal-qwen-vl:{model_id}"
            data["extracted_at"] = datetime.now(UTC).isoformat()
            return PageResult.model_validate(data)
        except Exception:
            # Defense-in-depth: if grammar lets through invalid JSON
            # (xgrammar version regression, schema corner case it doesn't
            # cover, etc.) we still want a scored row. Persist the raw
            # text on the failure path so we can diagnose the deviation;
            # on the happy path the structured dump captures everything.
            raw_dump_dir = Path("/tmp/modal-dump/modal-qwen-vl-raw")
            raw_dump_dir.mkdir(parents=True, exist_ok=True)
            (raw_dump_dir / f"{image_path.stem}.txt").write_text(text)
            return _wrap_raw_text_as_page_result(text, model_version=f"modal-qwen-vl:{model_id}")

    return transcribe


# -- helpers shared by adapters --------------------------------------------


def _torch_dtype(torch_module: Any, name: str):  # type: ignore[no-untyped-def]
    """Map a CLI dtype string to a torch dtype, or pass "auto" through.

    Forcing the dtype matters: torch_dtype="auto" picks fp32 when the
    config doesn't specify, which on a 7B VLM blows past 30GB before any
    activations. fp16 cuts that in half; bf16 has the same memory but
    better numerics on Apple Silicon.
    """
    if name == "auto":
        return "auto"
    table = {
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
        "fp32": torch_module.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; expected one of: auto, fp16, bf16, fp32")
    return table[name]


def _qwen_vl_wire_schema() -> dict[str, Any]:
    """Return a JSON Schema that drives grammar-constrained Qwen-VL decoding.

    `model_version` and `extracted_at` are populated by the adapter
    server-side, not by the model — including them in the grammar would
    force Qwen to invent a plausible-looking string for each. Strip them
    from `properties` and `required` so the constrained decoder only
    spends tokens on fields the model actually has signal for.
    """
    import copy

    schema = copy.deepcopy(PageResult.model_json_schema())
    for field in ("model_version", "extracted_at"):
        schema["properties"].pop(field, None)
        if field in schema.get("required", []):
            schema["required"].remove(field)
    return schema


# -- per-quadrant cropping (modal-qwen-vl-quad) ----------------------------


def _crop_header_strip(image: PILImage, layout: PageLayout) -> PILImage:
    """The header strip — date + page-level notes — above the body grid."""
    w, _ = image.size
    return image.crop((0, 0, w, layout.header_bottom_y))


def _crop_quadrants(image: PILImage, layout: PageLayout) -> dict[QuadrantPosition, PILImage]:
    """Split the page body into 4 quadrants on the detected grid lines.

    No bleed: the detected coordinates land on the printed grid divider
    itself, so a row to one side of the line belongs to one quadrant and
    the row on the other side belongs to its neighbor — no overlap is
    needed and bleeding the same row into two crops causes the model to
    transcribe it twice.
    """
    w, h = image.size
    return {
        "top_left": image.crop((0, layout.header_bottom_y, layout.column_mid_x, layout.body_mid_y)),
        "top_right": image.crop(
            (layout.column_mid_x, layout.header_bottom_y, w, layout.body_mid_y)
        ),
        "bottom_left": image.crop((0, layout.body_mid_y, layout.column_mid_x, h)),
        "bottom_right": image.crop((layout.column_mid_x, layout.body_mid_y, w, h)),
    }


def _png_bytes(image: PILImage) -> bytes:
    """PIL image -> PNG-encoded bytes for the Modal RPC payload."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _quadrant_fallback(text: str, position: QuadrantPosition) -> Quadrant:
    """One-row low-confidence Quadrant, used when JSON parse/validation fails.

    Tagged `notes="parse_failed"` so a downstream scorer can identify
    these and exclude them from accuracy stats — the substring matcher
    would otherwise reward a quadrant whose JSON failed but whose raw
    text happened to contain a truth row, conflating a structural
    failure with a real read.
    """
    return Quadrant(
        position=position,
        entries=[
            Entry(
                row_index=0,
                raw_text=text,
                confidence="low",
                notes="parse_failed",
            )
        ],
    )


def _quadrant_wire_schema(position: QuadrantPosition) -> dict[str, Any]:
    """Return a JSON Schema for ONE Quadrant with `position` pinned.

    Used by the per-quadrant adapter (`modal-qwen-vl-quad`): we crop the
    page into 4 sub-images and call the model once per quadrant. Pinning
    `position` to a singleton enum mechanically forbids the model from
    swapping labels — the grammar will reject any token that doesn't
    spell the exact position string we asked for. The adapter also
    overwrites `data["position"]` after json.loads as defense in depth.
    """
    import copy

    schema = copy.deepcopy(Quadrant.model_json_schema())
    schema["properties"]["position"] = {"enum": [position]}
    return schema


def _wrap_raw_text_as_page_result(text: str, *, model_version: str) -> PageResult:
    """Build a PageResult that puts the whole-page transcript in every quadrant.

    OCR-only models like Churro produce one long string of text without
    any quadrant attribution. The golden harness scores per-quadrant —
    'is this substring in some entry of THIS quadrant'. Putting the same
    transcript in all four quadrants means the row-substring check
    succeeds wherever the truth file located the row, while the
    header-level checks (hour, jock) still fail unless those fields
    actually appeared in the OCR output. That keeps the signal we want
    (does the model READ?) without penalizing it for not knowing the
    layout, which is a separate phase-2 problem.
    """
    quadrants = [
        Quadrant(
            position=pos,
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
        for pos in QUADRANT_ORDER  # type: ignore[arg-type]
    ]
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
    "churro": lambda a: make_churro_adapter(a.churro_model, device=a.device, dtype=a.dtype),
    "qwen-vl": lambda a: make_qwen_vl_adapter(a.qwen_model, device=a.device, dtype=a.dtype),
    "modal-churro": lambda a: make_modal_churro_adapter(a.churro_model),
    "modal-qwen-vl": lambda a: make_modal_qwen_vl_adapter(a.qwen_model),
    "modal-qwen-vl-quad": lambda a: make_modal_qwen_vl_quad_adapter(a.qwen_model),
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


def _wrap_with_dump(transcribe: TranscribeFn, dump_dir: Path) -> TranscribeFn:
    """Decorate a transcribe callable to also write each PageResult to disk.

    Useful for after-the-fact inspection: a CPU run takes ~7 min/page,
    so you want the raw transcript on disk even if rescoring against an
    updated truth file changes the verdict.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(image_path: Path) -> PageResult:
        result = transcribe(image_path)
        out = dump_dir / f"{image_path.stem}.json"
        out.write_text(result.model_dump_json(indent=2))
        return result

    return wrapped


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
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help=(
            "Where to run local-model inference. cpu is bounded by physical RAM and slow; "
            "mps is Apple Silicon's GPU pool — fast but can blow past system RAM on full-res "
            "pages because unified memory has no preallocation cap. Default: auto."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help=(
            "Weight dtype. 'auto' picks the model's config default (often fp32 for VLMs). "
            "fp16 / bf16 halve weight RAM and roughly halve activation RAM. Default: auto."
        ),
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help=(
            "If set, save each model's PageResult JSON to "
            "<dump-dir>/<model>/<stem>.json so the raw transcript can be "
            "inspected without re-running inference."
        ),
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
        if args.dump_dir is not None:
            transcribe = _wrap_with_dump(transcribe, args.dump_dir / model)
        outcomes_by_model[model] = run_calibration(cases, transcribe, on_complete=_print_progress)

    print()
    print(summarize(outcomes_by_model))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
