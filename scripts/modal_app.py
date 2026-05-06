"""Modal app for running VL transcription on a remote GPU.

Modal is the lowest-friction way to put a multi-GB VLM behind a callable:
no VM management, sub-minute cold starts, billed by the second. We use
it for calibration runs where the local Mac would either OOM (full-res
Qwen-VL-7B at MPS) or take half an hour per page (CPU/fp32).

This module defines two remote functions — `transcribe_churro` and
`transcribe_qwen_vl` — that load weights once per warm container and
return raw OCR text. The local-side wrappers in `calibrate_models.py`
read images, call these, and feed results into the same scoring harness
as the local adapters.

Setup (one-time):

    pip install modal
    modal token new

Usage from the calibration script:

    python scripts/calibrate_models.py --models modal-churro --golden-dir tests/golden

Cost reference (mid-2026):
  - A100-40GB on-demand: ~$1.10/hr; ~30-60s per page on Churro-3B / Qwen-VL-7B
  - One golden page therefore ~$0.01-0.02
  - Full 18K corpus run ~$100-200, comparable to Gemini 3 Pro pricing

The remote functions are decorated `@app.function`; the local side opens
an ephemeral `app.run()` per script invocation. For a corpus-scale run
you'd `modal deploy` the app once and call into the deployed endpoint
instead of spinning up ephemeral containers per script run.
"""

from __future__ import annotations

import io

# Modal is an optional dependency: only callers that pass --models modal-...
# need it installed. Lazy-import inside the function bodies keeps the rest
# of the script (gemini-stored, churro local, qwen-vl local) usable
# without it.
try:
    import modal
except ImportError as exc:  # pragma: no cover — exercised when modal is absent
    raise RuntimeError(
        "modal is required for the modal-* adapters. Install with `pip install modal` "
        "and run `modal token new` once to authenticate."
    ) from exc


# HF_HOME points the Hugging Face cache (download dir + lock dir) at the
# mounted Modal Volume below. Without this env, transformers writes to
# ~/.cache/huggingface on the ephemeral container disk and re-downloads
# the full checkpoint on every cold start.
_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers>=5.0",
        "torch>=2.4",
        "torchvision",
        "accelerate",
        "pillow",
        "sentencepiece",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

app = modal.App("flowsheet-digitization-vl")


# Caching weights on a Modal Volume avoids re-downloading the multi-GB
# checkpoint every cold start. First call populates the cache; subsequent
# calls mount it and reuse the on-disk shards.
_WEIGHTS_VOLUME = modal.Volume.from_name("hf-weights", create_if_missing=True)
_VOLUME_MOUNTS = {"/cache/huggingface": _WEIGHTS_VOLUME}


def _load_vl_model(model_id: str):  # type: ignore[no-untyped-def]
    """Shared loader: returns (processor, model) on the container's GPU."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return processor, model


@app.function(
    image=_image,
    gpu="A100-40GB",
    volumes=_VOLUME_MOUNTS,
    timeout=300,
)
def transcribe_churro(image_bytes: bytes, model_id: str = "stanford-oval/churro-3B") -> str:
    """OCR a flowsheet page on a remote A100 and return the raw transcript.

    Churro is a Qwen2.5-VL fine-tune so it shares the chat-template
    requirements of the local adapter. We send the image through the
    same prompt and decode the model's output verbatim.
    """
    from PIL import Image

    processor, model = _load_vl_model(model_id)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
    text: str = processor.batch_decode(output, skip_special_tokens=True)[0]
    return text


@app.function(
    image=_image,
    gpu="A100-40GB",
    volumes=_VOLUME_MOUNTS,
    timeout=600,
)
def transcribe_qwen_vl(
    image_bytes: bytes,
    schema_prompt: str,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> str:
    """Run Qwen-VL on a remote A100 with the project's structured-output prompt.

    The full schema prompt is passed in rather than imported here so this
    module has no project-internal dependencies — the function is a pure
    'image bytes + prompt -> string' RPC.
    """
    from PIL import Image

    processor, model = _load_vl_model(model_id)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": schema_prompt},
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
    text: str = processor.batch_decode(output, skip_special_tokens=True)[0]
    return text
