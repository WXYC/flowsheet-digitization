"""Pilot: re-extract pages 01-11 of 1990-04apr1318 against current Gemini
with the new notes prompt, then bake fresh verifier bundles into
data/pilot-bundles/. Does NOT touch .seed/verifier/, data/results/, or
jobs.db — the pilot output is isolated for spot-checking.

One-shot for the issue-#61 trial run. Run from repo root.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from core.gemini import GeminiClient, MediaResolution

PAGES = list(range(1, 12))
STEM = "1990-04apr1318"
PAGES_DIR = Path("data/pages/1990/April 1990") / STEM
RESULTS_DIR = Path("data/pilot-results") / STEM
BUNDLES_DIR = Path("data/pilot-bundles")


async def main() -> None:
    load_dotenv(override=False)
    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    media_resolution = MediaResolution.from_string(
        os.environ.get("GEMINI_MEDIA_RESOLUTION", "high")
    )

    from google import genai

    client = GeminiClient(
        sdk=genai.Client(api_key=api_key),
        model=model,
        media_resolution=media_resolution,
    )
    cached = await client.create_cache()
    print(f"model={model}  cache={'on' if cached else 'off'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    async def extract(p: int) -> tuple[int, Path | str]:
        img = PAGES_DIR / f"page-{p:02d}.png"
        out = RESULTS_DIR / f"page-{p:02d}.json"
        try:
            gemini_result = await client.extract_page(img)
            payload = gemini_result.model_dump(mode="json")
            payload["model_version"] = model
            payload["extracted_at"] = datetime.now(UTC).isoformat()
            out.write_text(json.dumps(payload, indent=2))
            return p, out
        except Exception as exc:  # noqa: BLE001
            return p, f"ERROR: {type(exc).__name__}: {exc}"

    results = await asyncio.gather(*(extract(p) for p in PAGES))

    extracted: list[tuple[int, Path]] = []
    for p, payload in results:
        if isinstance(payload, str):
            print(f"page {p:02d}: {payload}")
        else:
            print(f"page {p:02d}: extracted -> {payload}")
            extracted.append((p, payload))

    print("\n-- baking bundles --")
    for p, result_path in extracted:
        img = PAGES_DIR / f"page-{p:02d}.png"
        out = BUNDLES_DIR / f"{STEM}-page{p:02d}.bundle.json"
        cmd = [
            sys.executable,
            "-m",
            "scripts.make_verifier_bundle",
            str(result_path),
            str(img),
            "--out",
            str(out),
            "--pdf-path",
            f"1990/April 1990/{STEM}.pdf",
            "--page-number",
            str(p),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"page {p:02d}: baked -> {out}")
        except subprocess.CalledProcessError as exc:
            print(f"page {p:02d}: BAKE FAILED rc={exc.returncode}: {exc.stderr}")


if __name__ == "__main__":
    asyncio.run(main())
