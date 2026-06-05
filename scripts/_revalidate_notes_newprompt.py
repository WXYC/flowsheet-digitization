"""Re-extract the same 6-page sample as scripts/_revalidate_notes_2026_06_04.py
but against whatever core/prompts.py is currently on disk. Dumps results to
data/notes-revalidation-newprompt/ so the baseline measurement
(data/notes-revalidation-2026-06-04/) is preserved for diff.

One-shot measurement for issue #61. Run from repo root.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from core.gemini import GeminiClient, MediaResolution

PAGES = [2, 6, 9, 14, 16, 19]
PAGES_DIR = Path("data/pages/1990/April 1990/1990-04apr0106")
OUT_DIR = Path("data/notes-revalidation-newprompt")


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async def run_one(p: int) -> tuple[int, dict | str]:
        img = PAGES_DIR / f"page-{p:02d}.png"
        try:
            result = await client.extract_page(img)
            return p, result.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            return p, f"ERROR: {type(exc).__name__}: {exc}"

    results = await asyncio.gather(*(run_one(p) for p in PAGES))
    for p, payload in results:
        out = OUT_DIR / f"1990-04apr0106-page{p:02d}.fresh.json"
        if isinstance(payload, str):
            print(f"page {p:02d}: {payload}")
            out.write_text(json.dumps({"error": payload}, indent=2))
        else:
            out.write_text(json.dumps(payload, indent=2))
            print(f"page {p:02d}: wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
