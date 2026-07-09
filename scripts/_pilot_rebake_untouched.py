"""Re-extract + re-bake every deployed page Alex has not yet reviewed.

Untouched = a deployed bundle in `.seed/verifier/` for which no
`<stem>-page<NN>.verified.json` exists in `data/verifier-pulled-refresh/`.

Outputs land in `data/pilot-bundles-final/` (not into .seed/ directly) so
the bundles can be inspected before being copied into the deployment
seed dir. Already-baked 1990-04apr1318 bundles in `data/pilot-bundles-v2/`
are linked into the final dir without re-extraction.

One-shot script for issue-#61 trial-extension. Run from repo root.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from core.gemini import GeminiClient, MediaResolution

SEED_DIR = Path(".seed/verifier")
VERIFIED_DIR = Path("data/verifier-pulled-refresh")
PAGES_ROOT = Path("data/pages")

RESULTS_DIR = Path("data/pilot-results-final")
BUNDLES_DIR = Path("data/pilot-bundles-final")

# Already-baked bundles for 1990-04apr1318 (from the earlier pilot).
PREEXISTING_DIR = Path("data/pilot-bundles-v2")
PREEXISTING_STEM = "1990-04apr1318"

# Cap on concurrent Gemini calls in flight. Higher = faster wall-clock,
# more rate-limit risk.
_MAX_CONCURRENCY = 6

_BUNDLE_NAME = re.compile(r"^(?P<stem>.+)-page(?P<page>\d{2})\.bundle\.json$")
_VERIFIED_NAME = re.compile(r"^(?P<stem>.+)-page(?P<page>\d{2})\.verified\.json$")


def _index_dir(directory: Path, pattern: re.Pattern[str]) -> set[tuple[str, int]]:
    """Return `{(stem, page_number)}` for files in `directory` matching pattern."""
    out: set[tuple[str, int]] = set()
    if not directory.is_dir():
        return out
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            out.add((m.group("stem"), int(m.group("page"))))
    return out


def _pages_dir_for_stem(stem: str) -> Path:
    """Locate the page-image directory for a given bundle stem.

    Stems look like `1990-04apr0106` and live at
    `data/pages/<year>/<Month> <year>/<stem>/`. Rather than parse the
    stem, glob for the matching leaf directory — more robust to any
    future rename of the month-directory format.
    """
    matches = list(PAGES_ROOT.glob(f"*/*/{stem}"))
    if not matches:
        raise FileNotFoundError(f"no pages dir for stem {stem!r} under {PAGES_ROOT}")
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous pages dir for stem {stem!r}: {matches}")
    return matches[0]


def _pdf_path_for_stem(stem: str) -> str:
    """Canonical relative pdf_path for a given stem.

    Matches `_parse_job_key_from_result_path` output: the PDF sits one
    directory above the page-image dir with a `.pdf` extension. Example:
    `1990/April 1990/1990-04apr0106.pdf`.
    """
    pages_dir = _pages_dir_for_stem(stem)
    rel = pages_dir.relative_to(PAGES_ROOT)
    return f"{rel}.pdf"


def _untouched_pages() -> list[tuple[str, int]]:
    """All `(stem, page)` pairs in .seed/verifier/ not in verifier-pulled-refresh/."""
    deployed = _index_dir(SEED_DIR, _BUNDLE_NAME)
    verified = _index_dir(VERIFIED_DIR, _VERIFIED_NAME)
    return sorted(deployed - verified)


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

    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    # Link the already-baked 1318 bundles into the final dir verbatim.
    for src in PREEXISTING_DIR.glob(f"{PREEXISTING_STEM}-page*.bundle.json"):
        dst = BUNDLES_DIR / src.name
        shutil.copy2(src, dst)
        print(f"reused {src.name}")

    untouched = _untouched_pages()
    # Skip pages already covered by the linked-in preexisting bundles.
    targets = [(stem, page) for stem, page in untouched if stem != PREEXISTING_STEM]

    sem = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def extract(stem: str, page: int) -> tuple[str, int, Path | str]:
        async with sem:
            pages_dir = _pages_dir_for_stem(stem)
            img = pages_dir / f"page-{page:02d}.png"
            out_dir = RESULTS_DIR / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"page-{page:02d}.json"
            try:
                gemini_result = await client.extract_page(img)
                payload = gemini_result.model_dump(mode="json")
                payload["model_version"] = model
                payload["extracted_at"] = datetime.now(UTC).isoformat()
                out.write_text(json.dumps(payload, indent=2))
                return stem, page, out
            except Exception as exc:  # noqa: BLE001
                return stem, page, f"ERROR: {type(exc).__name__}: {exc}"

    tasks = [extract(stem, page) for stem, page in targets]
    print(f"extracting {len(tasks)} pages (max concurrency {_MAX_CONCURRENCY})")
    results = await asyncio.gather(*tasks)

    extracted: list[tuple[str, int, Path, Path]] = []
    for stem, page, payload in results:
        if isinstance(payload, str):
            print(f"{stem} page {page:02d}: {payload}")
            continue
        pages_dir = _pages_dir_for_stem(stem)
        extracted.append((stem, page, payload, pages_dir / f"page-{page:02d}.png"))

    print(f"\n-- baking {len(extracted)} bundles --")
    for stem, page, result_path, img in extracted:
        out = BUNDLES_DIR / f"{stem}-page{page:02d}.bundle.json"
        cmd = [
            sys.executable,
            "-m",
            "scripts.make_verifier_bundle",
            str(result_path),
            str(img),
            "--out",
            str(out),
            "--pdf-path",
            _pdf_path_for_stem(stem),
            "--page-number",
            str(page),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"{stem} page {page:02d}: baked -> {out}")
        except subprocess.CalledProcessError as exc:
            print(f"{stem} page {page:02d}: BAKE FAILED rc={exc.returncode}: {exc.stderr}")

    print(f"\n-- done. {len(list(BUNDLES_DIR.glob('*.bundle.json')))} bundles in {BUNDLES_DIR} --")


if __name__ == "__main__":
    asyncio.run(main())
