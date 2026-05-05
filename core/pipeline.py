"""Pipeline orchestration: discover → render → process → store.

Each step is a top-level async function so it composes easily and stays
testable. The module owns no state — all state lives in `JobStore` and on
disk under `data_root/`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.gemini import GeminiClient
from core.jobs import JobStore
from core.render import RenderError, count_pages, render_page


def _pages_dir_for(data_root: Path, pdf_relpath: str) -> Path:
    """Per-PDF directory for rendered page images, mirroring the PDF layout."""
    return data_root / "pages" / Path(pdf_relpath).with_suffix("")


def _results_dir_for(data_root: Path, pdf_relpath: str) -> Path:
    """Per-PDF directory for extraction-result JSON files."""
    return data_root / "results" / Path(pdf_relpath).with_suffix("")


def result_path_for(data_root: Path, pdf_relpath: str, page_number: int) -> Path:
    """Where the JSON result for a single page lives on disk."""
    width = max(2, len(str(page_number)))
    return _results_dir_for(data_root, pdf_relpath) / f"page-{page_number:0{width}d}.json"


async def discover_pdfs(store: JobStore, scans_root: Path) -> int:
    """Walk `scans_root` for PDFs and register one job per page.

    Returns the number of newly-registered jobs (idempotent on re-run).
    """
    if not scans_root.is_dir():
        raise FileNotFoundError(f"scans_root does not exist: {scans_root}")

    new_count = 0
    pdfs = sorted(scans_root.rglob("*.pdf"))
    for pdf in pdfs:
        try:
            n_pages = count_pages(pdf)
        except RenderError:
            # Skip unreadable PDFs; they'll be visible because nothing got registered.
            continue
        rel = pdf.relative_to(scans_root).as_posix()
        for page in range(1, n_pages + 1):
            existing = await store.get(rel, page)
            await store.register(rel, page)
            if existing is None:
                new_count += 1
    return new_count


async def render_pending(
    store: JobStore,
    scans_root: Path,
    data_root: Path,
    dpi: int,
    limit: int,
) -> int:
    """Render up to `limit` pending pages. Returns the count rendered."""
    pending = await store.next_pending_for_render(limit=limit)
    if not pending:
        return 0

    rendered = 0
    for job in pending:
        pdf_abs = scans_root / job.pdf_path
        out_dir = _pages_dir_for(data_root, job.pdf_path)
        try:
            image_path = await asyncio.to_thread(
                render_page, pdf_abs, job.page_number, out_dir, dpi
            )
        except RenderError as exc:
            await store.mark_failed(job.pdf_path, job.page_number, error=str(exc))
            continue
        await store.mark_rendered(job.pdf_path, job.page_number, image_path=image_path)
        rendered += 1
    return rendered


async def process_pending(
    store: JobStore,
    client: GeminiClient,
    data_root: Path,
    limit: int,
    max_attempts: int,
) -> int:
    """Send up to `limit` rendered pages through Gemini. Returns successes."""
    pending = await store.next_pending_for_process(limit=limit, max_attempts=max_attempts)
    if not pending:
        return 0

    completed = 0
    for job in pending:
        if job.image_path is None:
            await store.mark_failed(
                job.pdf_path,
                job.page_number,
                error="rendered job has no image_path on record",
            )
            continue

        try:
            page_result = await client.extract_page(Path(job.image_path))
        except Exception as exc:  # noqa: BLE001
            # The pipeline runs over thousands of pages; one transient SDK or
            # network error must not abort the whole run. Record and move on.
            await store.mark_failed(job.pdf_path, job.page_number, error=str(exc))
            continue

        result_path = result_path_for(data_root, job.pdf_path, job.page_number)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(page_result.model_dump_json(indent=2))

        await store.mark_completed(
            job.pdf_path,
            job.page_number,
            result_path=result_path,
            model_version=page_result.model_version,
        )
        completed += 1
    return completed
