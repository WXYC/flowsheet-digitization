"""Pipeline orchestration: discover → render → process → store.

Each step is a top-level async function so it composes easily and stays
testable. The module owns no state — all state lives in `JobStore` and on
disk under `data_root/`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from core.gemini import GeminiClient
from core.jobs import Job, JobStore
from core.render import RenderError, count_pages, render_page
from core.schema import PageResult

ProcessProgressCallback = Callable[[str, int, bool], None]
"""Called once per finished job with (pdf_path, page_number, success)."""


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
    limit: int,
    *,
    concurrency: int = 1,
) -> int:
    """Render up to `limit` pending pages. Returns the count rendered.

    `concurrency` controls how many pdfimages subprocesses may be in flight
    at once. The work is CPU-bound but each subprocess releases the GIL,
    so an asyncio semaphore + to_thread gives near-linear speedup up to
    roughly the number of physical cores.

    Per-job failures (RenderError) mark only that job failed and do not abort
    the rest of the batch.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    pending = await store.next_pending_for_render(limit=limit)
    if not pending:
        return 0

    sem = asyncio.Semaphore(concurrency)

    async def render_one(job: Job) -> bool:
        pdf_abs = scans_root / job.pdf_path
        out_dir = _pages_dir_for(data_root, job.pdf_path)
        async with sem:
            try:
                image_path = await asyncio.to_thread(render_page, pdf_abs, job.page_number, out_dir)
            except RenderError as exc:
                await store.mark_failed(job.pdf_path, job.page_number, error=str(exc))
                return False
        await store.mark_rendered(job.pdf_path, job.page_number, image_path=image_path)
        return True

    results = await asyncio.gather(*(render_one(j) for j in pending))
    return sum(results)


async def process_pending(
    store: JobStore,
    client: GeminiClient,
    data_root: Path,
    limit: int,
    max_attempts: int,
    *,
    concurrency: int = 1,
    on_complete: ProcessProgressCallback | None = None,
) -> int:
    """Send up to `limit` rendered pages through Gemini. Returns successes.

    `concurrency` controls how many Gemini calls may be in flight at once.
    Tune to fit the project's per-minute rate limit; on the paid tier, 4-8
    is a reasonable starting point. Each call holds its semaphore slot for
    the full duration of the API request (~30-45s/page on Gemini 3.1 Pro).

    If `on_complete` is provided, it's invoked once per finished job with
    `(pdf_path, page_number, success)`. The CLI uses this to drive a
    live progress bar; tests use it to assert per-job behaviour.

    Per-job failures (any exception from the SDK or schema layer) mark only
    that job failed and do not abort the rest of the batch.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    pending = await store.next_pending_for_process(limit=limit, max_attempts=max_attempts)
    if not pending:
        return 0

    sem = asyncio.Semaphore(concurrency)

    async def process_one(job: Job) -> bool:
        success = await _process_one_job(job, store, client, data_root, sem)
        if on_complete is not None:
            on_complete(job.pdf_path, job.page_number, success)
        return success

    results = await asyncio.gather(*(process_one(j) for j in pending))
    return sum(results)


async def _process_one_job(
    job: Job,
    store: JobStore,
    client: GeminiClient,
    data_root: Path,
    sem: asyncio.Semaphore,
) -> bool:
    if job.image_path is None:
        await store.mark_failed(
            job.pdf_path,
            job.page_number,
            error="rendered job has no image_path on record",
        )
        return False

    async with sem:
        try:
            gemini_result = await client.extract_page(Path(job.image_path))
        except Exception as exc:  # noqa: BLE001
            # Pipeline runs over thousands of pages; one transient SDK / network
            # error must not abort the whole run. Record and move on.
            await store.mark_failed(job.pdf_path, job.page_number, error=str(exc))
            return False

    # Wrap into a PageResult with caller-set `model_version` / `extracted_at`.
    # Dict-merge order means our values win even if a stale fixture somehow
    # supplied them — see core.schema module docstring for the underlying bug.
    payload = gemini_result.model_dump()
    payload["model_version"] = client.model
    payload["extracted_at"] = datetime.now(UTC)
    page_result = PageResult.model_validate(payload)

    result_path = result_path_for(data_root, job.pdf_path, job.page_number)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(page_result.model_dump_json(indent=2))

    await store.mark_completed(
        job.pdf_path,
        job.page_number,
        result_path=result_path,
        model_version=page_result.model_version,
    )
    return True
