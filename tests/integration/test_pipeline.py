"""Integration tests for the orchestration pipeline.

Mocks Gemini at the SDK boundary so no network is hit. Uses real pdftoppm
on hand-rolled blank PDFs (skipped if poppler is not installed).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.gemini import GeminiClient
from core.jobs import JobStatus, JobStore
from core.pipeline import (
    discover_pdfs,
    process_pending,
    render_pending,
    result_path_for,
)
from core.schema import Entry, PageResult, Quadrant
from tests.unit.test_render import _make_blank_pdf  # reuse the blank-PDF helper

POPPLER_AVAILABLE = shutil.which("pdftoppm") is not None and shutil.which("pdfinfo") is not None


def _build_page_result(date: str = "Monday 1 Jan '90") -> PageResult:
    return PageResult(
        page_date_raw=date,
        quadrants=[
            Quadrant(
                position="top_left",
                hour_raw="6AM",
                jock_raw="ALECIA",
                entries=[
                    Entry(
                        row_index=0,
                        raw_text="LED ZEP - TRAMPLED",
                        artist_guess="LED ZEP",
                        track_guess="TRAMPLED",
                        confidence="high",
                    )
                ],
            ),
            Quadrant(position="top_right", hour_raw=None, jock_raw=None, entries=[]),
            Quadrant(position="bottom_left", hour_raw=None, jock_raw=None, entries=[]),
            Quadrant(position="bottom_right", hour_raw=None, jock_raw=None, entries=[]),
        ],
        model_version="gemini-3.1-pro-preview",
        extracted_at=datetime.now(UTC),
    )


def _fake_gemini_client(parsed: PageResult | Exception) -> GeminiClient:
    response = MagicMock()
    if isinstance(parsed, Exception):
        generate_content = AsyncMock(side_effect=parsed)
    else:
        response.parsed = parsed
        generate_content = AsyncMock(return_value=response)
    sdk = MagicMock()
    sdk.aio.models.generate_content = generate_content
    return GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")


@pytest.fixture
async def store(tmp_path: Path) -> JobStore:
    s = JobStore(tmp_path / "jobs.db")
    await s.init()
    return s


@pytest.fixture
def scans_root(tmp_path: Path) -> Path:
    """Create a small scans/ tree with two PDFs (3 pages and 2 pages)."""
    root = tmp_path / "scans"
    (root / "1990" / "January 1990").mkdir(parents=True)
    (root / "1990" / "February 1990").mkdir(parents=True)
    _make_blank_pdf(root / "1990" / "January 1990" / "1990-01jan0106.pdf", n_pages=3)
    _make_blank_pdf(root / "1990" / "February 1990" / "1990-02feb0106.pdf", n_pages=2)
    return root


def test_result_path_for_mirrors_pdf_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    p = result_path_for(
        data_root=data_root,
        pdf_relpath="1990/January 1990/1990-01jan0106.pdf",
        page_number=1,
    )
    assert p == data_root / "results" / "1990" / "January 1990" / "1990-01jan0106" / "page-01.json"


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_discover_pdfs_registers_one_job_per_page(store: JobStore, scans_root: Path) -> None:
    n = await discover_pdfs(store, scans_root=scans_root)
    assert n == 5  # 3 + 2

    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 5}


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_discover_pdfs_is_idempotent(store: JobStore, scans_root: Path) -> None:
    await discover_pdfs(store, scans_root=scans_root)
    n_second = await discover_pdfs(store, scans_root=scans_root)
    assert n_second == 0  # nothing new
    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 5}


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_render_pending_renders_and_marks_rendered(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)

    n = await render_pending(store, scans_root=scans_root, data_root=data_root, dpi=72, limit=10)
    assert n == 5

    counts = await store.counts_by_status()
    assert counts == {JobStatus.RENDERED: 5}

    # Each rendered file is on disk.
    rendered = list(data_root.glob("pages/**/page-*.png"))
    assert len(rendered) == 5


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_process_pending_writes_json_and_marks_completed(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, dpi=72, limit=10)

    client = _fake_gemini_client(_build_page_result())
    n = await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    assert n == 5

    counts = await store.counts_by_status()
    assert counts == {JobStatus.COMPLETED: 5}

    # Result JSON exists and round-trips through PageResult.
    results = list(data_root.glob("results/**/page-*.json"))
    assert len(results) == 5
    sample = json.loads(results[0].read_text())
    PageResult.model_validate(sample)


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_process_pending_marks_failed_on_exception(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, dpi=72, limit=10)

    client = _fake_gemini_client(RuntimeError("rate limit exceeded"))
    n = await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    # Every job tried once and failed.
    assert n == 0  # zero successes

    counts = await store.counts_by_status()
    assert counts.get(JobStatus.FAILED) == 5

    # All five recorded the error message.
    for job in await store.next_pending_for_process(limit=10, max_attempts=99):
        assert job.last_error is not None
        assert "rate limit" in job.last_error
        assert job.attempts == 1


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
async def test_full_run_end_to_end(store: JobStore, scans_root: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, dpi=72, limit=10)

    client = _fake_gemini_client(_build_page_result())
    await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)

    # Re-running discover + render + process is idempotent (nothing repeats).
    assert (await discover_pdfs(store, scans_root=scans_root)) == 0
    assert (
        await render_pending(store, scans_root=scans_root, data_root=data_root, dpi=72, limit=10)
    ) == 0
    assert (
        await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    ) == 0

    counts = await store.counts_by_status()
    assert counts == {JobStatus.COMPLETED: 5}
