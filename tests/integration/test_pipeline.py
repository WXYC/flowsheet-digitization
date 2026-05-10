"""Integration tests for the orchestration pipeline.

Mocks Gemini at the SDK boundary so no network is hit. Uses the bundled
fixture PDF (real CCITT-G4 embedded images) when poppler is available.
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
from core.schema import Entry, GeminiPageResult, PageResult, Quadrant

POPPLER_AVAILABLE = shutil.which("pdfimages") is not None and shutil.which("pdfinfo") is not None
FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "three_pages_with_images.pdf"
FIXTURE_PDF_PAGES = 3


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
    """Create a small scans/ tree by copying the fixture PDF into two locations."""
    root = tmp_path / "scans"
    (root / "1990" / "January 1990").mkdir(parents=True)
    (root / "1990" / "February 1990").mkdir(parents=True)
    payload = FIXTURE_PDF.read_bytes()
    (root / "1990" / "January 1990" / "1990-01jan0106.pdf").write_bytes(payload)
    (root / "1990" / "February 1990" / "1990-02feb0106.pdf").write_bytes(payload)
    return root


SCANS_ROOT_TOTAL_PAGES = FIXTURE_PDF_PAGES * 2


def test_result_path_for_mirrors_pdf_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    p = result_path_for(
        data_root=data_root,
        pdf_relpath="1990/January 1990/1990-01jan0106.pdf",
        page_number=1,
    )
    assert p == data_root / "results" / "1990" / "January 1990" / "1990-01jan0106" / "page-01.json"


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_discover_pdfs_registers_one_job_per_page(store: JobStore, scans_root: Path) -> None:
    n = await discover_pdfs(store, scans_root=scans_root)
    assert n == 6  # 3 + 3

    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 6}


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_discover_pdfs_is_idempotent(store: JobStore, scans_root: Path) -> None:
    await discover_pdfs(store, scans_root=scans_root)
    n_second = await discover_pdfs(store, scans_root=scans_root)
    assert n_second == 0  # nothing new
    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 6}


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_render_pending_renders_and_marks_rendered(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)

    n = await render_pending(store, scans_root=scans_root, data_root=data_root, limit=10)
    assert n == 6

    counts = await store.counts_by_status()
    assert counts == {JobStatus.RENDERED: 6}

    # Each rendered file is on disk.
    rendered = list(data_root.glob("pages/**/page-*.png"))
    assert len(rendered) == 6


async def test_render_pending_respects_concurrency_limit(
    store: JobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peak in-flight render_page calls must not exceed `concurrency`.

    We don't need a real pdfimages call here — patch render_page to measure
    overlap and complete quickly.
    """
    import threading
    import time

    from core import pipeline as pipeline_mod

    # Register 12 jobs and mark them all as pending → next_pending_for_render returns them.
    for i in range(12):
        await store.register("scans/x.pdf", i + 1)

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    def fake_render(
        pdf_path: Path, page_number: int, out_dir: Path, *, force: bool = False
    ) -> Path:
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        time.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"page-{page_number:02d}.png"
        out.write_bytes(b"\x89PNG")
        return out

    monkeypatch.setattr(pipeline_mod, "render_page", fake_render)

    n = await render_pending(
        store,
        scans_root=tmp_path / "scans",
        data_root=tmp_path / "data",
        limit=12,
        concurrency=3,
    )
    assert n == 12
    assert state["peak"] <= 3
    # Sanity: with 12 jobs and concurrency 3, we should exceed serial-1 in flight.
    assert state["peak"] >= 2


async def test_render_pending_one_failure_does_not_kill_others(
    store: JobStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RenderError on one job must mark only that job failed and leave others intact."""
    from core import pipeline as pipeline_mod
    from core.render import RenderError

    for i in range(5):
        await store.register("scans/x.pdf", i + 1)

    def maybe_fail(pdf_path: Path, page_number: int, out_dir: Path, *, force: bool = False) -> Path:
        if page_number == 3:
            raise RenderError("simulated render failure on page 3")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"page-{page_number:02d}.png"
        out.write_bytes(b"\x89PNG")
        return out

    monkeypatch.setattr(pipeline_mod, "render_page", maybe_fail)

    n = await render_pending(
        store,
        scans_root=tmp_path / "scans",
        data_root=tmp_path / "data",
        limit=10,
        concurrency=3,
    )
    assert n == 4  # 4 succeeded, 1 failed

    counts = await store.counts_by_status()
    assert counts.get(JobStatus.RENDERED) == 4
    assert counts.get(JobStatus.FAILED) == 1


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_process_pending_writes_json_and_marks_completed(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, limit=10)

    client = _fake_gemini_client(_build_page_result())
    n = await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    assert n == 6

    counts = await store.counts_by_status()
    assert counts == {JobStatus.COMPLETED: 6}

    # Result JSON exists and round-trips through PageResult.
    results = list(data_root.glob("results/**/page-*.json"))
    assert len(results) == 6
    sample = json.loads(results[0].read_text())
    PageResult.model_validate(sample)


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_process_pending_marks_failed_on_exception(
    store: JobStore, scans_root: Path, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, limit=10)

    client = _fake_gemini_client(RuntimeError("rate limit exceeded"))
    n = await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    # Every job tried once and failed.
    assert n == 0  # zero successes

    counts = await store.counts_by_status()
    assert counts.get(JobStatus.FAILED) == 6

    # All five recorded the error message.
    for job in await store.next_pending_for_process(limit=10, max_attempts=99):
        assert job.last_error is not None
        assert "rate limit" in job.last_error
        assert job.attempts == 1


async def test_process_pending_respects_concurrency_limit(
    store: JobStore,
    tmp_path: Path,
) -> None:
    """Peak in-flight Gemini calls must not exceed `concurrency`.

    Uses a fake SDK whose generate_content sleeps briefly while incrementing
    a peak counter, so we observe real overlap.
    """
    import asyncio as _asyncio
    import threading

    data_root = tmp_path / "data"
    # Register and pre-render 12 jobs by hand (skipping the actual pdfimages step).
    for i in range(12):
        await store.register("scans/x.pdf", i + 1)
        img = tmp_path / f"img-{i + 1}.png"
        img.write_bytes(b"\x89PNG")
        await store.mark_rendered("scans/x.pdf", i + 1, image_path=img)

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}

    async def fake_generate(*args: object, **kwargs: object) -> MagicMock:
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        await _asyncio.sleep(0.02)
        with lock:
            state["in_flight"] -= 1
        response = MagicMock()
        response.parsed = _build_page_result()
        return response

    sdk = MagicMock()
    sdk.aio.models.generate_content = AsyncMock(side_effect=fake_generate)
    client = GeminiClient(sdk=sdk, model="m")

    n = await process_pending(
        store, client=client, data_root=data_root, limit=12, max_attempts=3, concurrency=3
    )
    assert n == 12
    assert state["peak"] <= 3
    # With 12 jobs and concurrency 3, expect actual overlap.
    assert state["peak"] >= 2


async def test_process_pending_calls_progress_callback_on_each_completion(
    store: JobStore, tmp_path: Path
) -> None:
    """The CLI uses this hook to render a live progress bar."""
    data_root = tmp_path / "data"
    for i in range(3):
        await store.register("scans/x.pdf", i + 1)
        img = tmp_path / f"img-{i + 1}.png"
        img.write_bytes(b"\x89PNG")
        await store.mark_rendered("scans/x.pdf", i + 1, image_path=img)

    client = _fake_gemini_client(_build_page_result())

    events: list[tuple[str, int, bool]] = []

    def on_complete(pdf_path: str, page_number: int, success: bool) -> None:
        events.append((pdf_path, page_number, success))

    n = await process_pending(
        store,
        client=client,
        data_root=data_root,
        limit=10,
        max_attempts=3,
        on_complete=on_complete,
    )
    assert n == 3
    assert len(events) == 3
    assert all(success for _, _, success in events)
    assert {p for p, _, _ in events} == {"scans/x.pdf"}


async def test_process_pending_progress_callback_fires_on_failure_too(
    store: JobStore, tmp_path: Path
) -> None:
    data_root = tmp_path / "data"
    await store.register("scans/x.pdf", 1)
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG")
    await store.mark_rendered("scans/x.pdf", 1, image_path=img)

    client = _fake_gemini_client(RuntimeError("boom"))

    events: list[tuple[str, int, bool]] = []
    n = await process_pending(
        store,
        client=client,
        data_root=data_root,
        limit=10,
        max_attempts=3,
        on_complete=lambda p, pg, ok: events.append((p, pg, ok)),
    )
    assert n == 0
    assert events == [("scans/x.pdf", 1, False)]


async def test_process_pending_overrides_model_version_and_extracted_at(
    store: JobStore, tmp_path: Path
) -> None:
    """The pipeline owns `model_version` and `extracted_at`, not Gemini.

    Even if the SDK somehow returned a `GeminiPageResult`-or-richer object
    with those fields populated, the on-disk JSON must reflect:
      - `model_version` == the id passed to the SDK (`GeminiClient(model=...)`)
      - `extracted_at` within seconds of `now`, in UTC

    This is the regression guard for the original bug (see core.schema
    module docstring): real run with `gemini-3.1-pro-preview` produced 4
    distinct hallucinated model_version values and timestamps off by 14
    months, all because those fields were inside the response_schema.
    """
    data_root = tmp_path / "data"
    await store.register("scans/x.pdf", 1)
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG")
    await store.mark_rendered("scans/x.pdf", 1, image_path=img)

    # GeminiPageResult intentionally has NO model_version / extracted_at —
    # this is the production shape after the split.
    gemini_result = GeminiPageResult(
        page_date_raw="Monday 1 Jan '90",
        quadrants=[
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
            for p in ("top_left", "top_right", "bottom_left", "bottom_right")
        ],
    )
    response = MagicMock()
    response.parsed = gemini_result
    sdk = MagicMock()
    sdk.aio.models.generate_content = AsyncMock(return_value=response)
    client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")

    before = datetime.now(UTC)
    n = await process_pending(store, client=client, data_root=data_root, limit=1, max_attempts=3)
    after = datetime.now(UTC)
    assert n == 1

    [result_file] = list(data_root.glob("results/**/page-*.json"))
    written = PageResult.model_validate_json(result_file.read_text())

    assert written.model_version == "gemini-3.1-pro-preview"
    assert before <= written.extracted_at <= after
    assert written.extracted_at.tzinfo is not None  # UTC, not naive


async def test_process_pending_overwrites_stale_metadata_in_parsed_result(
    store: JobStore, tmp_path: Path
) -> None:
    """Defense in depth: if a stale fixture, broken SDK, or older parsed
    result somehow slips a `PageResult` (subclass) past `extract_page`
    with bogus `model_version` / `extracted_at`, the pipeline still wins.

    Without this overwrite, the dict-merge in `_process_one_job` would
    have to be in the wrong order to silently propagate a hallucinated
    field. This test pins the merge order.
    """
    data_root = tmp_path / "data"
    await store.register("scans/x.pdf", 1)
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG")
    await store.mark_rendered("scans/x.pdf", 1, image_path=img)

    stale = PageResult(
        page_date_raw=None,
        quadrants=[
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[])
            for p in ("top_left", "top_right", "bottom_left", "bottom_right")
        ],
        model_version="gemini-2.5-pro",  # the bug — a hallucinated id
        extracted_at=datetime(2024, 1, 1, tzinfo=UTC),  # wrong by ~16 months
    )
    response = MagicMock()
    response.parsed = stale
    sdk = MagicMock()
    sdk.aio.models.generate_content = AsyncMock(return_value=response)
    client = GeminiClient(sdk=sdk, model="gemini-3.1-pro-preview")

    before = datetime.now(UTC)
    await process_pending(store, client=client, data_root=data_root, limit=1, max_attempts=3)
    after = datetime.now(UTC)

    [result_file] = list(data_root.glob("results/**/page-*.json"))
    written = PageResult.model_validate_json(result_file.read_text())

    assert written.model_version == "gemini-3.1-pro-preview"
    assert before <= written.extracted_at <= after


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
async def test_full_run_end_to_end(store: JobStore, scans_root: Path, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    await discover_pdfs(store, scans_root=scans_root)
    await render_pending(store, scans_root=scans_root, data_root=data_root, limit=10)

    client = _fake_gemini_client(_build_page_result())
    await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)

    # Re-running discover + render + process is idempotent (nothing repeats).
    assert (await discover_pdfs(store, scans_root=scans_root)) == 0
    assert (await render_pending(store, scans_root=scans_root, data_root=data_root, limit=10)) == 0
    assert (
        await process_pending(store, client=client, data_root=data_root, limit=10, max_attempts=3)
    ) == 0

    counts = await store.counts_by_status()
    assert counts == {JobStatus.COMPLETED: 6}
