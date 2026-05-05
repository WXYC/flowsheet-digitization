"""Tests for the SQLite job state machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.jobs import JobError, JobStatus, JobStore


@pytest.fixture
async def store(tmp_path: Path) -> JobStore:
    s = JobStore(tmp_path / "jobs.db")
    await s.init()
    return s


async def test_init_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    s1 = JobStore(db)
    await s1.init()
    s2 = JobStore(db)
    await s2.init()
    counts = await s2.counts_by_status()
    assert counts == {}


async def test_register_and_count(store: JobStore) -> None:
    await store.register("scans/a.pdf", page_number=1)
    await store.register("scans/a.pdf", page_number=2)
    await store.register("scans/b.pdf", page_number=1)
    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 3}


async def test_register_is_idempotent(store: JobStore) -> None:
    await store.register("scans/a.pdf", page_number=1)
    await store.register("scans/a.pdf", page_number=1)
    counts = await store.counts_by_status()
    assert counts == {JobStatus.PENDING: 1}


async def test_mark_rendered_then_processing_then_completed(
    store: JobStore, tmp_path: Path
) -> None:
    await store.register("scans/a.pdf", page_number=1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "img.png")
    await store.mark_processing("scans/a.pdf", 1)
    await store.mark_completed(
        "scans/a.pdf",
        1,
        result_path=tmp_path / "r.json",
        model_version="gemini-3.1-pro-preview",
    )
    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.image_path is not None and job.image_path.endswith("img.png")
    assert job.result_path is not None and job.result_path.endswith("r.json")
    assert job.model_version == "gemini-3.1-pro-preview"


async def test_mark_failed_increments_attempts(store: JobStore) -> None:
    await store.register("scans/a.pdf", page_number=1)
    await store.mark_failed("scans/a.pdf", 1, error="rate-limited")
    await store.mark_failed("scans/a.pdf", 1, error="rate-limited")
    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.attempts == 2
    assert job.last_error == "rate-limited"


async def test_completed_jobs_never_returned_for_processing(
    store: JobStore, tmp_path: Path
) -> None:
    await store.register("scans/a.pdf", 1)
    await store.register("scans/b.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_rendered("scans/b.pdf", 1, image_path=tmp_path / "b.png")
    await store.mark_processing("scans/a.pdf", 1)
    await store.mark_completed("scans/a.pdf", 1, result_path=tmp_path / "a.json", model_version="m")

    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=3)
    pdfs = [j.pdf_path for j in next_jobs]
    assert "scans/a.pdf" not in pdfs
    assert "scans/b.pdf" in pdfs


async def test_failed_retried_only_under_max_attempts(store: JobStore, tmp_path: Path) -> None:
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    for _ in range(3):
        await store.mark_failed("scans/a.pdf", 1, error="boom")

    # max_attempts=3 means: attempts < 3 → retry. After 3 attempts, no retry.
    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=3)
    assert next_jobs == []

    # max_attempts=4 → eligible again.
    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=4)
    assert len(next_jobs) == 1


async def test_retry_completed_requires_explicit(store: JobStore, tmp_path: Path) -> None:
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_completed("scans/a.pdf", 1, result_path=tmp_path / "a.json", model_version="m")

    # Without retry: still completed.
    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=3)
    assert next_jobs == []

    # Explicit retry resets to rendered (so process picks it up).
    await store.retry("scans/a.pdf", 1)
    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.status == JobStatus.RENDERED
    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=3)
    assert len(next_jobs) == 1


async def test_retry_unknown_job_raises(store: JobStore) -> None:
    with pytest.raises(JobError):
        await store.retry("scans/nope.pdf", 1)


async def test_low_confidence_is_terminal(store: JobStore, tmp_path: Path) -> None:
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_low_confidence(
        "scans/a.pdf", 1, result_path=tmp_path / "a.json", model_version="m"
    )

    next_jobs = await store.next_pending_for_process(limit=10, max_attempts=3)
    assert next_jobs == []
    counts = await store.counts_by_status()
    assert counts.get(JobStatus.LOW_CONFIDENCE) == 1


async def test_pending_for_render(store: JobStore, tmp_path: Path) -> None:
    await store.register("scans/a.pdf", 1)
    await store.register("scans/a.pdf", 2)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a1.png")

    pending = await store.next_pending_for_render(limit=10)
    assert [(j.pdf_path, j.page_number) for j in pending] == [("scans/a.pdf", 2)]
