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


async def test_init_enables_wal_mode(tmp_path: Path) -> None:
    """WAL mode is required for healthy concurrency between render workers.

    Without it, every writer briefly locks readers, which slows the
    render-pending dispatcher when several workers mark_rendered at once.
    """
    import aiosqlite

    s = JobStore(tmp_path / "jobs.db")
    await s.init()

    async with aiosqlite.connect(s.db_path) as db:
        cursor = await db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
    assert row is not None
    assert row[0].lower() == "wal"


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


async def test_retry_clears_completion_metadata(store: JobStore, tmp_path: Path) -> None:
    """`retry()` flips status back to RENDERED — but the previous run's
    `model_version` and `result_path` describe a completion that no
    longer exists. Leaving them set is a lie about state and an
    operational footgun: an offline analysis bucketing by `model_version`
    sees a stale id, and an idempotent migration that filters by
    `model_version IS NOT NULL` over-touches.

    Real incident: 185 rows in production reached `status=rendered` with
    a stale `model_version` after a bulk retry in May 2026. The reset
    code path cleared `last_error` but not `model_version` / `result_path`.
    """
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_completed(
        "scans/a.pdf",
        1,
        result_path=tmp_path / "a.json",
        model_version="gemini-3.1-pro-preview",
    )

    # Sanity: completed row has both fields set.
    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.model_version == "gemini-3.1-pro-preview"
    assert job.result_path == str(tmp_path / "a.json")

    await store.retry("scans/a.pdf", 1)

    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.status == JobStatus.RENDERED
    # The PRIOR completion's metadata must be cleared — no current
    # extraction backs them up.
    assert job.model_version is None
    assert job.result_path is None
    # `image_path` is preserved — the rendered PNG is still on disk and
    # is what re-processing will read.
    assert job.image_path == str(tmp_path / "a.png")


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


# -- verification tracking --------------------------------------------------


async def test_mark_verified_records_paths_without_changing_status(
    store: JobStore, tmp_path: Path
) -> None:
    """`mark_verified` updates the verification columns but leaves `status`
    alone — verification is orthogonal to the extraction state machine."""
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_completed("scans/a.pdf", 1, result_path=tmp_path / "a.json", model_version="m")

    verified = tmp_path / "a.verified.json"
    corrections = tmp_path / "a.corrections.json"
    matched = await store.mark_verified(
        "scans/a.pdf", 1, verified_path=verified, corrections_path=corrections
    )
    assert matched is True

    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.verified_at is not None
    assert job.verified_path == str(verified)
    assert job.corrections_path == str(corrections)


async def test_mark_verified_returns_false_when_no_matching_job(
    store: JobStore, tmp_path: Path
) -> None:
    """The verifier server may try to record verification for a test
    fixture that has no `jobs.db` row. Returns False instead of raising
    so the server can fall back to file-only persistence."""
    matched = await store.mark_verified(
        "scans/no-such.pdf",
        99,
        verified_path=tmp_path / "x.verified.json",
        corrections_path=tmp_path / "x.corrections.json",
    )
    assert matched is False


async def test_init_adds_late_columns_to_existing_db(tmp_path: Path) -> None:
    """`init()` against a pre-verification-column DB adds the columns
    without losing data."""
    import aiosqlite

    db_path = tmp_path / "old.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE jobs (
                pdf_path TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                image_path TEXT,
                result_path TEXT,
                model_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (pdf_path, page_number)
            )
            """
        )
        await db.commit()

    store = JobStore(db_path)
    await store.init()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        rows = await (await db.execute("PRAGMA table_info(jobs)")).fetchall()
    cols = {r["name"] for r in rows}
    assert "verified_at" in cols
    assert "verified_path" in cols
    assert "corrections_path" in cols
    # reviewer_id is the late column added by the OIDC PR. Without this
    # branch the migration on an existing prod jobs.db would silently
    # leave the column off and every mark_verified(reviewer_id=...) call
    # would persist no reviewer at all.
    assert "reviewer_id" in cols


# -- reviewer_id (OIDC PR) -------------------------------------------------


async def test_mark_verified_round_trips_reviewer_id(store: JobStore, tmp_path: Path) -> None:
    """`mark_verified(..., reviewer_id="u-123")` persists to `jobs.reviewer_id`.

    The verifier server denormalizes the OIDC user_id onto the row so
    per-reviewer queries don't have to parse every verified.json on disk.
    """
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")
    await store.mark_completed("scans/a.pdf", 1, result_path=tmp_path / "a.json", model_version="m")

    matched = await store.mark_verified(
        "scans/a.pdf",
        1,
        verified_path=tmp_path / "a.verified.json",
        corrections_path=tmp_path / "a.corrections.json",
        reviewer_id="u-123",
    )
    assert matched is True

    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.reviewer_id == "u-123"


async def test_mark_verified_accepts_none_reviewer_id(store: JobStore, tmp_path: Path) -> None:
    """`reviewer_id=None` (the default) writes NULL — the BasicAuth and
    no-auth deployments don't have a reviewer to credit and the column
    should accept that without a NOT NULL violation."""
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")

    matched = await store.mark_verified(
        "scans/a.pdf",
        1,
        verified_path=tmp_path / "a.verified.json",
        corrections_path=tmp_path / "a.corrections.json",
        # reviewer_id omitted; defaults to None.
    )
    assert matched is True

    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.reviewer_id is None


async def test_mark_verified_overwrites_reviewer_id_on_re_save(
    store: JobStore, tmp_path: Path
) -> None:
    """A second `mark_verified` call updates `reviewer_id` to the latest
    reviewer. Verification is "who last saved this page", not an
    append-only log — matches the file-side semantics where verified.json
    is rewritten by every save."""
    await store.register("scans/a.pdf", 1)
    await store.mark_rendered("scans/a.pdf", 1, image_path=tmp_path / "a.png")

    paths = {
        "verified_path": tmp_path / "a.verified.json",
        "corrections_path": tmp_path / "a.corrections.json",
    }
    await store.mark_verified("scans/a.pdf", 1, **paths, reviewer_id="first")
    await store.mark_verified("scans/a.pdf", 1, **paths, reviewer_id="second")

    job = await store.get("scans/a.pdf", 1)
    assert job is not None
    assert job.reviewer_id == "second"
