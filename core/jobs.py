"""SQLite-backed job state machine.

Each job is one (pdf_path, page_number) pair. Lifecycle:

    pending  ──render──▶ rendered ──process──▶ completed
       │                    │ │                    ▲
       │                    │ └──low_confidence────┤  (terminal in phase 1)
       │                    │                      │
       └────────────────────┴───── failed ─────────┘  (retried up to MAX_ATTEMPTS)

`completed` is terminal and is never automatically reprocessed; explicit
`retry()` flips it back to `rendered`. This protects already-extracted data
from accidental re-runs that could overwrite good results with worse ones.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import aiosqlite


class JobStatus(StrEnum):
    PENDING = "pending"
    RENDERING = "rendering"
    RENDERED = "rendered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    LOW_CONFIDENCE = "low_confidence"


class JobError(RuntimeError):
    """Raised on invalid state transitions or missing jobs."""


@dataclass(slots=True)
class Job:
    pdf_path: str
    page_number: int
    status: JobStatus
    attempts: int
    last_error: str | None
    image_path: str | None
    result_path: str | None
    model_version: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Self:
        return cls(
            pdf_path=row["pdf_path"],
            page_number=row["page_number"],
            status=JobStatus(row["status"]),
            attempts=row["attempts"],
            last_error=row["last_error"],
            image_path=row["image_path"],
            result_path=row["result_path"],
            model_version=row["model_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    pdf_path        TEXT NOT NULL,
    page_number     INTEGER NOT NULL,
    status          TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    image_path      TEXT,
    result_path     TEXT,
    model_version   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (pdf_path, page_number)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """Async SQLite job store.

    Each public method opens its own short-lived connection. That keeps the
    API simple at the cost of some per-call overhead — acceptable for an
    ETL with O(10K) pages, not for a hot-path service.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            # WAL lets concurrent readers run while a writer is active. With
            # multiple render workers each marking jobs `rendered`, this
            # avoids the writer-blocks-everyone behaviour of the default
            # rollback journal. The pragma is persistent across connections.
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def register(self, pdf_path: str, page_number: int) -> None:
        """Insert a pending job. No-op if the (pdf_path, page_number) already exists."""
        now = _now()
        async with self._connect() as db:
            await db.execute(
                """
                INSERT INTO jobs (
                    pdf_path, page_number, status, attempts,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(pdf_path, page_number) DO NOTHING
                """,
                (pdf_path, page_number, JobStatus.PENDING.value, now, now),
            )
            await db.commit()

    async def get(self, pdf_path: str, page_number: int) -> Job | None:
        async with self._connect() as db:
            cursor = await db.execute(
                "SELECT * FROM jobs WHERE pdf_path = ? AND page_number = ?",
                (pdf_path, page_number),
            )
            row = await cursor.fetchone()
            return Job.from_row(row) if row else None

    async def counts_by_status(self) -> dict[JobStatus, int]:
        async with self._connect() as db:
            cursor = await db.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            rows = await cursor.fetchall()
        return {JobStatus(r["status"]): r["n"] for r in rows}

    async def next_pending_for_render(self, limit: int) -> list[Job]:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY pdf_path, page_number
                LIMIT ?
                """,
                (JobStatus.PENDING.value, limit),
            )
            rows = await cursor.fetchall()
        return [Job.from_row(r) for r in rows]

    async def next_pending_for_process(self, limit: int, max_attempts: int) -> list[Job]:
        """Jobs ready to send to Gemini.

        Includes:
          * `rendered` (never tried)
          * `failed` rows whose `attempts` is below `max_attempts`
        """
        async with self._connect() as db:
            cursor = await db.execute(
                """
                SELECT * FROM jobs
                WHERE (status = ? OR (status = ? AND attempts < ?))
                ORDER BY pdf_path, page_number
                LIMIT ?
                """,
                (
                    JobStatus.RENDERED.value,
                    JobStatus.FAILED.value,
                    max_attempts,
                    limit,
                ),
            )
            rows = await cursor.fetchall()
        return [Job.from_row(r) for r in rows]

    async def mark_rendered(self, pdf_path: str, page_number: int, image_path: Path) -> None:
        await self._set_status(
            pdf_path,
            page_number,
            JobStatus.RENDERED,
            image_path=str(image_path),
        )

    async def mark_processing(self, pdf_path: str, page_number: int) -> None:
        await self._set_status(pdf_path, page_number, JobStatus.PROCESSING)

    async def mark_completed(
        self,
        pdf_path: str,
        page_number: int,
        result_path: Path,
        model_version: str,
    ) -> None:
        await self._set_status(
            pdf_path,
            page_number,
            JobStatus.COMPLETED,
            result_path=str(result_path),
            model_version=model_version,
            clear_error=True,
        )

    async def mark_low_confidence(
        self,
        pdf_path: str,
        page_number: int,
        result_path: Path,
        model_version: str,
    ) -> None:
        await self._set_status(
            pdf_path,
            page_number,
            JobStatus.LOW_CONFIDENCE,
            result_path=str(result_path),
            model_version=model_version,
            clear_error=True,
        )

    async def mark_failed(self, pdf_path: str, page_number: int, error: str) -> None:
        async with self._connect() as db:
            cursor = await db.execute(
                """
                UPDATE jobs
                SET status = ?, attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE pdf_path = ? AND page_number = ?
                """,
                (JobStatus.FAILED.value, error, _now(), pdf_path, page_number),
            )
            if cursor.rowcount == 0:
                raise JobError(f"no such job: {pdf_path} page {page_number}")
            await db.commit()

    async def retry(self, pdf_path: str, page_number: int) -> None:
        """Move a completed/low_confidence job back to `rendered` for reprocessing.

        Only allowed when the job has an `image_path` — we won't re-render here.
        """
        job = await self.get(pdf_path, page_number)
        if job is None:
            raise JobError(f"no such job: {pdf_path} page {page_number}")
        if job.image_path is None:
            raise JobError(f"cannot retry {pdf_path} page {page_number}: no image_path on record")
        async with self._connect() as db:
            await db.execute(
                """
                UPDATE jobs
                SET status = ?, last_error = NULL, updated_at = ?
                WHERE pdf_path = ? AND page_number = ?
                """,
                (JobStatus.RENDERED.value, _now(), pdf_path, page_number),
            )
            await db.commit()

    async def _set_status(
        self,
        pdf_path: str,
        page_number: int,
        status: JobStatus,
        *,
        image_path: str | None = None,
        result_path: str | None = None,
        model_version: str | None = None,
        clear_error: bool = False,
    ) -> None:
        sets = ["status = ?", "updated_at = ?"]
        params: list[object] = [status.value, _now()]
        if image_path is not None:
            sets.append("image_path = ?")
            params.append(image_path)
        if result_path is not None:
            sets.append("result_path = ?")
            params.append(result_path)
        if model_version is not None:
            sets.append("model_version = ?")
            params.append(model_version)
        if clear_error:
            sets.append("last_error = NULL")

        params.extend([pdf_path, page_number])
        sql = f"UPDATE jobs SET {', '.join(sets)} WHERE pdf_path = ? AND page_number = ?"

        async with self._connect() as db:
            cursor = await db.execute(sql, params)
            if cursor.rowcount == 0:
                raise JobError(f"no such job: {pdf_path} page {page_number}")
            await db.commit()
