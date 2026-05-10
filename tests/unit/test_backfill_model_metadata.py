"""Tests for the one-shot backfill script (`scripts/backfill_model_metadata.py`).

The script is throwaway, but a unit test pins the load-bearing behaviors:

  * hallucinated rows get rewritten, known-good rows do not (idempotency),
  * `extracted_at` matches the file's mtime, in UTC,
  * SQLite gets the equivalent UPDATE,
  * `--dry-run` is non-destructive.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.backfill_model_metadata import backfill


def _write_result(
    data_root: Path,
    *,
    relpath: str,
    page: int,
    model_version: str,
    extracted_at: str,
    mtime_ts: float | None = None,
) -> Path:
    """Write a synthetic result JSON; optionally clamp its mtime."""
    p = data_root / "results" / relpath / f"page-{page:02d}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "page_date_raw": "Monday 1 Jan '90",
                "quadrants": [
                    {
                        "position": pos,
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    }
                    for pos in ("top_left", "top_right", "bottom_left", "bottom_right")
                ],
                "model_version": model_version,
                "extracted_at": extracted_at,
                "oddities": [],
            }
        )
    )
    if mtime_ts is not None:
        os.utime(p, (mtime_ts, mtime_ts))
    return p


def _seed_jobs_db(db_path: Path, rows: list[tuple[str, int, str]]) -> None:
    """Seed a minimal jobs.db with just enough columns for the backfill SQL."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                pdf_path TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                model_version TEXT,
                PRIMARY KEY (pdf_path, page_number)
            )
            """
        )
        conn.executemany(
            "INSERT INTO jobs (pdf_path, page_number, model_version) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()


def _read_model_versions(db_path: Path) -> dict[tuple[str, int], str | None]:
    with sqlite3.connect(db_path) as conn:
        return {
            (row[0], row[1]): row[2]
            for row in conn.execute("SELECT pdf_path, page_number, model_version FROM jobs")
        }


@pytest.fixture
def truth_mtime() -> datetime:
    """A wall-clock time that's clearly distinct from the bogus values."""
    return datetime(2026, 5, 5, 21, 30, tzinfo=UTC)


def test_rewrites_hallucinated_model_version_and_extracted_at(
    tmp_path: Path, truth_mtime: datetime
) -> None:
    data_root = tmp_path / "data"
    p = _write_result(
        data_root,
        relpath="1990/January 1990/1990-01jan",
        page=1,
        model_version="gemini-2.5-pro",  # hallucinated
        extracted_at="2024-08-15T12:00:00+00:00",  # 16 months wrong
        mtime_ts=truth_mtime.timestamp(),
    )

    stats = backfill(
        data_root=data_root,
        jobs_db=data_root / "jobs.db",  # absent; SQLite step is a no-op
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )

    assert stats.json_rewritten == 1
    rewritten = json.loads(p.read_text())
    assert rewritten["model_version"] == "gemini-3.1-pro-preview"
    written = datetime.fromisoformat(rewritten["extracted_at"])
    # mtime is the truth signal — not the original hallucinated timestamp.
    assert abs((written - truth_mtime).total_seconds()) < 1


def test_known_good_model_left_untouched(tmp_path: Path) -> None:
    """Idempotency: a row already carrying the truth value must not be
    rewritten on a second pass."""
    data_root = tmp_path / "data"
    original = _write_result(
        data_root,
        relpath="1990/January 1990/1990-01jan",
        page=2,
        model_version="gemini-3.1-pro-preview",
        extracted_at="2026-05-05T21:30:00+00:00",
    )
    before = original.read_text()

    stats = backfill(
        data_root=data_root,
        jobs_db=data_root / "jobs.db",
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )

    assert stats.json_rewritten == 0
    assert stats.json_skipped_known_good == 1
    # Byte-for-byte identical: file was not touched.
    assert original.read_text() == before


def test_unreadable_json_files_are_counted_and_skipped(tmp_path: Path) -> None:
    """A corrupted result JSON must not abort the run, but the operator
    needs to see the miss in the final summary — bumping a dedicated
    `json_skipped_unreadable` counter surfaces it without forcing a log
    grep."""
    data_root = tmp_path / "data"
    bad = data_root / "results" / "1990" / "page-01.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{ this is not valid json")

    stats = backfill(
        data_root=data_root,
        jobs_db=data_root / "jobs.db",
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )
    assert stats.json_scanned == 1
    assert stats.json_skipped_unreadable == 1
    assert stats.json_rewritten == 0


def test_dry_run_makes_no_changes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    p = _write_result(
        data_root,
        relpath="1990/A/1990-01jan",
        page=3,
        model_version="gemini-2.0-flash",
        extracted_at="2024-08-15T12:00:00+00:00",
    )
    before = p.read_text()

    stats = backfill(
        data_root=data_root,
        jobs_db=data_root / "jobs.db",
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=True,
    )

    assert stats.json_rewritten == 1  # would have rewritten
    assert p.read_text() == before  # but didn't


def test_sqlite_rows_updated_for_hallucinated_model_versions(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "results").mkdir(parents=True)  # also creates data_root
    db = data_root / "jobs.db"
    _seed_jobs_db(
        db,
        rows=[
            ("1990/A.pdf", 1, "gemini-2.5-pro"),  # hallucinated
            ("1990/A.pdf", 2, "gemini-2.0-flash"),  # hallucinated
            ("1990/B.pdf", 1, "gemini-3.1-pro-preview"),  # truth
            ("1990/C.pdf", 1, None),  # never completed
        ],
    )

    stats = backfill(
        data_root=data_root,
        jobs_db=db,
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )
    assert stats.sqlite_rows_updated == 2

    after = _read_model_versions(db)
    assert after[("1990/A.pdf", 1)] == "gemini-3.1-pro-preview"
    assert after[("1990/A.pdf", 2)] == "gemini-3.1-pro-preview"
    assert after[("1990/B.pdf", 1)] == "gemini-3.1-pro-preview"  # untouched
    assert after[("1990/C.pdf", 1)] is None  # NULL stays NULL


def test_accepts_single_pass_iterator_for_known_good_models(
    tmp_path: Path, truth_mtime: datetime
) -> None:
    """`known_good_models` is typed `Iterable[str]`; the public contract
    must hold for single-pass iterators (generators), not just lists.

    Regression guard: an earlier draft consumed the iterable twice
    inside the SQLite helper (once for `?`-placeholders, once for the
    bind values), silently producing a binding-count mismatch when a
    generator was passed."""
    data_root = tmp_path / "data"
    (data_root / "results").mkdir(parents=True)
    db = data_root / "jobs.db"
    _seed_jobs_db(db, rows=[("1990/A.pdf", 1, "gemini-2.5-pro")])
    _write_result(
        data_root,
        relpath="1990/A",
        page=1,
        model_version="gemini-2.5-pro",
        extracted_at="2024-08-15T12:00:00+00:00",
        mtime_ts=truth_mtime.timestamp(),
    )

    stats = backfill(
        data_root=data_root,
        jobs_db=db,
        known_good_models=(name for name in ["gemini-3.1-pro-preview"]),  # generator
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )
    assert stats.json_rewritten == 1
    assert stats.sqlite_rows_updated == 1


def test_idempotent_on_repeat_run(tmp_path: Path, truth_mtime: datetime) -> None:
    """Run the script twice. After pass #1 every row carries the truth
    value; pass #2 is a no-op. This is the operator safeguard — running
    the same backfill twice cannot silently accumulate stale timestamps."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    db = data_root / "jobs.db"
    _seed_jobs_db(db, rows=[("1990/A.pdf", 1, "gemini-2.5-pro")])
    p = _write_result(
        data_root,
        relpath="1990/A",
        page=1,
        model_version="gemini-2.5-pro",
        extracted_at="2024-08-15T12:00:00+00:00",
        mtime_ts=truth_mtime.timestamp(),
    )

    first = backfill(
        data_root=data_root,
        jobs_db=db,
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )
    assert first.json_rewritten == 1
    assert first.sqlite_rows_updated == 1

    # Pin mtime back to truth; the rewrite touched the file and bumped it.
    os.utime(p, (truth_mtime.timestamp(), truth_mtime.timestamp()))

    second = backfill(
        data_root=data_root,
        jobs_db=db,
        known_good_models=["gemini-3.1-pro-preview"],
        target_model="gemini-3.1-pro-preview",
        dry_run=False,
    )
    assert second.json_rewritten == 0
    assert second.sqlite_rows_updated == 0
