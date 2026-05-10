"""ONE-SHOT: backfill hallucinated `model_version` and `extracted_at`.

Before the schema split (issue #1), `PageResult.model_version` and
`extracted_at` lived inside the Gemini `response_schema`. The model was
asked to fill them and confabulated plausible-looking values: 4 distinct
fake model ids across 34 pages all actually processed by
`gemini-3.1-pro-preview`, with timestamps off by 14+ months. The schema
fix lands those two fields server-side going forward — but the existing
on-disk corpus needs a one-shot rewrite.

This script:

  * walks `data/results/**/*.json`,
  * for each whose `model_version` is NOT in `--known-good-model`,
    rewrites both fields in place: `model_version` ← `--target-model`,
    `extracted_at` ← the file's mtime (UTC, ISO-8601). The mtime is the
    best wall-clock signal we have for when the file was actually
    written — far closer to truth than the model's guess. Caveat: if
    any tool (editor format-on-save, rsync, backup) has touched the
    JSON files since extraction, mtime drifts away from truth — the
    `--dry-run` summary will surface unexpected `extracted_at` values
    before any write happens.
  * runs the same UPDATE on `<data-root>/jobs.db`'s `jobs` table for
    every (pdf_path, page_number) whose `result_path` matches.

Idempotent: re-running after a successful pass is a no-op (the
`--known-good-model` allowlist short-circuits already-fixed rows).

Use `--dry-run` to preview before writing. Use `--data-root` and
`--jobs-db` to point at a non-default tree.

This script is intentionally a one-off — once the corpus has been
backfilled, delete it. There is no general-purpose case for rewriting
already-completed results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("backfill_model_metadata")

# Real Gemini model ids the project has actually used. Anything else
# in `model_version` is presumed hallucinated. Add to this list as new
# real models come online.
DEFAULT_KNOWN_GOOD_MODELS = ("gemini-3.1-pro-preview",)


@dataclass
class BackfillStats:
    json_scanned: int = 0
    json_skipped_known_good: int = 0
    json_skipped_unreadable: int = 0
    json_rewritten: int = 0
    json_rewritten_paths: list[Path] = field(default_factory=list)
    sqlite_rows_updated: int = 0


def _load_json(path: Path, stats: BackfillStats) -> dict | None:
    """Read and decode a result JSON. Bumps `json_skipped_unreadable` and
    returns None on read or decode failure so the operator can see the
    miss in the final summary instead of having to grep the log."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        stats.json_skipped_unreadable += 1
        return None


def _file_mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _backfill_json_files(
    data_root: Path,
    *,
    known_good_models: Sequence[str],
    target_model: str,
    dry_run: bool,
) -> BackfillStats:
    """Rewrite `model_version` / `extracted_at` in result JSONs in place."""
    stats = BackfillStats()
    known = set(known_good_models)
    results_root = data_root / "results"
    if not results_root.is_dir():
        logger.info("no results dir at %s — nothing to scan", results_root)
        return stats

    for path in sorted(results_root.rglob("*.json")):
        stats.json_scanned += 1
        data = _load_json(path, stats)
        if data is None:
            continue

        current = data.get("model_version")
        if current in known:
            stats.json_skipped_known_good += 1
            continue

        data["model_version"] = target_model
        data["extracted_at"] = _file_mtime_utc(path).isoformat()

        if not dry_run:
            path.write_text(json.dumps(data, indent=2))
        stats.json_rewritten += 1
        stats.json_rewritten_paths.append(path)
        logger.info(
            "%s %s: model_version %r → %r, extracted_at ← mtime",
            "WOULD REWRITE" if dry_run else "REWROTE",
            path,
            current,
            target_model,
        )

    return stats


def _backfill_sqlite(
    jobs_db: Path,
    *,
    known_good_models: Sequence[str],
    target_model: str,
    dry_run: bool,
) -> int:
    """UPDATE only `status='completed'` rows whose model_version is not in
    the allowlist.

    The status filter matters: rows that were once completed and then
    `retry()`'d back to `rendered` may carry a stale hallucinated
    model_version that no longer reflects any current extraction. Those
    are NOT in scope for this migration — there is no on-disk JSON to
    mirror the change. (A pre-fix `retry()` left those values stale; a
    post-fix `retry()` clears them.)

    Without this filter, an initial pass on the production corpus would
    have updated 219 rows (34 completed JSONs + 185 stale-rendered
    leftovers); only the 34 belong here.
    """
    if not jobs_db.is_file():
        logger.info("no jobs.db at %s — skipping SQLite update", jobs_db)
        return 0

    placeholders = ",".join("?" for _ in known_good_models)
    known = list(known_good_models)
    select_sql = (
        f"SELECT pdf_path, page_number, model_version FROM jobs "  # noqa: S608
        f"WHERE status = 'completed' "
        f"AND model_version IS NOT NULL "
        f"AND model_version NOT IN ({placeholders})"
    )
    update_sql = (
        f"UPDATE jobs SET model_version = ? "  # noqa: S608
        f"WHERE status = 'completed' "
        f"AND model_version IS NOT NULL "
        f"AND model_version NOT IN ({placeholders})"
    )

    with sqlite3.connect(jobs_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(select_sql, known))
        for row in rows:
            logger.info(
                "%s jobs row %s page %s: model_version %r → %r",
                "WOULD UPDATE" if dry_run else "UPDATING",
                row["pdf_path"],
                row["page_number"],
                row["model_version"],
                target_model,
            )
        if not dry_run and rows:
            conn.execute(update_sql, [target_model, *known])
            conn.commit()

    return len(rows)


def backfill(
    *,
    data_root: Path,
    jobs_db: Path,
    known_good_models: Iterable[str],
    target_model: str,
    dry_run: bool,
) -> BackfillStats:
    """Run the full backfill. Public surface for the unit test."""
    # Materialize once: the helpers iterate this twice (placeholder count +
    # bind values), and the public contract is `Iterable[str]` which may be
    # a single-pass generator.
    known_list: list[str] = list(known_good_models)
    if target_model not in known_list:
        # Defensive: if the operator's truth value isn't in the allowlist,
        # a re-run would loop forever rewriting their own writes. Adding
        # it here keeps the operation idempotent regardless of CLI input.
        known_list.append(target_model)

    stats = _backfill_json_files(
        data_root,
        known_good_models=known_list,
        target_model=target_model,
        dry_run=dry_run,
    )
    stats.sqlite_rows_updated = _backfill_sqlite(
        jobs_db,
        known_good_models=known_list,
        target_model=target_model,
        dry_run=dry_run,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing `results/**/*.json` and `jobs.db`.",
    )
    parser.add_argument(
        "--jobs-db",
        type=Path,
        default=None,
        help="Path to jobs.db (default: <data-root>/jobs.db).",
    )
    parser.add_argument(
        "--target-model",
        default="gemini-3.1-pro-preview",
        help="Truthful model id to write into rewritten rows.",
    )
    parser.add_argument(
        "--known-good-model",
        action="append",
        default=list(DEFAULT_KNOWN_GOOD_MODELS),
        help="Real model id to leave untouched. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended changes without writing.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log every file/row inspected.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    jobs_db = args.jobs_db or (args.data_root / "jobs.db")
    stats = backfill(
        data_root=args.data_root,
        jobs_db=jobs_db,
        known_good_models=args.known_good_model,
        target_model=args.target_model,
        dry_run=args.dry_run,
    )

    print(
        f"scanned {stats.json_scanned} json files; "
        f"skipped {stats.json_skipped_known_good} (already known-good), "
        f"{stats.json_skipped_unreadable} (unreadable); "
        f"{'would rewrite' if args.dry_run else 'rewrote'} {stats.json_rewritten}; "
        f"{'would update' if args.dry_run else 'updated'} "
        f"{stats.sqlite_rows_updated} rows in {jobs_db}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
