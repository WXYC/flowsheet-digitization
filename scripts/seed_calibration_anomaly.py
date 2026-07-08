"""Bootstrap the calibration anomaly bucket for the 5 named seed bundles.

Copies (via symlink) the seed bundles from `data/verifier/<stem>.bundle.json`
into the year-stratified calibration tree under
`data/calibration/<year>/anomaly/<stem>/bundle.json`.

The five seeds come from `project_bbox_sweep_result.md`: the strict
detector's pathology candidates. See plans/multi-reviewer-calibration.md
§Bootstrap for the rationale on why the existing hand-reviewed
`verified.json` files are deliberately NOT copied over.

Properties:

  * **Idempotent**: re-running creates nothing new and removes nothing.
  * **Validating**: refuses to create broken links if a seed bundle is
    missing from `data/verifier/`.
  * **Surgical**: touches only the seed bundles' directories.

Flags:

  * `--dry-run`: print operations without performing them.
  * `--refresh-reviewers`: rebuild `data/calibration/_reviewers.json`
    from the embedded `verified_by` blocks in existing `verified.*.json`
    submissions. Useful when a reviewer's real_name / dj_name changes on
    the auth server and the local mapping goes stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))
VERIFIER_DIR = DATA_ROOT / "verifier"
CALIBRATION_ROOT = DATA_ROOT / "calibration"

# The 5 pathology-bundle candidates from the strict detector
# (`project_bbox_sweep_result.md`).
SEED_STEMS: tuple[str, ...] = (
    "1990-04apr0106-page14",
    "1990-04apr0106-page28",
    "1990-04apr0106-page29",
    "1990-04apr0106-page34",
    "1990-04apr2430-page23",
)

BUCKET = "anomaly"

logger = logging.getLogger("seed_calibration_anomaly")


def year_from_stem(stem: str) -> str:
    """Extract the 4-digit year from a stem like `1990-04apr0106-page14`."""
    return stem.split("-", 1)[0]


def _short(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def seed_one(stem: str, *, dry_run: bool = False) -> str:
    """Create the calibration page dir for one seed stem.

    Returns a short status string: "created", "exists", or "would-create"
    (in dry-run mode).
    """
    year = year_from_stem(stem)
    page_dir = CALIBRATION_ROOT / year / BUCKET / stem
    bundle_link = page_dir / "bundle.json"
    bundle_source = VERIFIER_DIR / f"{stem}.bundle.json"

    if not bundle_source.is_file():
        raise SystemExit(f"missing source bundle: {bundle_source} — cannot seed {stem!r}")

    # `is_symlink()` catches broken links that `exists()` reports as False.
    if bundle_link.is_symlink() or bundle_link.exists():
        return "exists"

    if dry_run:
        return "would-create"

    page_dir.mkdir(parents=True, exist_ok=True)
    relative_target = Path("../../../..") / "verifier" / f"{stem}.bundle.json"
    bundle_link.symlink_to(relative_target)
    return "created"


def refresh_reviewers_mapping(*, dry_run: bool = False) -> None:
    """Rebuild `_reviewers.json` from the embedded verified_by blocks
    across every existing verified.*.json submission.

    Append-only per plan: an entry is only written on first sight of a
    given `user_id`. This flag exists to recover from a stale username /
    real_name / dj_name change on the auth server.
    """
    mapping_path = CALIBRATION_ROOT / "_reviewers.json"
    current: dict[str, dict[str, object]] = {}
    for verified_file in CALIBRATION_ROOT.rglob("verified.*.json"):
        try:
            payload = json.loads(verified_file.read_text())
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", verified_file, exc)
            continue
        reviewer = payload.get("reviewer") if isinstance(payload, dict) else None
        if not isinstance(reviewer, dict):
            continue
        user_id = reviewer.get("user_id")
        if not isinstance(user_id, str):
            continue
        short = _short(user_id)
        if short in current:
            continue
        current[short] = {
            "user_id": user_id,
            "username": reviewer.get("username"),
            "real_name": reviewer.get("real_name"),
            "dj_name": reviewer.get("dj_name"),
        }
    if dry_run:
        logger.info("would write %d reviewer entries to %s", len(current), mapping_path)
        return
    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = mapping_path.with_suffix(mapping_path.suffix + ".tmp")
    tmp.write_text(json.dumps(current, indent=2))
    os.replace(tmp, mapping_path)
    logger.info("wrote %d reviewer entries to %s", len(current), mapping_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print operations without performing them",
    )
    parser.add_argument(
        "--refresh-reviewers",
        action="store_true",
        help=(
            "rebuild data/calibration/_reviewers.json from existing "
            "verified.*.json submissions and exit (no seeding)"
        ),
    )
    args = parser.parse_args(argv)

    if args.refresh_reviewers:
        refresh_reviewers_mapping(dry_run=args.dry_run)
        return 0

    if not VERIFIER_DIR.is_dir():
        logger.error("verifier dir missing: %s", VERIFIER_DIR)
        return 1

    for stem in SEED_STEMS:
        status = seed_one(stem, dry_run=args.dry_run)
        logger.info("%s: %s", stem, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
