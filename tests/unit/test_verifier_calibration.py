"""Tests for the calibration-mode endpoints on `verifier/serve.py`.

Covers the read/write API surface added by the multi-reviewer calibration
plan. Uses the OIDC-enabled app configuration so `request.state.reviewer`
is populated the same way it will be in production.

Blind-review enforcement is the load-bearing property under test: no
reviewer may read another reviewer's `verified.<short>.json` until the
page reaches settlement (canonical.json + agreement.json land atomically).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import core.auth as auth_mod
from core.auth import ReviewerSession
from tests.unit.conftest import _reset_serve_module_state

ISSUER = "https://auth.example/auth"
CLIENT_ID = "flowsheet-test"
CLIENT_SECRET = "test-client-secret"
PUBLIC_URL = "https://flowsheet.example"
SESSION_SECRET = "x" * 64

STEM = "1990-04apr0106-page14"
YEAR = "1990"
BUCKET = "anomaly"


def _short(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _make_reviewer(user_id: str = "sub-a", username: str = "dj_a") -> ReviewerSession:
    return ReviewerSession(
        user_id=user_id,
        email=f"{username}@wxyc.org",
        username=username,
        real_name="Real Name",
        dj_name="Stage Name",
        role="dj",
    )


def _session_cookie(reviewer: ReviewerSession) -> str:
    return auth_mod.encode_session(reviewer)


def _set_oidc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WXYC_AUTH_ISSUER", ISSUER)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("WXYC_OIDC_CLIENT_SECRET", CLIENT_SECRET)
    monkeypatch.setenv("WXYC_SESSION_SECRET", SESSION_SECRET)
    monkeypatch.setenv("FLOWSHEET_PUBLIC_URL", PUBLIC_URL)


def _seed_bundle(root: Path, stem: str = STEM) -> Path:
    """Create a minimal seed bundle under data/verifier/ + a symlinked
    calibration page dir under data/calibration/<year>/<bucket>/<stem>/."""
    verifier_dir = root / "data" / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    src_bundle = verifier_dir / f"{stem}.bundle.json"
    src_bundle.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stem": stem,
                "image_path": "../pages/1990-04apr0106/page-14.png",
                "pdf_path": "1990-04apr0106.pdf",
                "page_number": 14,
                "model_version": "test",
                "extracted_at": "2026-01-01T00:00:00Z",
                "page_date_raw": "Mon 1 Jan 90",
                "comments_raw": None,
                "oddities": [],
                "quadrants": [
                    {
                        "position": "top_left",
                        "bbox": [0, 0, 640, 480],
                        "hour_raw": "6AM",
                        "jock_raw": "DJ ONE",
                        "entries": [
                            {
                                "row_index": 0,
                                "raw_text": "BEATLES - HELP",
                                "confidence": "high",
                                "type_raw": "H",
                                "notes": None,
                                "oddities": [],
                                "row_bbox": [0, 0, 640, 48],
                            }
                        ],
                        "oddities": [],
                    },
                    {
                        "position": "top_right",
                        "bbox": [640, 0, 1280, 480],
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                    {
                        "position": "bottom_left",
                        "bbox": [0, 480, 640, 960],
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                    {
                        "position": "bottom_right",
                        "bbox": [640, 480, 1280, 960],
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                ],
            }
        )
    )
    page_dir = root / "data" / "calibration" / YEAR / BUCKET / stem
    page_dir.mkdir(parents=True, exist_ok=True)
    (page_dir / "bundle.json").symlink_to(Path("../../../..") / "verifier" / f"{stem}.bundle.json")
    return page_dir


@pytest.fixture
def serve_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    _set_oidc_env(monkeypatch)

    import verifier.serve as serve_mod

    importlib.reload(serve_mod)
    _reset_serve_module_state(serve_mod)
    yield serve_mod
    monkeypatch.undo()
    importlib.reload(serve_mod)
    _reset_serve_module_state(serve_mod)


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _submission_body(rows_count: int = 1) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stem": STEM,
        "rows": [
            {
                "bundle_row_index": i,
                "edited_text": "BEATLES - HELP",
                "type_raw": "H",
                "notes": None,
                "spurious_flag": False,
            }
            for i in range(rows_count)
        ],
        "missing_row_markers": [],
    }


# --------------------------------------------------------------------------- #
# GET /api/calibration/<year>/<bucket>/<stem>/bundle
# --------------------------------------------------------------------------- #


async def test_bundle_read_serves_symlinked_content(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/bundle")
    assert r.status_code == 200
    body = r.json()
    assert body["stem"] == STEM


async def test_bundle_read_rejects_unauthenticated(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        r = await c.get(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/bundle",
            headers={"accept": "application/json"},
            follow_redirects=False,
        )
    assert r.status_code == 401


async def test_bundle_read_rejects_path_traversal_year(serve_app) -> None:
    reviewer = _make_reviewer()
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.get(f"/api/calibration/..%2F..%2Fetc/{BUCKET}/{STEM}/bundle")
    assert r.status_code in (400, 404)


# --------------------------------------------------------------------------- #
# POST /api/calibration/<year>/<bucket>/<stem>/draft
# --------------------------------------------------------------------------- #


async def test_draft_save_writes_draft_short_file(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft",
            json=_submission_body(),
        )
    assert r.status_code == 200
    draft_path = (
        tmp_path / "data" / "calibration" / YEAR / BUCKET / STEM / f"draft.{_short('sub-a')}.json"
    )
    assert draft_path.is_file()


async def test_draft_read_own_returns_content(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft",
            json=_submission_body(),
        )
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft")
    assert r.status_code == 200
    assert r.json()["stem"] == STEM


async def test_draft_read_no_draft_returns_404(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# POST /api/calibration/<year>/<bucket>/<stem>/submit
# --------------------------------------------------------------------------- #


async def test_submit_writes_verified_atomically(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
    assert r.status_code == 200
    verified_path = (
        tmp_path
        / "data"
        / "calibration"
        / YEAR
        / BUCKET
        / STEM
        / f"verified.{_short('sub-a')}.json"
    )
    assert verified_path.is_file()
    body = json.loads(verified_path.read_text())
    assert body["stem"] == STEM
    assert body["reviewer"]["user_id"] == "sub-a"


async def test_submit_second_by_same_reviewer_is_rejected(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        first = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        assert first.status_code == 200
        second = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
    assert second.status_code == 400


async def test_second_reviewer_submit_settles_page(serve_app, tmp_path: Path) -> None:
    """Two agreeing reviewers → canonical.json + agreement.json land."""
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        r1 = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        assert r1.status_code == 200
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-b")))
        r2 = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
    assert r2.status_code == 200
    page_dir = tmp_path / "data" / "calibration" / YEAR / BUCKET / STEM
    assert (page_dir / "canonical.json").is_file()
    assert (page_dir / "agreement.json").is_file()


# --------------------------------------------------------------------------- #
# Blind-review enforcement
# --------------------------------------------------------------------------- #


async def test_reading_other_reviewers_verified_pre_settlement_denied(
    serve_app, tmp_path: Path
) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        r = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        assert r.status_code == 200
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-b")))
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/verified/{_short('sub-a')}")
    assert r.status_code == 403


async def test_reading_own_verified_always_allowed(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        r = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        assert r.status_code == 200
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/verified/{_short('sub-a')}")
    assert r.status_code == 200


async def test_reading_other_reviewers_verified_post_settlement_allowed(
    serve_app, tmp_path: Path
) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        # Two agreeing reviewers → settlement.
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-b")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        # sub-a can now read sub-b's file (settled).
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/verified/{_short('sub-b')}")
    assert r.status_code == 200


async def test_reading_other_reviewers_draft_denied(serve_app, tmp_path: Path) -> None:
    """Drafts are owner-only regardless of settlement — you can't peek at
    an in-progress reading."""
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft",
            json=_submission_body(),
        )
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-b")))
        # sub-b's own draft doesn't exist → 404, not a leak of sub-a's.
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/draft")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Canonical / agreement reads
# --------------------------------------------------------------------------- #


async def test_canonical_read_denied_before_settlement(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        r = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/canonical")
    assert r.status_code == 404


async def test_canonical_and_agreement_read_after_settlement(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-b")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        canon = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/canonical")
        agree = await c.get(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/agreement")
    assert canon.status_code == 200
    assert agree.status_code == 200
    assert canon.json()["stem"] == STEM
    assert agree.json()["year"] == YEAR


# --------------------------------------------------------------------------- #
# Queue
# --------------------------------------------------------------------------- #


async def test_queue_lists_reviewer_eligible_pages(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.get("/api/calibration/queue")
    assert r.status_code == 200
    body = r.json()
    stems = [item["stem"] for item in body["pages"]]
    assert STEM in stems
    entry = next(item for item in body["pages"] if item["stem"] == STEM)
    assert entry["your_state"] == "not_started"
    assert entry["page_state"] == "awaiting_submissions"


async def test_queue_hides_already_submitted_pages(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(_make_reviewer("sub-a")))
        await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(),
        )
        r = await c.get("/api/calibration/queue")
    assert r.status_code == 200
    stems = [item["stem"] for item in r.json()["pages"]]
    assert STEM not in stems


# --------------------------------------------------------------------------- #
# Submission body validation
# --------------------------------------------------------------------------- #


async def test_submit_rejects_row_count_mismatch(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.post(
            f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit",
            json=_submission_body(rows_count=2),
        )
    assert r.status_code == 400


async def test_submit_rejects_out_of_range_missing_row_marker(serve_app, tmp_path: Path) -> None:
    _seed_bundle(tmp_path)
    body = _submission_body()
    body["missing_row_markers"] = [
        {
            "between_bundle_rows": [10, 11],
            "suggested_text": "OUT OF RANGE",
            "type_raw": None,
            "notes": None,
        }
    ]
    reviewer = _make_reviewer("sub-a")
    async with await _client(serve_app.app) as c:
        c.cookies.set("flowsheet_session", _session_cookie(reviewer))
        r = await c.post(f"/api/calibration/{YEAR}/{BUCKET}/{STEM}/submit", json=body)
    assert r.status_code == 400
