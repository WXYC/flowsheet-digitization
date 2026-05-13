"""Tests for `verifier/serve.py`.

`/api/save` is the load-bearing endpoint for the verifier UI — it
validates the verified payload as `PageResult`, guards against path
traversal via the bundle stem, writes both files to `data/verifier/`,
and conditionally updates `jobs.db` via `JobStore.mark_verified`.

These tests use httpx's ASGI transport to exercise the FastAPI app
in-process (no live server needed, no port collision with a running
`verifier/serve.py`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from core.jobs import JobStore
from core.schema import QUADRANT_ORDER, PageResult, Quadrant


def _page_result_dict() -> dict[str, Any]:
    """Minimal valid PageResult payload for the verified-export body."""
    return PageResult(
        page_date_raw="Mon 1 Jan 90",
        quadrants=[
            Quadrant(position=p, hour_raw=None, jock_raw=None, entries=[], oddities=[])
            for p in QUADRANT_ORDER
        ],
        comments_raw=None,
        oddities=[],
        model_version="test-model",
        extracted_at=datetime(2026, 5, 12, tzinfo=UTC),
    ).model_dump(mode="json")


def _corrections_dict() -> dict[str, Any]:
    return {
        "stem": "test",
        "model_version": "test-model",
        "extracted_at": "2026-05-12T00:00:00Z",
        "exported_at": "2026-05-12T00:00:01Z",
        "page_corrections": [],
        "quadrant_corrections": [],
        "row_corrections": [],
        "added_rows": [],
        "deleted_rows": [],
    }


@pytest.fixture
def serve_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a fresh FastAPI app rooted at `tmp_path` so each test
    starts with an empty `data/verifier/` and its own `jobs.db`."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    # Reimport to pick up the environment variable (DATA_ROOT is read at
    # module import time, not per-request).
    import importlib

    import verifier.serve as serve_mod

    importlib.reload(serve_mod)
    yield serve_mod
    # Restore the module to its default for any subsequent test that
    # imports it without the env override.
    monkeypatch.undo()
    importlib.reload(serve_mod)


async def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# -- /api/save body validation ---------------------------------------------


async def test_save_rejects_missing_verified(serve_app, tmp_path: Path) -> None:
    """Body must include `verified` and `corrections` objects."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={"stem": "abc", "corrections": _corrections_dict()},
        )
    assert r.status_code == 400
    assert "verified" in r.json()["detail"]


async def test_save_rejects_invalid_pageresult(serve_app, tmp_path: Path) -> None:
    """Verified payload must validate as `PageResult` — a malformed one
    is rejected before any file write."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "abc",
                "verified": {"quadrants": []},  # missing required fields, wrong shape
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 400
    assert "PageResult" in r.json()["detail"]
    # Nothing written.
    assert not (tmp_path / "data" / "verifier").exists()


async def test_save_rejects_path_traversal_stem(serve_app, tmp_path: Path) -> None:
    """`stem` containing `/`, `\\`, or `..` is refused so the server
    can't be tricked into writing outside `data/verifier/`. Whitespace-
    only stems are also rejected — they'd produce confusing ` .verified.json`
    files."""
    async with await _client(serve_app.app) as c:
        for bad in ("../escape", "a/b", "..", "a\\b", "", "   ", "\t"):
            r = await c.post(
                "/api/save",
                json={
                    "stem": bad,
                    "verified": _page_result_dict(),
                    "corrections": _corrections_dict(),
                },
            )
            assert r.status_code == 400, f"expected 400 for stem={bad!r}, got {r.status_code}"


# -- /api/save file persistence --------------------------------------------


async def test_save_writes_both_files(serve_app, tmp_path: Path) -> None:
    """A valid payload writes `<stem>.verified.json` and
    `<stem>.corrections.json` under `<DATA_ROOT>/verifier/`."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "page25",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["db_updated"] is False  # no pdf_path/page_number sent
    verifier_dir = tmp_path / "data" / "verifier"
    verified = verifier_dir / "page25.verified.json"
    corrections = verifier_dir / "page25.corrections.json"
    assert verified.is_file()
    assert corrections.is_file()
    # `verified.json` round-trips through `PageResult` — the on-disk file
    # is the consumable artifact, so the test pins its parseability.
    PageResult.model_validate_json(verified.read_text())
    # Corrections is opaque JSON — pin only that it's well-formed.
    json.loads(corrections.read_text())


async def test_save_strips_bundle_only_fields_via_pydantic_roundtrip(
    serve_app, tmp_path: Path
) -> None:
    """A client that leaks bundle-only fields (row_bbox, schema_version,
    etc.) shouldn't pollute the on-disk verified.json. The server's
    `PageResult.model_validate(...).model_dump_json(...)` round-trip
    strips unknown fields by Pydantic's default `extra='ignore'`."""
    polluted = _page_result_dict()
    # Simulate the UI accidentally leaking bundle metadata into the
    # verified payload (these don't belong on PageResult).
    polluted["schema_version"] = 2
    polluted["stem"] = "page25"
    polluted["image_path"] = "../tests/golden/x.png"
    polluted["quadrants"][0]["bbox"] = [0, 0, 100, 100]
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "polluted",
                "verified": polluted,
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    on_disk = json.loads((tmp_path / "data" / "verifier" / "polluted.verified.json").read_text())
    assert "schema_version" not in on_disk
    assert "stem" not in on_disk
    assert "image_path" not in on_disk
    # Per-quadrant bbox is bundle-only too.
    assert "bbox" not in on_disk["quadrants"][0]


async def test_save_overwrites_previous_files(serve_app, tmp_path: Path) -> None:
    """Re-saving the same stem overwrites — verification is the latest
    edit state, not an append-only log."""
    payload = {
        "stem": "p",
        "verified": _page_result_dict(),
        "corrections": _corrections_dict(),
    }
    async with await _client(serve_app.app) as c:
        await c.post("/api/save", json=payload)
        # Second save with a tweaked date.
        payload["verified"]["page_date_raw"] = "Tues 2 Jan 90"
        r2 = await c.post("/api/save", json=payload)
    assert r2.status_code == 200
    verified = tmp_path / "data" / "verifier" / "p.verified.json"
    assert json.loads(verified.read_text())["page_date_raw"] == "Tues 2 Jan 90"


# -- /api/save DB integration ----------------------------------------------


async def test_save_updates_jobs_db_when_job_key_matches(serve_app, tmp_path: Path) -> None:
    """When `pdf_path` + `page_number` are present AND `jobs.db` has a
    matching row, the verification is recorded via `JobStore.mark_verified`
    and `db_updated: true` is returned."""
    db_path = tmp_path / "data" / "jobs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = JobStore(db_path)
    await store.init()
    await store.register("1990/x.pdf", 1)
    await store.mark_rendered("1990/x.pdf", 1, image_path=tmp_path / "x.png")
    await store.mark_completed("1990/x.pdf", 1, result_path=tmp_path / "x.json", model_version="m")

    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "x-page-01",
                "pdf_path": "1990/x.pdf",
                "page_number": 1,
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["db_updated"] is True

    job = await store.get("1990/x.pdf", 1)
    assert job is not None
    assert job.verified_at is not None
    assert job.verified_path is not None and job.verified_path.endswith("x-page-01.verified.json")
    assert job.corrections_path is not None and job.corrections_path.endswith(
        "x-page-01.corrections.json"
    )


async def test_save_returns_db_updated_false_when_no_matching_job(
    serve_app, tmp_path: Path
) -> None:
    """A job key that doesn't match any row in `jobs.db` is not an error
    — the server writes files and reports `db_updated: false`. Lets test
    fixtures and ad-hoc pages save without a pre-registered job."""
    db_path = tmp_path / "data" / "jobs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize an empty jobs.db so the file exists but has no matching row.
    await JobStore(db_path).init()

    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "ghost",
                "pdf_path": "1990/no-such.pdf",
                "page_number": 99,
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["db_updated"] is False
    # Files still written.
    assert (tmp_path / "data" / "verifier" / "ghost.verified.json").is_file()


async def test_save_rejects_bool_page_number(serve_app, tmp_path: Path) -> None:
    """`isinstance(x, int)` is True for `bool` in Python — a malformed
    `page_number: true` would coerce to 1 and look up the wrong job
    row. Defensive: bool is rejected; the save still succeeds but with
    `db_updated: false` (treated as no-job-key)."""
    db_path = tmp_path / "data" / "jobs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = JobStore(db_path)
    await store.init()
    await store.register("1990/x.pdf", 1)

    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "bool-test",
                "pdf_path": "1990/x.pdf",
                "page_number": True,  # boolean, not real int
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["db_updated"] is False  # bool was rejected, files only
    # The job row at page 1 should NOT have been updated.
    job = await store.get("1990/x.pdf", 1)
    assert job is not None
    assert job.verified_at is None


async def test_save_writes_are_atomic_no_tmp_left_behind(serve_app, tmp_path: Path) -> None:
    """Atomic writes use `.tmp` siblings + os.replace. After a successful
    save, no `.tmp` files remain in data/verifier/."""
    async with await _client(serve_app.app) as c:
        await c.post(
            "/api/save",
            json={
                "stem": "atomic",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    verifier_dir = tmp_path / "data" / "verifier"
    tmp_files = list(verifier_dir.glob("*.tmp"))
    assert tmp_files == [], f"unexpected tmp files left: {tmp_files}"


async def test_save_writes_status_draft_by_default(serve_app, tmp_path: Path) -> None:
    """A Save with no `status` field writes corrections.json with
    `status: "draft"` — the default for a partial / in-progress page."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "draft-default",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["status"] == "draft"
    on_disk = json.loads(
        (tmp_path / "data" / "verifier" / "draft-default.corrections.json").read_text()
    )
    assert on_disk["status"] == "draft"


async def test_save_writes_status_complete_when_requested(serve_app, tmp_path: Path) -> None:
    """An explicit `status: "complete"` from the UI's Mark complete button
    persists as `"complete"`."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "mark-done",
                "status": "complete",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["status"] == "complete"


async def test_save_preserves_complete_on_subsequent_draft_save(serve_app, tmp_path: Path) -> None:
    """Once a page is `complete`, a subsequent plain Save (default
    `draft` or omitted status) does NOT downgrade it. Refining details
    on a completed page is a tweak-in-place, not a status change."""
    body_draft = {
        "stem": "preserve",
        "verified": _page_result_dict(),
        "corrections": _corrections_dict(),
    }
    body_complete = {**body_draft, "status": "complete"}
    async with await _client(serve_app.app) as c:
        await c.post("/api/save", json=body_complete)
        # Now save again with no status — should stay complete.
        r = await c.post("/api/save", json=body_draft)
    assert r.status_code == 200
    assert r.json()["status"] == "complete"
    on_disk = json.loads((tmp_path / "data" / "verifier" / "preserve.corrections.json").read_text())
    assert on_disk["status"] == "complete"


async def test_save_explicit_draft_reverts_complete(serve_app, tmp_path: Path) -> None:
    """An explicit `status: "draft"` from the UI's toggleable Mark complete
    button DOES revert a complete page back to draft. The preserve-on-disk
    rule only applies when the client omits the status field (plain Save)."""
    body = {
        "stem": "revert",
        "verified": _page_result_dict(),
        "corrections": _corrections_dict(),
    }
    body_complete = {**body, "status": "complete"}
    body_explicit_draft = {**body, "status": "draft"}
    async with await _client(serve_app.app) as c:
        await c.post("/api/save", json=body_complete)
        r = await c.post("/api/save", json=body_explicit_draft)
    assert r.status_code == 200
    assert r.json()["status"] == "draft"
    on_disk = json.loads((tmp_path / "data" / "verifier" / "revert.corrections.json").read_text())
    assert on_disk["status"] == "draft"


async def test_save_rejects_invalid_status(serve_app, tmp_path: Path) -> None:
    """Unknown status values are rejected — no silent fallback."""
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "bad-status",
                "status": "in-progress",  # not a valid value
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 400
    assert "status" in r.json()["detail"]


# -- /api/bundles ----------------------------------------------------------


def _write_bundle(verifier_dir: Path, stem: str, page_date_raw: str | None) -> None:
    """Drop a minimal bundle.json under the verifier directory for the
    /api/bundles enumeration tests."""
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / f"{stem}.bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stem": stem,
                "image_path": f"../tests/golden/{stem}.png",
                "pdf_path": None,
                "page_number": None,
                "model_version": "test",
                "extracted_at": "2026-05-12T00:00:00Z",
                "page_date_raw": page_date_raw,
                "comments_raw": None,
                "oddities": [],
                "quadrants": [],
            }
        )
    )


async def test_list_bundles_empty_when_no_dir(serve_app, tmp_path: Path) -> None:
    """No data/verifier/ directory → empty bundle list, not a 500."""
    async with await _client(serve_app.app) as c:
        r = await c.get("/api/bundles")
    assert r.status_code == 200
    assert r.json() == {"bundles": []}


async def test_list_bundles_classifies_three_states(serve_app, tmp_path: Path) -> None:
    """Three bundles → three states: incomplete (no corrections file),
    partial (corrections with status=draft), complete (corrections with
    status=complete). Sorted alphabetically by stem."""
    verifier_dir = tmp_path / "data" / "verifier"
    _write_bundle(verifier_dir, "a-untouched", "A")
    _write_bundle(verifier_dir, "b-draft", "B")
    _write_bundle(verifier_dir, "c-complete", "C")
    (verifier_dir / "b-draft.corrections.json").write_text(json.dumps({"status": "draft"}))
    (verifier_dir / "c-complete.corrections.json").write_text(json.dumps({"status": "complete"}))

    async with await _client(serve_app.app) as c:
        r = await c.get("/api/bundles")
    bundles = r.json()["bundles"]
    assert [b["stem"] for b in bundles] == ["a-untouched", "b-draft", "c-complete"]
    assert [b["status"] for b in bundles] == ["incomplete", "partial", "complete"]
    assert [b["page_date_raw"] for b in bundles] == ["A", "B", "C"]
    assert bundles[0]["url"] == "/verifier/?bundle=/data/verifier/a-untouched.bundle.json"


async def test_list_bundles_legacy_corrections_without_status_is_partial(
    serve_app, tmp_path: Path
) -> None:
    """A corrections.json from before status tracking landed (no `status`
    field) is classified as `partial` — they were saved, just not done."""
    verifier_dir = tmp_path / "data" / "verifier"
    _write_bundle(verifier_dir, "legacy", None)
    (verifier_dir / "legacy.corrections.json").write_text(json.dumps({"row_corrections": []}))
    async with await _client(serve_app.app) as c:
        r = await c.get("/api/bundles")
    assert r.json()["bundles"][0]["status"] == "partial"


async def test_list_bundles_surfaces_verified_at_timestamp(serve_app, tmp_path: Path) -> None:
    """`verified_at` reflects when the last Save / Mark-complete fired.
    Sourced from the verified.json mtime so the same /api/save flow keeps
    it accurate."""
    verifier_dir = tmp_path / "data" / "verifier"
    _write_bundle(verifier_dir, "stamped", None)
    (verifier_dir / "stamped.corrections.json").write_text(json.dumps({"status": "draft"}))
    (verifier_dir / "stamped.verified.json").write_text("{}")

    async with await _client(serve_app.app) as c:
        r = await c.get("/api/bundles")
    bundle = r.json()["bundles"][0]
    assert bundle["verified_at"] is not None
    # ISO format with timezone.
    assert "T" in bundle["verified_at"]


async def test_save_then_bundles_reflects_status_round_trip(serve_app, tmp_path: Path) -> None:
    """End-to-end: save with status=complete, then /api/bundles classifies
    that bundle as complete; save another with default status, listed as
    partial; un-saved bundle stays incomplete. Closes the integration gap
    between the save path and the listing path (they read/write the same
    corrections.json from different code paths)."""
    verifier_dir = tmp_path / "data" / "verifier"
    _write_bundle(verifier_dir, "a-incomplete", "A")
    _write_bundle(verifier_dir, "b-partial", "B")
    _write_bundle(verifier_dir, "c-complete", "C")

    async with await _client(serve_app.app) as c:
        # Save as draft.
        r = await c.post(
            "/api/save",
            json={
                "stem": "b-partial",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
        assert r.json()["status"] == "draft"
        # Save as complete.
        r = await c.post(
            "/api/save",
            json={
                "stem": "c-complete",
                "status": "complete",
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
        assert r.json()["status"] == "complete"
        # Now ask /api/bundles to classify all three.
        r = await c.get("/api/bundles")
    by_stem = {b["stem"]: b["status"] for b in r.json()["bundles"]}
    assert by_stem == {
        "a-incomplete": "incomplete",
        "b-partial": "partial",
        "c-complete": "complete",
    }


async def test_list_bundles_malformed_bundle_doesnt_break_index(serve_app, tmp_path: Path) -> None:
    """If one bundle.json is corrupted, the index still lists it (so the
    user can spot the problem) but with null metadata."""
    verifier_dir = tmp_path / "data" / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "broken.bundle.json").write_text("not json {{ \\")
    _write_bundle(verifier_dir, "good", "ok")
    async with await _client(serve_app.app) as c:
        r = await c.get("/api/bundles")
    bundles = r.json()["bundles"]
    by_stem = {b["stem"]: b for b in bundles}
    assert "broken" in by_stem
    assert by_stem["broken"]["page_date_raw"] is None
    assert by_stem["broken"]["status"] == "incomplete"
    assert by_stem["good"]["page_date_raw"] == "ok"


async def test_save_skips_db_when_no_jobs_db_file(serve_app, tmp_path: Path) -> None:
    """If `data/jobs.db` doesn't exist (no pipeline has run), Save still
    succeeds — no DB integration is attempted."""
    # tmp_path/data/jobs.db is absent.
    async with await _client(serve_app.app) as c:
        r = await c.post(
            "/api/save",
            json={
                "stem": "no-db",
                "pdf_path": "1990/x.pdf",
                "page_number": 1,
                "verified": _page_result_dict(),
                "corrections": _corrections_dict(),
            },
        )
    assert r.status_code == 200
    assert r.json()["db_updated"] is False
