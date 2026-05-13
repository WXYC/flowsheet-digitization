"""Dev server for the verifier UI.

Serves the repo's static files (verifier/, data/, tests/) and provides:

  POST /api/lookup    — same-origin proxy to request-o-matic /request
                        (request-o-matic doesn't emit CORS headers, and
                        a proxy is simpler than configuring CORS on a
                        third-party service).
  POST /api/save      — persist a verifier UI session: writes
                        <stem>.verified.json and <stem>.corrections.json
                        into data/verifier/, and (when the bundle carries
                        a `pdf_path`/`page_number` pair) updates
                        `jobs.db` via `JobStore.mark_verified`. Honors
                        an optional `status` field with preservation
                        semantics — see the handler docstring.
  GET  /api/bundles   — list every bundle in data/verifier/ with its
                        verification state (incomplete / partial /
                        complete) for the index page and Prev/Next nav.

Run:

    .venv/bin/python verifier/serve.py

Then open http://localhost:8765/verifier/?bundle=/data/verifier/<stem>.bundle.json
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.jobs import JobStore
from core.schema import PageResult

REQUEST_O_MATIC_URL = os.environ.get(
    "REQUEST_O_MATIC_URL",
    "https://request-o-matic-production.up.railway.app/api/v1/request",
)
PORT = int(os.environ.get("VERIFIER_PORT", "8765"))
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))
VERIFIER_DIR = DATA_ROOT / "verifier"
JOBS_DB_PATH = DATA_ROOT / "jobs.db"

app = FastAPI(docs_url=None, redoc_url=None)

# Cache of jobs.db paths whose `init()` migrations have already been
# applied this process. Avoids re-running PRAGMAs + ALTER-TABLE checks
# on every /api/save when jobs.db is unchanged across saves. Tests that
# swap DATA_ROOT via importlib.reload get a fresh empty set.
_initialized_jobs_dbs: set[Path] = set()


async def _open_jobs_store() -> JobStore | None:
    """Return an initialized `JobStore` if `jobs.db` is on disk, else None.

    Runs `JobStore.init()` once per (db_path, process) pair — subsequent
    calls hit the in-memory cache and skip the schema-migration round
    trip. Re-checks `is_file()` every call so a DB created after the
    server starts (e.g., user runs the pipeline mid-session) is picked
    up without a server restart.
    """
    if not JOBS_DB_PATH.is_file():
        return None
    store = JobStore(JOBS_DB_PATH)
    if JOBS_DB_PATH not in _initialized_jobs_dbs:
        await store.init()
        _initialized_jobs_dbs.add(JOBS_DB_PATH)
    return store


def _safe_stem(stem: str) -> str:
    """Reject anything that could escape `data/verifier/` via path traversal.

    Bundle stems come from image filenames (e.g. `1990-04apr0106-page25`)
    and are unlikely to contain `/`, but a hostile or malformed POST
    shouldn't let the verifier server write outside the verifier dir.
    Whitespace-only stems are also refused — they'd produce files named
    ` .verified.json` which are confusing and almost certainly a bug.
    """
    if not stem or not stem.strip() or "/" in stem or "\\" in stem or stem.startswith(".."):
        raise HTTPException(status_code=400, detail=f"invalid stem: {stem!r}")
    return stem


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via a `.tmp` sibling and `os.replace`.

    Two writes on the save path (verified + corrections) — atomic
    individual writes mean a partially-failed save leaves either both
    files at their pre-save state OR both at the new state, never a
    half-updated state where verified.json reflects the edit but
    corrections.json doesn't.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


@app.post("/api/lookup")
async def lookup(request: Request) -> JSONResponse:
    """Same-origin proxy to request-o-matic's /request endpoint.

    Accepts the same shape request-o-matic does (`{"message": "..."}`) and
    forwards verbatim. Adds skip_slack=True so the lookup doesn't post to
    Slack. Returns request-o-matic's response unchanged.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    message = (payload or {}).get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="missing 'message' field")

    forward_body = {"message": message, "skip_slack": True}
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(REQUEST_O_MATIC_URL, json=forward_body)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail=f"upstream timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(r.json())


_VALID_STATUSES = frozenset({"draft", "complete"})


def _resolve_status(incoming: str | None, corrections_path: Path) -> str:
    """Compute the corrections.json `status` field for this save.

    Rules (matching the UI's Save + toggleable Mark-complete design):
      - Any explicit valid status from the client (`"complete"` or
        `"draft"`) wins outright. `"draft"` means the user clicked the
        toggle on an already-complete page to revert it.
      - When the client omits the status field (plain Save), preserve
        an existing `"complete"` on disk — a refine-in-place edit, not
        a status change.
      - Otherwise the page is a draft.

    Returns one of `"draft"` or `"complete"`.
    """
    if incoming in _VALID_STATUSES:
        return incoming
    if corrections_path.is_file():
        try:
            existing = json.loads(corrections_path.read_text())
        except Exception:  # noqa: BLE001
            existing = {}
        if existing.get("status") == "complete":
            return "complete"
    return "draft"


@app.post("/api/save")
async def save(request: Request) -> JSONResponse:
    """Persist a verifier UI session to disk and (optionally) `jobs.db`.

    Expected body shape:

        {
          "stem": "<page stem>",
          "pdf_path": "<rel path to PDF>" | null,
          "page_number": <int> | null,
          "status": "draft" | "complete" | null,
          "verified": { ...PageResult... },
          "corrections": { ...corrections... }
        }

    Writes:
      - `data/verifier/<stem>.verified.json`  (validated as PageResult)
      - `data/verifier/<stem>.corrections.json` (verbatim JSON +
        server-resolved `status` field)

    Status resolution: `"complete"` wins; otherwise an existing
    on-disk `"complete"` is preserved across plain Saves. See
    `_resolve_status` for the table.

    If `pdf_path` and `page_number` are present, also calls
    `JobStore.mark_verified` to record the verification in `jobs.db`.
    For bundles without a job key (test fixtures), only the files are
    written and `db_updated` is False in the response.
    """
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {exc}") from exc

    stem = _safe_stem(str(payload.get("stem", "")))
    verified = payload.get("verified")
    corrections = payload.get("corrections")
    pdf_path = payload.get("pdf_path")
    page_number = payload.get("page_number")
    incoming_status = payload.get("status")
    if verified is None or corrections is None:
        raise HTTPException(
            status_code=400,
            detail="body must include `verified` and `corrections` objects",
        )
    if incoming_status is not None and incoming_status not in _VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"status must be one of {sorted(_VALID_STATUSES)} or omitted, "
                f"got {incoming_status!r}"
            ),
        )

    # Validate verified against PageResult and keep the parsed model so
    # the on-disk JSON is the Pydantic-normalized round-trip rather than
    # whatever the client happened to send. This makes the verified file
    # a canonical representation that bit-matches what the pipeline
    # writes, regardless of any extra fields or non-canonical datetime
    # formats the client may have included.
    try:
        validated = PageResult.model_validate(verified)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"verified payload not a valid PageResult: {exc}"
        ) from exc

    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    verified_path = VERIFIER_DIR / f"{stem}.verified.json"
    corrections_path = VERIFIER_DIR / f"{stem}.corrections.json"
    # Resolve status BEFORE we overwrite the existing corrections file,
    # so the preservation rule can see the prior on-disk value.
    resolved_status = _resolve_status(incoming_status, corrections_path)
    corrections_with_status = {"status": resolved_status, **(corrections or {})}
    _atomic_write_text(verified_path, validated.model_dump_json(indent=2))
    _atomic_write_text(corrections_path, json.dumps(corrections_with_status, indent=2))

    db_updated = False
    # `isinstance(x, int)` is True for `bool` in Python — explicitly reject
    # so a malformed `page_number: true` doesn't coerce to 1 and lookup
    # the wrong row.
    if pdf_path and isinstance(page_number, int) and not isinstance(page_number, bool):
        store = await _open_jobs_store()
        if store is not None:
            db_updated = await store.mark_verified(
                pdf_path=pdf_path,
                page_number=page_number,
                verified_path=verified_path,
                corrections_path=corrections_path,
            )

    # Report paths relative to DATA_ROOT.parent so the UI displays
    # `data/verifier/<stem>.verified.json` whether `data/` lives under
    # the repo root (production) or a tmp dir (tests). Always succeeds —
    # both written paths are under DATA_ROOT, which is a child of
    # DATA_ROOT.parent by construction.
    return JSONResponse(
        {
            "verified_path": str(verified_path.relative_to(DATA_ROOT.parent)),
            "corrections_path": str(corrections_path.relative_to(DATA_ROOT.parent)),
            "db_updated": db_updated,
            "status": resolved_status,
        }
    )


@app.get("/api/bundles")
async def list_bundles() -> JSONResponse:
    """Enumerate every bundle in data/verifier/ with its verification state.

    The state machine: `incomplete` (no corrections.json yet), `partial`
    (corrections.json with `status="draft"`), `complete` (corrections.json
    with `status="complete"`). Legacy corrections.json files without a
    `status` field default to `partial` — they were saved before status
    tracking landed, so they're at least in-progress.

    Returns:
        {
          "bundles": [
            {
              "stem": "...",
              "page_date_raw": "..." | null,
              "pdf_path": "..." | null,
              "page_number": int | null,
              "status": "incomplete" | "partial" | "complete",
              "verified_at": "<ISO timestamp>" | null,
              "url": "/verifier/?bundle=/data/verifier/<stem>.bundle.json"
            },
            ...
          ]
        }

    Sorted by stem so navigation is deterministic and matches alphanumeric
    page order (which the WXYC stem format respects: `page05 < page25`).
    """
    if not VERIFIER_DIR.is_dir():
        return JSONResponse({"bundles": []})

    bundles_out: list[dict[str, object]] = []
    for bundle_path in sorted(VERIFIER_DIR.glob("*.bundle.json")):
        stem = bundle_path.name.removesuffix(".bundle.json")
        try:
            bundle = json.loads(bundle_path.read_text())
        except Exception:  # noqa: BLE001
            # A malformed bundle file shouldn't break the whole index —
            # surface it as incomplete with null metadata so it still
            # appears and the user can see something's off.
            bundle = {}

        corrections_path = VERIFIER_DIR / f"{stem}.corrections.json"
        verified_path = VERIFIER_DIR / f"{stem}.verified.json"
        status, verified_at = _bundle_state(corrections_path, verified_path)

        bundles_out.append(
            {
                "stem": stem,
                "page_date_raw": bundle.get("page_date_raw"),
                "pdf_path": bundle.get("pdf_path"),
                "page_number": bundle.get("page_number"),
                "status": status,
                "verified_at": verified_at,
                "url": f"/verifier/?bundle=/data/verifier/{stem}.bundle.json",
            }
        )

    return JSONResponse({"bundles": bundles_out})


def _bundle_state(corrections_path: Path, verified_path: Path) -> tuple[str, str | None]:
    """Derive (status, verified_at) for one bundle from disk.

    `verified_at` is the mtime of `<stem>.verified.json` if it exists;
    None otherwise. Reflects when the last Save / Mark-complete happened.
    """
    if not corrections_path.is_file():
        return ("incomplete", None)
    try:
        corrections = json.loads(corrections_path.read_text())
    except Exception:  # noqa: BLE001
        corrections = {}
    raw_status = corrections.get("status")
    if raw_status == "complete":
        status = "complete"
    else:
        # Anything else (`draft`, missing, malformed) is `partial`. Legacy
        # corrections.json files written before status tracking landed end
        # up here, which is the right semantic — they're saved, not done.
        status = "partial"

    verified_at: str | None = None
    if verified_path.is_file():
        verified_at = datetime.fromtimestamp(verified_path.stat().st_mtime, tz=UTC).isoformat()
    return (status, verified_at)


# Static mounts. Each top-level dir we need to serve gets its own mount so
# the URL structure mirrors the repo layout — `image_path` in bundles is
# relative (`../pages/.../page-NN.png`), and the UI fetches relative to
# the bundle URL. `/data` honors the same DATA_ROOT override that writes
# use so the read and write sides stay in sync when DATA_ROOT is moved.
app.mount("/verifier", StaticFiles(directory=REPO_ROOT / "verifier", html=True), name="verifier")
app.mount("/data", StaticFiles(directory=DATA_ROOT, check_dir=False), name="data")
app.mount("/tests", StaticFiles(directory=REPO_ROOT / "tests"), name="tests")


def main() -> None:
    # Pass the app object directly rather than an import string — the script
    # is invoked as a file (verifier/serve.py), not as a package import.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
