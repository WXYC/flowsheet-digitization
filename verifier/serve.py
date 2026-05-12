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
                        `jobs.db` via `JobStore.mark_verified`.

Run:

    .venv/bin/python verifier/serve.py

Then open http://localhost:8765/verifier/?bundle=/data/verifier/<stem>.bundle.json
"""

from __future__ import annotations

import json
import os
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


def _safe_stem(stem: str) -> str:
    """Reject anything that could escape `data/verifier/` via path traversal.

    Bundle stems come from image filenames (e.g. `1990-04apr0106-page25`)
    and are unlikely to contain `/`, but a hostile or malformed POST
    shouldn't let the verifier server write outside the verifier dir.
    """
    if not stem or "/" in stem or "\\" in stem or stem.startswith(".."):
        raise HTTPException(status_code=400, detail=f"invalid stem: {stem!r}")
    return stem


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


@app.post("/api/save")
async def save(request: Request) -> JSONResponse:
    """Persist a verifier UI session to disk and (optionally) `jobs.db`.

    Expected body shape:

        {
          "stem": "<page stem>",
          "pdf_path": "<rel path to PDF>" | null,
          "page_number": <int> | null,
          "verified": { ...PageResult... },
          "corrections": { ...corrections... }
        }

    Writes:
      - `data/verifier/<stem>.verified.json`  (validated as PageResult)
      - `data/verifier/<stem>.corrections.json` (verbatim JSON)

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
    if verified is None or corrections is None:
        raise HTTPException(
            status_code=400,
            detail="body must include `verified` and `corrections` objects",
        )

    # Validate verified against PageResult so a bad client doesn't write
    # garbage that the pipeline can't load later.
    try:
        PageResult.model_validate(verified)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"verified payload not a valid PageResult: {exc}"
        ) from exc

    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    verified_path = VERIFIER_DIR / f"{stem}.verified.json"
    corrections_path = VERIFIER_DIR / f"{stem}.corrections.json"
    verified_path.write_text(json.dumps(verified, indent=2))
    corrections_path.write_text(json.dumps(corrections, indent=2))

    db_updated = False
    if pdf_path and isinstance(page_number, int):
        # Best-effort DB update. JobStore.init() is idempotent and applies
        # late-column migrations, so first-save against a pre-verification
        # jobs.db still works.
        if JOBS_DB_PATH.is_file():
            store = JobStore(JOBS_DB_PATH)
            await store.init()
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
        }
    )


# Static mounts. Each top-level dir we need to serve gets its own mount so
# the URL structure mirrors the repo layout — `image_path` in bundles is
# relative (`../pages/.../page-NN.png`), and the UI fetches relative to
# the bundle URL.
app.mount("/verifier", StaticFiles(directory=REPO_ROOT / "verifier", html=True), name="verifier")
app.mount("/data", StaticFiles(directory=REPO_ROOT / "data"), name="data")
app.mount("/tests", StaticFiles(directory=REPO_ROOT / "tests"), name="tests")


def main() -> None:
    # Pass the app object directly rather than an import string — the script
    # is invoked as a file (verifier/serve.py), not as a package import.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
