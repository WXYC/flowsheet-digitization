# Flowsheet Digitization — First-Pass Plan

## Goal

Digitize WXYC's 1990–2001 handwritten flowsheet PDFs (in `scans/`) into structured JSON using Gemini 3 Pro vision. First pass focuses **only** on the four-quadrant per-page structure (Date, Hour, Jock) and the per-row `Artist - Track` text. Special cases (continuations, double-height, crossed-out, comments field, and the playbox/hit-type column) are captured verbatim into a `notes` field and deferred to phase 2.

## Out of scope (phase 2)

- Reconciling artists/tracks against the WXYC library DB or `wxycCanonicalArtistNames`.
- Parsing the left-margin type column (H/M/L/Std/O/R/R⇒).
- Multi-row/continuation/double-height handling.
- The `Comments:` free-text field at the bottom of each page.
- Date normalization to ISO; we keep the raw string and let downstream consumers reconcile against the filename's year/month/day-range.
- Backfilling earlier or later years if any get added later.

## Architecture

Standalone Python repo, conventions parity with `library-metadata-lookup` and `request-o-matic`:

- Python 3.12+, `pyproject.toml`, top-level packages (no `src/` layout), `uv` managed.
- Pydantic v2 schemas, `pydantic-settings` for env config, `python-dotenv` for `.env`.
- ruff (pinned, e.g. `ruff==0.15.6`) + ruff format + mypy (pinned, e.g. `mypy==1.19.1`). Pinning matches `library-metadata-lookup` so a ruff/mypy release with new rules can't silently fail CI on an unrelated PR.
- pytest + pytest-asyncio + pytest-cov.
- aiosqlite for the job table (async parity with the rest of the org).
- `google-genai` (the unified Google GenAI SDK) for Gemini calls. Verified API:
  - Model: `gemini-3.1-pro-preview` (the earlier `gemini-3-pro-preview` was shut down March 2026; we'll use 3.1 Pro Preview and bump as Google releases stable Gemini 3.x).
  - Vision input: `types.Part.from_bytes(data=image_bytes, mime_type="image/png")`.
  - Structured output: `config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=PageResult)` where `PageResult` is our Pydantic class. Returns `response.parsed` as a `PageResult` instance.
  - `media_resolution=MEDIA_RESOLUTION_HIGH` (1120 tokens/image) for fine handwriting; calibrate against ULTRA_HIGH on golden pages.
- `pdf2image` (or direct `pdftoppm` subprocess) for PDF → PNG rendering.

### Package layout

```
flowsheet-digitization/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
├── scans/                       # already exists — input PDFs (gitignored if huge)
├── data/                        # gitignored — rendered PNGs, SQLite DB, output JSON
│   ├── pages/<year>/<pdf_name>/page-NN.png
│   ├── results/<year>/<pdf_name>/page-NN.json
│   └── jobs.db
├── core/
│   ├── __init__.py
│   ├── config.py                # Settings (API key, paths, concurrency)
│   ├── schema.py                # Pydantic models (PageResult, Quadrant, Entry)
│   ├── render.py                # PDF → PNG via pdftoppm
│   ├── gemini.py                # Gemini client + prompt + response parsing
│   ├── prompts.py               # Prompt template (separate file so it's reviewable)
│   ├── jobs.py                  # SQLite job table + state transitions
│   └── pipeline.py              # Orchestration: discover → render → process → store
├── cli.py                       # Typer entrypoint
├── tests/
│   ├── unit/
│   │   ├── test_schema.py
│   │   ├── test_render.py
│   │   └── test_jobs.py
│   ├── integration/
│   │   └── test_pipeline_with_mocked_gemini.py
│   └── golden/
│       ├── 1990-01jan0106-page01.png        # checked in
│       ├── 1990-01jan0106-page01.expected.json   # hand-transcribed truth
│       └── ... (2-3 more pages with diverse handwriting)
└── scripts/
    └── transcribe_golden.md     # instructions for adding new golden pages
```

## Pydantic schema (the contract)

```python
class Entry(BaseModel):
    row_index: int                       # 0-based within the quadrant
    raw_text: str                        # verbatim line, no normalization
    artist_guess: str | None
    track_guess: str | None
    confidence: Literal["high", "medium", "low"]
    notes: str | None = None             # "continuation", "crossed_out", "double_height", "illegible", etc.

class Quadrant(BaseModel):
    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"]
    hour_raw: str | None
    jock_raw: str | None
    entries: list[Entry]

class PageResult(BaseModel):
    page_date_raw: str | None
    quadrants: list[Quadrant]            # always length 4, ordered TL/TR/BL/BR
    model_version: str                   # which Gemini model produced this
    extracted_at: datetime
```

The Gemini call requests this same shape via `response_schema`. The same Pydantic class validates the response — single source of truth.

## Job state machine (SQLite)

One row per `(pdf_relative_path, page_number)`.

| Column | Purpose |
|---|---|
| `pdf_path` | relative to repo root, e.g. `scans/1990/January 1990/1990-01jan0106.pdf` |
| `page_number` | 1-based |
| `status` | `pending` / `rendering` / `rendered` / `processing` / `completed` / `failed` / `low_confidence` |
| `attempts` | int |
| `last_error` | nullable text |
| `image_path` | nullable, set after render |
| `result_path` | nullable, set after gemini |
| `model_version` | nullable |
| `created_at`, `updated_at` | timestamps |

State rules (data-safety):
- `completed` is **never** retried automatically. Surgical `--retry-page` flag for explicit re-runs.
- `low_confidence` is its own terminal state; eligible for re-OCR queue in phase 2.
- `failed` is automatically retried up to `MAX_ATTEMPTS` (default 3), with exponential backoff between runs.
- Renders are idempotent — if `image_path` exists on disk and matches expected name, skip rendering.

## CLI

Typer-based, single entrypoint `flowsheets`:

```
flowsheets discover [--root scans/]                  # populate jobs table from PDFs on disk
flowsheets render [--pdf <path>] [--workers N]       # render pending → rendered
flowsheets process [--limit N] [--workers N]         # rendered → completed via Gemini
flowsheets status                                    # counts by status
flowsheets retry-page <pdf_path> <page_number>       # explicit retry of a completed page
flowsheets validate-golden                           # run against tests/golden/, print accuracy
```

Each subcommand is also exposed as a Python function for tests.

## Prompt design

Stored as a separate file (`core/prompts.py`) so it can be reviewed and versioned independently.

Key principles:
1. **Faithful raw, separate guesses.** The model fills `raw_text` with what it sees (verbatim, no expansion of abbreviations) and puts inferred parsing in `artist_guess` / `track_guess`. If a row is unreadable, `raw_text` is best effort and `confidence` is `low`.
2. **Quadrant order is fixed.** TL/TR/BL/BR — Gemini is told to return exactly four quadrants in that order regardless of which the DJ filled first.
3. **No hallucination guard.** "If you cannot read a row, say `confidence: low` and put a partial transcription. Do not invent."
4. **Notes for the special cases we're skipping.** "If a line wraps onto a second physical row, mark `notes: continuation`. If a line is in double-height handwriting, mark `notes: double_height`. If it's crossed out, `notes: crossed_out`."

Prompt is short. Include one in-context example (the hand-transcribed JSON for a golden page would be too long; instead, a small synthetic example).

## Test strategy

Following the WXYC pytest-marker scheme (architecture A from the wiki):

- **Unit** (`tests/unit/`, no markers): schema validation, render-helper logic, jobs state transitions. No network, no real Gemini, no real PDFs (use a tiny generated 1-page PDF fixture).
- **Integration** (`tests/integration/`, marker TBD): pipeline orchestration with `gemini.py` mocked at the SDK boundary. Verifies job-state transitions, file outputs, idempotency.
- **Golden** (`tests/golden/`, marker `external_api` since real Gemini): runs on demand (not in CI) — `pytest -m external_api`. Computes per-field accuracy against hand-transcribed truth. CI runs this on a schedule, not per-PR, to avoid burning API budget on every push.

Per the org rule: TDD by default. Schema → write failing test for parse → implement parse → green. Same for render, jobs, pipeline.

## CI

- Lint (`ruff check`, `ruff format --check`).
- Typecheck (`mypy`).
- Unit + integration tests (no API key needed; Gemini is mocked).
- No golden tests in default CI (cost). Separate workflow for that.

## Deliverables for this PR

1. Repo skeleton (`pyproject.toml`, ruff/mypy config, `.gitignore`, CI workflow).
2. `core/schema.py` with Pydantic models + tests.
3. `core/render.py` (PDF→PNG via pdftoppm) + tests.
4. `core/jobs.py` (SQLite state machine) + tests.
5. `core/prompts.py` + `core/gemini.py` (real Gemini 3 call, but **gated by env var**; mocked in tests). Verify model id / SDK shape against live docs before writing this module.
6. `core/pipeline.py` orchestration + integration test.
7. `cli.py` with the subcommands above.
8. 2 golden pages hand-transcribed and checked in (one easy, one hard-handwriting).
9. README with quickstart + cost calibration note.
10. CLAUDE.md with the architecture, conventions, "phase 1 vs phase 2" scope split, and an explicit **Marker scheme** subsection (mirroring `request-o-matic`'s CLAUDE.md) listing the markers this repo uses (`external_api`, `slow`) and how CI selects them.

## Non-deliverables (deliberately)

- No Docker. (Add later if we deploy this somewhere; for a one-shot ETL it's overkill.)
- No PostHog / Sentry instrumentation in this PR. Add in phase 2 if we run at scale.
- No reconciliation pass against `@wxyc/shared` canonical artists. Phase 2.
- No re-OCR queue / row-cropping. Phase 2.
- No actual full-corpus run. This PR delivers the pipeline; running it is a follow-up.

## Calibration before scale-up

After the pipeline works end-to-end on the 2 golden pages, run it on ~20 random pages spanning multiple years and DJs. Inspect output by hand. Decide:
- Whether per-page confidence is good enough.
- Whether the prompt needs adjustment.
- Real cost per page (token usage logged).
- Whether to proceed to a full-corpus run.

Only then does a full run get scheduled.

## Open questions for review

1. **Gemini model id.** Should be the latest Gemini 3 Pro vision-capable model. I'll verify against current docs before writing `gemini.py`. Worth thinking-mode or not? (Thinking would be more accurate but more expensive on ~16K pages.)
2. **Image resolution.** I rendered samples at 300 DPI and they were readable. Lower (200 DPI) saves tokens; higher (400 DPI) might help on the worst handwriting. Will calibrate on golden pages.
3. **`scans/` in git?** They're ~3 MB per PDF × ~60 PDFs ≈ 180 MB. Decision for this PR: gitignore them, expose `SCANS_ROOT` env var (default `./scans`) so the user can point the pipeline at scans wherever they live (local FS, an external drive, a synced Dropbox/iCloud folder, or a future B2/S3 mount). README documents the option explicitly. Cloud-mount/B2/S3 wiring is phase 2 if needed.
4. **Async or sync?** Org default is async. PDF rendering is CPU-bound (good for `asyncio.to_thread`); Gemini calls are I/O-bound (good for native async). I'll go async for consistency with sibling repos, even though for this workload sync would also be fine.
