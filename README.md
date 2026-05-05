# flowsheet-digitization

OCR pipeline that turns 1990–2001 WXYC handwritten flowsheet PDFs into structured JSON using Gemini 3 vision.

## What it does

For each page of every PDF under `scans/`:

1. Extracts the embedded page image directly with `pdfimages -png`. The WXYC PDFs each wrap a single CCITT Group 4 (lossless 1-bit grayscale) image at 300 PPI native, so the extracted PNG is bit-for-bit identical to the source bitmap — no rasterization, no anti-aliasing, no DPI choice.
2. Sends the image to Gemini 3 with a Pydantic `response_schema` that defines the four-quadrant flowsheet layout.
3. Stores a JSON result file with the per-row `raw_text`, `artist_guess`, `track_guess`, `confidence`, and any phase-2 `notes` (continuation, double-height, crossed-out, illegible).
4. Tracks every page in a SQLite job table so reruns are idempotent and partial failures resume.

Phase 1 captures only the per-row "Artist – Track" text. The left-margin H/M/L/Std/O/R type column, multi-row continuations, double-height handwriting, the comments field, and reconciliation against the WXYC library DB are all phase 2 — see `PLAN.md`.

## Quickstart

```bash
# 1. Install dependencies (Python 3.12+, poppler for pdftoppm)
brew install poppler   # macOS; on Ubuntu: sudo apt-get install poppler-utils
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env: at minimum, set GEMINI_API_KEY (https://aistudio.google.com/apikey)
# Default SCANS_ROOT is ./scans. Point it elsewhere if your scans live on
# an external drive, a synced Dropbox/iCloud folder, or a future cloud mount.

# 3. Run the pipeline
.venv/bin/flowsheets discover                       # register one job per PDF page
.venv/bin/flowsheets render --limit 200             # extract embedded PNGs, parallel by default (RENDER_CONCURRENCY)
.venv/bin/flowsheets render --concurrency 8 --limit 1000   # override per-run
.venv/bin/flowsheets process --limit 50             # PNG -> Gemini -> JSON
.venv/bin/flowsheets status                         # show counts by status
```

Each step is resumable: rerun any subcommand and it picks up where it left off. Successful work is never repeated unless you explicitly call `retry-page`.

## Configuration

All settings live in `.env` (see `.env.example` for the canonical list):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | _(required)_ | API key from <https://aistudio.google.com/apikey>. |
| `SCANS_ROOT` | `./scans` | Where the PDFs live. Point at an external drive or synced folder if needed. |
| `DATA_ROOT` | `./data` | Where rendered PNGs, JSON results, and `jobs.db` are written. Gitignored. |
| `GEMINI_MODEL` | `gemini-3.1-pro-preview` | The original `gemini-3-pro-preview` was shut down March 2026; we use 3.1 Pro Preview. Bump as Google releases stable Gemini 3.x. |
| `GEMINI_MEDIA_RESOLUTION` | `high` | Vision token allocation: `low` / `medium` / `high` (1120 tokens) / `ultra_high`. `high` is the default for fine handwriting. |
| `RENDER_CONCURRENCY` | `4` | Parallel pdftoppm workers for `flowsheets render`. Override per-run with `--concurrency`. |
| `PROCESS_CONCURRENCY` | `4` | Reserved for future parallelism of the process step (currently sequential). |
| `MAX_ATTEMPTS` | `3` | Failed pages retry up to this many times across runs before sticking in `failed`. |

## Job lifecycle and data safety

```
pending  ──render──▶ rendered ──process──▶ completed
   │                    │                      ▲
   │                    │                      │  (no auto retry; only via retry-page)
   │                    │                      │
   └────────────────────┴───── failed ─────────┘
                                ▲
                                │ retried up to MAX_ATTEMPTS
```

`completed` is terminal and **never** reprocessed automatically. Successful Gemini extractions are not free — protecting them from accidental overwrite is a deliberate design choice. To re-extract a single page that you know was wrong:

```bash
flowsheets retry-page "1990/January 1990/1990-01jan0106.pdf" 5
```

Never use it for bulk resets. If you need to redo many pages, edit the job row in `data/jobs.db` directly with a targeted `WHERE` clause and confirm with a `SELECT` first.

## Development

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy core cli.py
.venv/bin/pytest
```

Tests are split into:

- `tests/unit/` — schema, render, jobs, prompts, gemini client (mocked SDK), CLI (mocked pipeline), golden comparison.
- `tests/integration/` — end-to-end pipeline orchestration with mocked Gemini and real `pdftoppm`.
- `tests/golden/` — rendered page images plus hand-transcribed truth JSON, used to spot-check extraction quality with the real API. See `tests/golden/README.md`.

The default test run **excludes** the `external_api` and `slow` markers; CI runs the same default. The golden-page external-API runner is a follow-up.

## Cost calibration

Gemini 3.1 Pro charges per input token; one 300-DPI flowsheet page at `media_resolution=high` is ~1120 image tokens plus ~600 prompt tokens. Across the full corpus (~16K pages) input cost lands in the low tens of dollars; output adds modestly. Run the pipeline against a 10–20 page sample first and inspect both quality and `usage_metadata` before scheduling a full run.

## Repo conventions

- Python 3.12, src-flat layout (`core/` + top-level `cli.py`), Pydantic v2.
- ruff (pinned) + mypy (pinned) so a tooling release with new rules cannot silently fail CI on an unrelated PR.
- pytest markers: `external_api` (real Gemini) and `slow` — both excluded from default runs. See `pyproject.toml`.
- See `CLAUDE.md` for in-repo architecture notes and the phase 1 / phase 2 split.
