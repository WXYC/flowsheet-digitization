# CLAUDE.md — flowsheet-digitization

## What this repo does

Pipeline that turns 1990–2001 WXYC handwritten flowsheet PDFs (in `scans/`) into structured JSON via Gemini 3 vision. Phase 1 captures the four-quadrant per-page layout (date, hour, jock) and per-row "Artist – Track" text. Phase 2 adds the left-margin type column (`Entry.type_raw`) and is rolling out continuation/double-height/crossed-out rows, the comments field, and reconciliation against the WXYC library.

See `README.md` for user-facing setup. See `PLAN.md` for the full design rationale and what's deliberately out of scope.

## Architecture (one-liner per module)

```
scans/                           input PDFs (gitignored; SCANS_ROOT)
data/                            outputs (gitignored; DATA_ROOT)
  pages/<rel-pdf>/page-NN.png    rendered images
  results/<rel-pdf>/page-NN.json extraction results (one PageResult per page)
  jobs.db                        SQLite job table

core/
  schema.py                      Pydantic models (PageResult/Quadrant/Entry).
                                 Single source of truth — same class is both
                                 the Gemini response_schema and the on-disk shape.
  prompts.py                     Extraction prompt (separate module so prompt
                                 changes are reviewable independently).
  gemini.py                      Async wrapper around google-genai. Dependency
                                 injects the SDK Client for testability.
  render.py                      pdfimages/pdfinfo wrappers. Extracts the
                                 embedded CCITT-G4 page image directly (no
                                 rasterization). Idempotent: skip if file
                                 exists.
  jobs.py                        aiosqlite-backed state machine:
                                 pending → rendered → processing → completed
                                                    ↘ failed → (retry) ↗
  pipeline.py                    Orchestration: discover_pdfs / render_pending /
                                 process_pending. No state of its own.
  golden.py                      Subset-comparison logic for hand-transcribed
                                 truth files in tests/golden/.
  page_layout.py                 Projection-profile detector for the printed
                                 grid lines on a flowsheet page; returns
                                 PageLayout (header_bottom_y, body_mid_y,
                                 column_mid_x). Used by the per-quadrant
                                 cropper in scripts/calibrate_models.py.

cli.py                           Typer entrypoint: `flowsheets <subcommand>`.
                                 Builds dependencies from env, calls into core.
```

## Why these choices

- **Pydantic models double as the response schema.** `response_schema=PageResult` means there's exactly one definition. Drift between "what Gemini returns" and "what we validate" is structurally impossible.
- **Job table tracks one row per `(pdf_path, page_number)`.** Resumable pipelines need a state record per unit-of-work. SQLite is plenty for ~16K pages.
- **`completed` never auto-retries.** Successful Gemini extractions are not cheap. Per the org data-safety rule, they are protected; explicit `retry-page` is the only path back into the pipeline.
- **Subset semantics for golden truth.** Handwriting is hard. Forcing exhaustive transcription would mean very few golden pages. Substring-match means partial truths are still useful.
- **CLI commands are thin wrappers over async pipeline functions.** Each subcommand assembles dependencies (env → store, env → Gemini client, paths) then awaits a `core.pipeline` function. This keeps tests focused: pipeline tests inject mocks; CLI tests mock the pipeline.

## Phase 1 vs phase 2

| | Phase 1 (this PR) | Phase 2 (later) |
|---|---|---|
| Per-row Artist/Track text | ✓ | refined |
| Four quadrants (date, hour, jock) | ✓ | refined |
| Confidence per row | ✓ | re-OCR queue for low-confidence |
| Special-case `notes` (continuation/double_height/crossed_out/illegible) | captured verbatim, parsed in phase 2 | parsed, structured |
| Left-margin type column (H/M/L/Std/O/R) | captured verbatim into `Entry.type_raw` (doodle-tolerant) | normalized + reconciled against rotation lists |
| Comments field | ignored | captured |
| Date normalization to ISO | raw only | reconciled with filename's year/range |
| Reconciliation against `@wxyc/shared` canonical artists | — | fuzzy-match + auto-correct |
| Bulk full-corpus run | not in this PR | calibrate first, then schedule |

## Marker scheme

Mirroring `request-o-matic` and `library-metadata-lookup`:

| Marker | Purpose | Default CI? |
|---|---|---|
| _(no marker)_ | Pure unit test or integration test with mocked external deps. Runs in CI on every PR. | yes |
| `external_api` | Hits the real Gemini API. Excluded from default CI to control cost; opt-in via `pytest -m external_api`. | no |
| `slow` | Tests that take more than a few seconds (e.g. wall-clock pdf rendering of large fixtures). | no |
| `calibration_dump` | Tests that read saved dumps under `/tmp/modal-dump/` and assume those dumps are current. Run manually after a fresh `scripts/calibrate_models.py --dump-dir ...`; opt-in via `pytest -m calibration_dump`. | no |

The default `pytest` run uses `addopts = "-m 'not external_api and not slow and not calibration_dump'"` (see `pyproject.toml`). A separate scheduled workflow can run `pytest -m external_api` against `tests/golden/` to detect prompt-quality regressions.

## Workflow conventions

- **TDD by default.** Failing test first, run to see red, implement, run to see green, refactor. Especially important for new modules — the discipline catches "the test doesn't actually test what it claims" bugs.
- **Don't introduce abstractions speculatively.** No `from_env` classmethods, no `@property` accessors, no `Protocol` types unless a current call site needs them. Add them when a second caller appears.
- **Single SQLite file, short-lived connections.** `JobStore` opens a connection per public method. Fine for ETL volume; if we ever need contention-aware operations, switch to a long-lived connection with explicit transactions.
- **No emojis in commit messages or output.** Not even console output.

## Adding a new pipeline step

1. Define the contract: what does it take, what does it return?
2. Write the failing test in `tests/unit/test_<step>.py` (or `tests/integration/` if it crosses module boundaries). Run it. Confirm it fails for the reason you expect.
3. Implement the step in `core/<step>.py`. Run the test. Iterate to green.
4. Refactor: dead code, redundant exception clauses, names that don't read well, tests that overspecify.
5. Wire it into `core/pipeline.py` and add an integration test there if it changes orchestration.
6. If user-visible, add a CLI subcommand in `cli.py`.

## Useful commands

```bash
.venv/bin/pytest                                      # default run (no external_api/slow)
.venv/bin/pytest tests/unit/test_jobs.py -v          # one file, verbose
.venv/bin/pytest -m external_api                      # opt in to real API tests
.venv/bin/ruff check .                                # lint
.venv/bin/ruff format --check .                       # format check
.venv/bin/mypy core cli.py                            # type check
.venv/bin/flowsheets status                           # see job counts
```
