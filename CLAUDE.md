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
  verifier/<stem>.bundle.json    pre-processor output: result + per-row bboxes
  verifier/<stem>.verified.json  verifier UI export: hand-corrected PageResult
  jobs.db                        SQLite job table

verifier/                        static SPA for manual row-by-row verification.
                                 Loads a bundle, renders each row's cropped
                                 image strip next to an editable text field,
                                 exports a corrected verified.json.

core/
  schema.py                      Pydantic models. GeminiPageResult is what
                                 the model returns (used as response_schema);
                                 PageResult adds caller-set model_version +
                                 extracted_at and is what lands on disk. Entry
                                 and Quadrant are shared between both.
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
                                 `partition_row_lines_by_quadrant(image,
                                 layout)` is the public hook the verifier
                                 pre-processor uses to compute per-row bboxes.
  continuations.py               Read-time merge of `notes="continuation"`
                                 rows into the prior entry's raw_text.
                                 Pure function; on-disk shape unchanged.

cli.py                           Typer entrypoint: `flowsheets <subcommand>`.
                                 Builds dependencies from env, calls into core.

scripts/
  make_verifier_bundle.py        PageResult JSON + page PNG -> verifier
                                 bundle.json with per-quadrant + per-row
                                 bboxes for the SPA to canvas-crop. Hard-codes
                                 SCHEMA_VERSION = 1; bump on incompatible
                                 schema changes.
  derive_truth.py                <stem>.verified.json -> <stem>.truth.json
                                 by extracting short uppercased substrings
                                 (page date tokens, jock prefix, artist
                                 portion of raw_text). Single source of
                                 truth for those rules — the UI doesn't
                                 derive truth itself.
```

## Why these choices

- **Pydantic models drive the response schema, with a deliberate split at the page level.** `response_schema=GeminiPageResult` covers everything the model fills (`page_date_raw`, `quadrants`, `oddities`); `PageResult` extends it with `model_version` and `extracted_at`, which the pipeline sets server-side. The split keeps the "one schema, one validator" property for the parts the model actually produces while preventing it from hallucinating caller-owned metadata.
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
| Special-case `notes` (continuation/double_height/crossed_out/illegible) | continuation merged at read-time via `core.continuations.merge_continuations` (on-disk JSON keeps the raw tag); double_height/crossed_out/illegible captured verbatim | double_height/crossed_out/illegible structured + filtered |
| Left-margin type column (H/M/L/Std/O/R) | captured verbatim into `Entry.type_raw` (doodle-tolerant) | normalized + reconciled against rotation lists |
| Comments field | captured verbatim into `GeminiPageResult.comments_raw` (null if blank/unreadable) | normalized / dedup-checked against entry text |
| Date normalization to ISO | raw only | reconciled with filename's year/range |
| Reconciliation against `@wxyc/shared` canonical artists | — | fuzzy-match + auto-correct |
| Bulk full-corpus run | not in this PR | calibrate first, then schedule |

## Research adapters

Production extraction goes through Gemini in `core/gemini.py`. The non-Gemini adapters in `scripts/calibrate_models.py` (`churro`, `qwen-vl`, `modal-churro`, `modal-qwen-vl`, `modal-qwen-vl-quad`, `local-quadrant-smoke`) are research-only: they exist as calibration regression alarms so a Gemini-side regression has something to compare against, and as a record of what we've tried. None of them are production candidates.

Quality on the 5-golden set (matched rows out of 76): Gemini 3 Pro 68/76; modal-qwen-vl-quad 50/76; modal-qwen-vl 50/76 (with grammar-constrained decoding, 24/76 before); modal-churro 36/76. The best Modal adapter sits ~24% behind Gemini on quality while costing ~6× more per page (`modal-qwen-vl-quad` ≈ $0.06-0.12/page). Any dollar figure quoted in `scripts/calibrate_models.py` for a Modal adapter at corpus scale (e.g. "~$1200-1800") is exploration spend if we calibrated it on the full corpus — it is not a planned spend.

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
