# Golden truth fixtures

This directory holds rendered flowsheet page images plus hand-transcribed "truth" JSON files used to spot-check Gemini extraction quality. The comparison uses **subset semantics** — your truth file does not need to be exhaustive. List only what you can read with high confidence; partial truths are still useful regression bait.

## File layout

For each page you want to validate, two files share a stem:

- `tests/golden/<stem>.png` — the rendered page image (300 DPI is the production default).
- `tests/golden/<stem>.truth.json` — the hand-transcribed truth.

The stem typically encodes the source PDF and page, e.g. `1990-01jan0106-page05`.

### Calibration-derived truth sub-layout

Truth files produced by the multi-reviewer calibration flow live under a nested path:

- `tests/golden/calibration/<year>/<stem>.truth.json`

These are emitted by `scripts/derive_truth.py --from canonical` from a settled `data/calibration/<year>/<bucket>/<stem>/canonical.json`. The provenance is the sibling `agreement.json` under `data/calibration/`; the truth file itself carries no reviewer metadata. Do NOT hand-edit these files — they will be silently overwritten by the next run of `derive_truth --from canonical`. To correct a calibration-derived truth, fix the reviewer submissions upstream (which is what the multi-reviewer protocol is *for*). Flat truth files (`tests/golden/*.truth.json`) remain unchanged and are the correct home for hand-transcribed regressions.

`core.golden.discover_truths(golden_dir)` uses `rglob` so both layouts are found by the same call — a single `pytest` invocation exercises both.

## Truth file shape

```json
{
  "page_date_substrings": ["Mon", "Jan", "90"],
  "quadrants": [
    {
      "position": "top_left",
      "hour_raw": "6AM",
      "jock_substring": "ALECIA",
      "rows": [
        {"raw_substring": "LED ZEP"},
        {"raw_substring": "TRAMPLED"}
      ]
    }
  ]
}
```

Field semantics:
- `page_date_substrings`: each must appear (case-insensitive) somewhere in the model's `page_date_raw`.
- `quadrants[*].position`: one of `top_left`, `top_right`, `bottom_left`, `bottom_right`. Omitted positions are not checked.
- `quadrants[*].hour_raw`: case-insensitive substring match on the model's `hour_raw`. Omit if you can't read it.
- `quadrants[*].jock_substring`: case-insensitive substring match on the model's `jock_raw`. Omit if you can't read it.
- `quadrants[*].rows[*].raw_substring`: case-insensitive substring that must appear in some entry's `raw_text` in that quadrant.

## Why subset semantics

Handwriting is hard. A "the model returned exactly these 17 entries" expectation is brittle and forces you to transcribe everything. With substrings:
- Add only the entries you're sure about.
- The model can transcribe extra rows you couldn't read — that's a win, not a failure.
- A future scanner upgrade or prompt tweak that legitimately reads more is visible as `matched_rows` going up.

## Adding a new golden page

1. Render the page from a real PDF in `scans/`:
   ```bash
   pdftoppm -r 300 -f <PAGE> -l <PAGE> -png \
     "scans/1990/January 1990/1990-01jan0106.pdf" \
     "tests/golden/1990-01jan0106-page<PAGE>"
   ```
   `pdftoppm` will write `1990-01jan0106-page05-05.png` (or similar). Rename to drop the trailing `-05` so the stem matches the truth file.

2. Open the PNG in any viewer and transcribe what you can read confidently into a sibling `.truth.json` file (use this directory's existing example as a template).

3. Run validation against your local Gemini key:
   ```bash
   GEMINI_API_KEY=... pytest tests/golden/ -m external_api
   ```
   (The external-API test runner is a follow-up; for now use `core.golden.compare` directly from a script or REPL.)

## Cost note

The full golden suite hits the real Gemini API. Each page is ~$0.005–0.02 depending on `media_resolution`. Keep this directory small: 5–10 pages is plenty to detect prompt regressions; bigger sets belong in a one-off calibration run, not in CI.
