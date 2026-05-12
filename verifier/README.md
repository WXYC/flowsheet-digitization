# Flowsheet verifier UI

A static, dependency-free single-page app for manually verifying flowsheet extraction output. Each row's cropped image strip is shown next to the model-detected text in an editable field. Hand-correct typos, mark hallucinated rows, add missed rows, then export a `verified.json` that flows back into the pipeline as ground truth.

## Run

The verifier ships with a tiny FastAPI server that does two things:

1. Serves `verifier/`, `data/`, and `tests/` as static files.
2. Proxies the **Check artists** lookups through `/api/lookup` to the request-o-matic `/request` endpoint (request-o-matic doesn't emit CORS headers, so a same-origin proxy is simpler than configuring CORS).

```bash
# from the repo root
.venv/bin/python verifier/serve.py
# default port is 8765; override with VERIFIER_PORT=9000 .venv/bin/python verifier/serve.py

# then open in a browser:
open "http://localhost:8765/verifier/?bundle=/data/verifier/<stem>.bundle.json"
```

If you want only the static side and don't need the artist-lookup button, `python -m http.server 8765` from the repo root still works — the Check-artists button will return 404s but everything else functions.

The `?bundle=...` URL param is the recommended path: the UI fetches the bundle, then resolves the bundle's `image_path` (relative path inside the JSON) and fetches the image too.

You can also load a bundle via the **Load bundle** file picker, in which case a second **Load image** picker appears. This path works without a server but you must pick both files manually.

## File layout

The bundle's `image_path` is **relative to the bundle file's directory**. The expected layout under the repo's `data/` directory:

```
data/
  pages/<rel-pdf>/<stem>.png            # source image
  results/<rel-pdf>/<stem>.json         # pipeline output (input to make_verifier_bundle)
  verifier/<stem>.bundle.json           # pre-processor output, references ../pages/<rel-pdf>/<stem>.png
  verifier/<stem>.verified.json         # UI export (download to this directory by convention)
tests/golden/<stem>.truth.json          # derive_truth output (optional destination)
```

## End-to-end workflow

1. Run the pipeline to produce `data/results/<rel-pdf>/<stem>.json`.
2. Generate a bundle:

   ```bash
   python -m scripts.make_verifier_bundle \
     data/results/<rel-pdf>/<stem>.json \
     data/pages/<rel-pdf>/<stem>.png \
     --out data/verifier/<stem>.bundle.json
   ```

3. Open the verifier and load the bundle.
4. Walk the page: each row shows a cropped image strip + the model's `raw_text`. Correct typos, set `type` and `notes` when needed, click ✗ to mark hallucinations, click **+ add row** to insert a row the model missed.
5. Edit the page-level fields: `page_date_raw`, `comments_raw`, `oddities`.
6. Click **Export verified** → downloads `<stem>.verified.json`. Move it to `data/verifier/`.
7. (Optional) Derive a `tests/golden/*.truth.json`:

   ```bash
   python -m scripts.derive_truth \
     data/verifier/<stem>.verified.json \
     --out tests/golden/<stem>.truth.json
   ```

## Bundle schema

```json
{
  "schema_version": 1,
  "stem": "<page stem>",
  "image_path": "<relative path to the page image>",
  "model_version": "<extraction model>",
  "extracted_at": "<ISO timestamp>",
  "page_date_raw": "...",
  "comments_raw": "...",
  "oddities": ["..."],
  "quadrants": [
    {
      "position": "top_left",
      "bbox": [x1, y1, x2, y2],
      "hour_raw": "6AM",
      "jock_raw": "Andrew",
      "entries": [
        {
          "row_index": 0,
          "raw_text": "...",
          "confidence": "high",
          "type_raw": "M",
          "notes": null,
          "oddities": [],
          "row_bbox": [x1, y1, x2, y2]
        }
      ],
      "oddities": []
    }
  ]
}
```

### Versioning

`schema_version` is currently `1`. Future incompatible changes bump the version; the UI shows an error banner if it sees an unsupported version. Keep `schema_version` set when archiving bundles so older bundles remain loadable.

## Saving

Clicking **Save** POSTs the current edit state to the server's `/api/save` endpoint, which:

1. Writes `data/verifier/<stem>.verified.json` — `PageResult`-shaped JSON validating against `core.schema.PageResult`. Bundle-only fields (`schema_version`, `stem`, `image_path`, `pdf_path`, `page_number`, per-entry `row_bbox`) are stripped before validation. Rows marked ✗ are excluded. Rows added via **+ add row** are included.
2. Writes `data/verifier/<stem>.corrections.json` — the delta between the loaded bundle and the verified state (shape below).
3. If the bundle has a non-null `pdf_path` + `page_number` (production-pipeline pages do; test fixtures don't), updates the matching `jobs.db` row via `JobStore.mark_verified` — setting `verified_at`, `verified_path`, and `corrections_path`.

The status bar reports the destination files and whether `jobs.db` was updated:

> Saved data/verifier/X.verified.json + data/verifier/X.corrections.json · 4 field correction(s), 0 added, 0 deleted · jobs.db updated.

If you'd rather have a downloadable file, open the saved JSON from `data/verifier/` directly.

The `corrections.json` shape:

```json
{
  "stem": "...",
  "model_version": "...",
  "extracted_at": "...",
  "exported_at": "...",
  "page_corrections": [
    {"field": "page_date_raw", "original": "...", "corrected": "..."}
  ],
  "quadrant_corrections": [
    {"position": "top_left", "field": "hour_raw", "original": "6AM", "corrected": "6PM"}
  ],
  "row_corrections": [
    {"position": "top_left", "row_index": 0, "field": "raw_text",
     "original": "Smiths-I wnat", "corrected": "Smiths-I want the one I can't have"}
  ],
  "verified_rows": [
    {"position": "top_left", "row_index": 0}
  ],
  "added_rows": [
    {"position": "top_left", "row_index": 12, "raw_text": "...",
     "type_raw": null, "notes": null}
  ],
  "deleted_rows": [
    {"position": "top_left", "row_index": 7, "original_raw_text": "..."}
  ]
}
```

The verified.json is the consumable artifact (plugs back into the pipeline as ground truth). The corrections.json is the audit record (preserves the original model output for analysis, separates "user reviewed and accepted" from "user reviewed and corrected" from "user never touched").

**Verification semantics**:
- A row's `verified` checkbox is **off** by default.
- Editing any field on a row (raw_text, type, notes) **auto-sets** `verified` to on.
- Toggling ✗ (delete) auto-sets `verified` to on (a deliberate action).
- The user can manually flip the checkbox to mark an unchanged row as reviewed.
- Rows added via **+ add row** are implicitly verified (they were typed by the user).

Truth derivation is a **separate Python tool** (`scripts/derive_truth.py`) rather than a UI button — the substring-extraction rules live in one place (Python, testable), not duplicated in JS.

## Check artists (request-o-matic lookup)

Click **Check artists** in the header to look up every row's text via the WXYC library + Discogs reconciliation pipeline. Each row gets a badge with the resolved artist + matched **release** (album / 12") and a confidence score.

**Important contrast**: the flowsheet records `Artist - Track`, but the library and Discogs match at the **release** level. The badge text is labeled `artist · album: "..."` (full track match) or `artist · sample release: "..."` (artist-only fallback) so this never looks like a near-track-match when it's a release-level result.

Badge states:

- **Green** — track found in the library on this release. High confidence the artist is right; the release shown is the album/single containing the played track.
- **Yellow** — one of:
  - **`⚠ artist-only · ...`**: the library has the artist but not this specific track. The "sample release" is whichever album of theirs the library indexed first — it's *not* a confirmation that the played track lives there.
  - **`⚠ postdates · ...`**: the matched release's year is after the flowsheet's page year. The page year is parsed from `page_date_raw` (1990 for `Thurs 4/5/90`, etc.); when `release_year > page_year` the match is almost certainly a later remix, reissue, or same-name band.
  - artwork confidence below 0.5.
- **Grey/italic** — no library match found. Could be a typo, a non-canonical name, or genuinely missing from the WXYC corpus (the library reflects current stock, not 1990 stock — ~30% of mid-density pages will have these).
- **Faded/italic** — stale. You edited the row after running Check; re-run to refresh.

The lookup goes through request-o-matic's LLM-driven request parser (artist normalization, fuzzy matching) before hitting the LML library search. The badge reflects request-o-matic's `library_results` and `artwork` fields — not LML's `/api/v1/lookup` directly, since the LLM correction layer is the load-bearing piece.

## Known rough edges (v1)

- **No autosave / localStorage.** Close the tab and unsaved edits are lost. Export before navigating away.
- **No batch loader.** One bundle at a time.
- **No keyboard shortcuts.** Mouse-driven only.
- **Confidence is not editable.** That field is a model artifact, not user truth.
- **Row crops use detected grid lines when available, even spacing otherwise.** A quadrant where the model over-emitted rows (more entries than handwritten lines) will show vertically squashed crops — visible but possibly mis-cropped at boundaries. Eye your way through it.
