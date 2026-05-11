# Flowsheet verifier UI

A static, dependency-free single-page app for manually verifying flowsheet extraction output. Each row's cropped image strip is shown next to the model-detected text in an editable field. Hand-correct typos, mark hallucinated rows, add missed rows, then export a `verified.json` that flows back into the pipeline as ground truth.

## Run

The UI is static HTML + JS + CSS. It needs a local HTTP server so the browser can fetch the bundle JSON and the page image relative to it.

```bash
# from the repo root
python -m http.server 8765

# then open in a browser:
open "http://localhost:8765/verifier/?bundle=/data/verifier/<stem>.bundle.json"
```

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

## Exports

Clicking **Export verified + corrections** downloads two files in sequence:

1. `<stem>.verified.json` — `PageResult`-shaped JSON validating against `core.schema.PageResult`. Bundle-only fields (`schema_version`, `stem`, `image_path`, per-entry `row_bbox`) are stripped. Rows marked ✗ are excluded. Rows added via **+ add row** are included.
2. `<stem>.corrections.json` — the delta between the loaded bundle and the verified export, plus the set of rows the user reviewed:

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

## Known rough edges (v1)

- **No autosave / localStorage.** Close the tab and unsaved edits are lost. Export before navigating away.
- **No batch loader.** One bundle at a time.
- **No keyboard shortcuts.** Mouse-driven only.
- **Confidence is not editable.** That field is a model artifact, not user truth.
- **Row crops use detected grid lines when available, even spacing otherwise.** A quadrant where the model over-emitted rows (more entries than handwritten lines) will show vertically squashed crops — visible but possibly mis-cropped at boundaries. Eye your way through it.
