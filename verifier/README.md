# Flowsheet verifier UI

A static, dependency-free single-page app for manually verifying flowsheet extraction output. Each row's cropped image strip is shown next to the model-detected text in an editable field. Hand-correct typos, mark hallucinated rows, add missed rows, then export a `verified.json` that flows back into the pipeline as ground truth.

> **Local-dev tool.** `verifier/serve.py` binds to `127.0.0.1` and has no authentication or CSRF protection on `/api/save` or `/api/bundles`. Do not expose it on a non-loopback interface; in particular, do not run it from a tmux/ssh session forwarded to a shared host without adding auth in front. The bundle-stem path-traversal guard is the only sanitization on writes.

## Run

The verifier ships with a tiny FastAPI server that does two things:

1. Serves `verifier/`, `data/`, and `tests/` as static files.
2. Proxies the **Check artists** lookups through `/api/lookup` to the request-o-matic `/request` endpoint (request-o-matic doesn't emit CORS headers, so a same-origin proxy is simpler than configuring CORS).

```bash
# from the repo root
.venv/bin/python verifier/serve.py
# default port is 8765; override with VERIFIER_PORT=9000 .venv/bin/python verifier/serve.py

# then open the index:
open "http://localhost:8765/verifier/"
```

The index page lists every bundle in `data/verifier/` with its verification state and an **Open next page that needs work** button. Click a row to open it. The status badge on each row mirrors the same state machine as the in-edit pill: `incomplete` (no save yet), `partial` (saved as draft), `complete` (marked complete).

The `?bundle=<path>` URL is still the way to deep-link a specific page (e.g., bookmarks, share links). Edit-mode navigation also exposes Prev / Next buttons and the keyboard shortcuts (`?` to see all).

You can also load a bundle via the **Load image** file picker if the page is served statically and the relative image path can't be fetched.

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
  "schema_version": 2,
  "pdf_path": "1990/April 1990/1990-04apr0106.pdf",
  "page_number": 25,
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

`schema_version` is currently `2`. v1 was the initial bundle shape; v2 added the optional `pdf_path` and `page_number` fields so `Save` can target the corresponding `jobs.db` row. Future incompatible changes bump the version; the UI shows an error banner if it sees an unsupported version. Keep `schema_version` set when archiving bundles so older bundles remain loadable.

## Saving

Two buttons share the right side of the header: **Save** and **Mark complete**. Both POST to `/api/save`. The only difference is the `status` field in the body — `Save` omits it (treated as `draft`), `Mark complete` sends `"complete"`.

Status semantics:

- **Incomplete** — no `<stem>.corrections.json` on disk. The bundle has never been saved.
- **Partial** — `corrections.json` exists with `"status": "draft"`. The user is in progress.
- **Complete** — `corrections.json` has `"status": "complete"`. The user explicitly marked the page done.

The server runs a small preservation rule so a plain Save on an already-complete page **does not** downgrade it. The user can refine details on a complete page without re-marking it. (If we ever need a "Revert to draft" affordance, that's a separate ticket.)

Save's three side effects:

1. Writes `data/verifier/<stem>.verified.json` — `PageResult`-shaped JSON validating against `core.schema.PageResult`. Bundle-only fields are stripped before validation. Rows marked ✗ are excluded; rows added via **+ add row** are included.
2. Writes `data/verifier/<stem>.corrections.json` — the delta between the loaded bundle and the verified state, plus a top-level `"status"` field.
3. If the bundle has a non-null `pdf_path` + `page_number`, updates the matching `jobs.db` row via `JobStore.mark_verified`.

The status bar reports the destination files, status, and whether `jobs.db` was updated:

> Saved as complete · data/verifier/X.verified.json + data/verifier/X.corrections.json · 4 field correction(s), 0 added, 0 deleted · jobs.db updated.

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
  "added_rows": [
    {"position": "top_left", "row_index": 12, "raw_text": "...",
     "type_raw": null, "notes": null}
  ],
  "deleted_rows": [
    {"position": "top_left", "row_index": 7, "original_raw_text": "..."}
  ]
}
```

The `verified.json` is the consumable artifact (plugs back into the pipeline as ground truth). The `corrections.json` is the audit record (preserves the original model output for diff analysis). Rows the user neither edited nor marked ✗ produce no entry in either file — by clicking Save, the user is implicitly endorsing every untouched row.

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

## Keyboard shortcuts

Press `?` anywhere in the editor to see the overlay. The current set:

| Key | Action |
|---|---|
| ⌘S / Ctrl+S | Save (draft) |
| ⌘⇧S / Ctrl+Shift+S | Mark complete |
| j / ⌘↓ | Focus next row's `raw_text` |
| k / ⌘↑ | Focus previous row |
| ⌘D / Ctrl+D | Toggle ✗ (delete) on focused row |
| n | Next bundle |
| p | Previous bundle |
| ? | Toggle shortcut overlay |
| Esc | Close overlay |

The single-letter keys (`j`, `k`, `n`, `p`, `?`) are ignored when the keyboard focus is in an `<input>`, `<textarea>`, or `<select>` — so typing in a row's text field works normally.

## Known rough edges (v1)

- **No autosave / localStorage.** Close the tab and unsaved edits are lost. Export before navigating away.
- **No batch loader.** One bundle at a time.
- **No keyboard shortcuts.** Mouse-driven only.
- **Confidence is not editable.** That field is a model artifact, not user truth.
- **Row crops use detected grid lines when available, even spacing otherwise.** A quadrant where the model over-emitted rows (more entries than handwritten lines) will show vertically squashed crops — visible but possibly mis-cropped at boundaries. Eye your way through it.
