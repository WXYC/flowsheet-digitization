// Flowsheet verifier — vanilla JS, no build step.
//
// Loads a bundle.json (produced by scripts/make_verifier_bundle.py) plus
// the page image it references, renders per-row canvas crops next to
// editable text fields, and exports:
//   1. <stem>.verified.json — PageResult-shaped corrected page
//   2. <stem>.corrections.json — delta vs the original bundle, plus the
//      set of rows the user marked verified
//
// Two load paths are supported:
//   1. Server-served bundle: fetch(bundle) then fetch(image) by relative
//      URL. Used when the page is opened via `python -m http.server`.
//   2. File-picker bundle: read the bundle as text, then prompt for the
//      image file separately.
//
// State is split:
//   state.originalBundle  — immutable snapshot of the loaded bundle. Never
//                           mutated; used as the diff baseline on export.
//   state.bundle          — working copy. Mutated by edits and UI flags
//                           (`_verified`, `_added`, `_deleted`).

"use strict";

const SUPPORTED_SCHEMA_VERSION = 2;

const state = {
  bundle: null,            // mutable working copy
  originalBundle: null,    // immutable snapshot for diffing
  pageImage: null,         // HTMLImageElement
  lookupConcurrency: 4,    // parallel /api/lookup requests
};

const $ = (sel, root = document) => root.querySelector(sel);

function setStatus(msg, kind = "info") {
  const el = $("#status");
  el.textContent = msg;
  el.className = kind === "error" ? "error" : "";
}

function cloneDeep(obj) {
  return JSON.parse(JSON.stringify(obj));
}

// ---- bundle loading ------------------------------------------------------

async function loadBundleFromUrlParam() {
  const params = new URLSearchParams(location.search);
  const path = params.get("bundle");
  if (!path) return false;
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
    const bundle = await r.json();
    await initBundle(bundle, { bundleUrl: path });
    return true;
  } catch (err) {
    setStatus(`Failed to load bundle: ${err.message}`, "error");
    return false;
  }
}

async function loadBundleFromFile(file) {
  try {
    const text = await file.text();
    const bundle = JSON.parse(text);
    await initBundle(bundle, { bundleUrl: null });
  } catch (err) {
    setStatus(`Failed to parse bundle: ${err.message}`, "error");
  }
}

async function initBundle(bundle, { bundleUrl }) {
  if (bundle.schema_version !== SUPPORTED_SCHEMA_VERSION) {
    setStatus(
      `Unsupported schema_version ${bundle.schema_version}; ` +
      `this UI supports v${SUPPORTED_SCHEMA_VERSION}.`,
      "error"
    );
    return;
  }
  state.originalBundle = cloneDeep(bundle);
  state.bundle = cloneDeep(bundle);
  state.pageImage = null;

  if (bundleUrl) {
    const imageUrl = new URL(bundle.image_path, new URL(bundleUrl, location.href));
    state.pageImage = await loadImage(imageUrl.href);
    finishInit();
  } else {
    $("#image-picker").hidden = false;
    setStatus("Bundle loaded. Pick the page image to continue.");
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load image ${src}`));
    img.src = src;
  });
}

async function loadImageFromFile(file) {
  const url = URL.createObjectURL(file);
  try {
    state.pageImage = await loadImage(url);
    finishInit();
  } catch (err) {
    setStatus(`Failed to load image: ${err.message}`, "error");
  }
}

function finishInit() {
  setStatus(
    `Loaded ${state.bundle.stem} ` +
    `(${state.pageImage.naturalWidth}×${state.pageImage.naturalHeight}px).`
  );
  $("#app").hidden = false;
  $("#save-verified").disabled = false;
  $("#toggle-page-view").disabled = false;
  $("#check-artists").disabled = false;
  $("#page-view-img").src = state.pageImage.src;
  renderPageMeta();
  renderQuadrants();
  // Show the full-page reference by default; verifiers asked for this
  // because the row crops need page context to be useful.
  togglePageView();
}

// ---- render: page meta ---------------------------------------------------

function renderPageMeta() {
  const dateEl = $("#page-date-raw");
  dateEl.value = state.bundle.page_date_raw ?? "";
  dateEl.addEventListener("input", () => {
    state.bundle.page_date_raw = dateEl.value || null;
  });

  const commentsEl = $("#comments-raw");
  commentsEl.value = state.bundle.comments_raw ?? "";
  commentsEl.addEventListener("input", () => {
    state.bundle.comments_raw = commentsEl.value || null;
  });

  const oddEl = $("#oddities");
  oddEl.value = (state.bundle.oddities ?? []).join("\n");
  oddEl.addEventListener("input", () => {
    state.bundle.oddities = oddEl.value
      .split("\n")
      .map(s => s.trim())
      .filter(Boolean);
  });
}

// ---- render: quadrants ---------------------------------------------------

function renderQuadrants() {
  const container = $("#quadrants-container");
  container.innerHTML = "";
  const tmpl = $("#quadrant-template");

  for (const quad of state.bundle.quadrants) {
    const node = tmpl.content.firstElementChild.cloneNode(true);
    $(".quadrant-title", node).textContent = quad.position;

    const hourEl = $(".hour-raw", node);
    hourEl.value = quad.hour_raw ?? "";
    hourEl.addEventListener("input", () => {
      quad.hour_raw = hourEl.value || null;
    });

    const jockEl = $(".jock-raw", node);
    jockEl.value = quad.jock_raw ?? "";
    jockEl.addEventListener("input", () => {
      quad.jock_raw = jockEl.value || null;
    });

    const rowsEl = $(".rows", node);
    for (const entry of quad.entries) {
      rowsEl.appendChild(buildRow(entry, quad));
    }

    $(".add-row", node).addEventListener("click", () => {
      const newEntry = {
        row_index: quad.entries.length,
        raw_text: "",
        confidence: "low",
        type_raw: null,
        notes: null,
        oddities: [],
        row_bbox: null,
        _added: true,
      };
      quad.entries.push(newEntry);
      rowsEl.appendChild(buildRow(newEntry, quad));
    });

    container.appendChild(node);
  }
}

function buildRow(entry, quad) {
  const tmpl = $("#row-template");
  const node = tmpl.content.firstElementChild.cloneNode(true);
  node.dataset.rowIndex = String(entry.row_index);

  const canvas = $(".row-crop", node);
  if (entry.row_bbox) {
    drawCrop(canvas, entry.row_bbox);
  } else {
    canvas.outerHTML = `<div class="row-crop no-crop">no crop (added row)</div>`;
  }

  const textEl = $(".raw-text", node);
  textEl.value = entry.raw_text;
  textEl.addEventListener("input", () => {
    entry.raw_text = textEl.value;
    // Edit invalidates the lookup badge — show as stale until re-check.
    const badge = $(".lookup-badge", node);
    if (badge && !badge.hidden) {
      badge.classList.add("stale");
      badge.title = "Click 'Check artists' to refresh.";
    }
  });

  const typeEl = $(".type-raw input", node);
  typeEl.value = entry.type_raw ?? "";
  typeEl.addEventListener("input", () => {
    entry.type_raw = typeEl.value || null;
  });

  const notesEl = $(".notes select", node);
  const syncNotesView = () => {
    notesEl.value = entry.notes ?? "";
    node.classList.toggle("has-notes", !!entry.notes);
  };
  syncNotesView();
  notesEl.addEventListener("change", () => {
    entry.notes = notesEl.value || null;
    syncNotesView();
  });

  $(".delete-row", node).addEventListener("click", () => {
    entry._deleted = !entry._deleted;
    node.classList.toggle("deleted", entry._deleted);
  });

  return node;
}

function drawCrop(canvas, bbox) {
  const [x1, y1, x2, y2] = bbox;
  const srcW = x2 - x1;
  const srcH = y2 - y1;
  if (srcW <= 0 || srcH <= 0) {
    canvas.outerHTML = `<div class="row-crop no-crop">empty bbox</div>`;
    return;
  }
  canvas.width = srcW;
  canvas.height = srcH;
  // Let CSS govern display size — the canvas's intrinsic aspect ratio is
  // preserved by `width: 100%; height: auto` in styles.css. This makes the
  // crop fill the available column width (full row when the side panel is
  // closed; narrower when the page-view panel pushes the editor).
  const ctx = canvas.getContext("2d");
  ctx.drawImage(state.pageImage, x1, y1, srcW, srcH, 0, 0, srcW, srcH);
}

// ---- export: PageResult verified.json -----------------------------------

function buildVerifiedExport() {
  // Strip bundle-only fields, per-entry row_bbox, and UI flags. Validates
  // as PageResult directly.
  return {
    page_date_raw: state.bundle.page_date_raw,
    quadrants: state.bundle.quadrants.map(quad => ({
      position: quad.position,
      hour_raw: quad.hour_raw,
      jock_raw: quad.jock_raw,
      entries: quad.entries
        .filter(e => !e._deleted)
        .map(e => ({
          row_index: e.row_index,
          raw_text: e.raw_text,
          type_raw: e.type_raw,
          confidence: e.confidence,
          notes: e.notes,
          oddities: e.oddities ?? [],
        })),
      oddities: quad.oddities ?? [],
    })),
    comments_raw: state.bundle.comments_raw,
    oddities: state.bundle.oddities ?? [],
    model_version: state.bundle.model_version,
    extracted_at: state.bundle.extracted_at,
  };
}

// ---- export: corrections.json (delta) -----------------------------------

// Fields that participate in row-level correction tracking. row_bbox is
// derived geometry, not user-editable text, so it never appears as a
// correction. confidence is model output, not user truth.
const ROW_TRACKED_FIELDS = ["raw_text", "type_raw", "notes"];

// Page-level and quadrant-level fields the verifier exposes for editing.
const PAGE_TRACKED_FIELDS = ["page_date_raw", "comments_raw"];
const QUADRANT_TRACKED_FIELDS = ["hour_raw", "jock_raw"];

function arraysEqual(a, b) {
  if (a == null && b == null) return true;
  if (a == null || b == null) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function findOriginalEntry(quadPosition, rowIndex) {
  const quad = state.originalBundle.quadrants.find(q => q.position === quadPosition);
  if (!quad) return null;
  return quad.entries.find(e => e.row_index === rowIndex) ?? null;
}

function findOriginalQuadrant(position) {
  return state.originalBundle.quadrants.find(q => q.position === position) ?? null;
}

function buildCorrectionsExport() {
  const page_corrections = [];
  for (const field of PAGE_TRACKED_FIELDS) {
    const orig = state.originalBundle[field] ?? null;
    const cur = state.bundle[field] ?? null;
    if (orig !== cur) {
      page_corrections.push({ field, original: orig, corrected: cur });
    }
  }
  if (!arraysEqual(state.originalBundle.oddities ?? [], state.bundle.oddities ?? [])) {
    page_corrections.push({
      field: "oddities",
      original: state.originalBundle.oddities ?? [],
      corrected: state.bundle.oddities ?? [],
    });
  }

  const quadrant_corrections = [];
  const row_corrections = [];
  const added_rows = [];
  const deleted_rows = [];

  for (const quad of state.bundle.quadrants) {
    const origQuad = findOriginalQuadrant(quad.position);
    if (origQuad) {
      for (const field of QUADRANT_TRACKED_FIELDS) {
        const orig = origQuad[field] ?? null;
        const cur = quad[field] ?? null;
        if (orig !== cur) {
          quadrant_corrections.push({
            position: quad.position,
            field,
            original: orig,
            corrected: cur,
          });
        }
      }
    }

    for (const entry of quad.entries) {
      // Added-and-then-deleted: dropped entirely, no signal worth keeping.
      if (entry._added && entry._deleted) continue;

      if (entry._added) {
        added_rows.push({
          position: quad.position,
          row_index: entry.row_index,
          raw_text: entry.raw_text,
          type_raw: entry.type_raw,
          notes: entry.notes,
        });
        continue;
      }

      if (entry._deleted) {
        const orig = findOriginalEntry(quad.position, entry.row_index);
        deleted_rows.push({
          position: quad.position,
          row_index: entry.row_index,
          original_raw_text: orig?.raw_text ?? null,
        });
        continue;
      }

      // Existing, not deleted: emit corrections per changed field.
      const orig = findOriginalEntry(quad.position, entry.row_index);
      if (orig) {
        for (const field of ROW_TRACKED_FIELDS) {
          const origVal = orig[field] ?? null;
          const curVal = entry[field] ?? null;
          if (origVal !== curVal) {
            row_corrections.push({
              position: quad.position,
              row_index: entry.row_index,
              field,
              original: origVal,
              corrected: curVal,
            });
          }
        }
      }
    }
  }

  return {
    stem: state.bundle.stem,
    model_version: state.bundle.model_version,
    extracted_at: state.bundle.extracted_at,
    exported_at: new Date().toISOString(),
    page_corrections,
    quadrant_corrections,
    row_corrections,
    added_rows,
    deleted_rows,
  };
}

// ---- file-download helpers -----------------------------------------------

async function saveAll() {
  if (!state.bundle) return;
  const btn = $("#save-verified");
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Saving…";

  const verified = buildVerifiedExport();
  const corrections = buildCorrectionsExport();
  const body = {
    stem: state.bundle.stem,
    pdf_path: state.bundle.pdf_path ?? null,
    page_number: state.bundle.page_number ?? null,
    verified,
    corrections,
  };

  try {
    const r = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const detail = await r.text();
      throw new Error(`/api/save ${r.status}: ${detail}`);
    }
    const result = await r.json();
    const n =
      corrections.row_corrections.length +
      corrections.page_corrections.length +
      corrections.quadrant_corrections.length;
    const dbBit = result.db_updated
      ? "jobs.db updated"
      : (body.pdf_path && body.page_number != null
          ? "no matching job row (files only)"
          : "files only (no job key)");
    setStatus(
      `Saved ${result.verified_path} + ${result.corrections_path} · ` +
      `${n} field correction(s), ${corrections.added_rows.length} added, ` +
      `${corrections.deleted_rows.length} deleted · ${dbBit}.`
    );
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

// ---- artist/track lookup (request-o-matic via /api/lookup proxy) --------

async function lookupOne(message) {
  const r = await fetch("/api/lookup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!r.ok) throw new Error(`/api/lookup ${r.status}: ${await r.text()}`);
  return await r.json();
}

// Same separator regex as `core.parse.parse_artist_track` (Python side).
// Pulls the artist out of a flowsheet `Artist - Track` string for
// comparison against the library-resolved artist.
const ARTIST_TRACK_SEPARATOR = /\s*[-–—]\s*/;

// Stop words and bibliographic prefixes that the WXYC corpus often
// drops or adds inconsistently. Stripping them prevents a "the" / "a"
// difference from making "the sundays" and "Sundays" look like a
// mismatch.
const ARTIST_TOKEN_STOPWORDS = new Set([
  "the", "a", "an", "and", "&", "feat", "featuring", "ft", "with", "vs", "presents",
]);

function _tokenize(s) {
  return s
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .map((t) => t.replace(/s$/, "")) // crude singularization: boys -> boy
    .filter((t) => t.length >= 2 && !ARTIST_TOKEN_STOPWORDS.has(t));
}

function parseInputArtist(rawText) {
  if (!rawText) return null;
  const parts = rawText.split(ARTIST_TRACK_SEPARATOR);
  return parts[0]?.trim() || null;
}

// Returns true when the resolved artist shares NO tokens with the input
// artist (after normalization, stop-word and trailing-s stripping).
// Conservative: when either side has zero meaningful tokens, returns
// false (no signal). Catches "Pure Joy → Coldcut" where the LLM
// fuzzy-matched on a track word; tolerates "the sundays → Sundays" and
// "Beastie Boy → Beastie Boys".
function artistTokensDisjoint(inputArtist, resolvedArtist) {
  if (!inputArtist || !resolvedArtist) return false;
  const a = new Set(_tokenize(inputArtist));
  const b = new Set(_tokenize(resolvedArtist));
  if (a.size === 0 || b.size === 0) return false;
  for (const t of a) if (b.has(t)) return false;
  return true;
}

// Parse a 2- or 4-digit year out of `page_date_raw`. The WXYC corpus spans
// 1990-2001, so 2-digit years 90-99 map to 19xx and 00-01 map to 20xx.
// Returns null when no plausible year is present.
function parsePageYear(pageDateRaw) {
  if (!pageDateRaw) return null;
  // Try 4-digit year first.
  const fourDigit = pageDateRaw.match(/\b(19\d{2}|20\d{2})\b/);
  if (fourDigit) return Number(fourDigit[1]);
  // Fall back to a 2-digit year. Scan every 2-digit token that's NOT
  // surrounded by other digits (so we don't pluck "19" or "90" out of
  // "1990"). 2-digit years in the WXYC corpus range (80-99 → 19xx,
  // 00-09 → 20xx) win; anything else (a month like "04" or a day like
  // "31") is skipped.
  const twoDigitMatches = pageDateRaw.matchAll(/(?<!\d)(\d{2})(?!\d)/g);
  for (const m of twoDigitMatches) {
    const n = Number(m[1]);
    if (n >= 80 && n <= 99) return 1900 + n;
    if (n >= 0 && n <= 9) return 2000 + n;
  }
  return null;
}

function badgeContentFor(data, pageYear, inputRawText) {
  const parsed = data.parsed || {};
  const artwork = data.artwork || {};
  const libResults = data.library_results || [];
  const libTop = libResults[0];
  if (!libTop && !artwork.artist) {
    return { kind: "empty", text: "no library match" };
  }

  // Prefer library_results (authoritative for the WXYC corpus). Fall back
  // to Discogs artwork for orientation.
  const artist = libTop?.artist || artwork.artist || parsed.artist || "?";
  // library_results[i].title and artwork.album both denote a RELEASE
  // (album / 12") in the library — never a track. The flowsheet records
  // tracks, so we label the field explicitly to avoid the visual conflict
  // with the flowsheet's "Artist - Track" shape.
  const release = libTop?.title || artwork.album || "";
  const conf = typeof artwork.confidence === "number" ? artwork.confidence : null;
  const releaseYear = typeof artwork.release_year === "number" ? artwork.release_year : null;

  // Fallback signal: the library reconciler couldn't find the specific
  // track and is returning the artist's catalog instead. `song_not_found`
  // is the canonical flag; `search_type === "song_as_artist"` means the
  // LLM parser couldn't identify the artist and reinterpreted the parsed
  // song as the artist (e.g. "Beastie Boy" treated as artist when LLM
  // missed the plural). Both mean: artist may be right, but the release
  // shown is unrelated to whatever track the DJ actually played.
  const fallback = data.song_not_found === true || data.search_type === "song_as_artist";

  // Anachronism: matched release postdates the flowsheet.
  const postdates = pageYear != null && releaseYear != null && releaseYear > pageYear;

  // Artist mismatch: resolved artist shares zero tokens with the artist
  // we parsed out of the flowsheet text. Catches request-o-matic
  // fuzzy-matching on a track word (Pure Joy → Coldcut via "Pieces")
  // even when the release year happens to be plausible.
  const inputArtist = parseInputArtist(inputRawText);
  const artistMismatch = artistTokensDisjoint(inputArtist, artist);

  const stampBits = [];
  if (releaseYear !== null) stampBits.push(String(releaseYear));
  if (conf !== null) stampBits.push(conf.toFixed(2));
  const stamp = stampBits.length ? ` (${stampBits.join(", ")})` : "";

  let text;
  let kind;
  if (fallback) {
    // "artist-only" makes it clear we have the artist but not the track,
    // and "sample release" disclaims the album shown is illustrative
    // (whichever release of theirs the library indexed first), not a
    // confirmation that this is where the played track lives.
    text = release
      ? `⚠ artist-only · ${artist} · sample release: "${release}"${stamp}`
      : `⚠ artist-only · ${artist}${stamp}`;
    kind = "hit-weak";
  } else {
    text = release
      ? `${artist} · album: "${release}"${stamp}`
      : `${artist}${stamp}`;
    kind = conf !== null && conf < 0.5 ? "hit-weak" : "hit-strong";
  }
  if (postdates) {
    text = "⚠ postdates · " + text;
    kind = "hit-weak";
  }
  if (artistMismatch) {
    text = `⚠ different artist (got "${artist}", expected "${inputArtist}") · ${text}`;
    kind = "hit-weak";
  }
  return { kind, text };
}

function applyBadge(rowEl, kind, text, title) {
  const badge = $(".lookup-badge", rowEl);
  if (!badge) return;
  badge.hidden = false;
  badge.className = `lookup-badge ${kind}`;
  badge.textContent = text;
  if (title) badge.title = title;
}

async function checkArtists() {
  if (!state.bundle) return;
  const btn = $("#check-artists");
  btn.disabled = true;
  const originalLabel = btn.textContent;
  const pageYear = parsePageYear(state.bundle.page_date_raw);

  // Collect every non-deleted, non-empty row with its DOM node.
  const work = [];
  for (const quad of state.bundle.quadrants) {
    const quadNode = [...$$(".quadrant")].find(
      (n) => $(".quadrant-title", n).textContent === quad.position
    );
    if (!quadNode) continue;
    const rowNodes = $$(".row", quadNode);
    for (let i = 0; i < quad.entries.length; i++) {
      const entry = quad.entries[i];
      if (entry._deleted || !entry.raw_text?.trim()) continue;
      const rowEl = rowNodes[i];
      if (!rowEl) continue;
      work.push({ rowEl, entry });
      applyBadge(rowEl, "loading", "…looking up", "");
    }
  }

  let done = 0;
  const total = work.length;
  const updateBtn = () => {
    btn.textContent = `Checking artists (${done}/${total})…`;
  };
  updateBtn();

  // Concurrency-limited fan-out.
  const queue = work.slice();
  async function worker() {
    while (queue.length) {
      const job = queue.shift();
      if (!job) break;
      try {
        const data = await lookupOne(job.entry.raw_text);
        const { kind, text } = badgeContentFor(data, pageYear, job.entry.raw_text);
        const aw = data.artwork || {};
        const title =
          `parsed_artist=${(data.parsed || {}).artist ?? "?"}; ` +
          `library_results=${(data.library_results || []).length}` +
          (aw.release_year ? `; release_year=${aw.release_year}` : "") +
          (pageYear ? `; page_year=${pageYear}` : "");
        applyBadge(job.rowEl, kind, text, title);
      } catch (err) {
        applyBadge(job.rowEl, "error", "lookup failed", String(err));
      }
      done++;
      updateBtn();
    }
  }
  await Promise.all(
    Array.from({ length: state.lookupConcurrency }, () => worker())
  );

  btn.disabled = false;
  btn.textContent = originalLabel;
  setStatus(
    `Checked ${total} row(s) via request-o-matic` +
    (pageYear ? ` (gating release_year > ${pageYear} as anachronistic).` : ".")
  );
}

function $$(sel, root = document) {
  return Array.from(root.querySelectorAll(sel));
}

function togglePageView() {
  const aside = $("#page-view");
  const main = $("main");
  const btn = $("#toggle-page-view");
  const open = !aside.classList.contains("is-open");
  aside.classList.toggle("is-open", open);
  main.classList.toggle("page-view-open", open);
  btn.classList.toggle("is-active", open);
  aside.setAttribute("aria-hidden", String(!open));
  btn.textContent = open ? "Hide page" : "Show page";
}

// ---- wiring --------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  $("#bundle-input").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) loadBundleFromFile(file);
  });
  $("#image-input").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) loadImageFromFile(file);
  });
  $("#save-verified").addEventListener("click", saveAll);
  $("#toggle-page-view").addEventListener("click", togglePageView);
  $("#check-artists").addEventListener("click", checkArtists);

  await loadBundleFromUrlParam();
});
