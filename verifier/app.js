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

const SUPPORTED_SCHEMA_VERSION = 1;

const state = {
  bundle: null,            // mutable working copy
  originalBundle: null,    // immutable snapshot for diffing
  pageImage: null,         // HTMLImageElement
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
  $("#export-verified").disabled = false;
  $("#toggle-page-view").disabled = false;
  $("#page-view-img").src = state.pageImage.src;
  renderPageMeta();
  renderQuadrants();
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
  });

  const typeEl = $(".type-raw input", node);
  typeEl.value = entry.type_raw ?? "";
  typeEl.addEventListener("input", () => {
    entry.type_raw = typeEl.value || null;
  });

  const notesEl = $(".notes select", node);
  notesEl.value = entry.notes ?? "";
  notesEl.addEventListener("change", () => {
    entry.notes = notesEl.value || null;
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

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportAll() {
  const verified = buildVerifiedExport();
  const corrections = buildCorrectionsExport();
  downloadJson(`${state.bundle.stem}.verified.json`, verified);
  downloadJson(`${state.bundle.stem}.corrections.json`, corrections);
  const n =
    corrections.row_corrections.length +
    corrections.page_corrections.length +
    corrections.quadrant_corrections.length;
  setStatus(
    `Exported verified + corrections (${n} field correction(s), ` +
    `${corrections.added_rows.length} added, ` +
    `${corrections.deleted_rows.length} deleted).`
  );
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
  $("#export-verified").addEventListener("click", exportAll);
  $("#toggle-page-view").addEventListener("click", togglePageView);

  await loadBundleFromUrlParam();
});
