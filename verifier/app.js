// Flowsheet verifier — vanilla JS, no build step.
//
// Two modes, switched on the presence of `?bundle=` in the URL:
//
//   1. INDEX mode (no ?bundle=): show the list of every bundle in
//      data/verifier/ with status (incomplete / partial / complete),
//      plus an "Open next incomplete" jump button.
//
//   2. EDIT mode (?bundle=<path>): the legacy per-page editor. Renders
//      per-row canvas crops next to editable text fields. Save POSTs to
//      /api/save which writes <stem>.verified.json + <stem>.corrections.json
//      and updates jobs.db when a job key is present. Mark complete sends
//      the same payload with status="complete".
//
// State is split:
//   state.originalBundle  — immutable snapshot of the loaded bundle. Never
//                           mutated; used as the diff baseline on save.
//   state.bundle          — working copy. Mutated by edits and UI flags
//                           (`_added`, `_deleted`).
//   state.bundleList      — cached /api/bundles response, used by Prev/Next
//                           and the keyboard shortcuts. Refreshed after a
//                           successful save so the status pill reflects truth.

"use strict";

const SUPPORTED_SCHEMA_VERSION = 2;

// ---- auth: /auth/me + 401 → /auth/login redirect ------------------------
//
// When the verifier is deployed with OIDC enabled, every /api/* and
// /verifier/* request goes through a session-cookie middleware. A
// missing or expired cookie produces:
//   * 302 → /auth/login?return_to=...     (for HTML nav)
//   * 401 JSON                              (for fetch() requests with
//                                             Accept: application/json)
//
// `authFetch` is the wrapper the SPA uses for every same-origin API
// call. On 401 it sends the user to /auth/login with the current path
// preserved as return_to, so after login they land back on the same
// page they were trying to use.
//
// When the deployment has NO auth gate (local-dev default), /auth/me
// returns 404 (no route installed). authFetch is a no-op around fetch
// in that case.

function redirectToLogin() {
  const returnTo = location.pathname + location.search;
  location.href = "/auth/login?return_to=" + encodeURIComponent(returnTo);
}

async function authFetch(input, init = {}) {
  // Mark the request as JSON so the middleware returns 401 JSON
  // instead of 302 — otherwise fetch() either follows the redirect
  // silently (in which case the SPA can't react) or treats it as an
  // opaque error.
  const headers = new Headers(init.headers || {});
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  const response = await fetch(input, { ...init, headers });
  if (response.status === 401) {
    redirectToLogin();
    // Throw so the calling code's catch surfaces a clear error rather
    // than trying to parse an empty body.
    throw new Error("authentication required");
  }
  return response;
}

async function renderReviewerName() {
  // Fetch the current reviewer on load and show their name in the
  // header. A 401 means the deployment has the gate enabled but our
  // cookie is gone — let the middleware redirect us on the next API
  // call rather than triggering an extra round-trip here. A 404 means
  // the deployment has no auth, so nothing to render.
  try {
    const r = await fetch("/auth/me", {
      headers: { "Accept": "application/json" },
    });
    if (!r.ok) return;
    const me = await r.json();
    const label = me.real_name || me.dj_name || me.username || me.email || "Signed in";
    for (const id of ["reviewer-name", "reviewer-name-index"]) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.hidden = false;
      el.textContent = label;
      el.title = me.email || "";
    }
  } catch {
    // Network error — silently leave the chip hidden. The next save
    // attempt will surface the failure via authFetch.
  }
}

const state = {
  bundle: null,            // mutable working copy
  originalBundle: null,    // immutable snapshot for diffing
  pageImage: null,         // HTMLImageElement
  lookupConcurrency: 4,    // parallel /api/lookup requests
  bundleList: null,        // cached array from /api/bundles
  focusedRowIndex: null,   // for j/k keyboard nav across rows
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
    // No-store: bundle JSONs are refreshed by the entrypoint on every
    // deploy (see scripts/railway_entrypoint.sh's overlay step). A
    // browser cache hit on a stale bundle would silently feed the old
    // bbox + entries into the UI.
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(`fetch ${path}: ${r.status}`);
    const bundle = await r.json();
    // If a verified.json exists alongside the bundle, overlay its data so
    // the volunteer's previously-saved edits appear pre-populated. Without
    // this, /api/save persists files on disk but the UI re-renders the
    // original Gemini output on every reload — edits "don't save."
    const verifiedPath = path.replace(/\.bundle\.json$/, ".verified.json");
    let overlaidFrom = null;  // timestamp string when overlay applied
    try {
      // `cache: "no-store"` is load-bearing: this file is rewritten on
      // every save (`POST /api/save`), and the `/data` static mount sends
      // ETag/Last-Modified but no Cache-Control. Without `no-store` the
      // browser will happily reuse a pre-save copy on the next page load,
      // which displays the OLD value and looks like the save never
      // happened.
      const vr = await fetch(verifiedPath, { cache: "no-store" });
      if (vr.ok) {
        const verified = await vr.json();
        applyVerifiedToBundle(bundle, verified);
        const lm = vr.headers.get("last-modified");
        overlaidFrom = lm || verified.extracted_at || "unknown";
      }
    } catch (e) {
      console.warn("verified.json fetch failed; using model output as-is", e);
    }
    await initBundle(bundle, { bundleUrl: path });
    if (overlaidFrom) {
      // Visible cue so the volunteer can tell at a glance whether their
      // prior edits were restored — if this message is missing on a page
      // they remember saving, the overlay path failed and Save would
      // appear "not to persist."
      setStatus(`Restored your saved edits (last saved ${overlaidFrom}).`);
    }
    return true;
  } catch (err) {
    setStatus(`Failed to load bundle: ${err.message}`, "error");
    return false;
  }
}

// Overlay a saved PageResult ("verified.json") onto the bundle so the UI
// shows the volunteer's prior edits as the starting state. The bundle's
// per-entry `row_bbox` is geometry (computed from the page image) and is
// preserved by matching entries on `row_index`; added rows get null.
function applyVerifiedToBundle(bundle, verified) {
  bundle.page_date_raw = verified.page_date_raw ?? null;
  bundle.comments_raw = verified.comments_raw ?? null;
  bundle.oddities = Array.isArray(verified.oddities) ? verified.oddities : [];
  for (const vq of verified.quadrants ?? []) {
    const bq = bundle.quadrants.find(q => q.position === vq.position);
    if (!bq) continue;
    bq.hour_raw = vq.hour_raw ?? null;
    bq.jock_raw = vq.jock_raw ?? null;
    bq.oddities = Array.isArray(vq.oddities) ? vq.oddities : [];
    const bboxByIndex = new Map(bq.entries.map(e => [e.row_index, e.row_bbox]));
    bq.entries = (vq.entries ?? []).map(e => ({
      ...e,
      row_bbox: bboxByIndex.get(e.row_index) ?? null,
    }));
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
  $("#mark-complete").disabled = false;
  $("#check-artists").disabled = false;
  $("#page-view-img").src = state.pageImage.src;
  renderPageMeta();
  renderQuadrants();
  // Best-effort: fetch the bundle list to enable Prev/Next and the
  // status pill. Failures don't block editing.
  refreshNavFromBundleList().catch(() => {});
}

// ---- nav + status pill --------------------------------------------------

async function fetchBundleList() {
  // No-store: status changes on every save and the index list / pill
  // must reflect that immediately. Tiny payload, no reason to cache.
  const r = await authFetch("/api/bundles", { cache: "no-store" });
  if (!r.ok) throw new Error(`/api/bundles ${r.status}`);
  const data = await r.json();
  state.bundleList = data.bundles || [];
  return state.bundleList;
}

// Find the current bundle's position in state.bundleList. Returns -1 if
// the current bundle isn't in the list (e.g., picked via file-picker, or
// the bundle was removed between page loads).
function currentBundleIndex() {
  if (!state.bundle || !state.bundleList) return -1;
  return state.bundleList.findIndex(b => b.stem === state.bundle.stem);
}

async function refreshNavFromBundleList({ pillFromList = true } = {}) {
  if (!state.bundle) return;
  try {
    await fetchBundleList();
  } catch (err) {
    // Non-fatal — Prev/Next stay disabled, no status pill.
    console.warn("bundle list fetch failed:", err);
    return;
  }
  const idx = currentBundleIndex();
  const total = state.bundleList.length;
  const posEl = $("#nav-position");
  const prevBtn = $("#prev-page");
  const nextBtn = $("#next-page");
  if (idx >= 0) {
    posEl.textContent = `${idx + 1} / ${total}`;
    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx === total - 1;
  } else {
    posEl.textContent = `? / ${total}`;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  }
  // After a save the pill is already authoritative from /api/save's
  // response — skip overwriting it from the (potentially-cached) list.
  // On the initial load there's no fresher source, so we use the list.
  if (pillFromList) {
    updateStatusPill(state.bundleList[idx]?.status ?? "incomplete");
  }
}

function updateStatusPill(status) {
  state.currentStatus = status;
  const pill = $("#status-pill");
  pill.hidden = false;
  pill.className = `status-pill status-${status}`;
  pill.textContent = status;
  // The Mark complete button is a toggle: click on a complete page
  // explicitly reverts to draft. Label and title reflect the action
  // the next click will take.
  const btn = $("#mark-complete");
  if (btn) {
    if (status === "complete") {
      btn.textContent = "Mark incomplete";
      btn.title = "Revert this page to draft (⌘⇧S)";
    } else {
      btn.textContent = "Mark complete";
      btn.title = "Mark this page complete (⌘⇧S)";
    }
  }
}

function navigateTo(bundle) {
  if (!bundle?.url) return;
  location.href = bundle.url;
}

function navigatePrev() {
  const idx = currentBundleIndex();
  if (idx > 0) navigateTo(state.bundleList[idx - 1]);
}

function navigateNext() {
  const idx = currentBundleIndex();
  if (idx >= 0 && idx < state.bundleList.length - 1) {
    navigateTo(state.bundleList[idx + 1]);
  }
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

function humanizePosition(pos) {
  // "top_left" → "Top left". Display only; the underscore form remains the
  // canonical identifier on the bundle data and the article's dataset.
  return pos.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
}

function renderQuadrants() {
  const container = $("#quadrants-container");
  container.innerHTML = "";
  const tmpl = $("#quadrant-template");

  for (const quad of state.bundle.quadrants) {
    const node = tmpl.content.firstElementChild.cloneNode(true);
    // Underlying position (e.g. "top_left") stays on the dataset so other
    // code can find this node by quadrant; the visible title is humanized.
    node.dataset.position = quad.position;
    $(".quadrant-title", node).textContent = humanizePosition(quad.position);

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

async function saveAll(requestedStatus = null) {
  // requestedStatus options:
  //   null         — plain Save; server applies the preserve-on-disk rule
  //   "complete"   — toggle clicked while not complete
  //   "draft"      — toggle clicked on an already-complete page (explicit revert)
  if (!state.bundle) return;
  const isToggle = requestedStatus !== null;
  const btn = isToggle ? $("#mark-complete") : $("#save-verified");
  const otherBtn = isToggle ? $("#save-verified") : $("#mark-complete");
  btn.disabled = true;
  otherBtn.disabled = true;
  const original = btn.textContent;
  btn.textContent =
    requestedStatus === "complete" ? "Marking…" :
    requestedStatus === "draft"    ? "Reverting…" :
                                     "Saving…";
  let saveOk = false;

  const verified = buildVerifiedExport();
  const corrections = buildCorrectionsExport();
  const body = {
    stem: state.bundle.stem,
    pdf_path: state.bundle.pdf_path ?? null,
    page_number: state.bundle.page_number ?? null,
    verified,
    corrections,
  };
  if (requestedStatus) body.status = requestedStatus;

  try {
    const r = await authFetch("/api/save", {
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
    // Trust the immediate response for the pill — it's authoritative for
    // this save. The follow-up list refresh is silent and updates the
    // Prev/Next state only (the pill is set just above and not touched
    // by the refresh path).
    updateStatusPill(result.status === "complete" ? "complete" : "partial");
    saveOk = true;
    setStatus(
      `Saved as ${result.status} · ${result.verified_path} + ${result.corrections_path} · ` +
      `${n} field correction(s), ${corrections.added_rows.length} added, ` +
      `${corrections.deleted_rows.length} deleted · ${dbBit}.`
    );
    refreshNavFromBundleList({ pillFromList: false }).catch(() => {});
  } catch (err) {
    setStatus(`Save failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    otherBtn.disabled = false;
    // Save button label is static — always restore. Mark complete button's
    // label is set by updateStatusPill on success; only restore on failure
    // (where it's still showing "Marking…" / "Reverting…").
    if (!isToggle || !saveOk) btn.textContent = original;
  }
}

const saveDraft = () => saveAll(null);
// Toggle: revert from complete back to draft when already complete,
// otherwise mark complete. The button label tracks the next-click action.
const toggleComplete = () =>
  saveAll(state.currentStatus === "complete" ? "draft" : "complete");

// ---- artist/track lookup (request-o-matic via /api/lookup proxy) --------

async function lookupOne(message) {
  const r = await authFetch("/api/lookup", {
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
    const quadNode = $(`.quadrant[data-position="${quad.position}"]`);
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

// ---- index mode ----------------------------------------------------------

async function showIndex() {
  $("#index-header").hidden = false;
  $("#index").hidden = false;
  const statusEl = $("#index-status");
  statusEl.textContent = "Loading…";
  try {
    await fetchBundleList();
    renderBundleList();
  } catch (err) {
    statusEl.textContent = `Failed to load bundle list: ${err.message}`;
    return;
  }
  const incompleteCount = state.bundleList.filter(b => b.status !== "complete").length;
  $("#open-next-incomplete").disabled = incompleteCount === 0;
  const counts = {
    incomplete: state.bundleList.filter(b => b.status === "incomplete").length,
    partial: state.bundleList.filter(b => b.status === "partial").length,
    complete: state.bundleList.filter(b => b.status === "complete").length,
  };
  statusEl.textContent =
    `${state.bundleList.length} bundle(s): ` +
    `${counts.incomplete} incomplete · ${counts.partial} partial · ${counts.complete} complete.`;
}

function renderBundleList() {
  const tbody = $("#bundle-list tbody");
  tbody.innerHTML = "";
  for (const bundle of state.bundleList) {
    const tr = document.createElement("tr");
    tr.dataset.stem = bundle.stem;
    tr.classList.toggle("is-complete", bundle.status === "complete");

    const statusTd = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `status-pill status-${bundle.status}`;
    pill.textContent = bundle.status;
    statusTd.appendChild(pill);
    tr.appendChild(statusTd);

    const stemTd = document.createElement("td");
    stemTd.className = "stem";
    stemTd.textContent = bundle.stem;
    tr.appendChild(stemTd);

    const dateTd = document.createElement("td");
    dateTd.textContent = bundle.page_date_raw || "—";
    tr.appendChild(dateTd);

    const tsTd = document.createElement("td");
    tsTd.className = "timestamp";
    tsTd.textContent = bundle.verified_at
      ? new Date(bundle.verified_at).toLocaleString()
      : "—";
    tr.appendChild(tsTd);

    tr.addEventListener("click", () => navigateTo(bundle));
    tbody.appendChild(tr);
  }
}

function openNextIncomplete() {
  if (!state.bundleList) return;
  const next = state.bundleList.find(b => b.status !== "complete");
  if (next) {
    navigateTo(next);
  } else {
    $("#index-status").textContent = "All pages complete!";
  }
}

// ---- keyboard shortcuts --------------------------------------------------

function isEditableTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return (
    tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable
  );
}

function focusedRowEl() {
  if (state.focusedRowIndex == null) return null;
  const rows = $$(".row");
  return rows[state.focusedRowIndex] || null;
}

function setFocusedRow(idx) {
  const rows = $$(".row");
  if (idx < 0 || idx >= rows.length) return;
  $$(".row.is-focused").forEach(el => el.classList.remove("is-focused"));
  state.focusedRowIndex = idx;
  const target = rows[idx];
  target.classList.add("is-focused");
  target.scrollIntoView({ block: "nearest", behavior: "smooth" });
  const input = $(".raw-text", target);
  if (input) input.focus();
}

function focusNextRow(delta) {
  const rows = $$(".row");
  if (!rows.length) return;
  let next = state.focusedRowIndex == null
    ? (delta > 0 ? 0 : rows.length - 1)
    : state.focusedRowIndex + delta;
  next = Math.max(0, Math.min(rows.length - 1, next));
  setFocusedRow(next);
}

function toggleShortcutOverlay() {
  const o = $("#shortcut-overlay");
  o.hidden = !o.hidden;
}

function installKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Esc closes overlay regardless of focus.
    if (e.key === "Escape" && !$("#shortcut-overlay").hidden) {
      toggleShortcutOverlay();
      e.preventDefault();
      return;
    }
    // ⌘S / Ctrl+S → save. Works even when an input has focus.
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      if (e.shiftKey) toggleComplete();
      else saveDraft();
      return;
    }
    // ⌘D / Ctrl+D → toggle delete on focused row. Also works in inputs.
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "d") {
      const row = focusedRowEl();
      if (row) {
        const btn = row.querySelector(".delete-row");
        if (btn) {
          btn.click();
          e.preventDefault();
        }
      }
      return;
    }
    // Non-modifier letter shortcuts are ignored when an input has focus,
    // so the verifier can type normally.
    if (isEditableTarget(e.target)) return;
    if (e.key === "?") {
      toggleShortcutOverlay();
      e.preventDefault();
    } else if (e.key === "j" || (e.metaKey && e.key === "ArrowDown")) {
      focusNextRow(1);
      e.preventDefault();
    } else if (e.key === "k" || (e.metaKey && e.key === "ArrowUp")) {
      focusNextRow(-1);
      e.preventDefault();
    } else if (e.key === "n") {
      navigateNext();
      e.preventDefault();
    } else if (e.key === "p") {
      navigatePrev();
      e.preventDefault();
    }
  });
  // Click outside the overlay card closes it.
  $("#shortcut-overlay").addEventListener("click", (e) => {
    if (e.target.id === "shortcut-overlay") toggleShortcutOverlay();
  });
  // Top-right floating help button and the card's × button both toggle.
  $("#show-shortcuts").addEventListener("click", toggleShortcutOverlay);
  $("#shortcut-overlay .overlay-close").addEventListener("click", toggleShortcutOverlay);
}

// ---- wiring --------------------------------------------------------------

// ---- in-app deploy detector ---------------------------------------------

// Captured on load; if /api/version drifts from this we surface the banner.
let INITIAL_APP_VERSION = null;
const VERSION_POLL_MS = 60_000;

async function fetchAppVersion() {
  try {
    const r = await fetch("/api/version", { cache: "no-store" });
    if (!r.ok) return null;
    const body = await r.json();
    return body.version ?? null;
  } catch {
    return null;
  }
}

function showUpdateBanner() {
  if (document.getElementById("update-banner")) return;
  const banner = document.createElement("div");
  banner.id = "update-banner";
  banner.innerHTML =
    '<span>A new version is available. Your saved work is preserved on the server.</span>' +
    '<button type="button" id="update-banner-reload">Reload now</button>';
  document.body.appendChild(banner);
  document.getElementById("update-banner-reload").addEventListener("click", () => {
    location.reload();
  });
}

async function checkForUpdate() {
  if (INITIAL_APP_VERSION === null) return;
  const current = await fetchAppVersion();
  if (current && current !== INITIAL_APP_VERSION) {
    showUpdateBanner();
  }
}

async function startVersionWatch() {
  INITIAL_APP_VERSION = await fetchAppVersion();
  if (!INITIAL_APP_VERSION) return;
  setInterval(checkForUpdate, VERSION_POLL_MS);
}

document.addEventListener("DOMContentLoaded", async () => {
  installKeyboardShortcuts();
  // Fire-and-forget — must not block bundle loading.
  startVersionWatch().catch(() => {});
  renderReviewerName().catch(() => {});
  const params = new URLSearchParams(location.search);
  const hasBundle = !!params.get("bundle");
  if (!hasBundle) {
    // Wire index-mode controls once, then render. `showIndex` populates
    // state and toggles the disabled state on this button — the listener
    // itself stays attached for the lifetime of the page.
    $("#open-next-incomplete").addEventListener("click", openNextIncomplete);
    await showIndex();
    return;
  }
  // Edit mode.
  $("#edit-header").hidden = false;
  $("#image-input").addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (file) loadImageFromFile(file);
  });
  $("#save-verified").addEventListener("click", saveDraft);
  $("#mark-complete").addEventListener("click", toggleComplete);
  $("#check-artists").addEventListener("click", checkArtists);
  $("#prev-page").addEventListener("click", navigatePrev);
  $("#next-page").addEventListener("click", navigateNext);

  await loadBundleFromUrlParam();
});
