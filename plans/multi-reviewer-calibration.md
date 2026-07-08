# Multi-reviewer calibration for the anomaly bucket

## Motivation

The verifier today writes a single `<stem>.verified.json` per page — one volunteer's reading is treated as ground truth. For the *calibration anomaly bucket* (the 5 known-pathology bundles from `project_bbox_sweep_result.md` today, and the per-year anomaly subset that issue #66's sampler will emit later), single-reviewer ground truth is too thin. The pages selected for calibration are precisely the ones where a single reviewer is most likely to misread, and a single reading gives no signal about how good "ground truth" itself is. Per the n=19 empirical failure modes, inter-reviewer agreement on the random bucket is roughly 90–95% on `raw_text`; on the anomaly bucket it is almost certainly lower, and we can't measure or correct for it without independent observations.

This plan introduces a blind multi-reviewer protocol for the anomaly bucket. Each page is reviewed independently by at least two reviewers; a third reviewer is recruited automatically when the first two disagree on any gating field. A pure-function merge step produces a canonical record per page, capturing per-row consensus, dissents, and inter-reviewer agreement statistics. The canonical record is the input to `derive_truth.py` for that page; the agreement statistics become the calibration noise-floor that issue #66's drift gate compares Gemini against — so that "Gemini's row-edit rate is 8%" can be read against "inter-reviewer agreement was 94%" rather than against an implicit assumption of perfect truth.

The work ships as two pull requests (backend, then SPA) plus a one-shot bootstrap script for the seed set. It does not touch the regular single-reviewer verifier flow, the random calibration bucket (covered by #66's separate budget rationale), or any corpus-wide processing.

## Scope

In scope:
- A new on-disk layout under `data/calibration/<year>/<bucket>/<stem>/` for anomaly-bucket pages.
- A pure-function merge that consumes per-reviewer submissions and produces `canonical.json` plus `agreement.json`.
- Server-side blind-review enforcement (per-reviewer file access control) in `verifier/serve.py`.
- New verifier endpoints for queue, draft auto-save, submit, and reads of bundle / per-reviewer / canonical / agreement files.
- An SPA calibration-review mode: blind reading, flag-spurious affordance, missing-row-marker affordance, confirm-to-submit gesture, calibration-mode banner.
- A one-shot bootstrap script for the 5 named bundles from `project_bbox_sweep_result.md`.
- `scripts/derive_truth.py` learns to read `canonical.json` and emit truth files under `tests/golden/calibration/<year>/<stem>.truth.json`.
- Schema versioning for the four new on-disk file shapes.
- Test fixtures, property tests for the comparator, and one real-bundle integration test.

Out of scope (deliberate):
- The corpus-wide single-reviewer verifier flow stays untouched.
- The random calibration bucket stays single-reviewer (issue #66 carries the budget rationale).
- LML / `@wxyc/shared` canonicalization in the agreement comparator (Phase 2; the comparator exposes a pluggable Tier-2 hook but does not implement one).
- The per-year sampler that emits the anomaly bucket list (issue #66, separate plan).
- Year-level rollup tooling beyond a small post-hoc reader script.
- Real-time presence (no "X others are reviewing this page" UX — that would compromise blind review).
- Recallable submissions, leaderboards, notifications, or retroactive backfill of existing single-reviewer pages.
- OIDC identity provisioning — that work has already landed (issues #67 and #68, closed). `core/auth.py` populates `request.state.reviewer` as a `ReviewerSession` with `user_id` (the OIDC `sub`), `email`, `username`, `real_name`, `dj_name`, and `role`. This plan consumes that surface; it does not modify it.

## Dependencies

The OIDC reviewer identity work is already merged to `main`. Specifically, `origin/main` is at commit `29581fe fix(verifier): address code-review findings (iter 2)`, whose tree contains `core/auth.py`, the `_SessionCookieMiddleware` in `verifier/serve.py` (installed when `OIDC_ENABLED` is truthy), and the `VerifiedBy` model on `core/schema.py:173` (with `verified_by: VerifiedBy | None` on `PageResult`). Verifiable in one command each:

```
git fetch origin main
git ls-tree origin/main core/auth.py                                        # blob present
git show origin/main:core/schema.py     | grep 'class VerifiedBy'           # class present
git show origin/main:verifier/serve.py  | grep '_SessionCookieMiddleware'   # middleware present
git show origin/main:verifier/serve.py  | grep '_PUBLIC_PATHS'              # allowlist present
```

This plan consumes that surface directly: every calibration endpoint reads `request.state.reviewer` (a `ReviewerSession`), and the on-disk reviewer block reuses the existing `core.schema.VerifiedBy` Pydantic model — same field names (`user_id`, `username`, `real_name`, `dj_name`, `verified_at`), same nullability rules, no new identity types. Because the dependency has already landed on `main`, the backend PR branches from `origin/main` directly and can merge as soon as its own review is clean; there is no upstream gate to wait on. (Historically these commits shipped through PRs merged before this plan was filed: OIDC client and cookie codec at `da40914`, the middleware + `verified_by` at `9f7e335`, and two review-fixup iterations at `8ad503c`..`29581fe`. All ancestors of `origin/main`.)

Independently of OIDC, the existing regular verifier flow (single-reviewer reads/writes under `data/verifier/<stem>.verified.json`) continues to function. This plan adds a parallel calibration tree; it does not modify the regular tree's serving code, schemas, or static mounts beyond opening one additional mount for `data/calibration/`.

## File layout

Each calibration page is a directory containing one shared bundle and one file per reviewer who has touched it:

```
data/calibration/<year>/<bucket>/<stem>/
    bundle.json             # relative symlink to data/verifier/<stem>.bundle.json
    draft.<short>.json      # zero or one per reviewer-in-progress; mutable; owner-readable only
    verified.<short>.json   # zero or one per submitted reviewer; immutable; access-gated
    canonical.json          # materializes only at completion
    agreement.json          # materializes only at completion
```

Where:
- `<year>` is the four-digit year parsed from the stem prefix (`1990-04apr0106-page14` → `1990`).
- `<bucket>` is `anomaly` for this plan. The dimension is reserved so that a future random-bucket extension fits without a path migration.
- `<stem>` is the bundle stem as used today by the verifier (e.g., `1990-04apr0106-page14`).
- `<short> = sha256(user_id).hexdigest()[:12]` — a 48-bit hex prefix of the reviewer's OIDC subject (`ReviewerSession.user_id`). Stable, filesystem-safe, short enough to read in `ls`, collision-negligible for any plausible reviewer pool size. The full identity lives inside the file in the `verified_by` block (a `VerifiedBy` instance); a mapping table at `data/calibration/_reviewers.json` exposes `short → {user_id, username, real_name, dj_name}` for ops debugging and is never consulted by the merge handler.

The `bundle.json` is a relative symlink (`../../../../verifier/<stem>.bundle.json`) so that future bundle regenerations propagate automatically without copies going stale invisibly. The verifier app's static mount follows symlinks by default.

Year-stratified placement matters for two reasons: it makes per-year drift-gate aggregation a directory walk rather than a metadata query, and it positions the layout to absorb #66's per-year sampler output without re-shaping. The `<bucket>` dimension is similarly forward-looking: if random-bucket multi-reviewer is ever added, the path generalises with no rename.

### State machine readable from disk

The presence and shape of files in a page directory is the page's state — there is no separate SQLite row tracking calibration state. Specifically:

- No `verified.*.json` files: page is fresh (or only drafts exist).
- One or more `verified.*.json` files, no `canonical.json`: page is awaiting more submissions or a tiebreaker.
- `canonical.json` and `agreement.json` both present: page is settled.

This is the load-bearing simplification of the design. It means the queue endpoint is a directory walk plus per-page comparator runs, the merge handler is the single writer of `canonical.json` (which makes "no second writes after settled" trivially enforceable), and there is no second-source-of-truth to drift.

## Protocol

Each page is reviewed blind: a reviewer working on page X cannot see what any other reviewer wrote for page X until the page reaches `verified` status. Blindness is enforced server-side (see *Blind-review enforcement* below). The default reviewer target per page is two; the target bumps to three when the first two disagree on any gating field. The target never bumps beyond three: when three reviewers fail to agree on a given field, the row's value for that field is set to an auto-illegible sentinel and the page closes.

### Gating vs informational fields

Four per-row decisions can trigger a `target_reviewers` bump from 2 to 3:

| Field | Gates escalation? | Auto-illegible sentinel on 1-1-1 |
|---|---|---|
| `raw_text` | yes | `__illegible__` |
| `type_raw` | yes | `_unknown` |
| `spurious_flag` (binary per bundle row) | yes (1-1 at N=2 → bump) | n/a — binary, always has majority at N=3 |
| missing-row marker presence (per gap between bundle rows) | yes (1 reports / 1 doesn't → bump) | n/a — binary |
| `notes` | no | n/a — values listed informationally |
| `confidence` | no | n/a — dropped entirely |

The **worst-of escalation rule**: a page's `target_reviewers` bumps from 2 to 3 if any row triggers disagreement on any of the four gating decisions. One disputed row anywhere on the page is enough.

`target_reviewers` is **monotonically ratcheting**: it only ever moves 2 → 3, never 3 → 2. Once a submission has revealed disagreement on any gating decision, the target stays at 3 for the remainder of the page's life even if a later submission would (hypothetically) reduce that specific disagreement — the disagreement happened, the auditor's record must reflect it. Concretely: the merge function recomputes `target = worst_of(target, per_submission_target)` on every invocation, never assigns `target = 2` after a `= 3` step. This is why the pseudocode below opens with `target = 2` and only ever assigns `target = 3` inside the loop.

### Auto-illegible vs auto-recover

When the third reviewer is recruited and still no majority emerges:

- `raw_text` 1-1-1 → row keeps a `__illegible__` raw_text in canonical; each reviewer's reading is preserved in `verification.raw_text_dissents`. Downstream, `derive_truth.py` skips this row (existing subset semantics handle skipping naturally).
- `type_raw` 1-1-1 → row keeps `_unknown` type_raw in canonical; readings preserved. Downstream consumers (Phase 2 rotation reconciliation) can inspect dissents.
- `spurious_flag` cannot 1-1-1 (binary at N=3 → guaranteed majority).
- Missing-row marker cannot 1-1-1 on presence (binary), but text agreement among reporters can fail; in that case the row is still injected with `__illegible__` text plus dissents, because the consensus is "Gemini missed a row here" even if reviewers don't agree on what.

The page transitions to `verified` once every gating decision has reached either majority/unanimous or auto-illegible. `target_reviewers` is captured in `agreement.json` as a frozen historical value.

## Row-set spine: bundle rows are pinned

The bundle's row list is the spine of the per-page consensus. A reviewer in calibration mode **cannot** free-form add or delete rows. They can:

- Edit `raw_text`, `type_raw`, and `notes` on a bundle row.
- Flag a bundle row as `spurious` (the SPA clears that row's text field; downstream the merge treats this as a negative vote on the row's existence).
- Insert a missing-row marker between bundle rows `N` and `N+1`, with suggested text, `type_raw`, and `notes`.

This is a deliberately less-powerful UI than the regular verifier (which exposes `add-row` at `app.js:343` and tracks `deleted_rows`). The trade-off is that the merge becomes a trivial O(rows) index-aligned scan instead of a sequence-alignment problem, while preserving the under-emit and over-emit signal that the n=19 empirical failure modes specifically named (3% under-emit, 0% over-emit).

The canonical row layout reflects the spine plus injections:

```jsonc
{
  "canonical_row_index": 7,         // 0..M-1 across all canonical rows
  "bundle_row_index": 6,            // index into bundle.rows; null for missing-row injections
  "raw_text": "BEATLES - HELP",     // null when spurious_majority wins
  "type_raw": "H",
  "notes": null,
  "confidence": "high",             // taken from the majority's submission; informational
  "verification": { ... }            // see Merge algorithm below
}
```

A canonical row's `canonical_row_index` is a contiguous 0..M-1 sequence assigned at merge time. Rows derived from bundle rows preserve `bundle_row_index`; missing-row injections carry `bundle_row_index: null` plus a sidecar `inserted_between_bundle_rows: [N, N+1]`.

## Comparator (Tier 1)

The `raw_text` agreement check is a pure function:

```python
def rows_agree(
    a: str,
    b: str,
    *,
    canonicalize: Callable[[str], str | None] | None = None,
) -> bool:
    """Tier 1: normalize and byte-compare. Tier 2: optional LML canonicalizer."""
    if _normalize_raw_text(a) == _normalize_raw_text(b):
        return True
    if canonicalize is not None:
        ca, cb = canonicalize(a), canonicalize(b)
        if ca and cb and ca == cb:
            return True
    return False
```

`_normalize_raw_text` applies:
1. NFKC unicode normalization.
2. Casefold (Unicode-correct lowercase).
3. `&` → ` and ` (with whitespace pad to avoid `Earth & Wind` → `Earthandwind`).
4. Strip characters: `.`, `'` (straight and smart), `"` (straight and smart), `,`.
5. Dash normalization: `–` (en-dash) and `—` (em-dash) → `-`.
6. Collapse runs of whitespace to a single space; strip leading and trailing whitespace.

Two characters are deliberately preserved: the artist/track separator ` - ` (load-bearing in WXYC flowsheet conventions) and the slash (`AC/DC`, `GZA/Genius`, where slash carries identity).

Tier 2 is an explicit `canonicalize` callable argument and is **not** implemented in this plan. The plan reserves the function signature so that when Phase 2's LML reconciliation lands (issue #62 territory), a one-line wiring change makes the merge LML-aware without re-shaping the comparator. The reserved signature is `canonicalize: Callable[[str], str | None]`: given a raw text string, return a canonicalized form, or `None` if canonicalization cannot be produced (LML miss, backend error, ambiguous match). Semantics inside `rows_agree`:

- Tier 1 is tried first (`_normalize_raw_text(a) == _normalize_raw_text(b)`). A Tier-1 match ends the check with `True`; the canonicalizer is never called on Tier-1 matches, so an LML outage cannot break equality on the easy cases.
- Tier 2 runs only when Tier 1 disagrees. Both inputs are canonicalized. Agreement is claimed **only** when both `canonicalize(a)` and `canonicalize(b)` return a non-`None` value **and** those values are equal. A single `None` from either input degrades the pair to "Tier 1 disagreement wins" — Tier 2 never converts a `None` into agreement, so an LML miss is fail-closed for agreement (the safe direction: we prefer to over-escalate to a 3rd reviewer than under-count real disagreement).
- The canonicalizer is a pure function from the caller's perspective. Exception propagation is left to the caller; the merge does not swallow LML errors. In practice the Tier-2 wiring at issue #62 time will wrap LML in a retry/timeout layer that returns `None` on hard failures.

The SPA in calibration mode stays LML-blind even after Tier 2 ships. The whole purpose of multi-reviewer is to measure independent human-reader agreement. Real-time LML autocomplete (e.g., "did you mean AC/DC?" while a reviewer types `ACDC`) would anchor reviewers on the lookup's first suggestion and inflate inter-reviewer agreement falsely. Tier 2's role is post-hoc reconciliation in the merge function, not assistive autocomplete in the editor.

### `type_raw` normalization

`type_raw` uses a separate normalization function before comparison:

1. NFKC normalize, casefold, strip whitespace.
2. Collapse the doodle/blank cluster: `{"", null, "?", "-", "doodle", "scribble"}` → sentinel `"_unknown"`.
3. Byte-compare normalized values.

The collapse is informed by `project_oddity_doodle_entry.md`: doodles in the type-column circle are an established pattern, not a transcription error. Reviewers will inconsistently transcribe them as `?`, `doodle`, or leave blank — folding these to a sentinel prevents spurious escalation while keeping the actual letter alphabet (H/M/L/S/Std/O/R) gating in the normal way. Per the user, the H/M/L/S core vocabulary is stable across the 1990–2001 corpus and remains in use today, so disagreement on these specific letters is real signal that should escalate.

### `notes` and `confidence`

`notes` is captured per-reviewer in `verification.notes_values` as a histogram (e.g., `{"null": 2, "continuation": 1}`) and does not gate escalation. The boundaries between `continuation`, `double_height`, `crossed_out`, and `illegible` are genuinely subjective; escalating on `notes` disagreement would 3rd-reviewer almost every page. The histogram preserves the disagreement as a signal for prompt iteration.

`confidence` is dropped from the agreement computation entirely. Per `project_empirical_failure_modes_n19.md`, confidence is anti-calibrated in Gemini's output; reviewer-edited confidence is noise. The canonical row's `confidence` is taken from whichever reviewer's submission won the `raw_text` majority — informational, not aggregated.

## Spurious-flag and missing-row consensus

### Spurious-flag

Binary vote per bundle row, "keep" vs "spurious":

| Tally at N=3 | Status | Effect on canonical |
|---|---|---|
| 3 keep | `unanimous_keep` | row stays; raw_text from text-agreement among the 3 |
| 2 keep / 1 spurious | `majority_keep` | row stays; raw_text from the 2 keepers; spurious flagger recorded as dissent |
| 1 keep / 2 spurious | `majority_spurious` | row's `raw_text` set to null; keeper's text recorded as dissent |
| 3 spurious | `unanimous_spurious` | row's `raw_text` set to null; no dissent |

At N=2, a 1-1 split on `spurious_flag` triggers escalation to N=3. When the SPA's flag-spurious affordance is engaged on a row, the row's text field is cleared client-side; the merge function additionally treats any spurious-flagged row as having no text contribution to the row's text-agreement tally, so text agreement among "keepers" is computed only over reviewers who did not flag spurious.

Downstream: `derive_truth.py` skips rows where the spurious status is `majority_spurious` or `unanimous_spurious` (no truth row to emit).

### Missing-row markers

A reviewer can insert a missing-row marker between bundle rows `N` and `N+1` with a suggested text. The merge applies two consensus axes:

1. **Gap consensus**: at least a majority of reviewers report a missing row at the same gap (identical `N`).
2. **Text consensus**: among reviewers who reported the gap, at least a majority agree on the text under Tier-1 normalization.

Decision matrix:

| Gap consensus | Text consensus | Action |
|---|---|---|
| Majority report gap | Majority agree on text | Inject row into canonical with status `under_emit_majority`, `raw_text` = majority text |
| Majority report gap | No text majority | Inject row, status `under_emit_no_text_agreement`, `raw_text` = `__illegible__`, all reported texts as dissents |
| Minority report gap | n/a | No injection; recorded in canonical-level `missing_row_reports` array; `derive_truth.py` ignores |
| N=2 split: one reports a gap, the other doesn't | n/a | Escalate to N=3 |

Injected rows take `bundle_row_index: null` and `inserted_between_bundle_rows: [N, N+1]`. `derive_truth.py` emits truth for `under_emit_majority` rows using the majority text.

## Merge algorithm

The merge is a pure function. Input: the page directory's `bundle.json` plus all `verified.*.json` files present. Output: `canonical.json` and `agreement.json` content.

The signature block below is the design contract; it uses the concrete Pydantic types defined in *Submission validation* (`CalibrationSubmission`) and *Schemas (final)* (`Canonical`, `Agreement`). `Bundle` is the parsed shape of `bundle.json` (the existing shape emitted by `scripts/make_verifier_bundle.py`, versioned by that script's `SCHEMA_VERSION`); the merge only reads `bundle.rows`, so an implementer may narrow the parameter to `bundle_row_count: int` if that is more convenient at the call site — the merge does not care about bbox coordinates or other bundle fields. Whichever concrete shape is chosen, the merge function's caller is responsible for passing `settled_at` (a caller-owned timestamp; see the split rationale in *Why these choices* / `PageResult`).

```python
def merge(
    bundle: Bundle,                            # or bundle_row_count: int; see above
    submissions: list[CalibrationSubmission],
    *,
    settled_at: datetime,                      # caller-injected, not fetched from a clock
) -> tuple[Canonical | None, Agreement | None, TargetReviewers]:
    """
    Inputs:
        bundle: parsed bundle.json (immutable across submissions).
        submissions: parsed verified.<short>.json files, one per reviewer.
        settled_at: timestamp stamped into canonical.json / agreement.json
            when settlement is reached; injected by the caller so the merge
            stays a pure function of its inputs (matches the PageResult /
            GeminiPageResult split rationale).
    Returns:
        canonical: the page's settled canonical form; None if not yet settled.
        agreement: the per-page agreement record; None if not yet settled.
        target_reviewers: 2 or 3, the live reviewer target given current state.
    """
```

Pseudocode (the implementation lives in `core/calibration_consensus.py` — named to disambiguate from the existing model-calibration harness in `core/calibration.py`, which scores extraction models against goldens):

```
target = 2
status_per_row = []
for i, bundle_row in enumerate(bundle.rows):
    votes = collect_per_row_votes(submissions, i)
    row_status = compute_row_status(votes)
    if row_status.needs_more_reviewers and len(submissions) < 3:
        target = 3
    status_per_row.append(row_status)

missing_row_gaps = collect_missing_row_reports(submissions)
gap_status = compute_gap_status(missing_row_gaps)
if any(gap.needs_more_reviewers and len(submissions) < 3 for gap in gap_status):
    target = 3

if len(submissions) < target:
    return None, None, target  # not yet settled

canonical = build_canonical(bundle, status_per_row, gap_status)
agreement = build_agreement(canonical, submissions, target)
return canonical, agreement, target
```

`compute_row_status` evaluates each of the four gating decisions independently and returns a row-level status that's the worst-of across them. `compute_gap_status` evaluates missing-row markers per gap.

The merge function is invoked synchronously by the submit handler immediately after a `verified.<short>.json` lands on disk. If `target_reviewers` is met and no new disagreement appears, the handler writes `canonical.json` and `agreement.json` atomically (write to temp, fsync, rename) in the page directory. If `target_reviewers` is now 3 (i.e., the freshly-submitted file pushed the count to 2 and revealed disagreement), no canonical is written and the page surfaces in queue queries as "awaiting third reviewer."

### Submit handler write ordering

A single submit POST performs up to four disk writes: the durable `verified.<short>.json`, then (if settlement is reached) `canonical.json` + `agreement.json`, and finally the auxiliary `_reviewers.json` mapping. The ordering matters because a mid-sequence crash must never leave a page-directory in a state the merge misreads.

Rules, in order:

1. **`verified.<short>.json` first**, atomically (write to `verified.<short>.json.tmp`, fsync, `os.replace` to final name). This is the reviewer's durable record; a crash after this step just means later steps run on the next request.
2. **Merge, then `canonical.json` + `agreement.json`** — same atomic pattern. Written only if `target_reviewers` is met and every gating decision is resolved. If a crash happens between the two files, the page directory has `canonical.json` but no `agreement.json` (or vice versa); the queue endpoint treats a directory with `canonical.json xor agreement.json` as *inconsistent* and skips it with a WARN log, prompting an operator to re-run the merge idempotently via `scripts/derive_truth.py --from canonical --replay-missing` (which also re-emits agreement from the canonical + submissions). Since both files derive purely from the on-disk submissions, replay is safe.
3. **`_reviewers.json` last**, non-fatal on error. "First-time submission for a given `user_id`" is detected by reading the current `_reviewers.json`, computing `<short>`, and checking membership — not by scanning the page directory for `verified.*.json` files, which would race against step 1 during concurrent submits by the same reviewer to different pages. On membership miss: read → append → write-temp → fsync → rename, guarded by a per-file `fcntl.flock` so concurrent submit handlers serialise their read-modify-write against each other. On any error at this step (I/O failure, unexpected schema, lock timeout), the handler logs `reviewers_mapping_stale` at WARN with `(user_id_short, exc)` and returns 200 to the client — the mapping is auditing-only, and the `--refresh-reviewers` bootstrap flag documented under `_reviewers.json` above will rebuild it from durable state.

The lock scope is deliberately narrow: only step 3 takes a lock, and only over `_reviewers.json`. Steps 1 and 2 are per-page directory writes that don't need cross-request coordination — the filename encodes the reviewer, so two reviewers submitting simultaneously to the same page write to different files, and one reviewer can't submit twice to the same page (blocked by cross-check #5 in *Submission validation*).

## Submission validation

Every submit POST is validated against a `CalibrationSubmission` Pydantic model. The model lives in `core/schema.py` alongside the existing `Entry`, `Quadrant`, `PageResult`, `GeminiPageResult`, and `VerifiedBy` models — the established convention is one Pydantic module for the project's domain types, and the existing OIDC work likewise extends `PageResult` with `verified_by` in place rather than splitting. A module-level `CALIBRATION_SCHEMA_VERSION = 1` constant sits next to the existing bundle-side `SCHEMA_VERSION` reference; the two versions cover different artifact families and evolve independently.

The new types added to `core/schema.py`:

```python
CALIBRATION_SCHEMA_VERSION = 1

class CalibrationRowSubmission(BaseModel):
    bundle_row_index: NonNegativeInt
    edited_text: str | None        # None if spurious_flag is True
    type_raw: str | None
    notes: str | None
    spurious_flag: bool

class MissingRowMarker(BaseModel):
    between_bundle_rows: tuple[int, int]
    suggested_text: str
    type_raw: str | None
    notes: str | None

    @model_validator(mode="after")
    def _adjacent(self) -> Self:
        a, b = self.between_bundle_rows
        if b != a + 1:
            raise ValueError("between_bundle_rows must be adjacent (N, N+1)")
        return self

class CalibrationSubmission(BaseModel):
    schema_version: Literal[1]
    stem: str
    # `reviewer` reuses the existing core.schema.VerifiedBy model so the
    # calibration-mode `verified_by` block is byte-identical in shape to
    # the regular-mode block (`user_id`, `username`, `real_name`,
    # `dj_name`, `verified_at`). Keeping a single VerifiedBy model means
    # any future identity-claim change ripples once, not twice.
    reviewer: VerifiedBy
    submitted_at: datetime
    rows: list[CalibrationRowSubmission]
    missing_row_markers: list[MissingRowMarker]
```

Cross-checks performed in the submit handler (not in the model, because they require the bundle):

1. `rows` is a complete list, one entry per bundle row, with `bundle_row_index` matching position.
2. Every `between_bundle_rows` pair satisfies `0 <= N < bundle.row_count`.
3. `reviewer.user_id` from the request body matches `request.state.reviewer.user_id` (confused-deputy defence; mirrors the same check the regular-mode `/api/save` handler in `verifier/serve.py` already performs).
4. The page is in `awaiting_submissions` (no `canonical.json` present).
5. The reviewer has not already submitted (no existing `verified.<short>.json` for their `<short>`); drafts may be overwritten freely.

Any failure returns 400 with a structured error envelope and writes nothing.

## Server-side blind-review enforcement

The verifier app gates per-reviewer file access. A new middleware-ish helper checks each calibration file read against the rule:

```python
def can_read_calibration_file(
    requester_user_id: str,
    file_path: Path,
) -> bool:
    parts = file_path.relative_to(CALIBRATION_ROOT).parts
    # parts is like ("1990", "anomaly", "<stem>", "<filename>")
    page_dir = CALIBRATION_ROOT / Path(*parts[:3])
    filename = parts[3]

    if filename == "bundle.json":
        return True  # any authenticated session
    if filename == "canonical.json" or filename == "agreement.json":
        return file_path.exists()  # presence implies settled; readable to anyone
    if filename.startswith("draft."):
        return _matches_owner(filename, requester_user_id)
    if filename.startswith("verified."):
        if _matches_owner(filename, requester_user_id):
            return True
        return (page_dir / "canonical.json").exists()  # settled → others may read
    return False
```

`_matches_owner(filename, requester_user_id)` recomputes `sha256(requester_user_id)[:12]` and compares against the filename's short prefix. The check is constant-time-equivalent at this scale; no timing-attack hardening is warranted.

Denied reads return 403 with no body content (do not leak which file shape was requested). The server logs denials at INFO level with `(requester_short, file_path, reason)` — small audit trail, helps catch SPA bugs.

The `_reviewers.json` mapping table is written by the submit handler on each first-time submission for a given `user_id` — the precise write ordering, membership check, and per-file lock discipline live in the *Submit handler write ordering* section under *Merge algorithm*. The merge function never reads it; only ops/admin tooling does.

## Endpoints

New routes on the verifier FastAPI app, all under `/api/calibration/` for JSON and `/calibration/` for HTML:

| Method | Path | Purpose |
|---|---|---|
| GET | `/calibration/` | HTML — landing page, lists the reviewer's queue. |
| GET | `/calibration/<year>/<bucket>/<stem>/` | HTML — serves the calibration-mode SPA for this page. |
| GET | `/api/calibration/queue` | JSON — the reviewer's queue (see ordering below). |
| GET | `/api/calibration/<year>/<bucket>/<stem>/bundle` | JSON — `bundle.json` content (after symlink follow). |
| GET | `/api/calibration/<year>/<bucket>/<stem>/draft` | JSON — the requesting reviewer's draft, if any. |
| POST | `/api/calibration/<year>/<bucket>/<stem>/draft` | Auto-save the requesting reviewer's working state. |
| POST | `/api/calibration/<year>/<bucket>/<stem>/submit` | Promote the draft to `verified.<short>.json` atomically; runs the merge. |
| GET | `/api/calibration/<year>/<bucket>/<stem>/verified/<short>` | JSON — gated by access rule. |
| GET | `/api/calibration/<year>/<bucket>/<stem>/canonical` | JSON — present only after settlement. |
| GET | `/api/calibration/<year>/<bucket>/<stem>/agreement` | JSON — present only after settlement. |

Routes under `/calibration/` and `/api/calibration/` join the existing OIDC session middleware's gated set; an unauthenticated request 302s to `/auth/login` per the OIDC plan, with `return_to` populated.

### Queue ordering

The queue endpoint returns the reviewer's eligible pages, with `your_state ∈ {not_started, draft, submitted}` and `page_state ∈ {awaiting_submissions, settled}`. Order, highest first:

1. `your_state == "draft"` — resume your own work.
2. `your_state == "not_started"` and `submissions_so_far == target_reviewers - 1` — your submission settles the page.
3. `your_state == "not_started"` — fresh work.
4. Suppressed: pages where you've already submitted, and pages already settled.

The "near-done first" rule deliberately minimises the wall-clock to canonical materialization, which matters because the calibration tree feeds drift-gate analysis downstream and settled pages are the unit of progress.

## SPA — calibration mode

The verifier SPA gains a calibration-mode variant. The mode is selected by URL prefix: `/calibration/<year>/<bucket>/<stem>/` enters calibration mode; `/verifier/?bundle=...` stays in the regular mode. The two modes share the row-rendering and bbox-crop code paths but differ on:

| Behaviour | Regular mode | Calibration mode |
|---|---|---|
| Add row | yes (`add-row` button) | no (the button is hidden) |
| Delete row | yes | no |
| Flag spurious | no | yes (per-row toggle; clears text when engaged) |
| Insert missing-row marker | no | yes (between-row button; opens text input) |
| LML autocomplete (future) | yes | **never** |
| Read other reviewers' files | n/a | only after page settled |
| Auto-save target | `data/verifier/<stem>.verified.json` | `draft.<short>.json` in the page dir |
| Submit gesture | save = submit | distinct Submit button with confirm-to-submit |

The calibration banner is persistent across the calibration page and queue: a coloured top bar (suggested: amber) with text "Calibration · blind review · do not compare notes with other reviewers." Calibration-mode pages also show, in the page header, a compact status pill: `your_state` plus `submissions_so_far / target_reviewers`. The pill does not name other reviewers and does not include presence ("X others editing"); blind-review integrity comes before presence-UX nice-to-haves.

Confirm-to-submit gesture: clicking Submit opens a modal with the text "This submission is final. Drafts are saved automatically; submission cannot be changed." plus a required typed confirmation (e.g., "submit"). Discouraging accidental submission is the whole point — the underlying immutability is what gives the agreement metric its calibration value.

## Bootstrap

`scripts/seed_calibration_anomaly.py` is a one-shot script run locally to populate the calibration tree with the 5 named seed bundles:

```python
SEED_STEMS = [
    "1990-04apr0106-page14",
    "1990-04apr0106-page28",
    "1990-04apr0106-page29",
    "1990-04apr0106-page34",
    "1990-04apr2430-page23",
]

def year_from_stem(stem: str) -> str:
    return stem.split("-", 1)[0]  # "1990-04apr0106-page14" → "1990"

def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for stem in SEED_STEMS:
        year = year_from_stem(stem)
        page_dir = repo / "data" / "calibration" / year / "anomaly" / stem
        page_dir.mkdir(parents=True, exist_ok=True)
        bundle_link = page_dir / "bundle.json"
        if bundle_link.exists() or bundle_link.is_symlink():
            continue
        bundle_source = repo / "data" / "verifier" / f"{stem}.bundle.json"
        if not bundle_source.exists():
            raise SystemExit(f"missing source bundle: {bundle_source}")
        relative_target = Path("../../../..") / "verifier" / f"{stem}.bundle.json"
        bundle_link.symlink_to(relative_target)
```

Properties:

- **Idempotent**: re-running creates nothing new and removes nothing.
- **Validating**: refuses to create broken links if a seed bundle is missing.
- **Surgical**: touches only the seed bundles' directories. Existing single-reviewer `verified.json` files (notably `data/verifier-pulled-refresh/1990-04apr0106-page14.verified.json`, which is the only seed-set page that has ever been hand-reviewed) are **not** copied or symlinked into the calibration tree. Multi-reviewer review starts fresh. Re-using the old file as one reviewer's submission would violate every property of the protocol: it wasn't produced blind, has no stable OIDC `sub`, and would corrupt the agreement metric.

A `--dry-run` flag is included to print the operations without performing them.

`data/calibration/_reviewers.json` is **not** created by the bootstrap. It starts absent and materialises on the first submission by any reviewer — the submit-handler code path in *Submit handler write ordering* handles the "file does not exist" case as an empty starting mapping. A brand-new calibration deployment therefore ships with zero reviewers on record; the mapping populates lazily as the seed set gets picked up. The `--refresh-reviewers` flag exists to rebuild the mapping from durable `verified.*.json` files after identity-field changes on the auth server; it is not needed to seed an empty deployment.

When issue #66's per-year sampler ships, it can call into this script's helper function with a programmatically-generated stem list, replacing the inlined `SEED_STEMS` constant; that refactor is one paragraph at that future date.

## Truth derivation wiring

`scripts/derive_truth.py` gains a `--from canonical` mode. When invoked with a `canonical.json` path, it walks the canonical's rows and emits truth using the existing substring-extraction rules, with new status handling:

| Canonical row status | Truth emission |
|---|---|
| `unanimous` (raw_text), `majority` (raw_text) | Emit row truth from `raw_text` (existing rules apply). |
| `illegible` (raw_text = `__illegible__`) | Skip — subset semantics naturally accommodate. |
| `majority_spurious`, `unanimous_spurious` | Skip — row asserted not to exist. |
| `under_emit_majority` (injected) | Emit row truth from the majority suggested text. |
| `under_emit_no_text_agreement` | Skip — gap consensus but no text consensus. |

The script also gains an optional `--year YYYY` argument; when omitted, the year is parsed from the stem prefix by splitting on the first `-` and asserting the prefix matches `^\d{4}$`. If the prefix is missing or non-numeric, both the CLI and `from_canonical()` raise `CanonicalReadError` with a message that names the offending stem — the pipeline stops loudly rather than emitting truth under a wrong-year directory or silently skipping the file. Default output path becomes `tests/golden/calibration/<year>/<stem>.truth.json`.

Auto-invocation: the verifier submit handler, after a successful merge that materialises `canonical.json`, calls `derive_truth.from_canonical(canonical_path)` synchronously. Manual CLI invocation keeps the old single-reviewer path (`derive_truth.py --from verified ...`) for the regular flow.

The auto-invoked function is the same shape as the module's existing helper, exposed as a plain function so the submit handler doesn't shell out:

```python
# scripts/derive_truth.py

def from_canonical(canonical_path: Path, *, out_root: Path | None = None) -> Path:
    """Read canonical.json at `canonical_path`, emit the corresponding
    <stem>.truth.json under `out_root` (defaults to
    `<repo>/tests/golden/calibration/<year>/`), and return the written path.

    Writes atomically (write-to-temp + fsync + rename). Overwrites any
    existing truth file for the same stem — the canonical is the sole
    source of truth for calibration-derived goldens.

    Raises:
        CanonicalReadError: canonical.json is missing, malformed, or its
            schema_version does not match CALIBRATION_SCHEMA_VERSION.
        TruthWriteError: the atomic rename failed.

    Returns:
        The absolute path of the written truth file.
    """
```

The submit handler wraps the call in try/except and treats any exception as **non-fatal for the HTTP response**: the settlement is already durable on disk (canonical.json + agreement.json fsynced), and re-deriving truth is a deterministic replay from that canonical. On exception, the handler logs `truth_derivation_failed` at ERROR with `(stem, canonical_path, exc)` and returns 200 to the client with a `warnings: ["truth_derivation_deferred"]` field in the submit response so the SPA can surface a small badge. A follow-up `scripts/derive_truth.py --from canonical --replay-missing` command (added to the same script) walks `data/calibration/**/canonical.json` and re-derives truth for any missing golden, so a transient failure at submit time self-heals on the next manual sweep. The alternative — 500-ing the submit — would leave the client thinking the submission failed when the durable state (the reviewer's canonical vote) has already landed. Prioritising the durable settlement over the derived artefact matches the project's data-safety posture.

The synchronous choice (rather than background task) is deliberate: canonical.json is small (~tens of KB), the truth extraction is a straight walk over the canonical rows applying substring rules, and it completes in low-milliseconds for realistic bundle sizes (~25–60 rows). No async task infrastructure buys us anything here; a plain function call keeps the code path debuggable and the failure surface visible to the integration test that exercises the settlement round-trip.

Existing flat truth files (`tests/golden/*.truth.json`) remain untouched in place — no git move, no rename, no edits. `core.golden`'s discovery switches from a flat `glob` to `rglob("*.truth.json")` so both layouts coexist transparently: flat files at the root of `tests/golden/` and new files nested under `tests/golden/calibration/<year>/` are both found by the same discovery code. The calibration sub-layout self-documents provenance — a truth file under `tests/golden/calibration/<year>/` was produced by the multi-reviewer protocol and carries the agreement-metric trail in the sibling `agreement.json`. Coexistence is verified explicitly in the acceptance criteria below by running tests against both layouts in the same pytest invocation.

## Schemas (final)

### `verified.<short>.json` (immutable submission)

```jsonc
{
  "schema_version": 1,
  "stem": "1990-04apr0106-page14",
  "reviewer": {
    "user_id": "...",            // Better Auth user.id (OIDC `sub` claim)
    "username": "...",
    "real_name": "...",
    "dj_name": "...",
    "verified_at": "2026-06-12T14:33:01Z"
  },
  "submitted_at": "2026-06-12T14:33:01Z",
  "rows": [
    {
      "bundle_row_index": 0,
      "edited_text": "BEATLES - HELP",
      "type_raw": "H",
      "notes": null,
      "spurious_flag": false
    },
    ...
  ],
  "missing_row_markers": [
    {
      "between_bundle_rows": [5, 6],
      "suggested_text": "PIXIES - DEBASER",
      "type_raw": "M",
      "notes": null
    }
  ]
}
```

### `draft.<short>.json`

Same shape as `verified.<short>.json` but mutable and owner-readable only. The `submitted_at` field is optional in drafts.

### `canonical.json` (post-settlement, immutable)

```jsonc
{
  "schema_version": 1,
  "stem": "1990-04apr0106-page14",
  "settled_at": "2026-06-12T14:35:11Z",
  "target_reviewers": 3,
  "rows": [
    {
      "canonical_row_index": 0,
      "bundle_row_index": 0,
      "inserted_between_bundle_rows": null,
      "raw_text": "BEATLES - HELP",
      "type_raw": "H",
      "notes": null,
      "confidence": "high",
      "verification": {
        "status": "majority",
        "raw_text_status": "majority",
        "raw_text_dissents": [
          { "reviewer_short": "a3b9e7c41f02", "raw_text": "BEATLES - HELO" }
        ],
        "type_raw_status": "unanimous",
        "type_raw_dissents": [],
        "spurious_flag_status": "unanimous_keep",
        "spurious_flag_votes": { "keep": 3, "spurious": 0 },
        "notes_values": { "null": 3 },
        "reviewer_shorts": ["a3b9e7c41f02", "9d27e6b5fa18", "f00b421ace30"]
      }
    },
    ...
  ],
  "missing_row_reports": [
    {
      "between_bundle_rows": [12, 13],
      "reporting_reviewer_shorts": ["a3b9e7c41f02"],
      "suggested_texts": ["LOU REED - PERFECT DAY"]
    }
  ]
}
```

The top-level `verification.status` is the worst-of across `raw_text_status`, `type_raw_status`, and `spurious_flag_status`. The `confidence` is taken from the majority `raw_text` reviewer.

### `agreement.json` (post-settlement, immutable)

```jsonc
{
  "schema_version": 1,
  "stem": "1990-04apr0106-page14",
  "year": "1990",
  "bucket": "anomaly",
  "target_reviewers": 3,
  "submissions": [
    { "reviewer_short": "a3b9e7c41f02", "submitted_at": "..." },
    { "reviewer_short": "9d27e6b5fa18", "submitted_at": "..." },
    { "reviewer_short": "f00b421ace30", "submitted_at": "..." }
  ],
  "row_status_histogram": {
    "unanimous": 18,
    "majority": 4,
    "illegible": 1,
    "spurious_majority": 2
  },
  "pair_concordance": [
    { "a": "a3b9e7c41f02", "b": "9d27e6b5fa18",
      "raw_text_agree_rate": 0.92, "type_raw_agree_rate": 1.0 },
    ...
  ]
}
```

No year-level rollup lives in `agreement.json`. A separate `scripts/calibration_report.py` walks `data/calibration/**/agreement.json` and prints per-year tables; that script is a small post-hoc reader and is not in the critical path of any settlement write, avoiding any write-write contention as multiple pages settle.

### `_reviewers.json` (ops-only mapping)

```jsonc
{
  "a3b9e7c41f02": { "user_id": "...", "username": "...", "real_name": "...", "dj_name": "..." },
  "9d27e6b5fa18": { ... }
}
```

Written by the submit handler on first-time-submission for a given `user_id`. Never read by the merge handler or the SPA — the no-read rule is enforced by code review, not a runtime gate (the file is on the same filesystem and any module *could* open it, but doesn't). Useful for translating short prefixes back to humans when reviewing agreement data via `scripts/calibration_report.py`. The repo's `.gitignore` already excludes everything under `data/` (`data/` rule at the top of the file), so no `.gitignore` change is needed; the mapping never enters version control by virtue of living under the blanket data-tree exclusion. Read access at runtime is unrestricted to authenticated sessions; the file does not carry secrets and the contents are already discoverable to any reviewer who knows another reviewer's name (they sit next to each other at the station).

**Append-only data-safety contract.** Entries are added on first-time submission for a given `user_id`; the submit handler never updates an existing entry's `username`, `real_name`, or `dj_name`, and never deletes one. If any of those fields change on the auth server, the local entry goes stale silently; the operator refreshes it by running `scripts/seed_calibration_anomaly.py --refresh-reviewers` (a small flag added to the bootstrap script that walks current `verified.*.json` files and rewrites `_reviewers.json` from their embedded `verified_by` blocks). Acceptable trade-off because the mapping is auditing-only and identity-field changes at WXYC are rare. This is consistent with the project-level data-safety rule: "Never delete, reset, or overwrite successfully collected data without explicit user approval."

## Schema versioning

`CALIBRATION_SCHEMA_VERSION = 1` is declared as a module constant in `core/schema.py` (the same module that hosts the calibration Pydantic models above). The four file shapes (`verified.<short>.json`, `draft.<short>.json`, `canonical.json`, `agreement.json`) each carry a top-level `schema_version` field that the verifier app validates on every read. A mismatched version returns a clear error banner ("schema version mismatch; this file is incompatible with the running app") rather than silently mis-parsing. Bumping the version is a deliberate act; all four shapes bump together so the cross-shape compatibility matrix stays trivially 1-to-1.

The bundle's existing `SCHEMA_VERSION` in `scripts/make_verifier_bundle.py` is **not** subsumed by `CALIBRATION_SCHEMA_VERSION`. The two version counters cover different artifacts and evolve independently.

## Testing

Per the project convention (CLAUDE.md: "TDD by default"), each module is written test-first. Concretely:

1. **`core/schema.py` calibration additions** — write Pydantic-validation cases in `tests/unit/test_calibration_schema.py` first (e.g., `test_missing_row_marker_requires_adjacent_indices`, `test_calibration_submission_round_trip`), confirm red, then add the models.
2. **`core/calibration_compare.py`** — write the comparator property/parametrize cases in `tests/unit/test_calibration_compare.py` first, confirm red, then implement `_normalize_raw_text`, `_normalize_type_raw`, and `rows_agree`.
3. **`core/calibration_consensus.py`** — write a failing fixture-driven case in `tests/unit/test_calibration_consensus.py` (e.g., `test_unanimous_all_agree`) first, then implement the merge; expand fixture-by-fixture until every scenario passes.
4. **Verifier endpoints in `verifier/serve.py`** — TestClient-based tests for each new route first (e.g., `test_submit_writes_verified_atomically`, `test_canonical_read_denied_before_settlement`), then implement.

Three test tiers cover the implementation surface.

### Synthetic per-reviewer fixtures

Under `tests/fixtures/calibration/<scenario_stem>/`, one directory per scenario:

- `unanimous_all_agree/` — three identical submissions; expect `canonical.status = unanimous` everywhere.
- `majority_n3/` — two agreeing submissions and one dissenter; expect `status = majority` with the dissent captured.
- `illegible_111/` — three different `raw_text` readings on the same row; expect `__illegible__` sentinel with three dissents.
- `n2_disagrees_bumps_to_3/` — two reviewers disagree; merge returns `target_reviewers=3` and no canonical until a third submission lands.
- `spurious_majority/` — two reviewers flag a row spurious; expect canonical row's text = null and `majority_spurious` status.
- `missing_row_majority/` — two reviewers insert the same missing-row marker; expect the row injected with `under_emit_majority` status.
- `missing_row_minority/` — only one reviewer reports a gap; expect no injection, sidecar `missing_row_reports` populated.
- `type_raw_doodle_cluster/` — reviewers write `""`, `"?"`, `"doodle"` for the same row; expect agreement under the `_unknown` collapse.

Each fixture directory contains the per-reviewer `verified.*.json` files plus `expected_canonical.json` and `expected_agreement.json`. The runner is one `@pytest.mark.parametrize` decorator over the fixture-root's immediate children, with each directory's name used as the test ID (so a failing case shows up as `test_merge_fixture[unanimous_all_agree]` rather than as an opaque integer index). Concrete shape:

```python
# tests/unit/test_calibration_consensus.py
from pathlib import Path
import json
import pytest
from core.calibration_consensus import merge
from core.schema import CalibrationSubmission

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "calibration"
SCENARIOS = sorted(p for p in FIXTURE_ROOT.iterdir() if p.is_dir())

def _load_submissions(scenario_dir: Path) -> list[CalibrationSubmission]:
    return [
        CalibrationSubmission.model_validate_json(p.read_text())
        for p in sorted(scenario_dir.glob("verified.*.json"))
    ]

def _load_bundle(scenario_dir: Path) -> dict:
    return json.loads((scenario_dir / "bundle.json").read_text())

@pytest.mark.parametrize("scenario_dir", SCENARIOS, ids=lambda p: p.name)
def test_merge_fixture(scenario_dir: Path) -> None:
    bundle = _load_bundle(scenario_dir)
    submissions = _load_submissions(scenario_dir)
    canonical, agreement, _target = merge(bundle, submissions)
    expected_canonical = json.loads((scenario_dir / "expected_canonical.json").read_text())
    expected_agreement = json.loads((scenario_dir / "expected_agreement.json").read_text())
    assert canonical.model_dump(mode="json") == expected_canonical
    assert agreement.model_dump(mode="json") == expected_agreement
```

Two collection-time invariants land alongside the parametrized test so a malformed fixture directory fails fast at collection rather than mid-run: `assert SCENARIOS, f"no fixture directories under {FIXTURE_ROOT}"` guards accidental empty runs, and a session-scoped fixture-validator (`test_fixture_files_are_complete`) asserts each directory contains at least two `verified.*.json`, exactly one `bundle.json`, and both `expected_*.json` files — surfacing a missing-file typo as a plain collection error before pytest tries to run the merge with an incomplete input. New fixtures land by adding a directory: no test-code change is required for the happy path; the validator will complain loudly if the file set is wrong.

This is a new convention for the project — existing tests use inline `@pytest.mark.parametrize` tables (e.g., `tests/unit/test_page_layout.py`, `tests/unit/test_make_verifier_bundle.py`). Calibration merge is the first case where per-scenario fixtures sprawl across multiple files (3 reviewer submissions + 1 bundle + 2 expected outputs per scenario), making a directory-per-scenario layout clearer than inline literals. The CLAUDE.md Testing section gains a one-line note: "Multi-file fixture scenarios live under `tests/fixtures/<module>/<scenario>/`; the per-module test parametrizes over that directory with `ids=lambda p: p.name`. Simple cases keep using inline `parametrize`."

### Real-bundle replay

A single integration test at `tests/integration/test_calibration_consensus.py` loads `data/verifier/1990-04apr0106-page28.bundle.json` (the smallest of the 5 seed bundles by row count), constructs three hand-built `verified.*.json` files that introduce a mix of agreement and disagreement, and asserts the merge produces the expected canonical and agreement outputs. This exercises the actual bundle shape including `row_bbox` round-trips and quadrant assignment for missing-row markers.

### Property tests on the comparator

`tests/unit/test_calibration_compare.py` covers `_normalize_raw_text` and `rows_agree`:

- Idempotency: `_normalize_raw_text(_normalize_raw_text(x)) == _normalize_raw_text(x)` for any string.
- NFKC stability: composed and decomposed forms of the same character compare equal.
- Case-fold: `BEATLES - HELP` equals `beatles - help`.
- Dash variants: `Beatles - Help`, `Beatles – Help`, `Beatles — Help` all agree.
- Ampersand expansion: `Simon & Garfunkel` agrees with `Simon and Garfunkel`.
- Punctuation strip: `R.E.M. - Losing My Religion` agrees with `REM - Losing My Religion`.
- Slash preserved: `AC/DC` does not agree with `ACDC` (genuine disagreement preserved).
- Separator preserved: `Beatles - Help` does not agree with `Beatles Help` (missing separator is a real disagreement).
- Whitespace collapse: `"BEATLES  -  HELP "` agrees with `"Beatles - Help"`.

Hypothesis is not introduced as a new dependency in this plan; the tests are hand-written parametrize tables with about 30 input pairs. If a property-testing framework becomes warranted later (e.g., when the LML Tier-2 hook lands), Hypothesis can be added then.

## PR plan

Two pull requests, sequenced.

### PR 1 — Backend

Worktree: `flowsheet-digitization-multi-reviewer-calibration-backend` (sibling to the main repo checkout, matching the repo convention seen in `flowsheet-digitization-oidc-auth` and `flowsheet-digitization-fix-review`). Branch: `feat/calibration-multi-reviewer-backend`, branched from `origin/main`.

Contents:

- `core/schema.py` — extended with `CalibrationRowSubmission`, `MissingRowMarker`, `CalibrationSubmission`, the canonical and agreement record shapes, and `CALIBRATION_SCHEMA_VERSION = 1`. No existing models are altered; this is purely additive and reuses the existing `VerifiedBy` model unchanged.
- `core/calibration_compare.py` — `_normalize_raw_text`, `_normalize_type_raw`, `rows_agree`, with the Tier-2 `canonicalize` argument as a parameter that no caller passes today. Coexists with `core/calibration.py` (the model-comparison harness); the names refer to different kinds of calibration (reviewer-text comparison vs model-vs-truth scoring) but the file-naming pair is intentional and called out in CLAUDE.md.
- `core/calibration_consensus.py` — the pure merge function plus its helpers. Named `_consensus` rather than `_merge` to disambiguate from the existing `core/calibration.py` model-comparison harness; this module's job is consensus over reviewer submissions, not score aggregation across model runs.
- `verifier/serve.py` — new endpoints, the access-control gate, and a static mount for `data/calibration/`. The middleware in `main` today is `_SessionCookieMiddleware` (installed when `OIDC_ENABLED`), which gates every path **not** in the small `_PUBLIC_PATHS` allowlist (`{"/auth/login", "/auth/callback", "/auth/logout", "/api/version"}`). New calibration routes register under `/api/calibration/` and `/calibration/` and are therefore gated by default; **`_PUBLIC_PATHS` is not modified**. No middleware-shape changes required. The plan does not add support for BasicAuth-only mode (the fallback middleware `_BasicAuthMiddleware`) — calibration requires OIDC because the confused-deputy cross-check in *Submission validation* compares `reviewer.user_id` from the POST body against `request.state.reviewer.user_id`, and only the OIDC session middleware populates `request.state.reviewer` with a `ReviewerSession` carrying `user_id`. Practically, deployments that run calibration must have `OIDC_ENABLED=1`; the seed-set operator laptop and the Railway deploy already do.
- `scripts/seed_calibration_anomaly.py` — the bootstrap script.
- `scripts/derive_truth.py` — `--from canonical` support and the automatic-invocation hook.
- `core/golden.py` — `rglob` switch for truth discovery.
- `CLAUDE.md` — new module entries; a "Calibration mode" subsection after the Phase-1/Phase-2 table.
- `verifier/README.md` — new "Calibration mode" section: file layout, status semantics, blind-review constraints.
- `tests/golden/README.md` — note on the new `calibration/<year>/` sub-layout and the no-edit-by-hand policy for calibration-derived truths.
- Tests as described in *Testing*.

No `.gitignore` change is needed: the existing `data/` rule (top of `.gitignore`) already excludes everything under `data/calibration/`, including `_reviewers.json`. Adding an explicit `data/calibration/_reviewers.json` entry would be redundant and risks reading as if `_reviewers.json` somehow needs a special carve-out it doesn't.

Aim for ~600–800 lines net, well under the project's 1000-line PR target.

### PR 2 — SPA

Worktree: `flowsheet-digitization-multi-reviewer-calibration-spa` (sibling to the main repo checkout). Branch: `feat/calibration-multi-reviewer-spa`, branched from `feat/calibration-multi-reviewer-backend` and rebased on it; when the backend merges, this rebases onto `main` for review. Operationally: after the backend PR merges, run `git fetch origin && git rebase origin/main` in the SPA worktree, then `git push --force-with-lease` to update the open SPA PR — preserving linear history and reflecting only the SPA-specific diff for reviewers.

Contents:

- `verifier/app.js` — calibration mode switch driven by URL path; flag-spurious toggle, missing-row-marker affordance, confirm-to-submit modal, calibration banner, queue page.
- `verifier/index.html` — minor template additions for the calibration banner and the missing-row-marker controls.
- `verifier/styles.css` — banner styling, calibration-mode visual distinctions.
- `verifier/README.md` — keyboard shortcuts, a screenshot of the calibration view.

Aim for ~500–700 lines net.

### Worktree convention note

Worktrees in this repo are **siblings** of the main checkout, named `<repo>-<slug>` (see existing examples `flowsheet-digitization-oidc-auth` and `flowsheet-digitization-fix-review`). Both PR worktrees above follow this convention; do not nest them under `.worktrees/` or any other subdirectory of the main checkout.

## GitHub issue + PR sequencing

One umbrella tracking issue: **Multi-reviewer calibration for the anomaly bucket**. Body is this plan (or a link to it once merged). Two sub-issues:

- **Backend: calibration merge + storage + bootstrap** — referenced by PR 1 with `Closes #<n>`.
- **SPA: calibration blind-review mode + queue UI** — referenced by PR 2 with `Closes #<n>`.

Sub-issues use GitHub's native Sub-issues relationship on the umbrella so completion tracking is structured rather than prose. Cross-references in bodies: each sub-issue mentions the umbrella; PR bodies use `Closes #<sub-issue>` so each PR auto-closes only its own sub-issue (avoids the mega-issue-closes-on-first-merge footgun).

Order:

1. Create umbrella + backend sub-issue, then open PR 1.
2. When PR 1 is in review, open the SPA sub-issue + PR 2.
3. PRs cross-reference each other ("backend prerequisite: #N"; "SPA follow-up: #M").

## Documentation updates

Backend PR — `CLAUDE.md` additions, exact one-liners to append to the Architecture module list (matching the existing one-line-per-module style):

```
core/
  calibration_compare.py         Pure normalization + agreement comparator
                                 for the multi-reviewer calibration flow.
                                 `_normalize_raw_text`, `_normalize_type_raw`,
                                 and `rows_agree(a, b, *, canonicalize=None)`
                                 where the Tier-2 canonicalize hook is
                                 reserved for future LML integration.
                                 Sibling of `calibration.py` (the existing
                                 model-comparison harness); distinct
                                 concerns, deliberately not folded together.
  calibration_consensus.py       Pure-function merge over per-reviewer
                                 submissions for a calibration page.
                                 Produces canonical + agreement records
                                 plus the live target_reviewers count.
                                 Single writer of canonical.json /
                                 agreement.json files on disk. Named
                                 `_consensus` (not `_merge`) to disambiguate
                                 from `calibration.py`'s model-vs-truth
                                 scoring.
  schema.py                      (extended) — calibration submission and
                                 canonical/agreement shapes added alongside
                                 existing PageResult / Entry / Quadrant /
                                 VerifiedBy. The `reviewer` block on
                                 `CalibrationSubmission` reuses the existing
                                 VerifiedBy unchanged.
                                 `CALIBRATION_SCHEMA_VERSION = 1` is the
                                 module constant for the four new file
                                 shapes (verified.<short>.json,
                                 draft.<short>.json, canonical.json,
                                 agreement.json).

scripts/
  seed_calibration_anomaly.py    One-shot bootstrap for the 5 named seed
                                 bundles. Creates data/calibration/<year>/
                                 anomaly/<stem>/ directories and symlinks
                                 each bundle.json from the regular verifier
                                 tree. Idempotent; safe to re-run.
                                 `--refresh-reviewers` rebuilds
                                 _reviewers.json from current verified.*.json
                                 files when reviewer identity fields change
                                 on the auth server.
```

A new "Calibration mode" subsection placed after the Phase-1/Phase-2 table covers, in this order: (1) the blind-review protocol summary, (2) the four on-disk file shapes, (3) where the merge runs (the submit handler), (4) the LML-blind-forever rule for the calibration SPA, and (5) the worst-of escalation rule.

- `verifier/README.md` — new "Calibration mode" section with the file layout, state semantics, and reviewer-facing guidance.
- `tests/golden/README.md` — brief note on the new sub-layout and the no-hand-edit policy for calibration-derived truths.
- Root `README.md` — one-paragraph pointer with a link to the verifier README's calibration section. No detailed content at the root.

SPA PR:

- `verifier/README.md` — keyboard shortcuts section refreshed; one screenshot of the calibration view with the banner visible.

## What this plan does not solve

- It does not retroactively multi-review any page that already has a single-reviewer `verified.json` in the regular tree. Page 14 is the only seed-set page in that state today; it will be re-reviewed from scratch in calibration.
- It does not address operator-side tooling for reviewing the agreement data (no leaderboard, no notification when a page needs a third reviewer, no admin dashboard). The queue endpoint and `calibration_report.py` are sufficient for one operator to drive a small reviewer pool by hand. Heavier tooling waits until the reviewer pool justifies it.
- It does not provision OIDC reviewer identity — that work is already in `main` (`core/auth.py`, `_SessionCookieMiddleware` in `verifier/serve.py`, `verified_by` on `PageResult`). If a future change reshapes `ReviewerSession` or `VerifiedBy`, the `reviewer: VerifiedBy` field on `CalibrationSubmission` in `core/schema.py` is the single touch point this plan adds.
- It does not change `core/jobs.py`. The calibration flow is parallel to the per-page jobs state machine; the OIDC plan's `jobs.reviewer_id` column remains the single-reviewer record for the regular flow and is unused by calibration.

## Acceptance criteria

Backend PR:

- [ ] `core/schema.py` calibration additions, `core/calibration_compare.py`, and `core/calibration_consensus.py` exist with full type coverage; mypy clean.
- [ ] `scripts/seed_calibration_anomaly.py` runs idempotently; after running, the 5 seed page directories exist with valid relative symlinks; running a second time is a no-op.
- [ ] `verifier/serve.py` exposes the new endpoints, gates access per the rules above, and serves `data/calibration/` as a static mount, all under the existing `_SessionCookieMiddleware` gate (no changes to `_PUBLIC_PATHS` and no middleware-shape changes).
- [ ] Submit handler atomically writes `verified.<short>.json` (write-to-temp + fsync + rename); runs the merge; writes `canonical.json` + `agreement.json` only when `target_reviewers` is met and all gating decisions are resolved.
- [ ] All comparator property tests pass.
- [ ] All synthetic-fixture merge tests pass.
- [ ] Real-bundle replay integration test passes.
- [ ] `derive_truth.py` `--from canonical` produces a truth file matching a hand-checked expectation for one settled fixture.
- [ ] Golden-truth coexistence: a single `pytest` invocation passes tests that load both a flat truth file (existing `tests/golden/<stem>.truth.json`) and a calibration-layout truth file (`tests/golden/calibration/1990/<stem>.truth.json`) via `core.golden`'s discovery.
- [ ] `CLAUDE.md` and `verifier/README.md` carry calibration sections; the Architecture module list contains entries for `core/calibration_compare.py`, `core/calibration_consensus.py`, the `core/schema.py` calibration additions, and `scripts/seed_calibration_anomaly.py`. The CLAUDE.md text explicitly calls out the coexistence of `core/calibration.py` (existing, model-vs-truth scoring) and the new `calibration_compare.py` + `calibration_consensus.py` pair (reviewer-text agreement and submission consensus) so future readers don't confuse the two concerns.

SPA PR:

- [ ] Calibration mode is selected by URL prefix; banner is visible; flag-spurious and missing-row-marker affordances function.
- [ ] Add-row and delete-row buttons are not present in calibration mode.
- [ ] Confirm-to-submit modal requires typed confirmation before POSTing.
- [ ] Queue page lists eligible pages with `near-done-first` ordering; auto-redirect after submit lands on the next eligible page.
- [ ] One screenshot in `verifier/README.md` shows the calibration view.

## Related

- `plans/oidc-auth.md` — reviewer identity dependency.
- `project_bbox_sweep_result.md` — the 5 named seed bundles.
- `project_empirical_failure_modes_n19.md` — the n=19 baseline metrics that motivate this work.
- `project_oddity_doodle_entry.md`, `project_oddity_overlay_pattern.md` — known oddity patterns the comparator's doodle-cluster normalization is informed by.
- Issue #66 — per-year drift sampling; consumes this plan's anomaly-bucket infrastructure when it ships.
- Issue #62 — Phase 2 reconciliation thresholds; the eventual home of the Tier-2 LML canonicalizer that plugs into the comparator's `canonicalize` argument.
