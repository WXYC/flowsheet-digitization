"""Pure-function consensus over reviewer submissions.

This is the merge step of the multi-reviewer calibration flow — takes the
per-reviewer `verified.<short>.json` submissions for one page, plus the
bundle row count, and produces the page's settled `canonical.json` and
`agreement.json`. See plans/multi-reviewer-calibration.md §Merge algorithm.

Deliberately not to be confused with `core.calibration` — that module
scores extraction models against goldens. This module's job is consensus
over reviewer submissions. Distinct concerns; the file-naming coincidence
is intentional and called out in CLAUDE.md.

The function is pure: no filesystem, no wall-clock. The caller injects
`settled_at`. The caller also owns the atomic write-to-temp + rename
that lands `canonical.json` and `agreement.json` on disk.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from itertools import combinations

from core.calibration_compare import (
    _normalize_raw_text,
    _normalize_type_raw,
    rows_agree,
)
from core.schema import (
    CalibrationAgreement,
    CalibrationCanonical,
    CalibrationRowSubmission,
    CalibrationSubmission,
    CanonicalRow,
    Confidence,
    MissingRowMarker,
    MissingRowReport,
    PairConcordance,
    RowDissent,
    RowStatus,
    RowVerification,
    SpuriousFlagStatus,
    SubmissionRecord,
)

_ILLEGIBLE = "__illegible__"
_TYPE_UNKNOWN_SENTINEL = "_unknown"


def _short(user_id: str) -> str:
    """Deterministic 12-hex-char short prefix of a reviewer's OIDC subject."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Per-row helpers
# --------------------------------------------------------------------------- #


def _spurious_status(spurious_votes: list[bool]) -> tuple[SpuriousFlagStatus | None, bool]:
    """Return (status, needs_more_reviewers).

    None status means "undecided — split at N=2, escalate to N=3."
    """
    keep = sum(1 for s in spurious_votes if not s)
    spurious = sum(1 for s in spurious_votes if s)
    n = len(spurious_votes)
    if n == 2 and keep == 1 and spurious == 1:
        return None, True
    if n <= 2:
        if spurious == 0:
            return "unanimous_keep", False
        if keep == 0:
            return "unanimous_spurious", False
        # Can't reach — n=2, 1-1 handled above; n=1 caller shouldn't call.
        return None, True
    # n >= 3
    if spurious == 0:
        return "unanimous_keep", False
    if keep == 0:
        return "unanimous_spurious", False
    if spurious > keep:
        return "majority_spurious", False
    return "majority_keep", False


def _text_consensus(
    texts: list[tuple[str, str]],  # (reviewer_short, text)
) -> tuple[str | None, str, list[RowDissent]]:
    """Compute majority text among keepers.

    Returns (chosen_text, status, dissents). status is one of
    "unanimous" | "majority" | "illegible" | "insufficient" — insufficient
    means N=2 with a split.
    """
    if not texts:
        # No keepers at all → caller (spurious_majority path) handles.
        return None, "illegible", []
    n = len(texts)
    # Group by normalized text.
    groups: dict[str, list[tuple[str, str]]] = {}
    for reviewer_short, text in texts:
        key = _normalize_raw_text(text)
        groups.setdefault(key, []).append((reviewer_short, text))
    largest_key = max(groups, key=lambda k: len(groups[k]))
    largest = groups[largest_key]
    if len(groups) == 1:
        return largest[0][1], "unanimous", []
    if len(largest) >= 2 and len(largest) > n // 2:
        # Strict majority (works for both N=2 all-agree — handled above —
        # and N=3 with 2 in the largest group).
        chosen = largest[0][1]
        dissents = [
            RowDissent(reviewer_short=rs, value=t)
            for group_key, group in groups.items()
            if group_key != largest_key
            for rs, t in group
        ]
        return chosen, "majority", dissents
    if n == 2:
        # 1-1 split at N=2 → not enough reviewers, escalate.
        return None, "insufficient", []
    # N=3 with no majority (1-1-1) → illegible.
    dissents = [
        RowDissent(reviewer_short=rs, value=t) for group in groups.values() for rs, t in group
    ]
    return _ILLEGIBLE, "illegible", dissents


def _type_raw_consensus(
    values: list[tuple[str, str | None]],  # (reviewer_short, raw_type_raw)
) -> tuple[str | None, str, list[RowDissent]]:
    """Same shape as `_text_consensus` but over `type_raw` with doodle-cluster fold.

    On agreement, returns the FIRST verbatim value from the winning group
    (preserving the reviewer's original casing / spelling), NOT the
    normalized key. The exception is the doodle cluster: when the winning
    key is `_unknown`, returns the sentinel `_unknown` string so downstream
    consumers can branch cleanly on it.
    """
    if not values:
        return _TYPE_UNKNOWN_SENTINEL, "unanimous", []
    n = len(values)
    groups: dict[str, list[tuple[str, str | None]]] = {}
    for reviewer_short, raw in values:
        key = _normalize_type_raw(raw)
        groups.setdefault(key, []).append((reviewer_short, raw))
    largest_key = max(groups, key=lambda k: len(groups[k]))

    def _pick_verbatim(key: str, group: list[tuple[str, str | None]]) -> str | None:
        if key == _TYPE_UNKNOWN_SENTINEL:
            return _TYPE_UNKNOWN_SENTINEL
        for _, raw in group:
            if raw is not None:
                return raw
        return None

    if len(groups) == 1:
        chosen = _pick_verbatim(largest_key, groups[largest_key])
        return chosen, "unanimous", []
    if len(groups[largest_key]) >= 2 and len(groups[largest_key]) > n // 2:
        chosen = _pick_verbatim(largest_key, groups[largest_key])
        dissents = [
            RowDissent(reviewer_short=rs, value=raw if raw is not None else "")
            for key, group in groups.items()
            if key != largest_key
            for rs, raw in group
        ]
        return chosen, "majority", dissents
    if n == 2:
        return None, "insufficient", []
    dissents = [
        RowDissent(reviewer_short=rs, value=raw if raw is not None else "")
        for group in groups.values()
        for rs, raw in group
    ]
    return _TYPE_UNKNOWN_SENTINEL, "unknown", dissents


def _worst_of_row_status(
    raw_text_status: str,
    type_raw_status: str,
    spurious_flag_status: SpuriousFlagStatus,
) -> RowStatus:
    """The row's headline status is the worst-of across gating fields.

    Ordered best → worst: unanimous, majority, majority_keep, illegible.
    Spurious-driven statuses override the text side.
    """
    if spurious_flag_status == "unanimous_spurious":
        return "unanimous_spurious"
    if spurious_flag_status == "majority_spurious":
        return "majority_spurious"
    # Otherwise text side dominates.
    if raw_text_status == "illegible":
        return "illegible"
    if raw_text_status == "majority" or type_raw_status == "majority":
        return "majority"
    if spurious_flag_status == "majority_keep":
        return "majority"
    return "unanimous"


# --------------------------------------------------------------------------- #
# Missing-row markers
# --------------------------------------------------------------------------- #


def _collect_markers_by_gap(
    submissions: list[CalibrationSubmission],
) -> dict[tuple[int, int], list[tuple[str, MissingRowMarker]]]:
    by_gap: dict[tuple[int, int], list[tuple[str, MissingRowMarker]]] = {}
    for sub in submissions:
        rs = _short(sub.reviewer.user_id)
        for marker in sub.missing_row_markers:
            by_gap.setdefault(marker.between_bundle_rows, []).append((rs, marker))
    return by_gap


def _classify_gap(
    reporters: list[tuple[str, MissingRowMarker]],
    total_reviewers: int,
) -> str:
    """Return one of: 'under_emit_majority', 'under_emit_no_text_agreement',
    'minority', 'insufficient' (N=2 with 1-1 split)."""
    n_reporters = len(reporters)
    threshold = total_reviewers // 2 + 1  # strict majority
    if n_reporters < threshold:
        # Minority in "no gap" direction — but N=2 split needs escalate.
        if total_reviewers == 2 and n_reporters == 1:
            return "insufficient"
        return "minority"
    # Gap majority present. Now check text.
    texts = [(rs, m.suggested_text) for rs, m in reporters]
    _, text_status, _ = _text_consensus(texts)
    if text_status == "unanimous" or text_status == "majority":
        return "under_emit_majority"
    if text_status == "insufficient":
        return "insufficient"
    return "under_emit_no_text_agreement"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def merge(
    *,
    stem: str,
    year: str,
    bucket: str,
    bundle_row_count: int,
    submissions: list[CalibrationSubmission],
    settled_at: datetime,
) -> tuple[CalibrationCanonical | None, CalibrationAgreement | None, int]:
    """Fold per-reviewer submissions into (canonical, agreement, target_reviewers).

    Returns (None, None, target) while the page is still awaiting more
    submissions. Returns settled values once every gating decision has
    resolved.
    """
    target = 2
    n = len(submissions)
    if n == 0:
        return None, None, target

    reviewer_shorts = [_short(sub.reviewer.user_id) for sub in submissions]

    # Per-row analysis.
    row_analyses: list[dict] = []
    for i in range(bundle_row_count):
        analysis = _analyze_row(i, submissions, reviewer_shorts)
        row_analyses.append(analysis)

    # Missing-row gaps.
    markers_by_gap = _collect_markers_by_gap(submissions)
    gap_classes: dict[tuple[int, int], str] = {}
    for gap, reporters in markers_by_gap.items():
        gap_classes[gap] = _classify_gap(reporters, n)

    # Determine target_reviewers: 3 if there was any dissent that required
    # (or would have required) escalation past N=2.
    has_dissent = False
    for analysis in row_analyses:
        # A row triggered escalation iff any of its gating fields is not
        # "unanimous" (or _unanimous_spurious_, which is a full-consensus
        # spurious vote — no dissent).
        text_status = analysis["text_status"]
        type_status = analysis["type_status"]
        spurious_status = analysis["spurious_status"]
        if text_status not in ("unanimous",):
            has_dissent = True
        if type_status not in ("unanimous",):
            has_dissent = True
        if spurious_status is None:
            # 1-1 split at N=2; the third reviewer either landed or is
            # still pending.
            has_dissent = True
        elif spurious_status in ("majority_keep", "majority_spurious"):
            has_dissent = True
    for cls in gap_classes.values():
        if cls == "insufficient":
            has_dissent = True
        elif cls == "under_emit_no_text_agreement":
            has_dissent = True
        elif cls == "under_emit_majority":
            # Injection-with-consensus at N=2 is unanimous among reporters;
            # at N=3 it means only the majority reported. If exactly one
            # reviewer did NOT report and >=2 did, that's dissent.
            pass  # handled by minority classification below
        elif cls == "minority":
            has_dissent = True
    if has_dissent:
        target = 3

    if n < target:
        return None, None, target

    # Build canonical rows.
    canonical_rows: list[CanonicalRow] = []
    canonical_index = 0
    for i, analysis in enumerate(row_analyses):
        canonical_rows.append(_build_canonical_row(canonical_index, i, analysis))
        canonical_index += 1
        # Any qualifying gap sitting after this bundle row?
        next_gap = (i, i + 1)
        if next_gap in gap_classes:
            cls = gap_classes[next_gap]
            if cls in ("under_emit_majority", "under_emit_no_text_agreement"):
                injection = _build_injection(
                    canonical_index, next_gap, markers_by_gap[next_gap], cls
                )
                canonical_rows.append(injection)
                canonical_index += 1

    # Minority-reported gaps: recorded but not injected.
    minority_reports: list[MissingRowReport] = []
    for gap, reporters in markers_by_gap.items():
        if gap_classes.get(gap) != "minority":
            continue
        minority_reports.append(
            MissingRowReport(
                between_bundle_rows=gap,
                reporting_reviewer_shorts=[rs for rs, _ in reporters],
                suggested_texts=[m.suggested_text for _, m in reporters],
            )
        )

    canonical = CalibrationCanonical(
        schema_version=1,
        stem=stem,
        settled_at=settled_at,
        target_reviewers=target,
        rows=canonical_rows,
        missing_row_reports=minority_reports,
    )

    agreement = _build_agreement(
        stem=stem,
        year=year,
        bucket=bucket,
        target=target,
        submissions=submissions,
        canonical_rows=canonical_rows,
        bundle_row_count=bundle_row_count,
    )
    return canonical, agreement, target


# --------------------------------------------------------------------------- #
# Row analysis + canonical row construction
# --------------------------------------------------------------------------- #


def _analyze_row(
    idx: int,
    submissions: list[CalibrationSubmission],
    reviewer_shorts: list[str],
) -> dict:
    """Analyse one bundle row across all submissions.

    Every submission has one entry per bundle row (validated upstream).
    """
    per_row: list[tuple[str, CalibrationRowSubmission]] = []
    for rs, sub in zip(reviewer_shorts, submissions, strict=True):
        entry = next((r for r in sub.rows if r.bundle_row_index == idx), None)
        if entry is None:
            # Should not happen once the submit handler validates completeness,
            # but tolerate it defensively: treat as if the reviewer said nothing.
            continue
        per_row.append((rs, entry))

    spurious_votes = [entry.spurious_flag for _, entry in per_row]
    spurious_status, spurious_needs_more = _spurious_status(spurious_votes)

    # Text side is computed over KEEPERS only.
    keeper_texts: list[tuple[str, str]] = [
        (rs, entry.edited_text)
        for rs, entry in per_row
        if not entry.spurious_flag and entry.edited_text is not None
    ]
    text_choice, text_status, text_dissents = _text_consensus(keeper_texts)

    # If majority-spurious, keeper's text becomes a dissent (they voted keep
    # but the majority said spurious).
    if spurious_status == "majority_spurious":
        text_choice = None
        text_dissents = [
            RowDissent(reviewer_short=rs, value=entry.edited_text or "")
            for rs, entry in per_row
            if not entry.spurious_flag and entry.edited_text is not None
        ]
        text_status = "majority"  # informational; row status dominated by spurious
    elif spurious_status == "unanimous_spurious":
        text_choice = None
        text_dissents = []
        text_status = "unanimous"

    # Type side computed over all reviewers (folding doodle cluster).
    type_values: list[tuple[str, str | None]] = [(rs, entry.type_raw) for rs, entry in per_row]
    type_choice, type_status, type_dissents = _type_raw_consensus(type_values)

    # Notes histogram.
    notes_counter: Counter[str] = Counter()
    for _, entry in per_row:
        notes_counter[entry.notes if entry.notes is not None else "null"] += 1

    # Confidence is dropped from the agreement computation per plan
    # §notes-and-confidence (`confidence` is anti-calibrated per the n=19
    # baseline; reviewer-edited confidence is noise). The canonical row's
    # `confidence` field is informational only.
    chosen_confidence: Confidence = "high"

    needs_more = (
        spurious_needs_more or text_status == "insufficient" or type_status == "insufficient"
    )

    return {
        "spurious_status": spurious_status,
        "spurious_needs_more": spurious_needs_more,
        "spurious_votes_counts": Counter(spurious_votes),
        "text_choice": text_choice,
        "text_status": text_status,
        "text_dissents": text_dissents,
        "type_choice": type_choice,
        "type_status": type_status,
        "type_dissents": type_dissents,
        "notes_counter": notes_counter,
        "confidence": chosen_confidence,
        "reviewer_shorts": [rs for rs, _ in per_row],
        "needs_more_reviewers": needs_more,
    }


def _build_canonical_row(canonical_idx: int, bundle_idx: int, analysis: dict) -> CanonicalRow:
    spurious_status: SpuriousFlagStatus = analysis["spurious_status"]
    row_status = _worst_of_row_status(
        analysis["text_status"], analysis["type_status"], spurious_status
    )

    text_status_field = (
        "illegible"
        if analysis["text_status"] == "illegible"
        else ("majority" if analysis["text_status"] == "majority" else "unanimous")
    )
    type_status_field = (
        "unknown"
        if analysis["type_status"] == "unknown"
        else ("majority" if analysis["type_status"] == "majority" else "unanimous")
    )

    verification = RowVerification(
        status=row_status,
        raw_text_status=text_status_field,
        raw_text_dissents=analysis["text_dissents"],
        type_raw_status=type_status_field,
        type_raw_dissents=analysis["type_dissents"],
        spurious_flag_status=spurious_status,
        spurious_flag_votes={
            "keep": analysis["spurious_votes_counts"].get(False, 0),
            "spurious": analysis["spurious_votes_counts"].get(True, 0),
        },
        notes_values=dict(analysis["notes_counter"]),
        reviewer_shorts=analysis["reviewer_shorts"],
    )

    return CanonicalRow(
        canonical_row_index=canonical_idx,
        bundle_row_index=bundle_idx,
        inserted_between_bundle_rows=None,
        raw_text=analysis["text_choice"],
        type_raw=(
            None
            if spurious_status in ("unanimous_spurious", "majority_spurious")
            else analysis["type_choice"]
        ),
        notes=None,
        confidence=analysis["confidence"],
        verification=verification,
    )


def _build_injection(
    canonical_idx: int,
    gap: tuple[int, int],
    reporters: list[tuple[str, MissingRowMarker]],
    classification: str,
) -> CanonicalRow:
    """Build a canonical row for a majority-reported missing-row injection."""
    texts = [(rs, m.suggested_text) for rs, m in reporters]
    text_choice, _text_status, dissents = _text_consensus(texts)
    reporter_shorts = [rs for rs, _ in reporters]

    if classification == "under_emit_no_text_agreement":
        text_choice = _ILLEGIBLE
        dissents = [RowDissent(reviewer_short=rs, value=m.suggested_text) for rs, m in reporters]
        text_status_field = "illegible"
        row_status: RowStatus = "under_emit_no_text_agreement"
    else:
        text_status_field = (
            "majority"
            if len({_normalize_raw_text(m.suggested_text) for _, m in reporters}) > 1
            else "unanimous"
        )
        row_status = "under_emit_majority"

    # type_raw among reporters.
    types = [(rs, m.type_raw) for rs, m in reporters]
    type_choice, type_status, type_dissents = _type_raw_consensus(types)
    if type_status == "unknown":
        type_status_field = "unknown"
    elif type_status == "majority":
        type_status_field = "majority"
    else:
        type_status_field = "unanimous"

    verification = RowVerification(
        status=row_status,
        raw_text_status=text_status_field,
        raw_text_dissents=dissents,
        type_raw_status=type_status_field,
        type_raw_dissents=type_dissents,
        spurious_flag_status="unanimous_keep",
        spurious_flag_votes={"keep": len(reporters), "spurious": 0},
        notes_values={"null": len(reporters)},
        reviewer_shorts=reporter_shorts,
    )

    return CanonicalRow(
        canonical_row_index=canonical_idx,
        bundle_row_index=None,
        inserted_between_bundle_rows=gap,
        raw_text=text_choice,
        type_raw=type_choice,
        notes=None,
        confidence="medium",
        verification=verification,
    )


# --------------------------------------------------------------------------- #
# Agreement statistics
# --------------------------------------------------------------------------- #


def _build_agreement(
    *,
    stem: str,
    year: str,
    bucket: str,
    target: int,
    submissions: list[CalibrationSubmission],
    canonical_rows: list[CanonicalRow],
    bundle_row_count: int,
) -> CalibrationAgreement:
    submission_records = [
        SubmissionRecord(
            reviewer_short=_short(sub.reviewer.user_id),
            submitted_at=sub.submitted_at,
        )
        for sub in submissions
    ]

    histogram: Counter[str] = Counter()
    for row in canonical_rows:
        histogram[row.verification.status] += 1

    pair_concordance = _compute_pair_concordance(submissions, bundle_row_count)

    return CalibrationAgreement(
        schema_version=1,
        stem=stem,
        year=year,
        bucket=bucket,
        target_reviewers=target,
        submissions=submission_records,
        row_status_histogram=dict(histogram),
        pair_concordance=pair_concordance,
    )


def _compute_pair_concordance(
    submissions: list[CalibrationSubmission],
    bundle_row_count: int,
) -> list[PairConcordance]:
    """Pairwise agreement rates over bundle rows only.

    For each pair, we count how many bundle rows the two reviewers agree
    on `raw_text` (using `rows_agree`) and `type_raw` (using normalized
    equality), divided by the total number of bundle rows.

    Rows where either reviewer flagged spurious count as agreement on
    `raw_text` iff both agreed on the spurious flag; disagreement
    otherwise. `type_raw` is compared under the doodle-cluster fold.
    """
    pairs: list[PairConcordance] = []
    for sub_a, sub_b in combinations(submissions, 2):
        rs_a = _short(sub_a.reviewer.user_id)
        rs_b = _short(sub_b.reviewer.user_id)
        text_agree = 0
        type_agree = 0
        for idx in range(bundle_row_count):
            row_a = _find_row(sub_a.rows, idx)
            row_b = _find_row(sub_b.rows, idx)
            if row_a is None or row_b is None:
                continue
            if _row_texts_agree(row_a, row_b):
                text_agree += 1
            if _normalize_type_raw(row_a.type_raw) == _normalize_type_raw(row_b.type_raw):
                type_agree += 1
        denominator = bundle_row_count if bundle_row_count > 0 else 1
        pairs.append(
            PairConcordance(
                a=rs_a,
                b=rs_b,
                raw_text_agree_rate=text_agree / denominator,
                type_raw_agree_rate=type_agree / denominator,
            )
        )
    return pairs


def _find_row(
    rows: Iterable[CalibrationRowSubmission], idx: int
) -> CalibrationRowSubmission | None:
    return next((r for r in rows if r.bundle_row_index == idx), None)


def _row_texts_agree(a: CalibrationRowSubmission, b: CalibrationRowSubmission) -> bool:
    """Agreement rule for pair-concordance text comparison.

    Both spurious → agree.
    One spurious, one keep → disagree.
    Both keep → text comparator.
    """
    if a.spurious_flag and b.spurious_flag:
        return True
    if a.spurious_flag != b.spurious_flag:
        return False
    if a.edited_text is None or b.edited_text is None:
        return a.edited_text == b.edited_text
    return rows_agree(a.edited_text, b.edited_text)
