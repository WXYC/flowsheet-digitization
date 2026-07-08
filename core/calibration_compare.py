"""Reviewer-text agreement comparator for the multi-reviewer calibration flow.

This module answers one question: given two reviewers' readings of the same
row, do we count them as agreeing? The answer feeds the per-row consensus
in `core.calibration_consensus`.

The comparator has two tiers:

  * **Tier 1 (this module)** — pure normalization + byte-compare. Handles
    the mechanical variance real reviewers produce (casefold, whitespace,
    dash variants, smart quotes, ampersand-vs-and, punctuation like
    R.E.M. vs REM).
  * **Tier 2 (future work)** — an optional `canonicalize` callable that
    consults LML (`library-metadata-lookup`) to fold `ACDC` and `AC/DC` to
    a common canonical form. Not implemented in this plan; the argument
    is reserved so the switch is one-line when Phase 2 reconciliation
    lands. Callers pass `canonicalize=None` today.

The comparator is deliberately kept out of the SPA: real-time LML
autocomplete in calibration mode would anchor reviewers on the first
suggestion and inflate inter-reviewer agreement falsely. Tier 2 belongs
in the merge step, not the editor.

Not to be confused with `core.calibration` — that module scores extraction
models against goldens. This module compares reviewer submissions to each
other. Different concerns, coincident naming; the file-name pair is
called out in CLAUDE.md.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

# Characters stripped during raw_text normalization. Deliberately NOT
# including `/` (identity-carrying — AC/DC vs ACDC) or `-` (separator —
# 'Artist - Track' vs 'Artist Track' is a real disagreement).
_STRIP_PUNCT = str.maketrans(
    {
        ".": "",
        ",": "",
        "'": "",   # straight single quote (U+0027)
        '"': "",   # straight double quote (U+0022)
        "‘": "",  # left single quotation mark
        "’": "",  # right single quotation mark (also used as apostrophe)
        "“": "",  # left double quotation mark
        "”": "",  # right double quotation mark
    }
)

# En- and em-dashes fold to hyphen (`-`), which is preserved.
_DASH_FOLD = str.maketrans({"–": "-", "—": "-"})

_WS = re.compile(r"\s+")

# Members of the doodle/blank cluster for `type_raw`. Informed by
# `project_oddity_doodle_entry.md`: doodles in the type-column circle are
# an established pattern, not a transcription error. Reviewers will
# inconsistently transcribe them as `?`, `doodle`, or leave blank —
# folding these to a sentinel prevents spurious escalation while keeping
# the actual letter alphabet (H/M/L/S/Std/O/R) gating in the normal way.
_DOODLE_CLUSTER: frozenset[str] = frozenset({"", "?", "-", "doodle", "scribble"})
_TYPE_RAW_UNKNOWN = "_unknown"


def _normalize_raw_text(text: str) -> str:
    """Canonicalize a reviewer's `raw_text` for comparison.

    Steps, in order:
      1. NFKC unicode normalization.
      2. Casefold (Unicode-correct lowercase).
      3. Dash normalization: en-dash and em-dash → `-`.
      4. Ampersand expansion: `&` → ` and ` (whitespace-padded).
      5. Strip decorative punctuation (`.`, `,`, straight and smart quotes).
      6. Collapse runs of whitespace to a single space; strip ends.

    Slash and hyphen are preserved deliberately.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = normalized.translate(_DASH_FOLD)
    normalized = normalized.replace("&", " and ")
    normalized = normalized.translate(_STRIP_PUNCT)
    normalized = _WS.sub(" ", normalized).strip()
    return normalized


def _normalize_type_raw(value: str | None) -> str:
    """Canonicalize a reviewer's `type_raw` for comparison.

    Returns the sentinel `_unknown` for the doodle/blank cluster
    (`""`, `None`, `"?"`, `"-"`, `"doodle"`, `"scribble"`, casefold
    equivalents). Otherwise: NFKC + casefold + strip whitespace.
    """
    if value is None:
        return _TYPE_RAW_UNKNOWN
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    if normalized in _DOODLE_CLUSTER:
        return _TYPE_RAW_UNKNOWN
    return normalized


def rows_agree(
    a: str,
    b: str,
    *,
    canonicalize: Callable[[str], str | None] | None = None,
) -> bool:
    """Tier-1 normalize-and-byte-compare, with an optional Tier-2 hook.

    Short-circuits on Tier-1 agreement — `canonicalize` is not called
    unless Tier-1 mismatches. When both sides canonicalize to truthy
    equal values, the pair agrees; when either side canonicalizes to
    None (unknown), fall through to disagreement.
    """
    if _normalize_raw_text(a) == _normalize_raw_text(b):
        return True
    if canonicalize is not None:
        ca, cb = canonicalize(a), canonicalize(b)
        if ca and cb and ca == cb:
            return True
    return False
