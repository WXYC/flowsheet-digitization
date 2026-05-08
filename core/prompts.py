"""Prompts for the extraction step.

Kept in its own module so prompt changes are reviewable independently of
client/orchestration code, and so we can A/B prompt variants without
touching the call site.

Phase 1 explicitly asks the model to:
  * preserve raw_text verbatim (no abbreviation expansion, no normalization),
  * always return four quadrants in fixed order even when blank,
  * never invent content; mark unreadable rows confidence=low,
  * tag special-case rows in `notes` and skip parsing them.

Three top-level prompts:

  * `PAGE_EXTRACTION_PROMPT` — Gemini and the page-level qwen-vl adapter.
    The model sees the whole page; the schema demands all four quadrants.

  * `QUADRANT_EXTRACTION_PROMPT_TEMPLATE` — the per-quadrant qwen-vl
    adapter (`modal-qwen-vl-quad`). The model sees one quadrant crop
    (with a small bleed band); the schema demands a single Quadrant.
    Use as `QUADRANT_EXTRACTION_PROMPT_TEMPLATE.format(position=...)`.

  * `HEADER_EXTRACTION_PROMPT` — the per-quadrant adapter's header-strip
    call. Pulls only `page_date_raw` and page-level oddities from the
    top band of the page.

The per-row guidance (raw_text / artist_guess / confidence / notes
tags / etc.) is duplicated across the page and quadrant prompts. They
must stay in sync. The shared row-level content is enforced by parallel
contract tests in `tests/unit/test_prompts.py`; if you change one,
update the other.
"""

from __future__ import annotations

PAGE_EXTRACTION_PROMPT = """\
You are transcribing one page of a 1990s WXYC handwritten radio flowsheet.

The form has a printed grid: a date at the top and four quadrants (top-left,
top-right, bottom-left, bottom-right). Each quadrant is one hour of broadcast
and has its own "Hour" and "Jock" (DJ) labels and a list of handwritten rows.

Each row generally reads "ARTIST - TRACK" (sometimes with the album in
parentheses). The left margin has a small printed type column (H, M, L, Std,
O, R) — IGNORE this column for now; do not include it in raw_text or guesses.

For every row, return:
  - row_index: 0-based position within the quadrant
  - raw_text: the line, transcribed verbatim. Do not expand abbreviations
    ("LED ZEP" stays "LED ZEP"). Do not fix spelling. Preserve case.
  - artist_guess: best-effort parse of the part left of the dash, or null
  - track_guess: best-effort parse of the part right of the dash, or null
  - confidence: "high" if the row is clearly legible, "medium" if you had to
    guess one or two characters, "low" if mostly illegible
  - notes: null in the common case. Use one of these tags only when relevant:
      * "continuation" — this row is a wrap of the previous row, not its own entry
      * "double_height" — handwriting takes up two physical rows of the grid
      * "crossed_out" — the entry is struck through
      * "illegible" — you could not read enough to attempt a transcription

Always return EXACTLY FOUR quadrants in this fixed order:
  1. top_left  2. top_right  3. bottom_left  4. bottom_right
Even if a quadrant is completely blank, include it with empty entries.

For each quadrant, capture:
  - position (one of the four labels above),
  - hour_raw (verbatim, e.g. "6AM", "7PM", "10°"; null if blank),
  - jock_raw (verbatim DJ name; null if blank).

Also capture:
  - page_date_raw: the date as written at the top of the page, verbatim
    (e.g. "Monday 1 Jan '90"). Null if blank or unreadable.

## Oddities — surface anything the schema doesn't model

Every Entry, Quadrant, and PageResult has an `oddities` list. Use it to
describe anything you see on the page that the rest of the schema does
not have a place for. This is how we discover what to formalize next; be
specific and honest — if something looks unusual, write a short sentence
about it.

  * Entry-level oddities: anything specific to ONE row that doesn't fit
    `notes` (which is reserved for: continuation, double_height,
    crossed_out, illegible). Examples:
      - "left margin has a hand-drawn arrow pointing to row 2"
      - "asterisk drawn next to this entry in the right margin"
      - "tiny annotation '7\"' at end of line"

  * Quadrant-level oddities: visual structure spanning multiple rows in
    THIS quadrant. Examples:
      - "rows 4-8 are inside a curly brace labeled 'ALL-REQUEST XMAS'"
      - "an arrow is drawn from row 3 down to row 6 (re-ordering)"
      - "rows 12-15 are bracketed with the label 'Smarty's Group/Album'"

  * Page-level oddities: anything OUTSIDE the four quadrants — content
    the schema simply has no field for. Examples:
      - "the entire page is rotated 180 degrees"
      - "Comments field at the bottom contains: 'declared today anti-Valentines Day...'"
      - "a weather note above the date reads: '25 degrees month wind chill 5'"
      - "a DJ-handoff note at the top of the right column says: 'F.S. Earl - Charles next'"
      - "marginal note in left margin near row 3 of top-left quadrant: 'Cool!'"

Each oddity should be one short sentence (under ~20 words). Empty list
if nothing unusual. Do not repeat things the schema already captures
(e.g. don't write an oddity that says "this row is crossed out" — that
belongs in `notes`).

CRITICAL RULES:
  * Never invent content. If you cannot read a row, give your best partial
    transcription and set confidence=low — do not fabricate.
  * Do not normalize abbreviations or spelling.
  * Do not include the left-margin type column letters in raw_text.
  * If the printed grid line is empty, do not emit an Entry for it.

Return only the structured JSON described by the response schema.
"""


QUADRANT_EXTRACTION_PROMPT_TEMPLATE = """\
You are transcribing ONE quadrant (one hour-block) of a 1990s WXYC
handwritten radio flowsheet. The image is a crop of a single cell from
the page's 2x2 grid; a small bleed band may include a few pixels of the
neighbor quadrants and the page header. Transcribe ONLY the content
that belongs to THIS hour-block; ignore content that bleeds in from
neighbors or the header.

This quadrant has its own "Hour" and "Jock" (DJ) labels and a list of
handwritten rows. Each row generally reads "ARTIST - TRACK" (sometimes
with the album in parentheses). The left margin has a small printed
type column (H, M, L, Std, O, R) — IGNORE this column for now; do not
include it in raw_text or guesses.

For every row, return:
  - row_index: 0-based position within the quadrant
  - raw_text: the line, transcribed verbatim. Do not expand abbreviations
    ("LED ZEP" stays "LED ZEP"). Do not fix spelling. Preserve case.
  - artist_guess: best-effort parse of the part left of the dash, or null
  - track_guess: best-effort parse of the part right of the dash, or null
  - confidence: "high" if the row is clearly legible, "medium" if you had to
    guess one or two characters, "low" if mostly illegible
  - notes: null in the common case. Use one of these tags only when relevant:
      * "continuation" — this row is a wrap of the previous row, not its own entry
      * "double_height" — handwriting takes up two physical rows of the grid
      * "crossed_out" — the entry is struck through
      * "illegible" — you could not read enough to attempt a transcription

For the quadrant itself, capture:
  - position: must be exactly "{position}" (the cell this crop came from)
  - hour_raw: verbatim hour label (e.g. "6AM", "7PM", "10°"); null if blank
  - jock_raw: verbatim DJ name; null if blank

## Oddities — surface anything the schema doesn't model

Both Entry and Quadrant have an `oddities` list. Use it to describe
anything you see in THIS quadrant that the rest of the schema does not
have a place for. Keep oddities scoped to this quadrant — page-level
notes (rotated page, comments field, header notes) are captured by a
separate call against the page header and must NOT be repeated here.

  * Entry-level oddities: anything specific to ONE row that doesn't fit
    `notes` (which is reserved for: continuation, double_height,
    crossed_out, illegible). Examples:
      - "left margin has a hand-drawn arrow pointing to row 2"
      - "asterisk drawn next to this entry in the right margin"
      - "tiny annotation '7\\"' at end of line"

  * Quadrant-level oddities: visual structure spanning multiple rows in
    THIS quadrant. Examples:
      - "rows 4-8 are inside a curly brace labeled 'ALL-REQUEST XMAS'"
      - "an arrow is drawn from row 3 down to row 6 (re-ordering)"
      - "rows 12-15 are bracketed with the label 'Smarty's Group/Album'"

Each oddity should be one short sentence (under ~20 words). Empty list
if nothing unusual. Do not repeat things the schema already captures
(e.g. don't write an oddity that says "this row is crossed out" — that
belongs in `notes`).

CRITICAL RULES:
  * Never invent content. If you cannot read a row, give your best partial
    transcription and set confidence=low — do not fabricate.
  * Do not normalize abbreviations or spelling.
  * Do not include the left-margin type column letters in raw_text.
  * If the printed grid line is empty, do not emit an Entry for it.
  * Do NOT emit page-level oddities here — only entry- and quadrant-level.

Return only the structured JSON described by the response schema.
"""


HEADER_EXTRACTION_PROMPT = """\
You are reading the top header strip of a 1990s WXYC handwritten radio
flowsheet page. The image is a horizontal slice from the very top of
the page — above the four hour-blocks of the broadcast grid. It
contains the date and any free-text annotations the DJs added there
(handoff notes, weather notes, marginal scribbles).

Capture:
  - page_date_raw: the date as written, verbatim (e.g. "Monday 1 Jan '90").
    Null if blank or unreadable.
  - oddities: a list of short sentences (under ~20 words each)
    describing free-text content in this header band that the schema
    has no field for.

Oddities here are PAGE-LEVEL ONLY: notes above the date, DJ-handoff
messages, weather notes, marginal text in the header band. Do NOT
include row-level or quadrant-level content — the four quadrants
below the header are captured by separate calls and have their own
oddity collections. Examples of what belongs here:
  - "the entire page is rotated 180 degrees"
  - "a weather note above the date reads: '25 degrees month wind chill 5'"
  - "a DJ-handoff note reads: 'F.S. Earl - Charles next'"

Each oddity is one short sentence. Empty list if nothing unusual.

Never invent content. If a marker is unreadable, leave it out rather
than guessing.

Return only the structured JSON described by the response schema.
"""
