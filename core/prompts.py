"""Prompts for the Gemini extraction step.

Kept in its own module so prompt changes are reviewable independently of
client/orchestration code, and so we can A/B prompt variants without
touching the call site.

Phase 1 explicitly asks the model to:
  * preserve raw_text verbatim (no abbreviation expansion, no normalization),
  * always return four quadrants in fixed order even when blank,
  * never invent content; mark unreadable rows confidence=low,
  * tag special-case rows in `notes` and skip parsing them.
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

CRITICAL RULES:
  * Never invent content. If you cannot read a row, give your best partial
    transcription and set confidence=low — do not fabricate.
  * Do not normalize abbreviations or spelling.
  * Do not include the left-margin type column letters in raw_text.
  * If the printed grid line is empty, do not emit an Entry for it.

Return only the structured JSON described by the response schema.
"""
