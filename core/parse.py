"""Read-time artist/track split for `Entry.raw_text`.

The Phase-1 schema asked Gemini to fill `artist_guess` and `track_guess`
alongside the verbatim `raw_text`. The split is deterministic (separate
on the first whitespace-bracketed dash) and producing it on the model
side spends output tokens for no analytic benefit. The fields were
dropped from the response schema; this module is what downstream
consumers call at read time to recover them.

Sibling pattern: `core.continuations.merge_continuations`,
`core.comments.normalize_comments`. Pure function, no I/O. The on-disk
shape stays verbatim; the split happens in memory on read.

Separator convention: a single ASCII hyphen, en-dash, or em-dash with
whitespace on both sides. Compound band names ("X-Ray Spex") have no
surrounding whitespace and are NOT split. Multiple separators in one
line split on the first; the track side keeps the rest.
"""

from __future__ import annotations

import re

# ASCII hyphen, en-dash (U+2013), em-dash (U+2014). Whitespace-bracketed
# so we don't split compound band names like "X-Ray Spex" on their
# internal hyphen.
_SEPARATOR = re.compile(r"\s+[-–—]\s+")


def parse_artist_track(raw_text: str | None) -> tuple[str | None, str | None]:
    """Split a row's verbatim text into best-effort (artist, track) parts.

    Returns:
      * `(None, None)` when the input is None, empty, or whitespace-only.
      * `(artist, None)` when no separator is present — the line names
        an artist with no track (common for early-90s flowsheets, e.g.
        "STEREOLAB" alone).
      * `(None, track)` when the separator opens the line with nothing
        before it (e.g. " - LA PARADOJA").
      * `(artist, None)` when the separator closes the line with nothing
        after it (e.g. "JUANA MOLINA - ").
      * `(artist, track)` for the dominant case ("ARTIST - TRACK"). When
        multiple separators appear, we split on the first only and the
        rest stays in the track.

    Each side is stripped of leading/trailing whitespace. Empty sides
    become None so callers have one sentinel for "nothing useful here".
    """
    if raw_text is None:
        return (None, None)
    if not raw_text.strip():
        return (None, None)
    # Split the unstripped text so a leading or trailing " - " still has
    # the whitespace on both sides of the dash that the regex needs.
    split = _SEPARATOR.split(raw_text, maxsplit=1)
    if len(split) == 1:
        return (raw_text.strip(), None)
    artist = split[0].strip() or None
    track = split[1].strip() or None
    return (artist, track)
