"""Tests for `core.parse.parse_artist_track`.

`parse_artist_track` is the read-time replacement for the
`Entry.artist_guess` / `Entry.track_guess` fields that used to live on
the Gemini response schema. The model now returns only `raw_text`; the
split happens here, deterministically, at read time.

Sibling pattern: `core.continuations.merge_continuations`,
`core.comments.normalize_comments`. Pure function, no I/O.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.parse import _SEPARATOR, parse_artist_track


class TestParseArtistTrack:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The dominant case: ASCII hyphen with single spaces.
            ("JUANA MOLINA - LA PARADOJA", ("JUANA MOLINA", "LA PARADOJA")),
            # Album in parens belongs to the track side, not split off.
            (
                "JUANA MOLINA - LA PARADOJA (DOGA)",
                ("JUANA MOLINA", "LA PARADOJA (DOGA)"),
            ),
        ],
    )
    def test_simple_ascii_hyphen(self, raw: str, expected: tuple[str | None, str | None]) -> None:
        assert parse_artist_track(raw) == expected

    @pytest.mark.parametrize(
        ("dash", "label"),
        [
            ("–", "en-dash"),
            ("—", "em-dash"),
        ],
    )
    def test_unicode_dashes_split_the_same(self, dash: str, label: str) -> None:
        """DJs sometimes write en/em-dashes; the model occasionally normalizes
        a hand-drawn dash to unicode. Treat them like ASCII hyphens so both
        on-disk shapes split consistently."""
        raw = f"JESSICA PRATT {dash} BACK, BABY"
        assert parse_artist_track(raw) == ("JESSICA PRATT", "BACK, BABY"), label

    def test_no_dash_returns_artist_only(self) -> None:
        """An entry with no separator is documented as artist-only. The
        Stereolab fixture case (artist-only row) appears in the WXYC
        canonical-artists test data; the dominant historical reading is
        that the line names the artist and omits the track."""
        assert parse_artist_track("STEREOLAB") == ("STEREOLAB", None)

    def test_multiple_dashes_split_on_first(self) -> None:
        """Hand-written track titles can themselves contain hyphens
        ("Duke Ellington & John Coltrane - In a Sentimental Mood"
        wouldn't trigger this — the artist '&' isn't a dash — but a
        title like 'X-Ray' on the track side will). Split on the first
        separator so the artist segment stays intact."""
        assert parse_artist_track("ARTIST - TRACK - PART TWO") == (
            "ARTIST",
            "TRACK - PART TWO",
        )

    def test_strips_whitespace_per_side(self) -> None:
        """Leading/trailing whitespace on either side gets stripped from
        both sides of the result. Internal whitespace inside artist or
        track is preserved."""
        assert parse_artist_track("  THE BAND   -   THE SONG  ") == ("THE BAND", "THE SONG")

    def test_none_input_returns_none_pair(self) -> None:
        assert parse_artist_track(None) == (None, None)

    def test_empty_string_returns_none_pair(self) -> None:
        assert parse_artist_track("") == (None, None)

    def test_whitespace_only_returns_none_pair(self) -> None:
        assert parse_artist_track("   ") == (None, None)

    def test_empty_artist_side_returns_none(self) -> None:
        """A line that opens with the separator (' - TRACK') has no artist;
        the function shouldn't promote an empty string into a real value."""
        assert parse_artist_track(" - LA PARADOJA") == (None, "LA PARADOJA")

    def test_empty_track_side_returns_none(self) -> None:
        """A trailing separator with nothing after it has no track."""
        assert parse_artist_track("JUANA MOLINA - ") == ("JUANA MOLINA", None)

    def test_hyphen_with_no_spaces_does_not_split(self) -> None:
        """Compound words like 'X-Ray Spex' must NOT be split on the
        internal hyphen — the convention is that the separator is a dash
        SURROUNDED by whitespace. This is the load-bearing constraint
        that lets the function work on a corpus that mixes hyphenated
        band names with separator dashes."""
        assert parse_artist_track("X-RAY SPEX") == ("X-RAY SPEX", None)

    def test_separator_matches_across_newlines(self) -> None:
        """The regex's `\\s+` class includes `\\n`, so a separator that
        straddles a newline still splits. Rare in practice — Gemini's
        `raw_text` is usually a single line — but pinning the behavior
        here means a future tightening (e.g. `[^\\S\\n]+` to exclude
        newlines) is a deliberate decision rather than a regression.
        """
        assert parse_artist_track("ARTIST\n - TRACK") == ("ARTIST", "TRACK")

    def test_idempotent_when_artist_only_recomputed(self) -> None:
        """Round-tripping the result through join+split must be a no-op
        for the artist-only case: a downstream consumer that re-stores
        the artist as `raw_text` and re-parses shouldn't flip the result."""
        artist, track = parse_artist_track("STEREOLAB")
        assert (artist, track) == ("STEREOLAB", None)
        assert parse_artist_track(artist) == ("STEREOLAB", None)


class TestJsParity:
    """Guard that `verifier/app.js`'s ARTIST_TRACK_SEPARATOR uses the same
    strict `\\s+` (whitespace-on-both-sides) form as `core.parse._SEPARATOR`.
    A drift to `\\s*` on the JS side would split "X-RAY SPEX" on the
    internal hyphen and disagree with the Python read-time split, so the
    verifier UI's artist lookup would ask for the wrong artist."""

    _APP_JS = Path(__file__).resolve().parents[2] / "verifier" / "app.js"
    _JS_LINE = re.compile(
        r"const\s+ARTIST_TRACK_SEPARATOR\s*=\s*/(?P<body>[^/]+)/(?P<flags>[a-z]*)\s*;"
    )

    def _js_pattern(self) -> str:
        m = self._JS_LINE.search(self._APP_JS.read_text())
        assert m is not None, "ARTIST_TRACK_SEPARATOR literal not found in verifier/app.js"
        return m.group("body")

    def test_js_separator_matches_python(self) -> None:
        assert self._js_pattern() == _SEPARATOR.pattern
