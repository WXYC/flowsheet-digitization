"""Tests for the reviewer-text agreement comparator.

Covers `_normalize_raw_text`, `_normalize_type_raw`, and the `rows_agree`
public predicate. See plans/multi-reviewer-calibration.md §Comparator.

The Tier-2 `canonicalize` argument to `rows_agree` is not implemented in
this plan; the tests here only exercise Tier-1 normalization plus the
argument's contract (short-circuit on tier-1 match; consult only if tier-1
mismatches).
"""

from __future__ import annotations

import unicodedata

import pytest

from core.calibration_compare import (
    _normalize_raw_text,
    _normalize_type_raw,
    rows_agree,
)


class TestNormalizeRawTextProperties:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            " ",
            "BEATLES - HELP",
            "R.E.M. - Losing My Religion",
            "AC/DC - Back In Black",
            "Simon & Garfunkel - The Boxer",
            "Beatles – Help",
            "  Beatles  -  Help  ",
            "Nilüfer Yanya - Melt Away",
        ],
    )
    def test_idempotent(self, value: str) -> None:
        once = _normalize_raw_text(value)
        twice = _normalize_raw_text(once)
        assert once == twice

    def test_nfkc_composed_and_decomposed_equal(self) -> None:
        composed = "Nilüfer Yanya"
        decomposed = unicodedata.normalize("NFD", composed)
        assert composed != decomposed  # sanity: they're byte-different
        assert _normalize_raw_text(composed) == _normalize_raw_text(decomposed)


class TestNormalizeRawTextRules:
    @pytest.mark.parametrize(
        "a,b",
        [
            # Casefold
            ("BEATLES - HELP", "beatles - help"),
            ("Beatles - Help", "BEATLES - HELP"),
            # Dash variants (hyphen, en-dash, em-dash)
            ("Beatles - Help", "Beatles – Help"),
            ("Beatles - Help", "Beatles — Help"),
            # Ampersand expansion (with padding so no glue-ups)
            ("Simon & Garfunkel", "Simon and Garfunkel"),
            ("Earth & Wind", "earth and wind"),
            # Punctuation strip
            ("R.E.M. - Losing My Religion", "REM - Losing My Religion"),
            ("R.E.M.", "REM"),
            ("Guns N' Roses", "Guns N Roses"),
            ('"Weird Al" Yankovic', "Weird Al Yankovic"),
            ("Peter, Paul and Mary", "Peter Paul and Mary"),
            # Smart quotes
            ("Guns N’ Roses", "Guns N Roses"),
            ("“Weird Al” Yankovic", "Weird Al Yankovic"),
            # Whitespace collapse
            ("  Beatles  -  Help  ", "Beatles - Help"),
            ("BEATLES\t-\tHELP", "BEATLES - HELP"),
        ],
    )
    def test_pairs_agree(self, a: str, b: str) -> None:
        assert rows_agree(a, b), f"expected agree: {a!r} vs {b!r}"

    @pytest.mark.parametrize(
        "a,b",
        [
            # Slash carries identity: AC/DC != ACDC.
            ("AC/DC", "ACDC"),
            ("GZA/Genius", "GZA Genius"),
            # Separator load-bearing: 'Artist - Track' vs 'Artist Track'.
            ("Beatles - Help", "Beatles Help"),
            # Different tracks.
            ("Beatles - Help", "Beatles - Yesterday"),
            # Different artists.
            ("Beatles - Help", "Stones - Help"),
        ],
    )
    def test_pairs_disagree(self, a: str, b: str) -> None:
        assert not rows_agree(a, b), f"expected disagree: {a!r} vs {b!r}"


class TestNormalizeTypeRaw:
    @pytest.mark.parametrize(
        "value",
        ["", "?", "-", "doodle", "scribble", "DOODLE", "SCRIBBLE"],
    )
    def test_doodle_cluster_collapses_to_unknown(self, value: str) -> None:
        assert _normalize_type_raw(value) == "_unknown"

    def test_none_collapses_to_unknown(self) -> None:
        assert _normalize_type_raw(None) == "_unknown"

    @pytest.mark.parametrize(
        "a,b",
        [
            ("H", "H"),
            ("H", "h"),
            ("Std", "std"),
            ("Std", "STD"),
        ],
    )
    def test_letters_normalize_equal(self, a: str, b: str) -> None:
        assert _normalize_type_raw(a) == _normalize_type_raw(b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("H", "M"),
            ("H", "L"),
            ("Std", "S"),
            ("H", "O"),
        ],
    )
    def test_different_letters_do_not_normalize_equal(self, a: str, b: str) -> None:
        assert _normalize_type_raw(a) != _normalize_type_raw(b)


class TestRowsAgreeTier2Hook:
    """The Tier-2 canonicalize hook is not implemented in this plan, but the
    contract is: if Tier-1 already matches, canonicalize is not called; if
    Tier-1 mismatches, canonicalize is called on both sides and only makes
    them agree if both canonicalizations are truthy and equal."""

    def test_canonicalize_not_called_on_tier1_match(self) -> None:
        calls: list[str] = []

        def canonicalize(s: str) -> str | None:
            calls.append(s)
            return s

        assert rows_agree("BEATLES", "beatles", canonicalize=canonicalize)
        assert calls == []

    def test_canonicalize_bridges_tier1_gap(self) -> None:
        def canonicalize(s: str) -> str | None:
            # Toy: fold ACDC and AC/DC to a common canonical form.
            return s.replace("/", "").casefold()

        # Without canonicalize, these disagree (slash preserved).
        assert not rows_agree("AC/DC", "ACDC")
        assert rows_agree("AC/DC", "ACDC", canonicalize=canonicalize)

    def test_canonicalize_none_side_falls_through(self) -> None:
        def canonicalize(s: str) -> str | None:
            return None  # signal: no canonical form known

        assert not rows_agree("Foo", "Bar", canonicalize=canonicalize)
