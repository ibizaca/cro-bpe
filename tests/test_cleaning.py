"""Regression tests for text normalization and Wikipedia residue cleaning."""

from __future__ import annotations

import pytest

from preprocessing.cleaning import clean_wikitext_residue, normalize_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stupac1\tstupac2", "stupac1 stupac2"),  # tabs separate words, they are not deleted
        ("a\t\tb", "a b"),
        ("red1\r\nred2\rred3", "red1\nred2\nred3"),
        ("ne­vidljivo​", "nevidljivo"),
        ("razmak i tanki", "razmak i tanki"),
        ("ǅamija i ǉubav", "Džamija i ljubav"),
        ("  višak   razmaka  \n\n\n\n novi red ", "višak razmaka\n\nnovi red"),
    ],
)
def test_normalize_whitespace_and_characters(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tom &amp; Jerry", "Tom & Jerry"),
        ("&#269;a&#x161;a", "čaša"),
        ("a&nbsp;b", "a b"),
        # Only complete entities are decoded: html.unescape on the whole text would turn
        # "&para" into "¶" and "&copy" into "©".
        ("?x=1&param=2 i &amp;", "?x=1&param=2 i &"),
        ("&copy2024 &copy;", "&copy2024 ©"),
        ("R&D bez entiteta", "R&D bez entiteta"),
    ],
)
def test_html_entities(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kraj<|endoftext|>Početak", "KrajPočetak"),
        ("a <|reserved_12|> b <|pad|> c", "a b c"),
        ("<| mask |> i <|endof text|> nisu tokeni", "<| mask |> i <|endof text|> nisu tokeni"),
    ],
)
def test_special_token_lookalikes_are_removed(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_is_idempotent() -> None:
    text = "Prvi\t red &amp; drugi\r\n\n\n\ntreći ǈ <|sep|>kraj"
    once = normalize_text(text)
    assert normalize_text(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Template parameters and table attributes are removed.
        ("Tekst url=http://x.hr access-date=2020 dalje", "Tekst dalje"),
        ('align=center style="color:red" tekst', "tekst"),
        # Prose that only looks like a parameter is kept.
        ("Vrijedi x=5 i y=3.", "Vrijedi x=5 i y=3."),
        ("Njegova težina=80 kg.", "Njegova težina=80 kg."),
        ("Formula E=mc2 je poznata.", "Formula E=mc2 je poznata."),
    ],
)
def test_wiki_template_residue(raw: str, expected: str) -> None:
    assert normalize_text(clean_wikitext_residue(raw)) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Prije {{citat|x}} poslije", "Prije poslije"),
        ("'''Zagreb''' je ''grad''.", "Zagreb je grad."),
        ("== Povijest ==\nTekst.", "Povijest\nTekst."),
        ("* stavka\n# druga", "stavka\ndruga"),
        ("{| class=x\n| ćelija\n|}\nTekst", "Tekst"),
        ("[http://primjer.hr Primjer] stranica", "Primjer stranica"),
        ("svirao je pjevanje|pjevač", "svirao je pjevač"),
    ],
)
def test_wiki_markup(raw: str, expected: str) -> None:
    assert normalize_text(clean_wikitext_residue(raw)) == expected
