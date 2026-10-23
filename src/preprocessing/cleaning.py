"""Text normalization applied to every document before quality filtering.

All functions are pure ``str -> str`` transforms so they can be reused at
tokenizer inference time and tested in isolation.
"""

import html
import re
import unicodedata

from preprocessing.charclasses import NEEDS_NORMALIZATION, classify

# Part of the preprocessing fingerprint: bump when a change to normalize_text or
# clean_wikitext_residue alters the output.
VERSION = 1

# Croatian digraph letters (U+01C4–U+01CC) are decomposed to two code points so
# ``Dž``/``Lj``/``Nj`` share merges with the far more common two-letter spelling.
# NFKC would do the same but also rewrites superscripts, fractions and other
# meaningful compatibility characters, so only a targeted table is applied.
_TRANSLATION = str.maketrans(
    {
        "Ǆ": "DŽ",
        "ǅ": "Dž",
        "ǆ": "dž",
        "Ǉ": "LJ",
        "ǈ": "Lj",
        "ǉ": "lj",
        "Ǌ": "NJ",
        "ǋ": "Nj",
        "ǌ": "nj",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)

_LINE_BREAKS = re.compile("\r\n?|[\x85\u2028\u2029\x0b\x0c]")
_SPACES = re.compile("[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
# C0/C1 controls (except \t, which becomes a space, and \n), soft hyphen, zero-width and
# bidi formatting marks, BOM, noncharacters
_INVISIBLE = re.compile(
    "[\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufffe\uffff]"
)
_HTML_ENTITY = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});")
# Text spelling a special token (``<|endoftext|>``, ``<|reserved_3|>``) would be encoded
# as that token by tokenizer.json but as plain text by tiktoken, so it never reaches training.
_SPECIAL_TOKEN_LOOKALIKE = re.compile(r"<\|[A-Za-z0-9_]+\|>")
_MULTI_SPACE = re.compile(r" {2,}")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _unescape_entity(match: re.Match[str]) -> str:
    return html.unescape(match.group())


def normalize_text(text: str) -> str:
    """Unicode, entity and whitespace normalization shared by all sources."""

    if "&" in text:
        # Only complete entities: html.unescape on the whole text also decodes "&para" in "&param=1".
        text = _HTML_ENTITY.sub(_unescape_entity, text)
    text = unicodedata.normalize("NFC", text)
    # Single vectorized scan instead of one regex pass per character class.
    if (classify(text) & NEEDS_NORMALIZATION).any():
        text = text.translate(_TRANSLATION)
        text = _LINE_BREAKS.sub("\n", text)
        text = _INVISIBLE.sub("", text)
        text = _SPACES.sub(" ", text)
    if "<|" in text:
        text = _SPECIAL_TOKEN_LOOKALIKE.sub("", text)
    if "  " in text:
        text = _MULTI_SPACE.sub(" ", text)
    if " \n" in text or "\n " in text:
        text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    if "\n\n\n" in text:
        text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─────────────────────────── Wikipedia residue ───────────────────────────────

# Lines that are table rows, template parameters, magic words or file/category links.
_WIKI_DROP_LINE = re.compile(r"^\s*(?:\||\{\||!|\{\{|\}\}|__[A-Z]+__|\[\[|(?:Datoteka|Slika|Kategorija|File|Image):)")
_WIKI_EXTERNAL_LINK = re.compile(r"\[(?:https?:)?//[^\s\]]+(?: ([^\]]*))?\]")
_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_WIKI_EMPHASIS = re.compile(r"'{2,}")
_WIKI_LIST_MARKER = re.compile(r"^[ \t]*[*#:;]+[ \t]*", re.MULTILINE)
# Template parameters (``url=…``, ``access-date=…``, ``align=…``) and unbalanced braces.
# Parameter names are ASCII, at least two characters and start a word, so prose such
# as ``x=5`` or ``težina=80`` is kept.
_WIKI_TEMPLATE_RESIDUE = re.compile(r"(?<![^\s|])[a-z][a-z0-9_-]+=\S*|\{\{|\}\}")
_WIKI_HEADING = re.compile(r"^(=+) *(.*?) *\1$", re.MULTILINE)
_WORD = re.compile(r"\S+")
_LABEL_WORD = re.compile(r"[^\W\d_]+")
_LINK_TARGET_WINDOW = 8


def _fold(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word.casefold()) if not unicodedata.combining(c))


def _repair_piped_links(line: str) -> str:
    """Remove ``target|`` left behind by the importer's ``[[target|label]]`` stripping.

    The link target is located by scanning back for the nearest word sharing a
    three-letter prefix with the label (Croatian labels are usually an inflected
    form of the target, e.g. ``pjevanje|pjevač``, ``Burkina Faso|Burkine Faso``).
    When no such word exists the pipe is replaced by a space.
    """

    parts = line.split("|")
    result = parts[0]
    for label in parts[1:]:
        label_word = _LABEL_WORD.match(label)
        if label_word is not None and len(label_word.group()) >= 2:
            prefix = _fold(label_word.group())[:3]
            words = list(_WORD.finditer(result))[-_LINK_TARGET_WINDOW:]
            for word in reversed(words):
                if _fold(word.group().lstrip("(\"'„")).startswith(prefix):
                    result = result[: word.start()]
                    break
            else:
                result += " "
        result += label
    return result


def clean_wikitext_residue(text: str) -> str:
    """Strip markup that survived the Wikipedia dump import."""

    text = _WIKI_EXTERNAL_LINK.sub(lambda match: match.group(1) or "", text)
    for _ in range(3):  # nested templates, innermost first
        text, replaced = _WIKI_TEMPLATE.subn("", text)
        if not replaced:
            break
    text = _WIKI_TEMPLATE_RESIDUE.sub(" ", text)
    text = _WIKI_EMPHASIS.sub("", text)
    text = _WIKI_HEADING.sub(r"\2", text)
    text = _WIKI_LIST_MARKER.sub("", text)
    lines = []
    for line in text.split("\n"):
        if _WIKI_DROP_LINE.match(line):
            continue
        if "|" in line:
            line = _repair_piped_links(line)
        lines.append(line)
    return "\n".join(lines)
