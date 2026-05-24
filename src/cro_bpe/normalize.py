import html
import re
import unicodedata

COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
REF_PATTERN = re.compile(r"<ref[^>/]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
SELF_CLOSING_REF_PATTERN = re.compile(r"<ref[^>]*/>", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
CATEGORY_PATTERN = re.compile(r"\[\[(?:Kategorija|Category):[^\]]+\]\]", re.IGNORECASE)
FILE_PATTERN = re.compile(r"\[\[(?:Datoteka|File|Slika):[^\]]+\]\]", re.IGNORECASE)
EXTERNAL_LINK_PATTERN = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
INTERNAL_LINK_PATTERN = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
HEADING_PATTERN = re.compile(r"(?m)^=+\s*(.*?)\s*=+$")
APOSTROPHE_PATTERN = re.compile(r"'{2,}")
TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}")
TABLE_LINE_PATTERN = re.compile(r"(?m)^(\{|\||!).*$")
WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_wiki_markup(text: str) -> str:
    cleaned_text = COMMENT_PATTERN.sub(" ", text)
    cleaned_text = REF_PATTERN.sub(" ", cleaned_text)
    cleaned_text = SELF_CLOSING_REF_PATTERN.sub(" ", cleaned_text)
    cleaned_text = CATEGORY_PATTERN.sub(" ", cleaned_text)
    cleaned_text = FILE_PATTERN.sub(" ", cleaned_text)
    cleaned_text = EXTERNAL_LINK_PATTERN.sub(lambda match: match.group(2) or " ", cleaned_text)

    previous_text = None
    while cleaned_text != previous_text:
        previous_text = cleaned_text
        cleaned_text = TEMPLATE_PATTERN.sub(" ", cleaned_text)

    cleaned_text = INTERNAL_LINK_PATTERN.sub(lambda match: match.group(2), cleaned_text)
    cleaned_text = HEADING_PATTERN.sub(lambda match: match.group(1), cleaned_text)
    cleaned_text = APOSTROPHE_PATTERN.sub("", cleaned_text)
    cleaned_text = TABLE_LINE_PATTERN.sub(" ", cleaned_text)
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    return cleaned_text


def normalize_croatian_text(text: str) -> str:
    cleaned_text = strip_wiki_markup(text)
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = cleaned_text.replace("\u00a0", " ")
    cleaned_text = unicodedata.normalize("NFC", cleaned_text)
    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text)
    return cleaned_text.strip()
