import re
import unicodedata

from cro_bpe.config import PreprocessingConfig

COMMENT_PATTERN = re.compile(r"<!--.*?-->", flags=re.DOTALL)
REF_PATTERN = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", flags=re.DOTALL | re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"</?[^>]+>")
TEMPLATE_PATTERN = re.compile(r"\{\{[^{}]*\}\}")
WIKILINK_PATTERN = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
CATEGORY_PATTERN = re.compile(r"\[\[(?:Category|Kategorija):[^\]]+\]\]", flags=re.IGNORECASE)
HEADING_PATTERN = re.compile(r"^=+\s*(.*?)\s*=+$", flags=re.MULTILINE)
BULLET_PATTERN = re.compile(r"^[*#:;]+\s*", flags=re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_TRANSLATION = str.maketrans({chr(codepoint): " " for codepoint in range(32) if chr(codepoint) not in "\n\t"})


def normalize_text(text: str, config: PreprocessingConfig) -> str:
    if config.unicode_normalization_form == "NFC":
        normalized_text = unicodedata.normalize("NFC", text)
    elif config.unicode_normalization_form == "NFKC":
        normalized_text = unicodedata.normalize("NFKC", text)
    else:
        raise ValueError(f"Unsupported normalization form '{config.unicode_normalization_form}'")

    normalized_text = normalized_text.translate(CONTROL_TRANSLATION)

    if config.strip_markup:
        normalized_text = strip_markup(normalized_text)

    normalized_text = WHITESPACE_PATTERN.sub(" ", normalized_text).strip()
    return normalized_text


def strip_markup(text: str) -> str:
    cleaned_text = COMMENT_PATTERN.sub(" ", text)
    cleaned_text = REF_PATTERN.sub(" ", cleaned_text)
    cleaned_text = CATEGORY_PATTERN.sub(" ", cleaned_text)
    cleaned_text = TEMPLATE_PATTERN.sub(" ", cleaned_text)
    cleaned_text = WIKILINK_PATTERN.sub(r"\1", cleaned_text)
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = HEADING_PATTERN.sub(r"\1", cleaned_text)
    cleaned_text = BULLET_PATTERN.sub("", cleaned_text)
    return cleaned_text


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha_characters = sum(1 for character in text if character.isalpha())
    return alpha_characters / len(text)
