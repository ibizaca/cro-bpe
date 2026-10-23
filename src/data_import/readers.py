"""Streaming document readers for every supported source format.

Each reader is a generator that yields :class:`DocumentRecord` instances
without loading the full corpus into memory.  A :func:`stream_source`
dispatcher routes to the correct reader based on the catalog entry.

Archive members that fail to parse are skipped with a warning and recorded in
the :class:`ReadReport` passed to :func:`stream_source`, so data loss is
visible in the source manifest instead of silent.
"""

import bz2
import gzip
import json
import logging
import lzma
import re
import tarfile
import zipfile
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from data_import.catalog import (
    HfDownloadSource,
    HfStreamSource,
    HttpSource,
    HuggingFaceSource,
    SourceEntry,
    SourceFormat,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────── document record ─────────────────────────────


@dataclass
class DocumentRecord:
    """Normalised document from any source format."""

    text: str
    source_id: str
    doc_id: str | None = None
    url: str | None = None
    language: str | None = None


@dataclass
class ReadReport:
    """Archive members skipped by a reader because they could not be parsed."""

    skipped_files: list[str] = field(default_factory=list)

    def skip(self, path: Path, member: str, documents_read: int, error: Exception) -> None:
        logger.warning("Skipping unparseable %s in %s after %d documents: %s", member, path.name, documents_read, error)
        self.skipped_files.append(f"{path.name}:{member}")


# ─────────────────────────────── shared helpers ───────────────────────────────


def source_filename(url: str) -> str:
    """Local file name for a source URL (query string removed)."""
    return Path(url.split("?")[0]).name or url.split("/")[-1]


def _first_lang_tag(value: Any) -> str | None:
    """Extract primary language tag from HPLT-style or plain lang field."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, dict):
            for key in ("language", "lang", "label"):
                if key in first:
                    return str(first[key])
    if isinstance(value, dict):
        for key in ("language", "lang", "label"):
            if key in value:
                return str(value[key])
    return None


# ─────────────────────── Wikimedia XML / BZip2 ───────────────────────────────

_MW_REDIRECT = re.compile(r"^#(?:REDIRECT|PREUSMJERI)\s*\[\[", re.IGNORECASE)
# Links into these namespaces render as images, categories or interlanguage
# links, not as article text; ``[[:Kategorija:X]]`` (leading colon) is a visible link.
_MW_DROP_NAMESPACES = frozenset({"datoteka", "slika", "file", "image", "media", "kategorija", "category"})
_MW_INTERLANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z]+)*$")
# Tags whose content is not running text (tables are parsed as ``table`` tags).
_MW_DROP_TAGS = frozenset(
    {"ref", "references", "gallery", "table", "math", "chem", "score", "timeline", "imagemap", "syntaxhighlight"}
)
_BLANK_LINES = re.compile(r"\n{3,}")
# Bold/italic quotes are left unparsed (parsing them makes the parser give up on
# whole articles with unbalanced quotes) and removed from the text afterwards.
_MW_STYLE_QUOTES = re.compile(r"'{2,}")


def _is_hidden_link(title: str) -> bool:
    prefix, colon, _ = title.strip().partition(":")
    prefix = prefix.strip().lower()
    return bool(colon) and (prefix in _MW_DROP_NAMESPACES or bool(_MW_INTERLANGUAGE.match(prefix)))


def _strip_wikitext(raw: str) -> str:
    """Plain article text: link labels kept; templates, tables, references, files and categories removed."""
    import mwparserfromhell
    from mwparserfromhell.nodes import Node

    if _MW_REDIRECT.match(raw.lstrip()):
        return ""
    code = mwparserfromhell.parse(raw, skip_style_tags=True)
    hidden: list[Node] = [link for link in code.filter_wikilinks() if _is_hidden_link(str(link.title))]
    hidden += [tag for tag in code.filter_tags() if str(tag.tag).strip().lower() in _MW_DROP_TAGS]
    for node in hidden:
        try:
            code.remove(node)
        except ValueError:  # already removed together with an enclosing node
            pass
    text = _MW_STYLE_QUOTES.sub("", code.strip_code(normalize=True, collapse=True))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def read_wikimedia_xml_bz2(path: Path, *, source_id: str) -> Generator[DocumentRecord]:
    """Stream main-namespace articles from a Wikimedia XML dump (bz2-compressed)."""
    from xml.etree import ElementTree as ET

    with bz2.open(path, "rb") as raw:
        context = ET.iterparse(raw, events=("start", "end"))

        # Detect namespace from root element
        ns = ""
        root = None
        for event, elem in context:
            if event == "start":
                root = elem
                m = re.match(r"\{([^}]+)\}", elem.tag)
                ns = f"{{{m.group(1)}}}" if m else ""
                break
        if root is None:
            raise ValueError(f"Wikimedia dump has no root element: {path}")

        in_page = False
        ns_val: str | None = None
        title = ""
        page_id = ""
        text_parts: list[str] = []

        for event, elem in context:
            local = elem.tag.replace(ns, "") if ns else elem.tag

            if event == "start" and local == "page":
                in_page = True
                ns_val = None
                title = ""
                page_id = ""
                text_parts = []
                continue

            if not in_page:
                elem.clear()
                continue

            if event == "end":
                if local == "ns":
                    ns_val = elem.text
                elif local == "title":
                    title = elem.text or ""
                elif local == "id" and not page_id:
                    page_id = elem.text or ""
                elif local == "text":
                    text_parts.append(elem.text or "")
                elif local == "page":
                    in_page = False
                    if ns_val == "0" and text_parts:
                        cleaned = _strip_wikitext("".join(text_parts))
                        if cleaned:
                            yield DocumentRecord(
                                text=cleaned,
                                source_id=source_id,
                                doc_id=page_id,
                                url=f"https://hr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                                language="hr",
                            )
                    elem.clear()
                    root.clear()


# ─────────────────────────── JSONL (gzip / zstd) ─────────────────────────────


def _open_gzip_text(path: Path) -> IO[str]:
    return gzip.open(path, "rt", encoding="utf-8", errors="replace")


def _open_zstd_text(path: Path) -> IO[str]:
    import zstandard

    # zstandard.open owns the file handle, so a failing decompressor setup cannot leak it.
    stream: IO[str] = zstandard.open(path, "rt", encoding="utf-8", errors="replace")
    return stream


def read_jsonl(
    path: Path,
    *,
    opener: Callable[[Path], IO[str]],
    source_id: str,
    text_field: str = "text",
    url_field: str | None = None,
    doc_id_field: str | None = None,
    language_field: str | None = None,
) -> Generator[DocumentRecord]:
    """Stream records from a compressed JSONL file (CLASSLA-web gzip, HPLT 3.0 zstd)."""
    with opener(path) as fh:
        for line_number, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                doc = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            text = doc.get(text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue
            yield DocumentRecord(
                text=text,
                source_id=source_id,
                doc_id=str(doc[doc_id_field]) if doc_id_field and doc_id_field in doc else None,
                url=doc.get(url_field) if url_field else None,
                language=_first_lang_tag(doc.get(language_field)) if language_field else None,
            )


# ─────────────────────────── XZ plain text ───────────────────────────────────


def read_xz_text(path: Path, *, source_id: str) -> Generator[DocumentRecord]:
    """Stream documents from an XZ-compressed plain-text file (CC-100 format).

    Documents are separated by blank lines; paragraphs by single newlines.
    """
    doc_lines: list[str] = []
    doc_index = 0

    with lzma.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "":
                if doc_lines:
                    text = "\n".join(doc_lines).strip()
                    if text:
                        yield DocumentRecord(
                            text=text,
                            source_id=source_id,
                            doc_id=str(doc_index),
                            language="hr",
                        )
                        doc_index += 1
                doc_lines = []
            else:
                doc_lines.append(line)

    if doc_lines:
        text = "\n".join(doc_lines).strip()
        if text:
            yield DocumentRecord(
                text=text,
                source_id=source_id,
                doc_id=str(doc_index),
                language="hr",
            )


# ──────────────────────── CWB vertical / GZip ────────────────────────────────

_VRT_TAG = re.compile(r"^<(/?)(\w+)((?:\s[^>]*)?)/?>\s*$")
_VRT_ATTR = re.compile(r'(\w+)="([^"]*)"')


def read_vertical_xml_gz(path: Path, *, source_id: str) -> Generator[DocumentRecord]:
    """Stream documents from a CWB vertical-format gzip file (hrWaC 2.1).

    Token lines contain: ``word\\tPOS\\tlemma``.  Text is reconstructed from
    column 0 only. hrWaC 2.1 has no document level: ``<p>`` elements sit
    directly under ``<corpus>``, each with its own ``url``, and consecutive
    paragraphs come from different pages (the release is paragraph-shuffled), so
    each ``<p>`` is emitted as one document. ``<g/>`` marks the next token as
    glued to the preceding token.
    """
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        doc_url: str | None = None
        doc_id: str | None = None
        doc_language: str | None = None
        text_parts: list[str] = []
        in_doc = False
        glue_before_next_token = False

        for raw in fh:
            line = raw.rstrip("\n")
            m = _VRT_TAG.match(line)

            if m:
                closing, tag_name = m.group(1), m.group(2).lower()
                if tag_name == "p":
                    if closing:
                        if in_doc and text_parts:
                            text = "".join(text_parts).strip()
                            if text:
                                yield DocumentRecord(
                                    text=text,
                                    source_id=source_id,
                                    doc_id=doc_id,
                                    url=doc_url,
                                    language=doc_language,
                                )
                        in_doc = False
                        text_parts = []
                        doc_url = None
                        doc_id = None
                        doc_language = None
                        glue_before_next_token = False
                    else:
                        in_doc = True
                        attrs = dict(_VRT_ATTR.findall(m.group(3)))
                        doc_url = attrs.get("url") or attrs.get("web_url")
                        doc_id = attrs.get("id")
                        doc_language = attrs.get("lang") or attrs.get("language")
                        text_parts = []
                        glue_before_next_token = False
                elif tag_name == "g" and not closing and in_doc:
                    glue_before_next_token = True
            elif in_doc and "\t" in line:
                word = line.split("\t", 1)[0]
                if word:
                    if text_parts and not glue_before_next_token:
                        text_parts.append(" ")
                    text_parts.append(word)
                    glue_before_next_token = False


# ────────────────────────── MaCoCu XML / ZIP ─────────────────────────────────


def read_macocu_xml_zip(path: Path, *, source_id: str, report: ReadReport) -> Generator[DocumentRecord]:
    """Stream documents from the MaCoCu monolingual XML ZIP archive."""
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".xml"):
                continue
            documents_read = 0
            with zf.open(name) as f:
                try:
                    context = ET.iterparse(f, events=("start", "end"))
                    _, root = next(context)
                    for event, elem in context:
                        if event != "end" or elem.tag.rsplit("}", 1)[-1] != "doc":
                            continue
                        parts = [
                            "".join(paragraph.itertext()).strip()
                            for paragraph in elem.iter()
                            if paragraph.tag.rsplit("}", 1)[-1] == "p"
                        ]
                        text = "\n\n".join(part for part in parts if part)
                        if text:
                            documents_read += 1
                            yield DocumentRecord(
                                text=text,
                                source_id=source_id,
                                doc_id=elem.get("id"),
                                url=elem.get("url"),
                                language=elem.get("lang") or elem.get("language"),
                            )
                        elem.clear()
                        root.clear()
                except ET.ParseError as error:
                    report.skip(path, name, documents_read, error)


# ─────────────────────────── ParlaMint TEI XML ───────────────────────────────

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_XML_NS = "http://www.w3.org/XML/1998/namespace"


def _iter_tei_utterances(root: Any, source_id: str, file_stem: str) -> Generator[DocumentRecord]:
    for u_elem in root.iter(f"{{{_TEI_NS}}}u"):
        uid = u_elem.get(f"{{{_XML_NS}}}id") or u_elem.get("id") or ""
        parts: list[str] = []
        for seg in u_elem.iter(f"{{{_TEI_NS}}}seg"):
            seg_text = "".join(seg.itertext()).strip()
            if seg_text:
                parts.append(seg_text)
        if parts:
            yield DocumentRecord(
                text=" ".join(parts),
                source_id=source_id,
                doc_id=f"{file_stem}:{uid}" if uid else None,
                language="hr",
            )


def read_tei_xml_tgz(path: Path, *, source_id: str, report: ReadReport) -> Generator[DocumentRecord]:
    """Stream utterances from a ParlaMint TEI XML tgz archive."""
    from xml.etree import ElementTree as ET

    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".xml"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            try:
                root = ET.parse(fh).getroot()
            except ET.ParseError as error:
                report.skip(path, member.name, 0, error)
                continue
            stem = Path(member.name).stem
            yield from _iter_tei_utterances(root, source_id, stem)


# ─────────────────────────── MultiEURLEX TAR.GZ ──────────────────────────────


def read_multieurlex_tar_gz(path: Path, *, source_id: str) -> Generator[DocumentRecord]:
    """Stream the Croatian field from every MultiEURLEX split."""
    wanted = {"train.jsonl", "dev.jsonl", "test.jsonl"}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if Path(member.name).name not in wanted:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for line_number, raw in enumerate(stream, start=1):
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"Invalid JSON in {path}:{member.name} at line {line_number}") from exc
                text_by_language = row.get("text")
                if not isinstance(text_by_language, dict):
                    continue
                text = text_by_language.get("hr")
                if isinstance(text, str) and text.strip():
                    yield DocumentRecord(
                        text=text,
                        source_id=source_id,
                        doc_id=str(row.get("celex_id", "")) or None,
                        language="hr",
                    )


# ─────────────────────────── Hugging Face ────────────────────────────────────

# Remote reads fetch 1 MiB blocks: large enough for few requests, small enough that
# column projection skips the large non-text column chunks.
_HF_REMOTE_BLOCK_SIZE = 1 << 20


def _iter_hf_rows(rows: list[dict[str, Any]], entry: HuggingFaceSource) -> Generator[DocumentRecord]:
    for example in rows:
        text = example.get(entry.text_field, "")
        if not isinstance(text, str) or not text.strip():
            continue
        doc_id = str(example[entry.doc_id_field]) if entry.doc_id_field and entry.doc_id_field in example else None
        url_raw = example.get(entry.url_field) if entry.url_field else None
        lang_raw = example.get(entry.language_field) if entry.language_field else None
        yield DocumentRecord(
            text=text,
            source_id=entry.id,
            doc_id=doc_id,
            url=str(url_raw) if url_raw else None,
            language=_first_lang_tag(lang_raw),
        )


def _read_parquet(source: Any, entry: HuggingFaceSource) -> Generator[DocumentRecord]:
    """Documents from one Parquet file (a path or an open file), reading only the entry's columns."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(source).iter_batches(batch_size=10_000, columns=entry.columns()):
        yield from _iter_hf_rows(batch.to_pylist(), entry)


def download_huggingface_files(entry: HfDownloadSource, src_dir: Path) -> list[Path]:
    """Download the entry's Hub Parquet files into *src_dir* (resumable) and return them.

    Files go to *src_dir* rather than the global ``~/.cache/huggingface``, so they
    are recorded in the manifest and deleted with the other source files after
    conversion.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(entry.hf_repo, repo_type="dataset", allow_patterns=entry.hf_data_files, local_dir=src_dir)
    files = sorted(src_dir.glob(entry.hf_data_files))
    if not files:
        raise FileNotFoundError(f"No files matching {entry.hf_data_files!r} in {entry.hf_repo}")
    return files


def read_huggingface_download(entry: HfDownloadSource, src_dir: Path) -> Generator[DocumentRecord]:
    """Read the Hub Parquet files downloaded into *src_dir*."""
    files = sorted(src_dir.glob(entry.hf_data_files))
    if not files:
        raise FileNotFoundError(f"No files matching {entry.hf_data_files!r} in {src_dir}")
    for path in files:
        yield from _read_parquet(path, entry)


def read_huggingface_stream(entry: HfStreamSource) -> Generator[DocumentRecord]:
    """Read the entry's Hub Parquet files remotely, fetching only the needed columns.

    Range requests through ``HfFileSystem`` download only the column chunks that
    are read (about 6 % of a ParlaSpeech file, whose audio column is skipped), and
    nothing is cached on disk. A gated repository raises ``GatedRepoError``.
    """
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    files = sorted(filesystem.glob(f"datasets/{entry.hf_repo}/{entry.hf_data_files}"))
    if not files:
        raise FileNotFoundError(f"No files matching {entry.hf_data_files!r} in {entry.hf_repo}")
    for name in files:
        with filesystem.open(name, block_size=_HF_REMOTE_BLOCK_SIZE) as remote:
            yield from _read_parquet(remote, entry)


# ─────────────────────────── dispatcher ──────────────────────────────────────

_HttpReader = Callable[[Path, HttpSource, ReadReport], Iterator[DocumentRecord]]


def _jsonl_reader(opener: Callable[[Path], IO[str]]) -> _HttpReader:
    def read(path: Path, entry: HttpSource, report: ReadReport) -> Iterator[DocumentRecord]:
        return read_jsonl(
            path,
            opener=opener,
            source_id=entry.id,
            text_field=entry.text_field,
            url_field=entry.url_field,
            doc_id_field=entry.doc_id_field,
            language_field=entry.language_field,
        )

    return read


HTTP_READERS: dict[SourceFormat, _HttpReader] = {
    SourceFormat.WIKIMEDIA_XML_BZ2: lambda path, entry, report: read_wikimedia_xml_bz2(path, source_id=entry.id),
    SourceFormat.JSONL_GZ: _jsonl_reader(_open_gzip_text),
    SourceFormat.JSONL_ZST: _jsonl_reader(_open_zstd_text),
    SourceFormat.XZ_TEXT: lambda path, entry, report: read_xz_text(path, source_id=entry.id),
    SourceFormat.VERTICAL_XML_GZ: lambda path, entry, report: read_vertical_xml_gz(path, source_id=entry.id),
    SourceFormat.MACOCU_XML_ZIP: lambda path, entry, report: read_macocu_xml_zip(
        path, source_id=entry.id, report=report
    ),
    SourceFormat.TEI_XML_TGZ: lambda path, entry, report: read_tei_xml_tgz(path, source_id=entry.id, report=report),
    SourceFormat.MULTIEURLEX_TAR_GZ: lambda path, entry, report: read_multieurlex_tar_gz(path, source_id=entry.id),
}


def stream_source(entry: SourceEntry, src_dir: Path, report: ReadReport) -> Generator[DocumentRecord]:
    """Route *entry* to the correct reader and yield :class:`DocumentRecord` objects.

    For HTTP and downloaded Hugging Face sources *src_dir* must already contain the
    downloaded files (see :func:`download_huggingface_files`); streamed Hugging
    Face sources are read remotely. Skipped archive members are recorded in *report*.
    """
    if isinstance(entry, HfDownloadSource):
        yield from read_huggingface_download(entry, src_dir)
        return
    if isinstance(entry, HfStreamSource):
        yield from read_huggingface_stream(entry)
        return

    reader = HTTP_READERS[entry.source_format]
    for url in entry.urls:
        path = src_dir / source_filename(url)
        if not path.exists():
            raise FileNotFoundError(f"Expected source file not found: {path}")
        yield from reader(path, entry, report)
