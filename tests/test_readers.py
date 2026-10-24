"""Source readers on tiny fixture files, one per format."""

import bz2
import gzip
import io
import json
import lzma
import tarfile
import zipfile
from pathlib import Path

import pytest
import zstandard

from data_import.catalog import SOURCES, SourceFormat
from data_import.readers import (
    ReadReport,
    _strip_wikitext,
    read_macocu_xml_zip,
    read_multieurlex_tar_gz,
    read_tei_xml_tgz,
    read_vertical_xml_gz,
    read_wikimedia_xml_bz2,
    read_xz_text,
    source_filename,
    stream_source,
)

_WIKI_MARKUP = ("|", "{{", "}}", "{|", "|}", "[[", "]]", "<ref", "''")

_JSONL_ROWS = [
    {"id": 1, "text": "Prvi dokument.", "u": "http://a.hr", "lang": ["hrv_Latn", "bos_Latn"]},
    {"id": 2, "text": "   "},  # blank text is skipped
    {"id": 3, "text": "Drugi dokument.", "u": "http://b.hr", "lang": "hr"},
]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[[Zagreb|glavni grad]] i [[Split]]", "glavni grad i Split"),
        ("A {{Infookvir|x={{lang|en|y}}|z=1}} B", "A B"),
        ("'''Zagreb''' je ''glavni'' grad.", "Zagreb je glavni grad."),
        ('Tekst.<ref>{{cite web|url=x}}</ref> Dalje.<ref name="a" />', "Tekst. Dalje."),
        ('Prije\n{| class="wikitable"\n! A !! B\n|-\n| 1 || 2\n|}\nPoslije', "Prije\n\nPoslije"),
        ("[[Datoteka:Z.jpg|mini|opis]] [[Slika:x.png]] Tekst [[Kategorija:Gradovi]] [[en:Zagreb]]", "Tekst"),
        ("[[Hrvatska|Hrvatske]].", "Hrvatske."),
        ("== Povijest ==\nPrvi spomen.", "Povijest\nPrvi spomen."),
        ("#PREUSMJERI [[Zagreb]]", ""),
        ("#REDIRECT [[Zagreb]]", ""),
    ],
)
def test_strip_wikitext(raw: str, expected: str) -> None:
    assert _strip_wikitext(raw) == expected


def test_strip_wikitext_leaves_no_markup_on_a_full_article() -> None:
    raw = """{{Infookvir grad|ime=Zagreb|slika=[[Datoteka:Z.jpg|250px]]|stanovnika={{formatnum:767131}}}}
'''Zagreb''' je [[glavni grad|glavni i najveći grad]] [[Hrvatska|Hrvatske]].<ref>{{cite web|url=x}}</ref>
[[Datoteka:Katedrala.jpg|mini|Zagrebačka katedrala]]
{| class="wikitable"
! Godina !! Stanovnika
|-
| 2021 || 767.131
|}
== Povijest ==
Prvi spomen je iz 1094.<ref name="a" />
[[Kategorija:Gradovi u Hrvatskoj]]"""
    text = _strip_wikitext(raw)
    assert not [marker for marker in _WIKI_MARKUP if marker in text]
    assert text == "Zagreb je glavni i najveći grad Hrvatske.\n\nPovijest\nPrvi spomen je iz 1094."


def test_read_wikimedia_keeps_articles_only(tmp_path: Path) -> None:
    dump = _wiki_dump(
        tmp_path / "wiki.xml.bz2",
        [
            ("Zagreb", "0", "'''Zagreb''' je [[glavni grad]]."),
            ("Razgovor:Zagreb", "1", "Talk page."),
            ("Zagreb (grad)", "0", "#PREUSMJERI [[Zagreb]]"),
            ("Lijepa naša", "0", "Himna."),
        ],
    )
    docs = list(read_wikimedia_xml_bz2(dump, source_id="hr-wikipedia"))
    assert [(d.doc_id, d.text, d.url) for d in docs] == [
        ("1", "Zagreb je glavni grad.", "https://hr.wikipedia.org/wiki/Zagreb"),
        ("4", "Himna.", "https://hr.wikipedia.org/wiki/Lijepa_naša"),
    ]


@pytest.mark.parametrize("source_format", [SourceFormat.JSONL_GZ, SourceFormat.JSONL_ZST])
def test_read_jsonl(tmp_path: Path, source_format: SourceFormat) -> None:
    payload = "\n".join(json.dumps(row) for row in _JSONL_ROWS).encode() + b"\n\n"
    entry = next(s for s in SOURCES if s.id == "hplt-3.0-hrv-latn").model_copy(
        update={"source_format": source_format, "urls": ["https://example.org/data/part.jsonl?x=1"]}
    )
    compressed = gzip.compress(payload) if source_format == SourceFormat.JSONL_GZ else zstandard.compress(payload)
    (tmp_path / "part.jsonl").write_bytes(compressed)
    docs = list(stream_source(entry, tmp_path, ReadReport()))
    assert [(d.doc_id, d.text, d.url, d.language) for d in docs] == [
        ("1", "Prvi dokument.", "http://a.hr", "hrv_Latn"),
        ("3", "Drugi dokument.", "http://b.hr", "hr"),
    ]


def test_read_jsonl_rejects_invalid_json(tmp_path: Path) -> None:
    entry = next(s for s in SOURCES if s.id == "classla-web-hr-2")
    (tmp_path / source_filename(entry.urls[0])).write_bytes(gzip.compress(b'{"text": "ok"}\n{broken\n'))
    with pytest.raises(ValueError, match="line 2"):
        list(stream_source(entry, tmp_path, ReadReport()))


def test_missing_source_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(stream_source(next(s for s in SOURCES if s.id == "cc-100-hr"), tmp_path, ReadReport()))


def test_read_xz_text(tmp_path: Path) -> None:
    path = tmp_path / "hr.txt.xz"
    path.write_bytes(lzma.compress(b"Prvi red.\nDrugi red.\n\n\nNovi dokument.\n\nZadnji"))
    docs = list(read_xz_text(path, source_id="cc-100-hr"))
    assert [(d.doc_id, d.text) for d in docs] == [
        ("0", "Prvi red.\nDrugi red."),
        ("1", "Novi dokument."),
        ("2", "Zadnji"),
    ]


def test_read_vertical_glues_tokens(tmp_path: Path) -> None:
    vertical = (
        '<corpus id="hrWaC2.1.01" lang="hr">\n'
        '<p url="http://a.hr" lang="hr">\n<s>\n'
        "Dobar\tA\tdobar\ndan\tN\tdan\n<g/>\n,\tZ\t,\nsvijete\tN\tsvijet\n<g/>\n!\tZ\t!\n</s>\n</p>\n"
        '<p url="http://b.hr">\n<s>\nDrugi\tA\tdrugi\n</s>\n</p>\n</corpus>\n'
    )
    path = tmp_path / "hrwac.xml.gz"
    path.write_bytes(gzip.compress(vertical.encode()))
    docs = list(read_vertical_xml_gz(path, source_id="hrwac-2.1"))
    assert [(d.text, d.url) for d in docs] == [("Dobar dan, svijete!", "http://a.hr"), ("Drugi", "http://b.hr")]


def test_read_macocu_reports_unparseable_files(tmp_path: Path) -> None:
    path = tmp_path / "macocu.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.xml", '<corpus><doc id="1" url="u"><p>Prvi.</p><p>Drugi odlomak.</p></doc></corpus>')
        archive.writestr("b.xml", '<corpus><doc id="2"><p>Prije greške.</p></doc><doc id="3"><p>Nedovrš')
        archive.writestr("c.xml", '<corpus><doc id="4"><p>Poslije.</p></doc></corpus>')
    report = ReadReport()
    docs = list(read_macocu_xml_zip(path, source_id="macocu-hr-2", report=report))
    assert [(d.doc_id, d.text) for d in docs] == [
        ("1", "Prvi.\n\nDrugi odlomak."),
        ("2", "Prije greške."),
        ("4", "Poslije."),
    ]
    assert report.skipped_files == ["macocu.zip:b.xml"]


def test_read_tei_reports_unparseable_files(tmp_path: Path) -> None:
    tei = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>'
        '<u xml:id="u1"><seg>Poštovani</seg><seg>zastupnici.</seg></u>'
        '<u xml:id="u2"><seg>Hvala.</seg></u></body></text></TEI>'
    )
    path = tmp_path / "parlamint.tgz"
    with tarfile.open(path, "w:gz") as archive:
        _add_member(archive, "ParlaMint-HR/2020/session.xml", tei.encode())
        _add_member(archive, "ParlaMint-HR/2020/broken.xml", b"<TEI><text>")
    report = ReadReport()
    docs = list(read_tei_xml_tgz(path, source_id="parlamint-hr-5", report=report))
    assert [(d.doc_id, d.text) for d in docs] == [("session:u1", "Poštovani zastupnici."), ("session:u2", "Hvala.")]
    assert report.skipped_files == ["parlamint.tgz:ParlaMint-HR/2020/broken.xml"]


def test_read_multieurlex_takes_croatian_only(tmp_path: Path) -> None:
    rows = [
        {"celex_id": "32010R0001", "text": {"en": "Regulation", "hr": "Uredba"}},
        {"celex_id": "32010R0002", "text": {"en": "Only English"}},
    ]
    path = tmp_path / "multi_eurlex.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        _add_member(archive, "data/train.jsonl", "\n".join(json.dumps(r) for r in rows).encode())
        _add_member(archive, "data/other.jsonl", json.dumps(rows[0]).encode())
    docs = list(read_multieurlex_tar_gz(path, source_id="multi-eurlex-hr"))
    assert [(d.doc_id, d.text) for d in docs] == [("32010R0001", "Uredba")]


def _wiki_dump(path: Path, pages: list[tuple[str, str, str]]) -> Path:
    ns = "http://www.mediawiki.org/xml/export-0.11/"
    body = "".join(
        f"<page><title>{title}</title><ns>{namespace}</ns><id>{i}</id>"
        f"<revision><id>{100 + i}</id><text>{text}</text></revision></page>"
        for i, (title, namespace, text) in enumerate(pages, start=1)
    )
    path.write_bytes(bz2.compress(f'<mediawiki xmlns="{ns}">{body}</mediawiki>'.encode()))
    return path


def _add_member(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))
