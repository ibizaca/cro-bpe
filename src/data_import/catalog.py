"""Catalog of Croatian text corpora for the downloading pipeline.

Every source that should be ingested, every source that is temporarily blocked
due to access restrictions, and every source that is definitively excluded are
all declared here.  The rest of the pipeline derives its behaviour entirely from
these declarations.
"""

from enum import Enum
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceFormat(str, Enum):
    """File format of an HTTP source; each has a reader in :mod:`data_import.readers`."""

    WIKIMEDIA_XML_BZ2 = "wikimedia_xml_bz2"
    JSONL_GZ = "jsonl_gz"
    JSONL_ZST = "jsonl_zst"
    XZ_TEXT = "xz_text"
    VERTICAL_XML_GZ = "vertical_xml_gz"
    MACOCU_XML_ZIP = "macocu_xml_zip"
    TEI_XML_TGZ = "tei_xml_tgz"
    MULTIEURLEX_TAR_GZ = "multieurlex_tar_gz"


class BlockedReason(str, Enum):
    ACCESS_SUSPENDED = "access_suspended"
    NO_CROATIAN_PARTITION = "no_croatian_partition"
    BCMS_ONLY = "bcms_only"
    SUPERSEDED = "superseded"
    NO_BULK_DOWNLOAD = "no_bulk_download"
    NOT_MONOLINGUAL = "not_monolingual"
    AUTH_REQUIRED = "auth_required"


class _Source(BaseModel):
    """Fields shared by every active source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable machine-readable identifier.")
    name: str = Field(description="Human-readable source name.")
    text_field: str = Field(default="text", description="Field name holding document text.")
    language_field: str | None = Field(default=None, description="Field holding language tag(s).")
    url_field: str | None = Field(default=None, description="Field holding source URL.")
    doc_id_field: str | None = Field(default=None, description="Field holding document ID.")
    version: str | None = None
    license: str | None = None
    notes: str | None = None


class HttpSource(_Source):
    """Archives downloaded over HTTP into ``data/raw/<id>/source`` and read locally."""

    kind: Literal["http"] = "http"
    source_format: SourceFormat
    urls: list[str] = Field(min_length=1, description="Ordered list of source-file URLs.")


class _HuggingFaceSource(_Source):
    hf_repo: str = Field(description="Hugging Face dataset repository ID.")
    hf_data_files: str = Field(description="Glob of the Parquet files to read, relative to the repository root.")
    requires_hf_token: bool = Field(default=False, description="Gated dataset; recorded as blocked without access.")

    def columns(self) -> list[str]:
        """The Parquet columns a reader needs; any other columns are never read."""
        fields = (self.text_field, self.doc_id_field, self.url_field, self.language_field)
        return [column for column in fields if column is not None]


class HfDownloadSource(_HuggingFaceSource):
    """Hub Parquet files downloaded into ``data/raw/<id>/source`` (resumable), then read locally."""

    kind: Literal["hf_download"] = "hf_download"


class HfStreamSource(_HuggingFaceSource):
    """Hub Parquet files read remotely, fetching only the needed columns.

    For repositories whose files also hold large non-text columns (ParlaSpeech
    audio): Parquet column projection downloads only the text column chunks.
    Nothing is written to disk.
    """

    kind: Literal["hf_stream"] = "hf_stream"


SourceEntry = Annotated[HttpSource | HfDownloadSource | HfStreamSource, Field(discriminator="kind")]
HuggingFaceSource = HfDownloadSource | HfStreamSource


class BlockedEntry(BaseModel):
    """A corpus that the pipeline cannot currently download; documented for completeness."""

    id: str
    name: str
    reason: BlockedReason
    note: str
    clarin_handle: str | None = None
    download_url: str | None = None
    source_format_hint: str | None = None
    license: str | None = None


class ExcludedEntry(BaseModel):
    """A corpus that is definitively excluded from ingestion."""

    id: str
    name: str
    reason: BlockedReason
    note: str


# ─────────────────────────── active sources ──────────────────────────────────

_HPLT3_BASE: Final = "https://data.hplt-project.org/three/sorted/hrv_Latn"
# Quality bins from highest (10) to lowest (5); one shard per bin for Croatian.
# Map file: https://data.hplt-project.org/three/sorted/hrv_Latn.map
HPLT3_QUALITY_BINS: Final[tuple[str, ...]] = ("10", "9", "8", "7", "6", "5")

SOURCES: Final[list[SourceEntry]] = [
    HttpSource(
        id="hr-wikipedia",
        name="Croatian Wikipedia (latest dump)",
        source_format=SourceFormat.WIKIMEDIA_XML_BZ2,
        urls=["https://dumps.wikimedia.org/hrwiki/latest/hrwiki-latest-pages-articles.xml.bz2"],
        version="latest",
        license="CC-BY-SA 4.0",
        notes="Canonical Croatian encyclopedia; namespace-0 articles only.",
    ),
    HttpSource(
        id="hplt-3.0-hrv-latn",
        name="HPLT 3.0 hrv_Latn (all quality bins)",
        source_format=SourceFormat.JSONL_ZST,
        urls=[f"{_HPLT3_BASE}/{b}_1.jsonl.zst" for b in HPLT3_QUALITY_BINS],
        text_field="text",
        language_field="lang",
        url_field="u",
        doc_id_field="id",
        version="3.0",
        license="CC0 1.0",
        notes=(
            "Data at data.hplt-project.org (not on Hugging Face directly). "
            "Quality bins 5–10 (Web Docs Scorer integer), one shard each for Croatian."
        ),
    ),
    HfDownloadSource(
        id="fineweb-2-hrv-latn",
        name="FineWeb-2 hrv_Latn",
        hf_repo="HuggingFaceFW/fineweb-2",
        hf_data_files="data/hrv_Latn/train/*.parquet",
        text_field="text",
        language_field="language",
        url_field="url",
        doc_id_field="id",
        version="1.0",
        license="ODC-By 1.0",
        notes="Config is 'hrv_Latn' (ISO 639-3 + script), not 'hr'.",
    ),
    HttpSource(
        id="classla-web-hr-2",
        name="CLASSLA-web.hr 2.0",
        source_format=SourceFormat.JSONL_GZ,
        urls=["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2079/CLASSLA-web.hr.2.0.jsonl.gz"],
        text_field="text",
        url_field="url",
        doc_id_field="id",
        version="2.0",
        license="CC0 1.0",
        notes="CLARIN.SI handle 11356/2079. 3.01B words; CC0/PUB access (no login required).",
    ),
    HttpSource(
        id="cc-100-hr",
        name="CC-100 Croatian",
        source_format=SourceFormat.XZ_TEXT,
        urls=["https://data.statmt.org/cc-100/hr.txt.xz"],
        version="2018-CC",
        license="Public domain (CC-NET pipeline)",
        notes="5.7 GB compressed; documents separated by blank lines.",
    ),
    HttpSource(
        id="macocu-hr-2",
        name="MaCoCu-hr 2.0",
        source_format=SourceFormat.MACOCU_XML_ZIP,
        urls=["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1806/MaCoCu-hr-2.0.xml.zip"],
        version="2.0",
        license="CC0 1.0",
        notes="Canonical CLARIN.SI handle 11356/1806.",
    ),
    HttpSource(
        id="hrwac-2.1",
        name="hrWaC 2.1",
        source_format=SourceFormat.VERTICAL_XML_GZ,
        urls=[
            f"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1064/hrWaC2.1.{part:02d}.xml.gz"
            for part in range(1, 15)
        ],
        version="2.1",
        license="CC-BY-SA 4.0",
        notes="Canonical CLARIN.SI handle 11356/1064; all 14 corpus parts.",
    ),
    HttpSource(
        id="parlamint-hr-5",
        name="ParlaMint-HR 5.0",
        source_format=SourceFormat.TEI_XML_TGZ,
        urls=["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2004/ParlaMint-HR.tgz"],
        version="5.0",
        license="CC-BY 4.0",
        notes="Croatian component only from canonical CLARIN.SI handle 11356/2004.",
    ),
    HttpSource(
        id="multi-eurlex-hr",
        name="MultiEURLEX hr",
        source_format=SourceFormat.MULTIEURLEX_TAR_GZ,
        urls=["https://huggingface.co/datasets/coastalcph/multi_eurlex/resolve/main/data/multi_eurlex.tar.gz"],
        text_field="text",
        doc_id_field="celex_id",
        version="1.0",
        license="CC-BY 4.0",
        notes="Croatian EU legal texts; ~15 k documents across all splits.",
    ),
    HfStreamSource(
        id="parlaspeech-hr",
        name="ParlaSpeech-HR 2.0 (text transcripts only)",
        hf_repo="classla/ParlaSpeech-HR",
        hf_data_files="data/train-*.parquet",
        text_field="text_normalised",
        version="2.0",
        license="CC-BY 4.0",
        notes=(
            "Uses text_normalised (parliamentary comments stripped). "
            "CLARIN.SI: http://hdl.handle.net/11356/1914. "
            "Only the text column is fetched; the 179 GB of audio is never downloaded."
        ),
    ),
    HfStreamSource(
        id="culturax-hr",
        name="CulturaX hr",
        hf_repo="uonlp/CulturaX",
        hf_data_files="hr/*.parquet",
        requires_hf_token=True,
        text_field="text",
        url_field="url",
        version="1.0",
        license="ODC-By / OSCAR license",
        notes=(
            "Gated dataset: requires a Hugging Face account, acceptance of the access "
            "agreement form, and the HF_TOKEN environment variable. "
            "If HF_TOKEN is absent or access is denied this source is recorded as blocked."
        ),
    ),
]

# ─────────────────────────── blocked sources ─────────────────────────────────
# Sources that become temporarily unavailable can be moved here without
# pretending that a blocked download completed successfully.
BLOCKED: Final[list[BlockedEntry]] = []

# ─────────────────────────── excluded sources ────────────────────────────────

EXCLUDED: Final[list[ExcludedEntry]] = [
    ExcludedEntry(
        id="oscar-21-23",
        name="OSCAR 21.09 / 22.01 / 23.01",
        reason=BlockedReason.ACCESS_SUSPENDED,
        note="Upstream access suspended as of plan date; not ingested until restored.",
    ),
    ExcludedEntry(
        id="mc4",
        name="mC4",
        reason=BlockedReason.NO_CROATIAN_PARTITION,
        note="No Croatian partition exists in mC4.",
    ),
    ExcludedEntry(
        id="hplt-1.2-hbs",
        name="HPLT 1.2 hbs / SH Wikipedia / hbs·sh·bs·sr partitions",
        reason=BlockedReason.BCMS_ONLY,
        note="Not Croatian-only; mixes Serbian, Bosnian, Montenegrin, and Croatian under the BCMS macro-language code.",
    ),
    ExcludedEntry(
        id="superseded",
        name="MaCoCu-hr 1.0 · CLASSLA-web.hr 1.0 · HPLT 2.0 · filtered copies",
        reason=BlockedReason.SUPERSEDED,
        note="Superseded by later versions or filtered copies of canonical sources ingested above.",
    ),
    ExcludedEntry(
        id="cnk-riznica",
        name="Croatian National Corpus / Riznica",
        reason=BlockedReason.NO_BULK_DOWNLOAD,
        note="No working bulk-download endpoint available.",
    ),
    ExcludedEntry(
        id="opus-parallel",
        name="OPUS parallel corpora · benchmark and annotation datasets",
        reason=BlockedReason.NOT_MONOLINGUAL,
        note=("Not substantive monolingual Croatian corpora. Would duplicate or distort the training distribution."),
    ),
]
