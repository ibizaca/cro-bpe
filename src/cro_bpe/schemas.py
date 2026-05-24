from pathlib import Path

from pydantic import BaseModel, Field


class ArtifactDigest(BaseModel):
    path: str = Field()
    sha256: str = Field()
    bytes_size: int = Field(ge=0)

    @classmethod
    def from_path(cls, path: Path, sha256: str, bytes_size: int) -> "ArtifactDigest":
        return cls(path=str(path), sha256=sha256, bytes_size=bytes_size)


class CharacterFrequency(BaseModel):
    character: str = Field()
    count: int = Field(ge=0)


class SourceManifest(BaseModel):
    source_id: str = Field()
    corpus_name: str = Field()
    language: str = Field()
    language_decision: str = Field()
    dataset_name: str = Field()
    dataset_config: str = Field()
    split: str = Field()
    revision: str | None = Field(default=None)
    streaming: bool = Field()
    text_field: str = Field()
    id_field: str = Field()
    title_field: str = Field()
    license: str | None = Field(default=None)
    homepage: str | None = Field(default=None)
    description: str | None = Field(default=None)
    rejected_language_markers: list[str] = Field(default_factory=list)
    statistics: dict[str, int] = Field(default_factory=dict)
    raw_documents: ArtifactDigest = Field()


class PreprocessingSummary(BaseModel):
    unicode_normalization_form: str = Field()
    strip_markup: bool = Field()
    deduplicate_exact: bool = Field()
    min_characters: int = Field()
    min_alpha_ratio: float = Field()
    statistics: dict[str, int] = Field(default_factory=dict)
    raw_documents: ArtifactDigest = Field()
    cleaned_corpus: ArtifactDigest = Field()


class CorpusDiagnostics(BaseModel):
    line_count: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    unique_characters: int = Field(ge=0)
    average_characters_per_line: float = Field(ge=0)
    min_line_length: int = Field(ge=0)
    max_line_length: int = Field(ge=0)
    length_bucket_counts: dict[str, int] = Field(default_factory=dict)
    top_characters: list[CharacterFrequency] = Field(default_factory=list)
    rare_characters: list[CharacterFrequency] = Field(default_factory=list)
    discarded_ratio: float = Field(ge=0)
    corpus: ArtifactDigest = Field()


class TokenizerArtifactManifest(BaseModel):
    model_type: str = Field()
    normalization: str = Field()
    pre_tokenizer: str = Field()
    vocab_size: int = Field(ge=0)
    min_frequency: int = Field(ge=0)
    special_tokens: list[str] = Field(default_factory=list)
    tokenizer_json: ArtifactDigest = Field()
    vocab_file: ArtifactDigest | None = Field(default=None)
    merges_file: ArtifactDigest | None = Field(default=None)
    config_snapshot: ArtifactDigest = Field()
    sample_text: str = Field()
    sample_tokens: list[str] = Field(default_factory=list)
    sample_ids: list[int] = Field(default_factory=list)
    decoded_text: str = Field()
