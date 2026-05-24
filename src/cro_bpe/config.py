from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    language_code: str = Field(default="hr")
    dataset_name: str = Field(default="Croatian Wikipedia")
    dump_url: str = Field(
        default="https://dumps.wikimedia.org/hrwiki/latest/hrwiki-latest-pages-articles-multistream.xml.bz2"
    )

    @field_validator("language_code")
    @staticmethod
    def validate_language_code(value: str) -> str:
        if value != "hr":
            raise ValueError("Only Croatian language code 'hr' is allowed.")
        return value

    @field_validator("dump_url")
    @staticmethod
    def validate_dump_url(value: str) -> str:
        if "dumps.wikimedia.org/hrwiki/" not in value:
            raise ValueError("Only the Croatian Wikipedia dump URL is allowed.")
        return value


class DirectoryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_directory_name: str = Field(default="data")
    logs_directory_name: str = Field(default="logs")

    @property
    def data_directory(self) -> Path:
        return self.project_root / self.data_directory_name

    @property
    def raw_directory(self) -> Path:
        return self.data_directory / "raw"

    @property
    def preprocessed_directory(self) -> Path:
        return self.data_directory / "preprocessed"

    @property
    def models_directory(self) -> Path:
        return self.data_directory / "models"

    @property
    def logs_directory(self) -> Path:
        return self.project_root / self.logs_directory_name


class CorpusConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_document_characters: int = Field(default=200, ge=1)
    article_limit: int | None = Field(default=None, ge=1)
    evaluation_document_limit: int = Field(default=1000, ge=1)


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    vocab_size: int = Field(default=32000, ge=256)
    min_frequency: int = Field(default=2, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: ["[UNK]", "[PAD]", "[BOS]", "[EOS]"])


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_size_bytes: int = Field(default=1024 * 1024, ge=1024)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: SourceConfig = Field(default_factory=SourceConfig)
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def dump_path(self) -> Path:
        return self.directories.raw_directory / Path(self.source.dump_url).name

    @property
    def corpus_path(self) -> Path:
        return self.directories.preprocessed_directory / "croatian_wikipedia_corpus.txt"

    @property
    def tokenizer_json_path(self) -> Path:
        return self.directories.models_directory / "tokenizer.json"

    @property
    def vocab_path(self) -> Path:
        return self.directories.models_directory / "vocab.json"

    @property
    def merges_path(self) -> Path:
        return self.directories.models_directory / "merges.txt"

    @property
    def metadata_path(self) -> Path:
        return self.directories.models_directory / "training_metadata.json"

    @property
    def evaluation_path(self) -> Path:
        return self.directories.models_directory / "evaluation.json"
