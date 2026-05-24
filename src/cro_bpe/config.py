import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, computed_field, model_validator

DISALLOWED_LANGUAGE_MARKERS = [
    "bosnian",
    "bosanski",
    "serbian",
    "srpski",
    "serbo-croatian",
    "serbocroatian",
    "sh",
]

APPROVED_SOURCES = {
    "croatian_wikipedia": {
        "dataset_name": "wikimedia/wikipedia",
        "dataset_config": "20231101.hr",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "title_field": "title",
        "language": "croatian",
        "corpus_name": "Croatian Wikipedia",
    }
}


class WorkflowConfig(BaseModel):
    run_download: bool = Field(default=True)
    run_preprocessing: bool = Field(default=True)
    run_feature_engineering: bool = Field(default=True)
    run_model_training: bool = Field(default=True)


class DatasetSourceConfig(BaseModel):
    source_id: str = Field(default="croatian_wikipedia")
    revision: str | None = Field(default=None)
    streaming: bool = Field(default=True)
    max_documents: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source_id(self) -> "DatasetSourceConfig":
        if self.source_id not in APPROVED_SOURCES:
            raise ValueError(f"Source '{self.source_id}' is not in the Croatian allowlist.")

        source_signature = " ".join(str(value).lower() for value in APPROVED_SOURCES[self.source_id].values())
        rejected_markers = [marker for marker in DISALLOWED_LANGUAGE_MARKERS if marker in source_signature]
        if rejected_markers:
            raise ValueError(
                f"Source '{self.source_id}' contains disallowed language markers: {', '.join(rejected_markers)}"
            )
        return self

    @computed_field
    @property
    def dataset_name(self) -> str:
        return APPROVED_SOURCES[self.source_id]["dataset_name"]

    @computed_field
    @property
    def dataset_config(self) -> str:
        return APPROVED_SOURCES[self.source_id]["dataset_config"]

    @computed_field
    @property
    def split(self) -> str:
        return APPROVED_SOURCES[self.source_id]["split"]

    @computed_field
    @property
    def text_field(self) -> str:
        return APPROVED_SOURCES[self.source_id]["text_field"]

    @computed_field
    @property
    def id_field(self) -> str:
        return APPROVED_SOURCES[self.source_id]["id_field"]

    @computed_field
    @property
    def title_field(self) -> str:
        return APPROVED_SOURCES[self.source_id]["title_field"]

    @computed_field
    @property
    def language(self) -> str:
        return APPROVED_SOURCES[self.source_id]["language"]

    @computed_field
    @property
    def corpus_name(self) -> str:
        return APPROVED_SOURCES[self.source_id]["corpus_name"]


class PreprocessingConfig(BaseModel):
    unicode_normalization_form: str = Field(default="NFC")
    strip_markup: bool = Field(default=True)
    deduplicate_exact: bool = Field(default=True)
    min_characters: int = Field(default=20, ge=1)
    min_alpha_ratio: float = Field(default=0.55, ge=0.0, le=1.0)


class DiagnosticsConfig(BaseModel):
    rare_character_threshold: int = Field(default=5, ge=1)
    top_character_limit: int = Field(default=128, ge=1)
    length_buckets: list[int] = Field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048])


class TokenizerTrainingConfig(BaseModel):
    model_type: str = Field(default="bpe")
    normalization: str = Field(default="NFC")
    pre_tokenizer: str = Field(default="byte_level")
    vocab_size: int = Field(default=32000, ge=256)
    min_frequency: int = Field(default=2, ge=1)
    special_tokens: list[str] = Field(default_factory=lambda: ["<s>", "</s>", "<unk>", "<pad>"])
    unknown_token: str = Field(default="<unk>")
    smoke_test_text: str = Field(default="Hrvatski jezik ima bogatu morfologiju i dijakritike.")

    @model_validator(mode="after")
    def validate_special_tokens(self) -> "TokenizerTrainingConfig":
        if self.unknown_token not in self.special_tokens:
            raise ValueError("The unknown token must be part of the special token set.")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Special tokens must be unique.")
        if self.vocab_size <= len(self.special_tokens):
            raise ValueError("Vocabulary size must exceed the number of special tokens.")
        return self


class ArtifactConfig(BaseModel):
    raw_documents_filename: str = Field(default="croatian_wikipedia_raw.jsonl")
    source_manifest_filename: str = Field(default="source_manifest.json")
    preprocessed_corpus_filename: str = Field(default="croatian_corpus.txt")
    preprocessing_summary_filename: str = Field(default="preprocessing_summary.json")
    diagnostics_filename: str = Field(default="corpus_diagnostics.json")
    tokenizer_filename: str = Field(default="croatian_bpe_tokenizer.json")
    tokenizer_prefix: str = Field(default="croatian_bpe")
    tokenizer_manifest_filename: str = Field(default="tokenizer_manifest.json")
    run_config_snapshot_filename: str = Field(default="run_config_snapshot.json")


class RunConfig(BaseModel):
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    source: DatasetSourceConfig = Field(default_factory=DatasetSourceConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    tokenizer: TokenizerTrainingConfig = Field(default_factory=TokenizerTrainingConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    mlflow_experiment_name: str = Field(default="croatian-bpe")
    mlflow_run_name: str = Field(default="croatian-wikipedia-bpe")

    @classmethod
    def from_environment(cls) -> "RunConfig":
        config_path = os.getenv("CRO_BPE_CONFIG_PATH")
        payload: dict[str, object] = {}

        if config_path:
            payload = json.loads(Path(config_path).read_text(encoding="utf-8"))

        _apply_optional_override(payload, ["source", "max_documents"], _read_int("CRO_BPE_MAX_DOCUMENTS"))
        _apply_optional_override(payload, ["source", "revision"], os.getenv("CRO_BPE_REVISION"))
        _apply_optional_override(payload, ["source", "streaming"], _read_bool("CRO_BPE_STREAMING"))
        _apply_optional_override(payload, ["tokenizer", "vocab_size"], _read_int("CRO_BPE_VOCAB_SIZE"))
        _apply_optional_override(payload, ["tokenizer", "min_frequency"], _read_int("CRO_BPE_MIN_FREQUENCY"))
        _apply_optional_override(payload, ["workflow", "run_download"], _read_bool("CRO_BPE_RUN_DOWNLOAD"))
        _apply_optional_override(payload, ["workflow", "run_preprocessing"], _read_bool("CRO_BPE_RUN_PREPROCESSING"))
        _apply_optional_override(
            payload,
            ["workflow", "run_feature_engineering"],
            _read_bool("CRO_BPE_RUN_FEATURE_ENGINEERING"),
        )
        _apply_optional_override(payload, ["workflow", "run_model_training"], _read_bool("CRO_BPE_RUN_MODEL_TRAINING"))
        return cls.model_validate(payload)


def _apply_optional_override(payload: dict[str, object], path: list[str], value: object) -> None:
    if value is None:
        return

    current: dict[str, object] = payload
    for key in path[:-1]:
        next_payload = current.get(key)
        if not isinstance(next_payload, dict):
            next_payload = {}
            current[key] = next_payload
        current = next_payload
    current[path[-1]] = value


def _read_bool(environment_key: str) -> bool | None:
    value = os.getenv(environment_key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {environment_key} must be boolean-like, got '{value}'.")


def _read_int(environment_key: str) -> int | None:
    value = os.getenv(environment_key)
    if value is None or not value.strip():
        return None
    return int(value)
