import json
import logging

from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.normalizers import Replace, Strip
from tokenizers.pre_tokenizers import Metaspace

from cro_bpe.config import PipelineConfig
from cro_bpe.normalize import normalize_croatian_text
from cro_bpe.wiki import download_croatian_wikipedia_dump, ensure_directories, iter_croatian_articles

logger = logging.getLogger(__name__)


class CorpusStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_articles_seen: int = Field(ge=0)
    retained_articles: int = Field(ge=0)
    retained_characters: int = Field(ge=0)
    retained_words: int = Field(ge=0)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    evaluated_documents: int = Field(ge=0)
    average_tokens_per_document: float = Field(ge=0)
    average_characters_per_token: float = Field(ge=0)
    exact_roundtrip_matches: int = Field(ge=0)


class TrainingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_url: str
    language_code: str
    dump_path: str
    corpus_path: str
    tokenizer_json_path: str
    vocab_path: str
    merges_path: str
    corpus_statistics: CorpusStatistics
    evaluation: EvaluationReport
    vocab_size: int = Field(ge=256)
    min_frequency: int = Field(ge=1)
    special_tokens: list[str]


def prepare_corpus(config: PipelineConfig) -> CorpusStatistics:
    ensure_directories(config)
    dump_path = download_croatian_wikipedia_dump(config)

    total_articles_seen = 0
    retained_articles = 0
    retained_characters = 0
    retained_words = 0

    with config.corpus_path.open("w", encoding="utf-8") as corpus_file:
        for article_text in iter_croatian_articles(dump_path=dump_path, article_limit=config.corpus.article_limit):
            total_articles_seen += 1
            if len(article_text) < config.corpus.min_document_characters:
                continue

            corpus_file.write(article_text)
            corpus_file.write("\n")
            retained_articles += 1
            retained_characters += len(article_text)
            retained_words += len(article_text.split())

    corpus_statistics = CorpusStatistics(
        total_articles_seen=total_articles_seen,
        retained_articles=retained_articles,
        retained_characters=retained_characters,
        retained_words=retained_words,
    )
    logger.info(
        "Prepared %s retained Croatian articles from %s parsed articles at %s",
        retained_articles,
        total_articles_seen,
        config.corpus_path,
    )
    return corpus_statistics


def build_tokenizer(config: PipelineConfig) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=config.tokenizer.special_tokens[0]))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), Replace(Regex(r"\s+"), " "), Strip()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Metaspace(replacement="▁", prepend_scheme="always")])
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    return tokenizer


def train_tokenizer(config: PipelineConfig) -> None:
    ensure_directories(config)
    if not config.corpus_path.exists():
        logger.info("Corpus file is missing; preparing it before training")
        prepare_corpus(config)

    tokenizer = build_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        show_progress=True,
        special_tokens=config.tokenizer.special_tokens,
    )
    tokenizer.train(files=[str(config.corpus_path)], trainer=trainer)
    tokenizer.save(str(config.tokenizer_json_path))
    tokenizer.model.save(str(config.directories.models_directory))
    logger.info("Saved tokenizer artifacts to %s", config.directories.models_directory)


def evaluate_tokenizer(config: PipelineConfig) -> EvaluationReport:
    tokenizer = Tokenizer.from_file(str(config.tokenizer_json_path))

    evaluated_documents = 0
    token_count = 0
    character_count = 0
    exact_roundtrip_matches = 0

    with config.corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            document = line.strip()
            if not document:
                continue

            encoding = tokenizer.encode(document)
            decoded_text = normalize_croatian_text(tokenizer.decode(encoding.ids))
            normalized_document = normalize_croatian_text(document)

            evaluated_documents += 1
            token_count += len(encoding.tokens)
            character_count += len(document)
            if decoded_text == normalized_document:
                exact_roundtrip_matches += 1

            if evaluated_documents >= config.corpus.evaluation_document_limit:
                break

    if evaluated_documents == 0 or token_count == 0:
        return EvaluationReport(
            evaluated_documents=evaluated_documents,
            average_tokens_per_document=0.0,
            average_characters_per_token=0.0,
            exact_roundtrip_matches=exact_roundtrip_matches,
        )

    return EvaluationReport(
        evaluated_documents=evaluated_documents,
        average_tokens_per_document=token_count / evaluated_documents,
        average_characters_per_token=character_count / token_count,
        exact_roundtrip_matches=exact_roundtrip_matches,
    )


def write_metadata(config: PipelineConfig, corpus_statistics: CorpusStatistics, evaluation: EvaluationReport) -> None:
    ensure_directories(config)
    metadata = TrainingMetadata(
        source_name=config.source.dataset_name,
        source_url=str(config.source.dump_url),
        language_code=config.source.language_code,
        dump_path=str(config.dump_path),
        corpus_path=str(config.corpus_path),
        tokenizer_json_path=str(config.tokenizer_json_path),
        vocab_path=str(config.vocab_path),
        merges_path=str(config.merges_path),
        corpus_statistics=corpus_statistics,
        evaluation=evaluation,
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        special_tokens=config.tokenizer.special_tokens,
    )
    config.metadata_path.write_text(json.dumps(metadata.model_dump(mode="json"), indent=2), encoding="utf-8")
    config.evaluation_path.write_text(json.dumps(evaluation.model_dump(mode="json"), indent=2), encoding="utf-8")


def run_training_pipeline(config: PipelineConfig) -> None:
    corpus_statistics = prepare_corpus(config)
    train_tokenizer(config)
    evaluation = evaluate_tokenizer(config)
    write_metadata(config, corpus_statistics, evaluation)
