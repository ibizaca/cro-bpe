from collections import Counter

from cro_bpe.config import RunConfig
from cro_bpe.schemas import CharacterFrequency, CorpusDiagnostics, PreprocessingSummary
from cro_bpe.utils import ProjectPaths, artifact_digest, get_logger, read_json, write_json

logger = get_logger(__name__)


def run(config: RunConfig, paths: ProjectPaths) -> CorpusDiagnostics:
    logger.info("Starting corpus diagnostics")

    corpus_path = paths.preprocessed_dir / config.artifacts.preprocessed_corpus_filename
    preprocessing_summary_path = paths.preprocessed_dir / config.artifacts.preprocessing_summary_filename
    if not corpus_path.exists():
        raise FileNotFoundError(f"Preprocessed corpus not found at {corpus_path}")

    character_counter: Counter[str] = Counter()
    bucket_counts = _initialize_bucket_counts(config.diagnostics.length_buckets)
    line_count = 0
    total_characters = 0
    total_bytes = 0
    min_line_length = 0
    max_line_length = 0

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            document = line.rstrip("\n")
            if not document:
                continue
            line_length = len(document)
            line_count += 1
            total_characters += line_length
            total_bytes += len(document.encode("utf-8"))
            min_line_length = line_length if min_line_length == 0 else min(min_line_length, line_length)
            max_line_length = max(max_line_length, line_length)
            character_counter.update(document)
            _add_to_bucket(bucket_counts, config.diagnostics.length_buckets, line_length)

    preprocessing_summary = PreprocessingSummary.model_validate(read_json(preprocessing_summary_path))
    discarded_documents = (
        preprocessing_summary.statistics["discarded_empty"]
        + preprocessing_summary.statistics["discarded_short"]
        + preprocessing_summary.statistics["discarded_low_signal"]
        + preprocessing_summary.statistics["discarded_duplicates"]
    )
    total_input_documents = preprocessing_summary.statistics["input_documents"] or 1

    diagnostics = CorpusDiagnostics(
        line_count=line_count,
        total_characters=total_characters,
        total_bytes=total_bytes,
        unique_characters=len(character_counter),
        average_characters_per_line=(total_characters / line_count) if line_count else 0.0,
        min_line_length=min_line_length,
        max_line_length=max_line_length,
        length_bucket_counts=bucket_counts,
        top_characters=[
            CharacterFrequency(character=character, count=count)
            for character, count in character_counter.most_common(config.diagnostics.top_character_limit)
        ],
        rare_characters=[
            CharacterFrequency(character=character, count=count)
            for character, count in sorted(character_counter.items(), key=lambda item: (item[1], item[0]))
            if count <= config.diagnostics.rare_character_threshold
        ],
        discarded_ratio=discarded_documents / total_input_documents,
        corpus=artifact_digest(corpus_path),
    )
    write_json(paths.features_dir / config.artifacts.diagnostics_filename, diagnostics.model_dump(mode="json"))
    logger.info("Computed diagnostics for %s documents", diagnostics.line_count)
    return diagnostics


def _initialize_bucket_counts(bucket_edges: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    lower_edge = 0
    for upper_edge in bucket_edges:
        counts[f"{lower_edge + 1}-{upper_edge}"] = 0
        lower_edge = upper_edge
    counts[f"{bucket_edges[-1] + 1}+"] = 0
    return counts


def _add_to_bucket(bucket_counts: dict[str, int], bucket_edges: list[int], line_length: int) -> None:
    lower_edge = 0
    for upper_edge in bucket_edges:
        if line_length <= upper_edge:
            bucket_counts[f"{lower_edge + 1}-{upper_edge}"] += 1
            return
        lower_edge = upper_edge
    bucket_counts[f"{bucket_edges[-1] + 1}+"] += 1
