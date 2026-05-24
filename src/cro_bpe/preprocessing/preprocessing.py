import hashlib

from cro_bpe.config import RunConfig
from cro_bpe.schemas import PreprocessingSummary
from cro_bpe.utils import ProjectPaths, artifact_digest, get_logger, iter_jsonl, write_json
from cro_bpe.utils.text import alpha_ratio, normalize_text

logger = get_logger(__name__)


def run(config: RunConfig, paths: ProjectPaths) -> PreprocessingSummary:
    logger.info("Starting preprocessing for Croatian corpus")

    raw_path = paths.raw_dir / config.artifacts.raw_documents_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset artifact not found at {raw_path}")

    output_path = paths.preprocessed_dir / config.artifacts.preprocessed_corpus_filename

    seen_hashes: set[str] = set()
    statistics = {
        "input_documents": 0,
        "kept_documents": 0,
        "discarded_empty": 0,
        "discarded_short": 0,
        "discarded_low_signal": 0,
        "discarded_duplicates": 0,
        "input_bytes": 0,
        "output_bytes": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(raw_path):
            statistics["input_documents"] += 1
            source_text = str(record.get("text", ""))
            statistics["input_bytes"] += len(source_text.encode("utf-8"))

            normalized_text = normalize_text(source_text, config.preprocessing)
            if not normalized_text:
                statistics["discarded_empty"] += 1
                continue
            if len(normalized_text) < config.preprocessing.min_characters:
                statistics["discarded_short"] += 1
                continue
            if alpha_ratio(normalized_text) < config.preprocessing.min_alpha_ratio:
                statistics["discarded_low_signal"] += 1
                continue

            if config.preprocessing.deduplicate_exact:
                text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    statistics["discarded_duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)

            handle.write(normalized_text + "\n")
            statistics["kept_documents"] += 1
            statistics["output_bytes"] += len((normalized_text + "\n").encode("utf-8"))

    summary = PreprocessingSummary(
        unicode_normalization_form=config.preprocessing.unicode_normalization_form,
        strip_markup=config.preprocessing.strip_markup,
        deduplicate_exact=config.preprocessing.deduplicate_exact,
        min_characters=config.preprocessing.min_characters,
        min_alpha_ratio=config.preprocessing.min_alpha_ratio,
        statistics=statistics,
        raw_documents=artifact_digest(raw_path),
        cleaned_corpus=artifact_digest(output_path),
    )
    write_json(paths.preprocessed_dir / config.artifacts.preprocessing_summary_filename, summary.model_dump(mode="json"))
    logger.info("Preprocessing retained %s documents", statistics["kept_documents"])
    return summary
