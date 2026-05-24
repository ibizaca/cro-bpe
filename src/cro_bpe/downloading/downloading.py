import json

from datasets import load_dataset, load_dataset_builder

from cro_bpe.config import DISALLOWED_LANGUAGE_MARKERS, RunConfig
from cro_bpe.schemas import SourceManifest
from cro_bpe.utils import ProjectPaths, artifact_digest, get_logger, write_json

logger = get_logger(__name__)


def run(config: RunConfig, paths: ProjectPaths) -> SourceManifest:
    logger.info("Starting controlled download for source '%s'", config.source.source_id)

    builder = load_dataset_builder(config.source.dataset_name, config.source.dataset_config)
    dataset = load_dataset(
        config.source.dataset_name,
        config.source.dataset_config,
        split=config.source.split,
        streaming=config.source.streaming,
        revision=config.source.revision,
    )

    raw_path = paths.raw_dir / config.artifacts.raw_documents_filename
    statistics = {
        "rows_seen": 0,
        "documents_written": 0,
        "empty_rows": 0,
        "rejected_rows": 0,
        "raw_text_bytes": 0,
    }

    with raw_path.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(dataset):
            statistics["rows_seen"] += 1
            raw_text = _read_field(row, config.source.text_field)
            if raw_text is None:
                statistics["rejected_rows"] += 1
                continue

            cleaned_text = raw_text.strip()
            if not cleaned_text:
                statistics["empty_rows"] += 1
                continue

            title = _read_field(row, config.source.title_field) or ""
            document_id = _read_field(row, config.source.id_field) or str(row_index)
            record = {
                "row_index": row_index,
                "document_id": document_id,
                "title": title,
                "text": cleaned_text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            statistics["documents_written"] += 1
            statistics["raw_text_bytes"] += len(cleaned_text.encode("utf-8"))

            if config.source.max_documents and statistics["documents_written"] >= config.source.max_documents:
                break

    manifest = SourceManifest(
        source_id=config.source.source_id,
        corpus_name=config.source.corpus_name,
        language=config.source.language,
        language_decision="approved_croatian_allowlist",
        dataset_name=config.source.dataset_name,
        dataset_config=config.source.dataset_config,
        split=config.source.split,
        revision=config.source.revision,
        streaming=config.source.streaming,
        text_field=config.source.text_field,
        id_field=config.source.id_field,
        title_field=config.source.title_field,
        license=builder.info.license,
        homepage=builder.info.homepage,
        description=builder.info.description,
        rejected_language_markers=[],
        statistics=statistics,
        raw_documents=artifact_digest(raw_path),
    )
    write_json(paths.raw_dir / config.artifacts.source_manifest_filename, manifest.model_dump(mode="json"))
    logger.info("Downloaded %s Croatian documents", statistics["documents_written"])
    return manifest


def _read_field(row: dict[str, object], field_name: str) -> str | None:
    value = row.get(field_name)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
