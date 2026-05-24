from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from cro_bpe.config import RunConfig
from cro_bpe.schemas import TokenizerArtifactManifest
from cro_bpe.utils import ProjectPaths, artifact_digest, get_logger, write_json

logger = get_logger(__name__)


def run(config: RunConfig, paths: ProjectPaths) -> TokenizerArtifactManifest:
    logger.info("Starting BPE tokenizer training")

    corpus_path = paths.preprocessed_dir / config.artifacts.preprocessed_corpus_filename
    if not corpus_path.exists():
        raise FileNotFoundError(f"Preprocessed corpus not found at {corpus_path}")

    config_snapshot_path = paths.models_dir / config.artifacts.run_config_snapshot_filename
    write_json(config_snapshot_path, config.model_dump(mode="json"))

    tokenizer = _build_tokenizer(config)
    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer.vocab_size,
        min_frequency=config.tokenizer.min_frequency,
        show_progress=True,
        special_tokens=config.tokenizer.special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([str(corpus_path)], trainer)

    tokenizer_json_path = paths.models_dir / config.artifacts.tokenizer_filename
    tokenizer.save(str(tokenizer_json_path))

    vocab_path, merges_path = _save_model_files(tokenizer=tokenizer, output_dir=paths.models_dir, prefix=config.artifacts.tokenizer_prefix)

    sample_encoding = tokenizer.encode(config.tokenizer.smoke_test_text)
    decoded_text = tokenizer.decode(sample_encoding.ids)

    manifest = TokenizerArtifactManifest(
        model_type=config.tokenizer.model_type,
        normalization=config.tokenizer.normalization,
        pre_tokenizer=config.tokenizer.pre_tokenizer,
        vocab_size=tokenizer.get_vocab_size(),
        min_frequency=config.tokenizer.min_frequency,
        special_tokens=config.tokenizer.special_tokens,
        tokenizer_json=artifact_digest(tokenizer_json_path),
        vocab_file=artifact_digest(vocab_path) if vocab_path else None,
        merges_file=artifact_digest(merges_path) if merges_path else None,
        config_snapshot=artifact_digest(config_snapshot_path),
        sample_text=config.tokenizer.smoke_test_text,
        sample_tokens=sample_encoding.tokens,
        sample_ids=sample_encoding.ids,
        decoded_text=decoded_text,
    )
    write_json(paths.models_dir / config.artifacts.tokenizer_manifest_filename, manifest.model_dump(mode="json"))
    logger.info("Finished tokenizer training with vocabulary size %s", manifest.vocab_size)
    return manifest


def _build_tokenizer(config: RunConfig) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=config.tokenizer.unknown_token))
    tokenizer.normalizer = _build_normalizer(config.tokenizer.normalization)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def _build_normalizer(normalization_name: str):
    normalized = normalization_name.strip().upper()
    if normalized == "NFC":
        return normalizers.NFC()
    if normalized == "NFKC":
        return normalizers.NFKC()
    raise ValueError(f"Unsupported normalization '{normalization_name}'")


def _save_model_files(tokenizer: Tokenizer, output_dir: Path, prefix: str) -> tuple[Path | None, Path | None]:
    saved_files = tokenizer.model.save(str(output_dir), prefix)
    vocab_path: Path | None = None
    merges_path: Path | None = None
    for file_path in saved_files:
        path = Path(file_path)
        if path.suffix == ".json":
            vocab_path = path
        elif path.suffix == ".txt":
            merges_path = path
    return vocab_path, merges_path
