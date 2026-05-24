# cro-bpe

Croatian byte-pair encoding pipeline built around a strict Croatian-only data policy.

## Data policy

This project only permits explicitly approved Croatian datasets.

- Allowed by default: Croatian Wikipedia via `wikimedia/wikipedia` with config `20231101.hr`
- Denied by policy: Bosnian, Serbian, Serbo-Croatian, mixed-language, or any non-Croatian source
- Source selection is deny-by-default and validated before the pipeline runs

## Pipeline structure

The pipeline preserves the existing four stages:

1. `downloading` - controlled dataset ingestion and source manifest generation
2. `preprocessing` - normalization, filtering, and exact deduplication
3. `feature_engineering` - corpus diagnostics and audit metadata
4. `model_training` - BPE tokenizer training and tokenizer artifact generation

## Usage with UV

Install dependencies:

```bash
uv sync --all-groups
```

Run the full pipeline:

```bash
uv run cro-bpe
```

Run on a smaller Croatian Wikipedia sample:

```bash
CRO_BPE_MAX_DOCUMENTS=1000 uv run cro-bpe
```

Run individual stages by toggling stage flags:

```bash
CRO_BPE_RUN_DOWNLOAD=false \
CRO_BPE_RUN_PREPROCESSING=false \
uv run cro-bpe
```

## Artifacts

Artifacts are written into the existing project directories:

- `/tmp/workspace/ibizaca/cro-bpe/data/raw`
  - raw Croatian Wikipedia JSONL export
  - `source_manifest.json`
- `/tmp/workspace/ibizaca/cro-bpe/data/preprocessed`
  - normalized corpus text file
  - `preprocessing_summary.json`
- `/tmp/workspace/ibizaca/cro-bpe/data/feature_engineered`
  - `corpus_diagnostics.json`
- `/tmp/workspace/ibizaca/cro-bpe/data/models`
  - tokenizer JSON
  - BPE vocab and merges files
  - `run_config_snapshot.json`
  - `tokenizer_manifest.json`
- `/tmp/workspace/ibizaca/cro-bpe/mlruns`
  - local MLflow tracking data
- `/tmp/workspace/ibizaca/cro-bpe/logs`
  - run logs

All stage manifests include reproducibility metadata and SHA-256 digests for the generated artifacts.

## Operational notes

- The default tokenizer backend is Hugging Face `tokenizers`
- The default tokenizer configuration trains a byte-level BPE model
- Croatian diacritics are preserved during preprocessing
- No tests are included by design for this task
