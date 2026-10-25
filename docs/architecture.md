# Architecture

How the pipeline is built: the stages, what they read and write, how reruns,
errors and the run lock work, and what each stage does to the data. The sources
themselves are described in [data.md](data.md), and the metrics and results in
[evaluation.md](evaluation.md).

## Pipeline overview

The pipeline has five stages, each a package under `src/`:

| Stage | Package | Reads | Writes |
|---|---|---|---|
| Data import | `data_import` | source archives and Hugging Face datasets | `data/raw/` |
| Preprocessing | `preprocessing` | `data/raw/` | `data/preprocessed/` |
| Feature engineering | `feature_engineering` | `data/preprocessed/` | `data/feature_engineered/` |
| Model training | `model_training` | `data/feature_engineered/train/` | `data/models/cro-bpe-<vocab>/` |
| Evaluation | `evaluation` | models and `data/feature_engineered/holdout/` | `evaluation.json`, `comparison.json` |

`src/utils/` holds what the stages share: data paths, the run lock, atomic file
writes, the Parquet schemas, fingerprints, typed JSON artifacts, experiment
tracking and logging.

## Running and rerunning

`src/main.py` runs the stages in order, and each stage's `run_step` turns it on
or off (commands in the [README](../README.md#running-the-pipeline)).

Each stage's settings object is built in `main.py` and passed to the stage, so
settings (including `data_root`) can be changed there. The `python -m` entry
points use the stage's default settings with `data_root` set to `./data`.
Stages that are already up to date are skipped (see each stage's rerun
behaviour below, and [data.md](data.md#rerun-behaviour) for data import), so
rerunning `main.py` only redoes what changed. Training and evaluation always
run again.

`main.py` writes a new timestamped log to `logs/` for every run and keeps the
50 most recent (`setup_logging(keep_logs=...)`). The `python -m` entry points
don't configure logging, so they write no log file and show only warnings and
errors.

Data problems end the run with a one-line error and exit code 1 instead of a
traceback: a source that is neither complete nor blocked (`MissingSourcesError`),
a missing output of an earlier stage (`MissingInputError`), a tiktoken export
that doesn't match `tokenizer.json` (`ExportMismatchError`), or another run
already using the same `data/` directory (`PipelineBusyError`). See
`src/utils/errors.py`. This applies to `main.py`; the `python -m` entry points
raise these errors with a traceback.

Only one run at a time can use a `data/` directory. `main.py` and each stage's
`python -m` entry point hold an exclusive lock on `data/.cro-bpe.lock`, so a
second run fails at once and names the PID of the first. The OS releases the
lock when the process exits, so a crash never leaves a stale lock.

## Data import

`src/data_import/` downloads every source in the catalog and converts it to
Parquet shards in `data/raw/<source-id>/`, with a `manifest.json` per source.
Sources, output layout, storage needs and rerun behaviour are described in
[data.md](data.md).

## Preprocessing

`src/preprocessing/` cleans every raw shard of every completely imported source
in parallel (one worker per shard, CPU count − 2 by default) into
`data/preprocessed/<source-id>/part-*.parquet` with columns `text`, `source`,
`id`, `hash` (xxh3-64 of the cleaned text) and `n_bytes`. Shards are written to a
`.tmp` file and renamed when finished.

Cleaning (`cleaning.py`):

- Decoding of complete HTML entities (`&amp;`, `&#269;`; `&param` in a URL is
  left alone), Unicode NFC, and decomposition of the digraph letters
  `ǅ ǈ ǋ` (and `ﬁ`-style ligatures) into two code points. Full NFKC is
  deliberately avoided because it rewrites superscripts and fractions.
- Removal of control, zero-width, bidi and soft-hyphen characters; tabs and
  other space characters become spaces; unified line breaks; collapsed spaces
  and blank lines.
- Removal of text that spells a special token (`<|endoftext|>`,
  `<|reserved_3|>`), which `tokenizer.json` would encode as that token.
- `hr-wikipedia` only: removal of residual wiki markup (templates, tables,
  template parameters such as `url=…` or `align=…`, headings, bold and italic
  quotes, list markers, external-link syntax, and lines that are file or
  category links) and repair of the `target|label` fragments left by link
  stripping.

Quality filters (`filters.py`, thresholds in `QualityThresholds`) reject
documents that are too short (< 20 characters or < 3 words), have an implausible
mean word length (outside 2–14 characters), are mostly non-letters (< 50 %),
have > 15 % non-Latin-script letters, or have > 30 % of characters in repeated
lines. No language identification is performed: the corpus is Croatian-only by
construction.

Each shard writes a `.stats.json` sidecar with rejection counts and a
fingerprint of the quality thresholds, the `VERSION` constants of
`preprocessing.py`, `cleaning.py`, `filters.py` and `charclasses.py`, and the
input shard's name and size. On rerun a shard is skipped only if its
fingerprint matches. Changing a threshold rebuilds automatically. After a code
change that alters the output, bump that module's `VERSION`. A change that
doesn't (a comment, a reformat) leaves the 217 GB corpus alone.
`tests/test_versions.py` pins each module's source hash next to its version, so
any code change fails the tests until `tests/versions.json` is updated. That's
the reminder to decide whether a bump is needed. Outputs whose input shard is
gone are deleted. Per-source totals and a source fingerprint go to
`data/preprocessed/<source-id>/stats.json`.

## Feature engineering (training corpus)

`src/feature_engineering/` turns the cleaned documents into the tokenizer corpus:

1. **Global exact deduplication** on `hash`, keeping the first copy in
   `SOURCE_PRIORITY` order (curated sources before web crawls).
2. **Global near-duplicate removal** (`near_dedup.py`, on by default with
   `near_dedup=True`). Five sources are Common Crawl derivatives, so the same
   page often appears again with small differences. Each document gets MinHash
   values over its word 5-grams (lowercased runs of letters and digits), in 14
   bands of 8 values (FineWeb's parameters). A document that shares a band
   hash with an earlier kept document, in `SOURCE_PRIORITY` order, is removed.
   Two documents with a Jaccard similarity of 0.8 share a band with probability
   0.92, at 0.75 with 0.77 and at 0.5 with 0.05. Documents with fewer than 5
   words have no shingles and are left to exact dedup.
3. **Holdout split**: up to 2,000 documents per source (≤ 1 % of the source's
   deduplicated documents), selected by the high 32 bits of the hash and never
   used for training. It is taken after both dedup steps, so no training
   document is a near-copy of a holdout document.
4. **Weighted sampling** to `target_train_bytes` (default 12 GB) using
   `MIXTURE_WEIGHTS`. Web crawls are downweighted relative to their raw size,
   and budget a small source cannot fill is redistributed to the others. Sampling
   is deterministic (on the low 32 bits of the hash), so reruns select the same
   documents.

Both lists are in `mixture.py`. A preprocessed source missing from
`SOURCE_PRIORITY` is ignored with a warning.

Output: `data/feature_engineered/{train,holdout}/part-*.parquet` (`text`,
`source`; shards of about 256 MiB of text) and `mixture.json` with per-source
statistics, including exact and near-duplicates removed. `mixture.json` holds a
fingerprint of the settings, `SOURCE_PRIORITY`, `MIXTURE_WEIGHTS`, the
`VERSION` of `feature_engineering.py`, `mixture.py` and `near_dedup.py`, and
every preprocessed source; the stage is skipped only when it matches, and is
rebuilt otherwise. It is written last, so an interrupted build is never reused.
It also records provenance (git commit and whether the tree was dirty, Python
and library versions).

The band hashes are the expensive part (every preprocessed document is read
once; `workers` processes in parallel). They are cached per preprocessed shard
in `data/feature_engineered/minhash/<source>/`, 112 bytes per document. The
file name holds a fingerprint of the shard (name and size), the MinHash
settings (`minhash_ngram`, `minhash_bands`, `minhash_rows`) and
`near_dedup.VERSION`, so a rebuild for other reasons (weights, sample size)
reuses them, and outdated files are deleted.

## Model training

`src/model_training/` trains a byte-level BPE with Hugging Face `tokenizers` and
exports every model to tiktoken (see [tiktoken export](#tiktoken-export)):

- NFC normalizer; a GPT-4-style pre-tokenizer regex without the English
  contraction rules (letter runs with an optional leading space, digit groups of
  ≤ 3 with an optional leading space, punctuation runs, whitespace) followed by
  byte-level mapping, so any input round-trips losslessly.
- One model per vocabulary size in `TrainingSettings.vocab_sizes` (default
  32,000 / 48,000 / 64,000, including the special tokens; sizes must be unique
  multiples of 1,000), minimum merge frequency 2, maximum token length 24
  characters.
- The largest size is trained once and the smaller ones are truncated from it
  (`train_once=True`). BPE merges are greedy and do not depend on the target
  size, so this gives the same models as separate runs (`tests/test_training.py`
  checks it) for about a third of the cost.
- 69 special tokens with the same low ids in every size, so models built on
  different sizes (ModernBERT, ELECTRA, small language models) share them:

  | id | token | use |
  |---|---|---|
  | 0 | `<\|endoftext\|>` | document separator / end of sequence |
  | 1 | `<\|pad\|>` | padding |
  | 2 | `<\|mask\|>` | masked-language-model target |
  | 3 | `<\|cls\|>` | sequence start for encoders |
  | 4 | `<\|sep\|>` | segment separator for encoders |
  | 5–68 | `<\|reserved_0\|>` … `<\|reserved_63\|>` | unused; for later chat, tool or fill-in-the-middle markers |

  `tokenizer.json` adds no template. Encoder model code wraps inputs as
  `<|cls|> … <|sep|>` itself, so the same file serves decoder models.

Output per size in `data/models/cro-bpe-<vocab>/`: `tokenizer.json`,
`vocab.json`, `merges.txt`, a tiktoken export (`cro-bpe-<vocab>.tiktoken`,
`tiktoken.json` and its copy `config.json`) and `training_config.json` (settings, corpus summary,
tracking run id, provenance). Training fails if the tiktoken export does not produce the same ids as
`tokenizer.json` on about 20,000 training documents spread over every training
shard (so every source is covered, not only the first) plus the whole holdout
set. Each model is built in `cro-bpe-<vocab>.tmp/` and renamed into place only
after that check and `training_config.json`, so a failed run never leaves a
model that looks complete; evaluation only uses models with
`training_config.json`.

### tiktoken export

tiktoken cannot train, so every trained model is also exported to tiktoken's
format (`model_training/export.py`). That gives the same ids with faster
encoding (see the [throughput results](evaluation.md#throughput)). The export is written next to
`tokenizer.json`:

- `cro-bpe-<vocab>.tiktoken`: one line per token, the base64-encoded token bytes
  and its rank. Ranks equal the `tokenizer.json` ids. The trainer assigns ids in
  merge order after the special tokens and the 256-byte alphabet, so id order is
  merge priority.
- `tiktoken.json`: encoding name, rank file name, the pre-tokenizer regex taken
  from the saved `tokenizer.json` (not the current constant in `tokenizer.py`),
  the special tokens with their ids, and `"normalization": "NFC"`.
- `config.json`: an exact copy of `tiktoken.json`. The Hub counts downloads by
  requests to `config.json` (the default for repositories without a registered
  library), and `transformers` requests it first, so the copy makes those loads
  count. `tiktoken.json` stays for code that fetches it by name.

Special tokens go only in `special_tokens`, not among the mergeable ranks.
tiktoken has no normalizer, so `load_tiktoken` returns an `NFCEncoding`, which
applies NFC before `encode`, `encode_ordinary` and their batch variants, as
`tokenizer.json` does. The equality gate (`verify_tiktoken`) compares
`encode_batch(..., allowed_special="all")` with `tokenizer.json`'s
`encode_batch_fast` ids and raises `ExportMismatchError` on any difference.
Training runs it before a model is moved into place, and evaluation runs it on
every holdout document.

### Experiment tracking

Each size gets its own MLflow run with the settings, corpus size, pre-tokenizer
pattern, training time (`training_minutes`) and the model directory as an
artifact. MLflow is used only through `utils/tracking.py` (`MlflowTracker`), with
a SQLite store in `data/models/mlflow.db` and artifacts in
`data/models/mlartifacts/`. Stages accept any `Tracker`, and the tests pass
`NullTracker` and block `import mlflow` entirely:

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/models/mlflow.db
```

## Evaluation

`src/evaluation/` measures every trained model on the holdout set against the
tokenizers of current models, multilingual and Croatian/BCMS ones included, and
against Llama-2 and GPT-2. It is described, with the results, in
[evaluation.md](evaluation.md).
