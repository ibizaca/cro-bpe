# cro-bpe

Croatian byte pair encoder — trained on a curated monolingual Croatian corpus.

General-purpose tokenizers are built mostly for English and split Croatian
words into many pieces, so Croatian text costs more tokens, more context and
more compute than it should. cro-bpe is a byte-level BPE tokenizer trained only
on Croatian: 12 GB sampled from ten openly licensed corpora (web crawls,
Wikipedia, parliamentary proceedings, EU legislation) after exact and
near-duplicate removal, in vocabularies of 32k, 48k and 64k tokens. Every model is also exported to tiktoken, which encodes
the same ids several times faster.

This repository holds the whole pipeline, from downloading the corpora to the
evaluation, so every model can be rebuilt from scratch.

## Results

On a holdout set of 18,131 Croatian documents (40 MB) from all sources.
"Tokens vs. cro-bpe-32k" is the number of tokens needed for the same text;
lossless means the tokenizer reproduces the text exactly:

| Tokenizer | Vocabulary | bytes/token | tokens/word | tokens vs. cro-bpe-32k | lossless |
|---|---|---|---|---|---|
| **cro-bpe-64k** | 64k | **4.76** | **1.41** | **0.92** | yes |
| **cro-bpe-32k** | 32k | 4.39 | 1.53 | 1.00 | yes |
| BERTić (BCMS encoder) | 32k | 4.36 | 1.54 | 1.01 | no |
| EuroLLM | 128k | 3.35 | 2.01 | 1.31 | yes |
| GPT-4o (`o200k_base`) | 200k | 3.28 | 2.05 | 1.34 | yes |
| Gemma 3 | 262k | 3.21 | 2.10 | 1.37 | yes |
| Llama 3 | 128k | 2.74 | 2.46 | 1.61 | yes |
| Llama 2 | 32k | 2.50 | 2.70 | 1.76 | yes |

cro-bpe-32k needs 24 % fewer tokens than the best current lossless tokenizer
(EuroLLM, 4× its vocabulary) and 43 % fewer than Llama 2 at the same size. Its
tiktoken export encodes 69 MB/s on 16 threads, faster than GPT-4o on tiktoken.
Full results, the other baselines, throughput and how the metrics are measured:
[docs/evaluation.md](docs/evaluation.md).

## Setup

Requirements: Python 3.14 (uv downloads it if it is missing) and a Unix-like OS,
because the run lock uses `fcntl`. The project is developed and run on Linux.

```bash
uv sync
```

`uv sync` installs the project into the environment together with the `dev`
dependency group (pytest, ruff, mypy). The stage packages are top-level imports,
so `PYTHONPATH=src` is not needed.

## Using the tokenizer

The models are built by the pipeline into `data/models/cro-bpe-<vocab>/`; they
are not stored in this repository.

```python
from pathlib import Path

from model_training.export import load_tiktoken
from model_training.tokenizer import load_tokenizer

model_dir = Path("data/models/cro-bpe-32k")
tokenizer = load_tokenizer(model_dir)  # Hugging Face tokenizers; offsets, word ids, padding
tokenizer.encode("Dobar dan, kako ste?").tokens

encoding = load_tiktoken(model_dir)  # tiktoken fast path, same ids; NFC-normalizes like tokenizer.json
encoding.encode_ordinary("Dobar dan, kako ste?")
encoding.encode("<|mask|>", allowed_special="all")  # [2]
```

`tokenizer.json` can also be loaded directly with
`tokenizers.Tokenizer.from_file` or `transformers.PreTrainedTokenizerFast`. For
inputs from the same kind of sources, apply
`preprocessing.cleaning.normalize_text` first to match the training
distribution.

## Running the pipeline

`src/main.py` toggles stages through `run_step`. Downloading is off and
preprocessing, feature engineering, training and evaluation are on:

```bash
uv run python src/main.py
# individual stages:
uv run python -m preprocessing.preprocessing
uv run python -m feature_engineering.feature_engineering
uv run python -m model_training.model_training
uv run python -m evaluation.evaluation
```

The data import is off by default because it downloads over 85 GB and needs
at least 300 GB of disk; see [docs/data.md](docs/data.md) before turning it on.

## Documentation

- [docs/architecture.md](docs/architecture.md): the stages, what each reads and
  writes, reruns, errors and the run lock, cleaning, deduplication and
  sampling, training and the tiktoken export, experiment tracking.
- [docs/data.md](docs/data.md): the source corpora with sizes and licenses,
  excluded sources and why, storage needs, and how the import works.
- [docs/evaluation.md](docs/evaluation.md): metrics, benchmark method and full
  results, including the comparison with current tokenizers.
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, tests and checks, and
  the project's conventions.

## Citation

If you use cro-bpe, please cite it:

```bibtex
@software{bizaca2026crobpe,
  author  = {Bizaca, Ivan},
  title   = {cro-bpe: A Byte-Level {BPE} Tokenizer for {Croatian}},
  year    = {2026},
  url     = {https://github.com/ibizaca/cro-bpe},
  license = {Apache-2.0}
}
```

GitHub also offers this citation (BibTeX and APA) under "Cite this repository",
from [CITATION.cff](CITATION.cff).

## License

The code is licensed under the [Apache License 2.0](LICENSE). The training
corpora keep their own licenses (CC0, CC-BY, CC-BY-SA, ODC-By and public
domain), listed per source in [docs/data.md](docs/data.md#source-inventory).
