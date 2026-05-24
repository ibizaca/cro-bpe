# cro-bpe

Croatian-only BPE tokenizer training pipeline built with UV.

## Scope

This project trains a tokenizer exclusively from Croatian data. The default corpus source is the official Croatian Wikipedia dump:

- https://dumps.wikimedia.org/hrwiki/latest/hrwiki-latest-pages-articles-multistream.xml.bz2

No Bosnian, Serbian, or any other non-Croatian corpus is referenced by the code or the default configuration.

## Usage

```bash
uv sync --all-groups
uv run cro-bpe train
```

Useful options:

```bash
uv run cro-bpe train --vocab-size 32000 --min-frequency 2
uv run cro-bpe train --article-limit 50000
uv run cro-bpe prepare-corpus
uv run cro-bpe download
```

Artifacts are written under `data/`:

- `data/raw/` for the downloaded Croatian Wikipedia dump
- `data/preprocessed/` for the normalized Croatian corpus
- `data/models/` for tokenizer artifacts and metadata
