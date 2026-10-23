# Data sources

The corpora cro-bpe is trained on, the ones left out and why, and how the
data import stage (`src/data_import/`) downloads them into `data/raw/`. For
what happens to the data afterwards, see [architecture.md](architecture.md).

## Source inventory

Sizes are the downloaded files and documents are the imported counts, both
from `manifest.json` of the current import:

| ID | Corpus | Download | Documents | License | Auth |
|---|---|---|---|---|---|
| `hr-wikipedia` | Croatian Wikipedia (latest dump) | 0.35 GB | 221,811 | CC-BY-SA 4.0 | None |
| `hplt-3.0-hrv-latn` | HPLT 3.0 hrv_Latn (quality bins 5–10, one shard each) | 50.8 GB | 31,156,692 | CC0 1.0 | None |
| `fineweb-2-hrv-latn` | FineWeb-2 hrv_Latn | Hub Parquet download | 6,195,824 | ODC-By 1.0 | None |
| `classla-web-hr-2` | CLASSLA-web.hr 2.0 | 8.4 GB | 5,922,531 | CC0 1.0 | None |
| `cc-100-hr` | CC-100 Croatian | 6.0 GB | 16,540,397 | Public domain | None |
| `macocu-hr-2` | MaCoCu-hr 2.0 | 7.6 GB | 8,106,239 | CC0 1.0 | None |
| `hrwac-2.1` | hrWaC 2.1 (14 parts) | 9.2 GB | 28,771,178 | CC-BY-SA 4.0 | None |
| `parlamint-hr-5` | ParlaMint-HR 5.0 | 0.42 GB | 504,334 | CC-BY 4.0 | None |
| `multi-eurlex-hr` | MultiEURLEX hr | 2.8 GB multilingual archive | 15,444 | CC-BY 4.0 | None |
| `parlaspeech-hr` | ParlaSpeech-HR 2.0 (text only) | streamed from the Hub | 867,581 | CC-BY 4.0 | None |
| `culturax-hr` | CulturaX hr | streamed from the Hub | — | ODC-By / OSCAR | **HF_TOKEN** |

The catalog is `src/data_import/catalog.py`. Every source is declared there, and
the rest of the stage derives its behaviour from those declarations.

## Excluded sources

| ID | Corpus | Reason for exclusion |
|---|---|---|
| `oscar-21-23` | OSCAR 21.09 / 22.01 / 23.01 | Upstream access is suspended; not imported until it is restored. |
| `mc4` | mC4 | Has no Croatian partition. |
| `hplt-1.2-hbs` | HPLT 1.2 hbs, SH Wikipedia, any `hbs`/`sh`/`bs`/`sr` partition | Not Croatian-only: mixes Serbian, Bosnian, Montenegrin and Croatian under the BCMS macro-language code. |
| `superseded` | MaCoCu-hr 1.0, CLASSLA-web.hr 1.0, HPLT 2.0, filtered copies | Superseded by the later versions above, or filtered copies of sources already imported. |
| `cnk-riznica` | Croatian National Corpus / Riznica | No working bulk-download endpoint. |
| `opus-parallel` | OPUS parallel corpora, benchmark and annotation datasets | Not substantial monolingual Croatian text; would duplicate or distort the training distribution. |

The list is `EXCLUDED` in `src/data_import/catalog.py`. Each entry is recorded
in `data/raw/<id>/manifest.json` with status `excluded` and the reason.

## Storage requirements

Allow at least **300 GB**. Source archives are deleted once their Parquet output
is verified, but during the import an archive and its output both exist on disk
(the HPLT shards alone are 51 GB). After the current run, `data/` holds 82 GB of
raw Parquet, 81 GB of preprocessed Parquet, 10 GB of near-dedup band hashes,
4.5 GB of training and holdout corpus and 163 MB of models, baseline tokenizers
and MLflow runs. Everything stays under `data/`; nothing is left in
`~/.cache/huggingface`. Exact usage changes as rolling web corpora are updated.

## Hugging Face token (CulturaX)

CulturaX requires accepting the [access agreement](https://huggingface.co/datasets/uonlp/CulturaX)
and exporting a token (`HUGGING_FACE_HUB_TOKEN` also works):

```bash
export HF_TOKEN=hf_...
```

Without the token, or when the access request has not been accepted, CulturaX is
recorded as `blocked` (`auth_required`) and all other sources continue normally.

## Output layout

```
data/raw/
  <source-id>/
    source/           ← downloaded files, deleted after conversion
    parquet/
      part-00000.parquet
      part-00001.parquet
    manifest.json     ← status, source/output checksums, row counts, skipped files
```

Excluded and blocked sources only get `manifest.json`.

Parquet columns are `text`, `source`, `id`, `url`, and `language`. Shards are
limited to an estimated **128 MiB of uncompressed UTF-8 field data or 100,000
rows**, whichever comes first. Each file uses Zstandard compression (level 3)
and 16,384-row groups. Expected writer peak memory is below 500 MiB per shard,
keeping individual preprocessing tasks comfortably below the available 64 GB
of RAM while supporting parallel scans and column projection.

## Source-specific notes

- `hr-wikipedia`: wikitext is parsed with `mwparserfromhell`. Link labels are
  kept; templates, tables, references, `<math>`/`<gallery>`-style tags, bold and
  italic quotes, and file, category and interlanguage links are removed.
  Namespace-0 articles only; redirects are skipped.
- `hrwac-2.1`: the release has no document level. Paragraphs sit directly under
  the corpus root with their own URL and are shuffled across pages, so each
  paragraph is one document. Text is rebuilt from the word column, and `<g/>`
  glues a token to the previous one.
- `cc-100-hr`: documents are separated by blank lines.
- `parlamint-hr-5`: each speech (`<u>` utterance) is one document.
- `multi-eurlex-hr`: the Croatian text of the train, dev and test splits.
- `fineweb-2-hrv-latn`: its Parquet files are downloaded into
  `data/raw/<id>/source/` (not the global Hugging Face cache), so an interrupted
  import resumes, the files get sizes and SHA-256 in `manifest.json`, and they
  are deleted after conversion like HTTP sources. `parlaspeech-hr` and
  `culturax-hr` are read remotely through `huggingface_hub.HfFileSystem`:
  Parquet column projection fetches only the text column (about 6 % of a
  ParlaSpeech file, whose audio column is skipped), and nothing is written to
  disk. ParlaSpeech uses `text_normalised`.
- Archive members that fail to parse (MaCoCu, ParlaMint XML) are skipped with a
  warning and listed under `skipped_files` in `manifest.json`.

## Rerun behaviour

Settings are in `DataImportSettings` (`src/data_import/data_import.py`):

- A source whose `manifest.json` records `"status": "complete"` and whose shards
  exist with the recorded sizes is skipped. Set `verify_existing=True` to also
  recompute every shard's SHA-256 (this reads the whole output).
- Interrupted HTTP downloads resume from their `.part` file, with up to
  `download_retries=8` attempts and exponential back-off capped at 60 s. Only
  408, 425, 429, 5xx, network and size errors are retried; other HTTP errors
  (404, 403, …) fail at once. A `.part` file the server reports as already
  complete (416 with a matching `Content-Range`) is kept and promoted.
- A finished download that is missing from `manifest.json` (a crash between the
  rename and the manifest save) is adopted, not downloaded again.
- Partial Parquet output is written under `parquet.tmp/` and re-created from
  scratch for that source after interruption.
- Original source archives are deleted after the source's Parquet output is
  written and verified, unless `keep_source_files=True`. Keeping them lets you
  re-run a fixed reader without downloading again. Their URLs, original sizes,
  and checksums remain in `manifest.json`. Leftover source files of a complete
  source are deleted on the next run.
- No document-level language filtering is performed during downloading.
- The stage fails unless every catalog source ends up `complete` or `blocked`,
  so a failed source cannot silently drop out of the mixture. Source ids in
  `allow_missing` only log a warning. When the download step is off, `main.py`
  runs the same check on the manifests on disk before preprocessing.
  `culturax-hr` is in `allow_missing` in `main.py`, because its access request
  is pending and its import failed.

## Running the import

The download step is off in `main.py` (`download_config.run_step=False`). Turn it
on there and run the pipeline, or run the stage alone:

```bash
uv run python src/main.py
# or just the download stage:
uv run python -m data_import.data_import
```

The standalone entry point uses the default `DataImportSettings`, which has an
empty `allow_missing`, so it ends with `MissingSourcesError` while CulturaX is
not imported. The other sources are imported anyway.
