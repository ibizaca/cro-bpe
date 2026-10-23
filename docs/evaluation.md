# Evaluation

How cro-bpe is measured on the holdout set, and the results. How the holdout
set is selected is described in
[architecture.md](architecture.md#feature-engineering-training-corpus).

## How evaluation works

`src/evaluation/` evaluates every trained size (`EvaluationSettings.vocab_sizes`)
on the holdout set, both through `tokenizer.json` and through its tiktoken
export, next to baseline tokenizers. A model without a tiktoken export is
exported first. The tiktoken export must give the same ids as `tokenizer.json`
on every holdout document, otherwise evaluation fails; it therefore shares the
model's quality metrics.

### Baselines

Llama-2 and GPT-2 are old and English-centric, so beating them says little.
The baselines (`EvaluationSettings.baselines`, name → Hugging Face Hub
repository) are therefore the tokenizers of current models, multilingual and
Croatian/BCMS ones included. Gated models are loaded from ungated mirrors, so no
token is needed:

| Name | Repository | Kind |
|---|---|---|
| `llama-3-128k` | `unsloth/Llama-3.2-1B` | byte-level BPE |
| `gemma-3-262k` | `unsloth/gemma-3-1b-it` | SentencePiece BPE, byte fallback |
| `qwen-3-151k` | `Qwen/Qwen3-0.6B` | byte-level BPE |
| `gpt-4o-200k` | `Xenova/gpt-4o` | byte-level BPE (`o200k_base`) |
| `mistral-nemo-131k` | `unsloth/Mistral-Nemo-Base-2407` | byte-level BPE (Tekken) |
| `eurollm-128k` | `utter-project/EuroLLM-1.7B` | SentencePiece BPE |
| `xlm-r-250k` | `FacebookAI/xlm-roberta-base` | SentencePiece Unigram |
| `bertic-32k` | `classla/bcms-bertic` | WordPiece (BCMS ELECTRA encoder) |
| `llama-2-32k` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | SentencePiece BPE, byte fallback |
| `gpt-2-50k` | `openai-community/gpt2` | byte-level BPE |

Files are cached in `data/models/baselines/`, and `comparison.json` records the
commit each one came from. A baseline that cannot be loaded is skipped with a
warning. BERTić ships only `vocab.txt`, so a `tokenizer.json` is built from it
(cased, as its `tokenizer_config.json` says) and saved under
`data/models/baselines/wordpiece/`.

`EvaluationSettings.tiktoken_builtin` (`"gpt-4o-200k (tiktoken)": "o200k_base"`)
adds GPT-4o on tiktoken, the engine OpenAI uses. Its rank file is cached in
`data/models/baselines/tiktoken/`. It is checked against the `Xenova/gpt-4o`
mirror on the holdout set; a difference is logged and recorded in
`tiktoken_mismatches` instead of failing the stage.

### Metrics

Per tokenizer:

- **Throughput** (measured before the quality metrics), in tokens/s and MB/s,
  for every model through both engines, and for the baselines in
  `EvaluationSettings.throughput_candidates` (default: Llama-3 and GPT-4o, both
  engines), because each tokenizer adds several minutes; about 60 min.
  - batched encoding of the whole holdout set at 1, 4 and all threads
    (`encode_batch_fast` on the Rust thread pool, `encode_ordinary_batch` for
    tiktoken);
  - sequential one-document-per-call encoding, including Python call overhead,
    on an evenly strided holdout sample of up to 16 MB, so every source is
    represented (12.9 MB in the current run);
  - batched decoding.

  Each measurement runs in a fresh subprocess (`evaluation/throughput_worker.py`)
  with `RAYON_NUM_THREADS` set (the Rust thread pool size is fixed once
  created), and candidates are interleaved round-robin over 5 rounds so their
  medians are comparable. A measurement repeats the workload for at least 10 s
  after a first warm-up run, which is reported separately as "burst". Sequential
  encoding and decoding are measured in the all-threads run. Median, min, max
  and spread ((max − min) / median) over rounds are reported.
- **Quality** (per holdout source; overall as `all`, where each source counts
  with its holdout size; and as `mixture`, where each source counts with its
  share of the training corpus, which reflects text like the training mix):
  bytes/token and characters/token (compression, higher is better),
  tokens/word over whitespace words (fertility, lower is better), split-word
  rate (share of pre-tokenized words needing more than one token; n/a without a
  pre-tokenizer) and round-trip failures (`decode(encode(text)) != text`).
- **Vocabulary**: size, share of the vocabulary used on the holdout set, mean
  token length, and Rényi efficiency (α = 2.5) of the token distribution, which
  penalizes vocabulary that is rarely used.
- **Rare-word fertility**: tokens/word by the word's frequency rank in a 200 MB
  training sample (top 1k / 1k–10k / 10k–100k / rest).

### Output

Results go to `data/models/cro-bpe-<vocab>/evaluation.json` (that model, its
tiktoken export and the baselines) and `data/models/comparison.json` (all
tokenizers, a summary table, the baselines' commits and the tiktoken mirror
checks). Both record the settings, the holdout set and the environment:
processor, CPU count, mean CPU clock before and after the benchmark, Python and
library versions. The CPU package temperature is recorded too, but only read
from Intel's `x86_pkg_temp` sensor, so it is empty on AMD. The metrics are
added to each model's MLflow training run under `eval.*`, `eval.tiktoken.*` and
`baseline.<name>.*`, with both JSON files as artifacts. The log ends with each
model's tokenization of four example sentences.

### How to read the numbers

- bytes/token and tokens/word are the fair measures across vocabulary sizes.
  Rényi efficiency is divided by log(vocabulary size) and vocabulary use is a
  share of the vocabulary, so neither can be compared between a 32k Croatian
  vocabulary and a 262k multilingual one.
- XLM-R and BERTić are encoder tokenizers and are lossy: XLM-R normalizes text
  and collapses whitespace, and WordPiece drops whitespace entirely and joins
  tokens with spaces when decoding. Their round-trip failures are expected, and
  they get whitespace for free, which flatters their compression a little.
  Unknown tokens are negligible (BERTić 0.015 % of tokens).

## Results

Models trained and evaluated on 2026-10-23 (`data/models/comparison.json`).

Corpus: 95.4 M documents after preprocessing (216.6 GB); 3.4 M exact duplicates
and 11.8 M near-duplicates (45 GB) removed; 12.0 GB / 8.5 M documents sampled
for training. On an AMD Ryzen 7 7800X3D (8 cores, 16 threads, 64 GB RAM),
preprocessing takes about 12 min with 14 workers. The corpus build takes about
7 min, plus about 19 min the first time for the near-dedup band hashes. BPE
training of the 64k model takes 4.5 min, and 48k and 32k are truncated from it
(about 5 min for all three, including the tiktoken export checks). Evaluation
takes about 64 min, almost all of it the throughput benchmark.

Before near-dedup, 772 of the then 18,154 holdout documents (4.3 %) had a
near-copy in the training corpus; now none can. The results barely moved
(cro-bpe-32k: 4.40 → 4.39 bytes/token), so the contamination had not inflated
the earlier numbers noticeably.

### Per source

Quality on the holdout set (18,131 documents, 40 MB). All documents round-trip
losslessly with every size, and the tiktoken export produces identical ids:

| Holdout source | Documents | 32k bytes/token | 48k | 64k | 32k tokens/word | 48k | 64k |
|---|---|---|---|---|---|---|---|
| all | 18,131 | 4.39 | 4.62 | 4.76 | 1.53 | 1.46 | 1.41 |
| mixture | — | 4.38 | 4.60 | 4.75 | 1.53 | 1.45 | 1.41 |
| hr-wikipedia | 2,000 | 3.92 | 4.13 | 4.28 | 1.77 | 1.68 | 1.62 |
| multi-eurlex-hr | 131 | 4.91 | 5.12 | 5.27 | 1.43 | 1.37 | 1.33 |
| parlamint-hr-5 | 2,000 | 5.01 | 5.19 | 5.29 | 1.27 | 1.22 | 1.20 |
| parlaspeech-hr | 2,000 | 4.97 | 5.14 | 5.25 | 1.26 | 1.22 | 1.19 |
| classla-web-hr-2 | 2,000 | 4.38 | 4.60 | 4.74 | 1.52 | 1.45 | 1.40 |
| macocu-hr-2 | 2,000 | 4.39 | 4.61 | 4.76 | 1.54 | 1.47 | 1.42 |
| hrwac-2.1 | 2,000 | 4.36 | 4.57 | 4.71 | 1.45 | 1.38 | 1.34 |
| fineweb-2-hrv-latn | 2,000 | 4.45 | 4.68 | 4.83 | 1.53 | 1.45 | 1.41 |
| hplt-3.0-hrv-latn | 2,000 | 4.30 | 4.53 | 4.68 | 1.57 | 1.49 | 1.44 |
| cc-100-hr | 2,000 | 4.34 | 4.57 | 4.71 | 1.53 | 1.46 | 1.41 |

### Compared with the baselines

"Tokens vs. 32k" is the number of tokens needed for the holdout set relative to
cro-bpe-32k; the fertility columns are tokens/word by the word's frequency rank
in the training corpus:

| Tokenizer | Vocab | bytes/token | tokens/word | tokens vs. 32k | top 1k | 1k–10k | 10k–100k | rarer | round-trip failures |
|---|---|---|---|---|---|---|---|---|---|
| cro-bpe-32k | 32,000 | 4.39 | 1.53 | 1.00 | 1.09 | 1.23 | 2.06 | 3.38 | 0 |
| cro-bpe-48k | 48,000 | 4.62 | 1.46 | 0.95 | 1.09 | 1.22 | 1.78 | 3.20 | 0 |
| cro-bpe-64k | 64,000 | **4.76** | **1.41** | **0.92** | 1.09 | 1.22 | **1.62** | **3.08** | 0 |
| bertic-32k | 32,000 | 4.36 | 1.54 | 1.01 | 1.09 | 1.24 | 2.10 | 3.38 | 13,755 |
| xlm-r-250k | 250,002 | 3.86 | 1.74 | 1.14 | **1.08** | 1.86 | 2.47 | 3.26 | 11,450 |
| eurollm-128k | 128,000 | 3.35 | 2.01 | 1.31 | 1.22 | 2.27 | 2.76 | 3.71 | 0 |
| gpt-4o-200k | 200,000 | 3.28 | 2.05 | 1.34 | 1.26 | 2.41 | 2.84 | 3.53 | 0 |
| gemma-3-262k | 262,145 | 3.21 | 2.10 | 1.37 | 1.28 | 2.45 | 2.87 | 3.67 | 0 |
| mistral-nemo-131k | 131,072 | 3.08 | 2.19 | 1.43 | 1.36 | 2.56 | 2.98 | 3.79 | 0 |
| llama-3-128k | 128,256 | 2.74 | 2.46 | 1.61 | 1.60 | 2.93 | 3.30 | 3.90 | 0 |
| qwen-3-151k | 151,669 | 2.60 | 2.59 | 1.69 | 1.66 | 3.12 | 3.47 | 4.14 | 0 |
| llama-2-32k | 32,000 | 2.50 | 2.70 | 1.76 | 1.74 | 3.19 | 3.59 | 4.41 | 0 |
| gpt-2-50k | 50,257 | 2.28 | 2.96 | 1.93 | 1.95 | 3.58 | 3.94 | 4.48 | 0 |

cro-bpe-32k needs 43 % fewer tokens than Llama-2 at the same vocabulary size.
Among lossless tokenizers, the best current one, EuroLLM's 128k vocabulary,
needs 31 % more tokens than cro-bpe-32k (cro-bpe-32k needs 24 % fewer), and
GPT-4o, Gemma 3 and Mistral Nemo need 34–43 % more, all with vocabularies 4–8×
larger. Llama-3 and Qwen3 are only slightly better than Llama-2 on Croatian.
The gap is in words outside the 1,000 most frequent: in the 1k–10k band the
current LLM tokenizers need 2.3–3.1 tokens per word, cro-bpe 1.2. The only
tokenizer level with cro-bpe-32k is BERTić, another BCMS-specific 32k
vocabulary; it uses 1 % more tokens even though it drops all whitespace (0.76 %
of cro-bpe-32k's tokens are whitespace-only), and it cannot reproduce the text.
XLM-R's huge vocabulary makes frequent words single tokens and splits the
rarest words slightly less than cro-bpe-32k (3.26 vs. 3.38 tokens/word), but
words ranked 1k–100k cost it 1.86 and 2.47 tokens, against 1.23 and 2.06.

Between the cro-bpe sizes, 48k and 64k compress 5.1 % and 8.4 % better than 32k.
Almost all of the gain is in words ranked 10k–100k (2.06 → 1.62 tokens/word);
frequent words are already single tokens at 32k. Rényi efficiency falls with
size (0.476, 0.451 and 0.435), because the extra tokens are rarer. Choosing
between compression and embedding size depends on the model; cro-bpe-32k stays
the default here. Vocabulary use is 99.5 % and not 100 %, because the 69
special tokens never occur in text.

`o200k_base` from tiktoken and the `Xenova/gpt-4o` mirror give identical ids on
all 18,131 documents.

### Throughput

Median of 5 interleaved rounds of sustained (≥ 10 s) runs on the same AMD Ryzen
7 7800X3D (8 cores, 16 threads). The runs are stable: the spread over rounds is
below 9 % for any measurement. The own models use the library's default BPE
cache (10,000 words).

| Tokenizer | batched, 1 thread | 4 threads | 16 threads | 16 threads, tokens/s | sequential | decode batched |
|---|---|---|---|---|---|---|
| cro-bpe-32k | 4.8 MB/s | 17.2 MB/s | 36.9 MB/s | 8.41 M | 4.0 MB/s | 14.1 M tok/s |
| cro-bpe-32k (tiktoken) | 23.4 MB/s | 60.7 MB/s | 69.1 MB/s | 15.73 M | 26.2 MB/s | 23.4 M tok/s |
| cro-bpe-48k | 4.8 MB/s | 17.1 MB/s | 37.3 MB/s | 8.08 M | 4.0 MB/s | 14.0 M tok/s |
| cro-bpe-48k (tiktoken) | 24.3 MB/s | 62.8 MB/s | 70.8 MB/s | 15.33 M | 27.4 MB/s | 22.5 M tok/s |
| cro-bpe-64k | 4.7 MB/s | 17.0 MB/s | 37.2 MB/s | 7.82 M | 4.0 MB/s | 14.4 M tok/s |
| cro-bpe-64k (tiktoken) | **24.9 MB/s** | **64.0 MB/s** | **72.0 MB/s** | 15.12 M | **28.4 MB/s** | 22.0 M tok/s |
| gpt-4o-200k | 4.4 MB/s | 15.5 MB/s | 32.8 MB/s | 10.02 M | 3.6 MB/s | 13.9 M tok/s |
| gpt-4o-200k (tiktoken) | 17.1 MB/s | 48.1 MB/s | 63.6 MB/s | **19.40 M** | 18.5 MB/s | **27.9 M tok/s** |
| llama-3-128k | 4.9 MB/s | 17.0 MB/s | 33.4 MB/s | 12.19 M | 3.9 MB/s | 13.7 M tok/s |

With `tokenizers`, the regex pre-tokenizer is the main encoding cost; the three
cro-bpe sizes encode at the same speed in MB/s, level with Llama-3 and faster
than GPT-4o on one thread, and faster than both on 16. The tiktoken export of
cro-bpe-32k encodes the same ids 4.9× faster batched on one thread, 1.9× faster
on 16 threads (it gains little beyond 4 threads) and 6.5× faster one document at
a time. It is also faster than GPT-4o on tiktoken in MB/s (1.4× on one thread,
1.1× on 16), which is what matters for a fixed amount of text; GPT-4o emits more
tokens per second only because it needs more tokens for the same text. The
other baselines are not benchmarked by default (`throughput_candidates`).
