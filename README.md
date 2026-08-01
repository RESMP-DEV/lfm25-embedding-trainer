# LFM2.5 Embedding Trainer

Fine-tune [LiquidAI/LFM2.5-Embedding-350M](https://huggingface.co/LiquidAI/LFM2.5-Embedding-350M)
for dense retrieval using your own query-document pairs.

This repository provides a small, auditable training pipeline rather than a framework-specific
dataset. It includes:

- symmetric multi-positive InfoNCE training;
- the checkpoint's native CLS pooling, asymmetric query/document prompts, and L2 normalization;
- deterministic document/group-safe train, development, and test splits;
- Recall@1/5/10 and mean reciprocal rank evaluation;
- Optuna hyperparameter selection on development MRR;
- optional online or offline Weights & Biases tracking;
- atomic progress files, pinned model revisions, dataset hashes, and training receipts; and
- JSONL embedding export for downstream vector databases or search systems.

## Install

Python 3.11–3.13 is supported. [uv](https://docs.astral.sh/uv/) is recommended:

```bash
git clone https://github.com/RESMP-DEV/lfm25-embedding-trainer.git
cd lfm25-embedding-trainer
uv sync --extra train --extra dev
```

PyTorch selects NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU automatically. PyTorch intentionally
reports ROCm devices as `cuda`; use `--device cuda` on AMD. ROCm requires a hardware-matched
PyTorch build rather than the default Linux PyTorch wheel. See [Fine-tuning on AMD ROCm](docs/AMD_ROCM.md).

The FP16 path is live-validated on an 8 GB Radeon RX 5700 (`gfx1010`) with ROCm 7.14 and
PyTorch 2.12. See the [sanitized hardware receipt](docs/receipts/amd-rx5700-rocm714-smoke.json).

## Why the embedding checkpoint?

`LFM2.5-Encoder-350M` is a bidirectional backbone. `LFM2.5-Embedding-350M` starts from that
architecture but has already received Liquid's retrieval-specific contrastive, multilingual,
distillation, and hard-negative training. It also defines the inference contract used by its
published retrieval results: CLS pooling plus `query: ` and `document: ` prefixes.

This project therefore fine-tunes the **embedding checkpoint**, not the encoder checkpoint.
Versions `0.1.x` and `0.2.x` incorrectly initialized from `LFM2.5-Encoder-350M`, applied mean
pooling, and omitted the asymmetric prompts. Version `0.3.0` corrects all three behaviors. Treat
checkpoints produced by the older path as a separate experiment; they are not drop-in compatible,
and moving to `0.3.0` requires rebuilding document embeddings.

An end-to-end one-step smoke run (including the real model download and checkpoint save) is:

```bash
uv run lfm25-embed train examples/pairs.jsonl \
  --config configs/smoke.toml --output models/smoke
```

## Pair format

Training data is UTF-8 JSONL with one object per query-positive pair:

```json
{"query":"reset a wireless router","positive":"Hold the reset button for ten seconds.","source":"support","source_id":"router-reset"}
```

Required fields:

| Field | Meaning |
|---|---|
| `query` | Query or task text presented to the retriever |
| `positive` | Document that should be retrieved |
| `source` | Dataset/domain label used for namespacing and stratified sampling |
| `source_id` | Stable document identifier; repeated queries may share it |
| `group_id` | Optional parent ID that keeps chunks or related records in one split |

If your fields have different names, map them without writing code:

```bash
uv run lfm25-embed prepare my-data.jsonl \
  --output data/pairs.jsonl \
  --query-field question \
  --positive-field answer \
  --id-field document_id \
  --source my-dataset \
  --group-field parent_document_id

uv run lfm25-embed validate data/pairs.jsonl
```

See [Training on your own data](docs/OWN_DATA.md) for dataset construction and evaluation
guidance.

## Split, train, and evaluate

Never tune on the test set. First create deterministic splits keyed by `source` and
`group_id`/`source_id`:

```bash
uv run lfm25-embed split data/pairs.jsonl --output-directory data/splits
```

For CPU or Apple MPS, start with the FP32 configuration:

```bash
uv run lfm25-embed train data/splits/train.jsonl \
  --config configs/train.toml \
  --output models/lfm25-my-data \
  --validation-pairs data/splits/dev.jsonl
```

For a BF16 CUDA GPU, use:

```bash
uv run lfm25-embed train data/splits/train.jsonl \
  --config configs/cuda.toml \
  --output models/lfm25-my-data \
  --validation-pairs data/splits/dev.jsonl \
  --device cuda
```

For an AMD Ryzen APU or Radeon GPU with a supported ROCm PyTorch build, start with validated
FP16 and the conservative AMD profile:

```bash
uv run lfm25-embed train data/splits/train.jsonl \
  --config configs/amd-rocm.toml \
  --output models/lfm25-my-data-rocm \
  --validation-pairs data/splits/dev.jsonl \
  --device cuda
```

The training receipt records `accelerator: "rocm"`, the HIP and PyTorch versions, and whether
fused AdamW was used. The portable AdamW path is selected on ROCm until the exact target has
been benchmarked.

Compare the pinned base model and tuned checkpoint on the same untouched test set:

```bash
uv run lfm25-embed evaluate LiquidAI/LFM2.5-Embedding-350M \
  data/splits/test.jsonl \
  --revision f35ae2c91d687658dbf1f2b449382f0b019b9808 \
  --output artifacts/base.json

uv run lfm25-embed evaluate models/lfm25-my-data \
  data/splits/test.jsonl \
  --output artifacts/tuned.json
```

## Weights & Biases

Tracking is optional. Authenticate once and select online mode:

```bash
uv run wandb login
uv run lfm25-embed train data/splits/train.jsonl \
  --config configs/cuda.toml \
  --validation-pairs data/splits/dev.jsonl \
  --wandb-mode online \
  --wandb-project my-embedding-project
```

Use `--wandb-mode offline` on disconnected machines, then run `wandb sync RUN_DIRECTORY`.
Training loss, learning rate, gradient norm, throughput, validation retrieval metrics, model
revision, and pair-file SHA-256 are recorded.

## Optuna sweep

The sweep searches learning rate, weight decay, warmup ratio, and contrastive temperature. It
selects by development MRR and never reads the test set:

```bash
uv run lfm25-embed sample data/splits/dev.jsonl \
  --output data/dev-sample.jsonl --per-source 250

uv run lfm25-embed sweep data/splits/train.jsonl data/dev-sample.jsonl \
  --config configs/cuda.toml \
  --output models/optuna-sweep \
  --trials 8 --steps-per-trial 250 \
  --device cuda \
  --wandb-mode online \
  --wandb-project my-embedding-project
```

The persistent SQLite study and `sweep_summary.json` make selection reproducible. Non-winning
trial weights are removed after the study; their configurations, receipts, metrics, and W&B
runs remain.

## Export embeddings

Input:

```json
{"id":"doc-1","text":"The document text to encode."}
```

Command:

```bash
uv run lfm25-embed embed models/lfm25-my-data documents.jsonl \
  --output artifacts/embeddings.jsonl \
  --prompt-name document
```

Each output row contains the original ID and a normalized floating-point vector. Import that
JSONL into the vector store of your choice. Use `--prompt-name query` when embedding live search
queries; `document` is the default because bulk export normally builds the document index.

## Implementation notes

The model repository uses Sentence Transformers plus custom Transformers code. This project loads
the complete embedding package, trains its retrieval-initialized weights through its native CLS
pooling path, applies the checkpoint's query/document prompts on the correct sides, and saves all
Sentence Transformers metadata. It also copies Liquid's bidirectional modeling file into each
checkpoint so a local save remains reloadable.

Keep `model_revision` pinned to a reviewed Hugging Face commit because `trust_remote_code=True`
executes code from that revision. The current remote code is compatible with Transformers 4.x but
not Transformers 5.x, so the training extra deliberately constrains `transformers>=4.56,<5`.

In-batch documents are negatives. When several rows in one batch point to the same document,
all matching columns are treated as positives, avoiding false-negative gradients. Long or
variable-length data can cause late memory spikes; calibrate batch size against representative
long examples rather than only short smoke data.

## Licensing

The source code in this repository is dual-licensed under your choice of:

- [Apache License 2.0](LICENSE-APACHE), or
- [MIT License](LICENSE-MIT).

The LiquidAI model weights and remote model code are separate works and are currently published
under the **LFM Open License v1.0**. Review the model card and its license before downloading,
fine-tuning, or redistributing weights. Your datasets and resulting checkpoints may also have
independent obligations. This repository does not relicense model weights or training data.
