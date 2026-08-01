# Training LFM2.5 Embeddings on Your Own Data

## 1. Define the retrieval task

Start with the behavior you need to measure. A query should represent realistic user input, and
its positive document should contain the item that must rank highly. Avoid using only document
titles as queries if production queries are questions, descriptions, or error messages.

Useful sources of supervision include:

- search queries paired with clicked or accepted documents;
- questions paired with verified answers or source passages;
- product descriptions paired with canonical catalog records;
- support requests paired with the resolution article; and
- human-authored paraphrases paired with the same document.

Obtain the rights required to process the data and train a model. Remove secrets and personal or
regulated information unless your entire training and tracking environment is authorized to
handle it.

## 2. Assign stable IDs

Every positive document needs a stable `source_id`. Multiple queries for the same document
should share the same ID; the loss treats those matches as multiple positives rather than false
negatives.

Use `group_id` when several records must remain together. For example, chunks from one manual
can have distinct `source_id` values and one shared parent `group_id`. The splitter hashes the
parent group so those chunks cannot leak across train, development, and test.

```json
{"query":"install the desktop client","positive":"Linux installation …","source":"docs","source_id":"install:linux","group_id":"install"}
```

`source` is a namespace, not a relevance label. It prevents IDs from different datasets from
colliding and supports deterministic source-stratified evaluation samples.

## 3. Convert and validate

If the input already has the canonical fields:

```bash
uv run lfm25-embed validate pairs.jsonl
```

For custom field names:

```bash
uv run lfm25-embed prepare raw.jsonl \
  --output data/pairs.jsonl \
  --query-field user_question \
  --positive-field document_text \
  --id-field document_id \
  --source-field collection \
  --group-field parent_id
```

Validation reports pair count, unique document count, unique group count, and source counts. It
rejects malformed JSON, missing required fields, and empty query/document text.

## 4. Freeze the splits

```bash
uv run lfm25-embed split data/pairs.jsonl \
  --output-directory data/splits \
  --dev-ratio 0.1 --test-ratio 0.1
```

The mapping is deterministic: adding unrelated documents does not reshuffle existing groups.
Record hashes of the source file and each split. Use development data for hyperparameter choice;
use test data only for the final matched comparison.

Time-dependent applications should add a second temporal test in which documents and queries
come from a later period. Hash splitting alone does not detect temporal memorization.

## 5. Establish the base-model result

Before training, evaluate the exact pinned base revision:

```bash
uv run lfm25-embed evaluate LiquidAI/LFM2.5-Embedding-350M \
  data/splits/test.jsonl \
  --revision f35ae2c91d687658dbf1f2b449382f0b019b9808 \
  --output artifacts/base.json
```

This is the control. A tuned score without the matched base result does not establish an
improvement.

## 6. Choose an initial configuration

Start conservatively:

- learning rate: `1e-5` to `5e-5`;
- warmup ratio: `0.03` to `0.1`;
- weight decay: `0.01` to `0.1`;
- temperature: `0.03` to `0.1`;
- BF16 on compatible NVIDIA hardware, FP16 on officially validated Ryzen/ROCm hardware,
  otherwise FP32; and
- the largest batch that survives representative long examples.

Contrastive learning benefits from larger batches because each batch supplies more negatives.
Gradient accumulation does not provide the same negative set as one physically larger batch.
When memory is limited, improve negative quality rather than assuming accumulation is equivalent.

`configs/train.toml` is a portable FP32 starting point. `configs/cuda.toml` is a bounded BF16
NVIDIA CUDA run, and `configs/amd-rocm.toml` is a bounded FP16 ROCm run. Edit `max_steps` and
batch size for your corpus and hardware. See [Fine-tuning on AMD ROCm](AMD_ROCM.md) before
installing PyTorch on an AMD system.

## 7. Track and sweep

Use W&B online, W&B offline, or disabled tracking. Do not place sensitive raw text in run names,
configuration values, or logs.

Run short Optuna trials against a fixed development sample, select the best parameters, and then
restart a full run from the original base model. Do not continue the short winning checkpoint if
the intended comparison is a full schedule from a common initialization.

## 8. Evaluate beyond aggregate recall

MRR and Recall@K are necessary but incomplete. Break results down by source, query type, length,
language, and difficulty. Maintain a small human-reviewed hard set containing:

- plausible but wrong near-duplicates;
- ambiguous or underspecified queries;
- rare terminology and spelling variants;
- long documents and long queries;
- documents updated after the training cutoff; and
- cases where retrieval should abstain or return no confident match.

Inspect regressions, not only averages. Weak labels can produce very high scores when the query
copies words from its positive document while still failing realistic search traffic.

## 9. Export and integrate

Use `lfm25-embed embed --prompt-name document` to create normalized vectors for the index, and use
`--prompt-name query` for live search inputs. The trainer and evaluator apply these roles
automatically. Omitting or swapping the checkpoint's asymmetric prompts silently changes retrieval
behavior.

Use exactly the same checkpoint, native CLS pooling, normalization, and maximum-length policy for
both indexed documents and live queries. Rebuild the document index whenever any of these changes.

Version the index by checkpoint and corpus hash. Build a new collection for a candidate model,
validate it, then switch traffic atomically. Keep the previous index available for rollback.
