from __future__ import annotations

import hashlib
import json
import math
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .data import _query_identifier
from .modeling import EmbeddingEncoder


@dataclass(slots=True)
class TrainConfig:
    model_id: str
    model_revision: str
    max_length: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    epochs: int
    temperature: float
    gradient_accumulation_steps: int
    seed: int
    max_steps: int | None = None
    log_every_steps: int = 10
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    fp16_initial_scale: float = 128.0

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        with path.open("rb") as handle:
            return cls(**tomllib.load(handle))


@dataclass(slots=True)
class TrackingConfig:
    mode: Literal["disabled", "offline", "online"] = "disabled"
    project: str = "lfm25-embedding-trainer"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def _identity_key(source: object, identifier: object) -> str:
    """Encode a namespaced identity without delimiter collisions."""
    return json.dumps([str(source), str(identifier)], ensure_ascii=False, separators=(",", ":"))


def _load_pairs(path: Path) -> list[tuple[str, str, str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_key = _identity_key(row["source"], _query_identifier(row))
            document_key = _identity_key(row["source"], row["source_id"])
            pairs.append((row["query"], row["positive"], query_key, document_key))
    if not pairs:
        raise ValueError("training pair file is empty")
    return pairs


def _relevance_by_query(
    pairs: list[tuple[str, str, str, str]],
) -> dict[str, set[str]]:
    relevance: dict[str, set[str]] = {}
    for _, _, query_key, document_key in pairs:
        relevance.setdefault(query_key, set()).add(document_key)
    return relevance


def _positive_mask(
    query_keys: list[str],
    document_keys: list[str],
    *,
    device,
    relevance_by_query: dict[str, set[str]] | None = None,
):
    """Build known edge labels; shared nodes do not imply transitive relevance."""
    import torch

    if len(query_keys) != len(document_keys):
        raise ValueError("query and document key counts must match")
    if relevance_by_query is None:
        relevance_by_query = {}
        for query_key, document_key in zip(query_keys, document_keys, strict=True):
            relevance_by_query.setdefault(query_key, set()).add(document_key)

    document_positions: dict[str, list[int]] = {}
    for index, document_key in enumerate(document_keys):
        document_positions.setdefault(document_key, []).append(index)
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, query_key in enumerate(query_keys):
        for document_key in relevance_by_query.get(query_key, set()):
            for column_index in document_positions.get(document_key, []):
                row_indices.append(row_index)
                column_indices.append(column_index)

    positive_mask = torch.zeros(
        (len(query_keys), len(document_keys)), dtype=torch.bool, device=device
    )
    if row_indices:
        coordinates = torch.tensor([row_indices, column_indices], dtype=torch.int64, device=device)
        positive_mask[coordinates[0], coordinates[1]] = True
    return positive_mask


def _multi_positive_loss(
    scores,
    query_keys: list[str],
    document_keys: list[str],
    relevance_by_query: dict[str, set[str]] | None = None,
):
    """InfoNCE that preserves known many-to-many query/document relevance."""
    import torch

    if scores.ndim != 2 or scores.shape != (len(query_keys), len(document_keys)):
        raise ValueError("scores must be square with one row per query/document pair")
    positive_mask = _positive_mask(
        query_keys,
        document_keys,
        device=scores.device,
        relevance_by_query=relevance_by_query,
    )

    def direction(logits, mask):
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
        positive_log_probability = torch.logsumexp(
            log_probabilities.masked_fill(~mask, float("-inf")), dim=1
        )
        return -positive_log_probability.mean()

    return (direction(scores, positive_mask) + direction(scores.T, positive_mask.T)) / 2


def train(
    pairs_path: Path,
    output_dir: Path,
    config_path: Path,
    device: str = "auto",
    tracking: TrackingConfig | None = None,
    validation_pairs_path: Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    config = TrainConfig.load(config_path)
    tracking = tracking or TrackingConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    pairs = _load_pairs(pairs_path)
    relevance_by_query = _relevance_by_query(pairs)
    loader = DataLoader(cast(Any, pairs), batch_size=config.batch_size, shuffle=True)
    encoder = EmbeddingEncoder(config.model_id, config.model_revision, device)
    if config.precision in {"fp16", "bf16"} and encoder.device_type != "cuda":
        raise ValueError("fp16 and bf16 training require a CUDA or ROCm device")
    if config.precision == "bf16":
        with torch.cuda.device(encoder.device):
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            raise ValueError("bf16 is not supported by this device; use fp16 or fp32")
    if encoder.device_type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(encoder.device)
    encoder.model.train()
    # Fused AdamW is a useful NVIDIA default, but its support varies across
    # ROCm device families and PyTorch builds. Prefer the portable implementation
    # until the exact AMD target has been probed.
    fused_optimizer = encoder.accelerator == "cuda"
    optimizer = AdamW(
        encoder.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=fused_optimizer,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.precision == "fp16",
        init_scale=config.fp16_initial_scale,
    )
    amp_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
    available_steps = math.ceil(len(loader) / config.gradient_accumulation_steps) * config.epochs
    update_steps = min(available_steps, config.max_steps or available_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(update_steps * config.warmup_ratio), update_steps
    )
    optimizer.zero_grad(set_to_none=True)
    step = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    started_clock = time.monotonic()
    run: Any = None
    if tracking.mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("install the train extra to enable W&B tracking") from exc
        wandb_directory = output_dir / "wandb"
        wandb_directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=tracking.project,
            entity=tracking.entity,
            name=tracking.run_name,
            group=tracking.group,
            mode=tracking.mode,
            dir=str(wandb_directory),
            config={
                **asdict(config),
                "pairs": len(pairs),
                "pairs_sha256": _sha256(pairs_path),
                "model_revision": config.model_revision,
                "device": encoder.device,
                "accelerator": encoder.accelerator,
                "fused_optimizer": fused_optimizer,
            },
            tags=["embedding-training", "lfm2.5-embedding", "contrastive"],
            job_type="train",
        )
    accumulated_loss = 0.0
    overflow_count = 0
    validation_metrics: dict[str, float] = {}
    try:
        for epoch in range(config.epochs):
            for batch_index, batch in enumerate(loader, 1):
                queries = list(batch[0])
                positives = list(batch[1])
                query_keys = list(batch[2])
                document_keys = list(batch[3])
                with torch.autocast(
                    device_type=encoder.device_type,
                    dtype=amp_dtype,
                    enabled=config.precision in {"fp16", "bf16"},
                ):
                    query_embeddings = encoder.encode_torch(
                        queries, config.max_length, prompt_name="query"
                    )
                    positive_embeddings = encoder.encode_torch(
                        positives, config.max_length, prompt_name="document"
                    )
                    scores = query_embeddings @ positive_embeddings.T / config.temperature
                    raw_loss = _multi_positive_loss(
                        scores, query_keys, document_keys, relevance_by_query
                    )
                accumulated_loss += float(raw_loss.detach().cpu())
                loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                should_update = (
                    batch_index % config.gradient_accumulation_steps == 0
                    or batch_index == len(loader)
                )
                if not should_update:
                    continue
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.0)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.is_enabled() and scaler.get_scale() < scale_before_step:
                    overflow_count += 1
                    accumulated_loss = 0.0
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        json.dumps(
                            {
                                "event": "fp16_overflow",
                                "epoch": epoch + 1,
                                "batch": batch_index,
                                "gradient_norm": float(gradient_norm.detach().cpu()),
                                "previous_scale": scale_before_step,
                                "new_scale": scaler.get_scale(),
                            }
                        ),
                        flush=True,
                    )
                    continue
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite gradient norm")
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                elapsed = time.monotonic() - started_clock
                metrics = {
                    "train/loss": accumulated_loss / config.gradient_accumulation_steps,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/gradient_norm": float(gradient_norm.detach().cpu()),
                    "train/epoch": epoch + batch_index / len(loader),
                    "train/examples_seen": step
                    * config.batch_size
                    * config.gradient_accumulation_steps,
                    "train/steps_per_second": step / elapsed,
                }
                accumulated_loss = 0.0
                if run is not None:
                    run.log(metrics, step=step)
                if step == 1 or step % config.log_every_steps == 0 or step == update_steps:
                    progress = {
                        "started_at": started_at.isoformat(),
                        "updated_at": datetime.now(UTC).isoformat(),
                        "step": step,
                        "total_steps": update_steps,
                        "wandb_mode": tracking.mode,
                        **metrics,
                    }
                    _write_progress(output_dir / "training_progress.json", progress)
                    print(json.dumps(progress), flush=True)
                if step >= update_steps:
                    break
            if step >= update_steps:
                break
        if step == 0:
            raise RuntimeError("training completed without a successful optimizer step")
        if validation_pairs_path is not None:
            from .evaluation import evaluate

            validation_metrics = {
                f"validation/{name}": value
                for name, value in evaluate(encoder, validation_pairs_path).items()
            }
            if run is not None:
                run.log(validation_metrics, step=step)
                for name, value in validation_metrics.items():
                    run.summary[name] = value
            print(json.dumps(validation_metrics), flush=True)
    finally:
        if run is not None:
            run.finish()
    encoder.save(output_dir)
    finished_at = datetime.now(UTC)
    receipt = {
        "config": asdict(config),
        "pairs": len(pairs),
        "pairs_sha256": _sha256(pairs_path),
        "optimizer_steps": step,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": time.monotonic() - started_clock,
        "device": encoder.device,
        "accelerator": encoder.accelerator,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "cuda_version": torch.version.cuda,
        "fused_optimizer": fused_optimizer,
        "fp16_overflow_count": overflow_count,
        "pooling": "cls",
        "prompts": {
            "query": encoder.model.prompts["query"],
            "document": encoder.model.prompts["document"],
        },
        "accelerator_name": (
            torch.cuda.get_device_name(encoder.device)
            if encoder.device_type == "cuda"
            else encoder.device
        ),
        "peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(encoder.device)
            if encoder.device_type == "cuda"
            else None
        ),
        "peak_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved(encoder.device)
            if encoder.device_type == "cuda"
            else None
        ),
        "tracking": asdict(tracking),
        "wandb_run_id": getattr(run, "id", None),
        "wandb_run_url": getattr(run, "url", None),
        "validation": validation_metrics,
    }
    (output_dir / "training_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
