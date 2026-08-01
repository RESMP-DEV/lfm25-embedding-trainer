from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .data import prepare_pairs, read_jsonl, validate_pairs
from .evaluation import evaluate as evaluate_model
from .modeling import EmbeddingEncoder
from .splitting import sample_pairs_by_source, split_pairs
from .sweep import run_sweep
from .training import TrackingConfig
from .training import train as train_model

app = typer.Typer(no_args_is_help=True)


@app.command("prepare")
def prepare(
    input_path: Path,
    output: Path = Path("data/pairs.jsonl"),
    query_field: str = "query",
    positive_field: str = "positive",
    id_field: str = "id",
    source: str = "default",
    source_field: str | None = None,
    group_field: str | None = None,
) -> None:
    """Map an arbitrary JSONL dataset into the trainer's pair schema."""
    count = prepare_pairs(
        input_path,
        output,
        query_field=query_field,
        positive_field=positive_field,
        id_field=id_field,
        source=source,
        source_field=source_field,
        group_field=group_field,
    )
    typer.echo(f"wrote {count} pairs to {output}")


@app.command("validate")
def validate(input_path: Path) -> None:
    """Validate pair fields and report dataset statistics."""
    typer.echo(json.dumps(validate_pairs(input_path), indent=2))


@app.command("split")
def split(
    input_path: Path,
    output_directory: Path = Path("data/splits"),
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    """Create deterministic, document-group-safe train/dev/test splits."""
    typer.echo(
        json.dumps(split_pairs(input_path, output_directory, dev_ratio, test_ratio), indent=2)
    )


@app.command("sample")
def sample(
    input_path: Path,
    output: Path = Path("data/evaluation-sample.jsonl"),
    per_source: Annotated[int, typer.Option(min=1)] = 250,
) -> None:
    """Create a deterministic source-stratified evaluation sample."""
    typer.echo(json.dumps(sample_pairs_by_source(input_path, output, per_source), indent=2))


@app.command("train")
def train(
    pairs: Path,
    output: Path = Path("models/lfm25-embeddings"),
    config: Path = Path("configs/train.toml"),
    device: str = typer.Option("auto", help="auto, cuda (including ROCm), mps, or cpu"),
    validation_pairs: Path | None = None,
    wandb_mode: str = typer.Option("disabled", help="disabled, offline, or online"),
    wandb_project: str = typer.Option("lfm25-embedding-trainer"),
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Fine-tune LFM2.5 Embedding with symmetric multi-positive InfoNCE."""
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise typer.BadParameter("must be disabled, offline, or online", param_hint="wandb-mode")
    train_model(
        pairs,
        output,
        config,
        device,
        TrackingConfig(
            mode=wandb_mode,
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name,
        ),
        validation_pairs,
    )


@app.command("sweep")
def sweep(
    pairs: Path,
    validation_pairs: Path,
    output: Path = Path("models/optuna-sweep"),
    config: Path = Path("configs/cuda.toml"),
    trials: Annotated[int, typer.Option(min=1)] = 6,
    steps_per_trial: Annotated[int, typer.Option(min=1)] = 250,
    device: str = typer.Option("auto", help="auto, cuda (including ROCm), mps, or cpu"),
    wandb_mode: str = typer.Option("online", help="disabled, offline, or online"),
    wandb_project: str = typer.Option("lfm25-embedding-trainer"),
    wandb_entity: str | None = None,
    study_name: str = typer.Option("lfm25-embedding-optuna"),
) -> None:
    """Select hyperparameters with Optuna using validation MRR."""
    if wandb_mode not in {"disabled", "offline", "online"}:
        raise typer.BadParameter("must be disabled, offline, or online", param_hint="wandb-mode")
    result = run_sweep(
        pairs,
        validation_pairs,
        config,
        output,
        trials=trials,
        steps_per_trial=steps_per_trial,
        device=device,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        study_name=study_name,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate")
def evaluate(
    model: str,
    pairs: Path,
    output: Path = Path("artifacts/evaluation.json"),
    revision: str = "main",
    device: str = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
) -> None:
    """Evaluate a Hub model or local checkpoint on labeled retrieval pairs."""
    encoder = EmbeddingEncoder(model, revision=revision, device=device)
    metrics = evaluate_model(encoder, pairs, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))


@app.command("embed")
def embed(
    model: str,
    input_path: Path,
    output: Path = Path("artifacts/embeddings.jsonl"),
    text_field: str = "text",
    id_field: str = "id",
    revision: str = "main",
    device: str = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    max_length: Annotated[int, typer.Option(min=1)] = 512,
    prompt_name: str = typer.Option(
        "document", help="query for search inputs or document for indexed passages"
    ),
) -> None:
    """Encode a JSONL file and write IDs with normalized embedding vectors."""
    if prompt_name not in {"query", "document"}:
        raise typer.BadParameter("must be query or document", param_hint="prompt-name")
    rows = list(read_jsonl(input_path))
    if not rows:
        raise typer.BadParameter("input dataset is empty", param_hint="input-path")
    encoder = EmbeddingEncoder(model, revision=revision, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [str(row[text_field]) for row in batch]
            vectors = encoder.encode(
                texts,
                max_length=max_length,
                prompt_name=prompt_name,
            )
            for row, vector in zip(batch, vectors, strict=True):
                handle.write(json.dumps({"id": row[id_field], "embedding": vector.tolist()}) + "\n")
    typer.echo(f"wrote {len(rows)} embeddings to {output}")


if __name__ == "__main__":
    app()
