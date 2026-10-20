from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from lfm25_embedding_trainer.sweep import run_sweep
from lfm25_embedding_trainer.training import TrainConfig


def _write_base_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'model_id = "LiquidAI/LFM2.5-Embedding-350M"',
                'model_revision = "f35ae2c91d687658dbf1f2b449382f0b019b9808"',
                "max_length = 512",
                "batch_size = 32",
                "learning_rate = 0.000025",
                "weight_decay = 0.018891200276189388",
                "warmup_ratio = 0.051818996064852074",
                "epochs = 1",
                "temperature = 0.03223743847396823",
                "gradient_accumulation_steps = 1",
                "seed = 42",
                "max_steps = 4",
                "log_every_steps = 1",
                'precision = "fp32"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_sweep_searches_batch_size_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch size is the in-batch negative count, so it must be searchable."""
    base_config = tmp_path / "base.toml"
    _write_base_config(base_config)
    observed: list[TrainConfig] = []

    def fake_train(
        pairs_path: Path,
        output_dir: Path,
        config_path: Path,
        device: str = "auto",
        tracking: Any = None,
        validation_pairs_path: Path | None = None,
    ) -> dict[str, Any]:
        config = TrainConfig.load(config_path)
        observed.append(config)
        return {
            "validation": {
                "validation/queries": 1.0,
                "validation/mrr": 0.5,
                "validation/recall_at_1": 0.5,
                "validation/recall_at_5": 0.5,
                "validation/recall_at_10": 0.5,
            },
            "wandb_run_id": "test",
        }

    monkeypatch.setattr("lfm25_embedding_trainer.sweep.train", fake_train)

    summary = run_sweep(
        tmp_path / "pairs.jsonl",
        tmp_path / "dev.jsonl",
        base_config,
        tmp_path / "out",
        trials=3,
        steps_per_trial=4,
        device="cpu",
        wandb_mode="disabled",
        study_name="batch-size-test",
        batch_sizes=(32, 128, 512),
    )

    assert observed, "sweep did not run any trial"
    searched = {config.batch_size for config in observed}
    assert searched <= {32, 128, 512}
    assert len(searched) > 1, "batch size was not actually varied across trials"
    assert summary["searched_batch_sizes"] == [32, 128, 512]
    assert "in-batch negative count" in summary["batch_size_note"]


def test_sweep_keeps_base_batch_size_when_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit search space the base batch size is preserved."""
    base_config = tmp_path / "base.toml"
    _write_base_config(base_config)
    observed: list[TrainConfig] = []

    def fake_train(
        pairs_path: Path,
        output_dir: Path,
        config_path: Path,
        device: str = "auto",
        tracking: Any = None,
        validation_pairs_path: Path | None = None,
    ) -> dict[str, Any]:
        observed.append(TrainConfig.load(config_path))
        return {
            "validation": {"validation/queries": 1.0, "validation/mrr": 0.5},
            "wandb_run_id": "test",
        }

    monkeypatch.setattr("lfm25_embedding_trainer.sweep.train", fake_train)

    summary = run_sweep(
        tmp_path / "pairs.jsonl",
        tmp_path / "dev.jsonl",
        base_config,
        tmp_path / "out",
        trials=2,
        steps_per_trial=4,
        device="cpu",
        wandb_mode="disabled",
        study_name="fixed-batch-test",
    )

    assert {config.batch_size for config in observed} == {32}
    assert summary["searched_batch_sizes"] is None


def test_train_config_batch_size_is_optional_to_replace() -> None:
    """The sweep replaces batch size alongside the other searched parameters."""
    base = TrainConfig(
        model_id="model",
        model_revision="main",
        max_length=512,
        batch_size=32,
        learning_rate=2.5e-5,
        weight_decay=0.018,
        warmup_ratio=0.05,
        epochs=1,
        temperature=0.032,
        gradient_accumulation_steps=1,
        seed=42,
    )
    assert replace(base, batch_size=512).batch_size == 512
    assert json.dumps({"batch_size": 512})
