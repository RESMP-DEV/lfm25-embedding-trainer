from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

from .training import TrackingConfig, TrainConfig, train


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_config(config: TrainConfig, path: Path) -> None:
    rows = [
        f"{name} = {_toml_value(value)}"
        for name, value in asdict(config).items()
        if value is not None
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_sweep(
    pairs_path: Path,
    validation_pairs_path: Path,
    base_config_path: Path,
    output_directory: Path,
    *,
    trials: int = 6,
    steps_per_trial: int = 250,
    device: str = "auto",
    wandb_mode: Literal["disabled", "offline", "online"] = "online",
    wandb_project: str = "lfm25-embedding-trainer",
    wandb_entity: str | None = None,
    study_name: str = "lfm25-embedding-optuna",
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("install the train extra to run Optuna sweeps") from exc

    output_directory.mkdir(parents=True, exist_ok=True)
    configs_directory = output_directory / "configs"
    configs_directory.mkdir(exist_ok=True)
    base = TrainConfig.load(base_config_path)
    storage = f"sqlite:///{(output_directory / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        load_if_exists=True,
    )

    def objective(trial) -> float:
        config = replace(
            base,
            learning_rate=trial.suggest_float("learning_rate", 5e-6, 6e-5, log=True),
            weight_decay=trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
            warmup_ratio=trial.suggest_float("warmup_ratio", 0.03, 0.15),
            temperature=trial.suggest_float("temperature", 0.025, 0.1, log=True),
            max_steps=steps_per_trial,
        )
        trial_name = f"trial-{trial.number:03d}"
        config_path = configs_directory / f"{trial_name}.toml"
        trial_directory = output_directory / trial_name
        _write_config(config, config_path)
        receipt = train(
            pairs_path,
            trial_directory,
            config_path,
            device,
            TrackingConfig(
                mode=wandb_mode,
                project=wandb_project,
                entity=wandb_entity,
                run_name=f"{study_name}-{trial_name}",
                group=study_name,
            ),
            validation_pairs_path,
        )
        metrics = receipt["validation"]
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("wandb_run_id", receipt["wandb_run_id"])
        return float(metrics["validation/mrr"])

    study.optimize(objective, n_trials=trials)
    best_number = study.best_trial.number
    for trial_directory in output_directory.glob("trial-*"):
        if trial_directory.name != f"trial-{best_number:03d}":
            (trial_directory / "model.safetensors").unlink(missing_ok=True)
    summary = {
        "study_name": study.study_name,
        "storage": storage,
        "completed_trials": len(study.trials),
        "best_trial": best_number,
        "best_validation_mrr": study.best_value,
        "best_parameters": study.best_params,
        "best_directory": str(output_directory / f"trial-{best_number:03d}"),
    }
    (output_directory / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
