from pathlib import Path

import torch

from lfm25_embedding_trainer.training import TrainConfig, _multi_positive_loss, _write_progress


def test_multi_positive_loss_accepts_duplicate_documents() -> None:
    scores = torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    duplicate_aware = _multi_positive_loss(
        scores, ["q1", "q2", "q3"], ["same", "same", "other"]
    )
    single_positive = torch.nn.functional.cross_entropy(scores, torch.arange(3))
    assert duplicate_aware < single_positive


def test_multi_positive_loss_accepts_multiple_documents_for_one_query() -> None:
    scores = torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    duplicate_aware = _multi_positive_loss(
        scores, ["same", "same", "other"], ["d1", "d2", "d3"]
    )
    single_positive = torch.nn.functional.cross_entropy(scores, torch.arange(3))
    assert duplicate_aware < single_positive


def test_train_config_supports_bounded_run(tmp_path: Path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        """model_id = "model"
model_revision = "revision"
max_length = 128
batch_size = 4
learning_rate = 0.00002
weight_decay = 0.1
warmup_ratio = 0.1
epochs = 1
temperature = 0.05
gradient_accumulation_steps = 2
seed = 42
max_steps = 10
log_every_steps = 2
precision = "bf16"
"""
    )
    config = TrainConfig.load(path)
    assert config.max_steps == 10
    assert config.log_every_steps == 2
    assert config.precision == "bf16"
    assert config.fp16_initial_scale == 128.0


def test_train_config_supports_rocm_fp16(tmp_path: Path) -> None:
    path = tmp_path / "amd-rocm.toml"
    path.write_text(
        """model_id = "model"
model_revision = "revision"
max_length = 128
batch_size = 4
learning_rate = 0.00002
weight_decay = 0.1
warmup_ratio = 0.1
epochs = 1
temperature = 0.05
gradient_accumulation_steps = 1
seed = 42
precision = "fp16"
fp16_initial_scale = 64.0
"""
    )
    config = TrainConfig.load(path)
    assert config.precision == "fp16"
    assert config.fp16_initial_scale == 64.0


def test_progress_receipt_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    _write_progress(path, {"step": 7})
    assert path.read_text() == '{\n  "step": 7\n}\n'
    assert not path.with_suffix(".json.partial").exists()
