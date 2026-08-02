from pathlib import Path

import torch

from lfm25_embedding_trainer.training import (
    TrainConfig,
    _identity_key,
    _load_pairs,
    _multi_positive_loss,
    _positive_mask,
    _write_progress,
)


def test_multi_positive_loss_accepts_duplicate_documents() -> None:
    scores = torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    duplicate_aware = _multi_positive_loss(scores, ["q1", "q2", "q3"], ["same", "same", "other"])
    expected_mask = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    assert torch.equal(
        _positive_mask(["q1", "q2", "q3"], ["same", "same", "other"], device="cpu"),
        expected_mask,
    )
    expected = -torch.logsumexp(
        torch.log_softmax(scores, dim=1).masked_fill(~expected_mask, -torch.inf), dim=1
    ).mean()
    assert torch.allclose(duplicate_aware, expected)


def test_multi_positive_loss_accepts_multiple_documents_for_one_query() -> None:
    scores = torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    duplicate_aware = _multi_positive_loss(scores, ["same", "same", "other"], ["d1", "d2", "d3"])
    expected_mask = torch.tensor([[True, True, False], [True, True, False], [False, False, True]])
    assert torch.equal(
        _positive_mask(["same", "same", "other"], ["d1", "d2", "d3"], device="cpu"),
        expected_mask,
    )
    expected = -torch.logsumexp(
        torch.log_softmax(scores, dim=1).masked_fill(~expected_mask, -torch.inf), dim=1
    ).mean()
    assert torch.allclose(duplicate_aware, expected)


def test_identity_key_cannot_collide_across_source_namespaces() -> None:
    assert _identity_key("a", "b:c") != _identity_key("a:b", "c")


def test_load_pairs_preserves_falsy_query_id(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"query":"same","query_id":0,"positive":"p1","source":"x","source_id":"1"}\n'
        '{"query":"same","query_id":false,"positive":"p2","source":"x","source_id":"2"}\n'
    )

    pairs = _load_pairs(path)

    assert pairs[0][2] == _identity_key("x", "0")
    assert pairs[1][2] == _identity_key("x", "False")


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
