from pathlib import Path
from types import SimpleNamespace

from lfm25_embedding_trainer.modeling import (
    _copy_compatible_remote_code,
    _patch_shortconv_seq_idx,
    accelerator_backend,
    device_type,
)


def test_rocm_reuses_cuda_device_but_is_identified() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip="7.2.1", cuda=None))
    assert accelerator_backend(torch_module, "cuda") == "rocm"


def test_nvidia_cuda_is_not_mislabeled_as_rocm() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0"))
    assert accelerator_backend(torch_module, "cuda") == "cuda"


def test_indexed_rocm_device_is_identified() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip="7.2.1", cuda=None))
    assert device_type("cuda:0") == "cuda"
    assert accelerator_backend(torch_module, "cuda:0") == "rocm"


def test_indexed_nvidia_device_is_not_mislabeled_as_rocm() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0"))
    assert device_type("cuda:1") == "cuda"
    assert accelerator_backend(torch_module, "cuda:1") == "cuda"


def test_shortconv_patch_ignores_transformers5_seq_idx() -> None:
    class Lfm2ShortConv:
        def slow_forward(self, value):
            return value + 1

    class Model:
        def modules(self):
            return [Lfm2ShortConv()]

    assert _patch_shortconv_seq_idx(Model()) is True
    assert Lfm2ShortConv().slow_forward(2, seq_idx="unused") == 3
    assert _patch_shortconv_seq_idx(Model()) is False


def test_saved_remote_code_accepts_seq_idx(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "def _noncausal_shortconv_forward(\n"
        "    self,\n"
        "    attention_mask: Optional[torch.Tensor] = None,\n"
        ") -> torch.Tensor:\n"
        "    return attention_mask\n"
    )
    destination = tmp_path / "destination.py"
    _copy_compatible_remote_code(source, destination)
    assert "    seq_idx=None,\n" in destination.read_text()
