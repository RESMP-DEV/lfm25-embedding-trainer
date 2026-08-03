import ast
import inspect
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
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    class LegacyShortConv(Lfm2ShortConv):
        def __init__(self):
            pass

        def slow_forward(self, value):
            return value + 1

    module = LegacyShortConv()

    class Model:
        def modules(self):
            return [module]

    assert _patch_shortconv_seq_idx(Model()) is True
    assert module.slow_forward(2, seq_idx="unused") == 3
    assert _patch_shortconv_seq_idx(Model()) is False
    assert "seq_idx" not in inspect.signature(LegacyShortConv.slow_forward).parameters


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
    tree = ast.parse(destination.read_text())
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_noncausal_shortconv_forward"
    )
    assert [argument.arg for argument in function.args.kwonlyargs] == ["seq_idx"]


def test_saved_remote_code_patch_tolerates_signature_formatting(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(
        "def _noncausal_shortconv_forward(self, hidden_states, attention_mask = None)->object:\n"
        "    return hidden_states\n"
    )
    destination = tmp_path / "destination.py"

    _copy_compatible_remote_code(source, destination)

    namespace: dict[str, object] = {}
    exec(destination.read_text(), namespace)
    signature = inspect.signature(namespace["_noncausal_shortconv_forward"])
    assert signature.parameters["seq_idx"].kind is inspect.Parameter.KEYWORD_ONLY
