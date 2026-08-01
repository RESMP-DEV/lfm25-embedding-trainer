from types import SimpleNamespace

from lfm25_embedding_trainer.modeling import accelerator_backend


def test_rocm_reuses_cuda_device_but_is_identified() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip="7.2.1", cuda=None))
    assert accelerator_backend(torch_module, "cuda") == "rocm"


def test_nvidia_cuda_is_not_mislabeled_as_rocm() -> None:
    torch_module = SimpleNamespace(version=SimpleNamespace(hip=None, cuda="13.0"))
    assert accelerator_backend(torch_module, "cuda") == "cuda"
