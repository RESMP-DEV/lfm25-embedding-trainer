from types import SimpleNamespace

from lfm25_embedding_trainer.modeling import accelerator_backend, device_type


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
