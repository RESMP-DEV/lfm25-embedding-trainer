from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

PromptName = Literal["query", "document"]


def _patch_shortconv_seq_idx(model: Any) -> bool:
    """Adapt Liquid's pinned non-causal short-conv patch to Transformers 5.x."""
    shortconv_type = next(
        (type(module) for module in model.modules() if type(module).__name__ == "Lfm2ShortConv"),
        None,
    )
    if shortconv_type is None:
        raise ValueError("embedding checkpoint does not contain an Lfm2ShortConv module")
    slow_forward = shortconv_type.slow_forward
    if "seq_idx" in inspect.signature(slow_forward).parameters:
        return False

    def compatible_slow_forward(self, *args, seq_idx=None, **kwargs):
        del seq_idx
        return slow_forward(self, *args, **kwargs)

    shortconv_type.slow_forward = compatible_slow_forward
    return True


def _copy_compatible_remote_code(source: Path, destination: Path) -> None:
    code = source.read_text(encoding="utf-8")
    function_start = code.find("def _noncausal_shortconv_forward(")
    function_end = code.find(") -> torch.Tensor:", function_start)
    if function_start < 0 or function_end < 0:
        raise ValueError("unexpected LFM2 remote-code layout")
    signature = code[function_start:function_end]
    if "seq_idx" not in signature:
        marker = "    attention_mask: Optional[torch.Tensor] = None,\n"
        marker_at = code.find(marker, function_start, function_end)
        if marker_at < 0:
            raise ValueError("cannot add Transformers 5.x seq_idx compatibility to remote code")
        insert_at = marker_at + len(marker)
        code = code[:insert_at] + "    seq_idx=None,\n" + code[insert_at:]
    destination.write_text(code, encoding="utf-8")


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type(device: str) -> str:
    """Return the accelerator type for an optionally indexed device string."""
    return device.partition(":")[0]


def accelerator_backend(torch_module: Any, device: str) -> str:
    """Return the physical accelerator while retaining PyTorch's device spelling.

    PyTorch intentionally exposes ROCm devices through the ``cuda`` API. The
    distinction is still useful for receipts and backend-specific safe defaults.
    """
    if device_type(device) == "cuda" and getattr(torch_module.version, "hip", None):
        return "rocm"
    return device_type(device)


class EmbeddingEncoder:
    """Prompt-aware LFM2.5 embedding checkpoint with its native CLS pooling."""

    def __init__(self, model_id: str, revision: str = "main", device: str = "auto") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.device = resolve_device(device)
        self.device_type = device_type(self.device)
        self.accelerator = accelerator_backend(torch, self.device)
        self.model = SentenceTransformer(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )
        _patch_shortconv_seq_idx(self.model)
        maximum_length = self.model.max_seq_length
        if maximum_length is None:
            raise ValueError("embedding checkpoint does not declare a maximum sequence length")
        self.maximum_length = maximum_length
        required_prompts = {"query", "document"}
        missing_prompts = required_prompts.difference(self.model.prompts)
        if missing_prompts:
            missing = ", ".join(sorted(missing_prompts))
            raise ValueError(
                f"model is not a prompt-aware embedding checkpoint; missing prompts: {missing}"
            )

    def encode_torch(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: PromptName = "document",
    ):
        if max_length > self.maximum_length:
            raise ValueError(
                f"max_length {max_length} exceeds the checkpoint limit {self.maximum_length}"
            )
        self.model.max_seq_length = max_length
        features = self.model.preprocess(
            cast(list[Any], texts), prompt=self.model.prompts[prompt_name]
        )
        features = {
            key: value.to(self.device) if isinstance(value, self.torch.Tensor) else value
            for key, value in features.items()
        }
        embeddings = self.model(features)["sentence_embedding"]
        return self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: PromptName = "document",
    ) -> np.ndarray:
        self.model.eval()
        with self.torch.inference_mode():
            return (
                self.encode_torch(texts, max_length=max_length, prompt_name=prompt_name)
                .float()
                .cpu()
                .numpy()
            )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        # Training may deliberately truncate batches below the architecture limit.
        # Do not persist that transient value as the checkpoint's maximum length.
        self.model.max_seq_length = self.maximum_length
        self.model.save_pretrained(str(directory), safe_serialization=True)
        remote_code = Path(inspect.getfile(self.model[0].auto_model.__class__))
        destination = directory / "modeling_lfm2_bidirectional.py"
        _copy_compatible_remote_code(remote_code, destination)
        shutil.copystat(remote_code, destination)
