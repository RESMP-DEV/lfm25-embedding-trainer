from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

PromptName = Literal["query", "document"]


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
        self.model.save_pretrained(str(directory), safe_serialization=True)
        remote_code = Path(inspect.getfile(self.model[0].auto_model.__class__))
        shutil.copy2(remote_code, directory / "modeling_lfm2_bidirectional.py")
