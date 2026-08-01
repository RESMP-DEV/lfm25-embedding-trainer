from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingEncoder:
    """Mean-pooled, L2-normalized LFM2.5 encoder wrapper."""

    def __init__(self, model_id: str, revision: str = "main", device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = resolve_device(device)
        self.tokenizer = cast(
            Any,
            AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True),
        )
        # The published safetensors keys are rooted at ``lfm2.*``. Loading the
        # advertised AutoModel body directly currently drops those weights as
        # unexpected and silently initializes a fresh body. Load the checkpoint's
        # real MLM architecture first, then train/use its populated backbone.
        self.wrapper = AutoModelForMaskedLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).to(self.device)
        self.model = self.wrapper.lfm2

    @staticmethod
    def pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)

    def encode_torch(self, texts: list[str], max_length: int = 512):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**tokens)
        embeddings = self.pool(outputs.last_hidden_state, tokens["attention_mask"])
        return self.torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode(self, texts: list[str], max_length: int = 512) -> np.ndarray:
        self.model.eval()
        with self.torch.inference_mode():
            return self.encode_torch(texts, max_length=max_length).float().cpu().numpy()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.wrapper.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
