from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

import numpy as np


class Encoder(Protocol):
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: Literal["query", "document"] = "document",
    ) -> np.ndarray: ...


def evaluate(model: Encoder, pairs_path: Path, batch_size: int = 32) -> dict[str, float]:
    rows = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("evaluation pair file is empty")
    corpus_by_id = {(row["source"], row["source_id"]): row["positive"] for row in rows}
    corpus_ids = list(corpus_by_id)
    corpus = list(corpus_by_id.values())
    target_by_id = {key: index for index, key in enumerate(corpus_ids)}

    def batched(texts: list[str], prompt_name: Literal["query", "document"]) -> np.ndarray:
        return np.concatenate(
            [
                model.encode(texts[i : i + batch_size], prompt_name=prompt_name)
                for i in range(0, len(texts), batch_size)
            ]
        )

    document_vectors = batched(corpus, "document")
    query_vectors = batched([row["query"] for row in rows], "query")
    ranks = []
    for query_vector, row in zip(query_vectors, rows, strict=True):
        order = np.argsort(-(document_vectors @ query_vector))
        target = target_by_id[(row["source"], row["source_id"])]
        ranks.append(int(np.where(order == target)[0][0]) + 1)
    return {
        "queries": float(len(ranks)),
        "mrr": float(np.mean([1 / rank for rank in ranks])),
        "recall_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "recall_at_5": float(np.mean([rank <= 5 for rank in ranks])),
        "recall_at_10": float(np.mean([rank <= 10 for rank in ranks])),
    }
