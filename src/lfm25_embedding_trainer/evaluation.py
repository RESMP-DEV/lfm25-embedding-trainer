from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .data import _query_identifier, _stable_id, read_jsonl


class Encoder(Protocol):
    def encode(
        self,
        texts: list[str],
        max_length: int = 512,
        prompt_name: Literal["query", "document"] = "document",
    ) -> np.ndarray: ...


def evaluate(model: Encoder, pairs_path: Path, batch_size: int = 32) -> dict[str, float]:
    rows = list(read_jsonl(pairs_path))
    if not rows:
        raise ValueError("evaluation pair file is empty")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    corpus_by_id: dict[tuple[str, str], str] = {}
    query_by_id: dict[tuple[str, str], str] = {}
    relevant_by_query: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for line_number, row in enumerate(rows, 1):
        source = _stable_id(row.get("source"), label=f"pair row {line_number} source")
        document_id = _stable_id(row.get("source_id"), label=f"pair row {line_number} source ID")
        query = row.get("query")
        positive = row.get("positive")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"pair row {line_number} has an empty query")
        if not isinstance(positive, str) or not positive.strip():
            raise ValueError(f"pair row {line_number} has an empty positive document")
        query_key = (source, _query_identifier(row))
        document_key = (source, document_id)
        previous_query = query_by_id.setdefault(query_key, query)
        if previous_query != query:
            raise ValueError(f"query identity {query_key!r} maps to multiple query texts")
        previous_document = corpus_by_id.setdefault(document_key, positive)
        if previous_document != positive:
            raise ValueError(f"document identity {document_key!r} maps to multiple texts")
        relevant_by_query.setdefault(query_key, set()).add(document_key)

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
    query_ids = list(query_by_id)
    query_vectors = batched(list(query_by_id.values()), "query")
    ranks = []
    for query_vector, query_id in zip(query_vectors, query_ids, strict=True):
        order = np.argsort(-(document_vectors @ query_vector))
        inverse_order = np.empty_like(order)
        inverse_order[order] = np.arange(len(order))
        relevant_targets = [target_by_id[key] for key in relevant_by_query[query_id]]
        ranks.append(min(int(inverse_order[target]) + 1 for target in relevant_targets))
    return {
        "queries": float(len(ranks)),
        "mrr": float(np.mean([1 / rank for rank in ranks])),
        "recall_at_1": float(np.mean([rank <= 1 for rank in ranks])),
        "recall_at_5": float(np.mean([rank <= 5 for rank in ranks])),
        "recall_at_10": float(np.mean([rank <= 10 for rank in ranks])),
    }
