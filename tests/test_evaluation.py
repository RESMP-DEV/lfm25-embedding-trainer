import json
from pathlib import Path

import numpy as np
import pytest

from lfm25_embedding_trainer.evaluation import evaluate


class FakeEncoder:
    vectors = {
        "document one": [1.0, 0.0],
        "document two": [0.0, 1.0],
        "query one": [1.0, 0.0],
        "query two": [0.0, 1.0],
    }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_evaluate_ranks_matching_documents_first(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"query": "query one", "positive": "document one", "source": "x", "source_id": "1"},
        {"query": "query two", "positive": "document two", "source": "x", "source_id": "2"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = evaluate(FakeEncoder(), path, batch_size=1)
    assert metrics == {
        "queries": 2.0,
        "mrr": 1.0,
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
    }


def test_evaluate_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        evaluate(FakeEncoder(), path)
