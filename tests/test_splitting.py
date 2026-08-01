import json
from pathlib import Path

from lfm25_embedding_trainer.splitting import sample_pairs_by_source, split_pairs


def test_split_keeps_document_queries_together(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {"query": "q1", "positive": "p", "source": "catalog", "source_id": "1"},
        {"query": "q2", "positive": "p", "source": "catalog", "source_id": "1"},
        {"query": "q3", "positive": "p", "source": "catalog", "source_id": "2"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "splits"
    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)
    assert sum(counts.values()) == 3
    locations = []
    for split in counts:
        contents = (output / f"{split}.jsonl").read_text()
        if '"source_id": "1"' in contents:
            locations.append(split)
    assert len(locations) == 1


def test_split_keeps_chunked_article_together(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query": "q1",
            "positive": "section 1",
            "source": "journal",
            "source_id": "article:s1",
            "group_id": "article",
        },
        {
            "query": "q2",
            "positive": "section 2",
            "source": "journal",
            "source_id": "article:s2",
            "group_id": "article",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "splits"
    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)
    locations = [
        split
        for split in counts
        if '"source_id": "article:s1"' in (output / f"{split}.jsonl").read_text()
        or '"source_id": "article:s2"' in (output / f"{split}.jsonl").read_text()
    ]
    assert len(locations) == 1


def test_sample_pairs_is_deterministic_and_stratified(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {"query": f"q{i}", "positive": "p", "source": source_name, "source_id": str(i)}
        for source_name in ("catalog", "manuals")
        for i in range(5)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    assert sample_pairs_by_source(source, first, 2) == {"catalog": 2, "manuals": 2}
    sample_pairs_by_source(source, second, 2)
    assert first.read_bytes() == second.read_bytes()
