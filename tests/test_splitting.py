import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lfm25_embedding_trainer import splitting
from lfm25_embedding_trainer.splitting import _DisjointSet, sample_pairs_by_source, split_pairs


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


def test_split_keeps_query_document_group_component_together(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query": "shared query",
            "query_id": "shared",
            "positive": "section 1",
            "source": "journal",
            "source_id": "article-a:s1",
            "group_id": "article-a",
        },
        {
            "query": "shared query",
            "query_id": "shared",
            "positive": "section 2",
            "source": "journal",
            "source_id": "article-b:s1",
            "group_id": "article-b",
        },
        {
            "query": "related query",
            "query_id": "related",
            "positive": "section 3",
            "source": "journal",
            "source_id": "article-b:s2",
            "group_id": "article-b",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "splits"

    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)

    assert sorted(counts.values()) == [0, 0, 3]


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


def test_sample_pairs_keeps_complete_query_relevance_sets(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query": "multi",
            "query_id": "multi",
            "positive": f"document {index}",
            "source": "catalog",
            "source_id": str(index),
        }
        for index in range(3)
    ] + [
        {
            "query": "single",
            "query_id": "single",
            "positive": "other document",
            "source": "catalog",
            "source_id": "other",
        }
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "sample.jsonl"

    counts = sample_pairs_by_source(source, output, per_source=2)

    sampled = [json.loads(line) for line in output.read_text().splitlines()]
    multi_rows = [row for row in sampled if row["query_id"] == "multi"]
    assert len(multi_rows) in {0, 3}
    assert counts == {"catalog": len(sampled)}


def test_disjoint_set_finds_root_in_deep_chain_without_recursion() -> None:
    components = _DisjointSet()
    nodes = [(str(index), "chain", "node") for index in range(5000)]
    for left, right in zip(nodes[1:], nodes[:-1], strict=True):
        components.union(left, right)

    assert components.find(nodes[0]) == nodes[-1]
    assert all(components.find(node) == nodes[-1] for node in nodes)


@pytest.mark.parametrize("empty_group", ["", "   "])
def test_split_treats_empty_group_id_as_absent(tmp_path: Path, empty_group: str) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query": f"q{index}",
            "positive": "p",
            "source": "catalog",
            "source_id": str(index),
            "group_id": empty_group,
        }
        for index in range(20)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "splits"

    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)

    assert sum(counts.values()) == 20
    assert sum(1 for count in counts.values() if count) > 1


def test_split_preserves_numeric_zero_group_id(tmp_path: Path) -> None:
    source = tmp_path / "pairs.jsonl"
    rows = [
        {
            "query": f"q{index}",
            "positive": "p",
            "source": "catalog",
            "source_id": str(index),
            "group_id": 0,
        }
        for index in range(20)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "splits"

    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)

    assert [count for count in counts.values() if count] == [20]


def test_split_pairs_streams_rows_instead_of_materializing_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_count = 500
    rows = [
        {"query": f"q{index}", "positive": "p", "source": "catalog", "source_id": str(index)}
        for index in range(row_count)
    ]
    source = tmp_path / "pairs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    alive = 0
    peak_alive = 0

    class TrackedRow(dict[str, object]):
        def __init__(self, row: dict[str, object]) -> None:
            nonlocal alive, peak_alive
            super().__init__(row)
            alive += 1
            peak_alive = max(peak_alive, alive)

        def __del__(self) -> None:
            nonlocal alive
            alive -= 1

    real_read_jsonl = splitting.read_jsonl
    passes = 0

    def tracked_read_jsonl(path: Path):
        nonlocal passes
        passes += 1
        for row in real_read_jsonl(path):
            yield TrackedRow(row)

    monkeypatch.setattr(splitting, "read_jsonl", tracked_read_jsonl)

    counts = split_pairs(source, tmp_path / "splits", dev_ratio=0.3, test_ratio=0.3)

    assert sum(counts.values()) == row_count
    assert passes == 2
    assert peak_alive <= 2


def test_split_caches_one_split_key_per_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    component_count = 4
    rows_per_component = 10
    rows = [
        {
            "query": f"q{component}-{index}",
            "positive": "p",
            "source": "catalog",
            "source_id": str(component * rows_per_component + index),
            "group_id": f"group-{component}",
        }
        for component in range(component_count)
        for index in range(rows_per_component)
    ]
    source = tmp_path / "pairs.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    real_sha256 = splitting.hashlib.sha256
    calls = 0

    def counting_sha256(data: bytes):
        nonlocal calls
        calls += 1
        return real_sha256(data)

    monkeypatch.setattr(splitting, "hashlib", SimpleNamespace(sha256=counting_sha256))
    output = tmp_path / "splits"

    counts = split_pairs(source, output, dev_ratio=0.3, test_ratio=0.3)

    assert sum(counts.values()) == component_count * rows_per_component
    assert calls == component_count

    def expected_split(component: int) -> str:
        nodes = [
            ("query", "catalog", f"q{component}-{index}") for index in range(rows_per_component)
        ]
        nodes.append(("document-group", "catalog", f"group-{component}"))
        key = json.dumps(sorted(nodes), ensure_ascii=False, separators=(",", ":")).encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        if bucket < 0.3:
            return "test"
        if bucket < 0.6:
            return "dev"
        return "train"

    for component in range(component_count):
        contents = (output / f"{expected_split(component)}.jsonl").read_text()
        for index in range(rows_per_component):
            assert f'"{component * rows_per_component + index}"' in contents
