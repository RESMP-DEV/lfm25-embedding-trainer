from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .data import _query_identifier, _stable_id, read_jsonl

IdentityNode = tuple[str, str, str]


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[IdentityNode, IdentityNode] = {}

    def find(self, node: IdentityNode) -> IdentityNode:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, left: IdentityNode, right: IdentityNode) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def split_pairs(
    input_path: Path,
    output_directory: Path,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, int]:
    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1:
        raise ValueError("dev_ratio and test_ratio must be nonnegative and sum to less than 1")
    rows = list(read_jsonl(input_path))
    components = _DisjointSet()
    row_nodes: list[tuple[IdentityNode, IdentityNode]] = []
    for line_number, row in enumerate(rows, 1):
        source = _stable_id(row.get("source"), label=f"pair row {line_number} source")
        query_node = ("query", source, _query_identifier(row))
        raw_group_id = (
            row["group_id"]
            if "group_id" in row and row["group_id"] is not None
            else row.get("source_id")
        )
        group_node = (
            "document-group",
            source,
            _stable_id(raw_group_id, label=f"pair row {line_number} group ID"),
        )
        components.union(query_node, group_node)
        row_nodes.append((query_node, group_node))

    nodes_by_root: dict[IdentityNode, list[IdentityNode]] = defaultdict(list)
    for node in components.parent:
        nodes_by_root[components.find(node)].append(node)

    output_directory.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0, "test": 0}
    handles = {
        split: (output_directory / f"{split}.jsonl").open("w", encoding="utf-8") for split in counts
    }
    try:
        for row, (query_node, _) in zip(rows, row_nodes, strict=True):
            component = sorted(nodes_by_root[components.find(query_node)])
            key = json.dumps(component, ensure_ascii=False, separators=(",", ":")).encode()
            bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
            if bucket < test_ratio:
                split = "test"
            elif bucket < test_ratio + dev_ratio:
                split = "dev"
            else:
                split = "train"
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return counts


def sample_pairs_by_source(
    input_path: Path, output_path: Path, per_source: int = 250
) -> dict[str, int]:
    if per_source < 1:
        raise ValueError("per_source must be positive")
    candidates: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = hashlib.sha256(line.encode()).hexdigest()
            candidates[row["source"]].append((digest, line))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with output_path.open("w", encoding="utf-8") as output:
        for source_name in sorted(candidates):
            selected = sorted(candidates[source_name])[:per_source]
            counts[source_name] = len(selected)
            for _, line in selected:
                output.write(line if line.endswith("\n") else line + "\n")
    return counts
