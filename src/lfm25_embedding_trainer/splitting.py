from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def split_pairs(
    input_path: Path,
    output_directory: Path,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, int]:
    if dev_ratio < 0 or test_ratio < 0 or dev_ratio + test_ratio >= 1:
        raise ValueError("dev_ratio and test_ratio must be nonnegative and sum to less than 1")
    output_directory.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "dev": 0, "test": 0}
    handles = {
        split: (output_directory / f"{split}.jsonl").open("w", encoding="utf-8") for split in counts
    }
    try:
        with input_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                group_id = row.get("group_id") or row["source_id"]
                key = f"{row['source']}:{group_id}".encode()
                bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
                if bucket < test_ratio:
                    split = "test"
                elif bucket < test_ratio + dev_ratio:
                    split = "dev"
                else:
                    split = "train"
                handles[split].write(line if line.endswith("\n") else line + "\n")
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
