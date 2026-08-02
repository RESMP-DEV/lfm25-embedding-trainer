from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REQUIRED_PAIR_FIELDS = ("query", "positive", "source", "source_id")


def _stable_id(value: object, *, label: str) -> str:
    """Normalize a stable identifier without turning null or empty IDs into data."""
    if value is None:
        raise ValueError(f"{label} is missing or null")
    normalized = str(value)
    if not normalized.strip():
        raise ValueError(f"{label} is empty")
    return normalized


def _query_identifier(row: dict[str, Any]) -> str:
    if "query_id" in row and row["query_id"] is not None:
        return _stable_id(row["query_id"], label="query_id")
    return _stable_id(row["query"], label="query")


def _write_jsonl_atomically(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".partial",
    )
    partial_path = Path(partial_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _reject_output_alias(output_path: Path, *input_paths: Path) -> None:
    output = output_path.resolve(strict=False)
    if any(output == path.resolve(strict=False) for path in input_paths):
        raise ValueError("output path must not alias an input path")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield row


def prepare_pairs(
    input_path: Path,
    output_path: Path,
    *,
    query_field: str = "query",
    positive_field: str = "positive",
    id_field: str = "id",
    source: str = "default",
    source_field: str | None = None,
    group_field: str | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in enumerate(read_jsonl(input_path), 1):
            try:
                query = row[query_field]
                positive = row[positive_field]
                source_id = row[id_field]
                source_name = row[source_field] if source_field else source
            except KeyError as exc:
                raise ValueError(
                    f"input row {line_number} is missing field {exc.args[0]!r}"
                ) from exc
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"input row {line_number} has an empty query")
            if not isinstance(positive, str) or not positive.strip():
                raise ValueError(f"input row {line_number} has an empty positive document")
            pair = {
                "query": query,
                "positive": positive,
                "source": str(source_name),
                "source_id": str(source_id),
            }
            if group_field is not None:
                if group_field not in row:
                    raise ValueError(f"input row {line_number} is missing field {group_field!r}")
                pair["group_id"] = str(row[group_field])
            output.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("input dataset is empty")
    return count


def link_retrieval_pairs(
    queries_path: Path,
    documents_path: Path,
    output_path: Path,
    *,
    query_field: str = "query",
    positive_ids_field: str = "positive_ids",
    document_id_field: str = "id",
    text_field: str = "text",
    group_field: str | None = "group_id",
    source: str = "default",
) -> int:
    """Join query relevance labels to a separate document corpus by stable ID."""
    _reject_output_alias(output_path, queries_path, documents_path)
    documents: dict[str, tuple[str, str]] = {}
    for line_number, row in enumerate(read_jsonl(documents_path), 1):
        try:
            raw_document_id = row[document_id_field]
            text = row[text_field]
        except KeyError as exc:
            raise ValueError(
                f"document row {line_number} is missing field {exc.args[0]!r}"
            ) from exc
        document_id = _stable_id(raw_document_id, label=f"document row {line_number} ID")
        if document_id in documents:
            raise ValueError(f"duplicate document ID {document_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"document row {line_number} has empty text")
        group_id = document_id
        if group_field is not None and row.get(group_field) is not None:
            group_id = _stable_id(row[group_field], label=f"document row {line_number} group ID")
        documents[document_id] = (text, group_id)
    if not documents:
        raise ValueError("document corpus is empty")

    pairs: list[dict[str, Any]] = []
    for line_number, row in enumerate(read_jsonl(queries_path), 1):
        try:
            query = row[query_field]
            positive_ids = row[positive_ids_field]
        except KeyError as exc:
            raise ValueError(f"query row {line_number} is missing field {exc.args[0]!r}") from exc
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query row {line_number} has an empty query")
        query_id = (
            _stable_id(row["id"], label=f"query row {line_number} ID")
            if "id" in row and row["id"] is not None
            else query
        )
        if not isinstance(positive_ids, list) or not positive_ids:
            raise ValueError(
                f"query row {line_number} field {positive_ids_field!r} must be a non-empty list"
            )
        seen_ids: set[str] = set()
        for raw_document_id in positive_ids:
            document_id = _stable_id(
                raw_document_id,
                label=f"query row {line_number} positive document ID",
            )
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            if document_id not in documents:
                raise ValueError(
                    f"query row {line_number} references unknown document ID {document_id!r}"
                )
            positive, group_id = documents[document_id]
            pairs.append(
                {
                    "query": query,
                    "query_id": query_id,
                    "positive": positive,
                    "source": source,
                    "source_id": document_id,
                    "group_id": group_id,
                }
            )
    if not pairs:
        raise ValueError("query dataset is empty")
    _write_jsonl_atomically(pairs, output_path)
    return len(pairs)


def validate_pairs(path: Path) -> dict[str, Any]:
    sources: Counter[str] = Counter()
    document_keys: set[tuple[str, str]] = set()
    groups: set[tuple[str, str]] = set()
    rows = 0
    for line_number, row in enumerate(read_jsonl(path), 1):
        missing = [field for field in REQUIRED_PAIR_FIELDS if field not in row]
        if missing:
            raise ValueError(f"pair row {line_number} is missing fields: {', '.join(missing)}")
        for field in ("query", "positive"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"pair row {line_number} has an empty {field}")
        source = str(row["source"])
        source_id = str(row["source_id"])
        group_id = str(row.get("group_id") or source_id)
        sources[source] += 1
        document_keys.add((source, source_id))
        groups.add((source, group_id))
        rows += 1
    if rows == 0:
        raise ValueError("pair dataset is empty")
    return {
        "pairs": rows,
        "unique_documents": len(document_keys),
        "unique_groups": len(groups),
        "sources": dict(sorted(sources.items())),
    }
