from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REQUIRED_PAIR_FIELDS = ("query", "positive", "source", "source_id")


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
    documents: dict[str, tuple[str, str]] = {}
    for line_number, row in enumerate(read_jsonl(documents_path), 1):
        try:
            document_id = str(row[document_id_field])
            text = row[text_field]
        except KeyError as exc:
            raise ValueError(
                f"document row {line_number} is missing field {exc.args[0]!r}"
            ) from exc
        if document_id in documents:
            raise ValueError(f"duplicate document ID {document_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"document row {line_number} has empty text")
        group_id = document_id
        if group_field is not None and row.get(group_field) is not None:
            group_id = str(row[group_field])
        documents[document_id] = (text, group_id)
    if not documents:
        raise ValueError("document corpus is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in enumerate(read_jsonl(queries_path), 1):
            try:
                query = row[query_field]
                positive_ids = row[positive_ids_field]
            except KeyError as exc:
                raise ValueError(
                    f"query row {line_number} is missing field {exc.args[0]!r}"
                ) from exc
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"query row {line_number} has an empty query")
            query_id = str(row.get("id") or query)
            if not isinstance(positive_ids, list) or not positive_ids:
                raise ValueError(
                    f"query row {line_number} field {positive_ids_field!r} "
                    "must be a non-empty list"
                )
            seen_ids: set[str] = set()
            for raw_document_id in positive_ids:
                document_id = str(raw_document_id)
                if document_id in seen_ids:
                    continue
                seen_ids.add(document_id)
                if document_id not in documents:
                    raise ValueError(
                        f"query row {line_number} references unknown document ID {document_id!r}"
                    )
                positive, group_id = documents[document_id]
                output.write(
                    json.dumps(
                        {
                            "query": query,
                            "query_id": query_id,
                            "positive": positive,
                            "source": source,
                            "source_id": document_id,
                            "group_id": group_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
    if count == 0:
        raise ValueError("query dataset is empty")
    return count


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
