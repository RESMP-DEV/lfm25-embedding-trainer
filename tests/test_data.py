import json
from pathlib import Path

import pytest

from lfm25_embedding_trainer.data import link_retrieval_pairs, prepare_pairs, validate_pairs


def test_prepare_maps_custom_fields_and_groups(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "How do I reset it?",
                "answer": "Hold the reset button for ten seconds.",
                "document": 7,
                "product": "router",
            }
        )
        + "\n"
    )
    output = tmp_path / "pairs.jsonl"
    assert (
        prepare_pairs(
            source,
            output,
            query_field="question",
            positive_field="answer",
            id_field="document",
            source="support",
            group_field="product",
        )
        == 1
    )
    row = json.loads(output.read_text())
    assert row == {
        "query": "How do I reset it?",
        "positive": "Hold the reset button for ten seconds.",
        "source": "support",
        "source_id": "7",
        "group_id": "router",
    }


def test_validate_reports_documents_sources_and_groups(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [
        {"query": "q1", "positive": "p", "source": "a", "source_id": "1"},
        {"query": "q2", "positive": "p", "source": "a", "source_id": "1"},
        {
            "query": "q3",
            "positive": "p2",
            "source": "b",
            "source_id": "2",
            "group_id": "g",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert validate_pairs(path) == {
        "pairs": 3,
        "unique_documents": 2,
        "unique_groups": 2,
        "sources": {"a": 2, "b": 1},
    }


def test_prepare_rejects_empty_text(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"id": 1, "query": "", "positive": "document"}\n')
    with pytest.raises(ValueError, match="empty query"):
        prepare_pairs(source, tmp_path / "pairs.jsonl")


def test_link_retrieval_pairs_joins_documents_and_preserves_groups(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(
        "\n".join(
            [
                json.dumps({"id": "a:1", "text": "First chunk", "group_id": "manual-a"}),
                json.dumps({"id": "a:2", "text": "Second chunk", "group_id": "manual-a"}),
            ]
        )
        + "\n"
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query": "How does A work?", "positive_ids": ["a:1", "a:2", "a:1"]}) + "\n"
    )
    output = tmp_path / "pairs.jsonl"

    assert link_retrieval_pairs(queries, documents, output, source="manuals") == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {
            "query": "How does A work?",
            "query_id": "How does A work?",
            "positive": "First chunk",
            "source": "manuals",
            "source_id": "a:1",
            "group_id": "manual-a",
        },
        {
            "query": "How does A work?",
            "query_id": "How does A work?",
            "positive": "Second chunk",
            "source": "manuals",
            "source_id": "a:2",
            "group_id": "manual-a",
        },
    ]


def test_link_retrieval_pairs_rejects_unknown_document(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"known","text":"Known document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query":"q","positive_ids":["missing"]}\n')

    with pytest.raises(ValueError, match="unknown document ID 'missing'"):
        link_retrieval_pairs(queries, documents, tmp_path / "pairs.jsonl")


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_link_retrieval_pairs_rejects_missing_or_empty_document_ids(
    tmp_path: Path, bad_id: object
) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text(json.dumps({"id": bad_id, "text": "document"}) + "\n")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query":"q","positive_ids":["document"]}\n')

    with pytest.raises(ValueError, match="document row 1 ID"):
        link_retrieval_pairs(queries, documents, tmp_path / "pairs.jsonl")


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_link_retrieval_pairs_rejects_missing_or_empty_positive_ids(
    tmp_path: Path, bad_id: object
) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"document","text":"document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query": "q", "positive_ids": [bad_id]}) + "\n")

    with pytest.raises(ValueError, match="positive document ID"):
        link_retrieval_pairs(queries, documents, tmp_path / "pairs.jsonl")


def test_link_retrieval_pairs_preserves_numeric_zero_query_id(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"document","text":"document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id":0,"query":"q","positive_ids":["document"]}\n')
    output = tmp_path / "pairs.jsonl"

    link_retrieval_pairs(queries, documents, output)

    assert json.loads(output.read_text())["query_id"] == "0"


def test_link_retrieval_pairs_treats_null_query_id_as_absent(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"document","text":"document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"id":null,"query":"q","positive_ids":["document"]}\n')
    output = tmp_path / "pairs.jsonl"

    link_retrieval_pairs(queries, documents, output)

    assert json.loads(output.read_text())["query_id"] == "q"


def test_link_retrieval_pairs_is_atomic_on_validation_error(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"known","text":"Known document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"query":"valid","positive_ids":["known"]}\n'
        '{"query":"invalid","positive_ids":["missing"]}\n'
    )
    output = tmp_path / "pairs.jsonl"
    output.write_text("existing output\n")

    with pytest.raises(ValueError, match="unknown document ID"):
        link_retrieval_pairs(queries, documents, output)

    assert output.read_text() == "existing output\n"
    assert not list(tmp_path.glob(".pairs.jsonl.*.partial"))


def test_link_retrieval_pairs_rejects_output_alias(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"known","text":"Known document"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query":"q","positive_ids":["known"]}\n')

    with pytest.raises(ValueError, match="must not alias"):
        link_retrieval_pairs(queries, documents, queries)

    assert queries.read_text() == '{"query":"q","positive_ids":["known"]}\n'


def test_link_retrieval_pairs_rejects_conflicting_query_id(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    documents.write_text('{"id":"one","text":"document one"}\n{"id":"two","text":"document two"}\n')
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"id":"shared","query":"first text","positive_ids":["one"]}\n'
        '{"id":"shared","query":"different text","positive_ids":["two"]}\n'
    )

    with pytest.raises(ValueError, match="reuses ID 'shared' with different query text"):
        link_retrieval_pairs(queries, documents, tmp_path / "pairs.jsonl")
