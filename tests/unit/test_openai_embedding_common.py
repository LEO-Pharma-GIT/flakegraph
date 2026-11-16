from __future__ import annotations

import pytest

from kg_processor.adapters.embeddings.openai_common import parse_embedding_vectors


def test_parse_embedding_vectors_orders_by_index_and_coerces_numbers() -> None:
    vectors = parse_embedding_vectors(
        {
            "data": [
                {"index": 1, "embedding": [2, 3.5]},
                {"index": 0, "embedding": [0.1, 1]},
            ]
        },
        expected_count=2,
    )

    assert vectors == [[0.1, 1.0], [2.0, 3.5]]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": [{"index": 0}]},
        {"data": [{"index": "0", "embedding": [0.1]}]},
        {"data": [{"index": 0, "embedding": ["bad"]}]},
    ],
)
def test_parse_embedding_vectors_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(ValueError, match="Embedding response"):
        parse_embedding_vectors(payload, expected_count=1)


def test_parse_embedding_vectors_rejects_duplicate_or_out_of_range_indexes() -> None:
    with pytest.raises(ValueError, match="complete 0-based permutation"):
        parse_embedding_vectors(
            {
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [2.0]},
                ]
            },
            expected_count=2,
        )
