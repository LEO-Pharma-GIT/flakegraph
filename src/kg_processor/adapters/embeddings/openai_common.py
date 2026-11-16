# SPDX-License-Identifier: Apache-2.0
"""Shared response parsing for OpenAI-shaped embedding APIs."""

from __future__ import annotations


def parse_embedding_vectors(payload: object, expected_count: int) -> list[list[float]]:
    """Return embedding vectors from an OpenAI-compatible response payload."""

    if not isinstance(payload, dict):
        raise ValueError("Embedding response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response must include a data list")
    if len(data) != expected_count:
        raise ValueError(
            f"Embedding response returned {len(data)} vectors for {expected_count} inputs"
        )
    rows: list[tuple[int, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Embedding response data items must be objects")
        index = item.get("index")
        embedding = item.get("embedding")
        if not isinstance(index, int):
            raise ValueError("Embedding response data item is missing an integer index")
        if not isinstance(embedding, list) or not all(
            isinstance(value, int | float) for value in embedding
        ):
            raise ValueError("Embedding response data item must include a numeric embedding list")
        rows.append((index, [float(value) for value in embedding]))
    indexes = sorted(index for index, _vector in rows)
    if indexes != list(range(expected_count)):
        raise ValueError(
            f"Embedding response indexes must be a complete 0-based permutation; received {indexes}"
        )
    return [vector for _index, vector in sorted(rows, key=lambda row: row[0])]
