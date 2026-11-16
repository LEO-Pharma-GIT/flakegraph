# SPDX-License-Identifier: Apache-2.0
"""Embedding text construction for graph retrieval artifacts.

The embedding text is intentionally bounded and derived from canonical graph
records, not raw provider output. That keeps vector generation stable across OCR
or LLM provider swaps once the graph rows are the same.
"""

from __future__ import annotations

from kg_processor.domain.graph import Chunk, Community, GraphEdge, GraphNode
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions


def embed_chunks(chunks: list[Chunk], provider: EmbeddingProvider, options: EmbedOptions) -> None:
    """Attach embeddings to chunk content in place."""

    vectors = provider.embed([chunk.content for chunk in chunks], options)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector


def embed_nodes(nodes: list[GraphNode], provider: EmbeddingProvider, options: EmbedOptions) -> None:
    """Attach embeddings to canonical node retrieval text in place."""

    texts = [
        (
            f"{node.name} ({node.primary_type})\n"
            f"Aliases: {', '.join(node.aliases)}\n"
            f"{node.description[:500]}"
        )
        for node in nodes
    ]
    vectors = provider.embed(texts, options)
    for node, vector in zip(nodes, vectors, strict=True):
        node.embedding = vector


def embed_edges(edges: list[GraphEdge], provider: EmbeddingProvider, options: EmbedOptions) -> None:
    """Attach embeddings to canonical edge retrieval text in place."""

    texts = [
        f"{edge.source_node_id} -[{edge.relation_type}]-> {edge.target_node_id}\n"
        f"{edge.description[:400]}"
        for edge in edges
    ]
    vectors = provider.embed(texts, options)
    for edge, vector in zip(edges, vectors, strict=True):
        edge.embedding = vector


def embed_communities(
    communities: list[Community],
    provider: EmbeddingProvider,
    options: EmbedOptions,
) -> None:
    """Attach embeddings to generated community report text in place."""

    vectors = provider.embed(
        [
            "\n".join(
                [
                    community.title,
                    community.summary,
                    community.rating_explanation,
                    *community.suggested_questions[:5],
                ]
            )
            for community in communities
        ],
        options,
    )
    for community, vector in zip(communities, vectors, strict=True):
        community.embedding = vector
