from __future__ import annotations

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.ports.embeddings import EmbedOptions


def test_hash_embeddings_are_deterministic_and_dimensioned() -> None:
    provider = HashEmbeddingProvider()
    options = EmbedOptions(model="hash", dimension=16)

    first = provider.embed(["Alice"], options)
    second = provider.embed(["Alice"], options)

    assert first == second
    assert len(first[0]) == 16
