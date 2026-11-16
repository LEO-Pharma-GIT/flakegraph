# SPDX-License-Identifier: Apache-2.0
"""Deterministic hash embeddings for smoke tests and fixture stability checks."""

from __future__ import annotations

import hashlib
import math

from kg_processor.ports.embeddings import EmbedOptions


class HashEmbeddingProvider:
    """Deterministic dimension-aware embeddings for tests and local dry-runs."""

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Return normalized deterministic vectors for each input text."""

        return [_embed_one(text, options.dimension) for text in texts]


def _embed_one(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}\x1f{text}".encode()).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    vector = values[:dimension]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
