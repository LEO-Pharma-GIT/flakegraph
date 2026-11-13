# SPDX-License-Identifier: Apache-2.0
"""Embedding provider port.

Embedding adapters must return exactly the configured dimension. The writers and
quality gates rely on that invariant before vectors are stored locally or cast to
Snowflake `VECTOR(FLOAT, n)`.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

# One batch of short texts through a hosted embedding model. Shared by every
# adapter so a stalled provider blocks a chunking worker for the same length of
# time whichever one is configured.
EMBEDDING_TIMEOUT_SECONDS = 120


class EmbedOptions(BaseModel):
    """Provider-neutral embedding model, dimension, and batching options."""

    model: str
    dimension: int
    batch_size: int = 96


class EmbeddingProvider(Protocol):
    """Computes embedding vectors for graph text in provider-specific adapters."""

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Return one vector per input text using the configured embedding model."""
        ...
