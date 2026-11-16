# SPDX-License-Identifier: Apache-2.0
"""Writer port for persisting a complete graph write batch."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from kg_processor.domain.graph import GraphWriteBatch


class GraphWriter(Protocol):
    """Persists a complete graph batch to a concrete storage backend."""

    def write(self, batch: GraphWriteBatch) -> None:
        """Persist the supplied graph batch atomically for the chosen backend."""
        ...

    def stage_publication(
        self, batch: GraphWriteBatch, *, publication_id: str, generation: int
    ) -> AbstractContextManager[Callable[[], None]]:
        """Stage an immutable batch and yield its fenced destination commit.

        Preparation must not replace the visible graph. The caller invokes the
        commit only after validating the durable publication's ownership.
        """
        ...
