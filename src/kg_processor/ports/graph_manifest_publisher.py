# SPDX-License-Identifier: Apache-2.0
"""Port for publishing finalized graphs to configured durable destinations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch


class GraphManifestPublisher(Protocol):
    """Publish a finalized graph without exposing concrete writers to application code."""

    def publish(
        self,
        manifest: GraphDatasetManifest,
        task_payload: Mapping[str, object],
    ) -> None:
        """Persist a manifest according to its final task's non-secret output contract."""
        ...

    def stage_batch_publication(
        self, batch: GraphWriteBatch, task_payload: Mapping[str, object]
    ) -> AbstractContextManager[Callable[[], None]]:
        """Stage a local-engine batch independently of the claiming worker's engine."""
        ...
