# SPDX-License-Identifier: Apache-2.0
"""Immutable artifact-store port for distributed stage handoffs."""

from __future__ import annotations

from typing import Any, Protocol

from kg_processor.domain.distributed import ArtifactKind, ArtifactRef, StoredArtifact


class ArtifactStore(Protocol):
    """Persist content-addressed payloads independently from task orchestration."""

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        """Store bytes idempotently and return their immutable artifact reference.

        ``identity_key`` distinguishes equal bytes that represent separate source
        records with different provenance. Derived artifacts normally omit it and
        remain content-addressed.
        """
        ...

    def get(self, artifact_id: str) -> StoredArtifact:
        """Load and checksum-verify an artifact or raise when absent or corrupt."""
        ...

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Load artifacts in caller order or raise if any item is missing or corrupt."""
        ...

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Return matching artifacts in stable creation and identity order.

        Only artifacts a succeeded task handed to ``complete_task`` count: an
        attempt that wrote its output and then lost its lease leaves an orphan
        the retry does not overwrite.
        """
        ...

    def get_run_artifact_ids(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[str]:
        """Return the ids ``get_run_artifacts`` would load, for a reader elsewhere."""
        ...

    def list_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
        *,
        linked: bool = True,
    ) -> list[ArtifactRef]:
        """Return the references ``get_run_artifacts`` would load, without payloads.

        A caller choosing artifacts by their metadata - the documents a shard
        covers - reads the references and loads only what it chose. Stage
        outputs are read ``linked``, as ``get_run_artifacts`` reads them; the
        sources a planner staged were nobody's output, so they are read with
        ``linked=False``.
        """
        ...
