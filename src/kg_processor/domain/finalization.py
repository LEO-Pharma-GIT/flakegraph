# SPDX-License-Identifier: Apache-2.0
"""Versioned manifests for graph datasets that remain partitioned at rest."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DatasetFile(BaseModel):
    """Describe one immutable file committed as part of a logical graph table."""

    uri: str
    size_bytes: int = Field(ge=0)


class DatasetTableManifest(BaseModel):
    """Describe one partitioned Parquet table without loading any of its rows."""

    name: str
    schema_version: int = Field(default=1, ge=1)
    row_count: int = Field(ge=0)
    files: list[DatasetFile] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Keep table identities suitable for manifests and destination mapping."""

        if not value.strip():
            raise ValueError("dataset table name must not be blank")
        return value


class GraphDatasetManifest(BaseModel):
    """Publish one complete immutable graph version as partitioned table metadata.

    The manifest is intentionally small enough for PostgreSQL and task outputs.
    Readers follow its file list directly; publication never reconstructs a
    ``GraphWriteBatch`` or copies all graph rows through the coordinator.
    """

    format: str = "flakegraph.graph-dataset-manifest/v1"
    run_id: str
    graph_id: str
    engine: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tables: dict[str, DatasetTableManifest]
    metrics: dict[str, Any] = Field(default_factory=dict)
    timings_seconds: dict[str, float] = Field(default_factory=dict)
    relation_weight_max: float = Field(default=10.0, gt=0.0)

    @field_validator("run_id", "graph_id", "engine")
    @classmethod
    def identity_must_not_be_blank(cls, value: str) -> str:
        """Reject manifests that cannot be tied to a graph version or engine."""

        if not value.strip():
            raise ValueError("graph dataset manifest identity must not be blank")
        return value

    def table(self, name: str) -> DatasetTableManifest:
        """Return a required logical table with a focused missing-table error."""

        try:
            return self.tables[name]
        except KeyError as exc:
            raise ValueError(f"graph dataset manifest is missing table {name!r}") from exc
