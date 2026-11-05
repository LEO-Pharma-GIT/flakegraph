# SPDX-License-Identifier: Apache-2.0
"""Job-queue models shared by local workers and Snowflake row leasing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobFileClaim(BaseModel):
    """A claimed source file row used by distributed file-queue workers."""

    job_id: str
    graph_id: str
    file_id: str
    source_uri: str | None = None
    checksum: str | None = None
    status: str = "CLAIMED"
    worker_id: str | None = None
    attempts: int = 0

    @field_validator("job_id", "graph_id", "file_id")
    @classmethod
    def required_ids_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ids before queue SQL can write ambiguous lease rows."""

        if not value.strip():
            raise ValueError("job file claim ids must not be blank")
        return value


class JobFileResult(BaseModel):
    """Summary of a processed file used to complete or fail queue rows."""

    file_id: str
    rows_written: int = Field(ge=0)
    row_counts: dict[str, int] = Field(default_factory=dict)
    stage: str = "done"
    ocr_provider: str | None = None
    llm_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    audit: dict[str, Any] = Field(default_factory=dict)
