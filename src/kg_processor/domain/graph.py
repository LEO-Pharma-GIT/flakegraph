# SPDX-License-Identifier: Apache-2.0
"""Canonical graph rows and extraction observations.

These models are deliberately close to the persisted local/Snowflake artifacts:
stable ids and provenance fields live here so provider adapters cannot invent a
different graph contract. The pipeline first works with observations, then
promotes grounded/merged records into canonical nodes, edges, evidence, and
community rows.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Chunk(BaseModel):
    """A stable text slice that becomes the unit of LLM graph extraction."""

    id: str
    graph_id: str = ""
    file_id: str
    document_id: str = ""
    page_number: int
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    token_count: int
    content_hash: str
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    ocr_generation_id: str | None = None
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def default_document_id(self) -> Self:
        """Use the file id as the document id for providers without a document id."""

        if not self.document_id:
            self.document_id = self.file_id
        return self


class ExtractedEntity(BaseModel):
    """One entity observation returned by an LLM for a specific chunk."""

    name: str
    type: str
    description: str
    source_chunk_id: str
    resolved_component_id: str | None = None
    quote: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    evidence_method: str = "llm"


# How a source states a relation: as a fact, finding, or observation
# (reported), or only as a claim, proposal, possible use, or hypothesis
# (claimed). A canonical edge is claimed only when every observation is.
RelationAssertion = Literal["reported", "claimed"]


class Quantity(BaseModel):
    """A value a relation states - a limit, a range, a result - as written.

    ``text`` is the exact source text; the bounds are parsed from it so values
    can be compared and filtered without re-reading the evidence.
    """

    text: str
    kind: str | None = None
    comparator: str | None = None
    unit: str | None = None
    low: float | None = None
    high: float | None = None


class ExtractedRelation(BaseModel):
    """One grounded relationship observation before canonical graph assembly.

    Resolved component IDs determine identity when present; observations without
    them use canonical endpoint names and types. The optional endpoint
    surfaces preserve the exact aliases, abbreviations, or document-context
    phrases that were verified in this observation's evidence.
    """

    source_name: str
    target_name: str
    source_component_id: str | None = None
    target_component_id: str | None = None
    # True only when the accepted source assertion referenced one mention twice.
    source_self_loop: bool = False
    source_surface: str | None = None
    target_surface: str | None = None
    source_type: str
    target_type: str
    relation_type: str
    description: str
    source_chunk_id: str
    quote: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    weight: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    quantities: list[Quantity] = Field(default_factory=list)
    proposed_relation_type: str | None = None
    evidence_method: str = "llm"
    assertion: RelationAssertion = "reported"
    # Set on a held relation; see ``RelationObservation.rejected_record_key``.
    rejected_record_key: str | None = None


class ExtractionResult(BaseModel):
    """Structured LLM extraction output before filtering and graph assembly.

    Held relations were rejected in their window only for their entity types;
    graph assembly admits those whose entities other documents type as the
    relation allows, and the rest stay rejected records.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    held_relations: list[ExtractedRelation] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Canonical merged entity node persisted to local artifacts or Snowflake.

    Aliases retain accepted source surfaces from every merged observation. They
    remain descriptive metadata rather than additional identities, allowing
    search, evaluation, and export consumers to recognize the canonical node
    without depending on one resolution-selected display spelling.
    """

    id: str
    graph_id: str
    normalized_name: str
    name: str
    primary_type: str
    types: list[str]
    aliases: list[str] = Field(default_factory=list)
    description: str
    embedding: list[float] | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    degree: int = 0
    rank: float | None = None


class GraphEdge(BaseModel):
    """Represent one canonical directed relationship between merged graph nodes.

    Repeated source assertions contribute weight, confidence, files, chunks, and
    evidence count without creating duplicate topology.
    """

    id: str
    graph_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    description: str
    weight: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_file_id: str
    source_file_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str]
    evidence_count: int = Field(default=0, ge=0)
    # Claimed only when every observation of the edge is claimed.
    assertion: RelationAssertion = "reported"
    embedding: list[float] | None = None


class EdgeObservation(BaseModel):
    """Preserve one file/chunk assertion contributing to a canonical edge.

    This provenance record lets reindexing remove one source independently while
    retaining support supplied by other documents.
    """

    id: str
    graph_id: str
    edge_id: str
    file_id: str
    chunk_id: str
    description: str
    weight: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_id: str
    quantities: list[Quantity] = Field(default_factory=list)
    # The type the model stated for this observation, when the ontology's
    # fallback relation kept it under the fallback's type instead.
    proposed_relation_type: str | None = None
    # Claimed when every assertion of the edge in this chunk only claims it.
    assertion: RelationAssertion = "reported"


class Evidence(BaseModel):
    """Grounded text span supporting a node, edge, or community finding."""

    id: str
    graph_id: str
    subject_id: str
    subject_kind: str
    file_id: str
    chunk_id: str
    page_number: int
    start_offset: int
    end_offset: int
    quote: str
    # How the evidence was found: llm - the model's quote or the sentence around
    # it; llm_located - the model pointed at the lines and they name the subject;
    # llm_confirmed - the model pointed at lines that state it in other words;
    # ocr_tolerant - the quote or a name matched only as OCR printed it, with
    # words run together or a letter misread;
    # llm_verified - lines that do not name both ends of a relation, which the
    # verifier, shown the document's opening, found to state it;
    # ontology_cue - an ontology evidence cue, without the model;
    # folder - the folder line of a document's location page, for the folder
    # entity the document is filed under and the links to it.
    method: str = "llm"


class EntitySource(BaseModel):
    """Per-file explanation and accepted surfaces supporting a merged entity.

    Surfaces include the batch display spelling so incremental reconciliation can
    rebuild aliases against an already-persisted node's display name.
    """

    id: str
    graph_id: str
    node_id: str
    file_id: str
    per_file_description: str
    mention_count: int
    aliases: list[str] = Field(default_factory=list)


class Community(BaseModel):
    """Represent a detected node group and its generated exploratory narrative.

    Stable membership identity, hierarchy links, structural rating, suggested
    questions, and embedding support inspection independently from model wording.
    """

    id: str
    graph_id: str
    stable_key: str
    level: int
    title: str
    summary: str
    rating: float
    rating_explanation: str = ""
    member_node_ids: list[str]
    parent_community_id: str | None = None
    child_community_ids: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class CommunityFinding(BaseModel):
    """A claim or takeaway associated with a generated community report."""

    id: str
    graph_id: str = ""
    community_id: str
    summary: str
    explanation: str


class DiscardedWindow(BaseModel):
    """A text window the model read and returned records for, all of which validation rejected.

    A window that yields nothing is ordinary - a title slide has no entities.
    A window where every returned record was rejected is different: the text
    was read, something was claimed about it, and none of it could be kept. For
    an entity window, what that text names is absent from the graph; for a
    relation window, its entities stand but no relation between them does. One
    pass over a window is one row, so a later pass over the same text is
    reported on its own. This row is that gap, named by document and pages so a
    reader can go and look, with the reasons counted so the pattern across a
    corpus is visible; every rejected record is listed in ``RejectedRecord``.
    """

    id: str
    graph_id: str = ""
    document_id: str
    file_id: str = ""
    # Which pass produced nothing: "entities" means the window contributed no
    # mentions at all; "relations" means its entities stand but no relation
    # between them survived.
    stage: Literal["entities", "relations"]
    window_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    extracted_records: int = 0
    # Why each record was rejected, by reason, e.g. {"ungrounded_quote": 3}.
    record_actions: dict[str, int] = Field(default_factory=dict)
    # The start of the window's text, so a reader can judge what was lost
    # without opening the source.
    preview: str = ""

    @field_validator("record_actions", mode="before")
    @classmethod
    def record_actions_from_any_table(cls, value: object) -> object:
        """Read the reasons however a table stored them.

        Spark writes the column as a Parquet map, which pyarrow hands back as
        a list of pairs; a JSON string is what a VARIANT or a pandas-written
        file may carry; the pipeline itself passes a dict.
        """

        if isinstance(value, str):
            value = json.loads(value) if value.strip() else {}
        if isinstance(value, list):
            value = dict(value)
        if isinstance(value, dict):
            return {str(key): int(count) for key, count in value.items() if count is not None}
        return value


class RejectedRecord(BaseModel):
    """An entity or relation the model stated that validation turned away.

    Every rejection is listed, wherever it happened - also in windows that kept
    other records - so what a graph lost, and why, can be read record by record.
    A duplicate of a kept record is not listed: the fact it states is in the
    graph. A relation the grounding fallback located is not listed either.
    """

    id: str
    graph_id: str = ""
    document_id: str
    file_id: str = ""
    window_id: str
    chunk_id: str = ""
    page_number: int | None = None
    kind: Literal["entity", "relation"]
    reason: str
    # The entity's name, or the relation's type.
    name: str = ""
    # The entity's type; empty for a relation.
    type: str = ""
    source: str = ""
    source_type: str = ""
    target: str = ""
    target_type: str = ""
    quote: str = ""


class FailedDocument(BaseModel):
    """A source document that could not be read, recorded instead of failing the run.

    Parsing is the one stage a document either passes whole or not at all: a
    file that is not what its name says, or that no parser can open, yields no
    text, so nothing from it is in the graph. One such file among thousands is
    no reason to lose the rest of a run, but it must not vanish either - this
    row names the file and says why, so a reader can see what the graph does
    not cover. A provider that rejects every document (bad credentials, say)
    still fails the run; that is not a property of any one document.
    """

    id: str
    graph_id: str = ""
    # The id the document would have had; documents are keyed by their file.
    document_id: str
    file_id: str
    source_uri: str = ""
    mime_type: str | None = None
    size_bytes: int | None = None
    # The parser that refused it.
    provider: str | None = None
    error_type: str = ""
    error_message: str = ""


class GraphWriteBatch(BaseModel):
    """Bundle every writer-ready artifact for one snapshot or incremental file batch.

    Local and Snowflake writers consume the same model, including graph rows,
    provenance, metrics, reports, trace events, and reindexing scope.
    """

    graph_id: str
    write_scope: Literal["graph_snapshot", "file_batch"] = "graph_snapshot"
    reindex_file_ids: list[str] = Field(default_factory=list)
    documents: list[dict[str, Any]]
    pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[Chunk]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    communities: list[Community]
    community_findings: list[CommunityFinding]
    discarded_windows: list[DiscardedWindow] = Field(default_factory=list)
    rejected_records: list[RejectedRecord] = Field(default_factory=list)
    failed_documents: list[FailedDocument] = Field(default_factory=list)
    run_report: dict[str, Any]
    extraction_trace: list[dict[str, Any]] = Field(default_factory=list)
    graph_metrics: dict[str, Any] = Field(default_factory=dict)
    edge_observations: list[EdgeObservation] = Field(default_factory=list)
    relation_weight_max: float = Field(default=10.0, gt=0.0)
