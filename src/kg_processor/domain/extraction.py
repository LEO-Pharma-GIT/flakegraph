# SPDX-License-Identifier: Apache-2.0
"""Intermediate extraction records kept separate from canonical graph rows.

The model first produces grounded mention observations.  Relations reference
those observations by local id, which makes dangling relation endpoints
impossible by construction.  Resolution later maps observations onto canonical
nodes while preserving every original mention and decision for auditability.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kg_processor.domain.graph import Chunk, Quantity, RelationAssertion


class ColumnRelation(BaseModel):
    """What one column of a catalogue dataset states about the thing its row names.

    One column names each row's subject. A relation column names a thing that
    ``relation_type`` links to the subject, with ``source`` saying which of the
    two is the relation's source; a value column is a value of the subject, or a
    quantity of the relation that ``of_column`` states; an alias column is a
    code of the subject itself.
    """

    column: str
    role: Literal["subject", "relation", "value", "alias"]
    entity_type: str | None = None
    relation_type: str | None = None
    source: Literal["subject", "column"] | None = None
    of_column: str | None = None


class ExtractionWindow(BaseModel):
    """Group adjacent chunks from one document into a bounded extraction task.

    Document identity is explicit so orchestration cannot accidentally infer facts
    between unrelated files batched for throughput.
    """

    id: str
    document_id: str
    chunks: list[Chunk]
    token_count: int
    # A catalogue dataset's column relations, read once from its header by
    # the dataset profile: a window past the first holds rows without it.
    dataset_columns: list[ColumnRelation] = Field(default_factory=list)


class EntityMention(BaseModel):
    """Represent one exact, typed entity mention accepted from source evidence.

    Mention identity remains separate from canonical graph nodes until conservative
    entity resolution has recorded how repeated surfaces should be grouped.
    """

    id: str
    name: str
    type: str
    description: str
    source_chunk_id: str
    quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    is_document_context: bool = False
    # The document's own subject (see EntityTypeDefinition.document_subject).
    is_document_subject: bool = False
    # llm: the name is in the model's quote; llm_located: the model pointed at
    # lines that name it; llm_confirmed: at lines that state it in other words;
    # ocr_tolerant: the quote or name matched only as OCR printed it - words run
    # together or a letter misread; folder: the folder the document is filed in,
    # on its location page, named by the ontology's folder entity.
    evidence_method: Literal["llm", "llm_located", "llm_confirmed", "ocr_tolerant", "folder"] = (
        "llm"
    )
    # The type extraction gave the mention when verification found it a thing
    # of another type and retyped it.
    proposed_type: str | None = None
    contextual_surfaces: list[str] = Field(default_factory=list)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class EntityExtractionOutcome(BaseModel):
    """Return grounded entities together with record-level provider diagnostics.

    Trace metadata explains accepted, rejected, repaired, and duplicate candidates
    without contaminating the persisted mention model.
    """

    entities: list[EntityMention] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class RelationObservation(BaseModel):
    """Represent one evidence-grounded predicate between accepted mention IDs.

    Endpoints cannot dangle because they reference first-pass mentions, and exact
    quote offsets preserve the assertion independently from canonical edge merging.
    """

    id: str
    source_entity_id: str
    target_entity_id: str
    source_surface: str | None = None
    target_surface: str | None = None
    relation_type: str
    description: str
    source_chunk_id: str
    quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    # llm: the model's quote or the sentence around it; llm_located: the model
    # pointed at the lines when string grounding could not place its quote;
    # llm_verified: the lines it pointed at do not name both ends, and the
    # verifier, shown the document's opening, found the relation stated;
    # ocr_tolerant: either, with a quote or endpoint matched only as OCR printed
    # it; ontology_cue: found by an ontology evidence cue without the model;
    # folder: an entity of a document linked to the document's folder entity,
    # on the location page's folder line.
    evidence_method: Literal[
        "llm", "llm_located", "llm_verified", "ocr_tolerant", "ontology_cue", "folder"
    ] = "llm"
    verification_required: bool = True
    quantities: list[Quantity] = Field(default_factory=list)
    # The type the model stated when the ontology's fallback relation replaced it.
    proposed_relation_type: str | None = None
    # Whether the text reports the relation as a fact or finding, or only
    # claims, proposes, or allows it ("useful for", "may be used").
    assertion: RelationAssertion = "reported"
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    # A held relation's rejected record, by ``rejected_records.rejected_record_key``;
    # finalization takes the record back when it admits the relation.
    rejected_record_key: str | None = None


class RelationExtractionOutcome(BaseModel):
    """Return accepted relation observations and record-level rejection diagnostics.

    A malformed sibling can be traced and dropped without losing valid relations
    from the same structured response. Held relations are grounded statements
    their window rejected only for their entity types; they are never kept here,
    only carried to finalization, which alone sees every type an entity has.
    """

    relations: list[RelationObservation] = Field(default_factory=list)
    held_relations: list[RelationObservation] = Field(default_factory=list)
    # Endpoints the response added for things the window's entities lacked.
    entities: list[EntityMention] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


# Why the verifier did not support a relation: its ends are only adjacent in a
# list, table, or layout; a citation names a source, not the subject; a table
# cell was read against the wrong row or column; the text does not state it;
# it runs the other way; an end is the wrong entity; composition wording was
# read as containment. "none" when supported, "omitted" when not answered.
VerificationReason = Literal[
    "none",
    "adjacency",
    "citation",
    "table_cell",
    "not_stated",
    "wrong_direction",
    "wrong_endpoint",
    "composition",
    "omitted",
]
# What the verifier found an entity to be: a specific named thing of its type or
# of another configured type, a thing no configured type covers, or a generic
# class term, a heading, a placeholder, or a value.
EntityVerdict = Literal[
    "specific", "wrong_type", "out_of_scope", "generic", "heading", "placeholder", "value"
]


class VerificationDecision(BaseModel):
    """Record an entailment verdict for one relation observation and exact evidence.

    Confidence remains available for filtering; the reason names which rule of
    evidence an unsupported relation failed.
    """

    relation_id: str
    verdict: Literal["supported", "contradicted", "insufficient"]
    reason: VerificationReason = "none"
    confidence: float = Field(ge=0.0, le=1.0)


class EntityVerificationDecision(BaseModel):
    """Record whether one entity mention names a specific thing of its type.

    Grounding proves a name is in the text, not that it names one thing: a
    class term, a heading, or a placeholder is grounded too.
    """

    entity_id: str
    verdict: EntityVerdict
    # The configured entity type the thing is: its own for most verdicts, the
    # other one for wrong_type.
    type: str = ""


class VerificationOutcome(BaseModel):
    """Return verifier decisions and provider metadata for extraction traces.

    The split keeps operational provenance outside the semantic decision records.
    """

    decisions: list[VerificationDecision] = Field(default_factory=list)
    entity_decisions: list[EntityVerificationDecision] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class ResolutionCandidate(BaseModel):
    """Describe a potentially equivalent mention pair considered for resolution.

    Lexical and embedding scores make candidate selection and adjudication auditable.
    """

    left_id: str
    right_id: str
    lexical_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class ResolutionDecision(BaseModel):
    """Record whether two mentions denote one entity and why that decision was made.

    Both merge and non-merge outcomes are retained so graph identity is reviewable.
    """

    left_id: str
    right_id: str
    same_entity: bool
    canonical_name: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str


class ExtractionObservations(BaseModel):
    """Persist window-parallel observations before graph-wide identity resolution.

    Extraction windows are independent by construction, so workers may produce
    these shards on different machines. Entity resolution remains a graph-wide
    barrier: combining shards before resolution preserves the same identity
    semantics as an in-process run over the complete corpus.
    """

    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationObservation] = Field(default_factory=list)
    held_relations: list[RelationObservation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
