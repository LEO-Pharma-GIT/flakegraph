"""Contracts for tying every document to the entity its folder names.

The folder a file is filed in can say what the file belongs to - a study, a
contract. The ontology's folder entity names that entity by the folder path,
links every entity of the document to it at finalization, and leaves out the
documents it excludes without extracting anything again.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from documents import chunk, grounded, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application import pipeline as pipeline_module
from kg_processor.application.chunking import chunk_document
from kg_processor.application.community_detection import detect_communities
from kg_processor.application.document_normalization import with_location_page
from kg_processor.application.extraction_contracts import (
    relation_extraction_request,
    relation_verification_request,
)
from kg_processor.application.folder_entities import folder_entity_mentions
from kg_processor.application.graph_merge import (
    GraphAssemblyResult,
    assemble_graph_with_decisions,
    prune_isolated_entities,
)
from kg_processor.application.graph_quality import evaluate_graph_quality
from kg_processor.application.llm_extractors import _relation_definition
from kg_processor.application.ontology import load_ontology
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.two_pass_extraction import (
    _relation_inventory_for_window,
    resolve_extraction_observations,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.documents import ParsedDocument, ParsedPage
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    ExtractionWindow,
    RelationObservation,
)
from kg_processor.domain.graph import (
    Chunk,
    Evidence,
    ExtractedEntity,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.ontology import FolderEntityDefinition, OntologyProfile
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard
from kg_processor.factories import build_pipeline
from kg_processor.ports.llm import (
    LlmCapabilities,
    RefusedChatResponseError,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

_PROFILE: dict[str, Any] = {
    "name": "studies",
    "description": "Studies and what their documents state.",
    "mode": "closed",
    "entity_types": [
        {"name": "STUDY", "description": "A research study a folder of documents belongs to."},
        {"name": "PRODUCT", "description": "A product a document describes."},
        {"name": "ORGANIZATION", "description": "A company or institution."},
    ],
    "relation_types": [
        {
            "name": "MADE_BY",
            "description": "A product is made by an organization.",
            "source_types": ["PRODUCT"],
            "target_types": ["ORGANIZATION"],
        }
    ],
}


def _profile(**folder: Any) -> dict[str, Any]:
    return {**_PROFILE, "folder_entity": {"entity_type": "STUDY", **folder}}


def test_a_folder_entity_adds_its_structural_relation_to_the_profile() -> None:
    profile = OntologyProfile.model_validate(_profile())

    relation = profile.relation("APPEARS_IN")
    assert relation is not None
    assert relation.structural
    assert relation.source_types == []
    assert relation.target_types == ["STUDY"]
    # Validating the stored profile again keeps the one relation it now names.
    again = OntologyProfile.model_validate(profile.model_dump(mode="json"))
    assert [item.name for item in again.relation_types] == ["MADE_BY", "APPEARS_IN"]


def test_a_folder_entity_names_an_entity_type_of_the_ontology() -> None:
    with pytest.raises(ValueError, match="folder entity type CONTRACT"):
        OntologyProfile.model_validate(_profile(entity_type="CONTRACT"))


def test_a_folder_relation_extraction_reads_from_text_is_refused() -> None:
    with pytest.raises(ValueError, match="extraction reads from text"):
        OntologyProfile.model_validate(_profile(relation_type="made by"))


def test_a_defined_folder_relation_must_link_any_entity_to_the_folder_type() -> None:
    narrow = {
        **_profile(relation_type="FILED_UNDER"),
        "relation_types": [
            *_PROFILE["relation_types"],
            {
                "name": "FILED_UNDER",
                "description": "Filed under a study.",
                "source_types": ["PRODUCT"],
                "target_types": ["STUDY"],
                "structural": True,
            },
        ],
    }
    with pytest.raises(ValueError, match="must link any entity type to STUDY"):
        OntologyProfile.model_validate(narrow)

    broad = {
        **narrow,
        "relation_types": [
            *_PROFILE["relation_types"],
            {"name": "FILED_UNDER", "description": "Filed under a study.", "structural": True},
        ],
    }
    assert OntologyProfile.model_validate(broad).relation("filed_under") is not None


def test_no_rule_keeps_a_stated_relation_under_a_structural_one() -> None:
    with pytest.raises(ValueError, match="structural relations cannot be"):
        OntologyProfile.model_validate({**_profile(), "fallback_relation": "APPEARS_IN"})


def test_folder_level_and_exclusions_read_paths_below_the_source_root() -> None:
    definition = FolderEntityDefinition(
        entity_type="STUDY",
        level=2,
        excluded_file_ids=["file_a"],
        excluded_paths=["Studies/Northwind Pricing Study/notes.txt"],
    )

    assert definition.folder_at_level("Studies/Northwind Pricing Study/raw") == (
        "Studies/Northwind Pricing Study"
    )
    assert definition.folder_at_level("Studies") is None
    assert definition.folder_at_level(None) is None
    assert definition.excludes("file_a", "anything.pdf")
    assert definition.excludes("file_b", "/Studies/Northwind Pricing Study/notes.txt")
    assert not definition.excludes("file_b", "Studies/Northwind Pricing Study/report.pdf")
    assert definition.excluded_path_folders() == {"Studies/Northwind Pricing Study"}
    with pytest.raises(ValueError):
        FolderEntityDefinition(entity_type="STUDY", level=0)


def test_extraction_reads_the_profile_without_structural_relations_or_exclusions() -> None:
    with_exclusions = load_ontology(
        None, [], None, inline=_profile(excluded_file_ids=["file_a"], excluded_paths=["a/b.pdf"])
    )
    without = load_ontology(None, [], None, inline=_profile())

    assert with_exclusions.extraction.relation("APPEARS_IN") is None
    assert with_exclusions.profile.relation("APPEARS_IN") is not None
    folder = with_exclusions.extraction.folder_entity
    assert folder is not None
    assert folder.excluded_file_ids == []
    assert folder.excluded_paths == []
    # Changing which documents a folder leaves out reuses every extraction.
    assert with_exclusions.extraction_checksum == without.extraction_checksum
    assert with_exclusions.checksum != without.checksum
    # A profile without the feature extracts under its own checksum.
    plain = load_ontology(None, [], None, inline=_PROFILE)
    assert plain.extraction_checksum == plain.checksum


def test_a_folder_entity_without_the_location_page_fails_at_config_load() -> None:
    with pytest.raises(ValueError, match="files.location_page must be on"):
        Settings.model_validate(
            {"ontology": {"profile": _profile()}, "files": {"location_page": False}}
        )

    assert Settings.model_validate({"ontology": {"profile": _profile()}}).files.location_page


class _NamingLlm:
    """Answer folder requests with one name per folder, or fail as told."""

    def __init__(
        self, names: dict[str, str] | None = None, failure: Exception | None = None
    ) -> None:
        self.names = names or {}
        self.failure = failure
        self.requests: list[StructuredCompletionRequest] = []

    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(supports_seed=True, max_output_tokens=1024)

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        name = self.names.get(payload["file_name"], "")
        return StructuredCompletionResult(
            payload={"name": name, "description": f"A pricing study of {name or 'nothing'}."}
        )


def _document(
    file_id: str, location: str, body: str, *, chunk_tokens: int = 200
) -> tuple[dict[str, Any], list[Chunk]]:
    """A document row and its chunks, location page first, as preparation builds them."""

    parsed = ParsedDocument(
        file_id=file_id,
        checksum=f"checksum-{file_id}",
        source_uri=f"/uploads/run/{location}",
        mime_type="text/plain",
        pages=[ParsedPage(page_number=1, markdown=body, raw_text=body)],
    )
    row = {
        "file_id": file_id,
        "source_uri": parsed.source_uri,
        "folder": location.rpartition("/")[0],
    }
    return row, chunk_document(with_location_page(parsed, location), "graph", chunk_tokens, 0)


def _pipeline(tmp_path: Path, profile: dict[str, Any], **graph: Any) -> KgProcessorPipeline:
    return build_pipeline(
        Settings.load(
            env={},
            overrides={
                "job": {"graph_id": "graph"},
                "ocr": {"provider": "builtin_text"},
                "llm": {"provider": "fake"},
                "embedding": {"provider": "hash", "dimension": 8},
                "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
                "cache": {"provider": "none"},
                "ontology": {"profile": profile},
                "graph": graph,
            },
        )
    )


def _mentions(
    llm: _NamingLlm, *documents: tuple[dict[str, Any], list[Chunk]], **folder: Any
) -> tuple[list[EntityMention], list[dict[str, Any]]]:
    return folder_entity_mentions(
        [chunk for _row, chunks in documents for chunk in chunks],
        {str(row["file_id"]): row for row, _chunks in documents},
        OntologyProfile.model_validate(_profile(**folder)),
        llm,
        model="test",
        timeout_seconds=5,
        seed=7,
        parallelism=2,
    )


def test_a_folder_mention_is_named_by_its_folder_and_grounded_on_the_folder_line() -> None:
    document = _document(
        "file_a",
        "Picked Root/northwind_pricing/2024/report.pdf",
        "Northwind compared shelf prices in four markets.",
    )
    llm = _NamingLlm({"report.pdf": "Northwind Pricing Study"})

    mentions, trace = _mentions(llm, document, level=2)

    [mention] = mentions
    location_chunk = document[1][0]
    assert mention.name == "northwind_pricing"
    assert mention.type == "STUDY"
    assert mention.is_document_context
    assert mention.evidence_method == "folder"
    assert mention.source_chunk_id == location_chunk.id
    assert mention.start_offset is not None and mention.end_offset is not None
    assert location_chunk.content[mention.start_offset : mention.end_offset] == mention.quote
    # The folder's own name at the level, read where the folder line writes it.
    assert location_chunk.content[: mention.start_offset].endswith("\nFolder: Picked Root/")
    assert mention.quote == "northwind_pricing"
    assert mention.description == (
        "A pricing study of Northwind Pricing Study. "
        "Its documents are filed in the folder Picked Root/northwind_pricing."
    )
    assert trace[0]["status"] == "named"
    assert trace[0]["name"] == "Northwind Pricing Study"
    # The call reads the folder, the file name, and the document's opening.
    [request] = llm.requests
    payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
    assert payload["folder"] == "Picked Root/northwind_pricing"
    assert payload["file_name"] == "report.pdf"
    assert payload["entity_type"]["name"] == "STUDY"
    assert "four markets" in payload["opening"][0]["content"]


@pytest.mark.parametrize(
    ("llm", "reason"),
    [
        (_NamingLlm(failure=RefusedChatResponseError("declined")), "refused"),
        (_NamingLlm(failure=TimeoutError("slow")), "error"),
        (_NamingLlm({}), "no_name"),
    ],
)
def test_a_folder_the_model_does_not_name_is_named_by_its_path(
    llm: _NamingLlm, reason: str
) -> None:
    mentions, trace = _mentions(
        llm, _document("file_a", "Contract ACME-0042/terms.pdf", "Payment terms apply.")
    )

    [mention] = mentions
    assert mention.name == "Contract ACME-0042"
    assert "Contract ACME-0042" in mention.description
    assert trace[0]["status"] == "folder_name"
    assert trace[0]["reason"] == reason


def test_a_file_above_the_folder_level_is_tied_to_no_folder() -> None:
    mentions, trace = _mentions(
        _NamingLlm(),
        _document("file_root", "readme.txt", "Read me first."),
        _document("file_top", "Studies/overview.pdf", "All studies."),
        _document("file_deep", "Studies/Northwind Pricing Study/report.pdf", "Prices."),
        level=2,
    )

    assert [mention.name for mention in mentions] == ["Northwind Pricing Study"]
    assert "Studies/Northwind Pricing Study" in mentions[0].description
    assert [event.get("reason") for event in trace[:2]] == [
        "no_folder_at_level",
        "no_folder_at_level",
    ]


def test_the_context_pass_adds_folder_mentions_without_document_context_types(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path, _profile())
    document = _document("file_a", "Northwind Pricing Study/report.pdf", "Prices rose.")

    entities, trace = pipeline._extract_document_context_entities(
        document[1], {"file_a": document[0]}
    )

    assert [(item.name, item.evidence_method) for item in entities] == [
        ("Northwind Pricing Study", "folder")
    ]
    assert any(event["stage"] == "folder_entity" for event in trace)


def test_relation_extraction_is_never_offered_a_folder_entity() -> None:
    document = _document("file_a", "Northwind Pricing Study/report.pdf", "Prices rose.")
    [folder], _trace = _mentions(_NamingLlm(), document)
    body = document[1][1]
    product = EntityMention(
        id="mention_product",
        name="Prices",
        type="PRODUCT",
        description="Prices.",
        source_chunk_id=body.id,
        **grounded(body.content, "Prices"),
    )

    inventory = _relation_inventory_for_window([folder, product], window(*document[1]))

    assert [item.id for item in inventory] == ["mention_product"]


# Read at level 2: below a picked root, and below an archive that holds a
# folder of the same name, which is the same study.
_CORPUS = (
    (
        "file_a",
        "Picked Root/Northwind Pricing Study/report.pdf",
        "Widget Pro is made by Acme Corp.",
    ),
    ("file_b", "Archive/Northwind Pricing Study/raw/prices.txt", "Widget Pro prices rose in May."),
    ("file_c", "Picked Root/Contract ACME-0042/terms.pdf", "Acme Corp supplies Gadget Max."),
    ("file_d", "Picked Root/notes.txt", "Gadget Max is also sold online."),
)
_LEVEL = 2
# The model names one folder two ways; the folder's name is still one entity.
_NAMES = {
    "report.pdf": "Northwind Pricing Study",
    "prices.txt": "NW Pricing",
    "terms.pdf": "Contract ACME-0042",
}


def _shards() -> list[ExtractedDocumentShard]:
    """Four documents in two folders and the source root, extracted and context-named."""

    shards: list[ExtractedDocumentShard] = []
    for file_id, location, body in _CORPUS:
        row, chunks = _document(file_id, location, body)
        row = {**row, "id": file_id, "graph_id": "graph", "mime_type": "text/plain"}
        folders, trace = _mentions(_NamingLlm(_NAMES), (row, chunks), level=_LEVEL)
        text = chunks[1]
        names = [
            ("Widget Pro", "PRODUCT"),
            ("Gadget Max", "PRODUCT"),
            ("Acme Corp", "ORGANIZATION"),
        ]
        mentions = [
            EntityMention(
                id=f"{file_id}-{name}",
                name=name,
                type=entity_type,
                description=f"{name} as {file_id} states it.",
                source_chunk_id=text.id,
                **grounded(text.content, name),
            )
            for name, entity_type in names
            if name in text.content
        ]
        relations = [
            RelationObservation(
                id=f"{file_id}-made-by",
                source_entity_id=f"{file_id}-Widget Pro",
                target_entity_id=f"{file_id}-Acme Corp",
                relation_type="MADE_BY",
                description="Widget Pro is made by Acme Corp.",
                source_chunk_id=text.id,
                **grounded(text.content, text.content),
            )
            for _ in [0]
            if "made by" in text.content
        ]
        prepared = PreparedDocumentShard(
            file_ids=[file_id],
            files_seen=1,
            documents_processed=1,
            document_rows=[row],
            chunks=chunks,
            document_context_entities=folders,
            trace=trace,
        )
        shards.append(
            ExtractedDocumentShard(
                prepared=prepared,
                observations=ExtractionObservations(
                    entities=[*folders, *mentions],
                    relations=relations,
                    chunk_count=len(chunks),
                    window_count=1,
                ),
            )
        )
    return shards


def _finalize(tmp_path: Path, **folder: Any) -> GraphWriteBatch:
    return _pipeline(
        tmp_path, _profile(level=_LEVEL, **folder), entity_resolution_enabled=False
    ).finalize_document_shards(_shards(), write=False)


def _links(batch: GraphWriteBatch) -> set[tuple[str, str]]:
    nodes = {node.id: node.name for node in batch.nodes}
    return {
        (nodes[edge.source_node_id], nodes[edge.target_node_id])
        for edge in batch.edges
        if edge.relation_type == "appears_in"
    }


def test_every_entity_of_a_document_is_linked_to_its_folder_entity(tmp_path: Path) -> None:
    batch = _finalize(tmp_path)

    studies = [node for node in batch.nodes if node.primary_type == "STUDY"]
    assert sorted(node.name for node in studies) == [
        "Contract ACME-0042",
        "Northwind Pricing Study",
    ]
    assert _links(batch) == {
        ("Widget Pro", "Northwind Pricing Study"),
        ("Acme Corp", "Northwind Pricing Study"),
        ("Acme Corp", "Contract ACME-0042"),
        ("Gadget Max", "Contract ACME-0042"),
    }
    # The two folders of that name, under different parents, are one study,
    # grounded on each document's folder line.
    [northwind] = [node for node in studies if node.name == "Northwind Pricing Study"]
    assert len(northwind.source_chunk_ids) == 2
    assert "Picked Root/Northwind Pricing Study" in northwind.description
    by_name = {node.name: node.id for node in batch.nodes}
    widget = next(
        edge
        for edge in batch.edges
        if edge.source_node_id == by_name["Widget Pro"] and edge.relation_type == "appears_in"
    )
    # Found in both of the folder's documents: one observation for each.
    assert widget.evidence_count == 2
    assert widget.weight == 2.0
    evidence = [item for item in batch.evidence if item.subject_id == widget.id]
    assert {item.method for item in evidence} == {"folder"}
    assert {item.quote for item in evidence} == {"Northwind Pricing Study"}
    # Folder links and the folder entity pass every gate, spans included.
    quality = next(event for event in batch.extraction_trace if event["stage"] == "quality")
    assert quality["ok"]
    assert not [check for check in quality["failed_checks"] if check["severity"] == "error"]
    chunks = {chunk.id: chunk for chunk in batch.chunks}
    for item in batch.evidence:
        chunk = chunks[item.chunk_id]
        start, end = item.start_offset - chunk.start_offset, item.end_offset - chunk.start_offset
        assert chunk.content[start:end] == item.quote


def test_an_excluded_document_keeps_its_entities_and_loses_its_folder(tmp_path: Path) -> None:
    batch = _finalize(tmp_path, excluded_file_ids=["file_c"])

    assert _links(batch) == {
        ("Widget Pro", "Northwind Pricing Study"),
        ("Acme Corp", "Northwind Pricing Study"),
    }
    # A folder whose every document is excluded is no entity at all.
    assert "Contract ACME-0042" not in {node.name for node in batch.nodes}
    assert "Gadget Max" in {node.name for node in batch.nodes}


def test_a_document_is_excluded_by_its_path_below_the_source_root(tmp_path: Path) -> None:
    batch = _finalize(tmp_path, excluded_paths=["Picked Root/Northwind Pricing Study/report.pdf"])

    assert _links(batch) == {
        ("Widget Pro", "Northwind Pricing Study"),
        ("Acme Corp", "Contract ACME-0042"),
        ("Gadget Max", "Contract ACME-0042"),
    }
    widget = next(edge for edge in batch.edges if edge.relation_type == "appears_in")
    assert widget.evidence_count == 1


def _node(node_id: str, chunk_id: str, primary_type: str = "PRODUCT") -> GraphNode:
    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=node_id,
        name=node_id,
        primary_type=primary_type,
        types=[primary_type],
        description=node_id,
        source_chunk_ids=[chunk_id],
    )


def _edge(source: str, target: str, relation_type: str, weight: float = 1.0) -> GraphEdge:
    return GraphEdge(
        id=f"{source}-{relation_type}-{target}",
        graph_id="graph",
        source_node_id=source,
        target_node_id=target,
        relation_type=relation_type,
        description="",
        weight=weight,
        source_file_id="file",
        source_chunk_ids=["chunk"],
    )


def _hub() -> tuple[list[GraphNode], list[GraphEdge]]:
    """Two pairs of related entities, every one of them filed under one folder."""

    nodes = [
        *(_node(name, f"chunk-{name}") for name in ("a1", "a2", "b1", "b2")),
        _node("folder", "chunk-location", "STUDY"),
    ]
    edges = [
        _edge("a1", "a2", "made_by"),
        _edge("b1", "b2", "made_by"),
        *(_edge(name, "folder", "appears_in", weight=10.0) for name in ("a1", "a2", "b1", "b2")),
    ]
    return nodes, edges


def test_communities_follow_topics_not_the_folder_every_entity_is_filed_under() -> None:
    nodes, edges = _hub()
    folder_evidence = Evidence(
        id="evidence-folder",
        graph_id="graph",
        subject_id="folder",
        subject_kind="node",
        file_id="file",
        chunk_id="chunk-location",
        page_number=0,
        start_offset=0,
        end_offset=6,
        quote="folder",
        method="folder",
    )

    result = detect_communities(
        "graph",
        nodes,
        edges,
        FakeLlmProvider(),
        min_community_size=2,
        evidence=[folder_evidence],
        structural_relation_types=frozenset({"appears_in"}),
    )

    members = sorted(sorted(item.member_node_ids) for item in result.communities)
    assert members == [["a1", "a2"], ["b1", "b2"]]


def test_graph_shape_checks_read_no_structural_edge() -> None:
    nodes, edges = _hub()
    lone = _node("lone", "chunk-lone")
    ontology = OntologyProfile.model_validate(_profile())
    rows = [*nodes, lone]

    def isolates(with_ontology: OntologyProfile | None) -> int:
        result = evaluate_graph_quality(
            rows,
            [*edges, _edge("lone", "folder", "appears_in")],
            [],
            None,
            ontology=with_ontology,
        )
        return next(check.count for check in result.checks if check.name == "isolate_ratio")

    # Only the folder's links reach the lone entity and the folder itself.
    assert isolates(ontology) == 2
    assert isolates(None) == 0


def test_dropping_isolated_entities_reads_structural_edges_from_entity_to_folder() -> None:
    nodes, edges = _hub()
    lone = _node("lone", "chunk-lone")
    lone_link = _edge("lone", "folder", "appears_in")
    assembly = GraphAssemblyResult(
        nodes=[*nodes, lone],
        edges=[*edges, lone_link],
        edge_observations=[],
        evidence=[],
        entity_sources=[],
        decisions=[],
    )

    pruned = prune_isolated_entities(assembly, frozenset({"appears_in"}))

    assert sorted(node.id for node in pruned.nodes) == ["a1", "a2", "b1", "b2", "folder"]
    assert lone_link.id not in {edge.id for edge in pruned.edges}
    assert {node.id: node.degree for node in pruned.nodes}["folder"] == 4


def test_new_exclusions_finalize_the_stored_extraction_without_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A revision with other exclusions reads the same shards: no extraction call is made."""

    asked: list[str] = []
    answer = FakeLlmProvider.complete_structured

    def recording(self: FakeLlmProvider, request: StructuredCompletionRequest) -> Any:
        asked.append(request.task_name)
        return answer(self, request)

    monkeypatch.setattr(FakeLlmProvider, "complete_structured", recording)
    shards = _shards()
    before = _pipeline(tmp_path, _profile(level=_LEVEL), entity_resolution_enabled=False)
    after = _pipeline(
        tmp_path,
        _profile(level=_LEVEL, excluded_file_ids=["file_a", "file_b"]),
        entity_resolution_enabled=False,
    )

    first = before.finalize_document_shards(shards, write=False)
    second = after.finalize_document_shards(shards, write=False)

    assert "Northwind Pricing Study" in {node.name for node in first.nodes}
    assert "Northwind Pricing Study" not in {node.name for node in second.nodes}
    assert not {"folder_entity", "entity_extraction", "relation_extraction"} & set(asked)
    # Extraction reads one profile either way, so a cached extraction is reused.
    assert before._ontology().extraction_checksum == after._ontology().extraction_checksum


@pytest.mark.parametrize(
    ("mode", "fallback", "kept_as"),
    [
        ("closed", False, None),
        ("hybrid", False, "RELATED_TO"),
        ("open", False, "RELATED_TO"),
        ("hybrid", True, "ASSOCIATED_WITH"),
    ],
)
def test_extraction_keeps_a_stated_relation_off_the_structural_label(
    mode: str, fallback: bool, kept_as: str | None
) -> None:
    """The folder relation's label is outside a statement's vocabulary, in every mode."""

    profile = {**_profile(), "mode": mode}
    if fallback:
        profile["relation_types"] = [
            *_PROFILE["relation_types"],
            {"name": "ASSOCIATED_WITH", "description": "Any stated connection."},
        ]
        profile["fallback_relation"] = "ASSOCIATED_WITH"
    extraction = load_ontology(None, [], None, inline=profile).extraction
    reasons: Counter[str] = Counter()

    definition, stated = _relation_definition("appears-in", extraction, reasons)

    if kept_as is None:
        assert definition is None
        assert reasons == {"invalid_relation_type": 1}
        return
    assert definition is not None
    assert definition.name == kept_as
    assert not definition.structural
    assert stated == "APPEARS_IN"
    assert reasons == {"reserved_relation_type": 1}


def test_a_profile_without_a_folder_entity_keeps_folder_wording_out_of_its_prompts() -> None:
    """Runs that do not tie documents to folders fingerprint and prompt without them.

    The values pin the profile and its relation prompts with no folder entity:
    folder wording or structural relations that leaked into them would move these.
    """

    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    assert ontology.checksum == "9b28487e5dcac0f85d41dd0f6bd047cd0faca63266c2447a8643a7b82faeee09"
    assert ontology.extraction_checksum == ontology.checksum
    text = "Kano Jigoro founded the Kodokan in Tokyo."
    source = chunk(text, chunk_id="chunk-1", file_id="file-1", token_count=7, content_hash="h")
    request_window = ExtractionWindow(
        id="window-1", document_id="file-1", chunks=[source], token_count=7
    )
    entities = [
        EntityMention(
            id="m-kano",
            name="Kano Jigoro",
            type="PERSON",
            description="Founder.",
            source_chunk_id="chunk-1",
            quote="Kano Jigoro",
            start_offset=0,
            end_offset=11,
        ),
        EntityMention(
            id="m-kodokan",
            name="Kodokan",
            type="SCHOOL",
            description="A school.",
            source_chunk_id="chunk-1",
            quote="Kodokan",
            start_offset=24,
            end_offset=31,
        ),
    ]
    relation = RelationObservation(
        id="r-1",
        source_entity_id="m-kodokan",
        target_entity_id="m-kano",
        relation_type="FOUNDED_BY",
        description="Founded.",
        source_chunk_id="chunk-1",
        quote=text,
        start_offset=0,
        end_offset=len(text),
    )
    extraction = relation_extraction_request(
        request_window,
        entities,
        ontology.extraction,
        model="m",
        timeout_seconds=30,
        max_relations=10,
        seed=7,
        previous_relations=None,
    )
    verification = relation_verification_request(
        request_window,
        entities,
        [relation],
        ontology.extraction,
        model="m",
        timeout_seconds=30,
        seed=7,
    )

    def fingerprint(request: StructuredCompletionRequest) -> str:
        return sha256_hex(f"{request.system}\n{request.user}")

    assert "structural" not in extraction.user
    assert fingerprint(extraction) == (
        "d316bf62bc12ace9e30e3428bab4f9d869a8738f6ca2be78515a57d2fe8cb1a0"
    )
    assert fingerprint(verification) == (
        "de7be9a858d8274092cc0a2e8b919f263e0df60a5239c14c9f3456130591a9f9"
    )


def test_a_folder_line_spread_over_chunks_still_grounds_the_folder_name() -> None:
    """The name must lie in one chunk; the line around it need not."""

    location = "Studies//Northwind Pricing Study/raw/prices.txt"
    document = _document("file_a", location, "Prices rose.", chunk_tokens=5)

    [mention], _trace = _mentions(_NamingLlm(), document, level=2)

    chunk_of = {item.id: item for item in document[1]}[mention.source_chunk_id]
    # The line's label is in the chunk before the one holding the name.
    assert "Folder:" not in chunk_of.content
    assert mention.name == "Northwind Pricing Study"
    assert chunk_of.content[mention.start_offset : mention.end_offset] == mention.quote
    assert mention.quote == "Northwind Pricing Study"


def test_a_folder_name_no_chunk_holds_whole_is_reported_as_such() -> None:
    long_name = " ".join(f"Phase{index}" for index in range(12))
    document = _document("file_a", f"{long_name}/report.pdf", "Prices rose.", chunk_tokens=4)

    mentions, trace = _mentions(_NamingLlm(), document)

    assert mentions == []
    assert trace[0]["reason"] == "folder_name_split"
    no_page = _document("file_b", "Studies/report.pdf", "Prices rose.")
    body_only = [item for item in no_page[1] if item.page_number != 0]
    _none, skipped = folder_entity_mentions(
        body_only,
        {"file_b": no_page[0]},
        OntologyProfile.model_validate(_profile()),
        _NamingLlm(),
        model="test",
        timeout_seconds=5,
        seed=7,
        parallelism=1,
    )
    assert skipped[0]["reason"] == "no_location_page"


def test_a_local_run_leaves_excluded_folder_mentions_out_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unsharded run resolves the same observations the distributed finalizers do."""

    corpus = tmp_path / "corpus"
    for location, text in (
        ("Northwind Pricing Study/report.txt", "Widget Pro is made by Acme Corp."),
        ("Contract ACME-0042/terms.txt", "Acme Corp supplies Gadget Max."),
    ):
        (corpus / location).parent.mkdir(parents=True, exist_ok=True)
        (corpus / location).write_text(text, encoding="utf-8")
    resolved: list[list[str]] = []

    def recording(observations: ExtractionObservations, *args: Any, **kwargs: Any) -> Any:
        resolved.append(
            [item.name for item in observations.entities if item.evidence_method == "folder"]
        )
        return resolve_extraction_observations(observations, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "resolve_extraction_observations", recording)
    settings = Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph"},
            "files": {"input_path": str(corpus)},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "ontology": {
                "profile": {
                    **_profile(excluded_paths=["Contract ACME-0042/terms.txt"]),
                    "mode": "hybrid",
                }
            },
            "graph": {"entity_resolution_enabled": False, "fail_on_quality_error": False},
        },
    )

    batch = build_pipeline(settings).run()

    assert resolved == [["Northwind Pricing Study"]]
    names = {node.id: node.name for node in batch.nodes}
    assert {
        names[edge.target_node_id] for edge in batch.edges if edge.relation_type == "appears_in"
    } == {"Northwind Pricing Study"}


def test_a_node_is_evidenced_in_a_chunk_by_a_text_mention_before_its_folder_mention() -> None:
    """Assembly and the Spark finalizer choose the same evidence, whatever the descriptions."""

    location = chunk(
        "Location: Northwind/report.pdf\nFolder: Northwind\nFile: report.pdf",
        chunk_id="chunk-0",
        page_number=0,
    )
    start = location.content.index("Northwind\nFile")
    folder = ExtractedEntity(
        name="Northwind",
        type="STUDY",
        description="A much longer description, as a folder entity's often is.",
        source_chunk_id="chunk-0",
        quote="Northwind",
        start_offset=start,
        end_offset=start + len("Northwind"),
        evidence_method="folder",
    )
    stated = folder.model_copy(
        update={
            "description": "Short.",
            "start_offset": 10,
            "end_offset": 19,
            "evidence_method": "llm",
        }
    )

    assembly = assemble_graph_with_decisions("graph", [location], [folder, stated], [], 10.0)

    assert [item.method for item in assembly.evidence] == ["llm"]
