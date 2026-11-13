"""A catalogue dataset's rows state relations through the columns its profile maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from documents import chunk, window

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.content_kinds import (
    catalogue_columns,
    dataset_columns,
    window_filter,
)
from kg_processor.application.extraction_contracts import (
    dataset_profile_request,
    relation_extraction_request,
)
from kg_processor.application.llm_extractors import LlmEntityExtractor
from kg_processor.application.two_pass_extraction import extract_graph_observations
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import ColumnRelation, EntityMention
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult

_FIXTURE = Path("tests/fixtures/datasets/component_catalogue.tsv")
_DOCUMENT = "catalogue"


def _ontology() -> OntologyProfile:
    return OntologyProfile(
        name="catalogue",
        description="Components, who supplies them, and the products they are used in.",
        entity_types=[
            EntityTypeDefinition(name="COMPONENT", description="A part a product is made of."),
            EntityTypeDefinition(name="ORGANIZATION", description="A company."),
            EntityTypeDefinition(name="PRODUCT", description="A finished product."),
        ],
        relation_types=[
            RelationTypeDefinition(
                name="SUPPLIES",
                description="An organization supplies a component.",
                source_types=["ORGANIZATION"],
                target_types=["COMPONENT"],
            ),
            RelationTypeDefinition(
                name="USED_IN",
                description="A component is used in a product, up to an amount.",
                source_types=["COMPONENT"],
                target_types=["PRODUCT"],
                quantities=True,
            ),
        ],
    )


# What a model proposes for the fixture's header: every column, one of them
# under a relation the ontology does not define.
_PROPOSED = [
    {"column": "Component", "role": "subject", "entity_type": "COMPONENT"},
    {"column": "Code", "role": "alias"},
    {
        "column": "Supplier",
        "role": "relation",
        "entity_type": "ORGANIZATION",
        "relation_type": "SUPPLIES",
    },
    {"column": "Used in", "role": "relation", "entity_type": "PRODUCT", "relation_type": "USED_IN"},
    {"column": "Max quantity", "role": "value", "of_column": "Used in"},
    {
        "column": "Remarks",
        "role": "relation",
        "entity_type": "PRODUCT",
        "relation_type": "DESCRIBED_BY",
    },
]
_KEPT = [
    {"column": "Component", "role": "subject", "entity_type": "COMPONENT"},
    {"column": "Code", "role": "alias"},
    {
        "column": "Supplier",
        "role": "relation",
        "entity_type": "ORGANIZATION",
        "relation_type": "SUPPLIES",
        "source": "column",
    },
    {
        "column": "Used in",
        "role": "relation",
        "entity_type": "PRODUCT",
        "relation_type": "USED_IN",
        "source": "subject",
    },
    {"column": "Max quantity", "role": "value", "of_column": "Used in"},
]


def _columns(proposed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept, rejected = catalogue_columns(proposed, _ontology())
    return [column.model_dump(exclude_none=True) for column in kept], rejected


def test_a_profile_keeps_the_column_relations_the_ontology_holds() -> None:
    """A relation's direction follows its type rules; an undefined relation is dropped."""

    kept, rejected = _columns(_PROPOSED)

    assert kept == _KEPT
    assert rejected == [
        {
            "column": "Remarks",
            "role": "relation",
            "reason": "unknown_relation",
            "relation_type": "DESCRIBED_BY",
        }
    ]


def test_columns_the_ontology_cannot_hold_are_turned_away_with_their_reason() -> None:
    subject = {"column": "Component", "role": "subject", "entity_type": "COMPONENT"}
    kept, rejected = _columns(
        [
            subject,
            {"column": "Part", "role": "subject", "entity_type": "COMPONENT"},
            {"column": "Maker", "role": "relation", "entity_type": "GADGET", "relation_type": ""},
            {
                "column": "Assembly",
                "role": "relation",
                "entity_type": "ORGANIZATION",
                "relation_type": "USED_IN",
            },
            {
                "column": "Supplier",
                "role": "relation",
                "entity_type": "ORGANIZATION",
                "relation_type": "supplies",
            },
            {"column": "Lead time", "role": "value", "of_column": "Supplier"},
            {"column": "Weight", "role": "value", "of_column": "Size"},
            {"column": "Price", "role": "value", "of_column": ""},
            {"column": " supplier ", "role": "alias"},
            {"column": "Row", "role": "none"},
            {"column": "", "role": "alias"},
        ]
    )

    assert [(column["column"], column["role"]) for column in kept] == [
        ("Component", "subject"),
        ("Supplier", "relation"),
        ("Price", "value"),
    ]
    assert [(record["column"], record["reason"]) for record in rejected] == [
        (" supplier ", "duplicate_column"),
        ("", "blank_column"),
        ("Part", "second_subject"),
        ("Maker", "unknown_entity_type"),
        ("Assembly", "relation_type_rules"),
        ("Lead time", "relation_carries_no_quantities"),
        ("Weight", "value_of_unmapped_column"),
    ]


def test_without_a_subject_no_column_is_kept() -> None:
    kept, rejected = _columns(
        [
            {"column": "Component", "role": "subject", "entity_type": "GADGET"},
            {"column": "Code", "role": "alias"},
        ]
    )

    assert kept == []
    assert [(record["column"], record["reason"]) for record in rejected] == [
        ("Component", "unknown_entity_type"),
        ("Code", "no_subject"),
    ]


def test_the_profile_request_offers_the_ontology_relations_for_the_columns() -> None:
    request = dataset_profile_request(
        window(chunk(_FIXTURE.read_text(encoding="utf-8"))),
        _ontology(),
        file_name="component_catalogue.tsv",
        model="fake",
        timeout_seconds=30,
        seed=17,
    )
    payload = _input(request)
    item = request.json_schema["properties"]["columns"]["items"]

    assert [relation["name"] for relation in payload["relation_types"]] == ["SUPPLIES", "USED_IN"]
    assert item["properties"]["relation_type"]["enum"] == ["SUPPLIES", "USED_IN", ""]
    assert item["properties"]["role"]["enum"] == ["subject", "relation", "value", "alias", "none"]


def test_a_profile_records_its_column_relations_on_the_trace() -> None:
    """Only a catalogue maps its columns; the columns turned away are listed."""

    for kind, kept, turned_away in (("catalogue", _KEPT, ["Remarks"]), ("other", [], [])):
        event = LlmEntityExtractor(_CatalogueLlm(kind=kind)).profile_dataset(
            window(chunk(_FIXTURE.read_text(encoding="utf-8"), document_id=_DOCUMENT)),
            _ontology(),
            file_name="component_catalogue.tsv",
            model="fake",
            timeout_seconds=30,
        )

        assert (event["kind"], event["columns"]) == (kind, kept)
        assert [record["column"] for record in event["rejected_columns"]] == turned_away


def test_catalogue_windows_go_to_relation_extraction_with_their_columns() -> None:
    """Only a catalogue's windows carry column relations; no other window's prompt changes."""

    columns = [ColumnRelation.model_validate(column) for column in _KEPT]
    trace = [
        {
            "stage": "dataset_profile",
            "document_id": _DOCUMENT,
            "kind": "catalogue",
            "columns": _KEPT,
        },
        {"stage": "dataset_profile", "document_id": "calc", "kind": "other", "columns": _KEPT},
    ]

    assert window_filter(trace, catalogue_relations=True)(_DOCUMENT) == (True, True)
    assert window_filter(trace, catalogue_relations=True)("calc") == (True, False)
    assert dataset_columns(trace) == {_DOCUMENT: columns}

    rows = window(chunk("Steel bolt\tSB-100\tNorthwind Metals", document_id=_DOCUMENT))
    request = relation_extraction_request(
        rows.model_copy(update={"dataset_columns": columns}),
        _mentions(rows.chunks[0], ["Steel bolt", "Northwind Metals"]),
        _ontology(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
        seed=17,
        previous_relations=None,
    )
    plain = relation_extraction_request(
        rows,
        _mentions(rows.chunks[0], ["Steel bolt", "Northwind Metals"]),
        _ontology(),
        model="fake",
        timeout_seconds=30,
        max_relations=10,
        seed=17,
        previous_relations=None,
    )

    assert _input(request)["dataset_columns"] == _KEPT
    assert "dataset_columns" not in _input(plain)


def test_catalogue_rows_yield_relations_grounded_in_their_rows() -> None:
    """Every row states its supplier and the product it is used in, with the amount.

    The second window holds rows without the header, rendered as a markdown
    table as some parsers write a sheet; the column relations still reach it.
    """

    header, *rows = _FIXTURE.read_text(encoding="utf-8").splitlines()
    chunks = [
        chunk("\n".join([header, *rows[:2]]), chunk_id="rows-1", document_id=_DOCUMENT),
        chunk(
            "\n".join("| " + " | ".join(row.split("\t")) + " |" for row in rows[2:]),
            chunk_id="rows-2",
            document_id=_DOCUMENT,
        ),
    ]
    llm = _CatalogueLlm(kind="catalogue", header=header.split("\t"))
    profile = LlmEntityExtractor(llm).profile_dataset(
        window(chunks[0]),
        _ontology(),
        file_name="component_catalogue.tsv",
        model="fake",
        timeout_seconds=30,
    )
    settings = GraphSettings(max_chunks_per_llm_call=1, gleaning_max_passes=0)

    observations = extract_graph_observations(
        chunks,
        llm,
        settings,
        _ontology(),
        "fake",
        30,
        window_filter=window_filter([profile], catalogue_relations=True),
        dataset_columns=dataset_columns([profile]),
    )

    names = {entity.id: entity.name for entity in observations.entities}
    stated = {
        (
            names[relation.source_entity_id],
            relation.relation_type,
            names[relation.target_entity_id],
            tuple(quantity.text for quantity in relation.quantities),
        )
        for relation in observations.relations
    }
    assert stated == {
        ("Northwind Metals", "SUPPLIES", "Steel bolt", ()),
        ("Steel bolt", "USED_IN", "Garden bench", ("12 pcs",)),
        ("Fjord Timber", "SUPPLIES", "Oak plank", ()),
        ("Oak plank", "USED_IN", "Garden bench", ("4 pcs",)),
        ("Northwind Metals", "SUPPLIES", "Brass hinge", ()),
        ("Brass hinge", "USED_IN", "Tool chest", ("2 pcs",)),
    }
    lines = [line for item in chunks for line in item.content.splitlines()]
    for relation in observations.relations:
        [row] = [line for line in lines if relation.quote in line]
        assert names[relation.source_entity_id] in row
        assert names[relation.target_entity_id] in row
    assert not [
        event["rejected_records"]
        for event in observations.trace
        if event.get("stage") == "relation_extraction" and event.get("rejected_records")
    ]
    assert [_input(request)["dataset_columns"] for request in llm.relation_requests] == [
        _KEPT,
        _KEPT,
    ]

    off = _CatalogueLlm(kind="catalogue", header=header.split("\t"))
    extract_graph_observations(
        chunks,
        off,
        settings,
        _ontology(),
        "fake",
        30,
        window_filter=window_filter([profile], catalogue_relations=False),
        dataset_columns=dataset_columns([profile]),
    )
    assert off.relation_requests == []


class _CatalogueLlm(FakeLlmProvider):
    """A model that reads a catalogue exactly as its columns say.

    It proposes the fixture's column relations, names each row's things, and
    states each row's relations by the column relations the request carries.
    """

    def __init__(self, *, kind: str, header: list[str] | None = None) -> None:
        self.kind = kind
        self.header = header or []
        self.relation_requests: list[StructuredCompletionRequest] = []

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        payload = _input(request)
        if request.task_name == "dataset_profile":
            return StructuredCompletionResult(
                payload={
                    "kind": self.kind,
                    "summary": "Components with their suppliers.",
                    "columns": _PROPOSED,
                }
            )
        if request.task_name == "entity_extraction":
            return StructuredCompletionResult(payload={"entities": self._entities(payload)})
        if request.task_name == "relation_extraction":
            self.relation_requests.append(request)
            return StructuredCompletionResult(payload={"relations": self._relations(payload)})
        return super().complete_structured(request)

    def _rows(self, payload: dict[str, Any]) -> list[tuple[str, str, dict[str, str]]]:
        """Each row's chunk, its text, and its cells by column."""

        rows = []
        for item in payload["chunks"]:
            for line in item["content"].splitlines():
                cells = [
                    cell.strip()
                    for cell in line.strip().strip("|").split("\t" if "\t" in line else "|")
                ]
                if cells != self.header:
                    rows.append((item["id"], line, dict(zip(self.header, cells, strict=False))))
        return rows

    def _entities(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        typed = {"Component": "COMPONENT", "Supplier": "ORGANIZATION", "Used in": "PRODUCT"}
        entities: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk_id, line, cells in self._rows(payload):
            for column, entity_type in typed.items():
                entities.setdefault(
                    (chunk_id, cells[column]),
                    {
                        "name": cells[column],
                        "type": entity_type,
                        "description": f"{cells[column]} is listed in the catalogue.",
                        "source_chunk_id": chunk_id,
                        "quote": line,
                        "confidence": 0.9,
                        "aliases": [],
                    },
                )
        return list(entities.values())

    def _relations(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        columns = payload["dataset_columns"]
        [subject] = [column["column"] for column in columns if column["role"] == "subject"]
        ids = {entity["name"]: entity["id"] for entity in payload["entities"]}
        relations = []
        for chunk_id, line, cells in self._rows(payload):
            for column in (item for item in columns if item["role"] == "relation"):
                ends = (cells[subject], cells[column["column"]])
                source, target = ends if column["source"] == "subject" else ends[::-1]
                relations.append(
                    {
                        "source_entity_id": ids[source],
                        "target_entity_id": ids[target],
                        "source_surface": source,
                        "target_surface": target,
                        "relation_type": column["relation_type"],
                        "description": f"{source} {column['relation_type']} {target}.",
                        "source_chunk_id": chunk_id,
                        "quote": line,
                        "confidence": 0.9,
                        "quantities": [
                            {
                                "text": cells[value["column"]],
                                "kind": "limit",
                                "unit": "",
                                "comparator": "",
                            }
                            for value in columns
                            if value.get("of_column") == column["column"]
                        ],
                    }
                )
        return relations


def _mentions(source: Chunk, names: list[str]) -> list[EntityMention]:
    types = {"Steel bolt": "COMPONENT", "Northwind Metals": "ORGANIZATION"}
    return [
        EntityMention(
            id=name,
            name=name,
            type=types[name],
            description="",
            source_chunk_id=source.id,
            quote=name,
        )
        for name in names
    ]


def _input(request: StructuredCompletionRequest) -> dict[str, Any]:
    return dict(json.loads(request.user.split("INPUT_JSON:\n", 1)[1]))
