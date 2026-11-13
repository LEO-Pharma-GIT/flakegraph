"""Cross-engine parity contracts executed on a real Spark runtime.

Local and Spark finalization publish into the same tables, so identity, weight,
rank, and rating must be computed identically. These contracts run the Spark
expressions against the reference Python rules on values chosen to expose any
independent reimplementation.
"""

# PySpark is an optional dependency, so its imports stay inside the guarded
# contracts that only run when the distributed extra and a JVM are installed.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.application.community_reports import relation_label, structural_rating
from kg_processor.application.entity_resolution import _has_numeric_mismatch
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.application.graph_merge import (
    claimed_only,
    normalize_entity_name,
    normalize_relation_type,
)
from kg_processor.application.spark_finalization import (
    _EMPTY_SCHEMAS,
    _EXTRACTED_STAGE_SCHEMA,
    _GRAPHFRAMES_COORDINATE,
    SparkFinalizationRequest,
    SparkGraphFinalizer,
    _claimed_only,
    _normalized_name,
    _normalized_relation_type,
    _numeric_mismatch,
    _relation_label,
    _stable_id,
    _structural_rating_column,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk, GraphEdge, Quantity, RelationAssertion
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard

# Values chosen so a rule reimplemented in SQL diverges from the reference rule:
# decomposed accents, case folding that changes length, ligatures, non-Latin
# scripts, whitespace runs, punctuation-only values, and oversized labels.
TRICKY_VALUES: tuple[str, ...] = (
    "Café Latte",
    "Café Latte",
    "  Spaced   Name  ",
    "line\tbreak\nname",
    "STRASSE",
    "Straße",
    "ﬁnance",
    "İstanbul",
    "Next.js",
    "Müller",
    "東京",
    "-- /// --",
    "",
    "Works-At",
    "works at",
    "very-long predicate " * 12,
    "x" * 200,
)
_GRAPH_ID = "parity-graph"
_ONTOLOGY_PROFILE: dict[str, Any] = {
    "name": "parity",
    "description": "Minimal profile exercising self-loop policy.",
    "mode": "hybrid",
    "entity_types": [{"name": "ENTITY", "description": "Any named thing."}],
    "relation_types": [
        {
            "name": "INTERACTS_WITH",
            "description": "One entity interacts with another.",
            "source_types": ["ENTITY"],
            "target_types": ["ENTITY"],
            "allow_self_loop": True,
        }
    ],
}


def _java_runtime_available() -> bool:
    """Distinguish macOS's Java launcher stub from an installed JVM."""

    try:
        return (
            subprocess.run(
                ["java", "-version"],
                check=False,
                capture_output=True,
                timeout=5,
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _spark_runtime_available() -> bool:
    """Require both optional Python packages and a working Java runtime."""

    return (
        find_spec("pyspark") is not None
        and find_spec("graphframes") is not None
        and _java_runtime_available()
    )


pytestmark = [
    pytest.mark.spark,
    pytest.mark.skipif(
        os.getenv("KG_RUN_SPARK_INTEGRATION") != "1" or not _spark_runtime_available(),
        reason=(
            "Set KG_RUN_SPARK_INTEGRATION=1 with the distributed-spark extra and a Java 17+ "
            "runtime installed to run Spark integration checks."
        ),
    ),
]
LINKED = frozenset({"prepared", "extracted"})


@pytest.fixture(scope="module")
def spark_session() -> Any:
    """Provide the local Spark session shared by every contract in this module.

    One JVM serves the whole test process and resolves declared packages only
    while a context starts, so this session carries what a full finalization run
    needs and stays available for the finalizer to adopt. Finalization releases
    the session it used, which is why this fixture never stops it.
    """

    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[2]")
        .appName("flakegraph-cross-engine-parity")
        .config("spark.ui.enabled", "false")
        .config("spark.jars.packages", _GRAPHFRAMES_COORDINATE)
        # Local executors live in this process, and a developer hostname can
        # resolve to a VPN or Wi-Fi interface that is not a valid loopback route.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def test_spark_canonical_identity_matches_the_local_rules(spark_session: Any) -> None:
    """Both engines must derive the same canonical keys and IDs for one surface."""

    from pyspark.sql import functions as F

    frame = spark_session.createDataFrame(
        [(index, value) for index, value in enumerate(TRICKY_VALUES)],
        "row_index int, value string",
    )
    computed = (
        frame.withColumn("normalized_name", _normalized_name(F.col("value")))
        .withColumn("relation_label", _normalized_relation_type(F.col("value")))
        .withColumn(
            "node_id",
            _stable_id(
                "node",
                F.lit(_GRAPH_ID),
                F.col("normalized_name"),
                F.lit("ENTITY"),
            ),
        )
        .collect()
    )

    assert len(computed) == len(TRICKY_VALUES)
    for row in computed:
        value = TRICKY_VALUES[row.row_index]
        assert row.normalized_name == normalize_entity_name(value)
        assert row.relation_label == normalize_relation_type(value)
        assert row.node_id == stable_id("node", _GRAPH_ID, normalize_entity_name(value), "ENTITY")


def test_spark_keeps_numerically_distinct_names_apart_like_the_local_rule(
    spark_session: Any,
) -> None:
    """A name pair one digit apart scores above auto-merge; both engines must refuse it."""

    from pyspark.sql import functions as F

    pairs = [
        ("ISO 27001", "ISO 27002"),
        ("GPT-4", "GPT-4"),
        ("Version 2.0", "Version 2,0"),
        ("Phase 1", "Phase 1 trial"),
        ("Alpha", "Alpha 1"),
        ("Beta", "Gamma"),
    ]
    frame = spark_session.createDataFrame(pairs, "left string, right string")
    computed = frame.withColumn(
        "numeric_mismatch", _numeric_mismatch(F.col("left"), F.col("right"))
    ).collect()

    for row in computed:
        left = _mention("l", row.left, "chunk", start_offset=0)
        right = _mention("r", row.right, "chunk", start_offset=0)
        assert row.numeric_mismatch == _has_numeric_mismatch(left, right), (row.left, row.right)


def test_spark_community_rating_matches_the_local_rule(spark_session: Any) -> None:
    """Ratings published by either engine must agree for identical structure."""

    from pyspark.sql import functions as F

    cases = [
        (3, 3, 2, 4, 1.0),
        (10, 5, 40, 4, 0.0),
        (1, 1, 0, 0, 0.5),
        (2, 3, 3, 6, 1.0),
    ]
    frame = spark_session.createDataFrame(
        cases,
        "member_count int, endpoint_pair_count int, reported_relations int, "
        "quotes int, confidence double",
    )
    computed = frame.withColumn(
        "structural_rating",
        _structural_rating_column(
            F.col("member_count"),
            F.col("endpoint_pair_count"),
            F.col("reported_relations"),
            F.col("quotes"),
            F.col("confidence"),
        ),
    ).collect()

    assert len(computed) == len(cases)
    for row in computed:
        rating, explanation = structural_rating(
            row.member_count,
            row.endpoint_pair_count,
            row.reported_relations,
            row.quotes,
            row.confidence,
        )
        assert row.structural_rating.rating == rating
        assert row.structural_rating.explanation == explanation
        assert 0.0 <= row.structural_rating.rating <= 10.0


def test_spark_assertion_aggregate_matches_the_local_rule(spark_session: Any) -> None:
    """An edge or observation is claimed in both engines only when all its assertions are."""

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    groups: dict[str, list[RelationAssertion]] = {
        "all-claimed": ["claimed", "claimed"],
        "one-reported": ["claimed", "reported", "claimed"],
        "all-reported": ["reported"],
        "single-claimed": ["claimed"],
    }
    frame = spark_session.createDataFrame(
        [(group, value) for group, values in groups.items() for value in values],
        "group_id string, assertion string",
    )
    aggregated = {
        row.group_id: row.assertion
        for row in frame.groupBy("group_id")
        .agg(_claimed_only(F.col("assertion")).alias("assertion"))
        .collect()
    }
    windowed = {
        row.group_id: row.assertion
        for row in frame.select(
            "group_id",
            _claimed_only(F.col("assertion"), Window.partitionBy("group_id")).alias("assertion"),
        ).collect()
    }

    expected = {group: claimed_only(values) for group, values in groups.items()}
    assert aggregated == expected
    assert windowed == expected
    assert expected["one-reported"] == "reported"
    assert expected["all-claimed"] == "claimed"


def test_spark_report_relation_label_matches_the_local_rule(spark_session: Any) -> None:
    """Both engines mark a claimed relation alike in a community report's context."""

    from pyspark.sql import functions as F

    cases: list[tuple[str, RelationAssertion]] = [
        ("protects", "claimed"),
        ("protects", "reported"),
    ]
    rows = (
        spark_session.createDataFrame(cases, "relation_type string, assertion string")
        .select(_relation_label(F.col("relation_type"), F.col("assertion")).alias("label"))
        .collect()
    )

    expected = [
        relation_label(
            GraphEdge(
                id="edge",
                graph_id=_GRAPH_ID,
                source_node_id="a",
                target_node_id="b",
                relation_type=relation_type,
                description="",
                weight=1.0,
                source_file_id="file",
                source_chunk_ids=[],
                assertion=assertion,
            )
        )
        for relation_type, assertion in cases
    ]
    assert [row.label for row in rows] == expected == ["protects (claimed)", "protects"]


def test_spark_edges_sum_observation_weight_and_reject_self_loops(
    spark_session: Any,
    tmp_path: Path,
) -> None:
    """Canonical edges carry summed observation weight, rank, and no self-loops.

    Finalization adopts the module session, so this contract runs last and leaves
    no session behind.
    """

    assert not spark_session.sparkContext._jsc.sc().isStopped()

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    run_id = "parity-run"
    first_text = "Alpha interacts with Beta."
    second_text = "alpha interacts with beta again."
    chunks = [
        _chunk("chunk-1", "file-1", first_text),
        _chunk("chunk-2", "file-1", second_text, start_offset=len(first_text)),
    ]
    confidence = 0.6
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        document_rows=[
            {
                "id": "document-1",
                "graph_id": _GRAPH_ID,
                "file_id": "file-1",
                "checksum": "checksum-1",
                "source_uri": "fixture.txt",
                "mime_type": "text/plain",
                "size_bytes": len(first_text) + len(second_text),
                "ocr_provider": "builtin_text",
                "kind": "dataset",
                "summary": None,
                "related_file_ids": [],
            }
        ],
        page_rows=[],
        block_rows=[],
        asset_rows=[],
        chunks=chunks,
        document_context_entities=[],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=[
                _mention("mention-alpha-1", "Alpha", "chunk-1", start_offset=0),
                _mention(
                    "mention-beta-1",
                    "Beta",
                    "chunk-1",
                    start_offset=first_text.index("Beta"),
                ),
                _mention("mention-alpha-2", "alpha", "chunk-2", start_offset=0),
                _mention(
                    "mention-beta-2",
                    "beta",
                    "chunk-2",
                    start_offset=second_text.index("beta"),
                ),
            ],
            relations=[
                _relation(
                    "relation-1",
                    "mention-alpha-1",
                    "mention-beta-1",
                    "chunk-1",
                    first_text,
                    confidence,
                ).model_copy(
                    update={
                        "quantities": [Quantity(text="0.5 %", kind="limit", low=0.5, high=0.5)],
                        "assertion": "claimed",
                    }
                ),
                _relation(
                    "relation-2",
                    "mention-alpha-2",
                    "mention-beta-2",
                    "chunk-2",
                    second_text,
                    confidence,
                ),
                # Two spellings of one entity resolve to a single canonical node.
                # The resulting loop is an assembly artifact, not an assertion the
                # ontology's self-loop allowance should preserve.
                _relation(
                    "relation-3",
                    "mention-alpha-1",
                    "mention-alpha-2",
                    "chunk-1",
                    first_text,
                    confidence,
                ),
            ],
            chunk_count=2,
            window_count=1,
            trace=[
                {
                    "stage": "dataset_profile",
                    "document_id": "file-1",
                    "kind": "catalogue",
                    "summary": "A list of interacting things.",
                }
            ],
        ),
    )
    store.put(
        f"{run_id}/prepared_document/prepared.json",
        prepared.model_dump_json().encode(),
        "application/json",
    )
    store.put(
        f"{run_id}/extracted_document/extracted.json",
        extracted.model_dump_json().encode(),
        "application/json",
    )
    settings = Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {"graph_id": _GRAPH_ID},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "ontology": {"profile": _ONTOLOGY_PROFILE},
            "distributed": {
                "artifact_uri": root.as_uri(),
                "finalization_engine": "spark",
                "spark_master": "local[2]",
                "spark_executor_instances": 2,
                "spark_executor_cores": 1,
                "spark_executor_memory": "1g",
            },
        },
    )

    manifest = SparkGraphFinalizer(settings).finalize(
        SparkFinalizationRequest(run_id=run_id, graph_id=_GRAPH_ID, attempt=1, artifact_ids=LINKED)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert len(batch.edges) == 1
    edge = batch.edges[0]
    assert edge.source_node_id != edge.target_node_id
    assert edge.weight == pytest.approx(2 * confidence)
    assert edge.weight <= settings.graph.relation_weight_max
    assert edge.evidence_count == 2
    assert {node.id for node in batch.nodes} == {edge.source_node_id, edge.target_node_id}
    assert all(node.rank == float(node.degree) for node in batch.nodes)
    assert all(node.degree == 1 for node in batch.nodes)
    assert manifest.metrics["hard_quality_error_count"] == 0
    # The values a relation states and a dataset's profile survive Spark as they
    # survive a local run.
    assert [
        [quantity.text for quantity in observation.quantities]
        for observation in sorted(batch.edge_observations, key=lambda item: item.chunk_id)
    ] == [["0.5 %"], []]
    # One reported observation makes the edge reported; each observation keeps its own.
    assert edge.assertion == "reported"
    assert [
        observation.assertion
        for observation in sorted(batch.edge_observations, key=lambda item: item.chunk_id)
    ] == ["claimed", "reported"]
    [document] = batch.documents
    assert (document["kind"], document["summary"]) == ("dataset", "A list of interacting things.")


def _chunk(chunk_id: str, file_id: str, content: str, *, start_offset: int = 0) -> Chunk:
    """Build a valid source chunk with the offsets evidence joins consume."""

    return Chunk(
        id=chunk_id,
        graph_id=_GRAPH_ID,
        file_id=file_id,
        document_id=file_id,
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
        token_count=len(content.split()),
        content_hash=sha256_hex(content),
    )


def _mention(entity_id: str, name: str, chunk_id: str, *, start_offset: int) -> EntityMention:
    """Build one grounded mention of a canonical entity."""

    return EntityMention(
        id=entity_id,
        name=name,
        type="ENTITY",
        description=f"A mention of {name}.",
        source_chunk_id=chunk_id,
        quote=name,
        start_offset=start_offset,
        end_offset=start_offset + len(name),
    )


def _relation(
    relation_id: str,
    source_entity_id: str,
    target_entity_id: str,
    chunk_id: str,
    quote: str,
    confidence: float,
) -> RelationObservation:
    """Build one grounded relation observation between accepted mentions."""

    return RelationObservation(
        id=relation_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type="INTERACTS_WITH",
        description="One entity interacts with another.",
        source_chunk_id=chunk_id,
        quote=quote,
        start_offset=0,
        end_offset=len(quote),
        confidence=confidence,
    )


def test_a_discarded_window_is_named_by_the_document_its_event_names(
    spark_session: Any, tmp_path: Path
) -> None:
    """The gap report exists to say *which* documents may be missing data."""

    import json

    artifacts = tmp_path / "extracted.jsonl"
    extracted_document = {
        "observations": {
            "trace": [
                {
                    "stage": "entity_extraction",
                    "window_id": "window_new",
                    "document_id": "file_named_by_event",
                    "chunk_ids": [],
                    "input_records": 1,
                    "accepted_records": 0,
                    "record_actions": {"ungrounded_quote": 1},
                }
            ]
        },
    }
    artifacts.write_text(json.dumps(extracted_document), encoding="utf-8")
    extracted = spark_session.read.schema(_EXTRACTED_STAGE_SCHEMA).json(str(artifacts))
    chunks = spark_session.createDataFrame([], _EMPTY_SCHEMAS["chunks"])

    finalizer = SparkGraphFinalizer(
        Settings.load(
            env={},
            overrides={
                "llm": {"provider": "fake"},
                "distributed": {"artifact_uri": tmp_path.as_uri()},
            },
        )
    )
    (row,) = finalizer._discarded_windows(spark_session, extracted, chunks, _GRAPH_ID).collect()

    assert row.document_id == "file_named_by_event"
    assert row.file_id == "file_named_by_event"
    assert dict(row.record_actions) == {"ungrounded_quote": 1}


def test_spark_names_an_unreadable_document_exactly_like_the_local_rule(
    spark_session: Any, tmp_path: Path
) -> None:
    """One trace event, two engines, the same failed_documents row.

    A worker records a document parsing gave up on as one event in that
    document's trace; the local pipeline and the Spark finalizer must turn it
    into the same row, id included, so the console reports one unreadable file
    once wherever the graph was built.
    """

    import json

    from kg_processor.application.failed_documents import (
        failed_document_event,
        failed_documents_from_trace,
    )
    from kg_processor.domain.documents import InputFile

    unreadable = InputFile(
        id="file_broken",
        path=tmp_path / "broken.pdf",
        source_uri="s3://corpus/broken.pdf",
        checksum="sha256:0",
        mime_type="application/pdf",
        size_bytes=18,
    )
    event = failed_document_event(
        unreadable,
        "fallback",
        {"error_type": "RuntimeError", "error_message": "Unsupported file type: txt"},
    )
    artifacts = tmp_path / "extracted.jsonl"
    artifacts.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"trace": [event], "observations": {"trace": []}},
                {"trace": [{"stage": "ocr", "file_id": "file_fine"}], "observations": {}},
            )
        ),
        encoding="utf-8",
    )
    extracted = spark_session.read.schema(_EXTRACTED_STAGE_SCHEMA).json(str(artifacts))
    finalizer = SparkGraphFinalizer(
        Settings.load(
            env={},
            overrides={
                "llm": {"provider": "fake"},
                "distributed": {"artifact_uri": tmp_path.as_uri()},
            },
        )
    )

    spark_rows = [
        row.asDict() for row in finalizer._failed_documents(extracted, _GRAPH_ID).collect()
    ]
    local_rows = [row.model_dump() for row in failed_documents_from_trace([event], _GRAPH_ID)]

    assert spark_rows == local_rows


def test_each_quality_check_counts_its_own_violations(spark_session: Any, tmp_path: Path) -> None:
    """Every check sees the one row seeded to break it, and only that row."""

    good = [0.0, 0.0]
    nodes = spark_session.createDataFrame(
        [
            ("n1", "CONCEPT", good, "kept"),
            ("n2", "CONCEPT", good, "kept"),
            ("no-evidence", "CONCEPT", good, "kept"),
            ("short-vector", "CONCEPT", [0.0], "kept"),
            ("no-vector", "CONCEPT", None, "kept"),
        ],
        "id string, primary_type string, embedding array<double>, unread string",
    )
    edges = spark_session.createDataFrame(
        [
            ("e1", "n1", "n2", "RELATED_TO", good, "kept"),
            ("orphan", "n1", "missing", "PART_OF", good, "kept"),
            ("twin-1", "n2", "n1", "CREATED_BY", good, "kept"),
            ("twin-2", "n2", "n1", "CREATED_BY", good, "kept"),
            ("loop", "n1", "n1", "LOCATED_IN", good, "kept"),
            ("no-evidence", "n2", "n2x", "OCCURRED_AT", None, "kept"),
        ],
        "id string, source_node_id string, target_node_id string, relation_type string, "
        "embedding array<double>, unread string",
    )
    evidence = spark_session.createDataFrame(
        [
            *(
                (f"ev-{subject}", kind, subject, "c1", 10, 15, "hello", "kept")
                for kind, subjects in (
                    ("node", ("n1", "n2", "short-vector", "no-vector")),
                    ("edge", ("e1", "orphan", "twin-1", "twin-2", "loop")),
                )
                for subject in subjects
            ),
            ("bad-span", "node", "n1", "c1", 10, 15, "other", "kept"),
        ],
        "id string, subject_kind string, subject_id string, chunk_id string, "
        "start_offset int, end_offset int, quote string, unread string",
    )
    chunks = spark_session.createDataFrame(
        [("c1", "hello world", 10, "kept")],
        "id string, content string, start_offset int, unread string",
    )
    finalizer = SparkGraphFinalizer(
        Settings.load(
            env={},
            overrides={
                "llm": {"provider": "fake"},
                "embedding": {"provider": "hash", "dimension": 2},
                "distributed": {"artifact_uri": tmp_path.as_uri()},
            },
        )
    )

    counts = finalizer._hard_quality_errors(spark_session, nodes, edges, evidence, chunks)

    assert counts == {
        "orphan_edges": 2,
        "duplicate_edges": 1,
        "unapproved_self_loops": 1,
        "ontology_constraints": 0,
        "nodes_without_evidence": 1,
        "edges_without_evidence": 1,
        "invalid_evidence_spans": 1,
        "invalid_node_embeddings": 2,
        "invalid_edge_embeddings": 1,
    }
