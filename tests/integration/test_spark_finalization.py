"""End-to-end contract test for partitioned Spark graph finalization."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.application.rejected_records import (
    rejected_record_metrics,
    rejected_records_from_trace,
)
from kg_processor.application.spark_finalization import (
    SparkFinalizationRequest,
    SparkGraphFinalizer,
    _is_stopped,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.consumption import ConsumptionEvent, TokenUsage
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import Chunk, GraphWriteBatch
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard

SparkSession = pytest.importorskip("pyspark.sql").SparkSession


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
    """Require both optional Python packages and a working Java runtime.

    Spark is intentionally an optional deployment backend. The ordinary development
    environment should skip these integration contracts, while environments installed
    with ``--extra distributed-spark`` exercise them whenever Java is available.
    """

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
# The stage objects these tests write, named as the finalizer sees them: an
# object's id is its file name without the media-type suffix.
LINKED = frozenset({"prepared", "extracted"})


def _assert_discarded_window_contract(
    manifest: GraphDatasetManifest, batch: GraphWriteBatch
) -> None:
    """The fully rejected window is a row a reader can act on and a number the report carries.

    Computed in Spark the same way the local pipeline computes it from its
    in-process trace, so one corpus reports the same gaps on either engine.
    """

    assert manifest.tables["discarded_windows"].row_count == 1
    (gap,) = batch.discarded_windows
    assert (gap.stage, gap.document_id, gap.file_id) == ("entities", "file-2", "file-2")
    assert (gap.page_start, gap.page_end, gap.extracted_records) == (1, 1, 2)
    assert gap.record_actions == {"ungrounded_quote": 2}
    assert gap.preview.startswith("The name judo refers to the same martial art.")
    assert manifest.metrics["discarded_windows"] == {
        "windows": 1,
        "documents": 1,
        "by_stage": {"entities": 1},
        "record_actions": {"ungrounded_quote": 2},
    }


def _assert_dataset_export_matches_batch_export(
    tmp_path: Path,
    manifest: GraphDatasetManifest,
    store: LocalBlobStore,
) -> None:
    """Exporting Spark's tables one batch at a time gives the files a whole-batch export gives."""

    reader = GraphDatasetReader(store)
    LocalArtifactsWriter(tmp_path / "whole").write(reader.read(manifest))
    LocalArtifactsWriter(tmp_path / "tables").write_dataset(manifest, reader)
    for table in sorted(path.name for path in (tmp_path / "whole").glob("*.parquet")):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "tables" / table),
            pd.read_parquet(tmp_path / "whole" / table),
        )


def _assert_homonyms_keep_their_own_ids(batch: GraphWriteBatch) -> None:
    """Two nodes share the name SRN and the type MODEL; each has an id of its own.

    Resolution keeps a bare acronym apart from a spelled-out one. Were both to
    take the id their shared name and type give, every later join on node id
    would multiply their rows.
    """

    assert len(batch.nodes) == 8
    assert len({node.id for node in batch.nodes}) == len(batch.nodes)
    assert sorted(node.name for node in batch.nodes if node.normalized_name == "srn") == [
        "SRN",
        "SRN",
    ]


def _assert_extraction_contract(extraction: dict[str, object]) -> None:
    """Check a fleet run reports extraction the way a local run does.

    Rejection and repair reasons are the only record of why a document lost
    entities or relations, and a fleet run is the one an operator can least
    easily inspect. Reporting them under different keys, or summing a different
    set of stages, would leave two runs over one corpus incomparable.
    """

    # Two entity calls: the window that kept records and the one that kept none.
    assert extraction["entity_calls"] == 2
    assert extraction["relation_calls"] == 1
    assert extraction["verification_calls"] == 1
    assert extraction["document_context_calls"] == 1
    # Verification raises reasons that a local run does not fold in, so its nine
    # rejections must not appear here either.
    assert extraction["record_actions"] == {
        "domain_or_range_violation": 4,
        "duplicate": 1,
        "ungrounded_quote": 7,
    }
    assert extraction["chunk_count"] == 2
    assert extraction["window_count"] == 1
    assert extraction["entity_mentions"] == 8
    assert extraction["relation_observations"] == 1


def test_spark_finalizer_writes_manifest_without_collecting_graph_tables(
    tmp_path: Path,
) -> None:
    """Resolve aliases, canonicalize endpoints, and publish readable Parquet tables."""

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    run_id = "spark-run"
    graph_id = "spark-graph"
    first_chunk_text = (
        "Judo influenced Brazilian Jiu-Jitsu. Elman simple recurrent network is also called SRN."
    )
    second_chunk_text = (
        "The name judo refers to the same martial art. "
        "Scene Representation Networks (SRN) render scenes. "
        "DA may abbreviate either Dario Amodei or Diogo Almeida."
    )
    chunks = [
        _chunk(
            "chunk-1",
            "file-1",
            first_chunk_text,
            start_offset=120,
        ),
        _chunk("chunk-2", "file-2", second_chunk_text),
    ]
    prepared = PreparedDocumentShard(
        file_ids=["file-1", "file-2"],
        files_seen=2,
        documents_processed=2,
        document_rows=[
            {
                "id": "document-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "checksum": "checksum-1",
                "source_uri": "fixture.txt",
                "mime_type": "text/plain",
                "size_bytes": 42,
                "ocr_provider": "builtin_text",
            }
        ],
        page_rows=[
            {
                "id": "page-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "page_number": 1,
                "markdown": "Judo influenced Brazilian Jiu-Jitsu.",
                "raw_text": "Judo influenced Brazilian Jiu-Jitsu.",
                "detected_language": "en",
            }
        ],
        block_rows=[
            {
                "id": "block-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "page_number": 1,
                "kind": "paragraph",
                "text": "Judo influenced Brazilian Jiu-Jitsu.",
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "metadata": {"confidence": 0.99, "layout": {"column": 1}},
            }
        ],
        asset_rows=[
            {
                "id": "asset-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "kind": "figure",
                "page_number": 1,
                "uri": "asset://figure-1",
                "metadata": {"caption": "Technique"},
            }
        ],
        chunks=chunks,
        # This focal entity is deliberately absent from ordinary observations.
        # Spark must retain compacted document context at the final barrier.
        document_context_entities=[
            _mention(
                "entity-bjj",
                "Brazilian Jiu-Jitsu",
                "chunk-1",
                start_offset=16,
            )
        ],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=[
                _mention("entity-judo-1", "Judo", "chunk-1", start_offset=0),
                _mention("entity-judo-2", "judo", "chunk-2", start_offset=9),
                _mention(
                    "entity-elman-srn",
                    "Elman simple recurrent network",
                    "chunk-1",
                    start_offset=first_chunk_text.index("Elman"),
                    entity_type="MODEL",
                    aliases=["SRN"],
                ),
                _mention(
                    "entity-scene-srn",
                    "SRN",
                    "chunk-2",
                    start_offset=second_chunk_text.index("SRN"),
                    entity_type="MODEL",
                    aliases=["Scene Representation Networks"],
                ),
                # The same acronym seen bare: resolution keeps it apart from the
                # spelled-out SRN above, so two nodes prefer the name SRN.
                _mention(
                    "entity-bare-srn",
                    "SRN",
                    "chunk-1",
                    start_offset=first_chunk_text.rindex("SRN"),
                    entity_type="MODEL",
                ),
                _mention(
                    "entity-da",
                    "DA",
                    "chunk-2",
                    start_offset=second_chunk_text.index("DA"),
                    entity_type="PERSON",
                ),
                _mention(
                    "entity-dario",
                    "Dario Amodei",
                    "chunk-2",
                    start_offset=second_chunk_text.index("Dario Amodei"),
                    entity_type="PERSON",
                ),
                _mention(
                    "entity-diogo",
                    "Diogo Almeida",
                    "chunk-2",
                    start_offset=second_chunk_text.index("Diogo Almeida"),
                    entity_type="PERSON",
                ),
            ],
            relations=[
                RelationObservation(
                    id="relation-1",
                    source_entity_id="entity-judo-1",
                    target_entity_id="entity-bjj",
                    relation_type="INFLUENCED",
                    description="Judo influenced Brazilian Jiu-Jitsu.",
                    source_chunk_id="chunk-1",
                    quote="Judo influenced Brazilian Jiu-Jitsu.",
                    start_offset=0,
                    end_offset=len("Judo influenced Brazilian Jiu-Jitsu."),
                )
            ],
            chunk_count=2,
            window_count=1,
            # The counters a worker raises per window. A fleet run has no other
            # record of why records were dropped or repaired.
            trace=[
                {
                    "stage": "entity_extraction",
                    "record_actions": {"ungrounded_quote": 2, "duplicate": 1},
                },
                {
                    "stage": "document_context_extraction",
                    "record_actions": {"ungrounded_quote": 3},
                },
                {
                    "stage": "relation_extraction",
                    "record_actions": {"domain_or_range_violation": 4},
                    "rejected_records": [
                        {
                            "kind": "relation",
                            "reason": "domain_or_range_violation",
                            "name": "INFLUENCED",
                            "source": "Judo",
                            "source_type": "MARTIAL_ART",
                            "target": "Kano",
                            "target_type": "PERSON",
                            "quote": "Judo influenced Kano.",
                            "chunk_id": "chunk-1",
                        }
                    ],
                },
                # Verification raises reasons of its own. A local run does not
                # fold them into record_actions, so neither may this one.
                {"stage": "relation_verification", "record_actions": {"invalid_schema": 9}},
                # A window the model returned records for and validation kept
                # none of: the gap the console reports per document.
                {
                    "stage": "entity_extraction",
                    "window_id": "window-2",
                    "document_id": "file-2",
                    "chunk_ids": ["chunk-2"],
                    "input_records": 2,
                    "accepted_records": 0,
                    "record_actions": {"ungrounded_quote": 2},
                    "rejected_records": [
                        {
                            "kind": "entity",
                            "reason": "ungrounded_quote",
                            "name": name,
                            "type": "MARTIAL_ART",
                            "quote": name,
                            "chunk_id": "chunk-2",
                        }
                        for name in ("Jiudo", "Jiu-do")
                    ],
                },
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
    settings = _settings(tmp_path, root, graph_id)
    manifest = SparkGraphFinalizer(settings).finalize(
        SparkFinalizationRequest(run_id=run_id, graph_id=graph_id, attempt=1, artifact_ids=LINKED)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert manifest.tables["nodes"].row_count == 8
    assert manifest.tables["edges"].row_count == 1
    # GraphFrames label propagation may resolve tied labels differently as Spark
    # partition order changes. FlakeGraph's contract is the validity and internal
    # consistency of the resulting communities, not a vendor algorithm's exact
    # label count for this deliberately small, highly connected fixture.
    _assert_community_contract(batch, manifest, settings.graph.min_community_size)
    assert all(item.size_bytes > 0 for table in manifest.tables.values() for item in table.files)
    assert manifest.metrics["dangling_edges"] == 0
    assert manifest.metrics["evidence_coverage"] == 1.0
    assert manifest.metrics["community_co_mention_strategy"] == "bounded_chunk_anchor_star"
    assert manifest.metrics["hard_quality_error_count"] == 0
    # A fleet run has to report the same rejection reasons a local run does, or
    # the runs whose workers are hardest to inspect are the ones that explain
    # nothing. Counts are summed across every stage that grounds records.
    _assert_extraction_contract(manifest.metrics["extraction"])
    assert not any(manifest.metrics["hard_quality_errors"].values())
    _assert_discarded_window_contract(manifest, batch)
    # Every rejected record is a row, the same rows and summary a local run
    # derives from the same trace.
    expected = rejected_records_from_trace(extracted.observations.trace, prepared.chunks, graph_id)
    assert len(expected) == 3
    assert sorted(
        (row.model_dump() for row in batch.rejected_records), key=lambda row: row["id"]
    ) == sorted((row.model_dump() for row in expected), key=lambda row: row["id"])
    assert manifest.metrics["rejected_records"] == rejected_record_metrics(expected)
    _assert_homonyms_keep_their_own_ids(batch)
    _assert_dataset_export_matches_batch_export(tmp_path, manifest, store)
    assert len(batch.edges) == 1
    assert batch.documents[0]["checksum"] == "checksum-1"
    assert batch.pages[0]["raw_text"] == "Judo influenced Brazilian Jiu-Jitsu."
    assert json.loads(batch.blocks[0]["metadata"])["layout"] == {"column": 1}
    assert json.loads(batch.assets[0]["metadata"])["caption"] == "Technique"
    assert batch.edges[0].source_node_id in {node.id for node in batch.nodes}
    assert batch.edges[0].target_node_id in {node.id for node in batch.nodes}
    assert {node.normalized_name for node in batch.nodes} == {
        "judo",
        "brazilianjiujitsu",
        "elmansimplerecurrentnetwork",
        "srn",
        "da",
        "darioamodei",
        "diogoalmeida",
    }
    assert all(node.embedding and len(node.embedding) == 8 for node in batch.nodes)
    assert batch.communities[0].summary.startswith("Community around")
    assert batch.community_findings[0].summary == "Connected entities"
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    for evidence in batch.evidence:
        chunk = chunks_by_id[evidence.chunk_id]
        local_start = evidence.start_offset - chunk.start_offset
        local_end = evidence.end_offset - chunk.start_offset
        assert 0 <= local_start < local_end <= len(chunk.content)
        assert chunk.content[local_start:local_end] == evidence.quote


def test_spark_finalizer_reads_only_the_objects_a_succeeded_task_linked(tmp_path: Path) -> None:
    """An attempt that wrote its shard and then lost its lease leaves an orphan behind.

    Executors read whole prefixes, and the retry does not overwrite the orphan:
    its shard hashes differently. Without the id filter every chunk of that
    document would appear twice.
    """

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        document_rows=[{"id": "document-1", "graph_id": "graph"}],
        chunks=[_chunk("chunk-1", "file-1", "plain lowercase text")],
    )
    orphan = prepared.model_copy(
        update={"chunks": [_chunk("chunk-1", "file-1", "plain lowercase text, seen again")]}
    )
    for name, shard in (("prepared", prepared), ("lost-lease", orphan)):
        store.put(
            f"run/prepared_document/{name}.json",
            shard.model_dump_json().encode(),
            "application/json",
        )
    for name, shard in (("extracted", prepared), ("lost-lease", orphan)):
        extracted = ExtractedDocumentShard(
            prepared=shard, observations=ExtractionObservations(chunk_count=1, window_count=1)
        )
        store.put(
            f"run/extracted_document/{name}.json",
            extracted.model_dump_json().encode(),
            "application/json",
        )

    manifest = SparkGraphFinalizer(_settings(tmp_path, root, "graph")).finalize(
        SparkFinalizationRequest(run_id="run", graph_id="graph", attempt=2, artifact_ids=LINKED)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert manifest.tables["chunks"].row_count == 1
    assert [chunk.content for chunk in batch.chunks] == ["plain lowercase text"]


def test_provider_threads_change_no_published_row(tmp_path: Path) -> None:
    """Four provider threads per task publish the graph one thread per task publishes.

    A chain of entities, each named in two chunks with two descriptions, gives
    description merges, community reports and embeddings several batches per
    partition once threads share the work. Threads also resize the provider
    partitions, so both runs cut different batches; the fake model answers each
    record on its own, which leaves any difference to the threads themselves.
    """

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    graph_id = "threaded-graph"
    names = [f"Kata Form {index:03d}" for index in range(120)]
    chunks: list[Chunk] = []
    mentions: list[EntityMention] = []
    relations: list[RelationObservation] = []
    for index, (source, target) in enumerate(itertools.pairwise(names)):
        text = f"{source} influenced {target}."
        chunk = _chunk(f"chunk-{index:03d}", f"file-{index:03d}", text)
        chunks.append(chunk.model_copy(update={"graph_id": graph_id}))
        # The longer description is the one the fake model keeps when it merges.
        for role, name, offset, description in (
            ("source", source, 0, f"{source} in {chunk.id}."),
            ("target", target, text.index(target), f"{target}, influenced in {chunk.id}."),
        ):
            mention = _mention(f"entity-{index:03d}-{role}", name, chunk.id, start_offset=offset)
            mentions.append(mention.model_copy(update={"description": description}))
        relations.append(
            RelationObservation(
                id=f"relation-{index:03d}",
                source_entity_id=f"entity-{index:03d}-source",
                target_entity_id=f"entity-{index:03d}-target",
                relation_type="INFLUENCED",
                description=text,
                source_chunk_id=chunk.id,
                quote=text,
                start_offset=0,
                end_offset=len(text),
            )
        )
    prepared = PreparedDocumentShard(
        file_ids=[chunk.file_id for chunk in chunks],
        files_seen=len(chunks),
        documents_processed=len(chunks),
        document_rows=[
            {"id": f"document-{chunk.file_id}", "graph_id": graph_id, "file_id": chunk.file_id}
            for chunk in chunks
        ],
        chunks=chunks,
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=mentions,
            relations=relations,
            chunk_count=len(chunks),
            window_count=len(chunks),
        ),
    )
    for run_id in ("one-thread", "four-threads"):
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

    settings = _settings(tmp_path, root, graph_id)
    tables: dict[str, dict[str, Any]] = {}
    # Two execution slots: a concurrency of two is one thread per task, eight is four.
    for run_id, concurrency in (("one-thread", 2), ("four-threads", 8)):
        threaded = settings.model_copy(
            update={
                "graph": settings.graph.model_copy(
                    update={"finalization_provider_concurrency": concurrency}
                )
            }
        )
        manifest = SparkGraphFinalizer(threaded).finalize(
            SparkFinalizationRequest(
                run_id=run_id, graph_id=graph_id, attempt=1, artifact_ids=LINKED
            )
        )
        batch = GraphDatasetReader(store).read(manifest)
        tables[run_id] = {
            "chunks": sorted((chunk.id, chunk.embedding) for chunk in batch.chunks),
            "nodes": sorted(node.model_dump_json() for node in batch.nodes),
            "edges": sorted(edge.model_dump_json() for edge in batch.edges),
            "community_reports": sorted(
                (sorted(community.member_node_ids), community.title, community.summary)
                for community in batch.communities
            ),
        }

    assert len(tables["one-thread"]["nodes"]) == len(names)
    assert sum(", influenced in chunk-" in node for node in tables["one-thread"]["nodes"]) == (
        len(names) - 1
    )
    assert tables["four-threads"] == tables["one-thread"]


@pytest.mark.parametrize("allowed", [True, False])
def test_spark_keeps_only_explicit_self_relations_the_ontology_allows(
    tmp_path: Path, allowed: bool
) -> None:
    """One mention at both ends is a stated loop; two spellings of one node never are.

    The local engine applies the same rule, so one corpus yields one edge set on
    either engine.
    """

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    graph_id = "loop-graph"
    text = "Kinase Alpha activates itself. KINASE ALPHA activates Kinase Alpha."
    chunk = _chunk("chunk-loop", "file-loop", text).model_copy(update={"graph_id": graph_id})
    second = text.index("KINASE ALPHA")
    last = text.rindex("Kinase Alpha")
    mentions = [
        _mention("entity-first", "Kinase Alpha", chunk.id, start_offset=0, entity_type="PROTEIN"),
        _mention(
            "entity-upper", "KINASE ALPHA", chunk.id, start_offset=second, entity_type="PROTEIN"
        ),
        _mention("entity-last", "Kinase Alpha", chunk.id, start_offset=last, entity_type="PROTEIN"),
    ]
    sentences = text.split(". ")

    def relation(relation_id: str, source: str, target: str, quote: str) -> RelationObservation:
        start = text.index(quote)
        return RelationObservation(
            id=relation_id,
            source_entity_id=source,
            target_entity_id=target,
            relation_type="ACTIVATES",
            description=quote,
            source_chunk_id=chunk.id,
            quote=quote,
            start_offset=start,
            end_offset=start + len(quote),
        )

    relations = [
        relation("relation-explicit", "entity-first", "entity-first", sentences[0]),
        relation("relation-merged", "entity-upper", "entity-last", sentences[1]),
    ]
    prepared = PreparedDocumentShard(
        file_ids=[chunk.file_id],
        files_seen=1,
        documents_processed=1,
        document_rows=[{"id": "document-loop", "graph_id": graph_id, "file_id": chunk.file_id}],
        chunks=[chunk],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=mentions, relations=relations, chunk_count=1, window_count=1
        ),
    )
    store.put(
        "loop-run/prepared_document/prepared.json",
        prepared.model_dump_json().encode(),
        "application/json",
    )
    store.put(
        "loop-run/extracted_document/extracted.json",
        extracted.model_dump_json().encode(),
        "application/json",
    )
    ontology = OntologyProfile(
        name="self-regulation",
        description="Protein self regulation",
        mode="closed",
        entity_types=[EntityTypeDefinition(name="PROTEIN", description="Named protein")],
        relation_types=[
            RelationTypeDefinition(
                name="ACTIVATES",
                description="A protein activates a protein",
                source_types=["PROTEIN"],
                target_types=["PROTEIN"],
                allow_self_loop=allowed,
            )
        ],
    )
    settings = _settings(tmp_path, root, graph_id)
    settings = settings.model_copy(
        update={
            "ontology": settings.ontology.model_copy(
                update={"profile": ontology.model_dump(mode="json")}
            )
        }
    )
    manifest = SparkGraphFinalizer(settings).finalize(
        SparkFinalizationRequest(
            run_id="loop-run", graph_id=graph_id, attempt=1, artifact_ids=LINKED
        )
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert len(batch.nodes) == 1
    # Only the stated loop survives, and only where the ontology allows it.
    assert [observation.description for observation in batch.edge_observations] == (
        [sentences[0]] if allowed else []
    )
    assert len(batch.edges) == (1 if allowed else 0)
    if allowed:
        (edge,) = batch.edges
        assert edge.source_node_id == edge.target_node_id == batch.nodes[0].id


def test_spark_finalizer_publishes_schema_correct_empty_graph(tmp_path: Path) -> None:
    """A corpus with no accepted mentions must still publish every logical table."""

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    prepared = PreparedDocumentShard(
        file_ids=["file-empty"],
        files_seen=1,
        documents_processed=1,
        document_rows=[
            {"id": "document-empty", "graph_id": "empty-graph", "file_id": "file-empty"}
        ],
        chunks=[_chunk("chunk-empty", "file-empty", "plain lowercase text")],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(chunk_count=1, window_count=1),
    )
    # Shaped as a revision that extracted the document again: its prepared
    # rows stay under the run that prepared it.
    store.put(
        "base-run/prepared_document/prepared.json",
        prepared.model_dump_json().encode(),
        "application/json",
    )
    store.put(
        "empty-run/extracted_document/extracted.json",
        extracted.model_dump_json().encode(),
        "application/json",
    )
    extraction_usage = ConsumptionEvent(
        graph_id="empty-graph",
        job_id="empty-run",
        run_id="empty-run",
        stage="entity_extraction",
        operation="completion",
        provider="fake",
        model="fake",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=5),
    )
    executor_usage = extraction_usage.model_copy(
        update={"event_id": "executor-event", "stage": "description_merge"}
    )
    store.put(
        "empty-run/consumption/executor-event.json",
        executor_usage.model_dump_json().encode(),
        "application/json",
    )
    manifest = SparkGraphFinalizer(_settings(tmp_path, root, "empty-graph")).finalize(
        SparkFinalizationRequest(
            run_id="empty-run",
            graph_id="empty-graph",
            attempt=1,
            artifact_ids=LINKED,
            prepared_run_ids=frozenset({"base-run"}),
            extracted_run_ids=frozenset({"empty-run"}),
            # A revision says how it came by each document it did not simply extract.
            document_provenance={"file-empty": "reextracted:relations"},
            consumption_events=(extraction_usage, extraction_usage),
        )
    )
    batch = GraphDatasetReader(store).read(manifest)
    assert [row["provenance"] for row in batch.documents] == ["reextracted:relations"]

    assert set(manifest.tables) == {
        "documents",
        "pages",
        "blocks",
        "assets",
        "chunks",
        "nodes",
        "edges",
        "edge_observations",
        "evidence",
        "entity_sources",
        "communities",
        "community_findings",
        "discarded_windows",
        "rejected_records",
        "failed_documents",
    }
    assert manifest.metrics["consumption"]["totals"]["total_tokens"] == 50
    assert len(manifest.metrics["consumption"]["events"]) == 2
    assert {row["job_id"] for row in manifest.metrics["consumption"]["events"]} == {"empty-run"}
    assert manifest.metrics["input_entity_mentions"] == 0
    assert manifest.metrics["discarded_windows"]["windows"] == 0
    assert batch.nodes == []
    assert batch.edges == []
    assert batch.discarded_windows == []


def _settings(tmp_path: Path, root: Path, graph_id: str) -> Settings:
    """Configure a local two-executor Spark finalization over ``root``."""

    return Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {"graph_id": graph_id},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
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


def _chunk(
    chunk_id: str,
    file_id: str,
    content: str,
    *,
    start_offset: int = 0,
) -> Chunk:
    """Build a valid source chunk with offsets used by evidence joins."""

    return Chunk(
        id=chunk_id,
        graph_id="spark-graph",
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


def _assert_community_contract(
    batch: GraphWriteBatch,
    manifest: GraphDatasetManifest,
    min_community_size: int,
) -> None:
    """Validate FlakeGraph's output contract without fixing GraphFrames labels."""

    community_count = manifest.tables["communities"].row_count
    assert community_count > 0
    assert manifest.tables["community_findings"].row_count == community_count
    assert len(batch.communities) == community_count
    assert len(batch.community_findings) == community_count
    node_ids = {node.id for node in batch.nodes}
    community_ids = {community.id for community in batch.communities}
    assert all(
        len(community.member_node_ids) >= min_community_size
        and set(community.member_node_ids) <= node_ids
        for community in batch.communities
    )
    assert {finding.community_id for finding in batch.community_findings} == community_ids


def _mention(
    entity_id: str,
    name: str,
    chunk_id: str,
    *,
    start_offset: int,
    entity_type: str = "MARTIAL_ART",
    aliases: list[str] | None = None,
) -> EntityMention:
    """Build one grounded mention for identity-resolution coverage."""

    return EntityMention(
        id=entity_id,
        name=name,
        type=entity_type,
        description=f"A mention of {name}.",
        source_chunk_id=chunk_id,
        quote=name,
        aliases=aliases or [],
        start_offset=start_offset,
        end_offset=start_offset + len(name),
    )


def test_two_runs_finalize_in_one_worker_process(tmp_path: Path) -> None:
    """Sequential finalizations publish both graphs and retain one live session.

    Finalizations in one worker are serialized and share a process-scoped Spark
    session until worker shutdown. Completing a run leaves that session alive
    for the next task.
    """

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    graph_id = "sequential-graph"
    for run_id in ("first-run", "second-run"):
        prepared = PreparedDocumentShard(
            file_ids=[f"file-{run_id}"],
            files_seen=1,
            documents_processed=1,
            document_rows=[{"id": f"document-{run_id}", "graph_id": graph_id}],
            chunks=[
                _chunk(f"chunk-{run_id}", f"file-{run_id}", "Judo influenced Brazilian Jiu-Jitsu.")
            ],
        )
        store.put(
            f"{run_id}/prepared_document/prepared.json",
            prepared.model_dump_json().encode(),
            "application/json",
        )
        store.put(
            f"{run_id}/extracted_document/extracted.json",
            ExtractedDocumentShard(
                prepared=prepared,
                observations=ExtractionObservations(chunk_count=1, window_count=1),
            )
            .model_dump_json()
            .encode(),
            "application/json",
        )

    settings = _settings(tmp_path, root, graph_id)
    finalizer = SparkGraphFinalizer(settings)
    sessions = []
    for run_id in ("first-run", "second-run"):
        manifest = finalizer.finalize(
            SparkFinalizationRequest(
                run_id=run_id, graph_id=graph_id, attempt=1, artifact_ids=LINKED
            )
        )
        assert manifest.run_id == run_id
        assert "documents" in manifest.tables
        sessions.append(SparkSession.getActiveSession())

    # Both completed runs share the same live process-scoped context.
    assert sessions[0] is sessions[1]
    assert sessions[1] is not None
    assert not _is_stopped(sessions[1])
