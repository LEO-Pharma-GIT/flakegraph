# SPDX-License-Identifier: Apache-2.0
"""Spark-backed graph finalization over immutable distributed stage artifacts.

The module imports PySpark lazily so ordinary local users do not pay its large
dependency or JVM startup cost. Kubernetes workers select this engine only when
shared object storage is configured. All graph-sized state remains in DataFrames;
the driver collects counts and file manifests, never entity or edge tables.
"""

# PySpark is an optional 455 MB dependency. Imports remain inside the scalable
# execution path so the normal local installation stays lightweight.
# ruff: noqa: PLC0415

from __future__ import annotations

import atexit
import hashlib
import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import batched
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from kg_processor.application.community_reports import (
    CLAIMED_RELATION_MARK,
    DEFAULT_MAX_RELATIONS_PER_COMMUNITY,
    MAX_EVIDENCE_QUOTES_PER_COMMUNITY,
    MAX_EVIDENCE_QUOTES_PER_RELATION,
    structural_rating,
)
from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.application.content_kinds import DATASET_PROFILE_STAGE, annotate_documents
from kg_processor.application.discarded_windows import (
    DISCARDED_WINDOW_STAGES,
    PREVIEW_CHARACTERS,
)
from kg_processor.application.enrichment_batching import (
    COMMUNITY_BATCH_SIZE,
    DESCRIPTION_BATCH_SIZE,
)
from kg_processor.application.entity_resolution import (
    MAX_SHORT_INITIALISM_LENGTH,
    MIN_EMBEDDING_FOR_LEXICAL_MERGE,
    MIN_SHORT_INITIALISM_LENGTH,
    UnionFind,
)
from kg_processor.application.failed_documents import FAILED_STATUS, PARSE_STAGE
from kg_processor.application.folder_entities import (
    FOLDER_EVIDENCE_METHOD,
    FOLDER_LINK_DESCRIPTION,
    excluded_file_ids,
)
from kg_processor.application.graph_merge import (
    normalize_entity_name,
    normalize_relation_type,
    structural_relation_types,
)
from kg_processor.application.ordered_map import bounded_map
from kg_processor.application.redaction import is_sensitive_key
from kg_processor.application.rejected_records import (
    MAX_REPORTED_TYPE_RULES,
    REJECTED_RECORD_FIELDS,
    REJECTING_STAGES,
)
from kg_processor.application.type_resolution import (
    TYPE_RESOLVED_METHOD,
    type_resolution_metrics,
    type_resolution_rules,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.consumption import ConsumptionEvent
from kg_processor.domain.distributed import TaskProgress
from kg_processor.domain.finalization import (
    DatasetFile,
    DatasetTableManifest,
    GraphDatasetManifest,
)

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame, SparkSession, WindowSpec

    from kg_processor.domain.ontology import OntologyProfile

_GRAPHFRAMES_COORDINATE = "io.graphframes:graphframes-spark4_2.13:0.12.1"
_TARGET_ROWS_PER_PARTITION = 100_000
_INPUT_PARTITIONS_PER_SLOT = 8
_MIN_PROVIDER_PARTITIONS_PER_SLOT = 2
_MAX_PROVIDER_PARTITIONS_PER_SLOT = 256
# Batches submitted ahead of the one a partition yields next. Threads that
# finish early start on these instead of idling behind a slow request, and a
# partition never holds more rows than these batches and their results.
_QUEUED_PROVIDER_BATCHES_PER_THREAD = 2
# Collecting is permitted only below these intermediate-graph bounds. The choice
# depends on actual resolution rows, never document count; larger corpora retain
# GraphFrames connected components with no user-supplied size limit.
_LOCAL_COMPONENT_MAX_VERTICES = 100_000
_LOCAL_COMPONENT_MAX_EDGES = 500_000
_MAX_DENORMALIZED_PROVENANCE_IDS = 1_000
_MAX_NODE_SURFACES = 1_000
_MAX_IDENTITY_BLOCK_TOKENS = 6
# Extraction derives a relation observation's weight from its confidence with this
# floor, so canonical edge weight stays comparable across engines that read stage
# artifacts, which persist confidence rather than the derived weight.
_MIN_OBSERVATION_WEIGHT = 0.1
_COMMUNITY_ITERATIONS = 10
_SPARK_APPLICATION_DIGEST_LENGTH = 16
_EXECUTOR_PROVIDER_LOCK = Lock()
_EXECUTOR_EMBEDDING_PROVIDERS: dict[str, Any] = {}
_EXECUTOR_LLM_PROVIDERS: dict[str, Any] = {}
_EXECUTOR_ENRICHMENT_PROVIDERS: dict[str, Any] = {}
_EXECUTOR_PIPELINE_CACHES: dict[str, Any] = {}
_EXECUTOR_PROVIDER_POOLS: dict[int, ThreadPoolExecutor] = {}
_MENTION_FIELDS = (
    "id",
    "name",
    "type",
    "description",
    "source_chunk_id",
    "quote",
    "confidence",
    "aliases",
    "start_offset",
    "end_offset",
)
_TABLE_NAMES = (
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
)
FINALIZATION_PHASES: tuple[tuple[str, str], ...] = (
    ("start_engine", "Starting Spark engine"),
    ("read_artifacts", "Reading document graph shards"),
    ("materialize_inputs", "Materializing extracted observations"),
    ("embed_chunks", "Embedding source chunks"),
    ("resolve_entities", "Resolving canonical entity identities"),
    ("assemble_graph", "Assembling nodes, edges, and evidence"),
    ("enrich_graph", "Merging descriptions and graph embeddings"),
    ("build_communities", "Detecting and summarizing communities"),
    ("quality_checks", "Running graph quality checks"),
    ("write_tables", "Writing partitioned graph tables"),
    ("publish_destination", "Publishing the graph destination"),
    ("persist_manifest", "Persisting the graph manifest"),
)
_FINALIZATION_PHASE_INDEX = {
    phase: index for index, (phase, _label) in enumerate(FINALIZATION_PHASES, start=1)
}


def finalization_progress(
    phase: str,
    *,
    completed: int,
    total: int,
    message: str | None = None,
) -> TaskProgress:
    """Build one validated progress record from the shared finalization phase order."""

    phase_index = _FINALIZATION_PHASE_INDEX[phase]
    default_message = FINALIZATION_PHASES[phase_index - 1][1]
    return TaskProgress(
        phase=phase,
        phase_index=phase_index,
        phase_total=len(FINALIZATION_PHASES),
        completed=completed,
        total=total,
        message=message or default_message,
    )


_TARGET_OUTPUT_ROWS_PER_FILE = {
    "documents": 250_000,
    "pages": 5_000,
    "blocks": 250_000,
    "assets": 100_000,
    "chunks": 25_000,
    "nodes": 25_000,
    "edges": 100_000,
    "edge_observations": 100_000,
    "evidence": 100_000,
    "entity_sources": 100_000,
    "communities": 25_000,
    "community_findings": 50_000,
    # A preview per row, so it packs like evidence.
    "discarded_windows": 100_000,
    # A quote per row, so it packs like evidence.
    "rejected_records": 100_000,
    "failed_documents": 100_000,
}
_EMPTY_SCHEMAS = {
    "documents": (
        "id string, graph_id string, file_id string, checksum string, source_uri string, "
        "mime_type string, size_bytes long, ocr_provider string, kind string, summary string, "
        "related_file_ids array<string>, folder string, provenance string"
    ),
    "pages": (
        "id string, graph_id string, file_id string, page_number int, markdown string, "
        "raw_text string, detected_language string"
    ),
    "blocks": (
        "id string, graph_id string, file_id string, page_number int, kind string, "
        "text string, bbox array<double>, metadata string"
    ),
    "assets": (
        "id string, graph_id string, file_id string, kind string, page_number int, "
        "uri string, metadata string"
    ),
    "chunks": (
        "id string, graph_id string, file_id string, document_id string, page_number int, "
        "chunk_index int, content string, start_offset int, end_offset int, token_count int, "
        "content_hash string, section_path array<string>, block_ids array<string>, "
        "asset_ids array<string>, ocr_generation_id string, embedding array<double>"
    ),
    "entities": (
        "id string, name string, type string, description string, source_chunk_id string, "
        "quote string, confidence double, aliases array<string>, start_offset int, end_offset int, "
        "is_document_context boolean, contextual_surfaces array<string>, evidence_method string"
    ),
    "relations": (
        "id string, source_entity_id string, target_entity_id string, relation_type string, "
        "description string, source_chunk_id string, quote string, confidence double, "
        "start_offset int, end_offset int, source_surface string, target_surface string, "
        "evidence_method string, verification_required boolean, "
        "quantities array<struct<text:string, kind:string, comparator:string, unit:string, "
        "low:double, high:double>>, proposed_relation_type string, rejected_record_key string, "
        "assertion string"
    ),
    "nodes": (
        "id string, graph_id string, normalized_name string, name string, "
        "primary_type string, types array<string>, aliases array<string>, description string, "
        "embedding array<double>, source_chunk_ids array<string>, degree int, rank double"
    ),
    "edges": (
        "id string, graph_id string, source_node_id string, target_node_id string, "
        "relation_type string, description string, weight double, confidence double, "
        "source_file_id string, source_file_ids array<string>, "
        "source_chunk_ids array<string>, evidence_count long, assertion string, "
        "embedding array<double>"
    ),
    "edge_observations": (
        "id string, graph_id string, edge_id string, file_id string, chunk_id string, "
        "description string, weight double, confidence double, evidence_id string, "
        "quantities array<struct<text:string, kind:string, comparator:string, unit:string, "
        "low:double, high:double>>, proposed_relation_type string, assertion string"
    ),
    "evidence": (
        "id string, graph_id string, subject_id string, subject_kind string, file_id string, "
        "chunk_id string, page_number int, start_offset int, end_offset int, quote string, "
        "method string"
    ),
    "entity_sources": (
        "id string, graph_id string, node_id string, file_id string, "
        "per_file_description string, mention_count long, aliases array<string>"
    ),
    "communities": (
        "id string, graph_id string, stable_key string, level int, title string, "
        "summary string, rating double, rating_explanation string, "
        "member_node_ids array<string>, parent_community_id string, "
        "child_community_ids array<string>, suggested_questions array<string>, "
        "embedding array<double>"
    ),
    "community_findings": (
        "id string, graph_id string, community_id string, summary string, explanation string"
    ),
    "discarded_windows": (
        "id string, graph_id string, document_id string, file_id string, stage string, "
        "window_id string, chunk_ids array<string>, page_start int, page_end int, "
        "extracted_records long, record_actions map<string,bigint>, preview string"
    ),
    "rejected_records": (
        "id string, graph_id string, document_id string, file_id string, window_id string, "
        "chunk_id string, page_number int, kind string, reason string, name string, "
        "type string, source string, source_type string, target string, target_type string, "
        "quote string"
    ),
    "failed_documents": (
        "id string, graph_id string, document_id string, file_id string, source_uri string, "
        "mime_type string, size_bytes long, provider string, error_type string, "
        "error_message string"
    ),
}
# Distributed stage artifacts have stable typed projections. Supplying schemas
# avoids Spark launching one schema-inference task per immutable document object,
# which is especially expensive for object stores and large corpora. Provider-
# defined OCR metadata is read as its complete compact JSON string rather than a
# shape inferred from whichever document Spark happens to inspect first.
_PREPARED_STAGE_SCHEMA = (
    f"document_rows array<struct<{_EMPTY_SCHEMAS['documents']}>>, "
    f"page_rows array<struct<{_EMPTY_SCHEMAS['pages']}>>, "
    f"block_rows array<struct<{_EMPTY_SCHEMAS['blocks']}>>, "
    f"asset_rows array<struct<{_EMPTY_SCHEMAS['assets']}>>, "
    f"chunks array<struct<{_EMPTY_SCHEMAS['chunks']}>>, "
    f"document_context_entities array<struct<{_EMPTY_SCHEMAS['entities']}>>"
)
# The values a relation states travel with each edge observation.
_EDGE_OBSERVATION_QUANTITIES_TYPE = (
    "array<struct<text:string, kind:string, comparator:string, unit:string, "
    "low:double, high:double>>"
)
# Only the fields the graph tables and the run report actually read are parsed.
# The trace is projected down to the stage that raised each event and its
# rejection and repair counters, so a run reports why records were dropped
# without every executor deserializing full window payloads.
_TRACE_SCHEMA = (
    "array<struct<stage:string, window_id:string, document_id:string, "
    "chunk_ids:array<string>, input_records:bigint, accepted_records:bigint, "
    "supported_records:bigint, skipped:string, kind:string, summary:string, "
    "record_actions:map<string,bigint>, rejected_records:array<struct<kind:string, "
    "reason:string, name:string, type:string, source:string, source_type:string, "
    "target:string, target_type:string, quote:string, chunk_id:string>>>>"
)
# The stages whose rejection and repair counters a run report sums, kept the same
# on both engines so one corpus reports one number wherever it was processed.
_RECORD_ACTION_STAGES = (
    "entity_extraction",
    "document_context_extraction",
    "relation_extraction",
)
_EXTRACTED_STAGE_SCHEMA = (
    "prepared struct<"
    f"document_context_entities:array<struct<{_EMPTY_SCHEMAS['entities']}>>>, "
    "observations struct<"
    f"entities:array<struct<{_EMPTY_SCHEMAS['entities']}>>, "
    f"relations:array<struct<{_EMPTY_SCHEMAS['relations']}>>, "
    f"held_relations:array<struct<{_EMPTY_SCHEMAS['relations']}>>, "
    "chunk_count:bigint, window_count:bigint, "
    f"trace:{_TRACE_SCHEMA}>, "
    # The artifact's own processing trace, read for a parse that failed, which
    # is the only record a document that could not be read has.
    "trace array<struct<stage:string, status:string, file_id:string, source_uri:string, "
    "mime_type:string, size_bytes:bigint, provider:string, error_type:string, "
    "error_message:string>>"
)


@dataclass(frozen=True)
class SparkFinalizationRequest:
    """Identify one immutable run attempt and the stage objects it may read.

    ``artifact_ids`` names the objects under the run's prefixes that a
    succeeded task produced. The prefixes also hold what an attempt wrote
    before losing its lease, and executors read prefixes, so the ids are what
    keeps a retry's orphan out of the graph.
    """

    run_id: str
    graph_id: str
    attempt: int
    artifact_ids: frozenset[str]
    # The runs whose prefixes hold those objects, per stage: the run itself,
    # and any earlier run of the graph a revision keeps documents from. Only
    # runs with linked objects of that stage are listed, because an empty
    # prefix is not a path Spark will read - a revision that extracts a
    # document again reads its prepared rows from the run that prepared it.
    prepared_run_ids: frozenset[str] = field(default_factory=frozenset)
    extracted_run_ids: frozenset[str] = field(default_factory=frozenset)
    # Each document's provenance label where it is not simply extracted by
    # this run - reused, revalidated, or extracted again - by file id.
    document_provenance: Mapping[str, str] = field(default_factory=dict)
    consumption_events: tuple[ConsumptionEvent, ...] = ()

    def stage_run_ids(self, sources: frozenset[str]) -> list[str]:
        """The run prefixes to read for one stage, the run's own first."""

        others = sorted(sources - {self.run_id})
        own = [self.run_id] if self.run_id in sources or not others else []
        return [*own, *others]


def _failed_document_metrics(rows: DataFrame) -> dict[str, Any]:
    """The same summary the local run report carries, computed in Spark."""

    from pyspark.sql import functions as F

    by_error = {
        str(row["error_type"]): int(row["documents"])
        for row in rows.groupBy(
            F.coalesce(F.nullif(F.col("error_type"), F.lit("")), F.lit("unknown")).alias(
                "error_type"
            )
        )
        .agg(F.count(F.lit(1)).alias("documents"))
        .collect()
    }
    return {"documents": sum(by_error.values()), "by_error_type": dict(sorted(by_error.items()))}


def _rejected_record_metrics(rows: DataFrame) -> dict[str, Any]:
    """The same summary the local run report carries, computed in Spark."""

    from pyspark.sql import functions as F

    totals = rows.agg(
        F.count(F.lit(1)).alias("records"),
        F.countDistinct("document_id").alias("documents"),
    ).first()
    by_kind = {
        str(row["kind"]): int(row["n"])
        for row in rows.groupBy("kind").agg(F.count(F.lit(1)).alias("n")).collect()
    }
    by_reason = sorted(
        (
            (f"{row['kind']}:{row['reason']}", int(row["n"]))
            for row in rows.groupBy("kind", "reason").agg(F.count(F.lit(1)).alias("n")).collect()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    rules = (
        rows.where((F.col("kind") == "relation") & (F.col("reason") == "domain_or_range_violation"))
        .groupBy("source_type", "name", "target_type")
        .agg(F.count(F.lit(1)).alias("n"))
        .orderBy(F.desc("n"), "source_type", "name", "target_type")
        .limit(MAX_REPORTED_TYPE_RULES)
        .collect()
    )
    return {
        "records": int(totals["records"] or 0) if totals else 0,
        "documents": int(totals["documents"] or 0) if totals else 0,
        "by_kind": dict(sorted(by_kind.items())),
        "by_reason": dict(by_reason),
        "broken_type_rules": [
            {
                "source_type": str(row["source_type"]),
                "relation_type": str(row["name"]),
                "target_type": str(row["target_type"]),
                "count": int(row["n"]),
            }
            for row in rules
        ],
    }


def _discarded_window_metrics(rows: DataFrame) -> dict[str, Any]:
    """The same summary the local run report carries, computed in Spark."""

    from pyspark.sql import functions as F

    totals = rows.agg(
        F.count(F.lit(1)).alias("windows"),
        F.countDistinct("document_id").alias("documents"),
    ).first()
    by_stage = {
        str(row["stage"]): int(row["windows"])
        for row in rows.groupBy("stage").agg(F.count(F.lit(1)).alias("windows")).collect()
    }
    reasons = {
        str(row["reason"]): int(row["total"])
        for row in rows.select(F.explode_outer("record_actions").alias("reason", "count"))
        .where(F.col("reason").isNotNull())
        .groupBy("reason")
        .agg(F.sum("count").alias("total"))
        .collect()
    }
    return {
        "windows": int(totals["windows"] or 0) if totals else 0,
        "documents": int(totals["documents"] or 0) if totals else 0,
        "by_stage": dict(sorted(by_stage.items())),
        "record_actions": dict(sorted(reasons.items())),
    }


# Spark permits one context per JVM. Both guards below exist because a worker
# process is long-lived and claims one finalize task after another.
_FINALIZATION_LOCK = Lock()


def _is_stopped(session: Any) -> bool:
    """Whether a session's context has been stopped and cannot serve work.

    Reads the JVM handle defensively: a session whose gateway has gone is as
    unusable as one explicitly stopped, and either way the answer is to build
    a fresh one.
    """

    try:
        return bool(session.sparkContext._jsc.sc().isStopped())
    except Exception:  # noqa: BLE001 - any failure to ask means it is unusable
        return True


def stop_process_session() -> None:
    """Stop this process's Spark context, if it has one. Registered at exit."""

    try:
        from pyspark.sql import SparkSession

        live = SparkSession.getActiveSession()
        if live is not None and not _is_stopped(live):
            live.stop()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        return


atexit.register(stop_process_session)


def _spark_application_name(request: SparkFinalizationRequest) -> str:
    """Return a DNS-safe Kubernetes identity unique to one durable attempt.

    Run IDs are user-controlled and can be long or contain characters Kubernetes
    rejects. A fixed hexadecimal digest keeps executor pods and ConfigMaps within
    platform limits while changing whenever either the run or attempt changes.
    """

    identity = f"{request.run_id}\0{request.attempt}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:_SPARK_APPLICATION_DIGEST_LENGTH]
    return f"flakegraph-{digest}"


def _with_provider_environment(builder: Any, distributed: Any) -> Any:
    """Pass provider settings to executors, keeping credentials as references.

    Spark renders ``executorEnv`` values as literal environment entries in every
    executor Pod spec, so a credential the driver received as a secret reference
    would leave the Secret and become readable by anything that can read pods in
    the namespace — including the executors themselves. Credentials are therefore
    forwarded as secret references, and not forwarded at all when no Secret is
    configured to hold them.
    """

    for name, value in os.environ.items():
        if not name.startswith(("KG_LLM_", "KG_EMBED_", "KG_EMBEDDING_")):
            continue
        if not is_sensitive_key(name):
            builder = builder.config(f"spark.executorEnv.{name}", value)
        elif reference := distributed.spark_provider_secret_refs.get(name):
            builder = builder.config(f"spark.kubernetes.executor.secretKeyRef.{name}", reference)
    return builder


class SparkGraphFinalizer:
    """Build canonical partitioned graph tables using Spark and GraphFrames.

    Provider-heavy extraction has already completed before this engine runs. The
    engine performs distributed flattening, bounded identity candidate generation,
    connected components, endpoint joins, edge aggregation, community detection,
    and table publication. Every table is written before its small manifest is
    returned to the leased coordinator.
    """

    def __init__(
        self,
        settings: Settings,
        progress_callback: Callable[[TaskProgress], None] | None = None,
    ) -> None:
        """Retain graph policy and an optional bounded task-progress callback."""

        if not settings.distributed.artifact_uri:
            raise ValueError("Spark finalization requires distributed.artifact_uri")
        self.settings = settings
        self.progress_callback = progress_callback

    def finalize(self, request: SparkFinalizationRequest) -> GraphDatasetManifest:
        """Execute all distributed graph phases and return a constant-size manifest.

        Serialized per process: Spark allows one context per JVM, so two
        finalizations in one worker cannot run side by side. Without this lock
        the second one's session creation fails on SPARK-2243 and its cleanup
        tears down the context the first one is still computing in.
        """

        with _FINALIZATION_LOCK:
            return self._finalize_exclusively(request)

    def _finalize_exclusively(self, request: SparkFinalizationRequest) -> GraphDatasetManifest:
        """Run one finalization, holding the process's only Spark context."""

        self.settings = self.settings.model_copy(
            update={
                "job": self.settings.job.model_copy(
                    update={"job_id": request.run_id, "graph_id": request.graph_id}
                )
            }
        )
        self._report_progress("start_engine", completed=0, total=1)
        spark = self._spark_session(request)
        self._report_progress("start_engine", completed=1, total=1)
        timings: dict[str, float] = {}
        prepared: DataFrame | None = None
        extracted: DataFrame | None = None
        checkpoint_path = ""
        try:
            checkpoint_root = _spark_uri(self.settings.distributed.artifact_uri or "")
            checkpoint_path = (
                f"{checkpoint_root}/{request.run_id}/checkpoints/attempt-{request.attempt:04d}"
            )
            spark.sparkContext.setCheckpointDir(checkpoint_path)
            started = perf_counter()
            self._report_progress("read_artifacts", completed=0, total=1)
            prepared, extracted = self._read_stage_artifacts(spark, request)
            self._report_progress("read_artifacts", completed=1, total=1)
            tables, metrics = self._build_graph_tables(spark, prepared, extracted, request)
            timings["distributed_graph_build"] = perf_counter() - started

            write_started = perf_counter()
            manifests = self._write_tables(spark, tables, request)
            metrics["output_rows"] = {name: table.row_count for name, table in manifests.items()}
            collector = ConsumptionCollector(
                request.graph_id, job_id=request.run_id, run_id=request.run_id
            )
            collector.merge(request.consumption_events)
            collector.merge(_executor_consumption_events(spark, self.settings))
            consumption = collector.report(self.settings.consumption.rate_card(), event_limit=100)
            consumption["event_ledger"] = {
                "task_store_run_id": request.run_id,
                "executor_prefix": (
                    f"{(self.settings.distributed.artifact_uri or '').rstrip('/')}/"
                    f"{request.run_id}/consumption/"
                ),
            }
            metrics["consumption"] = consumption
            timings["partitioned_table_write"] = perf_counter() - write_started
            timings["total"] = perf_counter() - started
            return GraphDatasetManifest(
                run_id=request.run_id,
                graph_id=request.graph_id,
                engine=f"apache-spark-{spark.version}/graphframes-0.12.1",
                tables=manifests,
                metrics=metrics,
                timings_seconds=timings,
                relation_weight_max=self.settings.graph.relation_weight_max,
            )
        finally:
            # Stage inputs are cached only for the lifetime of this attempt.
            # Explicit release keeps long-lived local Spark sessions and executor
            # storage tidy. Checkpoints are attempt-local recomputation aids, not
            # published artifacts, so remove their object-store prefix before the
            # Spark context and its configured credentials are torn down.
            if prepared is not None:
                prepared.unpersist(blocking=False)
            if extracted is not None:
                extracted.unpersist(blocking=False)
            if checkpoint_path:
                _delete_hadoop_prefix(spark, checkpoint_path)
            # The context is the process's, not this attempt's. Stopping it here
            # would end any other finalization sharing the JVM, and would throw
            # away warm executors the next run on this worker can reuse: on the
            # fleet that is 30-60 seconds of executor startup per graph. The
            # process stops it at exit instead.

    def _report_progress(
        self,
        phase: str,
        *,
        completed: int,
        total: int,
        message: str | None = None,
    ) -> None:
        """Publish one constant-size finalization update when a reporter is configured."""

        if self.progress_callback is None:
            return
        self.progress_callback(
            finalization_progress(
                phase,
                completed=completed,
                total=total,
                message=message,
            )
        )

    def _spark_session(self, request: SparkFinalizationRequest) -> SparkSession:
        """Return this process's Spark session, building it on first use.

        ``getOrCreate`` returns whatever session the JVM already holds and
        silently ignores the configuration asked for, so a session built for an
        earlier run would be reused without saying so. Here that reuse is the
        intent and is explicit; only a stopped session is rebuilt. Everything
        this builder configures is a property of the worker process - its
        master, its driver pod, its object-store credentials - and not of the
        run being finalized, so one session serves every run the worker claims.
        """

        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise RuntimeError(
                "Spark finalization requires `uv sync --extra distributed-spark`"
            ) from exc

        live = SparkSession.getActiveSession()
        if live is not None and not _is_stopped(live):
            return live

        distributed = self.settings.distributed
        application_name = _spark_application_name(request)
        builder = (
            SparkSession.builder.appName(application_name)
            .master(distributed.spark_master)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.speculation", "false")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            .config(
                "spark.redaction.regex",
                "(?i)secret|password|token|access[.]?key|api[._-]?key|credential",
            )
            # Stage artifacts are one immutable object per document. Bound file
            # scan tasks by actual execution slots so hundreds of thousands of
            # small JSON objects do not become hundreds of thousands of tasks.
            .config(
                "spark.sql.files.maxPartitionNum",
                max(
                    distributed.spark_executor_instances
                    * distributed.spark_executor_cores
                    * _INPUT_PARTITIONS_PER_SLOT,
                    1,
                ),
            )
            .config("spark.executor.instances", distributed.spark_executor_instances)
            .config("spark.executor.cores", distributed.spark_executor_cores)
            .config("spark.executor.memory", distributed.spark_executor_memory)
            # Python provider clients run outside the JVM heap. Reserving process
            # overhead explicitly prevents Kubernetes from killing an executor
            # when all of its task slots are making provider calls concurrently.
            .config(
                "spark.executor.memoryOverhead",
                distributed.spark_executor_memory_overhead,
            )
        )
        if not os.environ.get("FLAKEGRAPH_SPARK_BAKED_JARS"):
            builder = builder.config("spark.jars.packages", _GRAPHFRAMES_COORDINATE)
        builder = _with_provider_environment(builder, distributed)
        if distributed.spark_shuffle_partitions:
            builder = builder.config(
                "spark.sql.shuffle.partitions",
                distributed.spark_shuffle_partitions,
            )
        if distributed.spark_master.startswith("local"):
            # Laptop hostnames can resolve to VPN, Wi-Fi, or IPv6 interfaces that
            # are not valid loopback routes for Spark's local RPC server. Local
            # executors live in this process, so advertising loopback is both the
            # most portable and the narrowest network exposure. Kubernetes keeps
            # its routable pod-address configuration below.
            builder = builder.config("spark.driver.host", "127.0.0.1").config(
                "spark.driver.bindAddress",
                "127.0.0.1",
            )
        if distributed.spark_master.startswith("k8s://"):
            if not distributed.spark_image:
                raise ValueError("Kubernetes Spark finalization requires spark_image")
            builder = (
                builder.config("spark.kubernetes.container.image", distributed.spark_image)
                # A failed client-mode driver can leave its executor ConfigMap
                # until Kubernetes finishes deleting the owner pod. Explicitly
                # fencing the resource prefix by durable task attempt prevents a
                # retry from colliding with those terminating resources.
                .config("spark.kubernetes.resourceNamePrefix", application_name)
                .config(
                    "spark.kubernetes.authenticate.driver.serviceAccountName",
                    distributed.spark_service_account,
                )
                .config("spark.kubernetes.namespace", distributed.spark_namespace)
                .config(
                    "spark.kubernetes.executor.podTemplateFile",
                    distributed.spark_executor_pod_template,
                )
                .config(
                    "spark.kubernetes.executor.label.app.kubernetes.io/part-of",
                    "flakegraph",
                )
            )
            pod_ip = os.environ.get("POD_IP")
            if pod_ip:
                builder = (
                    builder.config("spark.driver.host", pod_ip)
                    .config("spark.driver.bindAddress", "0.0.0.0")
                    .config("spark.driver.port", "7078")
                    .config("spark.blockManager.port", "7079")
                )
            driver_pod_name = os.environ.get("POD_NAME")
            if driver_pod_name:
                # Client-mode Spark cannot infer its Kubernetes owner. Naming the
                # driver pod gives every executor an owner reference, so deleting
                # a failed/cancelled finalizer cannot leave expensive orphan pods.
                builder = builder.config(
                    "spark.kubernetes.driver.pod.name",
                    driver_pod_name,
                )
        if (distributed.artifact_uri or "").startswith("s3://"):
            builder = self._configure_s3(builder)
        return cast(SparkSession, builder.getOrCreate())

    def _configure_s3(self, builder: Any) -> Any:
        """Configure Hadoop S3A credentials and optional endpoint overrides.

        Transport security follows the configured endpoint scheme, and only an
        explicit ``http://`` endpoint disables TLS. Artifact objects and the SigV4
        credentials that sign every request would otherwise travel in plaintext
        whenever an operator points the engine at an ``https://`` endpoint.
        """

        distributed = self.settings.distributed
        builder = builder.config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )
        if distributed.artifact_endpoint_url:
            ssl_enabled = urlparse(distributed.artifact_endpoint_url).scheme.casefold() != "http"
            builder = (
                builder.config("spark.hadoop.fs.s3a.endpoint", distributed.artifact_endpoint_url)
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", str(ssl_enabled).lower())
            )
        if distributed.artifact_access_key_id:
            builder = builder.config(
                "spark.hadoop.fs.s3a.access.key",
                distributed.artifact_access_key_id,
            )
        if distributed.artifact_secret_access_key:
            builder = builder.config(
                "spark.hadoop.fs.s3a.secret.key",
                distributed.artifact_secret_access_key,
            )
        return builder

    def _read_stage_artifacts(
        self,
        spark: SparkSession,
        request: SparkFinalizationRequest,
    ) -> tuple[DataFrame, DataFrame]:
        """Read immutable stage objects once and reuse their parsed rows from disk.

        Graph construction branches into documents, chunks, mentions, relations,
        evidence, and quality checks. Spark's unified memory manager lets cached
        rows use otherwise-idle storage memory and evicts or spills them when joins
        need execution memory. This avoids forcing small corpora through local disk
        while still preventing repeated object-store parsing for large corpora.
        Lost cache blocks remain recomputable from immutable object storage.
        """

        from pyspark import StorageLevel
        from pyspark.sql import functions as F

        root = _spark_uri(self.settings.distributed.artifact_uri or "")
        prepared_path = [
            f"{root}/{run_id}/prepared_document"
            for run_id in request.stage_run_ids(request.prepared_run_ids)
        ]
        extracted_path = [
            f"{root}/{run_id}/extracted_document"
            for run_id in request.stage_run_ids(request.extracted_run_ids)
        ]
        # An object is named by its artifact id plus a media-type suffix, so
        # the id list is a filter on file names: a semi-join against a
        # broadcast of the ids keeps every row of the linked objects and none
        # of an orphan's, without listing the prefix on the driver.
        linked = F.broadcast(
            spark.createDataFrame([(item,) for item in sorted(request.artifact_ids)], "id string")
        )
        object_id = F.regexp_extract(F.input_file_name(), r"([^/]+?)(\.[^./]*)?$", 1)

        def only_linked(frame: DataFrame) -> DataFrame:
            return (
                frame.withColumn("_object_id", object_id)
                .join(linked, F.col("_object_id") == linked["id"], "left_semi")
                .drop("_object_id")
            )

        prepared = only_linked(
            spark.read.schema(_PREPARED_STAGE_SCHEMA).json(prepared_path)
        ).persist(StorageLevel.MEMORY_AND_DISK_DESER)
        extracted = only_linked(
            spark.read.schema(_EXTRACTED_STAGE_SCHEMA).json(extracted_path)
        ).persist(StorageLevel.MEMORY_AND_DISK_DESER)
        return prepared, extracted

    def _build_graph_tables(  # noqa: PLR0912, PLR0915 - each block is one table phase.
        self,
        spark: SparkSession,
        prepared: DataFrame,
        extracted: DataFrame,
        request: SparkFinalizationRequest,
    ) -> tuple[dict[str, DataFrame], dict[str, Any]]:
        """Transform nested stage records into canonical partitioned graph tables."""

        from graphframes import GraphFrame
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        graph_id = request.graph_id
        phase_timings: dict[str, float] = {}
        documents = _explode_struct(
            prepared, "document_rows", _EMPTY_SCHEMAS["documents"]
        ).dropDuplicates(["id"])
        pages = _explode_struct(prepared, "page_rows", _EMPTY_SCHEMAS["pages"]).dropDuplicates(
            ["id"]
        )
        blocks = _explode_struct(prepared, "block_rows", _EMPTY_SCHEMAS["blocks"]).dropDuplicates(
            ["id"]
        )
        assets = _explode_struct(prepared, "asset_rows", _EMPTY_SCHEMAS["assets"]).dropDuplicates(
            ["id"]
        )

        chunks = _explode_struct(
            prepared,
            "chunks",
            _EMPTY_SCHEMAS["chunks"],
        ).dropDuplicates(["id"])
        discarded_windows = self._discarded_windows(spark, extracted, chunks, request.graph_id)
        rejected_records = self._rejected_records(extracted, chunks, request.graph_id)
        failed_documents = self._failed_documents(extracted, request.graph_id)
        observed_mentions = _explode_struct(
            extracted,
            "observations.entities",
            _EMPTY_SCHEMAS["entities"],
        )
        context_mentions = _explode_struct(
            extracted,
            "prepared.document_context_entities",
            _EMPTY_SCHEMAS["entities"],
        )
        # Context entities are first-class graph observations even when their
        # grounding front matter is outside every body window. Unioning at the
        # final barrier protects local and distributed extraction alike; stable
        # mention IDs make the operation idempotent when a window also retained
        # the same context entity for relation extraction.
        mentions = observed_mentions.unionByName(context_mentions).dropDuplicates(["id"])
        relations = _explode_struct(
            extracted,
            "observations.relations",
            _EMPTY_SCHEMAS["relations"],
        ).dropDuplicates(["id"])
        held_relations = _explode_struct(
            extracted,
            "observations.held_relations",
            _EMPTY_SCHEMAS["relations"],
        ).dropDuplicates(["id"])
        ontology = self._ontology_profile()

        mentions = self._without_excluded_folder_mentions(
            mentions.where(
                (F.col("confidence") >= self.settings.graph.min_entity_confidence)
                & (F.length(F.trim(F.col("name"))) >= self.settings.graph.min_entity_name_length)
                & ~F.lower(F.trim(F.col("name"))).isin(
                    *[value.casefold() for value in self.settings.graph.entity_blocklist]
                )
            ).withColumn("normalized_name", _normalized_name(F.col("name"))),
            documents,
            chunks,
            ontology,
        ).checkpoint(eager=True)

        phase_started = perf_counter()
        self._report_progress("materialize_inputs", completed=0, total=1)
        mention_count = mentions.count()
        relation_count = relations.count()
        held_count = held_relations.count()
        self._report_progress("materialize_inputs", completed=1, total=1)
        phase_timings["input_materialization"] = perf_counter() - phase_started
        partitions = _effective_shuffle_partitions(
            mention_count + relation_count,
            self.settings.distributed.spark_executor_instances,
            self.settings.distributed.spark_executor_cores,
            self.settings.distributed.spark_shuffle_partitions,
        )
        spark.conf.set("spark.sql.shuffle.partitions", partitions)
        phase_started = perf_counter()
        self._report_progress("embed_chunks", completed=0, total=1)
        # Window workers do no embedding work. Encoding complete document chunks
        # here produces provider-sized batches, includes bibliography-only chunks,
        # and loads a local embedding model only in reusable Spark Python workers.
        chunks = self._with_embeddings(
            spark,
            chunks.drop("embedding"),
            id_column="id",
            text_column=F.col("content"),
            partitions=partitions,
        ).checkpoint(eager=True)
        self._report_progress("embed_chunks", completed=1, total=1)
        phase_timings["chunk_embeddings"] = perf_counter() - phase_started
        if mention_count == 0:
            for skipped_phase in (
                "resolve_entities",
                "assemble_graph",
                "enrich_graph",
                "build_communities",
                "quality_checks",
            ):
                self._report_progress(
                    skipped_phase,
                    completed=1,
                    total=1,
                    message="Skipped because no entity mentions were accepted",
                )
            tables = {
                "documents": self._annotated_documents(
                    spark, documents, extracted, request.document_provenance
                ),
                "pages": pages,
                "blocks": blocks,
                "assets": assets,
                "chunks": chunks,
                "discarded_windows": discarded_windows,
                "rejected_records": rejected_records,
                "failed_documents": failed_documents,
                **{
                    name: spark.createDataFrame([], _EMPTY_SCHEMAS[name])
                    for name in (
                        "nodes",
                        "edges",
                        "edge_observations",
                        "evidence",
                        "entity_sources",
                        "communities",
                        "community_findings",
                    )
                },
            }
            return tables, {
                "engine": "spark_graphframes",
                "input_entity_mentions": 0,
                "input_relation_observations": relation_count,
                "isolated_nodes": 0,
                "dangling_edges": 0,
                "evidence_coverage": 1.0,
                "discarded_windows": _discarded_window_metrics(discarded_windows),
                "rejected_records": _rejected_record_metrics(rejected_records),
                "type_resolution": type_resolution_metrics(held_count, 0),
                "failed_documents": _failed_document_metrics(failed_documents),
            }
        phase_started = perf_counter()
        self._report_progress("resolve_entities", completed=0, total=1)
        if self.settings.graph.entity_resolution_enabled:
            # Exact-name observations must remain separate evidence records, but
            # embedding every repetition wastes provider work and lets common
            # names consume bounded fuzzy-neighbor slots. One stable typed-name
            # representative is sufficient because exact identity edges below
            # connect every repetition to that representative's component.
            representative_window = Window.partitionBy("type", "normalized_name").orderBy(
                F.desc(F.length(F.coalesce("description", F.lit("")))),
                F.desc("confidence"),
                F.desc(F.length(F.coalesce("quote", F.lit("")))),
                F.asc("id"),
            )
            resolution_mentions = (
                mentions.withColumn(
                    "identity_rank",
                    F.row_number().over(representative_window),
                )
                .where(F.col("identity_rank") == 1)
                .drop("identity_rank")
            )
            resolution_mentions = self._with_embeddings(
                spark,
                resolution_mentions,
                id_column="id",
                text_column=F.concat_ws(
                    " ",
                    "type",
                    "name",
                    F.concat_ws(", ", "aliases"),
                    "description",
                ),
                partitions=partitions,
            ).checkpoint(eager=True)
        else:
            resolution_mentions = mentions
        phase_timings["mention_embeddings"] = perf_counter() - phase_started

        phase_started = perf_counter()
        # Identity generation contains executor-side LLM adjudication. Both the
        # emptiness branch and connected-components algorithm consume these rows,
        # so checkpoint once before either action. Without this boundary Spark can
        # replay the same provider calls and produce inconsistent retry histories.
        identity_edges, identity_metrics = self._identity_edges(
            spark,
            mentions,
            resolution_mentions,
            partitions,
        )
        identity_edges = identity_edges.checkpoint(eager=True)
        vertices = mentions.select(F.col("id")).dropDuplicates()
        identity_edge_count = identity_edges.count()
        if identity_edge_count:
            if (
                mention_count <= _LOCAL_COMPONENT_MAX_VERTICES
                and identity_edge_count <= _LOCAL_COMPONENT_MAX_EDGES
            ):
                component_rows = _connected_component_rows(
                    (str(row.id) for row in vertices.toLocalIterator(prefetchPartitions=True)),
                    (
                        (str(row.src), str(row.dst))
                        for row in identity_edges.select("src", "dst").toLocalIterator(
                            prefetchPartitions=True
                        )
                    ),
                )
                components = spark.createDataFrame(
                    component_rows,
                    "id string, component string",
                )
                component_engine = "bounded_union_find"
            else:
                raw_components = GraphFrame(
                    vertices,
                    identity_edges.select(F.col("src"), F.col("dst")),
                ).connectedComponents(
                    algorithm="two_phase",
                    # GraphFrames recommends disabling its manual high-degree
                    # broadcast so Spark AQE can handle skewed joins directly.
                    broadcastThreshold=-1,
                )
                # GraphFrames materializes and persists its result. A durable
                # checkpoint cuts the graph algorithm's lineage before downstream
                # joins, after which its internal cache can be released.
                components = raw_components.checkpoint(eager=True)
                raw_components.unpersist(blocking=True)
                component_engine = "graphframes_two_phase"
        else:
            components = vertices.withColumn("component", F.col("id"))
            component_engine = "no_identity_edges"
        phase_timings["identity_resolution"] = perf_counter() - phase_started
        self._report_progress("resolve_entities", completed=1, total=1)

        self._report_progress("assemble_graph", completed=0, total=1)

        # This join feeds canonical names, types, evidence, relations, and
        # descriptions. A single bounded lineage cut is cheaper than replaying it
        # through each independently materialized graph table.
        mention_components = mentions.join(components, "id", "inner").checkpoint(eager=True)
        name_counts = mention_components.groupBy("component", "normalized_name", "name").count()
        name_window = Window.partitionBy("component").orderBy(
            F.desc("count"),
            F.desc(F.length("name")),
            F.asc(F.lower("name")),
        )
        preferred_names = (
            name_counts.withColumn("choice", F.row_number().over(name_window))
            .where(F.col("choice") == 1)
            .select("component", "normalized_name", "name")
        )
        type_counts = mention_components.groupBy("component", "type").count()
        type_window = Window.partitionBy("component").orderBy(F.desc("count"), F.asc("type"))
        primary_types = (
            type_counts.withColumn("choice", F.row_number().over(type_window))
            .where(F.col("choice") == 1)
            .select("component", F.col("type").alias("primary_type"))
        )
        node_surfaces = _bounded_group_values(
            mention_components.select(
                "component",
                F.explode(
                    F.concat(
                        F.array("name"),
                        F.col("aliases"),
                    )
                ).alias("surface"),
            ),
            group_columns=("component",),
            value_column="surface",
            output_column="observed_surfaces",
            limit=_MAX_NODE_SURFACES,
        )
        node_source_chunks = _bounded_group_values(
            mention_components,
            group_columns=("component",),
            value_column="source_chunk_id",
            output_column="source_chunk_ids",
            limit=_MAX_DENORMALIZED_PROVENANCE_IDS,
        )
        node_groups = (
            mention_components.groupBy("component")
            .agg(
                F.sort_array(F.collect_set("type")).alias("types"),
                F.max_by("description", F.length("description")).alias("description"),
            )
            .join(node_surfaces, "component")
            .join(node_source_chunks, "component")
            .join(preferred_names, "component")
            .join(primary_types, "component")
            .withColumn("aliases", F.array_except("observed_surfaces", F.array("name")))
            .drop("observed_surfaces")
        )
        node_identity = node_groups.withColumn(
            "id",
            _stable_id(
                "node",
                F.lit(graph_id),
                F.col("normalized_name"),
                F.col("primary_type"),
            ),
        ).withColumn("graph_id", F.lit(graph_id))
        # Resolution deliberately keeps a bare short acronym apart from a mention
        # that spells it out (SAR may be a structure-activity relationship in one
        # document and something else in another), so two components can prefer
        # one name and type. The best-attested keeps the plain id; each other one
        # adds its component to the id, so every node row has an id of its own.
        homonym_window = Window.partitionBy("normalized_name", "primary_type").orderBy(
            F.desc(F.size("source_chunk_ids")), F.asc("component")
        )
        node_identity = (
            node_identity.withColumn("homonym_rank", F.row_number().over(homonym_window))
            .withColumn(
                "id",
                F.when(F.col("homonym_rank") == 1, F.col("id")).otherwise(
                    _stable_id(
                        "node",
                        F.lit(graph_id),
                        F.col("normalized_name"),
                        F.col("primary_type"),
                        F.col("component"),
                    )
                ),
            )
            .drop("homonym_rank")
        )
        nodes = (
            node_identity.withColumn("embedding", F.lit(None).cast("array<double>"))
            .withColumn("degree", F.lit(0))
            .withColumn("rank", F.lit(None).cast("double"))
            .select(
                "id",
                "graph_id",
                "normalized_name",
                "name",
                "primary_type",
                "types",
                "aliases",
                "description",
                "embedding",
                "source_chunk_ids",
                "degree",
                "rank",
            )
        )
        mention_to_node = (
            mention_components.select("id", "component")
            .join(
                node_identity.select("component", F.col("id").alias("node_id")),
                "component",
                "inner",
            )
            .select(F.col("id").alias("mention_id"), "node_id")
        ).checkpoint(eager=True)

        chunk_lookup = chunks.select(
            F.col("id").alias("chunk_id"),
            "file_id",
            "page_number",
            "content",
            F.col("start_offset").alias("chunk_start_offset"),
        )
        source_map = mention_to_node.select(
            F.col("mention_id").alias("source_entity_id"),
            F.col("node_id").alias("source_node_id"),
        )
        target_map = mention_to_node.select(
            F.col("mention_id").alias("target_entity_id"),
            F.col("node_id").alias("target_node_id"),
        )

        self_loop_relation_types = sorted(
            normalize_relation_type(definition.name)
            for definition in ontology.relation_types
            if definition.allow_self_loop
        )
        structural_types = sorted(structural_relation_types(ontology))

        def with_edges(endpoint_relations: DataFrame) -> DataFrame:
            """Locate relations with resolved endpoints in their chunks and name their edges."""

            located = endpoint_relations.join(
                chunk_lookup,
                endpoint_relations.source_chunk_id == chunk_lookup.chunk_id,
                "inner",
            ).withColumn(
                "normalized_relation_type",
                _normalized_relation_type(F.col("relation_type")),
            )
            # Distinct extracted spellings can resolve to the same canonical node
            # after alias and punctuation normalization. Such an assertion was not
            # a source-level self-loop, so it survives extraction and is rejected
            # here, where canonical endpoint identity is finally known. A source
            # assertion naming one mention at both ends is kept where the
            # ontology allows its predicate to loop.
            distinct_ends = F.col("source_node_id") != F.col("target_node_id")
            if self_loop_relation_types:
                distinct_ends = distinct_ends | (
                    (F.col("source_entity_id") == F.col("target_entity_id"))
                    & F.col("normalized_relation_type").isin(self_loop_relation_types)
                )
            return located.where(distinct_ends).withColumn(
                "edge_id",
                _stable_id(
                    "edge",
                    F.lit(graph_id),
                    F.col("source_node_id"),
                    F.col("target_node_id"),
                    F.col("normalized_relation_type"),
                ),
            )

        grounded_relations = with_edges(
            relations.where(F.col("confidence") >= self.settings.graph.min_relation_confidence)
            .join(source_map, "source_entity_id", "inner")
            .join(target_map, "target_entity_id", "inner")
        )
        type_resolved_count = 0
        if self.settings.graph.relation_type_resolution and held_count:
            chosen = self._type_resolved_relations(
                spark, held_relations, mention_components, node_identity, mention_to_node, chunks
            ).checkpoint(eager=True)
            type_resolved_count = chosen.count()
            # An admitted observation whose edge the chunk already supports adds
            # nothing; the graph states it, so it still counts as admitted.
            first_in_chunk = Window.partitionBy("edge_id", "chunk_id").orderBy(
                F.coalesce("start_offset", F.lit(0)),
                F.coalesce("end_offset", F.lit(0)),
                F.coalesce("rejected_record_key", F.lit("")),
            )
            admitted = (
                with_edges(chosen)
                .join(
                    grounded_relations.select("edge_id", "chunk_id"),
                    ["edge_id", "chunk_id"],
                    "left_anti",
                )
                .withColumn("chunk_rank", F.row_number().over(first_in_chunk))
                .where(F.col("chunk_rank") == 1)
                .drop("chunk_rank")
            )
            grounded_relations = grounded_relations.unionByName(admitted)
            rejected_records = self._rejected_records(
                extracted,
                chunks,
                graph_id,
                taken_back=chosen.select(
                    F.coalesce("rejected_record_key", F.lit("")).alias("record_key")
                ),
            )
        if ontology.folder_entity is not None:
            grounded_relations = grounded_relations.unionByName(
                with_edges(
                    self._folder_links(mention_components, mention_to_node, chunks, ontology)
                )
            )
        grounded_relations = grounded_relations.checkpoint(eager=True)
        # ``edge_id`` is a pure function of the grouping columns. Computing it on
        # each observation avoids a distinct-key shuffle followed by a join back
        # to the same relation rows; the aggregation below is the only required
        # exchange and preserves identical canonical IDs.
        edge_source_files = _bounded_group_values(
            grounded_relations,
            group_columns=("edge_id",),
            value_column="file_id",
            output_column="source_file_ids",
            limit=_MAX_DENORMALIZED_PROVENANCE_IDS,
        )
        edge_source_chunks = _bounded_group_values(
            grounded_relations,
            group_columns=("edge_id",),
            value_column="chunk_id",
            output_column="source_chunk_ids",
            limit=_MAX_DENORMALIZED_PROVENANCE_IDS,
        )
        edges = (
            grounded_relations.groupBy(
                "edge_id",
                "source_node_id",
                "target_node_id",
                "normalized_relation_type",
            )
            .agg(
                F.max_by("description", F.length("description")).alias("description"),
                # Canonical weight is the bounded sum of the weights its grounded
                # observations contribute, so repeated independent evidence
                # strengthens an edge while no edge can exceed the configured
                # maximum. Non-negative terms make the bounded sum identical to
                # clamping after every individual observation.
                F.least(
                    F.sum(_observation_weight(F.col("confidence"))),
                    F.lit(self.settings.graph.relation_weight_max),
                ).alias("weight"),
                F.max("confidence").alias("confidence"),
                # Document compaction has already deduplicated relation IDs and
                # endpoint/chunk joins are one-to-one. A second exact-distinct
                # set here only adds hot-key state for frequently repeated facts.
                F.count("id").alias("evidence_count"),
                _claimed_only(F.col("assertion")).alias("assertion"),
            )
            .join(edge_source_files, "edge_id")
            .join(edge_source_chunks, "edge_id")
            .withColumn("graph_id", F.lit(graph_id))
            .withColumn("source_file_id", F.element_at("source_file_ids", 1))
            .withColumn("embedding", F.lit(None).cast("array<double>"))
            .select(
                F.col("edge_id").alias("id"),
                "graph_id",
                "source_node_id",
                "target_node_id",
                F.col("normalized_relation_type").alias("relation_type"),
                "description",
                "weight",
                "confidence",
                "source_file_id",
                "source_file_ids",
                "source_chunk_ids",
                "evidence_count",
                "assertion",
                "embedding",
            )
            .checkpoint(eager=True)
        )

        if self.settings.graph.drop_isolated_entities:
            connected_nodes = _connected_nodes(edges, structural_types)
            if structural_types:
                # A structural edge keeps neither end: one whose other end is
                # gone goes too, and with it its observations and evidence.
                edges = edges.join(
                    connected_nodes.select(F.col("node_id").alias("source_node_id")),
                    "source_node_id",
                    "left_semi",
                ).join(
                    connected_nodes.select(F.col("node_id").alias("target_node_id")),
                    "target_node_id",
                    "left_semi",
                )
                grounded_relations = grounded_relations.join(
                    edges.select(F.col("id").alias("edge_id")), "edge_id", "left_semi"
                )
            nodes = nodes.join(connected_nodes, nodes.id == connected_nodes.node_id, "inner").drop(
                "node_id"
            )
            mention_to_node = mention_to_node.join(connected_nodes, "node_id", "inner")

        degree_rows = (
            edges.select(F.col("source_node_id").alias("node_id"))
            .unionByName(edges.select(F.col("target_node_id").alias("node_id")))
            .groupBy("node_id")
            .count()
        )
        nodes = (
            nodes.drop("degree", "rank")
            .join(degree_rows, nodes.id == degree_rows.node_id, "left")
            .drop("node_id")
            .withColumn("degree", F.coalesce("count", F.lit(0)))
            # Rank orders entities for exploration and report context. Canonical
            # degree is its structural definition, so every engine publishes the
            # same value and an isolated node ranks zero rather than unknown.
            .withColumn("rank", F.col("degree").cast("double"))
            .drop("count")
        )
        phase_started = perf_counter()
        nodes = nodes.checkpoint(eager=True)
        phase_timings["node_assembly"] = perf_counter() - phase_started
        phase_started = perf_counter()
        edges = self._with_embeddings(
            spark,
            edges.drop("embedding"),
            id_column="id",
            text_column=F.concat_ws(
                " ",
                "relation_type",
                "description",
                "source_node_id",
                "target_node_id",
            ),
            partitions=partitions,
        ).checkpoint(eager=True)
        phase_timings["edge_assembly_and_embeddings"] = perf_counter() - phase_started

        phase_started = perf_counter()
        edge_evidence = grounded_relations.withColumn(
            "evidence_id",
            _stable_id(
                "evidence",
                F.lit(graph_id),
                F.col("edge_id"),
                F.col("chunk_id"),
            ),
        )
        edge_observations = (
            edge_evidence.withColumn(
                "id",
                _stable_id(
                    "edge_observation",
                    F.lit(graph_id),
                    F.col("edge_id"),
                    F.col("chunk_id"),
                ),
            )
            .withColumn("graph_id", F.lit(graph_id))
            .select(
                "id",
                "graph_id",
                "edge_id",
                "file_id",
                F.col("chunk_id"),
                "description",
                F.lit(1.0).alias("weight"),
                "confidence",
                "evidence_id",
                "quantities",
                F.col("proposed_relation_type").cast("string").alias("proposed_relation_type"),
                _claimed_only(F.col("assertion"), Window.partitionBy("edge_id", "chunk_id")).alias(
                    "assertion"
                ),
            )
            .dropDuplicates(["id"])
        )
        relation_evidence = edge_evidence.select(
            F.col("evidence_id").alias("id"),
            F.lit(graph_id).alias("graph_id"),
            F.col("edge_id").alias("subject_id"),
            F.lit("edge").alias("subject_kind"),
            "file_id",
            F.col("chunk_id"),
            "page_number",
            (F.col("chunk_start_offset") + F.coalesce("start_offset", F.lit(0)))
            .cast("int")
            .alias("start_offset"),
            (F.col("chunk_start_offset") + F.coalesce("end_offset", F.length("quote"), F.lit(0)))
            .cast("int")
            .alias("end_offset"),
            F.coalesce("quote", F.lit("")).alias("quote"),
            F.col("evidence_method").alias("method"),
        ).dropDuplicates(["id"])

        mention_evidence_rows = (
            mention_components.join(
                mention_to_node,
                mention_components.id == mention_to_node.mention_id,
            )
            .join(chunk_lookup, mention_components.source_chunk_id == chunk_lookup.chunk_id)
            .withColumn(
                "evidence_id",
                _stable_id(
                    "evidence",
                    F.lit(graph_id),
                    F.col("node_id"),
                    F.col("chunk_id"),
                ),
            )
        )
        # One mention evidences a node in a chunk: the one with the longest
        # description, a mention the text makes before a folder mention, as
        # local assembly chooses.
        evidence_choice = Window.partitionBy("evidence_id").orderBy(
            (F.col("evidence_method") == F.lit(FOLDER_EVIDENCE_METHOD)).asc(),
            F.desc(F.length(F.coalesce(F.col("description"), F.lit("")))),
            F.asc("id"),
        )
        node_evidence = (
            mention_evidence_rows.withColumn("evidence_rank", F.row_number().over(evidence_choice))
            .where(F.col("evidence_rank") == 1)
            .select(
                F.col("evidence_id").alias("id"),
                F.lit(graph_id).alias("graph_id"),
                F.col("node_id").alias("subject_id"),
                F.lit("node").alias("subject_kind"),
                "file_id",
                F.col("chunk_id"),
                "page_number",
                (F.col("chunk_start_offset") + F.coalesce("start_offset", F.lit(0)))
                .cast("int")
                .alias("start_offset"),
                (
                    F.col("chunk_start_offset")
                    + F.coalesce("end_offset", F.length("quote"), F.lit(0))
                )
                .cast("int")
                .alias("end_offset"),
                F.coalesce("quote", F.lit("")).alias("quote"),
                F.col("evidence_method").alias("method"),
            )
        )
        evidence = (
            node_evidence.unionByName(relation_evidence)
            .dropDuplicates(["id"])
            .checkpoint(eager=True)
        )
        phase_timings["evidence_assembly"] = perf_counter() - phase_started

        entity_sources = (
            mention_evidence_rows.groupBy("node_id", "file_id")
            .agg(
                F.max_by("description", F.length("description")).alias("per_file_description"),
                # Mention IDs are unique before this one-to-one provenance join.
                F.count("id").alias("mention_count"),
                F.array_sort(
                    F.array_distinct(
                        F.flatten(
                            F.collect_list(
                                F.concat(
                                    F.array("name"),
                                    F.col("aliases"),
                                )
                            )
                        )
                    )
                ).alias("aliases"),
            )
            .withColumn(
                "id",
                _stable_id(
                    "entity_source",
                    F.lit(graph_id),
                    F.col("node_id"),
                    F.col("file_id"),
                ),
            )
            .withColumn("graph_id", F.lit(graph_id))
            .select(
                "id",
                "graph_id",
                "node_id",
                "file_id",
                "per_file_description",
                "mention_count",
                "aliases",
            )
        )
        phase_started = perf_counter()
        self._report_progress("assemble_graph", completed=1, total=1)
        self._report_progress("enrich_graph", completed=0, total=1)
        nodes, entity_sources, description_merge_records = self._merge_entity_descriptions(
            spark,
            nodes,
            entity_sources,
            mention_components,
            mention_to_node,
            evidence,
        )
        description_merge_elapsed = perf_counter() - phase_started
        node_embedding_started = perf_counter()
        nodes = self._with_embeddings(
            spark,
            nodes.drop("embedding"),
            id_column="id",
            text_column=F.concat_ws(
                " ",
                "primary_type",
                "name",
                F.concat_ws(", ", "aliases"),
                "description",
            ),
            partitions=partitions,
        ).checkpoint(eager=True)
        node_embedding_elapsed = perf_counter() - node_embedding_started
        phase_timings["description_merge"] = description_merge_elapsed
        phase_timings["node_embeddings"] = node_embedding_elapsed
        phase_timings["description_merge_and_node_embeddings"] = (
            description_merge_elapsed + node_embedding_elapsed
        )
        self._report_progress("enrich_graph", completed=1, total=1)

        phase_started = perf_counter()
        self._report_progress("build_communities", completed=0, total=1)
        communities = self._communities(
            GraphFrame,
            nodes,
            edges,
            evidence,
            graph_id,
        ).checkpoint(eager=True)
        community_detection_elapsed = perf_counter() - phase_started

        phase_started = perf_counter()
        communities, community_findings, community_report_records = self._community_reports(
            spark,
            communities,
            nodes,
            edges,
            evidence,
        )
        community_reporting_elapsed = perf_counter() - phase_started

        phase_started = perf_counter()
        communities = self._with_embeddings(
            spark,
            communities.drop("embedding"),
            id_column="id",
            text_column=F.concat_ws(" ", "title", "summary"),
            partitions=partitions,
        )
        community_embedding_elapsed = perf_counter() - phase_started
        phase_timings["community_detection"] = community_detection_elapsed
        phase_timings["community_reporting"] = community_reporting_elapsed
        phase_timings["community_embeddings"] = community_embedding_elapsed
        self._report_progress("build_communities", completed=1, total=1)

        phase_started = perf_counter()
        self._report_progress("quality_checks", completed=0, total=1)
        # A node only structural edges reach is as isolated as one with none.
        shape_endpoints = (
            edges.where(~F.col("relation_type").isin(structural_types))
            .select(F.col("source_node_id").alias("node_id"))
            .unionByName(
                edges.where(~F.col("relation_type").isin(structural_types)).select(
                    F.col("target_node_id").alias("node_id")
                )
            )
            .dropDuplicates(["node_id"])
            .withColumn("has_relation", F.lit(1))
        )
        node_quality = cast(
            "Any",
            nodes.join(shape_endpoints, nodes.id == shape_endpoints.node_id, "left")
            .agg(
                F.count(F.lit(1)).alias("node_count"),
                F.sum(F.when(F.col("has_relation").isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
                    "isolated_nodes"
                ),
            )
            .first(),
        )
        edge_quality = cast(
            "Any",
            edges.agg(
                F.count(F.lit(1)).alias("edge_count"),
                F.sum(F.when(F.col("evidence_count") > 0, F.lit(1)).otherwise(F.lit(0))).alias(
                    "supported_edges"
                ),
            ).first(),
        )
        hard_quality_errors = self._hard_quality_errors(
            spark,
            nodes,
            edges,
            evidence,
            chunks,
        )
        phase_timings["quality_checks"] = perf_counter() - phase_started
        self._report_progress("quality_checks", completed=1, total=1)
        hard_error_count = sum(hard_quality_errors.values())
        if hard_error_count and self.settings.graph.fail_on_quality_error:
            details = ", ".join(
                f"{name}={count}" for name, count in sorted(hard_quality_errors.items()) if count
            )
            raise ValueError(f"distributed graph quality checks failed: {details}")
        edge_count = int(edge_quality.edge_count or 0)
        supported_edges = int(edge_quality.supported_edges or 0)

        nodes = nodes.select(
            "id",
            "graph_id",
            "normalized_name",
            "name",
            "primary_type",
            "types",
            "aliases",
            "description",
            "embedding",
            "source_chunk_ids",
            F.col("degree").cast("int").alias("degree"),
            "rank",
        )
        documents = self._annotated_documents(
            spark, documents, extracted, request.document_provenance
        )
        tables = {
            "documents": documents,
            "pages": pages,
            "blocks": blocks,
            "assets": assets,
            "chunks": chunks,
            "nodes": nodes,
            "edges": edges,
            "edge_observations": edge_observations,
            "evidence": evidence,
            "entity_sources": entity_sources,
            "communities": communities,
            "community_findings": community_findings,
            "discarded_windows": discarded_windows,
            "rejected_records": rejected_records,
            "failed_documents": failed_documents,
        }
        metrics = {
            "engine": "spark_graphframes",
            "input_entity_mentions": mention_count,
            "input_relation_observations": relation_count,
            "shuffle_partitions": partitions,
            "entity_resolution_candidate_strategy": "typed_multikey_bounded_neighbors",
            "community_algorithm": "graphframes_label_propagation",
            "community_co_mention_strategy": "bounded_chunk_anchor_star",
            "community_iterations": _COMMUNITY_ITERATIONS,
            "isolated_nodes": int(node_quality.isolated_nodes or 0),
            "dangling_edges": hard_quality_errors["orphan_edges"],
            "evidence_coverage": supported_edges / edge_count if edge_count else 1.0,
            "hard_quality_error_count": hard_error_count,
            "hard_quality_errors": hard_quality_errors,
            "identity_links": identity_edge_count,
            "identity_component_engine": component_engine,
            "description_merge_records": description_merge_records,
            "description_merge_batch_size": DESCRIPTION_BATCH_SIZE,
            "community_report_records": community_report_records,
            "community_report_batch_size": COMMUNITY_BATCH_SIZE,
            "denormalized_provenance_id_limit": _MAX_DENORMALIZED_PROVENANCE_IDS,
            "node_surface_limit": _MAX_NODE_SURFACES,
            "identity_block_token_limit": _MAX_IDENTITY_BLOCK_TOKENS,
            **identity_metrics,
            "phase_timings_seconds": phase_timings,
            "extraction": self._extraction_metrics(extracted),
            "discarded_windows": _discarded_window_metrics(discarded_windows),
            "rejected_records": _rejected_record_metrics(rejected_records),
            "type_resolution": type_resolution_metrics(held_count, type_resolved_count),
            "failed_documents": _failed_document_metrics(failed_documents),
        }
        return tables, metrics

    def _type_resolved_relations(  # noqa: PLR0913 - the resolved graph a choice reads.
        self,
        spark: SparkSession,
        held_relations: DataFrame,
        mention_components: DataFrame,
        node_identity: DataFrame,
        mention_to_node: DataFrame,
        chunks: DataFrame,
    ) -> DataFrame:
        """Choose, for each held relation, the typing its endpoints' nodes admit.

        The rule is ``graph_merge._admitted_held_relations``: an endpoint stays
        on its own node or moves to a node of the same name and another type
        attested in enough documents, at least one moves, the rule table keeps
        the typing, and the fewest moves, best-attested nodes, type names and
        node ids decide. Rows keep the held relation's columns, with the chosen
        endpoint nodes, the kept relation type, and ``type_resolved`` evidence.
        """

        from pyspark.sql import Window
        from pyspark.sql import functions as F

        min_documents = self.settings.graph.relation_type_resolution_min_documents
        nodes = (
            mention_components.select("component", "source_chunk_id")
            .join(
                chunks.select(F.col("id").alias("source_chunk_id"), "document_id"),
                "source_chunk_id",
                "left",
            )
            .groupBy("component")
            .agg(F.countDistinct("document_id").alias("documents"))
            .join(
                node_identity.select(
                    "component",
                    F.col("id").alias("node_id"),
                    "normalized_name",
                    F.col("primary_type").alias("node_type"),
                ),
                "component",
            )
            .drop("component")
        )
        rules = spark.createDataFrame(
            [
                (stated, source_type, target_type, kept, replaced)
                for (stated, source_type, target_type), (kept, replaced) in sorted(
                    type_resolution_rules(self._ontology_profile()).items()
                )
            ],
            "relation_type string, source_type string, target_type string, "
            "kept_relation_type string, stated_relation_type string",
        )

        def endpoint(side: str) -> tuple[DataFrame, DataFrame]:
            """The endpoint's own node by name, and every node it may take."""

            own = nodes.select(
                F.col("node_id").alias(f"{side}_node_id"),
                F.col("normalized_name").alias(f"{side}_name"),
            )
            candidates = nodes.select(
                F.col("node_id").alias(f"{side}_node"),
                F.col("normalized_name").alias(f"{side}_name"),
                F.col("node_type").alias(f"{side}_type"),
                F.col("documents").alias(f"{side}_documents"),
            )
            return own, candidates

        own_source, source_candidates = endpoint("source")
        own_target, target_candidates = endpoint("target")
        typings = (
            held_relations.where(F.col("confidence") >= self.settings.graph.min_relation_confidence)
            .join(
                mention_to_node.select(
                    F.col("mention_id").alias("source_entity_id"),
                    F.col("node_id").alias("source_node_id"),
                ),
                "source_entity_id",
            )
            .join(
                mention_to_node.select(
                    F.col("mention_id").alias("target_entity_id"),
                    F.col("node_id").alias("target_node_id"),
                ),
                "target_entity_id",
            )
            .join(own_source, "source_node_id")
            .join(source_candidates, "source_name")
            .where(
                (F.col("source_node") == F.col("source_node_id"))
                | (F.col("source_documents") >= F.lit(min_documents))
            )
            .join(own_target, "target_node_id")
            .join(target_candidates, "target_name")
            .where(
                (F.col("target_node") == F.col("target_node_id"))
                | (F.col("target_documents") >= F.lit(min_documents))
            )
            .withColumn(
                "moved",
                (F.col("source_node") != F.col("source_node_id")).cast("int")
                + (F.col("target_node") != F.col("target_node_id")).cast("int"),
            )
            .where((F.col("moved") > 0) & (F.col("source_node") != F.col("target_node")))
            .join(F.broadcast(rules), ["relation_type", "source_type", "target_type"])
        )
        choice = Window.partitionBy("id").orderBy(
            F.asc("moved"),
            F.desc("source_documents"),
            F.desc("target_documents"),
            F.asc("source_type"),
            F.asc("target_type"),
            F.asc("source_node"),
            F.asc("target_node"),
        )
        return (
            typings.withColumn("choice", F.row_number().over(choice))
            .where(F.col("choice") == 1)
            .withColumn("relation_type", F.col("kept_relation_type"))
            .withColumn(
                "proposed_relation_type",
                F.coalesce(F.col("stated_relation_type"), F.col("proposed_relation_type")),
            )
            .withColumn("evidence_method", F.lit(TYPE_RESOLVED_METHOD))
            .withColumn("source_node_id", F.col("source_node"))
            .withColumn("target_node_id", F.col("target_node"))
            .select(*_column_names(_EMPTY_SCHEMAS["relations"]), "source_node_id", "target_node_id")
        )

    def _discarded_windows(
        self,
        spark: SparkSession,
        extracted: DataFrame,
        chunks: DataFrame,
        graph_id: str,
    ) -> DataFrame:
        """Name the windows whose every extracted record was rejected, as rows.

        The rule is ``discarded_windows.is_discarded_window_event``, expressed
        over the trace column so no executor ships its trace to the driver; the
        chunk table supplies the pages and the preview, the same way the local
        pipeline joins its in-process chunks.
        """

        from pyspark.sql import Window
        from pyspark.sql import functions as F

        stage_names = F.create_map(
            *[F.lit(item) for pair in DISCARDED_WINDOW_STAGES.items() for item in pair]
        )
        events = (
            extracted.select(F.explode_outer("observations.trace").alias("event"))
            .select("event.*")
            .where(
                F.col("stage").isin(list(DISCARDED_WINDOW_STAGES))
                & (F.coalesce(F.col("input_records"), F.lit(0)) > 0)
                & (F.coalesce(F.col("accepted_records"), F.lit(0)) == 0)
            )
            .withColumn("stage", stage_names[F.col("stage")])
        )
        members = (
            events.select("stage", "window_id", F.explode("chunk_ids").alias("chunk_id"))
            .join(
                chunks.select(
                    F.col("id").alias("chunk_id"),
                    F.col("file_id").alias("member_file_id"),
                    "page_number",
                    "chunk_index",
                    "content",
                ),
                "chunk_id",
                "left",
            )
            .withColumn(
                "content",
                F.regexp_replace(F.trim(F.coalesce(F.col("content"), F.lit(""))), r"\s+", " "),
            )
        )
        ordered = Window.partitionBy("stage", "window_id").orderBy("page_number", "chunk_index")
        joined_text = members.withColumn(
            "texts",
            F.collect_list(F.when(F.col("content") != "", F.col("content"))).over(
                ordered.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )
        per_window = joined_text.groupBy("stage", "window_id").agg(
            F.min("page_number").alias("page_start"),
            F.max("page_number").alias("page_end"),
            F.first("member_file_id", ignorenulls=True).alias("member_file_id"),
            F.first("texts").alias("texts"),
        )
        preview = F.array_join(F.col("texts"), " ")
        rows = (
            events.join(per_window, ["stage", "window_id"], "left")
            .select(
                F.sha2(
                    F.concat_ws(
                        "|", F.lit("discarded_window"), F.lit(graph_id), "stage", "window_id"
                    ),
                    256,
                ).alias("digest"),
                F.lit(graph_id).alias("graph_id"),
                "document_id",
                F.coalesce(F.col("member_file_id"), F.col("document_id")).alias("file_id"),
                "stage",
                "window_id",
                "chunk_ids",
                F.col("page_start").cast("int").alias("page_start"),
                F.col("page_end").cast("int").alias("page_end"),
                F.coalesce(F.col("input_records"), F.lit(0))
                .cast("long")
                .alias("extracted_records"),
                F.coalesce(
                    F.col("record_actions"), F.create_map().cast("map<string,bigint>")
                ).alias("record_actions"),
                F.when(
                    F.length(preview) > PREVIEW_CHARACTERS,
                    F.concat(F.rtrim(F.substring(preview, 1, PREVIEW_CHARACTERS - 1)), F.lit("…")),
                )
                .otherwise(preview)
                .alias("preview"),
            )
            .withColumn("id", F.concat(F.lit("discarded_window_"), F.substring("digest", 1, 32)))
            .drop("digest")
            .select(*_column_names(_EMPTY_SCHEMAS["discarded_windows"]))
            .dropDuplicates(["id"])
        )
        return rows

    def _rejected_records(
        self,
        extracted: DataFrame,
        chunks: DataFrame,
        graph_id: str,
        taken_back: DataFrame | None = None,
    ) -> DataFrame:
        """List every record validation turned away, as rows.

        The rule and the row identity are ``rejected_records.rejected_records_
        from_trace``, expressed over the trace column; the chunk table supplies
        the file and the page, as it does locally. ``taken_back`` holds, as
        ``record_key``, the ``rejected_record_key`` of each held relation the
        graph admitted; those records were not lost and have no row.
        """

        from pyspark.sql import functions as F

        records = (
            extracted.select(F.explode_outer("observations.trace").alias("event"))
            .where(F.col("event.stage").isin(sorted(REJECTING_STAGES)))
            .select(
                F.col("event.window_id").alias("window_id"),
                F.col("event.document_id").alias("event_document_id"),
                F.explode(F.coalesce(F.col("event.rejected_records"), F.array())).alias("r"),
            )
            .select(
                F.coalesce(F.col("window_id"), F.lit("")).alias("window_id"),
                "event_document_id",
                F.when(F.col("r.kind") == "relation", F.lit("relation"))
                .otherwise(F.lit("entity"))
                .alias("kind"),
                F.coalesce(F.nullif(F.col("r.reason"), F.lit("")), F.lit("rejected")).alias(
                    "reason"
                ),
                *[
                    F.coalesce(F.col(f"r.{field}"), F.lit("")).alias(field)
                    for field in REJECTED_RECORD_FIELDS
                ],
            )
        )
        located = records.join(
            chunks.select(
                F.col("id").alias("chunk_id"),
                F.col("file_id").alias("chunk_file_id"),
                F.col("document_id").alias("chunk_document_id"),
                "page_number",
            ),
            "chunk_id",
            "left",
        ).withColumn(
            "document_id",
            F.coalesce(
                F.nullif(F.col("event_document_id"), F.lit("")),
                F.col("chunk_document_id"),
                F.lit(""),
            ),
        )
        if taken_back is not None:
            located = located.withColumn(
                "record_key",
                _stable_id(
                    "rejected_record_key",
                    F.col("window_id"),
                    F.col("kind"),
                    F.col("reason"),
                    *[F.col(field) for field in REJECTED_RECORD_FIELDS],
                ),
            ).join(taken_back.dropDuplicates(["record_key"]), "record_key", "left_anti")
        digest = F.sha2(
            F.concat_ws(
                "\x1f",
                F.lit(graph_id),
                "window_id",
                "kind",
                "reason",
                *REJECTED_RECORD_FIELDS,
            ),
            256,
        )
        return (
            located.select(
                F.concat(F.lit("rejected_record_"), F.substring(digest, 1, 32)).alias("id"),
                F.lit(graph_id).alias("graph_id"),
                "document_id",
                F.coalesce(F.col("chunk_file_id"), F.col("document_id")).alias("file_id"),
                "window_id",
                "chunk_id",
                F.col("page_number").cast("int").alias("page_number"),
                "kind",
                "reason",
                *[field for field in REJECTED_RECORD_FIELDS if field != "chunk_id"],
            )
            .select(*_column_names(_EMPTY_SCHEMAS["rejected_records"]))
            .dropDuplicates(["id"])
        )

    def _failed_documents(self, extracted: DataFrame, graph_id: str) -> DataFrame:
        """Name the documents parsing gave up on, as rows.

        The rule is ``failed_documents.is_failed_document_event`` over each
        artifact's own trace, where a worker records the parse that failed;
        the id is ``failed_documents.failed_document_id``.
        """

        from pyspark.sql import functions as F

        events = (
            extracted.select(F.explode_outer("trace").alias("event"))
            .select("event.*")
            .where(
                (F.col("stage") == F.lit(PARSE_STAGE))
                & (F.col("status") == F.lit(FAILED_STATUS))
                & F.col("file_id").isNotNull()
                & (F.col("file_id") != F.lit(""))
            )
        )
        return (
            events.select(
                # stable_id's own scheme, so both engines key a file alike.
                F.sha2(F.concat_ws("\x1f", F.lit(graph_id), "file_id"), 256).alias("digest"),
                F.lit(graph_id).alias("graph_id"),
                F.col("file_id").alias("document_id"),
                "file_id",
                F.coalesce(F.col("source_uri"), F.lit("")).alias("source_uri"),
                "mime_type",
                F.col("size_bytes").cast("long").alias("size_bytes"),
                "provider",
                F.coalesce(F.col("error_type"), F.lit("")).alias("error_type"),
                F.coalesce(F.col("error_message"), F.lit("")).alias("error_message"),
            )
            .withColumn("id", F.concat(F.lit("failed_document_"), F.substring("digest", 1, 32)))
            .drop("digest")
            .select(*_column_names(_EMPTY_SCHEMAS["failed_documents"]))
            .dropDuplicates(["id"])
        )

    def _annotated_documents(
        self,
        spark: SparkSession,
        documents: DataFrame,
        extracted: DataFrame,
        provenance: Mapping[str, str],
    ) -> DataFrame:
        """Give documents their summaries, related files and provenance, as a local run does.

        Document rows are few - one per source file - so the driver applies the
        same rule the local pipeline uses rather than a second Spark version of it.
        """

        from pyspark.sql import functions as F

        profiles = [
            {"stage": DATASET_PROFILE_STAGE, **row.asDict()}
            for row in extracted.select(F.explode_outer("observations.trace").alias("event"))
            .where(F.col("event.stage") == DATASET_PROFILE_STAGE)
            .select("event.document_id", "event.kind", "event.summary")
            .collect()
        ]
        rows = [row.asDict(recursive=True) for row in documents.collect()]
        if not rows:
            return documents
        return spark.createDataFrame(
            annotate_documents(rows, profiles, provenance), schema=_EMPTY_SCHEMAS["documents"]
        )

    def _extraction_metrics(self, extracted: DataFrame) -> dict[str, Any]:
        """Summarize why records were dropped or repaired across the whole run.

        A local run reports these counters from its in-process trace. Without the
        same block a fleet run reports nothing at all, and the rejection reasons
        are missing from exactly the runs whose workers an operator cannot read.
        Counting is pushed into Spark so no executor sends its trace to the
        driver: only one row per stage and reason comes back.
        """

        from pyspark.sql import functions as F

        events = extracted.select(F.explode_outer("observations.trace").alias("event")).select(
            F.col("event.stage").alias("stage"),
            F.col("event.record_actions").alias("record_actions"),
            F.coalesce(F.col("event.input_records"), F.lit(0)).alias("input_records"),
            F.coalesce(F.col("event.supported_records"), F.lit(0)).alias("supported_records"),
            F.col("event.skipped").alias("skipped"),
        )
        # The same relation funnel a local run reports: proposals, empty calls
        # and verifier support, one row back from the executors.
        funnel = events.select(
            F.sum(F.when(F.col("stage") == "relation_extraction", F.col("input_records"))).alias(
                "relation_candidates"
            ),
            F.sum(
                F.when(
                    (F.col("stage") == "relation_extraction")
                    & F.col("skipped").isNull()
                    & (F.col("input_records") == 0),
                    1,
                )
            ).alias("empty_relation_calls"),
            F.sum(F.when(F.col("stage") == "relation_verification", F.col("input_records"))).alias(
                "verification_candidates"
            ),
            F.sum(
                F.when(F.col("stage") == "relation_verification", F.col("supported_records"))
            ).alias("verification_supported"),
        ).collect()[0]
        stage_counts = {
            str(row["stage"]): int(row["events"])
            for row in events.groupBy("stage").agg(F.count(F.lit(1)).alias("events")).collect()
            if row["stage"]
        }
        reason_rows = (
            # The same stages a local run counts. Verification raises reasons of
            # its own, and folding those in here would report a different total
            # than the local pipeline does for the identical corpus, under a key
            # that gives a reader no way to tell the two apart.
            events.where(F.col("stage").isin(list(_RECORD_ACTION_STAGES)))
            .select(F.explode_outer("record_actions").alias("reason", "count"))
            .where(F.col("reason").isNotNull())
            .groupBy("reason")
            .agg(F.sum("count").alias("total"))
            .collect()
        )
        record_actions = {
            str(row["reason"]): int(row["total"] or 0)
            for row in reason_rows
            if int(row["total"] or 0)
        }
        totals = extracted.select(
            F.sum("observations.chunk_count").alias("chunks"),
            F.sum("observations.window_count").alias("windows"),
            F.sum(F.size(F.coalesce("observations.entities", F.array()))).alias("mentions"),
            F.sum(F.size(F.coalesce("observations.relations", F.array()))).alias("relations"),
        ).collect()[0]
        return {
            "strategy": "two_pass",
            "chunk_count": int(totals["chunks"] or 0),
            "window_count": int(totals["windows"] or 0),
            "entity_calls": stage_counts.get("entity_extraction", 0),
            "relation_calls": stage_counts.get("relation_extraction", 0),
            "verification_calls": stage_counts.get("relation_verification", 0),
            "document_context_calls": stage_counts.get("document_context_extraction", 0),
            "entity_mentions": int(totals["mentions"] or 0),
            "relation_observations": int(totals["relations"] or 0),
            "relation_candidates": int(funnel["relation_candidates"] or 0),
            "empty_relation_calls": int(funnel["empty_relation_calls"] or 0),
            "verification": {
                "candidates": int(funnel["verification_candidates"] or 0),
                "supported": int(funnel["verification_supported"] or 0),
            },
            "record_actions": dict(sorted(record_actions.items())),
        }

    def _without_excluded_folder_mentions(
        self,
        mentions: DataFrame,
        documents: DataFrame,
        chunks: DataFrame,
        ontology: OntologyProfile,
    ) -> DataFrame:
        """Leave out the folder mentions of documents the folder entity excludes.

        Exclusions are few and named by file id or location, so the driver
        resolves them from the documents in the folders they name.
        """

        from pyspark.sql import functions as F

        definition = ontology.folder_entity
        if definition is None:
            return mentions
        excluded = set(definition.excluded_file_ids)
        folders = sorted(definition.excluded_path_folders())
        if folders:
            # The folder as the definition reads paths: backslashes as
            # slashes, without empty folders.
            folder = F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(F.coalesce(F.col("folder"), F.lit("")), r"\\", "/"),
                    "/+",
                    "/",
                ),
                "^/|/$",
                "",
            )
            candidates = documents.where(folder.isin(folders)).select(
                "file_id", "folder", "source_uri"
            )
            excluded |= excluded_file_ids(
                definition, (row.asDict() for row in candidates.toLocalIterator())
            )
        if not excluded:
            return mentions
        excluded_chunks = chunks.where(F.col("file_id").isin(sorted(excluded))).select(
            F.col("id").alias("source_chunk_id")
        )
        excluded_mentions = (
            mentions.where(F.col("evidence_method") == F.lit(FOLDER_EVIDENCE_METHOD))
            .join(F.broadcast(excluded_chunks), "source_chunk_id", "left_semi")
            .select("id")
        )
        return mentions.join(excluded_mentions, "id", "left_anti")

    def _folder_links(
        self,
        mention_components: DataFrame,
        mention_to_node: DataFrame,
        chunks: DataFrame,
        ontology: OntologyProfile,
    ) -> DataFrame:
        """Link every entity of a document to its folder entity, as the local finalizer does.

        One link per entity and document, whichever of its mentions there is
        read, grounded in the folder line; none for an entity resolved into
        the folder entity itself. The rows are relation observations with
        their resolved endpoints, ready for edge assembly.
        """

        from pyspark.sql import functions as F

        definition = ontology.folder_entity
        if definition is None:
            raise ValueError("folder links need an ontology folder entity")
        relation = ontology.relation(definition.relation_type)
        relation_type = relation.name if relation is not None else definition.relation_type
        located = mention_components.join(
            mention_to_node.select(F.col("mention_id").alias("id"), "node_id"), "id"
        ).join(chunks.select(F.col("id").alias("source_chunk_id"), "file_id"), "source_chunk_id")
        is_folder = F.col("evidence_method") == F.lit(FOLDER_EVIDENCE_METHOD)
        folders = located.where(is_folder).select(
            "file_id",
            F.col("id").alias("target_entity_id"),
            F.col("node_id").alias("target_node_id"),
            "source_chunk_id",
            "quote",
            "start_offset",
            "end_offset",
        )
        members = located.where(~is_folder).select(
            "file_id",
            F.col("id").alias("member_id"),
            F.col("node_id").alias("source_node_id"),
        )
        prefix, _, suffix = FOLDER_LINK_DESCRIPTION.partition("{folder}")
        return (
            members.join(folders, "file_id")
            .where(F.col("source_node_id") != F.col("target_node_id"))
            .groupBy(
                "source_node_id",
                "target_node_id",
                "target_entity_id",
                "source_chunk_id",
                "quote",
                "start_offset",
                "end_offset",
            )
            .agg(F.min("member_id").alias("source_entity_id"))
            .select(
                _stable_id(
                    "folder_link",
                    F.col("source_node_id"),
                    F.col("target_node_id"),
                    F.col("source_chunk_id"),
                ).alias("id"),
                "source_entity_id",
                "target_entity_id",
                F.lit(relation_type).alias("relation_type"),
                F.concat(F.lit(prefix), F.col("quote"), F.lit(suffix)).alias("description"),
                "source_chunk_id",
                "quote",
                F.lit(1.0).alias("confidence"),
                "start_offset",
                "end_offset",
                F.lit(None).cast("string").alias("source_surface"),
                F.lit(None).cast("string").alias("target_surface"),
                F.lit(FOLDER_EVIDENCE_METHOD).alias("evidence_method"),
                F.lit(False).alias("verification_required"),
                F.array().cast(_EDGE_OBSERVATION_QUANTITIES_TYPE).alias("quantities"),
                F.lit(None).cast("string").alias("proposed_relation_type"),
                F.lit(None).cast("string").alias("rejected_record_key"),
                F.lit("reported").alias("assertion"),
                "source_node_id",
                "target_node_id",
            )
        )

    def _ontology_profile(self) -> OntologyProfile:
        """Load the same validated ontology used by extraction and quality checks.

        The profile is tiny driver-side configuration. Centralizing the load keeps
        post-resolution self-loop filtering and the full quality-rule join aligned
        without embedding ontology-specific predicates in Spark transformations.
        """

        from kg_processor.application.ontology import load_ontology

        return load_ontology(
            self.settings.ontology.profile_path,
            self.settings.graph.entity_types,
            self.settings.graph.relation_types,
            inline=self.settings.ontology.profile,
        ).profile

    def _hard_quality_errors(
        self,
        spark: SparkSession,
        nodes: DataFrame,
        edges: DataFrame,
        evidence: DataFrame,
        chunks: DataFrame,
    ) -> dict[str, int]:
        """Count all publication-blocking graph violations without collecting rows.

        This preserves the local quality contract with only counts reaching the
        driver. Every frame is first cut to the columns the checks read, and each
        check is its own Spark job: the checks share no work, and together in one
        job they hold their shuffles at once. Progress names the running check,
        so a run that stops here says which one it stopped on. Ontology rules are
        tiny configuration data and are broadcast by Spark when joined to canonical
        edges.
        """

        from pyspark.sql import functions as F

        embedding_valid = (
            F.col("embedding").isNotNull()
            & (F.size("embedding") == F.lit(self.settings.embedding.dimension))
        ).alias("embedding_valid")
        nodes = nodes.select("id", "primary_type", embedding_valid)
        edges = edges.select(
            "id", "source_node_id", "target_node_id", "relation_type", embedding_valid
        )
        evidence = evidence.select(
            "id",
            "subject_kind",
            "subject_id",
            "chunk_id",
            "start_offset",
            "end_offset",
            "quote",
        )

        valid_nodes = nodes.select(F.col("id").alias("valid_node_id"))
        orphan_edges = (
            edges.join(
                valid_nodes,
                edges.source_node_id == valid_nodes.valid_node_id,
                "left_anti",
            )
            .unionByName(
                edges.join(
                    valid_nodes,
                    edges.target_node_id == valid_nodes.valid_node_id,
                    "left_anti",
                )
            )
            .dropDuplicates(["id"])
        )
        duplicate_edges = (
            edges.groupBy("source_node_id", "target_node_id", "relation_type")
            .count()
            .where(F.col("count") > 1)
        )

        ontology = self._ontology_profile()
        # Rules are registered under the same bounded label rule that produces
        # persisted relation types, which is the only spelling a canonical edge
        # can carry, so every declared predicate reaches its own constraint.
        rule_rows = [
            (
                normalize_relation_type(label),
                definition.source_types,
                definition.target_types,
                definition.allow_self_loop,
            )
            for definition in ontology.relation_types
            for label in [definition.name, *definition.aliases]
        ]
        rules = spark.createDataFrame(
            rule_rows,
            "relation_key string, source_types array<string>, "
            "target_types array<string>, allow_self_loop boolean",
        )
        source_types = nodes.select(
            F.col("id").alias("source_node_id"),
            F.col("primary_type").alias("source_type"),
        )
        target_types = nodes.select(
            F.col("id").alias("target_node_id"),
            F.col("primary_type").alias("target_type"),
        )
        typed_edges = (
            edges.join(source_types, "source_node_id", "left")
            .join(target_types, "target_node_id", "left")
            .withColumn("relation_key", _normalized_relation_type(F.col("relation_type")))
            .join(F.broadcast(rules), "relation_key", "left")
        )
        self_loops = typed_edges.where(
            (F.col("source_node_id") == F.col("target_node_id"))
            & ~F.coalesce(F.col("allow_self_loop"), F.lit(False))
        )
        if ontology.mode == "open":
            ontology_errors = typed_edges.limit(0)
        else:
            type_errors = (
                (F.size("source_types") > 0)
                & ~F.array_contains("source_types", F.col("source_type"))
            ) | (
                (F.size("target_types") > 0)
                & ~F.array_contains("target_types", F.col("target_type"))
            )
            # A hybrid profile applies domain/range constraints to predicates it
            # declares while preserving grounded open-vocabulary predicates. A
            # closed profile additionally rejects every unknown predicate.
            ontology_errors = typed_edges.where(
                type_errors | (F.lit(ontology.mode == "closed") & F.col("allow_self_loop").isNull())
            )

        node_evidence = (
            evidence.where(F.col("subject_kind") == "node")
            .select(F.col("subject_id").alias("evidence_subject_id"))
            .dropDuplicates()
        )
        edge_evidence = (
            evidence.where(F.col("subject_kind") == "edge")
            .select(F.col("subject_id").alias("evidence_subject_id"))
            .dropDuplicates()
        )
        nodes_without_evidence = nodes.join(
            node_evidence,
            nodes.id == node_evidence.evidence_subject_id,
            "left_anti",
        )
        edges_without_evidence = edges.join(
            edge_evidence,
            edges.id == edge_evidence.evidence_subject_id,
            "left_anti",
        )

        evidence_rows = evidence.alias("evidence").join(
            chunks.select(
                F.col("id").alias("joined_chunk_id"),
                F.col("content").alias("chunk_content"),
                F.col("start_offset").alias("chunk_start_offset"),
            ).alias("chunk"),
            F.col("evidence.chunk_id") == F.col("chunk.joined_chunk_id"),
            "left",
        )
        local_start = F.col("evidence.start_offset") - F.col("chunk_start_offset")
        span_length = F.col("evidence.end_offset") - F.col("evidence.start_offset")
        source_span = F.substring(F.col("chunk_content"), local_start + F.lit(1), span_length)
        normalized_span = F.regexp_replace(F.lower(F.trim(source_span)), r"\s+", " ")
        normalized_quote = F.regexp_replace(F.lower(F.trim(F.col("evidence.quote"))), r"\s+", " ")
        invalid_evidence = evidence_rows.where(
            F.col("joined_chunk_id").isNull()
            | (local_start < 0)
            | (span_length <= 0)
            | (local_start + span_length > F.length("chunk_content"))
            | (normalized_span != normalized_quote)
        )

        checks = {
            "orphan_edges": orphan_edges,
            "duplicate_edges": duplicate_edges,
            "unapproved_self_loops": self_loops,
            "ontology_constraints": ontology_errors,
            "nodes_without_evidence": nodes_without_evidence,
            "edges_without_evidence": edges_without_evidence,
            "invalid_evidence_spans": invalid_evidence,
            "invalid_node_embeddings": nodes.where(~F.col("embedding_valid")),
            "invalid_edge_embeddings": edges.where(~F.col("embedding_valid")),
        }
        counts: dict[str, int] = {}
        for name, frame in checks.items():
            self._report_progress(
                "quality_checks", completed=0, total=1, message=f"Checking {name}"
            )
            counts[name] = frame.count()
        return counts

    def _identity_edges(
        self,
        spark: SparkSession,
        mentions: DataFrame,
        resolution_mentions: DataFrame,
        partitions: int,
    ) -> tuple[DataFrame, dict[str, int]]:
        """Build sparse identity links from full evidence and compact candidates.

        ``mentions`` retains every observation so exact aliases connect all source
        evidence. ``resolution_mentions`` contains one embedded representative per
        typed normalized name, bounding semantic comparisons and provider work by
        distinct identities rather than corpus-wide mention frequency.
        """

        from pyspark.sql import Window
        from pyspark.sql import functions as F

        surfaces = mentions.select(
            "id",
            "type",
            F.explode(F.array_distinct(F.concat(F.array("name"), F.col("aliases")))).alias(
                "surface"
            ),
        ).withColumn("normalized_surface", _normalized_name(F.col("surface")))
        surface_letters = F.regexp_replace(F.col("surface"), r"[^\p{L}]", "")
        surfaces = surfaces.withColumn(
            "short_initialism",
            (
                F.length(surface_letters).between(
                    MIN_SHORT_INITIALISM_LENGTH,
                    MAX_SHORT_INITIALISM_LENGTH,
                )
            )
            & (surface_letters == F.upper(surface_letters)),
        )
        expansion_flags = surfaces.groupBy("id").agg(
            F.max((~F.col("short_initialism")).cast("int")).alias("has_expanded_surface")
        )
        # A short acronym is exact identity evidence only when the mention has no
        # competing expansion. Expanded aliases still merge ordinary abbreviation
        # variants, while labels such as SRN can denote different named systems.
        exact_surfaces = surfaces.join(expansion_flags, "id").where(
            (~F.col("short_initialism")) | (F.col("has_expanded_surface") == F.lit(0))
        )
        exact_roots = exact_surfaces.groupBy("type", "normalized_surface").agg(
            F.min("id").alias("root")
        )
        exact_edges = (
            exact_surfaces.join(exact_roots, ["type", "normalized_surface"])
            .where(F.col("id") != F.col("root"))
            .select(F.col("id").alias("src"), F.col("root").alias("dst"))
            .dropDuplicates()
        )
        if not self.settings.graph.entity_resolution_enabled:
            return exact_edges, {
                "identity_candidates": 0,
                "identity_automatic_candidates": 0,
                "identity_adjudication_candidates": 0,
            }

        candidate_mentions = resolution_mentions
        # Candidate generation needs only identity, type, and normalized surface.
        # Carrying descriptions, quotes, aliases, and embeddings through the
        # replicated window shuffle multiplies its byte volume by three. Rich
        # payloads are joined only after candidate IDs have been deduplicated.
        # A single prefix block misses common variants whose informative token is
        # not first (surname-only people, qualified method names, and acronyms).
        # Each unique typed name therefore emits a bounded set of deterministic
        # prefix, token, and initials keys. Ordered-neighbor windows below keep
        # every key linear rather than constructing a block-wide self-join.
        token_surface = F.regexp_replace(
            F.lower(F.trim(F.col("name"))),
            r"[^\p{L}\p{N}]+",
            " ",
        )
        all_name_tokens = F.filter(
            F.split(F.trim(token_surface), r"\s+"),
            lambda token: F.length(token) >= F.lit(3),
        )
        name_tokens = F.slice(all_name_tokens, 1, _MAX_IDENTITY_BLOCK_TOKENS)
        initials = F.aggregate(
            all_name_tokens,
            F.lit(""),
            lambda value, token: F.concat(value, F.substring(token, 1, 1)),
        )
        initials_keys = (
            F.when(
                F.size(all_name_tokens) >= F.lit(2),
                F.array(F.concat(F.lit("initials:"), initials)),
            )
            .when(
                (F.size(all_name_tokens) == F.lit(1)) & (F.length("normalized_name") <= F.lit(10)),
                F.array(F.concat(F.lit("initials:"), F.col("normalized_name"))),
            )
            .otherwise(F.array().cast("array<string>"))
        )
        block_keys = F.array_distinct(
            F.concat(
                F.array(F.concat(F.lit("prefix:"), F.substring("normalized_name", 1, 3))),
                F.transform(
                    name_tokens,
                    lambda token: F.concat(F.lit("token:"), token),
                ),
                initials_keys,
            )
        )
        blocked_keys = candidate_mentions.select(
            "id", "type", "name", "normalized_name"
        ).withColumn("block_key", F.explode(block_keys))
        ambiguous_initialism_keys = (
            blocked_keys.where(F.col("block_key").startswith("initials:"))
            .withColumn(
                "expanded_name",
                F.when(F.size(all_name_tokens) >= F.lit(2), F.col("normalized_name")),
            )
            .groupBy("type", "block_key")
            .agg(F.countDistinct("expanded_name").alias("expanded_name_count"))
            .where(F.col("expanded_name_count") > F.lit(1))
            .select("type", "block_key")
            .withColumn("ambiguous_initialism", F.lit(True))
        )
        # Prefix blocks keep a narrow length policy. Token blocks
        # use coarser buckets so qualified and shortened names can meet without
        # allowing common words to form one hot partition. Initials blocks are
        # already compact and intentionally ignore surface length.
        length_bucket = (
            F.when(
                F.col("block_key").startswith("prefix:"),
                F.floor(F.length("normalized_name") / F.lit(4)),
            )
            .when(
                F.col("block_key").startswith("token:"),
                F.floor(F.length("normalized_name") / F.lit(12)),
            )
            .otherwise(F.lit(0))
        )
        candidate_buckets = F.when(
            F.col("block_key").startswith("initials:"),
            F.array(F.lit(0)),
        ).otherwise(
            F.array(
                length_bucket - F.lit(1),
                length_bucket,
                length_bucket + F.lit(1),
            )
        )
        blocked = (
            blocked_keys.join(ambiguous_initialism_keys, ["type", "block_key"], "left")
            .fillna(False, subset=["ambiguous_initialism"])
            .withColumn("candidate_bucket", F.explode(candidate_buckets))
        )
        neighbor_window = (
            Window.partitionBy("type", "block_key", "candidate_bucket")
            .orderBy("normalized_name", "id")
            .rowsBetween(
                -self.settings.graph.resolution_max_candidates_per_mention,
                -1,
            )
        )
        record = F.struct("id", "normalized_name")
        # One bounded ordered window yields the same preceding-neighbor set as N
        # separate ``lag`` branches, while requiring one exchange and sort rather
        # than repeating both for every configured candidate slot.
        neighboring = (
            blocked.withColumn("neighbors", F.collect_list(record).over(neighbor_window))
            .withColumn("right", F.explode("neighbors"))
            .select(
                record.alias("left"),
                "right",
                F.col("block_key").startswith("initials:").alias("initialism_candidate"),
                "ambiguous_initialism",
            )
        )
        pair_names = (
            neighboring.where(F.col("left.id") != F.col("right.id"))
            .select(
                F.least(F.col("left.id"), F.col("right.id")).alias("left_id"),
                F.greatest(F.col("left.id"), F.col("right.id")).alias("right_id"),
                F.when(
                    F.col("left.id") < F.col("right.id"),
                    F.col("left.normalized_name"),
                )
                .otherwise(F.col("right.normalized_name"))
                .alias("left_normalized_name"),
                F.when(
                    F.col("left.id") < F.col("right.id"),
                    F.col("right.normalized_name"),
                )
                .otherwise(F.col("left.normalized_name"))
                .alias("right_normalized_name"),
                "initialism_candidate",
                "ambiguous_initialism",
            )
            .groupBy(
                "left_id",
                "right_id",
                "left_normalized_name",
                "right_normalized_name",
            )
            .agg(
                F.max(F.col("initialism_candidate").cast("int"))
                .cast("boolean")
                .alias("initialism_candidate"),
                F.max(F.col("ambiguous_initialism").cast("int"))
                .cast("boolean")
                .alias("ambiguous_initialism"),
            )
            # Pairwise LLM decisions cannot observe other edges that will enter
            # connected-components resolution. When one typed initialism expands
            # to multiple full names, treating the initialism key as evidence can
            # transitively collapse those identities. Disable only that key;
            # independent lexical and embedding evidence remains available.
            .withColumn(
                "initialism_candidate",
                F.col("initialism_candidate") & ~F.col("ambiguous_initialism"),
            )
            .withColumn(
                "lexical_score",
                F.lit(1.0)
                - F.levenshtein("left_normalized_name", "right_normalized_name")
                / F.greatest(
                    F.length("left_normalized_name"),
                    F.length("right_normalized_name"),
                ),
            )
        )
        left_details = candidate_mentions.select(
            F.col("id").alias("left_id"),
            F.struct(*[F.col(name).alias(name) for name in _MENTION_FIELDS]).alias("left_mention"),
            F.col("embedding").alias("left_embedding"),
        )
        right_details = candidate_mentions.select(
            F.col("id").alias("right_id"),
            F.struct(*[F.col(name).alias(name) for name in _MENTION_FIELDS]).alias("right_mention"),
            F.col("embedding").alias("right_embedding"),
        )
        pairs = (
            pair_names.join(left_details, "left_id", "inner")
            .join(right_details, "right_id", "inner")
            .withColumn(
                "embedding_score",
                _cosine_score(F.col("left_embedding"), F.col("right_embedding")),
            )
            .drop(
                "left_normalized_name",
                "right_normalized_name",
                "ambiguous_initialism",
                "left_embedding",
                "right_embedding",
            )
            .where(
                (
                    (
                        F.col("lexical_score")
                        >= F.lit(self.settings.graph.resolution_candidate_threshold)
                    )
                    | (
                        (
                            F.col("embedding_score")
                            >= F.lit(self.settings.graph.resolution_candidate_threshold)
                        )
                        & (
                            F.col("lexical_score")
                            >= F.lit(self.settings.graph.resolution_embedding_lexical_floor)
                        )
                    )
                    | F.col("initialism_candidate")
                )
                & ~_numeric_mismatch(F.col("left_mention.name"), F.col("right_mention.name"))
            )
        )
        candidate_score = F.when(F.col("initialism_candidate"), F.col("embedding_score")).otherwise(
            F.greatest(F.col("lexical_score"), F.col("embedding_score"))
        )
        ranking = Window.partitionBy("left_id").orderBy(
            F.desc(candidate_score),
            F.desc("lexical_score"),
            F.desc("embedding_score"),
            F.asc("right_id"),
        )
        bounded = (
            pairs.withColumn("candidate_rank", F.row_number().over(ranking))
            .where(
                F.col("candidate_rank")
                <= F.lit(self.settings.graph.resolution_max_candidates_per_mention)
            )
            .repartition(partitions, "left_id")
        ).checkpoint(eager=True)
        auto_condition = (
            (F.col("lexical_score") >= F.lit(self.settings.graph.resolution_lexical_auto_merge))
            & (F.col("embedding_score") >= F.lit(MIN_EMBEDDING_FOR_LEXICAL_MERGE))
        ) | (
            (F.col("embedding_score") >= F.lit(self.settings.graph.resolution_embedding_auto_merge))
            & (F.col("lexical_score") >= F.lit(self.settings.graph.resolution_candidate_threshold))
        )
        candidate_counts = cast(
            "Any",
            bounded.agg(
                F.count(F.lit(1)).alias("total"),
                F.sum(F.when(auto_condition, F.lit(1)).otherwise(F.lit(0))).alias("automatic"),
            ).first(),
        )
        total_candidates = int(candidate_counts.total or 0)
        automatic_candidates = int(candidate_counts.automatic or 0)
        automatic = bounded.where(auto_condition).select(
            F.col("left_id").alias("src"), F.col("right_id").alias("dst")
        )
        adjudicated = self._adjudicated_identity_edges(
            spark,
            bounded.where(~auto_condition),
            _adaptive_provider_partitions(
                total_candidates - automatic_candidates,
                self.settings.distributed.spark_executor_instances,
                self.settings.distributed.spark_executor_cores,
                self.settings.graph.resolution_adjudication_batch_size,
                _provider_threads_per_task(self.settings),
            ),
        )
        return (
            exact_edges.unionByName(automatic).unionByName(adjudicated).dropDuplicates(),
            {
                "identity_candidates": total_candidates,
                "identity_automatic_candidates": automatic_candidates,
                "identity_adjudication_candidates": total_candidates - automatic_candidates,
            },
        )

    def _with_embeddings(
        self,
        spark: SparkSession,
        frame: DataFrame,
        *,
        id_column: str,
        text_column: Column,
        partitions: int,
    ) -> DataFrame:
        """Embed independent row batches on executors and join by stable identity."""

        materialized = frame.checkpoint(eager=True)
        selected = materialized.select(
            materialized[id_column].alias("record_id"),
            text_column.alias("embedding_text"),
        ).repartition(partitions, "record_id")
        # Shuffle blocks arrive in no fixed order. Sorting fixes which rows share
        # a provider batch, so a retried task sends the requests the first
        # attempt sent. The same holds for every provider phase below.
        selected = selected.sortWithinPartitions("record_id")
        settings_payload = self.settings.model_dump(mode="json")
        embedded = spark.createDataFrame(
            selected.rdd.mapPartitions(lambda rows: _embed_partition(rows, settings_payload)),
            "record_id string, embedding array<double>",
        )
        return materialized.join(
            embedded,
            materialized[id_column] == embedded.record_id,
            "left",
        ).drop("record_id")

    def _adjudicated_identity_edges(
        self,
        spark: SparkSession,
        candidates: DataFrame,
        partitions: int,
    ) -> DataFrame:
        """Adjudicate uncertain candidates in executor-local, bounded LLM batches."""

        selected = candidates.select(
            "left_id",
            "right_id",
            "lexical_score",
            "embedding_score",
            "left_mention",
            "right_mention",
        ).repartition(partitions, "left_id")
        selected = selected.sortWithinPartitions("left_id", "right_id")
        settings_payload = self.settings.model_dump(mode="json")
        decisions = spark.createDataFrame(
            selected.rdd.mapPartitions(lambda rows: _adjudicate_partition(rows, settings_payload)),
            "left_id string, right_id string, same_entity boolean, "
            "confidence double, reason string",
        )
        return decisions.where("same_entity = true").selectExpr("left_id AS src", "right_id AS dst")

    def _communities(
        self,
        graph_frame_type: Any,
        nodes: DataFrame,
        edges: DataFrame,
        evidence: DataFrame,
        graph_id: str,
    ) -> DataFrame:
        """Detect distributed communities and create bounded report-ready rows.

        Complete node-to-chunk provenance comes from the normalized evidence
        table. Node rows intentionally carry only a bounded convenience sample,
        which must not change co-mention topology for frequently observed nodes.
        """

        from pyspark import StorageLevel
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        spark = nodes.sparkSession
        structural_types = sorted(structural_relation_types(self._ontology_profile()))
        # Structural edges, and the folder line that grounds a folder entity,
        # say how the sources are filed rather than what they state.
        relation_topology = edges.where(
            (F.col("source_node_id") != F.col("target_node_id"))
            & ~F.col("relation_type").isin(structural_types)
        ).select(
            F.col("source_node_id").alias("src"),
            F.col("target_node_id").alias("dst"),
        )
        chunk_members = (
            evidence.where(
                (F.col("subject_kind") == F.lit("node"))
                & (F.coalesce(F.col("method"), F.lit("")) != F.lit(FOLDER_EVIDENCE_METHOD))
            )
            .select("chunk_id", F.col("subject_id").alias("node_id"))
            # ``drop_isolated_entities`` may intentionally remove canonical
            # vertices after evidence was assembled. Keep topology endpoints
            # within the exact vertex set handed to GraphFrames.
            .join(nodes.select(F.col("id").alias("node_id")), "node_id", "inner")
            .dropDuplicates()
        )
        chunk_window = Window.partitionBy("chunk_id").orderBy("node_id")
        bounded_chunk_members = chunk_members.withColumn(
            "chunk_rank", F.row_number().over(chunk_window)
        ).where(F.col("chunk_rank") <= F.lit(30))
        # A full co-mention clique grows quadratically (30 members produce 435
        # shuffle rows) although community topology only needs every member to be
        # connected. A deterministic anchor star retains that connectivity with at
        # most 29 rows and keeps corpus-wide topology linear in extracted mentions.
        anchored_members = bounded_chunk_members.withColumn(
            "anchor_node_id",
            F.first("node_id").over(
                chunk_window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            ),
        )
        co_mention_topology = (
            anchored_members.where(F.col("node_id") != F.col("anchor_node_id"))
            .select(
                F.col("anchor_node_id").alias("src"),
                F.col("node_id").alias("dst"),
            )
            .dropDuplicates()
        )
        forward_topology = relation_topology.unionByName(co_mention_topology).dropDuplicates()
        topology_edges = (
            forward_topology.unionByName(
                forward_topology.select(F.col("dst").alias("src"), F.col("src").alias("dst"))
            )
            .dropDuplicates()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        if not topology_edges.limit(1).count():
            topology_edges.unpersist(blocking=False)
            return spark.createDataFrame(
                [],
                "id string, graph_id string, stable_key string, level int, title string, "
                "summary string, rating double, rating_explanation string, "
                "member_node_ids array<string>, parent_community_id string, "
                "child_community_ids array<string>, suggested_questions array<string>, "
                "embedding array<double>",
            )
        topology = graph_frame_type(
            nodes.select("id"),
            # A self-message prevents two-node components from swapping labels
            # forever on alternating LPA iterations. These edges exist only in
            # the community topology and never become graph facts.
            topology_edges.unionByName(
                nodes.select(F.col("id").alias("src"), F.col("id").alias("dst"))
            ),
        )
        raw_labels = topology.labelPropagation(maxIter=_COMMUNITY_ITERATIONS)
        labels = raw_labels.checkpoint(eager=True)
        topology_edges.unpersist(blocking=False)
        raw_labels.unpersist(blocking=True)
        member_window = Window.partitionBy("label").orderBy(F.asc("id"))
        # Oversized labels are split deterministically into bounded report groups.
        expanded = labels.withColumn("member_rank", F.row_number().over(member_window)).withColumn(
            "subgroup",
            F.floor(
                (F.col("member_rank") - F.lit(1)) / F.lit(self.settings.graph.max_community_size)
            ),
        )
        bounded_members = (
            expanded.join(nodes.select("id", "name"), "id")
            .groupBy("label", "subgroup")
            .agg(
                F.sort_array(F.collect_set("id")).alias("member_node_ids"),
                F.sort_array(F.collect_set("name")).alias("member_names"),
            )
            .where(F.size("member_node_ids") >= F.lit(self.settings.graph.min_community_size))
        )
        return cast(
            "DataFrame",
            bounded_members.withColumn(
                "stable_key",
                _stable_id(
                    "community_key",
                    F.lit(0),
                    F.concat_ws(",", "member_node_ids"),
                    length=40,
                ),
            )
            .withColumn(
                "id",
                _stable_id("community", F.lit(graph_id), F.col("stable_key")),
            )
            .withColumn("graph_id", F.lit(graph_id))
            .withColumn("level", F.lit(0))
            .withColumn(
                "title",
                F.concat_ws(", ", F.slice("member_names", 1, 3)),
            )
            .withColumn(
                "summary",
                F.concat(
                    F.lit("Community containing "),
                    F.size("member_node_ids"),
                    F.lit(" connected entities."),
                ),
            )
            .withColumn("rating", F.lit(0.0))
            .withColumn("rating_explanation", F.lit("structural community"))
            .withColumn("parent_community_id", F.lit(None).cast("string"))
            .withColumn("child_community_ids", F.array().cast("array<string>"))
            .withColumn("suggested_questions", F.array().cast("array<string>"))
            .withColumn("embedding", F.lit(None).cast("array<double>"))
            .select(
                "id",
                "graph_id",
                "stable_key",
                "level",
                "title",
                "summary",
                "rating",
                "rating_explanation",
                "member_node_ids",
                "parent_community_id",
                "child_community_ids",
                "suggested_questions",
                "embedding",
            ),
        )

    def _community_reports(
        self,
        spark: SparkSession,
        communities: DataFrame,
        nodes: DataFrame,
        edges: DataFrame,
        evidence: DataFrame,
    ) -> tuple[DataFrame, DataFrame, int]:
        """Generate bounded community narratives independently on executors."""

        from pyspark import StorageLevel
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        community_count = communities.count()
        if not community_count:
            return (
                communities,
                spark.createDataFrame([], _EMPTY_SCHEMAS["community_findings"]),
                0,
            )
        provider_partitions = _adaptive_provider_partitions(
            community_count,
            self.settings.distributed.spark_executor_instances,
            self.settings.distributed.spark_executor_cores,
            provider_batch_size=COMMUNITY_BATCH_SIZE,
            threads_per_task=_provider_threads_per_task(self.settings),
        )

        memberships = communities.select(
            F.col("id").alias("community_id"),
            F.explode("member_node_ids").alias("node_id"),
        )
        member_context = (
            memberships.join(nodes, memberships.node_id == nodes.id, "inner")
            .groupBy("community_id")
            .agg(
                F.sort_array(
                    F.collect_list(
                        F.concat(
                            F.col("name"),
                            F.lit(" ["),
                            F.col("primary_type"),
                            F.lit("] degree="),
                            F.col("degree").cast("string"),
                            F.lit(": "),
                            F.substring(F.col("description"), 1, 500),
                        )
                    )
                ).alias("members")
            )
        )
        source_members = memberships.select(
            "community_id", F.col("node_id").alias("source_node_id")
        )
        target_members = memberships.select(
            F.col("community_id").alias("target_community_id"),
            F.col("node_id").alias("target_node_id"),
        )
        source_names = nodes.select(
            F.col("id").alias("source_node_id"), F.col("name").alias("source_name")
        )
        target_names = nodes.select(
            F.col("id").alias("target_node_id"), F.col("name").alias("target_name")
        )
        internal = (
            edges.join(source_members, "source_node_id", "inner")
            .join(target_members, "target_node_id", "inner")
            .where(F.col("community_id") == F.col("target_community_id"))
            .drop("target_community_id")
            .join(source_names, "source_node_id", "inner")
            .join(target_names, "target_node_id", "inner")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        relation_window = Window.partitionBy("community_id").orderBy(
            F.desc("weight"), F.asc("relation_type"), F.asc("id")
        )
        report_edges = (
            internal.withColumn("relation_rank", F.row_number().over(relation_window))
            .where(F.col("relation_rank") <= F.lit(DEFAULT_MAX_RELATIONS_PER_COMMUNITY))
            .withColumn(
                "relation_label", _relation_label(F.col("relation_type"), F.col("assertion"))
            )
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        relation_context = report_edges.groupBy("community_id").agg(
            F.collect_list(
                F.concat(
                    F.col("source_name"),
                    F.lit(" --"),
                    F.col("relation_label"),
                    F.lit("--> "),
                    F.col("target_name"),
                    F.lit(" weight="),
                    F.col("weight").cast("string"),
                    F.lit(": "),
                    F.substring(F.col("description"), 1, 500),
                )
            ).alias("relations")
        )
        evidence_window = Window.partitionBy("community_id", "edge_id").orderBy(
            "file_id", "page_number", "start_offset"
        )
        evidence_context = (
            report_edges.select("community_id", F.col("id").alias("edge_id"), "relation_label")
            .join(evidence, F.col("edge_id") == evidence.subject_id, "inner")
            .withColumn("evidence_rank", F.row_number().over(evidence_window))
            .where(F.col("evidence_rank") <= F.lit(MAX_EVIDENCE_QUOTES_PER_RELATION))
            .groupBy("community_id")
            .agg(
                F.slice(
                    F.collect_list(
                        F.concat(
                            F.col("relation_label"),
                            F.lit(": "),
                            F.substring(F.col("quote"), 1, 700),
                        )
                    ),
                    1,
                    MAX_EVIDENCE_QUOTES_PER_COMMUNITY,
                ).alias("evidence_quotes")
            )
        )
        # Density counts distinct member endpoint pairs, so parallel predicates and
        # opposite directions between the same two entities describe one connection
        # exactly as they do when the graph is assembled locally.
        endpoint_pair = F.concat_ws(
            "\u001f",
            F.least(F.col("source_node_id"), F.col("target_node_id")),
            F.greatest(F.col("source_node_id"), F.col("target_node_id")),
        )
        edge_stats = internal.groupBy("community_id").agg(
            F.count_distinct(endpoint_pair).alias("endpoint_pair_count"),
            F.avg("confidence").alias("mean_confidence"),
        )
        contexts = (
            communities.join(member_context, communities.id == member_context.community_id)
            .drop("community_id")
            .join(relation_context, communities.id == relation_context.community_id, "left")
            .drop("community_id")
            .join(evidence_context, communities.id == evidence_context.community_id, "left")
            .drop("community_id")
            .join(edge_stats, communities.id == edge_stats.community_id, "left")
            .drop("community_id")
            .withColumn("relations", F.coalesce("relations", F.array().cast("array<string>")))
            .withColumn(
                "evidence_quotes",
                F.coalesce("evidence_quotes", F.array().cast("array<string>")),
            )
            .repartition(provider_partitions, "id")
            .sortWithinPartitions("id")
        ).persist(StorageLevel.MEMORY_AND_DISK)
        # This single materialization traverses every context branch and fills
        # both persisted intermediates. Separate ``count()`` actions on
        # ``internal`` and ``report_edges`` would scan the same joins twice before
        # producing any report input, which becomes expensive on corpus-scale
        # graphs without changing cache contents or provider cardinality.
        contexts.count()
        settings_payload = self.settings.model_dump(mode="json")
        summaries = spark.createDataFrame(
            contexts.select(
                "id", "title", "members", "relations", "evidence_quotes"
            ).rdd.mapPartitions(
                lambda rows: _summarize_community_partition(rows, settings_payload)
            ),
            "community_id string, generated_title string, generated_summary string, "
            "generated_questions array<string>, "
            "findings array<struct<summary:string,explanation:string>>",
        )
        enriched = (
            contexts.join(summaries, contexts.id == summaries.community_id, "inner")
            .withColumn("title", F.col("generated_title"))
            .withColumn("summary", F.col("generated_summary"))
            .withColumn("suggested_questions", F.col("generated_questions"))
            .withColumn(
                "structural_rating",
                _structural_rating_column(
                    F.size("member_node_ids"),
                    F.coalesce("endpoint_pair_count", F.lit(0)),
                    F.size("relations"),
                    F.size("evidence_quotes"),
                    F.coalesce("mean_confidence", F.lit(0.0)),
                ),
            )
            .withColumn("rating", F.col("structural_rating.rating"))
            .withColumn("rating_explanation", F.col("structural_rating.explanation"))
            .drop("structural_rating")
        ).checkpoint(eager=True)
        # Both the community table and its exploded findings consume the same
        # generated report. The eager lineage cut above ensures each community is
        # summarized exactly once instead of replaying model calls per output table.
        contexts.unpersist(blocking=False)
        report_edges.unpersist(blocking=False)
        internal.unpersist(blocking=False)
        findings = (
            enriched.select(
                F.col("id").alias("community_id"),
                "graph_id",
                F.posexplode("findings").alias("finding_index", "finding"),
            )
            .withColumn(
                "id",
                _stable_id(
                    "finding",
                    F.col("community_id"),
                    F.col("finding_index"),
                    F.col("finding.summary"),
                ),
            )
            .select(
                "id",
                "graph_id",
                "community_id",
                F.col("finding.summary").alias("summary"),
                F.col("finding.explanation").alias("explanation"),
            )
        )
        return enriched.select(*communities.columns), findings, community_count

    def _merge_entity_descriptions(
        self,
        spark: SparkSession,
        nodes: DataFrame,
        entity_sources: DataFrame,
        mentions: DataFrame,
        mention_to_node: DataFrame,
        evidence: DataFrame,
    ) -> tuple[DataFrame, DataFrame, int]:
        """Synthesize repeated descriptions as independent executor-side calls."""

        from pyspark.sql import Window
        from pyspark.sql import functions as F

        descriptions = (
            mentions.join(
                mention_to_node,
                mentions.id == mention_to_node.mention_id,
                "inner",
            )
            .where(F.length(F.trim("description")) > 0)
            .select("node_id", F.trim("description").alias("description"))
            .dropDuplicates(["node_id", "description"])
        )
        description_window = Window.partitionBy("node_id").orderBy(
            F.desc(F.length("description")), F.asc("description")
        )
        description_context = (
            descriptions.withColumn("description_rank", F.row_number().over(description_window))
            .where(
                F.col("description_rank")
                <= F.lit(self.settings.graph.description_merge_max_descriptions)
            )
            .groupBy("node_id")
            .agg(F.collect_list("description").alias("descriptions"))
            .where(
                F.size("descriptions")
                >= F.lit(self.settings.graph.description_merge_min_observations)
            )
        )
        description_count = description_context.count()
        if not description_count:
            return nodes, entity_sources, 0
        provider_partitions = _adaptive_provider_partitions(
            description_count,
            self.settings.distributed.spark_executor_instances,
            self.settings.distributed.spark_executor_cores,
            provider_batch_size=DESCRIPTION_BATCH_SIZE,
            threads_per_task=_provider_threads_per_task(self.settings),
        )

        evidence_window = Window.partitionBy("subject_id").orderBy(
            "file_id", "page_number", "start_offset"
        )
        evidence_context = (
            evidence.where(F.col("subject_kind") == F.lit("node"))
            .withColumn("evidence_rank", F.row_number().over(evidence_window))
            .where(
                F.col("evidence_rank") <= F.lit(self.settings.graph.description_merge_max_evidence)
            )
            .groupBy(F.col("subject_id").alias("node_id"))
            .agg(F.collect_list("quote").alias("evidence_quotes"))
        )
        contexts = (
            description_context.join(
                nodes.select(F.col("id").alias("node_id"), "name", "primary_type"),
                "node_id",
                "inner",
            )
            .join(evidence_context, "node_id", "left")
            .withColumn(
                "evidence_quotes",
                F.coalesce("evidence_quotes", F.array().cast("array<string>")),
            )
            .repartition(provider_partitions, "node_id")
            .sortWithinPartitions("node_id")
        )
        settings_payload = self.settings.model_dump(mode="json")
        merged = spark.createDataFrame(
            contexts.rdd.mapPartitions(
                lambda rows: _merge_description_partition(rows, settings_payload)
            ),
            "node_id string, merged_description string",
        ).checkpoint(eager=True)
        # Nodes and per-file entity sources share one canonical description. A
        # durable boundary prevents Spark from invoking the model independently
        # for each branch and keeps both tables semantically identical on retries.
        updated_nodes = (
            nodes.join(merged, nodes.id == merged.node_id, "left")
            .withColumn(
                "description",
                F.coalesce("merged_description", "description"),
            )
            .drop("node_id", "merged_description")
        )
        updated_sources = (
            entity_sources.join(merged, "node_id", "left")
            .withColumn(
                "per_file_description",
                F.coalesce("merged_description", "per_file_description"),
            )
            .drop("merged_description")
        )
        return updated_nodes, updated_sources, description_count

    def _write_tables(
        self,
        spark: SparkSession,
        tables: dict[str, DataFrame],
        request: SparkFinalizationRequest,
    ) -> dict[str, DatasetTableManifest]:
        """Write every logical table before exposing their immutable file manifests."""

        from pyspark import StorageLevel

        root = _spark_uri(self.settings.distributed.artifact_uri or "")
        attempt_root = f"{root}/{request.run_id}/graph_dataset/attempt-{request.attempt:04d}"
        manifests: dict[str, DatasetTableManifest] = {}
        self._report_progress("write_tables", completed=0, total=len(_TABLE_NAMES))
        for table_index, name in enumerate(_TABLE_NAMES, start=1):
            frame = tables[name].persist(StorageLevel.MEMORY_AND_DISK)
            path = f"{attempt_root}/{name}"
            try:
                row_count = frame.count()
                rows_per_file = _TARGET_OUTPUT_ROWS_PER_FILE[name]
                # coalesce only ever removes partitions, so a small table lands
                # in a few files without a publication shuffle, while
                # maxRecordsPerFile still splits any oversized partition.
                published = frame.coalesce(max(1, math.ceil(row_count / rows_per_file)))
                (
                    published.write.mode("overwrite")
                    .option("maxRecordsPerFile", rows_per_file)
                    .parquet(path)
                )
                # Spark may publish only the ``_SUCCESS`` marker for an empty
                # DataFrame. Reading that directory back cannot infer a Parquet
                # schema, even though the write itself succeeded. Empty tables
                # have no immutable data files to expose in their manifest.
                committed_files = [] if row_count == 0 else spark.read.parquet(path).inputFiles()
                manifests[name] = DatasetTableManifest(
                    name=name,
                    row_count=row_count,
                    files=[
                        DatasetFile(uri=uri, size_bytes=_hadoop_file_size(spark, uri))
                        for uri in committed_files
                    ],
                )
                self._report_progress(
                    "write_tables",
                    completed=table_index,
                    total=len(_TABLE_NAMES),
                    message=f"Writing graph tables: {table_index}/{len(_TABLE_NAMES)} ({name})",
                )
            finally:
                frame.unpersist(blocking=False)
        return manifests


def _connected_nodes(edges: DataFrame, structural_types: Sequence[str]) -> DataFrame:
    """The nodes an isolate filter keeps, as ``graph_merge.prune_isolated_entities`` reads them.

    A node is kept by a relation that is not structural, and the target of a
    structural edge is kept when its source is.
    """

    from pyspark.sql import functions as F

    def endpoints(frame: DataFrame) -> DataFrame:
        return (
            frame.select(F.col("source_node_id").alias("node_id"))
            .unionByName(frame.select(F.col("target_node_id").alias("node_id")))
            .dropDuplicates(["node_id"])
        )

    structural = F.col("relation_type").isin(list(structural_types))
    connected = endpoints(edges.where(~structural))
    if not structural_types:
        return connected
    reached = (
        edges.where(structural)
        .join(connected.select(F.col("node_id").alias("source_node_id")), "source_node_id")
        .select(F.col("target_node_id").alias("node_id"))
    )
    return connected.unionByName(reached).dropDuplicates(["node_id"])


def _column_names(ddl_schema: str) -> list[str]:
    """The column names of a DDL schema string, in order."""

    names: list[str] = []
    depth = 0
    for part in ddl_schema.split(","):
        stripped = part.strip()
        if depth == 0 and stripped:
            names.append(stripped.split(" ", 1)[0])
        depth += part.count("<") - part.count(">")
    return names


def _explode_struct(frame: DataFrame, field: str, empty_schema: str) -> DataFrame:
    """Explode structs or return a schema-correct table for an all-empty array.

    JSON inference represents an array that is empty in every input as
    ``array<string>``. Explicit schemas keep empty graph tables writable and make
    their Parquet contracts independent of whether a particular run has rows.
    """

    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StructType

    data_type: Any = frame.schema
    for part in field.split("."):
        if not isinstance(data_type, StructType) or part not in data_type.names:
            return frame.sparkSession.createDataFrame([], empty_schema)
        data_type = data_type[part].dataType
    if not isinstance(data_type, ArrayType) or not isinstance(data_type.elementType, StructType):
        return frame.sparkSession.createDataFrame([], empty_schema)

    return (
        frame.select(F.explode_outer(F.col(field)).alias("row"))
        .where(F.col("row").isNotNull())
        .select("row.*")
    )


def _bounded_group_values(
    frame: DataFrame,
    *,
    group_columns: tuple[str, ...],
    value_column: str,
    output_column: str,
    limit: int,
) -> DataFrame:
    """Collect deterministic convenience arrays without unbounded group state.

    Spark's ``collect_set`` must retain every value in a hot key before a later
    ``slice`` can trim the result. Deduplicating and ranking first means at most
    ``limit`` rows reach the final aggregate for any canonical node or edge. The
    normalized evidence and observation tables remain complete; this helper is
    only for denormalized arrays carried on graph rows for compact inspection.
    """

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    if not group_columns:
        raise ValueError("bounded group collection requires grouping columns")
    if limit <= 0:
        raise ValueError("bounded group collection limit must be positive")
    selected = (
        frame.select(*group_columns, F.col(value_column).alias("bounded_value"))
        .where(F.col("bounded_value").isNotNull())
        .dropDuplicates([*group_columns, "bounded_value"])
    )
    rank_window = Window.partitionBy(*group_columns).orderBy(F.asc("bounded_value"))
    return (
        selected.withColumn("bounded_rank", F.row_number().over(rank_window))
        .where(F.col("bounded_rank") <= F.lit(limit))
        .groupBy(*group_columns)
        .agg(F.sort_array(F.collect_list("bounded_value")).alias(output_column))
    )


def _normalized_name_value(value: str | None) -> str | None:
    """Return the canonical entity merge key for one surface, preserving nulls."""

    return None if value is None else normalize_entity_name(value)


def _normalized_name(value: Column) -> Column:
    """Apply the canonical entity-name identity rule to a Spark column.

    Node IDs are hashed from this value, so the distributed engine evaluates the
    reference rule itself instead of an equivalent-looking SQL expression: Spark
    SQL offers neither Unicode NFC composition nor full case folding, and either
    gap mints a different node ID for the same entity, which breaks idempotent
    merges of local and distributed runs into shared tables. The rule runs once
    per surface; every later identity stage joins on the resulting key.
    """

    from pyspark.sql import functions as F

    return F.udf(_normalized_name_value, "string", useArrow=True)(value)


def _normalized_relation_type_value(value: str | None) -> str | None:
    """Return the bounded canonical relation label for one predicate, preserving nulls."""

    return None if value is None else normalize_relation_type(value)


def _normalized_relation_type(value: Column) -> Column:
    """Apply the canonical bounded relation-label rule to a Spark column.

    Canonical edge IDs are hashed from this value and ontology rules are keyed by
    it, so the distributed engine reuses the reference rule for the same reason
    canonical names do: case folding and Unicode-aware whitespace splitting have
    no SQL equivalent that agrees on every predicate an extractor can emit.
    """

    from pyspark.sql import functions as F

    return F.udf(_normalized_relation_type_value, "string", useArrow=True)(value)


def _claimed_only(assertion: Column, window: WindowSpec | None = None) -> Column:
    """Aggregate assertions as ``graph_merge.claimed_only``: claimed only when every one is.

    Without ``window`` it is a group aggregate; with one, the same over each row's window.
    """

    from pyspark.sql import functions as F

    all_claimed = F.min(assertion == F.lit("claimed"))
    if window is not None:
        all_claimed = all_claimed.over(window)
    return F.when(all_claimed, F.lit("claimed")).otherwise(F.lit("reported"))


def _relation_label(relation_type: Column, assertion: Column) -> Column:
    """The relation type a report context names, as ``community_reports.relation_label``."""

    from pyspark.sql import functions as F

    return F.concat(
        relation_type,
        F.when(assertion == F.lit("claimed"), F.lit(CLAIMED_RELATION_MARK)).otherwise(F.lit("")),
    )


def _observation_weight(confidence: Column) -> Column:
    """Return the per-observation edge weight contributed by one relation row.

    Distributed stage artifacts persist confidence rather than the weight
    extraction derives from it, so that derivation is repeated here. The floor
    keeps a low-confidence assertion contributing measurable support.
    """

    from pyspark.sql import functions as F

    return F.greatest(F.coalesce(confidence, F.lit(0.0)), F.lit(_MIN_OBSERVATION_WEIGHT))


def _structural_rating_value(
    member_count: int | None,
    endpoint_pair_count: int | None,
    reported_relation_count: int | None,
    evidence_quote_count: int | None,
    mean_edge_confidence: float | None,
) -> tuple[float, str]:
    """Score one community with the shared structural rating rule, tolerating nulls."""

    return structural_rating(
        member_count or 0,
        endpoint_pair_count or 0,
        reported_relation_count or 0,
        evidence_quote_count or 0,
        mean_edge_confidence or 0.0,
    )


def _structural_rating_column(
    member_count: Column,
    endpoint_pair_count: Column,
    reported_relation_count: Column,
    evidence_quote_count: Column,
    mean_edge_confidence: Column,
) -> Column:
    """Rate one community row with the reference structural rating rule.

    Both engines publish ratings into the same community table, so the rule runs
    from its single definition rather than from a parallel SQL translation that
    can round or clamp differently. Communities are few relative to graph rows,
    which keeps per-row evaluation immaterial to finalization cost.
    """

    from pyspark.sql import functions as F

    rate = F.udf(
        _structural_rating_value,
        "struct<rating:double,explanation:string>",
        useArrow=True,
    )
    return rate(
        member_count,
        endpoint_pair_count,
        reported_relation_count,
        evidence_quote_count,
        mean_edge_confidence,
    )


def _stable_id(prefix: str, *parts: Column, length: int = 32) -> Column:
    """Build the same SHA-256 semantic IDs as ``domain.ids.stable_id`` in Spark SQL."""

    from pyspark.sql import functions as F

    joined = F.concat_ws("\u001f", *[part.cast("string") for part in parts])
    return F.concat(F.lit(f"{prefix}_"), F.substring(F.sha2(joined, 256), 1, length))


def _adaptive_partitions(
    row_count: int,
    executor_instances: int,
    executor_cores: int,
) -> int:
    """Size shuffles from rows while exposing work to every executor core."""

    execution_slots = executor_instances * executor_cores
    return max(execution_slots, math.ceil(row_count / _TARGET_ROWS_PER_PARTITION), 1)


def _effective_shuffle_partitions(
    row_count: int,
    executor_instances: int,
    executor_cores: int,
    configured_partitions: int,
) -> int:
    """Honor an operator override or derive a conservative SQL partition count.

    Explicit Spark tuning is commonly based on measured row width, skew, and spill,
    information a generic row-count heuristic cannot recover. Zero retains adaptive
    defaults; a positive value remains authoritative throughout finalization.
    """

    if configured_partitions < 0:
        raise ValueError("configured Spark shuffle partitions must not be negative")
    if configured_partitions:
        return configured_partitions
    return _adaptive_partitions(row_count, executor_instances, executor_cores)


def _connected_component_rows(
    vertex_ids: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Resolve a bounded identity graph with the local engine's union-find.

    The Spark caller checks strict row limits before collecting. The same
    lexical-minimum root rule labels components here as in local resolution,
    so both engines agree on a representative for one cluster.
    """

    union_find = UnionFind(list(vertex_ids))
    for left_id, right_id in edges:
        union_find.union(left_id, right_id)
    return sorted((vertex_id, union_find.find(vertex_id)) for vertex_id in union_find.parent)


def _provider_threads_per_task(settings: Settings) -> int:
    """Share finalization's provider concurrency among Spark's execution slots.

    Every slot (an executor core) runs one task, and each provider task runs
    this many batches at once, so the fleet keeps slots x threads provider
    calls in flight. Rounding down keeps that total within the configured
    concurrency; a slot always runs at least one call, as a sequential task
    would. Because the total is what is configured, adding executors gives
    each task a smaller share instead of multiplying the load a model server
    sees.
    """

    execution_slots = (
        settings.distributed.spark_executor_instances * settings.distributed.spark_executor_cores
    )
    return max(1, settings.graph.finalization_provider_concurrency // execution_slots)


def _adaptive_provider_partitions(
    row_count: int,
    executor_instances: int,
    executor_cores: int,
    provider_batch_size: int,
    threads_per_task: int,
) -> int:
    """Give each provider task one round of batches for its threads.

    SQL transformations benefit from large partitions, but an LLM-backed task
    waits on network requests of variable latency. A task runs
    ``threads_per_task`` batches at once, so a partition of that many batches
    finishes in about one request's latency and a stage's tail is no longer
    than its slowest request. Fewer, larger partitions are what let the threads
    work: a partition of one batch leaves all but one of them idle.

    Two tasks per execution slot remain the minimum, so Spark can hand a free
    slot more work while another waits on a slow request. At corpus scale the
    maximum of 256 tasks per slot bounds scheduler metadata; only those capped
    partitions hold more than one round, which their threads work through in
    order. Partition size does not set memory: rows stream from Spark, and a
    task holds only the batches it has submitted.
    """

    if row_count < 0:
        raise ValueError("provider row count must not be negative")
    if executor_instances <= 0 or executor_cores <= 0:
        raise ValueError("Spark executor counts and cores must be positive")
    if provider_batch_size <= 0:
        raise ValueError("provider batch size must be positive")
    if threads_per_task <= 0:
        raise ValueError("provider threads per task must be positive")
    if row_count == 0:
        return 1
    execution_slots = executor_instances * executor_cores
    minimum = min(row_count, execution_slots * _MIN_PROVIDER_PARTITIONS_PER_SLOT)
    target = math.ceil(row_count / (provider_batch_size * threads_per_task))
    maximum = execution_slots * _MAX_PROVIDER_PARTITIONS_PER_SLOT
    return min(row_count, max(minimum, min(target, maximum)))


def _numeric_mismatch(left_name: Column, right_name: Column) -> Column:
    """Mirror ``entity_resolution._has_numeric_mismatch`` for a pair of raw names.

    Contract numbers, model generations and version labels differ by one digit
    yet score above the lexical auto-merge threshold, so a pair whose numeric
    token sequences differ is neither merged nor sent for adjudication.
    """

    from pyspark.sql import functions as F

    pattern = r"(\d+(?:[.,]\d+)?)"
    return F.regexp_extract_all(left_name, F.lit(pattern)) != F.regexp_extract_all(
        right_name, F.lit(pattern)
    )


def _cosine_score(left: Column, right: Column) -> Column:
    """Return cosine similarity using native higher-order Spark expressions."""

    from pyspark.sql import functions as F

    products = F.zip_with(left, right, lambda x, y: x * y)
    numerator = F.aggregate(products, F.lit(0.0), lambda total, value: total + value)
    left_squared = F.transform(left, lambda value: value * value)
    right_squared = F.transform(right, lambda value: value * value)
    left_norm = F.sqrt(F.aggregate(left_squared, F.lit(0.0), lambda total, value: total + value))
    right_norm = F.sqrt(F.aggregate(right_squared, F.lit(0.0), lambda total, value: total + value))
    valid = (
        left.isNotNull()
        & right.isNotNull()
        & (F.size(left) == F.size(right))
        & (left_norm > F.lit(0.0))
        & (right_norm > F.lit(0.0))
    )
    return F.when(valid, numerator / (left_norm * right_norm)).otherwise(F.lit(0.0))


def _map_provider_batches[TRow, TResult](
    rows: Iterable[TRow],
    batch_size: int,
    call: Callable[[Sequence[TRow]], list[TResult]],
    settings: Settings,
) -> Iterator[TResult]:
    """Run one partition's provider batches concurrently, yielding rows in input order.

    Batches are cut from the partition's rows exactly as a sequential loop would
    cut them, and results come back in that order whatever order the provider
    answers in, so concurrency changes how long a partition takes and never
    what it produces. Rows are read on this thread only; the pool's threads see
    the batches handed to them. A failed call raises here, failing the Spark
    task, whose retry recomputes the whole partition: Spark keeps no rows from
    a failed attempt, so none are written twice.
    """

    threads = _provider_threads_per_task(settings)
    results = bounded_map(
        _executor_provider_pool(threads),
        call,
        batched(rows, batch_size),
        buffersize=threads * _QUEUED_PROVIDER_BATCHES_PER_THREAD,
    )
    for batch_results in results:
        yield from batch_results


def _embed_partition(
    rows: Any,
    settings_payload: dict[str, Any],
) -> Any:
    """Embed one executor partition in provider-sized batches with bounded memory."""

    from kg_processor.ports.embeddings import EmbedOptions

    settings = Settings.model_validate(settings_payload)
    provider = _executor_embedding_provider(settings)
    options = EmbedOptions(
        model=settings.embedding.model,
        dimension=settings.embedding.dimension,
        batch_size=settings.embedding.batch_size,
    )
    yield from _map_provider_batches(
        rows,
        settings.embedding.batch_size,
        lambda batch: _embed_row_batch(provider, options, batch),
        settings,
    )


def _embed_row_batch(
    provider: Any,
    options: Any,
    rows: Sequence[Any],
) -> list[tuple[str, list[float]]]:
    """Call one embedding provider batch and preserve stable row order."""

    vectors = provider.embed([str(row.embedding_text or "") for row in rows], options)
    return [
        (str(row.record_id), [float(value) for value in vector])
        for row, vector in zip(rows, vectors, strict=True)
    ]


def _adjudicate_partition(
    rows: Any,
    settings_payload: dict[str, Any],
) -> Any:
    """Adjudicate uncertain identity rows in bounded executor-local LLM calls."""

    from kg_processor.ports.llm import StructuredCompletionProvider

    settings = Settings.model_validate(settings_payload)
    llm = _executor_llm_provider(settings)
    if not isinstance(llm, StructuredCompletionProvider):
        raise ValueError("Spark entity resolution requires structured LLM completion")
    yield from _map_provider_batches(
        rows,
        settings.graph.resolution_adjudication_batch_size,
        lambda batch: _adjudicate_row_batch(batch, settings, llm),
        settings,
    )


def _adjudicate_row_batch(
    rows: Sequence[Any],
    settings: Settings,
    llm: Any,
) -> list[tuple[str, str, bool, float, str]]:
    """Convert Spark rows to domain records and return reconciled decisions."""

    from kg_processor.application.entity_resolution import adjudicate_resolution_candidates
    from kg_processor.domain.extraction import EntityMention, ResolutionCandidate

    mentions_by_id: dict[str, EntityMention] = {}
    candidates: list[ResolutionCandidate] = []
    for row in rows:
        left = EntityMention.model_validate(row.left_mention.asDict(recursive=True))
        right = EntityMention.model_validate(row.right_mention.asDict(recursive=True))
        mentions_by_id[left.id] = left
        mentions_by_id[right.id] = right
        candidates.append(
            ResolutionCandidate(
                left_id=str(row.left_id),
                right_id=str(row.right_id),
                lexical_score=float(row.lexical_score),
                embedding_score=float(row.embedding_score),
            )
        )
    decisions = adjudicate_resolution_candidates(
        candidates,
        list(mentions_by_id.values()),
        llm=llm,
        model=settings.llm.model,
        timeout_seconds=settings.llm.timeout_seconds,
        batch_size=settings.graph.resolution_adjudication_batch_size,
        minimum_merge_confidence=settings.graph.resolution_llm_merge_min_confidence,
        seed=settings.graph.deterministic_seed,
    )
    return [
        (
            decision.left_id,
            decision.right_id,
            decision.same_entity,
            decision.confidence,
            decision.reason,
        )
        for decision in decisions
    ]


def _summarize_community_partition(
    rows: Any,
    settings_payload: dict[str, Any],
) -> Any:
    """Generate each bounded community report in its executor partition."""

    from kg_processor.application.enrichment_batching import summarize_community_requests
    from kg_processor.ports.llm import CommunitySummaryRequest

    settings = Settings.model_validate(settings_payload)
    llm = _executor_enrichment_llm_provider(settings)

    def summarize_batch(
        batch: Sequence[Any],
    ) -> list[tuple[str, str, str, list[str], list[dict[str, str]]]]:
        """Convert one bounded Spark row batch into provider-generated tuples."""

        requests = [
            CommunitySummaryRequest(
                title_seed=str(row.title),
                members=[str(value) for value in row.members],
                relations=[str(value) for value in row.relations],
                evidence_quotes=[str(value) for value in row.evidence_quotes],
            )
            for row in batch
        ]
        results = summarize_community_requests(
            llm,
            requests,
            model=settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            seed=settings.graph.deterministic_seed,
        )
        return [
            (
                str(row.id),
                result.title,
                result.summary,
                result.suggested_questions,
                [
                    {"summary": summary, "explanation": explanation}
                    for summary, explanation in result.findings
                ],
            )
            for row, result in zip(batch, results, strict=True)
        ]

    yield from _map_provider_batches(rows, COMMUNITY_BATCH_SIZE, summarize_batch, settings)


def _merge_description_partition(
    rows: Any,
    settings_payload: dict[str, Any],
) -> Any:
    """Merge repeated node descriptions without returning rows to the driver."""

    from kg_processor.application.enrichment_batching import merge_description_requests
    from kg_processor.ports.llm import DescriptionMergeRequest

    settings = Settings.model_validate(settings_payload)
    llm = _executor_enrichment_llm_provider(settings)

    def merge_batch(batch: Sequence[Any]) -> list[tuple[str, str]]:
        """Return non-empty merged descriptions for one bounded Spark row batch."""

        requests = [
            DescriptionMergeRequest(
                entity_name=str(row.name),
                entity_type=str(row.primary_type),
                descriptions=[str(value) for value in row.descriptions],
                evidence=[str(value) for value in row.evidence_quotes],
            )
            for row in batch
        ]
        results = merge_description_requests(
            llm,
            requests,
            model=settings.llm.model,
            timeout_seconds=settings.llm.timeout_seconds,
            seed=settings.graph.deterministic_seed,
        )
        return [
            (str(row.node_id), result.description.strip()[:1000])
            for row, result in zip(batch, results, strict=True)
            if result.description.strip()
        ]

    yield from _map_provider_batches(rows, DESCRIPTION_BATCH_SIZE, merge_batch, settings)


def _executor_embedding_provider(settings: Settings) -> Any:
    """Reuse one embedding adapter and local model per Python worker process.

    Spark invokes ``mapPartitions`` repeatedly while building mentions, nodes,
    edges, and communities. Python workers survive those tasks, so retaining the
    adapter prevents SentenceTransformers from reloading identical model weights
    for every partition and lets remote adapters retain HTTP connection pools.
    """

    from kg_processor.factories import build_embedding_provider

    key = _executor_provider_key("embedding", settings)
    with _EXECUTOR_PROVIDER_LOCK:
        provider = _EXECUTOR_EMBEDDING_PROVIDERS.get(key)
        if provider is None:
            provider = build_embedding_provider(settings)
            _EXECUTOR_EMBEDDING_PROVIDERS[key] = provider
        return provider


def _executor_llm_provider(settings: Settings) -> Any:
    """Reuse one LLM adapter and its keep-alive pool per Python worker process.

    Identity adjudication, description synthesis, and community reporting are
    separate Spark phases but use the same configured provider. Reuse removes
    repeated adapter setup while preserving phase-specific prompts and outputs.
    """

    from kg_processor.factories import build_llm_provider

    key = _executor_provider_key("llm", settings)
    with _EXECUTOR_PROVIDER_LOCK:
        provider = _EXECUTOR_LLM_PROVIDERS.get(key)
        if provider is None:
            provider = build_llm_provider(settings, _executor_consumption_collector(settings))
            _EXECUTOR_LLM_PROVIDERS[key] = provider
        return provider


def _executor_consumption_collector(settings: Settings) -> ConsumptionCollector:
    """Persist billed executor calls independently of Spark task recomputation.

    Provider work can be billed even when Spark discards a failed partition.
    Immutable event objects therefore live outside the partition's output commit
    and survive retries; the finalizer merges them once by event identity.
    """

    from kg_processor.factories import build_blob_store

    blob = build_blob_store(settings)
    if blob is None:
        raise ValueError("Spark consumption accounting requires shared object storage")

    def persist(event: ConsumptionEvent) -> None:
        """Write one immutable provider event under the durable run's prefix."""

        blob.put(
            f"{settings.job.job_id}/consumption/{event.event_id}.json",
            event.model_dump_json().encode("utf-8"),
            "application/json",
        )

    return ConsumptionCollector(
        settings.job.graph_id,
        job_id=settings.job.job_id,
        run_id=settings.job.job_id,
        sink=persist,
    )


def _executor_consumption_events(
    spark: SparkSession, settings: Settings
) -> Iterator[ConsumptionEvent]:
    """Stream the run's durable executor event ledger after all tables materialize."""

    prefix = _spark_uri(settings.distributed.artifact_uri or "")
    location = f"{prefix}/{settings.job.job_id}/consumption"
    jvm = spark._jvm
    if jvm is None:
        raise RuntimeError("Spark consumption accounting requires the Hadoop filesystem bridge")
    path = jvm.org.apache.hadoop.fs.Path(location)
    filesystem = path.getFileSystem(spark._jsc.hadoopConfiguration())
    if not filesystem.exists(path):
        return
    for row in spark.read.text(location).toLocalIterator():
        yield ConsumptionEvent.model_validate_json(row.value)


def _executor_enrichment_llm_provider(settings: Settings) -> Any:
    """Reuse an LLM wrapped in the durable pipeline cache for Spark enrichment."""

    from kg_processor.application.enrichment_cache import CachedEnrichmentLlmProvider
    from kg_processor.factories import build_cache

    key = _executor_provider_key("enrichment", settings)
    with _EXECUTOR_PROVIDER_LOCK:
        provider = _EXECUTOR_ENRICHMENT_PROVIDERS.get(key)
    if provider is not None:
        return provider
    llm = _executor_llm_provider(settings)
    cache = build_cache(settings)
    provider = (
        CachedEnrichmentLlmProvider(
            llm,
            cache,
            graph_id=settings.job.graph_id,
            provider=settings.llm.provider,
            model=settings.llm.model,
        )
        if cache is not None
        else llm
    )
    with _EXECUTOR_PROVIDER_LOCK:
        existing = _EXECUTOR_ENRICHMENT_PROVIDERS.setdefault(key, provider)
        if existing is provider and cache is not None:
            _EXECUTOR_PIPELINE_CACHES[key] = cache
    if existing is not provider and cache is not None:
        close = getattr(cache, "close", None)
        if callable(close):
            close()
    return existing


def _executor_provider_pool(threads: int) -> ThreadPoolExecutor:
    """Reuse one provider thread pool of each size per Python worker process.

    Spark runs one task at a time in a Python worker and reuses the worker for
    the next, so the pool's threads outlive a partition. Adapters that keep
    state per thread, such as a Snowflake session, then open it once per
    thread rather than once per partition.
    """

    with _EXECUTOR_PROVIDER_LOCK:
        pool = _EXECUTOR_PROVIDER_POOLS.get(threads)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="provider")
            _EXECUTOR_PROVIDER_POOLS[threads] = pool
        return pool


def _executor_provider_key(kind: str, settings: Settings) -> str:
    """Hash effective settings into a non-secret process-local cache identity."""

    payload = settings.model_dump_json().encode("utf-8")
    return f"{kind}:{hashlib.sha256(payload).hexdigest()}"


def _clear_executor_provider_caches() -> None:
    """Release process-local adapters, caches and provider threads for tests and shutdown.

    Production Python workers normally exit with their executor. Explicit closure
    is useful to tests and embedded Spark sessions that execute several unrelated
    configurations in one long-lived interpreter.
    """

    with _EXECUTOR_PROVIDER_LOCK:
        pools = list(_EXECUTOR_PROVIDER_POOLS.values())
        providers = [
            *_EXECUTOR_EMBEDDING_PROVIDERS.values(),
            *_EXECUTOR_LLM_PROVIDERS.values(),
        ]
        caches = list(_EXECUTOR_PIPELINE_CACHES.values())
        _EXECUTOR_PROVIDER_POOLS.clear()
        _EXECUTOR_EMBEDDING_PROVIDERS.clear()
        _EXECUTOR_LLM_PROVIDERS.clear()
        _EXECUTOR_ENRICHMENT_PROVIDERS.clear()
        _EXECUTOR_PIPELINE_CACHES.clear()
    # Calls still running finish before the adapters they use are closed.
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)
    for provider in providers:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    for cache in caches:
        close = getattr(cache, "close", None)
        if callable(close):
            close()


atexit.register(_clear_executor_provider_caches)


def _spark_uri(uri: str) -> str:
    """Translate public S3 URIs to Hadoop's S3A scheme and remove trailing slashes."""

    normalized = uri.rstrip("/")
    return f"s3a://{normalized[5:]}" if normalized.startswith("s3://") else normalized


def _hadoop_file_size(spark: SparkSession, uri: str) -> int:
    """Read one committed file length through the configured Hadoop filesystem."""

    context = cast("Any", spark.sparkContext)
    path = context._jvm.org.apache.hadoop.fs.Path(uri)  # noqa: SLF001
    configuration = context._jsc.hadoopConfiguration()  # noqa: SLF001
    return int(path.getFileSystem(configuration).getFileStatus(path).getLen())


def _delete_hadoop_prefix(spark: SparkSession, uri: str) -> None:
    """Delete one attempt-owned Spark scratch prefix without masking run results."""

    try:
        context = cast("Any", spark.sparkContext)
        path = context._jvm.org.apache.hadoop.fs.Path(uri)  # noqa: SLF001
        configuration = context._jsc.hadoopConfiguration()  # noqa: SLF001
        path.getFileSystem(configuration).delete(path, True)
    except Exception:  # pragma: no cover - cleanup is best effort during teardown
        return
