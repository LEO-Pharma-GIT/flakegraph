# SPDX-License-Identifier: Apache-2.0
"""End-to-end orchestration for one configured knowledge-graph job.

The pipeline deliberately wires only ports and application services. Providers
can change through configuration, but the order of normalization, chunking,
extraction, filtering, merge, embeddings, community reports, quality gates, and
writer persistence remains one provider-independent implementation.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from kg_processor.application.cache_keys import build_extraction_cache_key, build_ocr_cache_key
from kg_processor.application.chunking import chunk_document, compute_ordered_chunk_hash
from kg_processor.application.community_detection import detect_communities
from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.application.content_kinds import (
    annotate_documents,
    content_kind,
    dataset_columns,
    relative_location,
    source_roots,
    window_filter,
)
from kg_processor.application.discarded_windows import (
    discarded_windows_from_trace,
    is_discarded_window_event,
)
from kg_processor.application.document_normalization import (
    as_image_file,
    build_document_artifacts,
    file_display_name,
    image_document,
    with_location_page,
)
from kg_processor.application.enrichment_cache import CachedEnrichmentLlmProvider
from kg_processor.application.extraction_windows import (
    build_document_context_windows,
    build_extraction_windows,
)
from kg_processor.application.failed_documents import (
    failed_document_event,
    failed_documents_from_trace,
)
from kg_processor.application.folder_entities import (
    excluded_file_ids,
    folder_entity_mentions,
    folder_links,
    without_excluded_folder_mentions,
)
from kg_processor.application.graph_descriptions import synthesize_entity_descriptions
from kg_processor.application.graph_embeddings import (
    embed_chunks,
    embed_communities,
    embed_edges,
    embed_nodes,
)
from kg_processor.application.graph_filter import (
    EntityFilterResult,
    RelationFilterResult,
    filter_entities_with_decisions,
    filter_relations_with_decisions,
)
from kg_processor.application.graph_merge import (
    GraphAssemblyResult,
    assemble_graph_with_decisions,
    prune_isolated_entities,
    structural_relation_types,
)
from kg_processor.application.graph_quality import (
    GraphQualityError,
    GraphQualityResult,
    evaluate_graph_quality,
)
from kg_processor.application.llm_extractors import (
    LlmEntityExtractor,
    LlmRelationExtractor,
    LlmRelationVerifier,
)
from kg_processor.application.metered_llm import MeteredLlmProvider
from kg_processor.application.ontology import LoadedOntology, load_ontology
from kg_processor.application.ordered_map import ordered_map
from kg_processor.application.progress import (
    ProgressEvent,
    ProgressSink,
    elapsed_ms,
    error_metadata,
)
from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.application.rejected_records import rejected_records_from_trace
from kg_processor.application.revalidation import (
    ReplayExtractors,
    RevalidationRequest,
    begin_revalidation,
    revalidate_entity_window,
    revalidate_relation_window,
)
from kg_processor.application.run_report import (
    RunProviders,
    RunReportRequest,
    build_run_report_artifacts,
)
from kg_processor.application.two_pass_extraction import (
    extract_entity_observations,
    extract_graph_observations,
    extract_relation_observations,
    resolve_extraction_observations,
)
from kg_processor.application.type_resolution import TypeResolution, type_resolution_rules
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import Settings, require_location_page_for_folder_entity
from kg_processor.domain.documents import InputFile, ParsedDocument
from kg_processor.domain.extraction import (
    ColumnRelation,
    EntityMention,
    ExtractionObservations,
    ExtractionWindow,
)
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    Evidence,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.jobs import JobFileClaim, JobFileResult
from kg_processor.domain.stages import (
    DocumentEntityInventoryShard,
    DocumentRevalidationShard,
    EntityWindowShard,
    ExtractedDocumentShard,
    PreparedDocumentShard,
    RelationWindowShard,
    RevalidatedRelationWindowShard,
    RevalidationWindow,
    RevalidationWindowShard,
    combine_extracted_shards,
)
from kg_processor.ports.cache import PipelineCache
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor, RelationVerifier
from kg_processor.ports.file_source import FileSource
from kg_processor.ports.graph_writer import GraphWriter
from kg_processor.ports.llm import LlmProvider, StructuredCompletionProvider
from kg_processor.ports.ocr import NoTextReadError, OcrOptions, OcrProvider

logger = logging.getLogger(__name__)


def _bind_extractor[T](extractor: T, llm: StructuredCompletionProvider) -> T:
    """Rebind application LLM extractors without rebuilding reusable local models."""

    if isinstance(extractor, (LlmEntityExtractor, LlmRelationExtractor, LlmRelationVerifier)):
        scoped = copy(extractor)
        scoped.llm = llm
        return scoped
    return extractor


class KgProcessorPipeline:
    """Coordinate one end-to-end graph build through provider-neutral interfaces.

    The pipeline owns stage ordering, caching, progress, quality gates, and artifact
    assembly. Concrete file, OCR, LLM, embedding, writer, and extraction adapters
    are injected so local, on-prem, and Snowflake runs share the same behavior.
    """

    def __init__(
        self,
        settings: Settings,
        file_source: FileSource,
        ocr: OcrProvider,
        llm: LlmProvider,
        embeddings: EmbeddingProvider,
        writer: GraphWriter,
        cache: PipelineCache | None = None,
        claimed_files: list[JobFileClaim] | None = None,
        consumption: ConsumptionCollector | None = None,
        write_scope_override: Literal["graph_snapshot", "file_batch"] | None = None,
        progress_sink: ProgressSink | None = None,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        relation_verifier: RelationVerifier | None = None,
        replay_extractors: ReplayExtractors | None = None,
    ) -> None:
        """Store configured adapters and optional queue, cache, and progress context.

        Extraction-stage overrides support modular algorithms such as GLiNER without
        introducing provider conditionals into orchestration. Ontology loading is
        deferred and cached per pipeline instance. ``replay_extractors`` builds
        the configured extractors over another provider, which revalidation
        answers with stored records.
        """

        self.settings = settings
        self.file_source = file_source
        self.ocr = ocr
        self.llm = llm
        self.embeddings = embeddings
        self.writer = writer
        self.cache = cache
        self.claimed_files = claimed_files or []
        self.consumption = consumption or ConsumptionCollector(
            settings.job.graph_id, job_id=settings.job.job_id
        )
        self.write_scope_override = write_scope_override
        self.progress_sink = progress_sink
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.relation_verifier = relation_verifier
        self.replay_extractors = replay_extractors
        self._loaded_ontology: LoadedOntology | None = None
        self._owns_adapters = True
        # File-queue workers should isolate bad source files from the rest of a
        # claimed batch. Snapshot/local runs intentionally remain fail-fast so
        # operators notice a broken corpus instead of writing a partial graph.
        self._failed_job_file_results: list[JobFileResult] = []

    def for_settings(
        self, settings: Settings, *, consumption: ConsumptionCollector | None = None
    ) -> KgProcessorPipeline:
        """Bind reusable adapters to one run's vocabulary and accounting.

        Provider clients may be shared across runs, but their metering wrappers
        and extractors refer to the run's collector. Rebinding both prevents a
        prior run's OCR pages or LLM calls from entering another graph's report.
        """

        collector = consumption or ConsumptionCollector(
            settings.job.graph_id, job_id=settings.job.job_id, run_id=settings.job.job_id
        )
        delegate = self.llm.delegate if isinstance(self.llm, MeteredLlmProvider) else self.llm
        llm = MeteredLlmProvider(
            delegate, collector, provider=settings.llm.provider, model=settings.llm.model
        )
        variant = KgProcessorPipeline(
            settings=settings,
            file_source=self.file_source,
            ocr=self.ocr,
            llm=llm,
            embeddings=self.embeddings,
            writer=self.writer,
            cache=self.cache,
            claimed_files=self.claimed_files,
            consumption=collector,
            write_scope_override=self.write_scope_override,
            progress_sink=self.progress_sink,
            entity_extractor=_bind_extractor(self.entity_extractor, llm),
            relation_extractor=_bind_extractor(self.relation_extractor, llm),
            relation_verifier=_bind_extractor(self.relation_verifier, llm),
            replay_extractors=self.replay_extractors,
        )
        variant._owns_adapters = False
        return variant

    def for_task(self, task_id: str, attempt: int) -> KgProcessorPipeline:
        """Keep one task attempt's provider calls attributable under parallel slots."""

        return self.for_settings(
            self.settings, consumption=self.consumption.for_task(task_id, attempt)
        )

    def close(self) -> None:
        """Release adapters and cache sessions owned by this pipeline instance."""

        if not self._owns_adapters:
            return

        closed: set[int] = set()
        failures: list[Exception] = []
        for dependency in (
            self.cache,
            self.ocr,
            self.llm,
            self.embeddings,
            self.writer,
            self.file_source,
        ):
            if dependency is None or id(dependency) in closed:
                continue
            closed.add(id(dependency))
            close = getattr(dependency, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise ExceptionGroup("Failed to close pipeline dependencies", failures)

    def run(self) -> GraphWriteBatch:
        """Execute every configured stage and return the exact persisted write batch.

        Empty snapshots are rejected, queue-scoped OCR failures are isolated, and
        hard graph quality failures occur before writer persistence. The returned
        batch contains artifacts, provenance, metrics, and trace data for inspection.
        """

        self._failed_job_file_results = []
        # The trace starts before OCR so failed or empty file-claim selections
        # are explainable in the same artifact stream as LLM/filter/merge events.
        files, file_source_trace = self._list_files_with_progress()
        if self.claimed_files and not files:
            claimed = ", ".join(claim.file_id for claim in self.claimed_files)
            raise ValueError(
                "No files from the configured file source matched claimed KG_JOB_FILE rows: "
                f"{claimed}"
            )
        if not self.claimed_files and not files:
            # A full run writes with graph_snapshot scope, and snapshot writers
            # intentionally replace the previous graph. Refusing empty input
            # here prevents a typoed path or over-tight include glob from
            # deleting an otherwise valid graph snapshot.
            raise ValueError(
                "No input files matched the configured file source; refusing to write an "
                "empty graph snapshot."
            )
        trace: list[dict[str, Any]] = [file_source_trace]
        prepared = self._parse_and_chunk_files(files)
        documents = prepared.documents
        chunks = prepared.chunks
        document_rows = prepared.document_rows
        page_rows = prepared.page_rows
        block_rows = prepared.block_rows
        asset_rows = prepared.asset_rows
        ocr_cache_hits = prepared.ocr_cache_hits
        trace.extend(prepared.trace)
        trace.append(
            {
                "stage": "chunking",
                "chunks": len(chunks),
                "ordered_chunk_hash": compute_ordered_chunk_hash(chunks),
            }
        )
        self._emit_progress("chunking", "completed", counts={"chunks": len(chunks)})

        embed_options = self._embed_options()
        self._embed_chunks_with_progress(chunks, embed_options)
        trace.append(
            {
                "stage": "chunk_embeddings",
                "provider": self.settings.embedding.provider,
                "model": self.settings.embedding.model,
                "dimension": self.settings.embedding.dimension,
                "chunks": len(chunks),
            }
        )

        (
            extraction,
            extraction_cache_id,
            extraction_cache_hit,
            context_trace,
        ) = self._extract_graph_with_progress(
            chunks,
            embed_options,
            {str(row.get("file_id")): row for row in document_rows},
        )
        trace.extend(context_trace)
        trace.append(
            {
                "stage": "graph_extraction",
                "cache_id": extraction_cache_id,
                "cache_hit": extraction_cache_hit,
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "entities": len(extraction.entities),
                "relations": len(extraction.relations),
                "metadata": extraction.provider_metadata,
            }
        )
        return self._finalize(
            chunks=chunks,
            extraction=extraction,
            trace=trace,
            embed_options=embed_options,
            files_seen=len(files),
            documents_processed=len(documents),
            document_rows=document_rows,
            page_rows=page_rows,
            block_rows=block_rows,
            asset_rows=asset_rows,
            ocr_cache_hits=ocr_cache_hits,
            extraction_cache_hit=extraction_cache_hit,
            write=True,
        )

    def prepare_documents(
        self,
        files: list[InputFile],
        *,
        record_failures: bool = True,
    ) -> PreparedDocumentShard:
        """Run OCR, normalization, and chunking for a durable document task.

        Distributed coordinators pass an explicit immutable file selection, so this
        method does not enumerate the configured source or infer queue ownership.
        The output contains no provider objects and can be persisted between worker
        pools or replayed during failure recovery.

        ``record_failures=False`` raises a document's parse failure instead of
        recording it, for a caller that retries first: a coordinator's task has
        an attempt budget, and a parser that was briefly unreachable should cost
        an attempt, not the document.
        """

        if not files:
            raise ValueError("document preparation requires at least one input file")
        self._failed_job_file_results = []
        prepared = self._parse_and_chunk_files(files, record_failures=record_failures)
        trace = [*prepared.trace]
        trace.append(
            {
                "stage": "chunking",
                "chunks": len(prepared.chunks),
                "ordered_chunk_hash": compute_ordered_chunk_hash(prepared.chunks),
            }
        )
        self._emit_progress(
            "chunking",
            "completed",
            counts={"chunks": len(prepared.chunks)},
        )
        file_ids = sorted(
            {
                str(row.get("file_id", ""))
                for row in prepared.document_rows
                if str(row.get("file_id", "")).strip()
            }
        )
        return PreparedDocumentShard(
            file_ids=file_ids,
            files_seen=len(files),
            documents_processed=len(prepared.documents),
            document_rows=prepared.document_rows,
            page_rows=prepared.page_rows,
            block_rows=prepared.block_rows,
            asset_rows=prepared.asset_rows,
            chunks=prepared.chunks,
            ocr_cache_hits=prepared.ocr_cache_hits,
            trace=trace,
        )

    def extract_document_context(
        self,
        prepared: PreparedDocumentShard,
    ) -> PreparedDocumentShard:
        """Enrich one prepared shard with reusable document-context mentions.

        This method is a separate durable stage so OCR workers never need an LLM
        and every body window can share one context result. An ontology without a
        context-enabled type takes the same path but performs no provider call.
        """

        subjects, trace = self._extract_document_context_entities(
            prepared.chunks,
            {str(row.get("file_id")): row for row in prepared.document_rows},
        )
        return prepared.model_copy(
            deep=True,
            update={
                "document_context_entities": subjects,
                "trace": [*prepared.trace, *trace],
            },
        )

    def extract_window_entities(
        self,
        prepared: PreparedDocumentShard,
    ) -> EntityWindowShard:
        """Extract only entity mentions from one queue-sized text window.

        Splitting entity and relation provider calls lets distributed execution
        compact a complete document vocabulary between two parallel queue waves.
        The returned shard excludes prepared pages and blocks to keep object-store
        traffic bounded for long documents. ``prepared`` holds the chunks of
        exactly one window, which the document-context task cut.
        """

        ontology = self._ontology()
        observations = extract_entity_observations(
            prepared.chunks,
            self._graph_llm(),
            self.settings.graph,
            ontology.extraction,
            self.settings.llm.model,
            self.settings.llm.timeout_seconds,
            entity_extractor=self.entity_extractor,
            document_context_entities=prepared.document_context_entities,
        )
        _note_discarded_windows(observations.trace, prepared.file_ids)
        _require_one_window(observations.window_count, "entity")
        return EntityWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
            entities=observations.entities,
            trace=observations.trace,
        )

    def extract_window_relations(
        self,
        prepared: PreparedDocumentShard,
        document_entities: list[EntityMention],
        document_openings: dict[str, str] | None = None,
        *,
        dataset_columns: list[ColumnRelation] | None = None,
    ) -> RelationWindowShard:
        """Extract relations from one window using a document-wide entity list.

        Every returned relation remains grounded in this window's source chunks,
        while either endpoint may have been introduced in another window. The
        immutable inventory is read once by the worker and is not copied into the
        output artifact. The window holds only some of its document's chunks, so
        the document's opening comes with the inventory rather than from them. A
        catalogue dataset's rows are read by the column relations its profile kept.
        """

        ontology = self._ontology()
        observations = extract_relation_observations(
            prepared.chunks,
            document_entities,
            self._graph_llm(),
            self.settings.graph,
            ontology.extraction,
            self.settings.llm.model,
            self.settings.llm.timeout_seconds,
            relation_extractor=self.relation_extractor,
            relation_verifier=self.relation_verifier,
            dataset_columns={file_id: dataset_columns or [] for file_id in prepared.file_ids},
            document_openings=document_openings or {},
        )
        _require_one_window(observations.window_count, "relation")
        return RelationWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
            relations=observations.relations,
            held_relations=observations.held_relations,
            entities=observations.entities,
            trace=observations.trace,
        )

    def begin_revalidation(
        self,
        prepared: PreparedDocumentShard,
        observations: ExtractionObservations,
    ) -> tuple[DocumentRevalidationShard, list[RevalidationWindow]]:
        """Validate one document's stored context records and place the rest by window.

        The windows are replayed apart, a few at a time, by the two window
        methods below.
        """

        return begin_revalidation(
            prepared.file_ids, prepared.chunks, observations, self._revalidation()
        )

    def revalidate_entity_window(self, window: RevalidationWindowShard) -> EntityWindowShard:
        """Replay one window's stored entity records through the current extractor."""

        return revalidate_entity_window(window, self._revalidation())

    def revalidate_relation_window(
        self,
        window: RevalidationWindowShard,
        inventory: DocumentEntityInventoryShard,
    ) -> RevalidatedRelationWindowShard:
        """Replay and verify one window's stored relations against the document entities."""

        return revalidate_relation_window(window, inventory, self._revalidation())

    def _revalidation(self) -> RevalidationRequest:
        """What a revalidation runs with: the configured extractors, verifier and ontology.

        The stored records stand in for the model's extraction answers,
        replayed through extractors built as the configured ones are; the
        grounding fallbacks and the verifier still reach the model.
        """

        if self.replay_extractors is None:
            raise ValueError("revalidation needs the configured extractors to replay records")
        llm = self._graph_llm()
        return RevalidationRequest(
            llm=llm,
            extractors=self.replay_extractors,
            verifier=self.relation_verifier
            or LlmRelationVerifier(llm, self.settings.graph.deterministic_seed),
            settings=self.settings.graph,
            ontology=self._ontology().extraction,
            model=self.settings.llm.model,
            timeout_seconds=self.settings.llm.timeout_seconds,
        )

    def finalize_document_shards(
        self,
        shards: list[ExtractedDocumentShard],
        *,
        write: bool = True,
    ) -> GraphWriteBatch:
        """Resolve, assemble, validate, and optionally publish a complete graph run.

        This graph-wide barrier deterministically combines every successful document
        shard before identity resolution. Publication occurs only after hard quality
        gates pass, so retries cannot expose a partially finalized graph snapshot.
        """

        if not shards:
            raise ValueError("graph finalization requires at least one extracted shard")
        if self.claimed_files:
            raise ValueError("distributed graph finalization cannot use file-queue claims")
        ordered = sorted(
            shards,
            key=lambda shard: tuple(
                sorted(str(row.get("source_uri", "")) for row in shard.prepared.document_rows)
            ),
        )
        prepared = [shard.prepared for shard in ordered]
        chunks = [chunk for shard in prepared for chunk in shard.chunks]
        ontology = self._ontology()
        # A document tied to no folder loses its folder mention before
        # resolution, as the Spark finalizer leaves it out of its mentions.
        observations = without_excluded_folder_mentions(
            combine_extracted_shards(ordered),
            chunks,
            self._folder_exclusions([row for shard in prepared for row in shard.document_rows]),
        )
        embed_options = self._embed_options()
        self._embed_chunks_with_progress(chunks, embed_options)
        extraction = resolve_extraction_observations(
            observations,
            self._graph_llm(),
            self.embeddings,
            embed_options,
            self.settings.graph,
            ontology.profile,
            ontology.checksum,
            ontology.source,
            self.settings.llm.model,
            self.settings.llm.timeout_seconds,
            resolution_progress=lambda completed, total, candidates, failures: self._emit_progress(
                "graph_extraction",
                "progress",
                counts={
                    "resolution_batches_completed": completed,
                    "resolution_batches_total": total,
                    "resolution_candidates": candidates,
                    "resolution_failed_batches": failures,
                },
                metadata={"phase": "entity_resolution"},
            ),
        )
        trace = [event for shard in ordered for event in shard.trace]
        trace.append(
            {
                "stage": "graph_extraction",
                "strategy": "two_pass_resolved",
                "entities": len(extraction.entities),
                "relations": len(extraction.relations),
                "document_shards": len(ordered),
            }
        )
        return self._finalize(
            chunks=chunks,
            extraction=extraction,
            trace=trace,
            embed_options=embed_options,
            files_seen=sum(shard.files_seen for shard in prepared),
            documents_processed=sum(shard.documents_processed for shard in prepared),
            document_rows=[row for shard in prepared for row in shard.document_rows],
            page_rows=[row for shard in prepared for row in shard.page_rows],
            block_rows=[row for shard in prepared for row in shard.block_rows],
            asset_rows=[row for shard in prepared for row in shard.asset_rows],
            ocr_cache_hits=sum(shard.ocr_cache_hits for shard in prepared),
            extraction_cache_hit=False,
            write=write,
        )

    def _finalize(
        self,
        *,
        chunks: list[Chunk],
        extraction: ExtractionResult,
        trace: list[dict[str, Any]],
        embed_options: EmbedOptions,
        files_seen: int,
        documents_processed: int,
        document_rows: list[dict[str, Any]],
        page_rows: list[dict[str, Any]],
        block_rows: list[dict[str, Any]],
        asset_rows: list[dict[str, Any]],
        ocr_cache_hits: int,
        extraction_cache_hit: bool,
        write: bool,
    ) -> GraphWriteBatch:
        """Turn a resolved extraction into a published graph, the same way for every entry.

        A local run and a distributed finalizer arrive here with the extraction
        produced differently, and from this point must not differ at all: the
        filters, assembly, embeddings, communities and quality gates are what
        make the two paths produce the same graph for the same corpus.
        """

        filtered = self._link_folders(
            self._filter_extraction(extraction, chunks, trace), chunks, document_rows, trace
        )
        assembly, descriptions_merged = self._assemble_and_describe_graph(
            chunks,
            filtered.entities,
            filtered.relations,
            trace,
            held_relations=filtered.held_relations,
        )
        self._embed_graph_with_progress(assembly.nodes, assembly.edges, embed_options)
        trace.append(
            {
                "stage": "graph_embeddings",
                "nodes": len(assembly.nodes),
                "edges": len(assembly.edges),
            }
        )
        communities, findings = self._detect_and_embed_communities(
            assembly.nodes,
            assembly.edges,
            assembly.evidence,
            embed_options,
            trace,
        )
        batch, quality_result = self._build_write_batch(
            files_seen=files_seen,
            documents_processed=documents_processed,
            document_rows=document_rows,
            page_rows=page_rows,
            block_rows=block_rows,
            asset_rows=asset_rows,
            chunks=chunks,
            extraction=extraction,
            entities=filtered.entities,
            relations=filtered.relations,
            entity_filter=filtered.entity_filter,
            relation_filter=filtered.relation_filter,
            assembly=assembly,
            descriptions_merged=descriptions_merged,
            communities=communities,
            findings=findings,
            trace=trace,
            ocr_cache_hits=ocr_cache_hits,
            extraction_cache_hit=extraction_cache_hit,
        )
        if self.settings.graph.fail_on_quality_error and not quality_result.ok:
            raise GraphQualityError(quality_result)
        if batch.write_scope == "graph_snapshot" and not batch.nodes and not batch.edges:
            raise ValueError(
                "Extraction produced an empty graph; refusing to publish an empty graph snapshot."
            )
        if write:
            self._write_with_progress(batch)
        return batch

    def _embed_options(self) -> EmbedOptions:
        """Build the shared provider-neutral embedding request for every stage."""

        return EmbedOptions(
            model=self.settings.embedding.model,
            dimension=self.settings.embedding.dimension,
            batch_size=self.settings.embedding.batch_size,
        )

    def _graph_llm(self) -> LlmProvider:
        """Return the strict graph LLM required by distributed extraction stages."""

        if not isinstance(self.llm, StructuredCompletionProvider):
            raise ValueError("distributed extraction requires strict structured completion")
        return self.llm

    def _list_files_with_progress(self) -> tuple[list[InputFile], dict[str, Any]]:
        file_source_started = perf_counter()
        self._emit_progress("file_source", "started")
        available_files = self.file_source.list_files()
        files = self._select_claimed_files(available_files)
        trace_event = {
            "stage": "file_source",
            "files_available": len(available_files),
            "files_seen": len(files),
            "claimed_files": len(self.claimed_files),
        }
        self._emit_progress(
            "file_source",
            "completed",
            counts={
                "files_available": len(available_files),
                "files_seen": len(files),
                "claimed_files": len(self.claimed_files),
            },
            elapsed_ms=elapsed_ms(file_source_started, perf_counter()),
        )
        return files, trace_event

    def _embed_chunks_with_progress(
        self,
        chunks: list[Chunk],
        embed_options: EmbedOptions,
    ) -> None:
        # Chunk embeddings happen before extraction so chunk vectors are
        # available even when later graph quality checks are inspected.
        started = perf_counter()
        self._emit_progress("chunk_embeddings", "started", counts={"chunks": len(chunks)})
        embed_chunks(chunks, self.embeddings, embed_options)
        self._emit_progress(
            "chunk_embeddings",
            "completed",
            counts={"chunks": len(chunks)},
            metadata={
                "provider": self.settings.embedding.provider,
                "model": self.settings.embedding.model,
                "dimension": self.settings.embedding.dimension,
            },
            elapsed_ms=elapsed_ms(started, perf_counter()),
        )

    def _extract_graph_with_progress(
        self,
        chunks: list[Chunk],
        embed_options: EmbedOptions,
        documents: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[ExtractionResult, str, bool, list[dict[str, Any]]]:
        """Run cached extraction with shared document context and report progress."""

        started = perf_counter()
        batch_count = len(
            build_extraction_windows(
                chunks,
                self.settings.graph.extraction_window_tokens,
                self.settings.graph.max_chunks_per_llm_call,
            )
        )
        self._emit_progress(
            "graph_extraction",
            "started",
            counts={
                "chunks": len(chunks),
                "batches_completed": 0,
                "batches_total": batch_count,
            },
            metadata={
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "parallelism": self.settings.graph.extraction_parallelism,
                "strategy": "two_pass",
            },
        )
        extraction, cache_id, cache_hit, context_trace = self._extract_graph(
            chunks, embed_options, documents
        )
        self._emit_progress(
            "graph_extraction",
            "completed",
            counts={
                "entities": len(extraction.entities),
                "relations": len(extraction.relations),
            },
            metadata={
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "cache_hit": cache_hit,
                "cache_id": cache_id,
                "strategy": extraction.provider_metadata.get("strategy", "two_pass"),
            },
            elapsed_ms=elapsed_ms(started, perf_counter()),
        )
        return extraction, cache_id, cache_hit, context_trace

    def _embed_graph_with_progress(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        embed_options: EmbedOptions,
    ) -> None:
        started = perf_counter()
        self._emit_progress(
            "graph_embeddings",
            "started",
            counts={"nodes": len(nodes), "edges": len(edges)},
        )
        embed_nodes(nodes, self.embeddings, embed_options)
        embed_edges(edges, self.embeddings, embed_options)
        self._emit_progress(
            "graph_embeddings",
            "completed",
            counts={"nodes": len(nodes), "edges": len(edges)},
            elapsed_ms=elapsed_ms(started, perf_counter()),
        )

    def _write_with_progress(self, batch: GraphWriteBatch) -> None:
        started = perf_counter()
        counts = _write_counts(batch)
        metadata = {"provider": self.settings.writer.provider, "scope": batch.write_scope}
        self._emit_progress("write", "started", counts=counts, metadata=metadata)
        try:
            self.writer.write(batch)
        except Exception as exc:
            self._emit_progress(
                "write",
                "failed",
                counts=counts,
                metadata={**metadata, **error_metadata(exc)},
                elapsed_ms=elapsed_ms(started, perf_counter()),
            )
            raise
        self._emit_progress(
            "write",
            "completed",
            counts=counts,
            metadata=metadata,
            elapsed_ms=elapsed_ms(started, perf_counter()),
        )

    def job_file_results(self, batch: GraphWriteBatch) -> list[JobFileResult]:
        """Build KG_JOB_FILE completion payloads for a file-queue run."""

        return build_job_file_results(
            batch,
            self.settings.ocr.provider,
            self.settings.llm.provider,
            self.settings.embedding.model,
            self.settings.embedding.dimension,
        )

    def failed_job_file_results(self) -> list[JobFileResult]:
        """Return per-file failures captured during a claimed file-queue run."""

        return list(self._failed_job_file_results)

    def _select_claimed_files(self, files: list[InputFile]) -> list[InputFile]:
        if not self.claimed_files:
            return files
        # File-queue workers may see a large stage or blob prefix but should
        # only process the rows they leased. The claim identity is copied onto
        # the InputFile so writer scopes and KG_JOB_FILE completion use the
        # Snowflake queue row as the source of truth.
        selected: list[InputFile] = []
        consumed_claim_ids: set[str] = set()
        for file in files:
            claim = _matching_claim(
                file,
                self.claimed_files,
                source_root=self.settings.files.input_path,
            )
            if claim is None:
                continue
            if claim.file_id in consumed_claim_ids:
                raise ValueError(f"Multiple discovered files matched claimed file {claim.file_id}")
            consumed_claim_ids.add(claim.file_id)
            selected.append(_file_with_claim_identity(file, claim))
        return selected

    def _parse_and_chunk_files(
        self,
        files: list[InputFile],
        *,
        record_failures: bool = True,
    ) -> _PreparedFiles:
        documents: list[ParsedDocument] = []
        chunks: list[Chunk] = []
        document_rows: list[dict[str, Any]] = []
        page_rows: list[dict[str, Any]] = []
        block_rows: list[dict[str, Any]] = []
        asset_rows: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        ocr_cache_hits = 0

        for file in files:
            ocr_options = self._ocr_options()
            ocr_started = perf_counter()
            self._emit_progress(
                "ocr",
                "started",
                file_id=file.id,
                metadata={
                    "provider": self.settings.ocr.provider,
                    "source_uri": file.source_uri,
                    "mime_type": file.mime_type,
                },
            )
            try:
                parsed, ocr_cache_id, cache_hit = self._parse_file_or_image(file, ocr_options)
            except Exception as exc:
                failure_metadata = error_metadata(exc)
                self._emit_progress(
                    "ocr",
                    "failed",
                    file_id=file.id,
                    metadata={
                        "provider": self.settings.ocr.provider,
                        "source_uri": file.source_uri,
                        **failure_metadata,
                    },
                    elapsed_ms=elapsed_ms(ocr_started, perf_counter()),
                )
                # A provider that refuses every document (bad credentials, no
                # permission) is not this document's failure and fails the run.
                if is_systemic_provider_error(exc) or not record_failures:
                    raise
                # Anything else is this document: record it, with why, and go
                # on, so one unreadable file does not cost the rest of the run.
                # Every finalizer turns the event into a failed_documents row
                # the console reports.
                if self.claimed_files:
                    # The Snowflake queue also keeps a per-file result row.
                    self._failed_job_file_results.append(
                        _job_file_failure_result(
                            file,
                            self.settings.job.graph_id,
                            self.settings.ocr.provider,
                            self.settings.llm.provider,
                            self.settings.embedding.model,
                            self.settings.embedding.dimension,
                            failure_metadata,
                        )
                    )
                trace.append(
                    failed_document_event(file, self.settings.ocr.provider, failure_metadata)
                )
                logger.warning(
                    "could not read %s (%s: %s); recorded as a failed document",
                    file.source_uri,
                    failure_metadata.get("error_type"),
                    failure_metadata.get("error_message"),
                )
                continue
            if cache_hit:
                ocr_cache_hits += 1
            documents.append(parsed)
            location = file.location or relative_location(
                file.source_uri, source_roots(self.settings)
            )
            if self.settings.files.location_page:
                parsed = with_location_page(parsed, location)
            artifacts = build_document_artifacts(
                self.settings.job.graph_id,
                file,
                parsed,
                ocr_cache_id,
                cache_hit,
                location,
            )
            trace.append(artifacts.trace_event)
            document_rows.append(artifacts.document)
            page_rows.extend(artifacts.pages)
            block_rows.extend(artifacts.blocks)
            asset_rows.extend(artifacts.assets)
            chunks.extend(
                chunk_document(
                    parsed,
                    self.settings.job.graph_id,
                    self.settings.graph.chunk_token_size,
                    self.settings.graph.chunk_token_overlap,
                )
            )
            self._emit_progress(
                "ocr",
                "completed",
                file_id=file.id,
                counts={
                    "pages": len(parsed.pages),
                    "blocks": sum(len(page.blocks) for page in parsed.pages),
                    "assets": len(parsed.assets),
                },
                metadata={
                    "provider": self.settings.ocr.provider,
                    "cache_hit": cache_hit,
                    "cache_id": ocr_cache_id,
                },
                elapsed_ms=elapsed_ms(ocr_started, perf_counter()),
            )

        return _PreparedFiles(
            documents=documents,
            chunks=chunks,
            document_rows=document_rows,
            page_rows=page_rows,
            block_rows=block_rows,
            asset_rows=asset_rows,
            ocr_cache_hits=ocr_cache_hits,
            trace=trace,
        )

    def _filter_extraction(
        self,
        extraction: ExtractionResult,
        chunks: list[Chunk],
        trace: list[dict[str, Any]],
    ) -> _FilteredExtraction:
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        entity_filter = filter_entities_with_decisions(
            extraction.entities,
            chunks_by_id,
            min_confidence=self.settings.graph.min_entity_confidence,
            min_name_length=self.settings.graph.min_entity_name_length,
            blocklist=self.settings.graph.entity_blocklist,
        )
        relation_filter = filter_relations_with_decisions(
            extraction.relations,
            entity_filter.kept,
            chunks_by_id,
            min_confidence=self.settings.graph.min_relation_confidence,
            require_endpoint_grounding=self.settings.graph.require_relation_endpoint_grounding,
        )
        # Held relations meet the same bar before assembly may admit them; the
        # ones it drops stay what they already are, rejected records.
        held_filter = filter_relations_with_decisions(
            extraction.held_relations,
            entity_filter.kept,
            chunks_by_id,
            min_confidence=self.settings.graph.min_relation_confidence,
            require_endpoint_grounding=self.settings.graph.require_relation_endpoint_grounding,
        )
        trace.append(
            {
                "stage": "filtering",
                "entities_before": len(extraction.entities),
                "entities_after": len(entity_filter.kept),
                "relations_before": len(extraction.relations),
                "relations_after": len(relation_filter.kept),
                "dropped_entities_by_reason": entity_filter.dropped_reason_counts(),
                "dropped_relations_by_reason": relation_filter.dropped_reason_counts(),
            }
        )
        self._emit_progress(
            "filtering",
            "completed",
            counts={
                "entities_before": len(extraction.entities),
                "entities_after": len(entity_filter.kept),
                "relations_before": len(extraction.relations),
                "relations_after": len(relation_filter.kept),
            },
        )
        trace.extend(
            decision.to_trace_event()
            for decision in entity_filter.decisions
            if decision.action == "dropped"
        )
        trace.extend(
            decision.to_trace_event()
            for decision in relation_filter.decisions
            if decision.action == "dropped"
        )
        return _FilteredExtraction(
            entities=entity_filter.kept,
            relations=relation_filter.kept,
            entity_filter=entity_filter,
            relation_filter=relation_filter,
            held_relations=held_filter.kept,
        )

    def _folder_exclusions(self, document_rows: list[dict[str, Any]]) -> set[str]:
        """The documents, by file id, the ontology's folder entity leaves out."""

        definition = self._ontology().profile.folder_entity
        return excluded_file_ids(definition, document_rows) if definition else set()

    def _link_folders(
        self,
        filtered: _FilteredExtraction,
        chunks: list[Chunk],
        document_rows: list[dict[str, Any]],
        trace: list[dict[str, Any]],
    ) -> _FilteredExtraction:
        """Link every entity of a document to its folder's entity, as the Spark finalizer does."""

        profile = self._ontology().profile
        if profile.folder_entity is None:
            return filtered
        excluded = self._folder_exclusions(document_rows)
        entities, links = folder_links(
            filtered.entities, {chunk.id: chunk for chunk in chunks}, profile, excluded
        )
        trace.append(
            {
                "stage": "folder_links",
                "relations": len(links),
                "excluded_documents": len(excluded),
            }
        )
        return replace(filtered, entities=entities, relations=[*filtered.relations, *links])

    def stage_graph_publication(
        self,
        batch: GraphWriteBatch,
        publication_id: str,
        generation: int,
    ) -> AbstractContextManager[Callable[[], None]]:
        """Prepare a graph snapshot and expose its short fenced publication step.

        Distributed orchestration validates the active publication lease while
        invoking the commit callback; expensive writer staging happens beforehand.
        """

        return self.writer.stage_publication(
            batch, publication_id=publication_id, generation=generation
        )

    def _assemble_and_describe_graph(
        self,
        chunks: list[Chunk],
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        trace: list[dict[str, Any]],
        *,
        held_relations: list[ExtractedRelation],
    ) -> tuple[GraphAssemblyResult, int]:
        merge_started = perf_counter()
        self._emit_progress(
            "merge",
            "started",
            counts={"entities": len(entities), "relations": len(relations)},
        )
        assembly = assemble_graph_with_decisions(
            self.settings.job.graph_id,
            chunks,
            entities,
            relations,
            self.settings.graph.relation_weight_max,
            held_relations=held_relations,
            ontology=self._ontology().profile,
            type_resolution=(
                TypeResolution(
                    type_resolution_rules(self._ontology().profile),
                    self.settings.graph.relation_type_resolution_min_documents,
                )
                if self.settings.graph.relation_type_resolution
                else None
            ),
        )
        if self.settings.graph.drop_isolated_entities:
            assembly = prune_isolated_entities(
                assembly, structural_relation_types(self._ontology().profile)
            )
        enrichment_llm = self._enrichment_llm()
        description_result = synthesize_entity_descriptions(
            assembly.nodes,
            assembly.entity_sources,
            entities,
            assembly.evidence,
            enrichment_llm,
            self.settings.graph.description_merge_min_observations,
            self.settings.graph.description_merge_max_descriptions,
            self.settings.graph.description_merge_max_evidence,
            self.settings.graph.finalization_provider_concurrency,
            model=self.settings.llm.model,
            timeout_seconds=self.settings.llm.timeout_seconds,
            seed=self.settings.graph.deterministic_seed,
        )
        trace.append(
            {
                "stage": "merge",
                "nodes": len(assembly.nodes),
                "edges": len(assembly.edges),
                "evidence": len(assembly.evidence),
                "entity_sources": len(assembly.entity_sources),
                "descriptions_merged": description_result.merged_count,
                "decision_actions": assembly.decision_action_counts(),
                "decision_reasons": assembly.decision_reason_counts(),
            }
        )
        self._emit_progress(
            "merge",
            "completed",
            counts={
                "nodes": len(assembly.nodes),
                "edges": len(assembly.edges),
                "evidence": len(assembly.evidence),
                "entity_sources": len(assembly.entity_sources),
                "descriptions_merged": description_result.merged_count,
            },
            elapsed_ms=elapsed_ms(merge_started, perf_counter()),
        )
        trace.extend(decision.to_trace_event() for decision in assembly.decisions)
        trace.extend(description_result.trace_events)
        return assembly, description_result.merged_count

    def _detect_and_embed_communities(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        evidence: list[Evidence],
        embed_options: EmbedOptions,
        trace: list[dict[str, Any]],
    ) -> tuple[list[Community], list[CommunityFinding]]:
        """Detect weighted communities, generate reports, and embed final summaries.

        Exact edge evidence is supplied to report prompts, while progress callbacks
        and trace events expose report completion independently from graph extraction.
        """

        community_started = perf_counter()
        self._emit_progress(
            "community_detection",
            "started",
            counts={"nodes": len(nodes), "edges": len(edges)},
        )
        community_result = detect_communities(
            self.settings.job.graph_id,
            nodes,
            edges,
            self._enrichment_llm(),
            self.settings.graph.min_community_size,
            self.settings.graph.finalization_provider_concurrency,
            report_progress=lambda completed, total: self._emit_progress(
                "community_detection",
                "progress",
                counts={"reports_completed": completed, "reports_total": total},
            ),
            evidence=evidence,
            deterministic_seed=self.settings.graph.deterministic_seed,
            max_community_size=self.settings.graph.max_community_size,
            resolution=self.settings.graph.community_resolution,
            co_mention_weight=self.settings.graph.community_co_mention_weight,
            model=self.settings.llm.model,
            timeout_seconds=self.settings.llm.timeout_seconds,
            algorithm=self.settings.graph.community_algorithm,
            structural_relation_types=structural_relation_types(self._ontology().profile),
        )
        communities = community_result.communities
        findings = community_result.findings
        embed_communities(communities, self.embeddings, embed_options)
        self._emit_progress(
            "community_detection",
            "completed",
            counts={"communities": len(communities), "findings": len(findings)},
            elapsed_ms=elapsed_ms(community_started, perf_counter()),
        )
        trace.append(
            {
                "stage": "community_detection",
                "communities": len(communities),
                "findings": len(findings),
            }
        )
        trace.extend(community_result.trace_events)
        return communities, findings

    def _enrichment_llm(self) -> LlmProvider:
        """Wrap post-extraction LLM calls in the configured durable cache."""

        if self.cache is None:
            return self.llm
        return CachedEnrichmentLlmProvider(
            self.llm,
            self.cache,
            graph_id=self.settings.job.graph_id,
            provider=self.settings.llm.provider,
            model=self.settings.llm.model,
        )

    def _build_write_batch(
        self,
        *,
        files_seen: int,
        documents_processed: int,
        document_rows: list[dict[str, Any]],
        page_rows: list[dict[str, Any]],
        block_rows: list[dict[str, Any]],
        asset_rows: list[dict[str, Any]],
        chunks: list[Chunk],
        extraction: ExtractionResult,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        entity_filter: EntityFilterResult,
        relation_filter: RelationFilterResult,
        assembly: GraphAssemblyResult,
        descriptions_merged: int,
        communities: list[Community],
        findings: list[CommunityFinding],
        trace: list[dict[str, Any]],
        ocr_cache_hits: int,
        extraction_cache_hit: bool,
    ) -> tuple[GraphWriteBatch, GraphQualityResult]:
        """Evaluate final quality and assemble all persistence and report artifacts.

        Provenance and quality diagnostics are added before constructing the writer
        batch so local and Snowflake adapters persist an identical audit surface.
        """

        # Quality is evaluated before writer persistence so fail-before-write and
        # diagnostic mode both use the same checks and serialized details.
        trace.append(self._run_provenance_event(document_rows, extraction))
        quality_result = evaluate_graph_quality(
            assembly.nodes,
            assembly.edges,
            assembly.evidence,
            self.settings.embedding.dimension,
            chunks=chunks,
            communities=communities,
            ontology=self._ontology().profile,
        )
        trace.append(
            {
                "stage": "quality",
                "ok": quality_result.ok,
                "failed_checks": [
                    {
                        "name": check.name,
                        "severity": check.severity,
                        "count": check.count,
                        "details": check.details,
                    }
                    for check in quality_result.checks
                    if not check.ok
                ],
            }
        )
        self._emit_progress(
            "quality",
            "completed",
            counts={
                "failed_errors": len(quality_result.failed_errors()),
                "warnings": len(quality_result.warnings()),
            },
            metadata={"ok": quality_result.ok},
        )
        file_ids = sorted(
            {
                str(row.get("file_id", ""))
                for row in document_rows
                if str(row.get("file_id", "")).strip()
            }
        )
        # A local extraction keeps its window events with its result, not in the
        # run trace; a fleet run's shards carry them in the trace. Read both.
        window_events = [
            *trace,
            *(
                event
                for event in extraction.provider_metadata.get("trace", [])
                if isinstance(event, dict)
            ),
        ]
        discarded_windows = discarded_windows_from_trace(
            window_events, chunks, self.settings.job.graph_id
        )
        rejected_records = rejected_records_from_trace(
            window_events,
            chunks,
            self.settings.job.graph_id,
            taken_back=assembly.type_resolved_keys,
        )
        failed_documents = failed_documents_from_trace(trace, self.settings.job.graph_id)
        # A file batch replaces what the graph holds for each of its files, so a
        # file that failed this time is in scope too: a failure row from an
        # earlier attempt goes, and so do rows from when it last parsed.
        reindex_file_ids = sorted({*file_ids, *(row.file_id for row in failed_documents)})
        write_scope: Literal["graph_snapshot", "file_batch"] = self.write_scope_override or (
            "file_batch" if self.claimed_files else "graph_snapshot"
        )
        # Full local runs replace a graph snapshot; leased Snowflake queue
        # workers only reindex the claimed files so parallel batches do not
        # erase each other's output.
        report_artifacts = build_run_report_artifacts(
            RunReportRequest(
                job_id=self.settings.job.job_id,
                graph_id=self.settings.job.graph_id,
                consumption=self.consumption.report(self.settings.consumption.rate_card()),
                write_scope=write_scope,
                file_ids=file_ids,
                files_seen=files_seen,
                documents_processed=documents_processed,
                block_rows=block_rows,
                asset_rows=asset_rows,
                chunks=chunks,
                extraction=extraction,
                entities=entities,
                relations=relations,
                entity_filter=entity_filter,
                relation_filter=relation_filter,
                assembly=assembly,
                descriptions_merged=descriptions_merged,
                communities=communities,
                findings=findings,
                ocr_cache_hits=ocr_cache_hits,
                extraction_cache_hit=extraction_cache_hit,
                providers=RunProviders(
                    ocr=self.settings.ocr.provider,
                    llm=self.settings.llm.provider,
                    embedding=self.settings.embedding.provider,
                    writer=self.settings.writer.provider,
                    cache=self.settings.cache.provider,
                ),
                embedding_dimension=self.settings.embedding.dimension,
                graph_settings=self.settings.graph,
                quality_result=quality_result,
                discarded_windows=discarded_windows,
                rejected_records=rejected_records,
                failed_documents=failed_documents,
            )
        )
        return (
            GraphWriteBatch(
                graph_id=self.settings.job.graph_id,
                write_scope=write_scope,
                reindex_file_ids=reindex_file_ids,
                documents=annotate_documents(document_rows, trace),
                pages=page_rows,
                blocks=block_rows,
                assets=asset_rows,
                chunks=chunks,
                nodes=assembly.nodes,
                edges=assembly.edges,
                edge_observations=assembly.edge_observations,
                evidence=assembly.evidence,
                entity_sources=assembly.entity_sources,
                communities=communities,
                community_findings=findings,
                discarded_windows=discarded_windows,
                rejected_records=rejected_records,
                failed_documents=failed_documents,
                run_report=report_artifacts.run_report,
                extraction_trace=trace,
                graph_metrics=report_artifacts.graph_metrics,
                relation_weight_max=self.settings.graph.relation_weight_max,
            ),
            quality_result,
        )

    def _run_provenance_event(
        self,
        document_rows: list[dict[str, Any]],
        extraction: ExtractionResult,
    ) -> dict[str, Any]:
        """Capture redacted effective inputs needed to reproduce an extraction run.

        The event records input checksums, ontology identity, model choices, seed,
        prompt fingerprints, and the resolved configuration without exposing secrets.
        """

        ontology = self._ontology()
        extraction_trace = extraction.provider_metadata.get("trace", [])
        prompt_fingerprints = [
            {
                "name": str(item.get("prompt_name", "")),
                "revision": str(item.get("prompt_revision", "")),
                "sha256": str(item.get("prompt_sha256", item.get("prompt_checksum", ""))),
            }
            for item in extraction_trace
            if isinstance(item, dict) and item.get("prompt_name")
        ]
        return {
            "stage": "run_provenance",
            "effective_config": redact_sensitive_data(
                self.settings.model_dump(mode="json", by_alias=True)
            ),
            "input_manifest": [
                {
                    "file_id": str(row.get("file_id", "")),
                    "checksum": str(row.get("checksum", "")),
                    "source_uri": str(row.get("source_uri", "")),
                }
                for row in document_rows
            ],
            "ontology": {
                "name": ontology.profile.name,
                "source": ontology.source,
                "checksum": ontology.checksum,
            },
            "models": {
                "llm": self.settings.llm.model,
                "embedding": self.settings.embedding.model,
            },
            "deterministic_seed": self.settings.graph.deterministic_seed,
            "prompt_fingerprints": prompt_fingerprints,
        }

    def _parse_file(
        self,
        file: InputFile,
        ocr_options: OcrOptions,
    ) -> tuple[ParsedDocument, str, bool]:
        provider_settings: dict[str, Any] | None = None
        if self.settings.ocr.provider == "generic_http":
            provider_settings = self.settings.generic_http_ocr.model_dump(mode="json")
        elif self.settings.ocr.provider == "fallback":
            # Routing policy changes which normalized document enters chunking,
            # so warm caches must not preserve text accepted under old thresholds.
            provider_settings = {
                "primary_provider": self.settings.ocr.fallback_primary_provider,
                "secondary_provider": self.settings.ocr.fallback_secondary_provider,
                "min_characters_per_page": (self.settings.ocr.fallback_min_characters_per_page),
                "max_sparse_page_ratio": self.settings.ocr.fallback_max_sparse_page_ratio,
                "max_unbroken_text_ratio": (self.settings.ocr.fallback_max_unbroken_text_ratio),
                "max_fragmented_text_ratio": (self.settings.ocr.fallback_max_fragmented_text_ratio),
                "max_unspaced_page_ratio": self.settings.ocr.fallback_max_unspaced_page_ratio,
                "max_interleaved_column_page_ratio": (
                    self.settings.ocr.fallback_max_interleaved_column_page_ratio
                ),
            }
            fallback_providers = {
                self.settings.ocr.fallback_primary_provider,
                self.settings.ocr.fallback_secondary_provider,
            }
            if "generic_http" in fallback_providers:
                provider_settings["generic_http"] = self.settings.generic_http_ocr.model_dump(
                    mode="json"
                )
            if "tesseract_internal" in fallback_providers:
                provider_settings["tesseract_internal"] = {
                    "command": self.settings.ocr.tesseract_command,
                    "pdf_renderer_command": self.settings.ocr.tesseract_pdf_renderer_command,
                    "dpi": self.settings.ocr.tesseract_dpi,
                }
        cache_key = build_ocr_cache_key(
            file,
            self.settings.ocr.provider,
            ocr_options,
            provider_settings,
        )
        cached = self.cache.get_ocr_document(cache_key) if self.cache is not None else None
        if cached is not None:
            return cached, cache_key.id, True
        parsed = self.ocr.parse(file, ocr_options)
        # Recorded after the call, so a cache hit contributes nothing: it was
        # never billed, and counting it would make re-running an unchanged
        # corpus look as expensive as the first run. The adapter that ran is
        # what gets billed, not the routing policy that chose it.
        provider = str(parsed.provider_metadata.get("provider", self.settings.ocr.provider))
        self.consumption.record(
            stage="ocr",
            operation="parse_document",
            provider=provider,
            model=self.settings.ocr.consumption_model(provider),
            pages=len(parsed.pages),
            file_id=file.id,
        )
        if self.cache is not None:
            self.cache.put_ocr_document(cache_key, parsed)
        return parsed, cache_key.id, False

    def _parse_file_or_image(
        self,
        file: InputFile,
        options: OcrOptions,
    ) -> tuple[ParsedDocument, str, bool]:
        """Parse a file; an image with no text is kept as an image, not a failure.

        A photograph or micrograph has nothing for a parser to read, and that is
        what it is rather than something that went wrong: it stays in the graph
        as an asset of its own document.
        """

        is_image = content_kind(file.mime_type, file.source_uri) == "image"
        try:
            parsed, cache_id, cache_hit = self._parse_file(file, options)
        except NoTextReadError:
            if not is_image:
                raise
            return image_document(file), "", False
        return (as_image_file(file, parsed) if is_image else parsed), cache_id, cache_hit

    def _extract_graph(
        self,
        chunks: list[Chunk],
        embed_options: EmbedOptions,
        documents: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[ExtractionResult, str, bool, list[dict[str, Any]]]:
        """Run grounded extraction and persist the result after a cache miss.

        Strict structured completion keeps provider transports behind one schema
        and grounding contract owned by the application layer.

        The documents the folder entity excludes lose their folder mentions
        before resolution, as a distributed finalizer leaves them out. The
        cached result is resolved, so its key names those documents; with none
        among the chunks it is the extraction's own key.
        """

        ontology = self._ontology()
        excluded = sorted(self._folder_exclusions(list((documents or {}).values())))
        cache_key = build_extraction_cache_key(
            self.settings.job.graph_id,
            chunks,
            self.settings.llm.provider,
            self.settings.llm.model,
            self.settings.graph,
            self.settings.llm.timeout_seconds,
            (
                sha256_hex(json.dumps([ontology.extraction_checksum, excluded]))
                if excluded
                else ontology.extraction_checksum
            ),
            self.settings.extractors,
            self.settings.embedding,
            self.settings.llm,
        )
        cached = self.cache.get_extraction_result(cache_key) if self.cache is not None else None
        if cached is not None:
            return cached, cache_key.id, True, _cached_document_context_trace(cached)
        document_context_entities, context_trace = self._extract_document_context_entities(
            chunks, documents
        )
        graph_llm = self._graph_llm()
        observations = extract_graph_observations(
            chunks,
            graph_llm,
            self.settings.graph,
            ontology.extraction,
            self.settings.llm.model,
            self.settings.llm.timeout_seconds,
            window_progress=lambda completed, total, entities, relations: self._emit_progress(
                "graph_extraction",
                "progress",
                counts={
                    "batches_completed": completed,
                    "batches_total": total,
                    "batch_entities": entities,
                    "batch_relations": relations,
                },
            ),
            entity_extractor=self.entity_extractor,
            relation_extractor=self.relation_extractor,
            relation_verifier=self.relation_verifier,
            document_context_entities=document_context_entities,
            window_filter=window_filter(
                context_trace, catalogue_relations=self.settings.graph.catalogue_relations
            ),
            dataset_columns=dataset_columns(context_trace),
        )
        observations = without_excluded_folder_mentions(
            observations.model_copy(update={"trace": [*context_trace, *observations.trace]}),
            chunks,
            set(excluded),
        )
        extraction = resolve_extraction_observations(
            observations,
            graph_llm,
            self.embeddings,
            embed_options,
            self.settings.graph,
            ontology.extraction,
            ontology.extraction_checksum,
            ontology.source,
            self.settings.llm.model,
            self.settings.llm.timeout_seconds,
            resolution_progress=lambda completed, total, candidates, failures: self._emit_progress(
                "graph_extraction",
                "progress",
                counts={
                    "resolution_batches_completed": completed,
                    "resolution_batches_total": total,
                    "resolution_candidates": candidates,
                    "resolution_failed_batches": failures,
                },
                metadata={
                    "phase": "entity_resolution",
                    "parallelism": self.settings.graph.finalization_provider_concurrency,
                },
            ),
        )
        # The extraction trace begins with the document-context trace, so one
        # check covers context and body provider failures alike.
        if self.cache is not None and not _extraction_degraded(extraction):
            self.cache.put_extraction_result(cache_key, extraction)
        return extraction, cache_key.id, False, context_trace

    def _extract_document_context_entities(
        self,
        chunks: list[Chunk],
        documents: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[EntityMention], list[dict[str, Any]]]:
        """Extract grounded discourse anchors for each configured source document.

        Ontology authors opt types into document-context behavior. Front matter is
        bounded independently of document length, and each resulting immutable
        mention can be reused by every parallel extraction window for that file.
        A dataset's opening is profiled instead: its kind decides how much of it
        the extraction passes read (content_kinds). With a folder entity, every
        document also gets the mention of the entity its folder names
        (folder_entities).
        """

        ontology = self._ontology().extraction
        folders, folder_trace = folder_entity_mentions(
            chunks,
            documents or {},
            ontology,
            self._graph_llm(),
            model=self.settings.llm.model,
            timeout_seconds=self.settings.llm.timeout_seconds,
            seed=self.settings.graph.deterministic_seed,
            parallelism=self.settings.graph.extraction_parallelism,
        )
        datasets = {
            file_id: row
            for file_id, row in (documents or {}).items()
            if row.get("kind") == "dataset"
        }
        has_context_types = any(item.document_context for item in ontology.entity_types)
        skipped = [
            {
                "stage": "document_context_extraction",
                "skipped": True,
                "reason": "ontology_has_no_document_context_type",
            }
        ]
        if not has_context_types and not datasets:
            return folders, [*skipped, *folder_trace]
        windows = [
            window
            for window in build_document_context_windows(
                chunks,
                self.settings.graph.document_context_tokens,
                self.settings.graph.document_context_max_chunks,
            )
            if has_context_types or window.document_id in datasets
        ]
        extractor = LlmEntityExtractor(
            self._graph_llm(),
            self.settings.graph.deterministic_seed,
        )

        def extract_window(
            _index: int, window: ExtractionWindow
        ) -> tuple[list[EntityMention], dict[str, Any]]:
            if window.document_id in datasets:
                return [], extractor.profile_dataset(
                    window,
                    ontology,
                    file_name=file_display_name(datasets[window.document_id]),
                    model=self.settings.llm.model,
                    timeout_seconds=self.settings.llm.timeout_seconds,
                )
            outcome = extractor.extract_document_context_entities(
                window,
                ontology,
                model=self.settings.llm.model,
                timeout_seconds=self.settings.llm.timeout_seconds,
                max_entities=self.settings.graph.max_document_context_entities,
            )
            return outcome.entities, outcome.trace

        failures: list[Exception] = []

        def contain(
            _index: int, window: ExtractionWindow, exc: Exception
        ) -> tuple[list[EntityMention], dict[str, Any]]:
            failures.append(exc)
            return [], {
                "stage": "document_context_error",
                "window_id": window.id,
                "document_id": window.document_id,
                "error": error_metadata(exc),
            }

        # A snapshot can contain many documents, and every context prompt is
        # independent. Running these calls serially delayed the already-parallel
        # body-window stage by one complete provider round trip per document. The
        # bounded pool reuses the same throughput control as ordinary extraction.
        ordered = ordered_map(
            windows,
            extract_window,
            parallelism=self.settings.graph.extraction_parallelism,
            on_complete=lambda completed, result: self._emit_document_context_progress(
                completed, len(windows), result[0]
            ),
            contain=contain,
        )

        subjects = [*(entity for entities, _event in ordered for entity in entities), *folders]
        trace = [
            *([] if has_context_types else skipped),
            *(event for _entities, event in ordered),
            *folder_trace,
        ]
        if failures and len(failures) == len(windows):
            trace.append(
                {
                    "stage": "document_context_summary",
                    "status": "degraded",
                    "windows": len(windows),
                    "failed_windows": len(failures),
                    "entities": 0,
                }
            )
        return subjects, trace

    def _emit_document_context_progress(
        self,
        completed: int,
        total: int,
        entities: list[EntityMention],
    ) -> None:
        """Report document-prefix completions without exposing provider payloads."""

        self._emit_progress(
            "graph_extraction",
            "progress",
            counts={
                "context_documents_completed": completed,
                "context_documents_total": total,
                "context_entities": len(entities),
            },
            metadata={
                "phase": "document_context",
                "parallelism": min(self.settings.graph.extraction_parallelism, total),
            },
        )

    def _ontology(self) -> LoadedOntology:
        """Load and validate the effective ontology once per pipeline instance.

        Caching guarantees extraction, quality checks, provenance, and cache keys all
        use the same parsed profile and checksum throughout a run.
        """

        if self._loaded_ontology is None:
            loaded = load_ontology(
                self.settings.ontology.profile_path,
                self.settings.graph.entity_types,
                self.settings.graph.relation_types,
                inline=self.settings.ontology.profile,
            )
            require_location_page_for_folder_entity(
                loaded.profile.folder_entity is not None, self.settings.files.location_page
            )
            self._loaded_ontology = loaded
        return self._loaded_ontology

    def _ocr_options(self) -> OcrOptions:
        return OcrOptions(
            language=self.settings.ocr.language,
            page_range=self.settings.ocr.page_range,
            timeout_seconds=self.settings.ocr.timeout_seconds,
            model_cache_dir=str(self.settings.ocr.model_cache_dir)
            if self.settings.ocr.model_cache_dir
            else None,
            method=self.settings.ocr.mineru_method,
            backend=self.settings.ocr.mineru_backend,
            effort=self.settings.ocr.mineru_effort,
            api_url=self.settings.ocr.mineru_api_url,
            api_key=self.settings.ocr.mineru_api_key,
            server_url=self.settings.ocr.mineru_server_url,
            start_page_id=self.settings.ocr.mineru_start_page_id,
            end_page_id=self.settings.ocr.mineru_end_page_id,
            formula=self.settings.ocr.mineru_formula,
            table=self.settings.ocr.mineru_table,
            image_analysis=self.settings.ocr.mineru_image_analysis,
            client_side_output_generation=(self.settings.ocr.mineru_client_side_output_generation),
            tesseract_command=self.settings.ocr.tesseract_command,
            tesseract_pdf_renderer_command=self.settings.ocr.tesseract_pdf_renderer_command,
            tesseract_dpi=self.settings.ocr.tesseract_dpi,
            snowflake_parse_mode=self.settings.ocr.snowflake_parse_mode,
            snowflake_extract_images=self.settings.ocr.snowflake_extract_images,
            snowflake_page_split=self.settings.ocr.snowflake_page_split,
        )

    def _emit_progress(
        self,
        stage: str,
        status: str,
        *,
        file_id: str | None = None,
        counts: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        if self.progress_sink is None:
            return
        self.progress_sink.emit(
            ProgressEvent(
                job_id=self.settings.job.job_id,
                graph_id=self.settings.job.graph_id,
                stage=stage,
                status=status,
                file_id=file_id,
                counts=counts or {},
                metadata=metadata or {},
                elapsed_ms=elapsed_ms,
            )
        )


@dataclass(frozen=True)
class _PreparedFiles:
    """Parsed documents and document rows produced before graph extraction."""

    documents: list[ParsedDocument]
    chunks: list[Chunk]
    document_rows: list[dict[str, Any]]
    page_rows: list[dict[str, Any]]
    block_rows: list[dict[str, Any]]
    asset_rows: list[dict[str, Any]]
    ocr_cache_hits: int
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class _FilteredExtraction:
    """Filtered extraction observations plus their filter decision records."""

    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    entity_filter: EntityFilterResult
    relation_filter: RelationFilterResult
    held_relations: list[ExtractedRelation]


def build_job_file_results(
    batch: GraphWriteBatch,
    ocr_provider: str,
    llm_provider: str,
    embedding_model: str,
    embedding_dimension: int,
) -> list[JobFileResult]:
    """Summarize per-file persistence for Snowflake queue completion and auditing.

    Shared canonical nodes and edges are attributed through source chunks/files,
    while each table's row count remains visible without querying all graph tables.
    """

    chunk_ids_by_file: dict[str, set[str]] = {}
    for chunk in batch.chunks:
        chunk_ids_by_file.setdefault(chunk.file_id, set()).add(chunk.id)
    file_ids = sorted(
        {
            str(row.get("file_id", ""))
            for row in batch.documents
            if str(row.get("file_id", "")).strip()
        }
    )
    results: list[JobFileResult] = []
    for file_id in file_ids:
        file_chunk_ids = chunk_ids_by_file.get(file_id, set())
        node_rows = sum(
            1 for node in batch.nodes if file_chunk_ids.intersection(node.source_chunk_ids)
        )
        edge_rows = sum(1 for edge in batch.edges if file_id in edge.source_file_ids)
        # KG_JOB_FILE is the app-facing progress surface, so keep a compact
        # per-file row breakdown there instead of forcing operators to derive
        # it from all graph tables after the fact.
        row_counts = {
            "documents": sum(1 for row in batch.documents if row.get("file_id") == file_id),
            "pages": sum(1 for row in batch.pages if row.get("file_id") == file_id),
            "blocks": sum(1 for row in batch.blocks if row.get("file_id") == file_id),
            "assets": sum(1 for row in batch.assets if row.get("file_id") == file_id),
            "chunks": sum(1 for chunk in batch.chunks if chunk.file_id == file_id),
            "nodes": node_rows,
            "edges": edge_rows,
            "edge_observations": sum(
                1 for item in batch.edge_observations if item.file_id == file_id
            ),
            "evidence": sum(1 for item in batch.evidence if item.file_id == file_id),
            "entity_sources": sum(1 for item in batch.entity_sources if item.file_id == file_id),
        }
        rows_written = sum(row_counts.values())
        quality = batch.graph_metrics.get("quality", {})
        results.append(
            JobFileResult(
                file_id=file_id,
                rows_written=rows_written,
                row_counts=row_counts,
                stage="written",
                ocr_provider=ocr_provider,
                llm_provider=llm_provider,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
                audit={
                    "graph_id": batch.graph_id,
                    "write_scope": batch.write_scope,
                    "quality_ok": quality.get("ok") if isinstance(quality, dict) else None,
                },
            )
        )
    return results


def _job_file_failure_result(
    file: InputFile,
    graph_id: str,
    ocr_provider: str,
    llm_provider: str,
    embedding_model: str,
    embedding_dimension: int,
    failure_metadata: dict[str, object],
) -> JobFileResult:
    """Build a queue-row failure summary without pretending graph rows were written."""

    return JobFileResult(
        file_id=file.id,
        rows_written=0,
        row_counts={},
        stage="ocr_failed",
        ocr_provider=ocr_provider,
        llm_provider=llm_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        audit={
            "graph_id": graph_id,
            "write_scope": "file_batch",
            "source_uri": file.source_uri,
            "error": failure_metadata,
        },
    )


def _write_counts(batch: GraphWriteBatch) -> dict[str, int]:
    """Return a complete table-level artifact count summary for progress and run reporting."""

    return {
        "documents": len(batch.documents),
        "pages": len(batch.pages),
        "blocks": len(batch.blocks),
        "assets": len(batch.assets),
        "chunks": len(batch.chunks),
        "nodes": len(batch.nodes),
        "edges": len(batch.edges),
        "edge_observations": len(batch.edge_observations),
        "evidence": len(batch.evidence),
        "entity_sources": len(batch.entity_sources),
        "communities": len(batch.communities),
        "community_findings": len(batch.community_findings),
        "trace_events": len(batch.extraction_trace),
    }


def _matching_claim(
    file: InputFile,
    claims: list[JobFileClaim],
    *,
    source_root: Path | None = None,
) -> JobFileClaim | None:
    file_keys = _file_match_keys(file, source_root=source_root)
    matches: list[JobFileClaim] = []
    for claim in claims:
        claim_keys = _claim_match_keys(claim)
        if file_keys.intersection(claim_keys) or (
            claim.source_uri and _source_paths_match(file, claim.source_uri)
        ):
            matches.append(claim)
    unique = {claim.file_id: claim for claim in matches}
    if len(unique) > 1:
        raise ValueError(f"Discovered file {file.source_uri} matched multiple queue claims")
    return next(iter(unique.values()), None)


def _file_with_claim_identity(file: InputFile, claim: JobFileClaim) -> InputFile:
    update: dict[str, object] = {"id": claim.file_id}
    if claim.checksum:
        update["checksum"] = claim.checksum
    if claim.source_uri and _is_absolute_source_uri(claim.source_uri):
        update["source_uri"] = claim.source_uri
    return file.model_copy(update=update)


def _file_match_keys(file: InputFile, *, source_root: Path | None = None) -> set[str]:
    relative_path = ""
    if source_root is not None:
        try:
            root = source_root if source_root.is_dir() else source_root.parent
            relative_path = file.path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return {
        key
        for key in {
            file.id,
            file.source_uri,
            file.path.as_posix(),
            relative_path,
            _relative_source_path(file.source_uri),
        }
        if key
    }


def _claim_match_keys(claim: JobFileClaim) -> set[str]:
    return {
        key
        for key in {
            claim.file_id,
            claim.source_uri,
            _relative_source_path(claim.source_uri or ""),
        }
        if key
    }


def _source_paths_match(file: InputFile, source_uri: str) -> bool:
    normalized_source = _relative_source_path(source_uri)
    if not normalized_source:
        return False
    return (
        file.source_uri.rstrip("/") == source_uri.strip().rstrip("/")
        or _relative_source_path(file.source_uri) == normalized_source
    )


def _relative_source_path(source_uri: str) -> str:
    stripped = source_uri.strip()
    if not stripped:
        return ""
    if stripped.startswith("@") and "/" in stripped:
        return stripped.split("/", 1)[1].strip("/")
    if "://" in stripped:
        return stripped.split("://", 1)[1].split("/", 1)[-1].strip("/")
    return stripped.strip("/")


def _is_absolute_source_uri(source_uri: str) -> bool:
    return source_uri.startswith(("@", "file://", "http://", "https://", "azblob://"))


def _require_one_window(window_count: int, phase: str) -> None:
    """Refuse a window task whose chunks rebuilt into other than the one window it names.

    A document's windows are cut once, by its context task, and compaction
    counts one result per window. Chunks that rebuild into more windows, or
    none, would make that count disagree with what was extracted.
    """

    if window_count != 1:
        raise ValueError(
            f"{phase} window task rebuilt {window_count} windows from its chunks, not one"
        )


def _extraction_degraded(extraction: ExtractionResult) -> bool:
    """Keep provider failures retryable by never caching their partial results.

    Successful responses whose records fail validation are complete decisions,
    so their discarded-window events remain eligible for caching.
    """

    return any(
        isinstance(event, dict) and str(event.get("stage", "")).endswith("_error")
        for event in extraction.provider_metadata.get("trace", [])
    )


def _cached_document_context_trace(extraction: ExtractionResult) -> list[dict[str, Any]]:
    """Replay document annotations retained inside a complete cached extraction."""

    return [
        dict(event)
        for event in extraction.provider_metadata.get("trace", [])
        if isinstance(event, dict)
        and (
            str(event.get("stage", "")).startswith("document_context")
            or event.get("stage") == "dataset_profile"
        )
    ]


def _note_discarded_windows(
    trace: Sequence[Mapping[str, Any]],
    file_ids: Sequence[str],
) -> None:
    """Say so when a window's every extracted record was rejected by validation.

    A window that yields no entities is ordinary — a title slide or a page of
    furniture genuinely contains none, and the model correctly returns nothing.
    A window where the model returned records and validation rejected *all* of
    them is not ordinary: the work happened, was paid for, and was thrown away,
    and what that text says is now absent from the graph.

    Validation rejection is a completed extraction outcome: the task succeeds
    and its trace supplies a ``discarded_windows`` row for per-document reporting
    in both finalizers. Provider exceptions remain failed work and are distinct
    from this outcome. Logging keeps the validation gap visible to operators.
    """

    for event in trace:
        if is_discarded_window_event(event):
            reasons = ", ".join(
                f"{name}={count}"
                for name, count in sorted((event.get("record_actions") or {}).items())
            )
            logger.warning(
                "extraction discarded every record for %s (window %s): %s extracted, "
                "none accepted (%s); recorded as a discarded window",
                list(file_ids),
                event.get("window_id"),
                event.get("input_records"),
                reasons or "no reasons recorded",
            )
