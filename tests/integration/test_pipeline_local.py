from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.graph_quality import GraphQualityError
from kg_processor.application.inspect import inspect_local_graph
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.progress import ProgressEvent
from kg_processor.config.settings import Settings
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.jobs import JobFileClaim
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult
from kg_processor.ports.ocr import OcrOptions


def test_pipeline_writes_local_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen. Acme Corp supports Contoso Health.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    batch = pipeline.run()

    assert batch.run_report["files_processed"] == 1
    assert batch.nodes
    assert batch.edges
    assert batch.graph_metrics["quality"]["ok"]
    assert (output_dir / "nodes.parquet").exists()
    assert len(pd.read_parquet(output_dir / "nodes.parquet")) == len(batch.nodes)
    documents = pd.read_parquet(output_dir / "documents.parquet")
    pages = pd.read_parquet(output_dir / "pages.parquet")
    blocks = pd.read_parquet(output_dir / "blocks.parquet")
    assets = pd.read_parquet(output_dir / "assets.parquet")
    chunks = pd.read_parquet(output_dir / "chunks.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    findings = pd.read_parquet(output_dir / "community_findings.parquet")
    assert set(["id", "graph_id", "file_id"]).issubset(documents.columns)
    assert documents.loc[0, "id"] == documents.loc[0, "file_id"]
    assert set(["id", "graph_id", "file_id", "page_number"]).issubset(pages.columns)
    assert set(["id", "graph_id", "file_id", "page_number", "kind", "metadata"]).issubset(
        blocks.columns
    )
    assert set(["id", "graph_id", "file_id", "kind", "metadata"]).issubset(assets.columns)
    assert set(["id", "graph_id", "document_id", "section_path"]).issubset(chunks.columns)
    assert set(["rating_explanation", "suggested_questions"]).issubset(communities.columns)
    assert set(["id", "graph_id", "community_id"]).issubset(findings.columns)
    assert documents.loc[0, "graph_id"] == "graph"
    assert pages.loc[0, "graph_id"] == "graph"
    assert chunks.loc[0, "graph_id"] == "graph"
    assert communities.loc[0, "rating_explanation"]
    assert findings.loc[0, "graph_id"] == "graph"


def test_pipeline_refuses_empty_full_snapshot_runs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    writer = RecordingWriter()
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=writer,
        cache=build_cache(settings),
    )

    with pytest.raises(ValueError, match="No input files matched"):
        pipeline.run()

    assert writer.batch is None


def test_pipeline_processes_packaged_manifest_corpus(tmp_path: Path) -> None:
    output_dir = tmp_path / "manifest-out"
    settings = Settings.load(
        Path("tests/fixtures/configs/manifest.yaml"),
        overrides={
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
            "graph": {
                "chunk_token_size": 100000,
                "chunk_token_overlap": 0,
                "max_chunks_per_llm_call": 2000,
                "max_entities_per_batch": 80,
                "max_relations_per_batch": 80,
                "gleaning_max_passes": 0,
            },
        },
    )

    batch = _build_pipeline(settings).run()
    inspection = inspect_local_graph(output_dir)

    assert batch.run_report["files_processed"] == 10
    assert batch.run_report["files_seen"] == 10
    assert len(batch.documents) == 10
    assert len(batch.pages) >= 26
    assert batch.chunks
    assert batch.nodes
    assert batch.evidence
    assert batch.graph_metrics["providers"] == {
        "ocr": "builtin_text",
        "llm": "fake",
        "embedding": "hash",
        "writer": "local_artifacts",
        "cache": "none",
    }
    assert batch.graph_metrics["quality"]["ok"] is True
    assert inspection["schema"]["ok"] is True
    assert inspection["quality"]["ok"] is True
    assert inspection["tables"]["documents"] == 10
    assert inspection["tables"]["pages"] >= 26
    assert {document["source_uri"] for document in batch.documents} == {
        "file://data/martial_arts/files/martial-arts-schools.pptx",
        "file://data/martial_arts/files/martial-arts-timeline.html",
        "file://data/martial_arts/files/martial-arts-interview.docx",
        "file://data/martial_arts/files/martial-arts-overview.pdf",
        "file://data/martial_arts/files/martial-arts-lineages.pdf",
        "file://data/martial_arts/files/martial-arts-crossroads.md",
        "file://data/martial_arts/files/martial-arts-living-heritage.txt",
        "file://data/martial_arts/files/martial-arts-olympic-program.md",
        "file://data/martial_arts/files/martial-arts-rules-and-regulators.md",
        "file://data/martial_arts/files/smoke.txt",
    }


def test_pipeline_emits_structured_progress_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "progress-job", "graph_id": "progress-graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    progress = RecordingProgressSink()
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        progress_sink=progress,
    )

    batch = pipeline.run()
    records = [event.as_log_record() for event in progress.events]

    assert batch.run_report["files_processed"] == 1
    assert all(record["event"] == "kg_processor.progress" for record in records)
    assert all(record["job_id"] == "progress-job" for record in records)
    assert all(record["graph_id"] == "progress-graph" for record in records)
    assert ("file_source", "started") in _stage_statuses(records)
    assert ("file_source", "completed") in _stage_statuses(records)
    assert ("ocr", "started") in _stage_statuses(records)
    assert ("ocr", "completed") in _stage_statuses(records)
    assert ("graph_extraction", "completed") in _stage_statuses(records)
    assert ("quality", "completed") in _stage_statuses(records)
    assert ("write", "started") in _stage_statuses(records)
    assert ("write", "completed") in _stage_statuses(records)
    ocr_completed = next(
        record for record in records if record["stage"] == "ocr" and record["status"] == "completed"
    )
    write_completed = next(
        record
        for record in records
        if record["stage"] == "write" and record["status"] == "completed"
    )
    ocr_metadata = ocr_completed["metadata"]
    write_counts = write_completed["counts"]
    assert isinstance(ocr_metadata, dict)
    assert isinstance(write_counts, dict)
    assert ocr_metadata["provider"] == "builtin_text"
    assert write_counts["nodes"] == len(batch.nodes)


def test_pipeline_persists_ocr_assets_and_chunk_asset_references(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir, "location_page": False},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=AssetOcrProvider(),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    batch = pipeline.run()

    assert batch.assets == [
        {
            "id": "asset_1",
            "graph_id": "graph",
            "file_id": batch.documents[0]["file_id"],
            "kind": "image",
            "page_number": 1,
            "uri": "file:///figure-1.png",
            "metadata": {"caption": "Figure 1"},
        }
    ]
    assert batch.blocks[0]["id"] == "block_1"
    assert batch.blocks[0]["bbox"] is None
    assert batch.chunks[0].asset_ids == ["asset_1"]
    assert batch.chunks[0].block_ids == ["block_1"]
    assets = pd.read_parquet(output_dir / "assets.parquet")
    blocks = pd.read_parquet(output_dir / "blocks.parquet")
    chunks = pd.read_parquet(output_dir / "chunks.parquet")
    assert blocks.loc[0, "id"] == "block_1"
    assert blocks.loc[0, "metadata"] == "{}"
    assert assets.loc[0, "id"] == "asset_1"
    assert assets.loc[0, "metadata"] == '{"caption":"Figure 1"}'
    assert chunks.loc[0, "asset_ids"] == ["asset_1"]


def test_pipeline_reuses_local_cache_on_second_run(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
            "cache": {"provider": "local", "path": cache_dir},
        }
    )

    first = _build_pipeline(settings).run()
    second = _build_pipeline(settings).run()

    assert first.graph_metrics["cache"] == {
        "ocr_hits": 0,
        "ocr_misses": 1,
        "extraction_hit": False,
    }
    assert second.graph_metrics["cache"] == {
        "ocr_hits": 1,
        "ocr_misses": 0,
        "extraction_hit": True,
    }
    assert (cache_dir / "ocr").exists()
    assert (cache_dir / "extraction").exists()


def test_pipeline_records_filter_decision_trace_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "gleaning_max_passes": 0,
                "entity_blocklist": ["Acme Corp"],
            },
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    batch = _build_pipeline(settings).run()

    dropped = [
        event
        for event in batch.extraction_trace
        if event["stage"] == "filter_decision" and event["action"] == "dropped"
    ]
    assert any(
        event["kind"] == "entity"
        and event["reason"] == "blocklisted_entity"
        and event["name"] == "Acme Corp"
        for event in dropped
    )
    assert batch.graph_metrics["filtering"]["dropped_entities_by_reason"] == {
        "blocklisted_entity": 1
    }
    assert (output_dir / "extraction_trace.jsonl").exists()


def test_pipeline_records_merge_decision_trace_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp. Alice Smith supports Acme Corp.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    batch = _build_pipeline(settings).run()

    merge_events = [event for event in batch.extraction_trace if event["stage"] == "merge_decision"]
    assert merge_events
    assert batch.graph_metrics["merge"]["decision_actions"]["created"] > 0
    assert "canonical_node_created" in batch.graph_metrics["merge"]["decision_reasons"]
    assert (output_dir / "extraction_trace.jsonl").exists()


def test_pipeline_processes_only_claimed_job_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    skipped = input_dir / "skipped.txt"
    claimed = input_dir / "claimed.txt"
    skipped.write_text("Bob Jones works at Other Corp.", encoding="utf-8")
    claimed.write_text("Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=[
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                file_id="app-file-claimed",
                source_uri=claimed.resolve().as_uri(),
            )
        ],
    )

    batch = pipeline.run()
    job_file_results = pipeline.job_file_results(batch)

    assert batch.run_report["files_processed"] == 1
    assert batch.documents[0]["file_id"] == "app-file-claimed"
    assert batch.documents[0]["source_uri"] == claimed.resolve().as_uri()
    assert batch.extraction_trace[0] == {
        "stage": "file_source",
        "files_available": 2,
        "files_seen": 1,
        "claimed_files": 1,
    }
    assert [result.file_id for result in job_file_results] == ["app-file-claimed"]
    assert job_file_results[0].rows_written > 0
    assert job_file_results[0].rows_written == sum(job_file_results[0].row_counts.values())
    assert job_file_results[0].row_counts["documents"] == 1
    assert job_file_results[0].row_counts["chunks"] == len(batch.chunks)
    assert job_file_results[0].audit == {
        "graph_id": "graph",
        "write_scope": "file_batch",
        "quality_ok": True,
    }
    assert job_file_results[0].ocr_provider == "builtin_text"
    assert skipped.resolve().as_uri() not in {str(row.get("source_uri")) for row in batch.documents}


def test_pipeline_continues_claimed_batch_after_one_ocr_failure(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    bad = input_dir / "bad.txt"
    good = input_dir / "good.txt"
    bad.write_text("This file simulates a corrupt source document.", encoding="utf-8")
    good.write_text("Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=FailingForPathOcrProvider(fail_path=bad),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=[
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                file_id="bad-claim",
                source_uri=bad.resolve().as_uri(),
            ),
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                file_id="good-claim",
                source_uri=good.resolve().as_uri(),
            ),
        ],
    )

    batch = pipeline.run()
    completed = pipeline.job_file_results(batch)
    failed = pipeline.failed_job_file_results()

    assert [row["file_id"] for row in batch.documents] == ["good-claim"]
    assert [result.file_id for result in completed] == ["good-claim"]
    assert [result.file_id for result in failed] == ["bad-claim"]
    assert failed[0].stage == "ocr_failed"
    assert failed[0].rows_written == 0
    assert failed[0].audit["graph_id"] == "graph"
    assert "simulated OCR failure" in str(failed[0].audit["error"])
    # The graph names it too, and the batch reindexes it: a later success for
    # the file must replace this failure row.
    assert [row.file_id for row in batch.failed_documents] == ["bad-claim"]
    assert "bad-claim" in batch.reindex_file_ids


def _two_file_settings(tmp_path: Path) -> tuple[Settings, Path, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    bad = input_dir / "bad.txt"
    good = input_dir / "good.txt"
    bad.write_text("This file simulates a document no parser can open.", encoding="utf-8")
    good.write_text("Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    return settings, bad, good


def _pipeline_failing_for(settings: Settings, bad: Path) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=FailingForPathOcrProvider(fail_path=bad),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )


def test_pipeline_records_an_unreadable_document_and_builds_the_rest(tmp_path: Path) -> None:
    """One file no parser can open costs that file, not the run, and is named."""

    settings, bad, good = _two_file_settings(tmp_path)

    batch = _pipeline_failing_for(settings, bad).run()

    assert [row["source_uri"] for row in batch.documents] == [good.resolve().as_uri()]
    assert batch.nodes
    (failure,) = batch.failed_documents
    assert failure.source_uri == bad.resolve().as_uri()
    assert failure.document_id == failure.file_id
    assert failure.error_type == "RuntimeError"
    assert failure.error_message == "simulated OCR failure"
    assert batch.graph_metrics["failed_documents"] == {
        "documents": 1,
        "by_error_type": {"RuntimeError": 1},
    }
    written = pd.read_parquet(tmp_path / "out" / "failed_documents.parquet")
    assert written["file_id"].tolist() == [failure.file_id]
    assert written["error_message"].tolist() == ["simulated OCR failure"]
    report = inspect_local_graph(tmp_path / "out")
    assert report["tables"]["failed_documents"] == 1


def test_pipeline_fails_the_run_when_no_document_can_be_read(tmp_path: Path) -> None:
    """Recording failures never turns an empty graph into a successful run."""

    settings, bad, good = _two_file_settings(tmp_path)
    good.unlink()

    with pytest.raises(ValueError, match="empty graph"):
        _pipeline_failing_for(settings, bad).run()


def test_prepare_documents_raises_the_failure_when_asked_to(tmp_path: Path) -> None:
    """A caller with attempts left retries the document instead of recording it."""

    settings, bad, _good = _two_file_settings(tmp_path)
    pipeline = _pipeline_failing_for(settings, bad)
    (bad_file,) = [file for file in pipeline.file_source.list_files() if file.path == bad]

    with pytest.raises(RuntimeError, match="simulated OCR failure"):
        pipeline.prepare_documents([bad_file], record_failures=False)

    recorded = pipeline.prepare_documents([bad_file])
    assert recorded.documents_processed == 0
    assert [event["file_id"] for event in recorded.trace if event.get("status") == "failed"] == [
        bad_file.id
    ]


def test_pipeline_reraises_systemic_ocr_failure_for_claimed_batch(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "source.txt"
    source.write_text("Alice works at Acme.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )

    class _UnauthorizedOcr:
        def parse(self, _file: InputFile, _options: OcrOptions) -> ParsedDocument:
            raise RuntimeError("401 unauthorized: invalid API key")

    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=_UnauthorizedOcr(),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=[
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                file_id="source",
                source_uri=source.resolve().as_uri(),
            )
        ],
    )

    with pytest.raises(RuntimeError, match="unauthorized"):
        pipeline.run()


def test_pipeline_refuses_empty_graph_snapshot_replacement(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Readable prose that names nothing: the document parses, and the graph it
    # produces is empty, which is the condition under test.
    (input_dir / "unnamed.txt").write_text(
        "the meeting covered several topics and ended without decisions.\n",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir, "location_page": False},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )

    with pytest.raises(ValueError, match="refusing to publish"):
        _build_pipeline(settings).run()

    assert not (tmp_path / "out").exists()


def test_pipeline_claim_matching_uses_source_paths_not_basenames(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    first_dir = input_dir / "first"
    second_dir = input_dir / "second"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    skipped = first_dir / "shared.txt"
    claimed = second_dir / "shared.txt"
    skipped.write_text("Bob Jones works at Other Corp.", encoding="utf-8")
    claimed.write_text("Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=[
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                # Queue systems often use display-ish file ids. This must not
                # become a basename match when two source paths share a name.
                file_id="shared.txt",
                source_uri="second/shared.txt",
            )
        ],
    )

    batch = pipeline.run()

    assert batch.run_report["files_processed"] == 1
    assert batch.documents[0]["file_id"] == "shared.txt"
    assert batch.documents[0]["source_uri"] == claimed.resolve().as_uri()
    assert skipped.resolve().as_uri() not in {str(row.get("source_uri")) for row in batch.documents}


def test_pipeline_fails_before_writing_when_quality_checks_fail(tmp_path: Path) -> None:
    settings = _quality_gate_settings(tmp_path, fail_on_quality_error=True)
    writer = RecordingWriter()
    pipeline = _build_pipeline_with_bad_embeddings(settings, writer)

    with pytest.raises(GraphQualityError, match="node_embedding_dimensions") as exc_info:
        pipeline.run()

    failed_checks = {check.name for check in exc_info.value.result.checks if not check.ok}
    assert "node_embedding_dimensions" in failed_checks
    assert writer.batch is None


def test_pipeline_can_write_failed_quality_for_diagnostics(tmp_path: Path) -> None:
    settings = _quality_gate_settings(tmp_path, fail_on_quality_error=False)
    writer = RecordingWriter()
    pipeline = _build_pipeline_with_bad_embeddings(settings, writer)

    batch = pipeline.run()

    assert writer.batch is batch
    assert batch.graph_metrics["quality"]["ok"] is False
    quality_events = [event for event in batch.extraction_trace if event["stage"] == "quality"]
    assert len(quality_events) == 1
    assert quality_events[0]["ok"] is False
    assert {check["name"] for check in quality_events[0]["failed_checks"]} >= {
        "node_embedding_dimensions",
        "edge_embedding_dimensions",
    }


def _build_pipeline(settings: Settings) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )


def _build_pipeline_with_bad_embeddings(
    settings: Settings,
    writer: RecordingWriter,
) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=WrongDimensionEmbeddingProvider(),
        writer=writer,
        cache=build_cache(settings),
    )


def _quality_gate_settings(
    tmp_path: Path,
    *,
    fail_on_quality_error: bool,
) -> Settings:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen. Acme Corp supports Contoso Health.",
        encoding="utf-8",
    )
    return Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "gleaning_max_passes": 0,
                "fail_on_quality_error": fail_on_quality_error,
            },
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )


class WrongDimensionEmbeddingProvider:
    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        return [[0.1] for _ in texts]


class AssetOcrProvider:
    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        text = file.path.read_text(encoding="utf-8")
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=[
                ParsedPage(
                    page_number=1,
                    markdown=text,
                    raw_text=text,
                    blocks=[
                        LayoutBlock(
                            id="block_1",
                            page_number=1,
                            kind="paragraph",
                            text=text,
                        )
                    ],
                )
            ],
            assets=[
                ParsedAsset(
                    id="asset_1",
                    kind="image",
                    page_number=1,
                    uri="file:///figure-1.png",
                    metadata={"caption": "Figure 1"},
                )
            ],
            provider_metadata={"provider": "asset_ocr"},
        )


class FailingForPathOcrProvider:
    def __init__(self, fail_path: Path) -> None:
        self.fail_path = fail_path

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        if file.path == self.fail_path:
            raise RuntimeError("simulated OCR failure")
        text = file.path.read_text(encoding="utf-8")
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=[
                ParsedPage(
                    page_number=1,
                    markdown=text,
                    raw_text=text,
                    blocks=[
                        LayoutBlock(
                            id=f"block-{file.id}",
                            page_number=1,
                            kind="paragraph",
                            text=text,
                        )
                    ],
                )
            ],
            provider_metadata={"provider": "failing_path_ocr"},
        )


class RecordingWriter:
    def __init__(self) -> None:
        self.batch: GraphWriteBatch | None = None

    def write(self, batch: GraphWriteBatch) -> None:
        self.batch = batch

    @contextmanager
    def stage_publication(
        self, batch: GraphWriteBatch, *, publication_id: str, generation: int
    ) -> Iterator[Callable[[], None]]:
        yield lambda: self.write(batch)


class RecordingProgressSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _stage_statuses(records: list[dict[str, object]]) -> set[tuple[object, object]]:
    return {(record["stage"], record["status"]) for record in records}


class _InventingLlm(FakeLlmProvider):
    """The fake model, plus one entity per window whose quote the text does not contain."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        result = super().complete_structured(request)
        if not request.task_name.startswith("entity_extraction"):
            return result
        chunk_id = json.loads(request.user[request.user.index("{") :])["chunks"][0]["id"]
        invented = {
            "name": "Imaginary Corp",
            "type": result.payload["entities"][0]["type"]
            if result.payload["entities"]
            else "CONCEPT",
            "description": "Named nowhere in the text.",
            "source_chunk_id": chunk_id,
            "quote": "Imaginary Corp was founded",
            "confidence": 1.0,
            "aliases": [],
        }
        return result.model_copy(
            update={
                "payload": {**result.payload, "entities": [*result.payload["entities"], invented]}
            }
        )


def test_a_local_run_lists_every_rejected_record_and_its_gaps(tmp_path: Path) -> None:
    """A local run reports what validation turned away, as a fleet run does."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "people.txt").write_text("Alice Smith works at Acme Corp.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir, "location_page": False},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    pipeline = _build_pipeline(settings)
    pipeline.llm = _InventingLlm()

    batch = pipeline.run()

    rejected = [(row.kind, row.reason, row.name) for row in batch.rejected_records]
    assert ("entity", "ungrounded_quote", "Imaginary Corp") in rejected
    assert batch.graph_metrics["rejected_records"]["records"] == len(batch.rejected_records)
    assert (tmp_path / "out" / "rejected_records.parquet").exists()
