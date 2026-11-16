"""Prove split extraction execution preserves local graph semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import ExtractionObservations
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.stages import ExtractedDocumentShard
from kg_processor.factories import build_pipeline


def test_split_extraction_stages_match_single_process_pipeline(tmp_path: Path) -> None:
    """Local and split execution must produce identical canonical graph artifacts.

    This exercises real OCR normalization, chunking, two-pass extraction, entity
    resolution, merge, embeddings, communities, and quality logic. Only providers
    are deterministic test implementations, keeping the assertion fast and focused
    on FlakeGraph's own stage boundaries.
    """

    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "judo.txt").write_text(
        "Jigoro Kano founded Kodokan Judo in Tokyo. Kodokan Judo teaches Randori.",
        encoding="utf-8",
    )
    (input_path / "karate.txt").write_text(
        "Okinawan Karate teaches Kata. Gichin Funakoshi introduced Karate in Japan.",
        encoding="utf-8",
    )
    settings = _settings(input_path, tmp_path / "local-output")

    local_batch = build_pipeline(settings).run()

    split_pipeline = build_pipeline(settings)
    files = split_pipeline.file_source.list_files()
    extracted: list[ExtractedDocumentShard] = []
    for input_file in reversed(files):
        prepared = split_pipeline.prepare_documents([input_file])
        contextualized = split_pipeline.extract_document_context(prepared)
        entities = split_pipeline.extract_window_entities(contextualized)
        relations = split_pipeline.extract_window_relations(
            contextualized,
            entities.entities,
        )
        extracted.append(
            ExtractedDocumentShard(
                prepared=prepared,
                observations=ExtractionObservations(
                    entities=entities.entities,
                    relations=relations.relations,
                    trace=[*contextualized.trace, *entities.trace, *relations.trace],
                    chunk_count=len(prepared.chunks),
                    window_count=1,
                ),
            )
        )
    split_batch = split_pipeline.finalize_document_shards(extracted, write=False)

    _assert_semantic_batch_equality(local_batch, split_batch)


def test_split_execution_names_an_unreadable_document_like_the_local_pipeline(
    tmp_path: Path,
) -> None:
    """A file no parser can open is recorded, not fatal, on both execution shapes.

    The split side does what a worker does on a document's last attempt: the
    preparation records the failure, yields no chunks, and the document's
    extracted record is empty but for that trace.
    """

    input_path = tmp_path / "input"
    input_path.mkdir()
    (input_path / "judo.txt").write_text(
        "Jigoro Kano founded Kodokan Judo in Tokyo. Kodokan Judo teaches Randori.",
        encoding="utf-8",
    )
    (input_path / "broken.pdf").write_text("this is not a pdf", encoding="utf-8")
    settings = _settings(input_path, tmp_path / "local-output")

    local_batch = build_pipeline(settings).run()

    split_pipeline = build_pipeline(settings)
    extracted: list[ExtractedDocumentShard] = []
    for input_file in split_pipeline.file_source.list_files():
        prepared = split_pipeline.prepare_documents([input_file])
        if not prepared.chunks:
            extracted.append(
                ExtractedDocumentShard(
                    prepared=prepared,
                    observations=ExtractionObservations(chunk_count=0, window_count=0),
                    trace=prepared.trace,
                )
            )
            continue
        contextualized = split_pipeline.extract_document_context(prepared)
        entities = split_pipeline.extract_window_entities(contextualized)
        relations = split_pipeline.extract_window_relations(contextualized, entities.entities)
        extracted.append(
            ExtractedDocumentShard(
                prepared=prepared,
                observations=ExtractionObservations(
                    entities=entities.entities,
                    relations=relations.relations,
                    trace=[*contextualized.trace, *entities.trace, *relations.trace],
                    chunk_count=len(prepared.chunks),
                    window_count=1,
                ),
            )
        )
    split_batch = split_pipeline.finalize_document_shards(extracted, write=False)

    (failure,) = local_batch.failed_documents
    assert failure.source_uri.endswith("/broken.pdf")
    assert failure.error_message
    assert [row["source_uri"].rsplit("/", 1)[-1] for row in local_batch.documents] == ["judo.txt"]
    _assert_semantic_batch_equality(local_batch, split_batch)


def _settings(input_path: Path, output_path: Path) -> Settings:
    """Build the deterministic two-pass profile used by both execution shapes."""

    return Settings.load(
        env={},
        overrides={
            "job": {"job_id": "equivalence", "graph_id": "equivalence-graph"},
            "files": {"source": "local", "input_path": input_path},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "model": "hash", "dimension": 8},
            "graph": {
                "chunk_token_size": 80,
                "chunk_token_overlap": 10,
                "gleaning_max_passes": 0,
                "verify_relations": True,
                "fail_on_quality_error": False,
            },
            "writer": {"provider": "local_artifacts", "output_path": output_path},
            "cache": {"provider": "none"},
        },
    )


def _assert_semantic_batch_equality(
    local_batch: GraphWriteBatch,
    split_batch: GraphWriteBatch,
) -> None:
    """Compare every persisted semantic table while excluding timing-only metadata."""

    assert local_batch.graph_id == split_batch.graph_id
    assert local_batch.documents == split_batch.documents
    assert local_batch.pages == split_batch.pages
    assert local_batch.blocks == split_batch.blocks
    assert local_batch.assets == split_batch.assets
    for field in (
        "chunks",
        "nodes",
        "edges",
        "evidence",
        "entity_sources",
        "communities",
        "community_findings",
        "discarded_windows",
        "failed_documents",
    ):
        assert _dump_models(getattr(local_batch, field)) == _dump_models(
            getattr(split_batch, field)
        )
    stable_report_fields = {
        "graph_id",
        "files_seen",
        "files_processed",
        "chunks_created",
        "entities_extracted",
        "relations_extracted",
        "nodes_created",
        "edges_created",
        "evidence_created",
        "ordered_chunk_hash",
    }
    assert {key: local_batch.run_report[key] for key in stable_report_fields} == {
        key: split_batch.run_report[key] for key in stable_report_fields
    }


def _dump_models(values: list[Any]) -> list[dict[str, Any]]:
    """Serialize Pydantic rows into exact JSON-compatible comparison values."""

    dumped: list[dict[str, Any]] = []
    for value in values:
        assert isinstance(value, BaseModel)
        dumped.append(value.model_dump(mode="json"))
    return dumped
