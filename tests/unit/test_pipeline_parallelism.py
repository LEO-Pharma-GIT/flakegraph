"""Concurrency contracts for provider-bound pipeline stages."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
from typing import Any, cast

import pytest

from kg_processor.application.llm_extractors import LlmEntityExtractor
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import Settings
from kg_processor.domain.documents import InputFile, ParsedDocument, ParsedPage
from kg_processor.domain.extraction import EntityExtractionOutcome, EntityMention, ExtractionWindow
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.factories import (
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.ocr import NoTextReadError


def test_pipeline_close_releases_each_unique_dependency_once(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"input_path": tmp_path},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    class _Closeable:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    dependency = _Closeable()
    shared = cast(Any, dependency)
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=shared,
        ocr=shared,
        llm=shared,
        embeddings=shared,
        writer=shared,
        cache=shared,
    )

    pipeline.close()

    assert dependency.close_calls == 1


def test_document_context_calls_run_concurrently_and_return_in_source_order(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Parallelize independent prefixes without making persisted order nondeterministic."""

    pipeline = _context_pipeline(tmp_path, "parallel-context")
    chunks = [_context_chunk(index) for index in range(3)]
    all_started = Event()
    state_lock = Lock()
    started = 0

    def extract_context(
        _extractor: LlmEntityExtractor,
        window: ExtractionWindow,
        _ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
    ) -> EntityExtractionOutcome:
        """Block each fake request until all document calls have entered concurrently."""

        nonlocal started
        _ = (model, timeout_seconds, max_entities)
        with state_lock:
            started += 1
            if started == len(chunks):
                all_started.set()
        if not all_started.wait(timeout=1.0):
            raise AssertionError("document-context requests were executed serially")
        index = int(window.document_id.rsplit("-", 1)[1])
        return EntityExtractionOutcome(
            entities=[
                EntityMention(
                    id=f"mention-{index}",
                    name=f"Paper {index}",
                    type="PAPER",
                    description=f"Paper {index}",
                    source_chunk_id=window.chunks[0].id,
                    quote=f"Paper {index}",
                    is_document_context=True,
                )
            ],
            trace={"window_index": index},
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        LlmEntityExtractor,
        "extract_document_context_entities",
        extract_context,
    )

    entities, trace = pipeline._extract_document_context_entities(chunks)

    assert [entity.name for entity in entities] == ["Paper 0", "Paper 1", "Paper 2"]
    assert [event["window_index"] for event in trace] == [0, 1, 2]


def test_all_document_context_failures_are_reported_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _context_pipeline(tmp_path, "failed-context")

    def fail_context(*_args: object, **_kwargs: object) -> EntityExtractionOutcome:
        raise ValueError("malformed optional context response")

    monkeypatch.setattr(
        LlmEntityExtractor,
        "extract_document_context_entities",
        fail_context,
    )

    entities, trace = pipeline._extract_document_context_entities(
        [_context_chunk(index) for index in range(3)]
    )

    assert entities == []
    assert [event["stage"] for event in trace[:-1]] == ["document_context_error"] * 3
    assert trace[-1] == {
        "stage": "document_context_summary",
        "status": "degraded",
        "windows": 3,
        "failed_windows": 3,
        "entities": 0,
    }


def _context_pipeline(tmp_path: Path, job_id: str) -> KgProcessorPipeline:
    """Build a pipeline whose document-context extraction runs three windows at once."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"job_id": job_id, "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "ontology": {"profile_path": "data/deep_learning_papers/ontology.yaml"},
            "graph": {"extraction_parallelism": 3},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    )


def _context_chunk(index: int) -> Chunk:
    """Build one deterministic prefix chunk for an independent source document."""

    content = f"Paper {index} presents a method."
    return Chunk(
        id=f"chunk-{index}",
        graph_id="graph",
        file_id=f"file-{index}",
        document_id=f"document-{index}",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=6,
        content_hash=sha256_hex(content),
    )


def test_a_dataset_is_profiled_instead_of_searched_for_document_context(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """A spreadsheet's opening decides its kind; other documents keep context extraction."""

    pipeline = _context_pipeline(tmp_path, "dataset-profile")
    chunks = [_context_chunk(index) for index in range(2)]
    profiled: list[tuple[str, str]] = []

    def profile(
        _extractor: LlmEntityExtractor,
        window: ExtractionWindow,
        _ontology: OntologyProfile,
        *,
        file_name: str,
        model: str,
        timeout_seconds: int,
    ) -> dict[str, object]:
        _ = (model, timeout_seconds)
        profiled.append((window.document_id, file_name))
        return {"stage": "dataset_profile", "document_id": window.document_id, "kind": "other"}

    monkeypatch.setattr(LlmEntityExtractor, "profile_dataset", profile)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        LlmEntityExtractor,
        "extract_document_context_entities",
        lambda *_args, **_kwargs: EntityExtractionOutcome(
            trace={"stage": "document_context_extraction"}
        ),
    )

    _entities, trace = pipeline._extract_document_context_entities(  # noqa: SLF001
        chunks,
        {
            "document-0": {"kind": "dataset", "source_uri": "file:///in/Vitic%20list.xlsx"},
            "document-1": {"kind": "document", "source_uri": "file:///in/paper.pdf"},
        },
    )

    assert profiled == [("document-0", "Vitic list.xlsx")]
    assert [event["stage"] for event in trace] == ["dataset_profile", "document_context_extraction"]


def test_an_image_without_text_is_kept_as_an_image_document(tmp_path: Path) -> None:
    """A photograph with nothing to read is an image asset, not a failed parse."""

    pipeline = _context_pipeline(tmp_path, "image-without-text")
    (tmp_path / "input" / "micrograph.tif").write_bytes(b"II*\x00")
    (tmp_path / "input" / "scan.pdf").write_bytes(b"%PDF-1.4")

    class _Blind:
        def parse(self, file: object, options: object) -> object:
            raise NoTextReadError("no text")

    pipeline.ocr = cast(Any, _Blind())
    prepared = pipeline._parse_and_chunk_files(pipeline.file_source.list_files())  # noqa: SLF001

    [image] = prepared.document_rows
    assert image["kind"] == "image"
    assert [row["kind"] for row in prepared.asset_rows] == ["image"]
    failed = [event for event in prepared.trace if event.get("status") == "failed"]
    assert [event["source_uri"].rsplit("/", 1)[-1] for event in failed] == ["scan.pdf"]


def test_an_image_the_parser_only_pictured_is_kept_as_an_image_document(tmp_path: Path) -> None:
    """A parse holding nothing but a reference to the picture is an image without text."""

    pipeline = _context_pipeline(tmp_path, "image-reference-only")
    (tmp_path / "input" / "nozzle.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "input" / "label.png").write_bytes(b"\x89PNG")

    class _Picturing:
        def parse(self, file: InputFile, options: object) -> ParsedDocument:
            text = "![](images/0a1b.jpg)" + (
                "\n\nGranulator G-10" if "label" in file.source_uri else ""
            )
            return ParsedDocument(
                file_id=file.id,
                checksum=file.checksum,
                source_uri=file.source_uri,
                mime_type=file.mime_type,
                pages=[ParsedPage(page_number=1, markdown=text, raw_text=text)],
            )

    pipeline.ocr = cast(Any, _Picturing())
    prepared = pipeline._parse_and_chunk_files(pipeline.file_source.list_files())  # noqa: SLF001

    assert sorted(row["kind"] for row in prepared.document_rows) == ["image", "image"]
    # Both keep the picture itself; only the label has text to extract.
    assert [row["kind"] for row in prepared.asset_rows] == ["image", "image"]
    read = [chunk.content for chunk in prepared.chunks if chunk.page_number > 0]
    assert len(read) == 1
    assert "Granulator G-10" in read[0]


def test_each_document_starts_with_a_location_page(tmp_path: Path) -> None:
    """Page 0 names the folders and file, so their names can be read and grounded."""

    pipeline = _context_pipeline(tmp_path, "location-page")
    folder = tmp_path / "input" / "Formulation Lab" / "Trial 3"
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_text("Granulation with lactose.")

    prepared = pipeline._parse_and_chunk_files(pipeline.file_source.list_files())  # noqa: SLF001

    [document] = prepared.document_rows
    assert document["folder"] == "Formulation Lab/Trial 3"
    first = prepared.chunks[0]
    assert first.page_number == 0
    assert first.content.startswith("Location: Formulation Lab/Trial 3/notes.txt")
    assert any("lactose" in chunk.content for chunk in prepared.chunks[1:])


def test_a_location_carried_with_a_file_wins_over_the_readers_roots(tmp_path: Path) -> None:
    """A staged copy read under another root keeps the folders it was discovered in."""

    pipeline = _context_pipeline(tmp_path, "carried-location")
    (tmp_path / "input" / "notes.txt").write_text("Granulation with lactose.")
    files = [
        file.model_copy(update={"location": "Lab A/Trial 3/notes.txt"})
        for file in pipeline.file_source.list_files()
    ]

    prepared = pipeline._parse_and_chunk_files(files)  # noqa: SLF001

    [document] = prepared.document_rows
    assert document["folder"] == "Lab A/Trial 3"
    assert prepared.chunks[0].content.startswith("Location: Lab A/Trial 3/notes.txt")
