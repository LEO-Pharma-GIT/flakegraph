from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr import builtin_text
from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider
from kg_processor.application.inspect import (
    ArtifactTable,
    _artifact_identity_check,
    compare_local_graph_artifacts,
    inspect_local_graph,
)
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import Settings
from kg_processor.factories import (
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.ocr import OcrOptions


def test_original_pdf_fixture_is_parseable() -> None:
    source = LocalFileSource(Path("data/martial_arts/files/martial-arts-overview.pdf"))
    file = source.list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

    assert len(parsed.pages) == 4
    assert sum(len(page.raw_text) for page in parsed.pages) > 5000


def test_pdfium_fallback_parses_fixture_inside_resource_limits() -> None:
    pages = builtin_text._run_guarded_pdfium_fallback(
        Path("data/martial_arts/files/martial-arts-overview.pdf")
    )

    assert len(pages) == 4
    assert sum(len(page) for page in pages) > 5000


def test_original_text_native_office_and_html_fixtures_are_parseable() -> None:
    sample_paths = [
        Path("data/martial_arts/files/martial-arts-timeline.html"),
        Path("data/martial_arts/files/martial-arts-interview.docx"),
        Path("data/martial_arts/files/martial-arts-schools.pptx"),
        Path("data/martial_arts/files/martial-arts-lineages.pdf"),
        Path("data/martial_arts/files/martial-arts-crossroads.md"),
        Path("data/martial_arts/files/martial-arts-living-heritage.txt"),
        Path("data/martial_arts/files/martial-arts-olympic-program.md"),
        Path("data/martial_arts/files/martial-arts-rules-and-regulators.md"),
    ]

    for path in sample_paths:
        file = LocalFileSource(path).list_files()[0]
        parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

        assert parsed.pages
        assert sum(len(page.raw_text.strip()) for page in parsed.pages) > 4000


def test_inspect_reports_written_graph(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir, "location_page": False},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"output_path": output_dir},
        }
    )
    KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    ).run()

    inspection = inspect_local_graph(output_dir)

    assert inspection["tables"]["nodes"] > 0
    assert inspection["tables"]["documents"] == 1
    assert inspection["tables"]["pages"] == 1
    assert inspection["run_report"]["files_processed"] == 1
    assert inspection["graph_metrics"]["counts"]["nodes"] > 0
    assert inspection["quality"]["ok"]
    assert inspection["schema"]["ok"]
    assert inspection["top_entities"]
    assert inspection["top_entities"][0]["evidence_count"] > 0
    assert inspection["top_relations"]
    assert inspection["top_relations"][0]["source"]
    assert inspection["top_relations"][0]["target"]
    assert inspection["community_summaries"]
    assert inspection["community_summaries"][0]["member_count"] > 0
    assert inspection["community_summaries"][0]["rating_explanation"]
    assert inspection["community_summaries"][0]["suggested_questions"]
    assert inspection["trace_events"] > 0


def test_compare_local_graph_artifacts_accepts_stable_reruns(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )

    _run_pipeline_for_output(input_dir, first_output)
    _run_pipeline_for_output(input_dir, second_output)

    comparison = compare_local_graph_artifacts(first_output, second_output)

    assert comparison["ok"]
    assert all(check["ok"] for check in comparison["checks"])


def test_compare_local_graph_artifacts_reports_identity_drift(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    _run_pipeline_for_output(input_dir, first_output)
    _run_pipeline_for_output(input_dir, second_output)
    nodes_path = second_output / "nodes.parquet"
    nodes = pd.read_parquet(nodes_path)
    nodes.loc[0, "id"] = "node_unstable"
    nodes.to_parquet(nodes_path, index=False)

    comparison = compare_local_graph_artifacts(first_output, second_output)

    assert not comparison["ok"]
    node_check = next(
        check for check in comparison["checks"] if check["name"] == "nodes_stable_identity"
    )
    assert not node_check["ok"]
    assert node_check["missing_from_right"]
    assert node_check["missing_from_left"] == ["node_unstable"]


def test_compare_local_graph_artifacts_reports_schema_drift(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    _run_pipeline_for_output(input_dir, first_output)
    _run_pipeline_for_output(input_dir, second_output)
    nodes_path = second_output / "nodes.parquet"
    nodes = pd.read_parquet(nodes_path).drop(columns=["description"])
    nodes.to_parquet(nodes_path, index=False)

    comparison = compare_local_graph_artifacts(first_output, second_output)

    assert not comparison["ok"]
    node_schema_check = next(
        check for check in comparison["checks"] if check["name"] == "nodes_schema"
    )
    node_identity_check = next(
        check for check in comparison["checks"] if check["name"] == "nodes_stable_identity"
    )
    assert not node_schema_check["ok"]
    assert node_schema_check["right_missing_required"] == ["description"]
    assert node_identity_check["ok"]


def test_compare_local_graph_artifacts_reports_value_drift(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    _run_pipeline_for_output(input_dir, first_output)
    _run_pipeline_for_output(input_dir, second_output)
    nodes_path = second_output / "nodes.parquet"
    nodes = pd.read_parquet(nodes_path)
    nodes.loc[0, "description"] = "Unexpected changed description"
    nodes.to_parquet(nodes_path, index=False)

    comparison = compare_local_graph_artifacts(first_output, second_output)

    node_identity_check = next(
        check for check in comparison["checks"] if check["name"] == "nodes_stable_identity"
    )
    assert not node_identity_check["ok"]
    assert node_identity_check["value_mismatches"]


def test_artifact_identity_detects_large_array_and_duplicate_drift() -> None:
    left_embedding = np.zeros(1536)
    right_embedding = np.zeros(1536)
    right_embedding[1000] = 1.0
    columns = ["id", "embedding"]

    embedding_check = _artifact_identity_check(
        "nodes",
        ArtifactTable(True, columns, [{"id": "node-1", "embedding": left_embedding}]),
        ArtifactTable(True, columns, [{"id": "node-1", "embedding": right_embedding}]),
    )
    duplicate_check = _artifact_identity_check(
        "nodes",
        ArtifactTable(
            True,
            columns,
            [
                {"id": "node-1", "embedding": left_embedding},
                {"id": "node-1", "embedding": left_embedding},
            ],
        ),
        ArtifactTable(True, columns, [{"id": "node-1", "embedding": left_embedding}]),
    )

    assert not embedding_check["ok"]
    assert embedding_check["value_mismatches"] == ["node-1"]
    assert not duplicate_check["ok"]
    assert duplicate_check["left_rows"] == 2
    assert duplicate_check["right_rows"] == 1


def _run_pipeline_for_output(input_dir: Path, output_dir: Path) -> None:
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "stable-graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"output_path": output_dir},
        }
    )
    KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    ).run()
