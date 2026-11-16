# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for complete, configuration-sensitive extraction caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pandas as pd
import pytest

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


def _settings(root: Path, **llm_changes: Any) -> Settings:
    return Settings.load(
        overrides={
            "job": {"job_id": "cache-test", "graph_id": "cache-graph"},
            "files": {
                "input_path": root / "input",
                "location_page": False,
                "include_globs": ["**/*"],
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake", **llm_changes},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"gleaning_max_passes": 0},
            "cache": {"provider": "local", "path": root / "cache"},
            "writer": {"provider": "local_artifacts", "output_path": root / "out"},
        }
    )


def _pipeline(settings: Settings, llm: Any) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings,
        build_file_source(settings),
        build_ocr_provider(settings),
        llm,
        build_embedding_provider(settings),
        build_writer(settings),
        build_cache(settings),
    )


def _input(root: Path, *, second_document: bool = False) -> None:
    (root / "input").mkdir()
    (root / "input/alice.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8"
    )
    if second_document:
        (root / "input/bob.txt").write_text(
            "Bob Jones works at Beta Corp in London.", encoding="utf-8"
        )


def _topology(batch: GraphWriteBatch) -> tuple[list[str], list[tuple[str, str, str]]]:
    names = {node.id: node.name for node in batch.nodes}
    return sorted(names.values()), sorted(
        (names[edge.source_node_id], names[edge.target_node_id], edge.relation_type)
        for edge in batch.edges
    )


def _transport_run(settings: Settings, option: str) -> tuple[GraphWriteBatch, list[str]]:
    calls: list[str] = []
    fake = FakeLlmProvider()

    def respond(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        schema = data["response_format"]["json_schema"]
        task = schema["name"]
        calls.append(task)
        result = fake.complete_structured(
            StructuredCompletionRequest(
                task_name=task,
                model=settings.llm.model,
                system=data["messages"][0]["content"],
                user=data["messages"][1]["content"],
                json_schema=schema["schema"],
            )
        )
        limited = (
            (option == "endpoint" and request.url.host == "first.example")
            or (option == "max_output_tokens" and data["max_tokens"] == 1000)
            or (option == "api_version" and request.url.params.get("api-version") == "first")
        )
        if limited and task == "entity_extraction":
            result.payload["entities"] = [
                entity for entity in result.payload["entities"] if entity["name"] != "Copenhagen"
            ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(result.payload)}, "finish_reason": "stop"}
                ]
            },
        )

    llm = build_llm_provider(settings)
    llm._client.close()  # type: ignore[attr-defined]  # noqa: SLF001
    llm._client = httpx.Client(transport=httpx.MockTransport(respond))  # type: ignore[attr-defined]  # noqa: SLF001
    try:
        return _pipeline(settings, llm).run(), calls
    finally:
        llm._client.close()  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.parametrize("option", ["endpoint", "max_output_tokens", "api_version"])
def test_changed_effective_llm_options_match_a_cold_cache(tmp_path: Path, option: str) -> None:
    """Use supported adapters and complete wire responses to detect stale graphs."""

    _input(tmp_path)
    settings = _settings(
        tmp_path,
        provider="azure_openai" if option == "api_version" else "openai_compatible",
        model="deployment",
        endpoint="https://first.example/v1",
        api_key="synthetic",
        max_output_tokens=1000,
        api_version="first",
    )
    initial, initial_calls = _transport_run(settings, option)
    changes: dict[str, str | int] = {
        "endpoint": "https://second.example/v1",
        "max_output_tokens": 8000,
        "api_version": "second",
    }
    changed = settings.model_copy(deep=True)
    setattr(changed.llm, option, changes[option])
    warm, warm_calls = _transport_run(changed, option)
    cold_settings = changed.model_copy(deep=True)
    cold_settings.cache.path = tmp_path / "cold-cache"
    cold, cold_calls = _transport_run(cold_settings, option)

    assert initial_calls and warm_calls and cold_calls
    assert _topology(initial) != _topology(warm)
    assert _topology(warm) == _topology(cold)
    assert not warm.graph_metrics["cache"]["extraction_hit"]
    assert all(batch.graph_metrics["quality"]["ok"] for batch in (initial, warm, cold))

    rotated = changed.model_copy(deep=True)
    rotated.llm.api_key = "rotated-synthetic"
    reused, reused_calls = _transport_run(rotated, option)
    assert reused.graph_metrics["cache"]["extraction_hit"]
    assert reused_calls == []
    assert _topology(reused) == _topology(cold)


class _ProfileProvider(FakeLlmProvider):
    """Return a deterministic dataset summary and count profile requests."""

    def __init__(self) -> None:
        self.profiles = 0

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        if request.task_name == "dataset_profile":
            self.profiles += 1
            return StructuredCompletionResult(
                payload={
                    "kind": "measurements",
                    "summary": "A synthetic measurement dataset.",
                    "columns": [],
                }
            )
        return super().complete_structured(request)


def _workbook(path: Path) -> None:
    """Write a minimal OOXML dataset without an optional spreadsheet library."""

    sheet_type = "application/vnd.openxmlformats-officedocument.spreadsheetml"
    relation_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    members = {
        "[Content_Types].xml": f"""<Types
            xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
            <Default Extension="rels"
                ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
            <Default Extension="xml" ContentType="application/xml"/>
            <Override PartName="/xl/workbook.xml"
                ContentType="{sheet_type}.sheet.main+xml"/>
            <Override PartName="/xl/worksheets/sheet1.xml"
                ContentType="{sheet_type}.worksheet+xml"/>
            </Types>""",
        "_rels/.rels": f"""<Relationships
            xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId1"
                Type="{relation_type}/officeDocument"
                Target="xl/workbook.xml"/></Relationships>""",
        "xl/workbook.xml": """<workbook
            xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <sheets><sheet name="Measurements" sheetId="1" r:id="rId1"/></sheets></workbook>""",
        "xl/_rels/workbook.xml.rels": f"""<Relationships
            xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
            <Relationship Id="rId1"
                Type="{relation_type}/worksheet"
                Target="worksheets/sheet1.xml"/></Relationships>""",
        "xl/worksheets/sheet1.xml": """<worksheet
            xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
            <row r="1"><c r="A1" t="inlineStr"><is><t>Notes</t></is></c></row>
            <row r="2"><c r="A2" t="inlineStr"><is><t>Alice Smith works at Acme Corp.
            Acme Corp supports Contoso Health.</t></is></c></row></sheetData></worksheet>""",
    }
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)


def test_cached_profile_is_replayed_into_persisted_document_summary(tmp_path: Path) -> None:
    (tmp_path / "input").mkdir()
    _workbook(tmp_path / "input/measurements.xlsx")
    settings = _settings(tmp_path)
    provider = _ProfileProvider()
    cold = _pipeline(settings, provider).run()
    warm = _pipeline(settings, provider).run()
    persisted = pd.read_parquet(tmp_path / "out/documents.parquet")

    assert provider.profiles == 1
    assert warm.graph_metrics["cache"]["extraction_hit"]
    assert (
        cold.documents[0]["summary"] == warm.documents[0]["summary"] == persisted.iloc[0]["summary"]
    )
    assert warm.documents[0]["summary"] == "A synthetic measurement dataset."
    assert (
        len([event for event in warm.extraction_trace if event["stage"] == "dataset_profile"]) == 1
    )
    assert _topology(cold) == _topology(warm)


class _RecoveringProvider(FakeLlmProvider):
    """Fail one independent body window or return intentionally invalid records."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[str] = []

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        self.calls.append(request.task_name)
        bob = "Bob Jones" in request.user
        if bob and request.task_name == f"{self.mode}_extraction":
            raise RuntimeError("Temporary synthetic window failure")
        result = super().complete_structured(request)
        if bob and self.mode == "rejection" and request.task_name == "entity_extraction":
            for entity in result.payload["entities"]:
                entity.update(name="Absent synthetic entity", quote="Absent synthetic quotation")
        return result


@pytest.mark.parametrize("stage", ["entity", "relation"])
def test_failed_body_windows_are_retried_after_provider_recovery(
    tmp_path: Path, stage: str
) -> None:
    _input(tmp_path, second_document=True)
    settings = _settings(tmp_path)
    provider = _RecoveringProvider(stage)
    first = _pipeline(settings, provider).run()
    assert not list((tmp_path / "cache/extraction").glob("*.json"))
    provider.mode = "healthy"
    provider.calls.clear()
    recovered = _pipeline(settings, provider).run()
    assert not recovered.graph_metrics["cache"]["extraction_hit"]
    assert provider.calls.count(f"{stage}_extraction") == 2
    assert _topology(first) != _topology(recovered)
    provider.calls.clear()
    cached = _pipeline(settings, provider).run()
    assert cached.graph_metrics["cache"]["extraction_hit"]
    assert provider.calls == []
    assert _topology(cached) == _topology(recovered)


def test_successful_validation_rejections_remain_cacheable(tmp_path: Path) -> None:
    _input(tmp_path, second_document=True)
    settings = _settings(tmp_path)
    provider = _RecoveringProvider("rejection")
    first = _pipeline(settings, provider).run()
    assert first.rejected_records
    assert first.graph_metrics["discarded_windows"]
    provider.mode = "healthy"
    provider.calls.clear()
    cached = _pipeline(settings, provider).run()
    assert cached.graph_metrics["cache"]["extraction_hit"]
    assert provider.calls == []
    assert _topology(first) == _topology(cached)
