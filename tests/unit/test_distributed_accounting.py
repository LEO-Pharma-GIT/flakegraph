"""Run-scoped distributed metering and durable finalization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_distributed_execution import MemoryDistributedStore, _settings

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.application.graph_filter import EntityFilterResult, RelationFilterResult
from kg_processor.application.graph_merge import GraphAssemblyResult
from kg_processor.application.metered_llm import MeteredLlmProvider
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.spark_finalization import (
    _clear_executor_provider_caches,
    _executor_llm_provider,
)
from kg_processor.domain.consumption import ConsumptionEvent, TokenUsage
from kg_processor.domain.distributed import RunDefinition, TaskStage
from kg_processor.domain.graph import ExtractionResult, GraphWriteBatch
from kg_processor.factories import build_distributed_pipeline
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


class UsageProvider(FakeLlmProvider):
    """Return normalized synthetic usage without accessing a provider service."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Attach measurable usage to a deterministic fake completion."""

        return (
            super()
            .complete_structured(request)
            .model_copy(update={"usage": TokenUsage(prompt_tokens=100, completion_tokens=20)})
        )


class LedgerStore(MemoryDistributedStore):
    """Share immutable event records between independently constructed workers."""

    def __init__(self) -> None:
        super().__init__()
        self.events: dict[tuple[str, str], ConsumptionEvent] = {}

    def record_consumption(self, run_id: str, event: ConsumptionEvent) -> None:
        """Persist one event idempotently like the durable store."""

        self.events.setdefault((run_id, event.event_id), event)

    def consumption_events(self, run_id: str) -> list[ConsumptionEvent]:
        """Return only the requested durable run's records."""

        return [event for (run, _), event in self.events.items() if run == run_id]


def _request() -> StructuredCompletionRequest:
    return StructuredCompletionRequest(
        task_name="entity_extraction", model="fake", system="", user="", json_schema={}
    )


def _batch(pipeline: KgProcessorPipeline, file_id: str) -> GraphWriteBatch:
    """Assemble the actual persisted report without invoking graph providers."""

    batch, _ = pipeline._build_write_batch(
        files_seen=1,
        documents_processed=1,
        document_rows=[
            {"file_id": file_id, "source_uri": f"synthetic://{file_id}", "checksum": "s"}
        ],
        page_rows=[],
        block_rows=[],
        asset_rows=[],
        chunks=[],
        extraction=ExtractionResult(),
        entities=[],
        relations=[],
        entity_filter=EntityFilterResult([], []),
        relation_filter=RelationFilterResult([], []),
        assembly=GraphAssemblyResult([], [], [], [], [], []),
        descriptions_merged=0,
        communities=[],
        findings=[],
        trace=[],
        ocr_cache_hits=0,
        extraction_cache_hit=False,
    )
    return batch


def test_run_and_task_collectors_persist_once_and_merge_other_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Separate extraction and finalization processes retain every billed call."""

    monkeypatch.setattr(
        "kg_processor.factories._build_raw_llm_provider", lambda _settings: UsageProvider()
    )
    settings = _settings(tmp_path)
    store = LedgerStore()
    store.run = RunDefinition(
        id="run-a", graph_id=settings.job.graph_id, config={}, config_digest="x"
    )
    stages = {TaskStage.EXTRACT_ENTITY_WINDOW, TaskStage.FINALIZE_GRAPH}
    first_pipeline = build_distributed_pipeline(settings, stages)
    first = DistributedWorker(settings, "first", stages, first_pipeline, store, store)
    run = first._run_context("run-a")
    pipeline = run.pipeline
    assert isinstance(pipeline, KgProcessorPipeline)
    task_pipeline = pipeline.for_task("task-a", 2)
    assert isinstance(task_pipeline.llm, MeteredLlmProvider)
    assert task_pipeline.entity_extractor.llm is task_pipeline.llm  # type: ignore[union-attr]
    task_pipeline.llm.complete_structured(_request())
    assert len(store.events) == 1
    event = store.consumption_events("run-a")[0]
    assert (event.run_id, event.job_id, event.task_id, event.task_attempt) == (
        "run-a",
        "run-a",
        "task-a",
        2,
    )
    store.record_consumption("run-a", event)
    assert len(store.events) == 1

    second_pipeline = build_distributed_pipeline(settings, stages)
    second = DistributedWorker(settings, "second", stages, second_pipeline, store, store)
    final_pipeline = second._run_context("run-a").pipeline
    assert isinstance(final_pipeline, KgProcessorPipeline)
    second._refresh_consumption("run-a", final_pipeline)
    second._refresh_consumption("run-a", final_pipeline)
    report = _batch(final_pipeline, "file-a")
    assert report.graph_metrics["consumption"]["totals"]["total_tokens"] == 120
    assert len(report.graph_metrics["consumption"]["events"]) == 1
    assert report.run_report["job_id"] == "run-a"
    provenance = next(row for row in report.extraction_trace if row["stage"] == "run_provenance")
    assert provenance["effective_config"]["job"]["job_id"] == "run-a"
    first_pipeline.close()
    second_pipeline.close()


def test_ocr_events_and_reports_remain_isolated_for_each_durable_run(tmp_path: Path) -> None:
    """A reused worker and graph identity never reuse another run's collector."""

    settings = _settings(tmp_path)
    store = LedgerStore()
    stages = {TaskStage.PREPARE_DOCUMENT, TaskStage.FINALIZE_GRAPH}
    pipeline = build_distributed_pipeline(settings, stages)
    worker = DistributedWorker(settings, "worker", stages, pipeline, store, store)
    reports = []
    collectors = []
    for run_id, file_id, pages in [("run-a", "file-a", 3), ("run-b", "file-b", 2)]:
        store.run = RunDefinition(
            id=run_id, graph_id=settings.job.graph_id, config={}, config_digest="x"
        )
        scoped = worker._run_context(run_id).pipeline
        assert isinstance(scoped, KgProcessorPipeline)
        collectors.append(scoped.consumption)
        scoped.consumption.record(
            stage="ocr",
            operation="parse_document",
            provider="mineru_api",
            model="mineru_api",
            pages=pages,
            file_id=file_id,
        )
        reports.append(_batch(scoped, file_id))
    assert collectors[0] is not collectors[1]
    assert [report.run_report["job_id"] for report in reports] == ["run-a", "run-b"]
    assert [report.graph_metrics["consumption"]["totals"]["pages"] for report in reports] == [3, 2]
    assert [row["file_id"] for row in reports[1].graph_metrics["consumption"]["events"]] == [
        "file-b"
    ]
    pipeline.close()


def test_spark_executor_metering_survives_provider_cache_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executor usage objects retain distinct billed calls across task retries."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "job": settings.job.model_copy(update={"job_id": "spark-run"}),
            "distributed": settings.distributed.model_copy(
                update={"artifact_uri": (tmp_path / "objects").as_uri()}
            ),
        }
    )
    monkeypatch.setattr(
        "kg_processor.factories._build_raw_llm_provider", lambda _settings: UsageProvider()
    )
    _clear_executor_provider_caches()
    try:
        _executor_llm_provider(settings).complete_structured(_request())
        _clear_executor_provider_caches()
        _executor_llm_provider(settings).complete_structured(_request())
    finally:
        _clear_executor_provider_caches()
    objects = list((tmp_path / "objects" / "spark-run" / "consumption").glob("*.json"))
    collector = ConsumptionCollector(settings.job.graph_id, job_id="spark-run", run_id="spark-run")
    for _ in range(2):
        collector.merge(ConsumptionEvent.model_validate_json(path.read_text()) for path in objects)
    assert len(collector.events) == 2
    assert sum(event.usage.total_tokens for event in collector.events) == 240


def test_billed_failure_is_durable_before_retry(tmp_path: Path) -> None:
    """Billed failed calls retain distinct event identities from later attempts."""

    store = LedgerStore()
    collector = ConsumptionCollector("graph", job_id="run", run_id="run")
    collector.bind_sink(lambda event: store.record_consumption("run", event))
    error = ValueError("synthetic billed failure")
    error.usage = TokenUsage(prompt_tokens=7, completion_tokens=3)  # type: ignore[attr-defined]

    class FailingProvider(UsageProvider):
        """Expose normalized billed usage on a failed synthetic completion."""

        def complete_structured(
            self, request: StructuredCompletionRequest
        ) -> StructuredCompletionResult:
            """Fail without network or external provider work."""

            raise error

    provider = MeteredLlmProvider(
        FailingProvider(), collector.for_task("task", 1), provider="fake", model="fake"
    )
    with pytest.raises(ValueError, match="synthetic billed failure"):
        provider.complete_structured(_request())
    assert len(store.consumption_events("run")) == 1
    retry = MeteredLlmProvider(
        UsageProvider(), collector.for_task("task", 2), provider="fake", model="fake"
    )
    retry.complete_structured(_request())
    events = store.consumption_events("run")
    assert [event.task_attempt for event in events] == [1, 2]
    assert sum(event.usage.total_tokens for event in events) == 130
    assert len({event.event_id for event in events}) == 2


@pytest.mark.parametrize(
    ("call", "stage"),
    [
        ("merge_entity_description", "description_merge"),
        ("summarize_community", "community_report"),
    ],
)
def test_billed_enrichment_failure_is_recorded(call: str, stage: str) -> None:
    """Finalization calls that fail after billing keep their usage too."""

    error = ValueError("synthetic billed failure")
    error.usage = TokenUsage(prompt_tokens=11, completion_tokens=2)  # type: ignore[attr-defined]

    class FailingProvider(UsageProvider):
        """Fail every enrichment call after reporting what it consumed."""

        def merge_entity_description(
            self, request: DescriptionMergeRequest
        ) -> DescriptionMergeResult:
            """Fail without network or external provider work."""

            raise error

        def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
            """Fail without network or external provider work."""

            raise error

    collector = ConsumptionCollector("graph", job_id="run", run_id="run")
    provider = MeteredLlmProvider(FailingProvider(), collector, provider="fake", model="fake")
    with pytest.raises(ValueError, match="synthetic billed failure"):
        getattr(provider, call)(None)
    [event] = collector.events
    assert (event.stage, event.usage.total_tokens) == (stage, 13)
