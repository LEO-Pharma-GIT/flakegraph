"""Contracts for bounded executor-side provider concurrency."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any

import pytest
from llm_fakes import CountingLlm

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.consumption import ConsumptionCollector
from kg_processor.application.enrichment_batching import (
    COMMUNITY_BATCH_SIZE,
    DESCRIPTION_BATCH_SIZE,
)
from kg_processor.application.spark_finalization import (
    SparkFinalizationRequest,
    _adaptive_provider_partitions,
    _adjudicate_partition,
    _clear_executor_provider_caches,
    _connected_component_rows,
    _effective_shuffle_partitions,
    _embed_partition,
    _executor_embedding_provider,
    _executor_enrichment_llm_provider,
    _merge_description_partition,
    _provider_threads_per_task,
    _spark_application_name,
    _summarize_community_partition,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import EntityMention
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import (
    DescriptionMergeRequest,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


def test_spark_application_identity_fences_every_durable_attempt() -> None:
    """Prevent retry collisions without trusting user-controlled run IDs."""

    first = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=1,
            artifact_ids=frozenset(),
        )
    )
    repeated = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=1,
            artifact_ids=frozenset(),
        )
    )
    retry = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=2,
            artifact_ids=frozenset(),
        )
    )

    assert first == repeated
    assert first != retry
    assert first.startswith("flakegraph-")
    assert len(first) == 27
    assert first.replace("-", "").isalnum()


def test_bounded_connected_components_preserve_singletons_and_transitive_links() -> None:
    """Match connected-component semantics without depending on edge input order."""

    vertices = ["d", "c", "b", "a", "single"]
    forward = _connected_component_rows(vertices, [("b", "c"), ("a", "b"), ("d", "c")])
    reverse = _connected_component_rows(vertices, [("d", "c"), ("a", "b"), ("b", "c")])

    expected = [
        ("a", "a"),
        ("b", "a"),
        ("c", "a"),
        ("d", "a"),
        ("single", "single"),
    ]
    assert forward == expected
    assert reverse == expected


def test_provider_partitions_reduce_stragglers_without_unbounded_task_counts() -> None:
    """Size network work more finely while bounding scheduler load by fleet size."""

    # With one thread per task, a partition holds one batch and four four-core
    # executors get at least two work units per slot.
    assert _adaptive_provider_partitions(10, 4, 4, provider_batch_size=1, threads_per_task=1) == 10
    assert (
        _adaptive_provider_partitions(14_000, 4, 4, provider_batch_size=20, threads_per_task=1)
        == 700
    )
    assert (
        _adaptive_provider_partitions(2_068, 4, 4, provider_batch_size=8, threads_per_task=1) == 259
    )
    assert (
        _adaptive_provider_partitions(698, 4, 4, provider_batch_size=2, threads_per_task=1) == 349
    )
    # Extremely large provider stages stop at 256 tasks per execution slot.
    assert (
        _adaptive_provider_partitions(1_000_000, 4, 4, provider_batch_size=1, threads_per_task=1)
        == 4_096
    )
    assert _adaptive_provider_partitions(0, 4, 4, provider_batch_size=1, threads_per_task=1) == 1


def test_provider_partitions_give_each_thread_a_batch() -> None:
    """Threads only help when a partition holds a batch for each of them."""

    # Four threads per task: one partition carries four batches, not one.
    assert (
        _adaptive_provider_partitions(14_000, 4, 4, provider_batch_size=20, threads_per_task=4)
        == 175
    )
    assert (
        _adaptive_provider_partitions(2_068, 4, 4, provider_batch_size=8, threads_per_task=4) == 65
    )
    # A small stage keeps two tasks per slot, so no slot waits for work.
    assert (
        _adaptive_provider_partitions(500, 4, 4, provider_batch_size=16, threads_per_task=4) == 32
    )
    # The per-slot task cap still binds at corpus scale.
    assert (
        _adaptive_provider_partitions(10_000_000, 4, 4, provider_batch_size=1, threads_per_task=4)
        == 4_096
    )


@pytest.mark.parametrize(
    ("concurrency", "instances", "cores", "threads"),
    [(64, 4, 4, 4), (60, 4, 4, 3), (16, 4, 4, 1), (2, 4, 4, 1), (8, 2, 1, 4)],
)
def test_provider_concurrency_is_shared_among_execution_slots(
    concurrency: int,
    instances: int,
    cores: int,
    threads: int,
) -> None:
    """Slots times threads never exceeds the configured total, and a slot runs at least one."""

    settings = _concurrency_settings(concurrency, instances=instances, cores=cores)

    assert _provider_threads_per_task(settings) == threads


def test_explicit_shuffle_partition_override_remains_authoritative() -> None:
    """Preserve measured operator tuning instead of replacing it from row counts."""

    assert _effective_shuffle_partitions(10_000_000, 4, 4, 73) == 73
    assert _effective_shuffle_partitions(10_000_000, 4, 4, 0) >= 16
    with pytest.raises(ValueError, match="must not be negative"):
        _effective_shuffle_partitions(10, 1, 1, -1)


@pytest.mark.parametrize(
    ("rows", "instances", "cores", "batch_size", "threads"),
    [(-1, 1, 1, 1, 1), (1, 0, 1, 1, 1), (1, 1, 0, 1, 1), (1, 1, 1, 0, 1), (1, 1, 1, 1, 0)],
)
def test_provider_partitions_reject_invalid_capacity(
    rows: int,
    instances: int,
    cores: int,
    batch_size: int,
    threads: int,
) -> None:
    """Reject invalid sizing inputs before they reach a Spark repartition call."""

    with pytest.raises(ValueError):
        _adaptive_provider_partitions(rows, instances, cores, batch_size, threads)


def test_executor_embedding_provider_is_reused_for_identical_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid rebuilding an adapter or reloading model weights for every partition."""

    settings = Settings.load(
        env={},
        overrides={
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
        },
    )
    built: list[object] = []

    def build(_settings: Settings) -> object:
        """Record factory calls and return a unique provider sentinel."""

        provider = object()
        built.append(provider)
        return provider

    _clear_executor_provider_caches()
    monkeypatch.setattr("kg_processor.factories.build_embedding_provider", build)
    try:
        first = _executor_embedding_provider(settings)
        second = _executor_embedding_provider(settings)
    finally:
        _clear_executor_provider_caches()

    assert first is second
    assert built == [first]


def test_spark_enrichment_provider_reuses_durable_cache_across_worker_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph"},
            "llm": {"provider": "fake", "model": "fake"},
            "cache": {"provider": "local", "path": tmp_path / "cache"},
            "distributed": {"artifact_uri": (tmp_path / "artifacts").as_uri()},
        },
    )
    built: list[CountingLlm] = []

    def build(_settings: Settings, _consumption: Any = None) -> CountingLlm:
        provider = CountingLlm()
        built.append(provider)
        return provider

    request = DescriptionMergeRequest(
        entity_name="Alice",
        entity_type="PERSON",
        descriptions=["Alice", "Alice is an engineer."],
        evidence=["Alice works at Acme."],
    )
    monkeypatch.setattr("kg_processor.factories.build_llm_provider", build)
    _clear_executor_provider_caches()
    try:
        first = _executor_enrichment_llm_provider(settings)
        first.merge_entity_description(request)
        _clear_executor_provider_caches()
        second = _executor_enrichment_llm_provider(settings)
        second.merge_entity_description(request)
    finally:
        _clear_executor_provider_caches()

    assert len(built) == 2
    assert sum(provider.description_calls for provider in built) == 1


class _InFlight:
    """Count provider calls in flight and answer early calls last.

    Every third call is slow, so later batches finish first and results arrive
    out of submission order whenever calls overlap.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.calls = 0
        self.current = 0
        self.peak = 0

    @contextmanager
    def call(self) -> Iterator[int]:
        with self._lock:
            self.calls += 1
            number = self.calls
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            sleep(0.03 if number % 3 == 1 else 0.005)
            yield number
        finally:
            with self._lock:
                self.current -= 1


class _InFlightLlm(FakeLlmProvider):
    """Answer like the fake LLM while counting overlapping calls."""

    def __init__(self, in_flight: _InFlight, fail_on_call: int | None) -> None:
        self.in_flight = in_flight
        self.fail_on_call = fail_on_call

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        with self.in_flight.call() as number:
            if number == self.fail_on_call:
                raise RuntimeError("provider unavailable")
            return super().complete_structured(request)


class _InFlightEmbeddings(HashEmbeddingProvider):
    """Embed like the hash provider while counting overlapping calls."""

    def __init__(self, in_flight: _InFlight) -> None:
        self.in_flight = in_flight

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        with self.in_flight.call():
            return super().embed(texts, options)


class _Struct(dict[str, Any]):
    """A mapping with the ``asDict`` a Spark struct value offers."""

    def asDict(self, recursive: bool = False) -> dict[str, Any]:  # noqa: N802 - Spark's name
        del recursive
        return dict(self)


def _description_rows(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            node_id=f"node-{index:04d}",
            name=f"Entity {index}",
            primary_type="CONCEPT",
            descriptions=[f"Entity {index}.", f"Entity {index}, described at more length."],
            evidence_quotes=[f"Entity {index} appears here."],
        )
        for index in range(count)
    ]


def _community_rows(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=f"community-{index:04d}",
            title=f"Community {index}",
            members=[f"Member {index}a", f"Member {index}b"],
            relations=[f"Member {index}a RELATED_TO Member {index}b"],
            evidence_quotes=[f"Member {index}a works with Member {index}b."],
        )
        for index in range(count)
    ]


def _mention(mention_id: str, name: str) -> _Struct:
    return _Struct(
        EntityMention(
            id=mention_id,
            name=name,
            type="CONCEPT",
            description=f"A mention of {name}.",
            source_chunk_id="chunk-1",
            quote=name,
        ).model_dump(mode="json")
    )


def _candidate_rows(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            left_id=f"left-{index:04d}",
            right_id=f"right-{index:04d}",
            lexical_score=0.85,
            embedding_score=0.9,
            left_mention=_mention(f"left-{index:04d}", f"Entity {index}"),
            right_mention=_mention(f"right-{index:04d}", f"Entity {index}s"),
        )
        for index in range(count)
    ]


def _embedding_rows(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(record_id=f"record-{index:04d}", embedding_text=f"Text number {index}.")
        for index in range(count)
    ]


# Each phase's partition function, its rows, and its batch size under
# ``_concurrency_settings``. Every case runs sixteen batches.
_PHASES: dict[str, tuple[Any, Any, int]] = {
    "descriptions": (_merge_description_partition, _description_rows, DESCRIPTION_BATCH_SIZE),
    "communities": (_summarize_community_partition, _community_rows, COMMUNITY_BATCH_SIZE),
    "adjudication": (_adjudicate_partition, _candidate_rows, 4),
    "embeddings": (_embed_partition, _embedding_rows, 4),
}


def _concurrency_settings(concurrency: int, *, instances: int = 2, cores: int = 2) -> Settings:
    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8, "batch_size": 4},
            "cache": {"provider": "none"},
            "graph": {
                "finalization_provider_concurrency": concurrency,
                "resolution_adjudication_batch_size": 4,
            },
            "distributed": {
                "spark_executor_instances": instances,
                "spark_executor_cores": cores,
            },
        },
    )


def _run_partition(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    concurrency: int,
    *,
    fail_on_call: int | None = None,
    consumed: list[Any] | None = None,
) -> tuple[list[Any], _InFlight]:
    """Run one phase's partition function in-process against counting providers.

    Rows land in ``consumed`` as the partition yields them, so a caller can
    inspect what came out before a failure.
    """

    partition, rows, batch_size = _PHASES[phase]
    in_flight = _InFlight()
    monkeypatch.setattr(
        "kg_processor.factories.build_llm_provider",
        lambda _settings, _collector=None: _InFlightLlm(in_flight, fail_on_call),
    )
    monkeypatch.setattr(
        "kg_processor.factories.build_embedding_provider",
        lambda _settings: _InFlightEmbeddings(in_flight),
    )
    monkeypatch.setattr(
        "kg_processor.application.spark_finalization._executor_consumption_collector",
        lambda settings: ConsumptionCollector(settings.job.graph_id, job_id=settings.job.job_id),
    )
    payload = _concurrency_settings(concurrency).model_dump(mode="json")
    output = consumed if consumed is not None else []
    _clear_executor_provider_caches()
    try:
        output.extend(partition(iter(rows(16 * batch_size)), payload))
    finally:
        _clear_executor_provider_caches()
    return output, in_flight


@pytest.mark.parametrize("phase", sorted(_PHASES))
def test_concurrent_provider_batches_keep_the_sequential_output_and_order(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Four threads per task produce exactly the rows one thread does, in the same order.

    Four execution slots share a concurrency of sixteen: four calls per task.
    Every third call is slow, so batches complete out of order, and the
    counting provider shows the pool reached its limit without passing it.
    """

    sequential, one_at_a_time = _run_partition(monkeypatch, phase, concurrency=4)
    concurrent, overlapping = _run_partition(monkeypatch, phase, concurrency=16)

    assert concurrent == sequential
    assert len(sequential) == 16 * _PHASES[phase][2]
    assert one_at_a_time.peak == 1
    assert overlapping.peak == 4
    assert overlapping.calls == one_at_a_time.calls == 16


def test_a_failed_provider_batch_fails_the_partition_without_reordering_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing call is the task's error, and what came out before it is an in-order prefix.

    Spark discards a failed task's rows and reruns its partition, so what the
    partition owes Spark is the error and no row out of place. Batches queued
    behind the failure are cancelled rather than sent.
    """

    expected, _ = _run_partition(monkeypatch, "descriptions", concurrency=16)
    consumed: list[Any] = []
    with pytest.raises(RuntimeError, match="provider unavailable"):
        _run_partition(
            monkeypatch, "descriptions", concurrency=16, fail_on_call=6, consumed=consumed
        )

    assert consumed == expected[: len(consumed)]
    assert len(consumed) < len(expected)
    retried, _ = _run_partition(monkeypatch, "descriptions", concurrency=16)
    assert retried == expected


def test_each_stage_is_read_from_the_runs_that_hold_it() -> None:
    """A revision that extracts documents again prepared none of its own.

    Its extracted rows are under its own prefix and the base run's, its
    prepared rows only under the base run's; listing a prefix a run never
    wrote fails the read outright.
    """

    request = SparkFinalizationRequest(
        run_id="revision",
        graph_id="graph",
        attempt=1,
        artifact_ids=frozenset({"a", "b"}),
        prepared_run_ids=frozenset({"base"}),
        extracted_run_ids=frozenset({"base", "revision"}),
    )

    assert request.stage_run_ids(request.prepared_run_ids) == ["base"]
    assert request.stage_run_ids(request.extracted_run_ids) == ["revision", "base"]
    # A run with nothing linked yet reads only its own prefix.
    assert request.stage_run_ids(frozenset()) == ["revision"]
