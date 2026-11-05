"""Pricing and aggregation of recorded consumption."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kg_processor.application.consumption import ConsumptionCollector, locality_for
from kg_processor.application.metered_llm import MeteredLlmProvider
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import OcrSettings, Settings
from kg_processor.domain.consumption import (
    ConsumptionEvent,
    Locality,
    Rate,
    RateCard,
    TokenUsage,
    group_by,
    price_event,
    summarize,
)
from kg_processor.domain.documents import InputFile, ParsedDocument, ParsedPage
from kg_processor.ports.ocr import OcrOptions


def _card() -> RateCard:
    return RateCard(
        rates={
            "snowflake_cortex:llama3.3-70b": Rate(
                prompt_usd_per_million=1.0, completion_usd_per_million=3.0
            ),
            "snowflake_cortex:*": Rate(usd_per_page=0.01),
            "openai:gpt-5.6": Rate(prompt_usd_per_million=2.0, completion_usd_per_million=8.0),
        },
        local_reference="openai:gpt-5.6",
    )


def _event(**overrides: object) -> ConsumptionEvent:
    defaults: dict[str, object] = {
        "graph_id": "g",
        "stage": "extraction",
        "operation": "completion",
        "provider": "snowflake_cortex",
        "model": "llama3.3-70b",
        "usage": TokenUsage(
            prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000
        ),
    }
    return ConsumptionEvent.model_validate({**defaults, **overrides})


def test_token_cost_uses_separate_prompt_and_completion_rates() -> None:
    """Completion tokens usually cost several times what prompt tokens do."""

    cost, priced = price_event(_event(), _card())

    assert priced
    assert cost == 4.0


def test_pages_are_priced_for_document_parsing() -> None:
    cost, priced = price_event(
        _event(
            operation="parse_document",
            model="ai_parse_document",
            usage=TokenUsage(),
            pages=250,
        ),
        _card(),
    )

    assert priced
    assert round(cost, 4) == 2.5


def test_a_model_without_its_own_rate_falls_back_to_the_provider() -> None:
    """A new model must be priced approximately, never silently as free."""

    _, priced = price_event(_event(model="some-new-model", usage=TokenUsage(), pages=10), _card())

    assert priced


def test_an_unpriced_call_is_reported_rather_than_hidden() -> None:
    """A total that omits calls it could not price reads authoritative but is wrong."""

    totals = summarize([_event(provider="mystery", model="unknown")], _card())

    assert totals.billed_usd == 0.0
    assert totals.unpriced_calls == 1
    assert not totals.is_complete


def test_local_work_accrues_avoided_cost_not_billed_cost() -> None:
    """Local runs bill nothing; their value is what was not spent elsewhere."""

    totals = summarize(
        [_event(provider="ollama", model="qwen3.8", locality=Locality.LOCAL)],
        _card(),
    )

    assert totals.billed_usd == 0.0
    # Priced at the reference hosted model: 1M prompt at $2 + 1M completion at $8.
    assert totals.avoided_usd == 10.0
    assert totals.is_complete


def test_avoided_cost_needs_a_stated_alternative() -> None:
    """Without a reference model, a saving would be unfalsifiable."""

    card = _card().model_copy(update={"local_reference": None})

    totals = summarize([_event(provider="ollama", locality=Locality.LOCAL)], card)

    assert totals.avoided_usd == 0.0
    assert totals.unpriced_calls == 1


def test_grouping_separates_spend_by_attribute() -> None:
    totals = group_by(
        [
            _event(stage="ocr", operation="parse_document", usage=TokenUsage(), pages=100),
            _event(stage="extraction"),
        ],
        "stage",
        _card(),
    )

    assert set(totals) == {"ocr", "extraction"}
    assert totals["ocr"].pages == 100
    assert totals["extraction"].total_tokens == 2_000_000


def test_a_rate_that_ignores_pages_does_not_price_a_parse() -> None:
    """A card pricing tokens but not pages would report parsing as free.

    That is the same silent zero that reporting unpriced calls exists to prevent,
    so coverage is checked per dimension rather than per rate.
    """

    card = RateCard(rates={"openai:gpt-5.6": Rate(prompt_usd_per_million=2.0)})

    cost, priced = price_event(
        _event(provider="openai", model="gpt-5.6", usage=TokenUsage(), pages=1_000),
        card,
    )

    assert cost == 0.0
    assert not priced


def test_local_ocr_saving_is_flagged_when_no_page_reference_exists() -> None:
    """ "We saved nothing" and "we cannot say" must not look identical."""

    card = RateCard(
        rates={"openai:gpt-5.6": Rate(prompt_usd_per_million=2.0)},
        local_reference="openai:gpt-5.6",
    )

    totals = summarize(
        [
            _event(
                stage="ocr",
                operation="parse_document",
                provider="builtin_text",
                model="builtin_text",
                locality=Locality.LOCAL,
                usage=TokenUsage(),
                pages=500,
            )
        ],
        card,
    )

    assert totals.avoided_usd == 0.0
    assert totals.unpriced_calls == 1
    assert not totals.is_complete


def test_an_adapter_that_forgets_usage_falls_back_to_provider_metadata() -> None:
    """Zero tokens means "not reported", not "free".

    An adapter can build its result without usage while still carrying the
    provider's own figures in metadata. Read as reported, that call would count
    no tokens and therefore no cost.
    """

    collector = ConsumptionCollector("g")
    collector.record_completion(
        stage="community_report",
        provider="snowflake_cortex",
        model="llama3.3-70b",
        usage=TokenUsage(),
        metadata={"usage": {"prompt_tokens": 900, "completion_tokens": 100}},
    )

    recorded = collector.events[0].usage

    assert recorded.prompt_tokens == 900
    assert recorded.completion_tokens == 100


def test_a_reported_usage_is_never_overridden_by_metadata() -> None:
    """The result's own figures win when it actually reported them."""

    collector = ConsumptionCollector("g")
    collector.record_completion(
        stage="extraction",
        provider="snowflake_cortex",
        model="llama3.3-70b",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        metadata={"usage": {"prompt_tokens": 999}},
    )

    assert collector.events[0].usage.prompt_tokens == 10


def test_a_window_that_exhausts_its_repair_budget_reports_what_it_burned() -> None:
    """Every attempt of a failed structured call was billed.

    A window that never produces usable output still consumed six completions.
    Recording nothing for it reports an expensive failure as free, which is the
    one direction a spend total must never be wrong in.
    """

    collector = ConsumptionCollector(graph_id="graph-1")
    error = ValueError("could not parse the reply after every attempt")
    error.usage = TokenUsage(prompt_tokens=4_000, completion_tokens=800)  # type: ignore[attr-defined]
    provider = MeteredLlmProvider(
        cast(Any, _RaisingProvider(error)),
        collector,
        provider="openai_compatible",
        model="test-model",
    )

    with pytest.raises(ValueError):
        provider.complete_structured(
            cast(Any, SimpleNamespace(task_name="entity_extraction", model=None))
        )

    events = list(collector.events)
    assert len(events) == 1, "a billed failure must be recorded exactly once"
    assert events[0].usage.total_tokens == 4_800
    assert events[0].stage == "entity_extraction"


def test_a_failure_that_was_never_billed_records_nothing() -> None:
    """A transport error before any tokens were consumed is not a cost."""

    collector = ConsumptionCollector(graph_id="graph-1")
    provider = MeteredLlmProvider(
        cast(Any, _RaisingProvider(OSError("connection refused"))),
        collector,
        provider="openai_compatible",
        model="test-model",
    )

    with pytest.raises(OSError):
        provider.complete_structured(cast(Any, SimpleNamespace(task_name="ocr", model=None)))

    assert list(collector.events) == []


class _RaisingProvider:
    """Fail every structured call with one prepared error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def capabilities(self) -> object:
        raise NotImplementedError

    def complete_structured(self, request: object) -> object:
        raise self.error


def _shipped_card() -> RateCard:
    return Settings.load(Path("configs/app-defaults.yaml")).consumption.rate_card()


@pytest.mark.parametrize(
    "model",
    [
        "mistral-large2",
        "claude-opus-5",
        "claude-sonnet-5",
        "openai-gpt-5.6-sol",
        "llama3.3-70b",
        "snowflake-arctic-embed-m-v1.5",
        "snowflake-arctic-embed-l-v2.0",
    ],
)
def test_the_shipped_card_prices_the_models_this_deployment_can_select(model: str) -> None:
    """Every model reachable from the UI must resolve to a rate.

    The Cortex model is free text in the app, so a gap here does not fail — it
    reports a real call as costing nothing, which is the failure this whole
    module exists to prevent.
    """

    assert _shipped_card().rate_for("snowflake_cortex", model) is not None


def test_document_parsing_is_priced_per_mode() -> None:
    """AI_PARSE_DOCUMENT bills Layout at over five times the OCR rate.

    One rate for both modes would misreport whichever mode was not chosen, so
    the modes are priced separately and the pipeline records which one ran.
    """

    card = _shipped_card()
    layout = card.rate_for("snowflake_cortex", "snowflake_cortex-layout")
    ocr = card.rate_for("snowflake_cortex", "snowflake_cortex-ocr")
    assert layout is not None and ocr is not None
    assert layout.credits_per_page > ocr.credits_per_page * 5


def test_a_fallback_parse_is_billed_to_the_provider_that_ran() -> None:
    """The routing policy is not a provider; booking it as one hid hosted pages as local."""

    settings = Settings.load(
        overrides={
            "ocr": {
                "provider": "fallback",
                "fallback_primary_provider": "builtin_text",
                "fallback_secondary_provider": "mineru_api",
            },
        }
    )
    file = InputFile(
        id="f",
        path=Path("a.pdf"),
        source_uri="a.pdf",
        checksum="c",
        mime_type="application/pdf",
        size_bytes=1,
    )
    parsed = ParsedDocument(
        file_id="f",
        checksum="c",
        source_uri="a.pdf",
        mime_type="application/pdf",
        pages=[ParsedPage(page_number=1, markdown="x", raw_text="x")],
        provider_metadata={"provider": "mineru_api"},
    )
    ocr = SimpleNamespace(parse=lambda _file, _options: parsed)
    stub = cast(Any, SimpleNamespace())
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=stub,
        ocr=cast(Any, ocr),
        llm=stub,
        embeddings=stub,
        writer=stub,
    )

    pipeline._parse_file(file, OcrOptions())

    (event,) = pipeline.consumption.events
    assert (event.provider, event.model, event.locality) == (
        "mineru_api",
        "mineru_api",
        Locality.HOSTED,
    )


def test_the_ocr_settings_carry_the_parse_mode_into_billing() -> None:
    """The mode has to reach the rate card, or pricing it separately is moot."""

    assert OcrSettings(provider="snowflake_cortex").consumption_model() == "snowflake_cortex-ocr"
    assert (
        OcrSettings(provider="snowflake_cortex", snowflake_parse_mode="LAYOUT").consumption_model()
        == "snowflake_cortex-layout"
    )
    assert OcrSettings(provider="tesseract_internal").consumption_model() == "tesseract_internal"


def test_local_runs_price_against_a_stated_hosted_alternative() -> None:
    """Avoided cost is only meaningful next to the alternative it avoided.

    The shipped card names the alternative but leaves the credit price at zero,
    because that figure comes from the operator's Snowflake contract; the
    saving is reported once the operator supplies it.
    """

    shipped = _shipped_card()
    assert shipped.local_reference
    assert shipped.usd_per_credit == 0.0

    card = shipped.model_copy(update={"usd_per_credit": 2.0})
    totals = summarize([_event(provider="vllm_local", locality=Locality.LOCAL)], card)
    assert totals.billed_usd == 0.0
    assert totals.avoided_usd > 0.0
    assert totals.unpriced_calls == 0


@pytest.mark.parametrize(
    ("provider", "locality"),
    [
        ("tesseract_internal", Locality.LOCAL),
        ("mineru_internal", Locality.LOCAL),
        ("vllm_local", Locality.LOCAL),
        ("sentence_transformers", Locality.LOCAL),
        ("mineru_api", Locality.HOSTED),
        ("snowflake_cortex", Locality.HOSTED),
        ("azure_openai", Locality.HOSTED),
    ],
)
def test_a_provider_is_billed_where_the_registry_says_it_runs(
    provider: str, locality: Locality
) -> None:
    """The names that settings can hold are the ones that must classify right.

    A second list of local providers could spell one under an alias no settings
    object ever carries, and bill it as hosted.
    """

    assert locality_for(provider) == locality


def test_bounded_event_detail_preserves_complete_consumption_aggregates() -> None:
    """A small manifest can retain full totals without embedding the audit ledger."""

    collector = ConsumptionCollector("graph", job_id="run", run_id="run")
    for _ in range(125):
        collector.record_completion(
            stage="extraction",
            provider="openai",
            model="gpt-5.6",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
        )
    report = collector.report(_card(), event_limit=100)
    events, totals, by_stage = report["events"], report["totals"], report["by_stage"]
    assert isinstance(events, list)
    assert isinstance(totals, dict)
    assert isinstance(by_stage, dict)
    assert report["event_count"] == 125
    assert report["events_truncated"] is True
    assert len(events) == 100
    assert totals["calls"] == 125
    assert totals["total_tokens"] == 15_000
    assert by_stage["extraction"] == totals
    assert totals["billed_usd"] == pytest.approx(0.045)
    full_events = collector.report(_card())["events"]
    assert isinstance(full_events, list)
    assert len(full_events) == 125
