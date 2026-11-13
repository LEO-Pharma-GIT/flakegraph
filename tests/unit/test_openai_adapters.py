from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from http_fakes import Handler, ScriptedClient

from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from kg_processor.adapters.llm.azure_openai import AzureOpenAILlmProvider
from kg_processor.adapters.llm.ollama import OllamaLlmProvider
from kg_processor.adapters.llm.openai_common import ChatCompletion, coerce_string_list
from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.adapters.llm.vllm_local import VllmLocalLlmProvider
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    DescriptionMergeRequest,
    StructuredCompletionRequest,
)


def test_openai_string_lists_drop_null_and_non_string_values() -> None:
    assert coerce_string_list([None, " Valid question? ", 7, ""]) == ["Valid question?"]


_ENTITIES = {
    "entities": [
        {
            "name": "Alice Smith",
            "type": "PERSON",
            "description": "Alice Smith is present.",
            "source_chunk_id": "chunk_1",
            "confidence": 1,
            "aliases": [],
        }
    ],
    "relations": [],
}
_MERGED = {"description": "Alice Smith works at Acme Corp."}


def _chat(content: object) -> httpx.Response:
    """Return one chat completion whose message is ``content``, serialized unless a string."""

    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def _embedding() -> httpx.Response:
    return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]})


def _embeddings_rejecting(condition: Callable[[dict[str, Any]], bool], error: object) -> Handler:
    """Embed every batch except the ones ``condition`` picks, which get a 400 with ``error``."""

    def handler(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        return httpx.Response(400, json=error) if condition(payload) else _embedding()

    return handler


def _system_prompt(payload: dict[str, Any]) -> str:
    return str(payload["messages"][0]["content"])


def _openai(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
    """Answer the compatible adapter's chat and embedding calls."""

    if url.endswith("/embeddings"):
        return _embedding()
    assert url.endswith("/chat/completions"), f"Unexpected URL: {url}"
    assert payload["max_tokens"] == 8192
    if "merge observed descriptions" in _system_prompt(payload):
        return _chat(_MERGED)
    if payload["model"] == "community":
        return _chat(
            {
                "title": "Acme",
                "summary": "Acme community",
                "rating": 7,
                "rating_explanation": "Important because Alice is connected to Acme.",
                "findings": [{"summary": "Finding", "explanation": "Because"}],
                "suggested_questions": ["How is Alice connected to Acme?"],
            }
        )
    return _chat(_ENTITIES)


def _vllm(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
    assert url == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in headers
    assert payload["model"] == "qwen2.5"
    assert payload["max_tokens"] == 16384
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    return _chat(_ENTITIES)


def _azure(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
    """Answer deployment-addressed chat and embedding calls."""

    assert "api-version=2025-01-01-preview" in url
    assert headers["api-key"] == "secret"
    if url.endswith("/embeddings?api-version=2025-01-01-preview"):
        return _embedding()
    assert url.endswith("/chat/completions?api-version=2025-01-01-preview"), url
    assert payload["max_tokens"] == 8192
    assert payload["reasoning_effort"] == "none"
    system = _system_prompt(payload)
    if "merge observed descriptions" in system:
        return _chat(_MERGED)
    if "knowledge-graph community" in system:
        return _chat(
            {
                "title": "Acme",
                "summary": "Acme community",
                "rating": "6.5",
                "rating_explanation": "Important because the relation is high weight.",
                "findings": [{"summary": "Finding", "explanation": "Because"}],
                "suggested_questions": ["What does Acme connect to?"],
            }
        )
    return _chat(_ENTITIES)


def _azure_negotiating(
    url: str, headers: dict[str, str], payload: dict[str, Any]
) -> httpx.Response:
    """Return the structured capability errors newer models emit until the request fits."""

    for parameter, code in (
        ("max_tokens", "unsupported_parameter"),
        ("temperature", "unsupported_value"),
        ("reasoning_effort", "unsupported_parameter"),
    ):
        if parameter in payload:
            return httpx.Response(400, json={"error": {"code": code, "param": parameter}})
    return _chat('{"ok":true}')


def test_openai_compatible_llm_runs_structured_and_enrichment_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ScriptedClient(_openai)
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="community")
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="graph",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=16384,
        )
    )
    summary = provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )
    description = provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert result.provider_metadata["provider"] == "openai_compatible"
    assert result.provider_metadata["task_name"] == "entities"
    assert summary.title == "Acme"
    assert summary.rating_explanation == "Important because Alice is connected to Acme."
    assert summary.suggested_questions == ["How is Alice connected to Acme?"]
    assert summary.provider_metadata["provider"] == "openai_compatible"
    assert summary.provider_metadata["model"] == "community"
    assert summary.provider_metadata["prompt_name"] == "community_report"
    assert description.description == "Alice Smith works at Acme Corp."
    assert description.provider_metadata["provider"] == "openai_compatible"
    assert description.provider_metadata["model"] == "community"
    assert description.provider_metadata["prompt_name"] == "entity_description_merge"
    assert client.opened == 1


def test_openai_compatible_llm_requires_explicit_default_model() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="")


def test_openai_structured_completion_retries_malformed_transport_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure malformed transport JSON triggers exactly one schema-preserving regeneration.

    Repair attempts must appear in provider metadata.
    """

    def handler(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        assert payload["response_format"]["type"] == "json_schema"
        return _chat("{broken" if len(client.requests) == 1 else '{"records": []}')

    client = ScriptedClient(handler)
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", "model")

    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="records",
            model="model",
            system="Return records.",
            user="Input",
            json_schema={
                "type": "object",
                "properties": {"records": {"type": "array", "items": {"type": "string"}}},
                "required": ["records"],
            },
        )
    )

    assert result.payload == {"records": []}
    assert result.provider_metadata["format_repair_attempts"] == 1
    assert len(client.requests) == 2


def test_openai_structured_completion_retries_empty_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry an empty choice message that fails before JSON parsing begins."""

    responses: list[str | ValueError] = [ValueError("empty message"), '{"records":[]}']
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", "model")

    def chat_content(*_args: object, **_kwargs: object) -> ChatCompletion:
        """Return one transport failure followed by a complete JSON response."""

        response = responses.pop(0)
        if isinstance(response, ValueError):
            raise response
        return ChatCompletion(content=response)

    monkeypatch.setattr(provider, "_chat_content", chat_content)
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="records",
            model="model",
            system="Return records.",
            user="Input",
            json_schema={"type": "object", "properties": {"records": {"type": "array"}}},
        )
    )

    assert result.payload == {"records": []}
    assert result.provider_metadata["format_repair_attempts"] == 1


def test_vllm_local_llm_uses_openai_compatible_chat_without_required_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "Client", ScriptedClient(_vllm).open)
    provider = VllmLocalLlmProvider("http://localhost:8000/v1", model="qwen2.5")
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="qwen2.5",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=16384,
        )
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert result.provider_metadata["provider"] == "vllm_local"
    assert provider.capabilities().supports_seed is True
    assert provider.capabilities().max_output_tokens == 16384


def test_openai_compatible_embeddings_validate_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", ScriptedClient(_openai).open)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")

    vectors = provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


def test_azure_openai_llm_uses_deployment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", ScriptedClient(_azure).open)
    provider = AzureOpenAILlmProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
        "gpt-4.1-mini",
    )
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="gpt-4.1-mini",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=8192,
        )
    )
    description = provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )
    summary = provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert description.description == "Alice Smith works at Acme Corp."
    assert description.provider_metadata["provider"] == "azure_openai"
    assert description.provider_metadata["model"] == "gpt-4.1-mini"
    assert description.provider_metadata["prompt_name"] == "entity_description_merge"
    assert summary.title == "Acme"
    assert summary.rating == 6.5
    assert summary.rating_explanation == "Important because the relation is high weight."
    assert summary.findings == [("Finding", "Because")]
    assert summary.suggested_questions == ["What does Acme connect to?"]
    assert summary.provider_metadata["provider"] == "azure_openai"
    assert summary.provider_metadata["model"] == "gpt-4.1-mini"
    assert summary.provider_metadata["prompt_name"] == "community_report"


def test_azure_openai_advertises_dense_relation_output_budget() -> None:
    """Allow complete relation objects that exceed an 8K output ceiling."""

    provider = AzureOpenAILlmProvider(
        "https://example.openai.azure.com",
        "secret",
        "2025-01-01-preview",
        "deployment",
    )

    assert provider.capabilities().max_output_tokens == 16384


def test_azure_openai_negotiates_newer_chat_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache model capability errors so only the first request requires retries."""

    client = ScriptedClient(_azure_negotiating)
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = AzureOpenAILlmProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
        "new-model",
    )
    request = StructuredCompletionRequest(
        task_name="new_model",
        model="new-model",
        system="Return JSON.",
        user="Confirm.",
        json_schema={"type": "object"},
        max_tokens=128,
    )

    first = provider.complete_structured(request)
    second = provider.complete_structured(request)

    assert first.payload == {"ok": True}
    assert second.payload == {"ok": True}
    payloads = client.payloads
    assert len(payloads) == 5
    assert "max_tokens" in payloads[0]
    assert "max_completion_tokens" in payloads[1]
    assert "temperature" in payloads[1]
    assert "temperature" not in payloads[2]
    assert "reasoning_effort" in payloads[2]
    assert "reasoning_effort" not in payloads[3]
    assert "reasoning_effort" not in payloads[4]


def test_azure_openai_embeddings_use_deployment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "Client", ScriptedClient(_azure).open)
    provider = AzureOpenAIEmbeddingProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
    )

    vectors = provider.embed(["Alice"], EmbedOptions(model="text-embedding-3-small", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


@pytest.mark.parametrize(
    ("build", "model"),
    [
        (
            lambda: OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret"),
            "text-embedding-3-small",
        ),
        (
            lambda: AzureOpenAIEmbeddingProvider(
                "https://example.test", "secret", "2025-01-01-preview"
            ),
            "ada-deployment",
        ),
    ],
)
def test_embedding_adapters_retry_without_unsupported_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[], OpenAICompatibleEmbeddingProvider],
    model: str,
) -> None:
    # The rejection has to name the parameter before the adapter may drop it.
    client = ScriptedClient(
        _embeddings_rejecting(
            lambda payload: "dimensions" in payload, {"error": "unsupported dimensions"}
        )
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = build()

    vectors = provider.embed(["Alice"], EmbedOptions(model=model, dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]
    assert "dimensions" in client.payloads[0]
    assert "dimensions" not in client.payloads[1]
    provider.close()


def test_embedding_adapters_keep_the_requested_width_after_an_unrelated_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rejected batch must not narrow every later vector in the process."""

    client = ScriptedClient(
        _embeddings_rejecting(
            lambda payload: len(payload["input"]) > 1,
            {
                "error": {
                    "code": "context_length_exceeded",
                    "message": "This model's maximum context length is 8192 tokens",
                    "param": "input",
                }
            },
        )
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")
    options = EmbedOptions(model="text-embedding-3-small", dimension=3)

    with pytest.raises(httpx.HTTPStatusError):
        provider.embed(["Alice", "Acme"], options)

    assert provider.embed(["Alice"], options) == [[0.1, 0.2, 0.3]]
    assert all("dimensions" in payload for payload in client.payloads)
    provider.close()


def test_generic_embedding_omits_dimensions_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ScriptedClient(lambda *_: _embedding())
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")

    vectors = provider.embed(["Alice"], EmbedOptions(model="custom-model", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]
    assert client.payloads == [{"model": "custom-model", "input": ["Alice"]}]
    provider.close()


@pytest.mark.parametrize("name", ["compatible", "azure", "vllm", "ollama"])
def test_output_ceiling_applies_to_enrichment_and_structured_calls(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    response = {
        "description": "Alice works at Acme.",
        "title": "Acme",
        "summary": "Acme community",
        "rating": 1,
        "rating_explanation": "Fixture",
        "findings": [],
        "suggested_questions": [],
    }

    def answer(url: str, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        limit = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if name == "ollama":
            limit = payload["options"]["num_predict"]
        assert limit == 512
        content = json.dumps(response)
        return httpx.Response(
            200,
            json={
                "message": {"content": content},
                "choices": [{"message": {"content": content}}],
                "done": True,
            },
        )

    client = ScriptedClient(answer)
    monkeypatch.setattr(httpx, "Client", client.open)
    provider: OpenAICompatibleLlmProvider
    if name == "azure":
        provider = AzureOpenAILlmProvider(
            "https://example.test", "fixture", "2024-12-01-preview", "m", max_output_tokens=512
        )
    elif name == "vllm":
        provider = VllmLocalLlmProvider("http://example.test/v1", model="m", max_output_tokens=512)
    elif name == "ollama":
        provider = OllamaLlmProvider("http://example.test", "m", max_output_tokens=512)
    else:
        provider = OpenAICompatibleLlmProvider(
            "http://example.test/v1", "fixture", model="m", max_output_tokens=512
        )
    provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice",
            entity_type="PERSON",
            descriptions=["Alice is present", "Alice works at Acme"],
            evidence=[],
        )
    )
    provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Alice"], relations=[])
    )
    provider.complete_structured(
        StructuredCompletionRequest(
            task_name="fixture",
            model="m",
            system="Fixture",
            user="Fixture",
            json_schema={"type": "object"},
            max_tokens=8192,
        )
    )
    assert len(client.requests) == 3
