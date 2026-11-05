"""Contracts for Ollama's native structured chat transport."""

from __future__ import annotations

import httpx
import pytest
from http_fakes import ScriptedClient

from kg_processor.adapters.llm.ollama import OllamaLlmProvider
from kg_processor.ports.llm import StructuredCompletionRequest


def test_ollama_uses_native_schema_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reserve output tokens for structured content rather than reasoning traces."""

    client = ScriptedClient(
        lambda *_: httpx.Response(
            200, json={"message": {"role": "assistant", "content": '{"ok":true}'}}
        )
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    provider = OllamaLlmProvider(
        endpoint="http://localhost:11434/v1",
        model="qwen3.8:27b-q4_K_M",
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="ollama_contract",
            model="qwen3.8:27b-q4_K_M",
            system="Return JSON.",
            user="Confirm.",
            json_schema=schema,
            max_tokens=128,
        )
    )

    url, _, payload = client.requests[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == schema
    assert payload["options"] == {
        "temperature": 0.0,
        "num_predict": 128,
        "num_ctx": 32768,
    }
    assert result.payload == {"ok": True}
    assert result.provider_metadata["provider"] == "ollama"
