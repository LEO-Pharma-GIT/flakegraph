# SPDX-License-Identifier: Apache-2.0
"""Azure OpenAI/Foundry chat adapter with deployment-specific transport."""

from __future__ import annotations

from functools import partial

import httpx

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    chat_completion_from_payload,
    send_with_http_retry,
)
from kg_processor.adapters.llm.openai_compatible import (
    DEFAULT_CHAT_MAX_TOKENS,
    OpenAICompatibleLlmProvider,
)
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS, LlmCapabilities

_STRUCTURED_CHAT_MAX_TOKENS = 16384
_BAD_REQUEST_STATUS = 400


class AzureOpenAILlmProvider(OpenAICompatibleLlmProvider):
    """Reuse shared chat orchestration with Azure deployment-specific HTTP details."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        default_deployment: str,
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: int | None = None,
    ) -> None:
        """Configure Azure URL negotiation and the default enrichment deployment."""

        super().__init__(
            endpoint=endpoint,
            api_key=api_key,
            model=default_deployment,
            provider_name="azure_openai",
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )
        self.api_version = api_version
        self.default_deployment = default_deployment
        self._uses_max_completion_tokens = False
        self._omits_temperature = False
        self._omits_reasoning_effort = False

    def capabilities(self) -> LlmCapabilities:
        """Advertise Azure's larger structured-response token budget."""

        return LlmCapabilities(
            supports_seed=False,
            max_output_tokens=self._max_output_tokens or _STRUCTURED_CHAT_MAX_TOKENS,
        )

    def _chat_content(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        response_format: dict[str, object] | None = None,
        seed: int | None = None,
    ) -> ChatCompletion:
        """Submit one Azure request while negotiating deployment capabilities."""

        url = (
            f"{self.endpoint}/openai/deployments/{model}/chat/completions"
            f"?api-version={self.api_version}"
        )
        response: httpx.Response | None = None
        for _attempt in range(4):
            token_parameter = (
                "max_completion_tokens" if self._uses_max_completion_tokens else "max_tokens"
            )
            request_payload: dict[str, object] = {
                token_parameter: max_tokens,
                "response_format": response_format or {"type": "json_object"},
                "messages": messages,
            }
            if not self._omits_temperature:
                request_payload["temperature"] = temperature
            if not self._omits_reasoning_effort:
                request_payload["reasoning_effort"] = "none"
            if seed is not None:
                request_payload["seed"] = seed
            response = send_with_http_retry(
                partial(
                    _post_chat,
                    self._client,
                    url,
                    str(self.api_key),
                    request_payload,
                    timeout_seconds,
                )
            )
            if response.is_success:
                break
            unsupported = _unsupported_parameter(response)
            if unsupported == "max_tokens" and "max_tokens" in request_payload:
                self._uses_max_completion_tokens = True
                continue
            if unsupported == "temperature" and "temperature" in request_payload:
                self._omits_temperature = True
                continue
            if unsupported == "reasoning_effort" and "reasoning_effort" in request_payload:
                self._omits_reasoning_effort = True
                continue
            response.raise_for_status()
        if response is None:
            raise RuntimeError("Azure OpenAI request was not attempted")
        response.raise_for_status()
        return chat_completion_from_payload(response.json())


def _post_chat(
    client: httpx.Client,
    url: str,
    api_key: str,
    request_payload: dict[str, object],
    timeout_seconds: float,
) -> httpx.Response:
    """Submit one Azure request attempt through a retained connection pool."""

    return client.post(
        url, headers={"api-key": api_key}, json=request_payload, timeout=timeout_seconds
    )


def _unsupported_parameter(response: httpx.Response) -> str | None:
    """Return the parameter named by a safely parsed Azure capability error."""

    if response.status_code != _BAD_REQUEST_STATUS:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict) or error.get("code") not in {
        "unsupported_parameter",
        "unsupported_value",
        "invalid_value",
    }:
        return None
    parameter = error.get("param")
    return parameter if isinstance(parameter, str) else None
