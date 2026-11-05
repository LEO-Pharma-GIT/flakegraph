# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter implemented through the OpenAI-compatible chat surface."""

from __future__ import annotations

from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS, LlmCapabilities

_VLLM_MAX_OUTPUT_TOKENS = 16384


class VllmLocalLlmProvider(OpenAICompatibleLlmProvider):
    """Identify a local vLLM server while reusing the compatible transport.

    The specialization keeps provider metadata and factory selection explicit
    without duplicating the OpenAI-compatible request and validation logic.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        model: str = "",
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
        max_output_tokens: int | None = None,
    ) -> None:
        """Configure the inherited transport with vLLM-specific provider identity.

        Authentication remains optional because local vLLM deployments commonly
        run on a trusted network without an API key.
        """

        super().__init__(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            provider_name="vllm_local",
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )

    def capabilities(self) -> LlmCapabilities:
        """Advertise deterministic seeds and the tested local decode budget.

        Dense relation windows can legitimately produce more than 8,192 tokens
        of schema-constrained JSON. The chart-managed server has ample context
        and KV capacity for a 16,384-token completion, while unrelated
        OpenAI-compatible providers retain their conservative inherited ceiling.

        On a fleet shared with interactive callers an operator will often want a
        tighter bound than this, because completion length is what sets how long
        a queue-jumping request waits. A configured ceiling therefore wins.
        """

        return LlmCapabilities(
            supports_seed=True,
            max_output_tokens=self._max_output_tokens or _VLLM_MAX_OUTPUT_TOKENS,
        )

    def _chat_request_overrides(self) -> dict[str, object]:
        """Disable free-form reasoning before strict JSON generation.

        Qwen enables thinking in its chat template by default. vLLM exposes
        that reasoning separately from ``message.content`` and counts it toward
        the completion budget, so structured extraction can otherwise exhaust
        its budget without producing a JSON object. The template switch is a
        vLLM extension and therefore belongs in this adapter, not in prompts or
        the provider-neutral completion request.
        """

        return {"chat_template_kwargs": {"enable_thinking": False}}
