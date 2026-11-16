# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible embedding adapter with strict dimension validation."""

from __future__ import annotations

import httpx

from kg_processor.adapters.embeddings.openai_common import parse_embedding_vectors
from kg_processor.adapters.llm.openai_common import send_with_http_retry
from kg_processor.ports.embeddings import EMBEDDING_TIMEOUT_SECONDS, EmbedOptions

_DIMENSIONS_PARAMETER = "dimensions"
_REJECTED_REQUEST_STATUSES = {400, 422}


class OpenAICompatibleEmbeddingProvider:
    """Embeds text through any OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, endpoint: str, api_key: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client()
        self._dimensions_unsupported = False

    def close(self) -> None:
        """Release retained keep-alive connections owned by this adapter."""

        self._client.close()

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Batch texts through the endpoint and validate returned dimensions."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), options.batch_size):
            batch = texts[start : start + options.batch_size]

            include_dimensions = not self._dimensions_unsupported

            def send(
                batch: list[str] = batch,
                include_dimensions: bool = include_dimensions,
            ) -> httpx.Response:
                return self._post_embeddings(batch, options, include_dimensions=include_dimensions)

            response = send_with_http_retry(send)
            requested_dimensions = self._uses_dimensions_parameter(options, include_dimensions)
            if requested_dimensions and _rejects_dimensions_parameter(response):
                self._dimensions_unsupported = True

                def send_without_dimensions(batch: list[str] = batch) -> httpx.Response:
                    return self._post_embeddings(batch, options, include_dimensions=False)

                response = send_with_http_retry(send_without_dimensions)
            response.raise_for_status()
            batch_vectors = parse_embedding_vectors(response.json(), len(batch))
            for vector in batch_vectors:
                if len(vector) != options.dimension:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {options.dimension}, got {len(vector)}"
                    )
            vectors.extend(batch_vectors)
        return vectors

    def _uses_dimensions_parameter(
        self,
        options: EmbedOptions,
        include_dimensions: bool,
    ) -> bool:
        """Return whether this request shape carries the optional dimensions field."""

        return include_dimensions and options.model.startswith("text-embedding-3-")

    def _post_embeddings(
        self,
        batch: list[str],
        options: EmbedOptions,
        *,
        include_dimensions: bool = True,
    ) -> httpx.Response:
        """Submit one OpenAI-compatible embedding batch."""

        payload: dict[str, object] = {"model": options.model, "input": batch}
        if include_dimensions and options.model.startswith("text-embedding-3-"):
            payload[_DIMENSIONS_PARAMETER] = options.dimension
        return self._client.post(
            f"{self.endpoint}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
        )


def _rejects_dimensions_parameter(response: httpx.Response) -> bool:
    """Return whether a rejection names the optional dimensions parameter.

    Dropping the parameter narrows every later vector in the process to the
    model's native width, so the provider has to say that the parameter is the
    problem. An oversized batch, an invalid model, or a quota rejection arrives
    with the same status and must leave the requested width intact.
    """

    if response.status_code not in _REJECTED_REQUEST_STATUSES:
        return False
    error = _response_error(response)
    if isinstance(error, str):
        return _DIMENSIONS_PARAMETER in error.casefold()
    if not isinstance(error, dict):
        return False
    parameter = error.get("param")
    if isinstance(parameter, str) and parameter.strip().casefold() == _DIMENSIONS_PARAMETER:
        return True
    message = error.get("message")
    return isinstance(message, str) and _DIMENSIONS_PARAMETER in message.casefold()


def _response_error(response: httpx.Response) -> object:
    """Return the provider's error field from a rejected embedding response."""

    try:
        payload = response.json()
    except ValueError:
        return None
    return payload.get("error") if isinstance(payload, dict) else None
