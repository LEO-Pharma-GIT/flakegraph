# SPDX-License-Identifier: Apache-2.0
"""Snowflake Cortex embedding adapter using the Cortex REST embedding endpoint."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from kg_processor.adapters.snowflake import ConnectorFactory, SnowflakeConnectionConfig
from kg_processor.adapters.snowflake_cortex_rest import EMBED_PATH, CortexRestClient
from kg_processor.ports.embeddings import EMBEDDING_TIMEOUT_SECONDS, EmbedOptions

# Documented request limits of the embedding endpoint: at most this many texts
# per request, each at most this many characters.
_MAX_TEXTS_PER_REQUEST = 1280
_MAX_TEXT_CHARACTERS = 4096


class SnowflakeCortexEmbeddingProvider:
    """Embeds text with the Cortex REST ``inference:embed`` endpoint."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
        client: CortexRestClient | None = None,
    ) -> None:
        """Bind the credentials the REST client authenticates with."""

        self.config = config
        self.connector_factory = connector_factory
        self._client = client or CortexRestClient.from_config(config, connector_factory)

    def close(self) -> None:
        """Release retained HTTP connections and any session held for credentials."""

        self._client.close()

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Embed ordered text in bounded batches, one request per batch.

        Batching is enforced here because callers may pass a complete corpus.
        The configured batch size is honoured up to the endpoint's own limit.

        Text longer than the endpoint accepts is cut to its limit rather than
        failing the whole graph at the last stage of a run. That length is far
        beyond the configured chunk size and the bounded node, edge, and
        community texts, so in practice nothing is cut.

        Each request is bounded by the shared embedding timeout so a stalled
        provider releases the calling worker.
        """

        if not texts:
            return []
        batch_size = min(options.batch_size, _MAX_TEXTS_PER_REQUEST)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text[:_MAX_TEXT_CHARACTERS] for text in texts[start : start + batch_size]]
            events = self._client.post(
                EMBED_PATH,
                {"model": options.model, "text": batch},
                timeout_seconds=EMBEDDING_TIMEOUT_SECONDS,
                operation=f"embedding with model {options.model}",
            )
            for vector in _ordered_vectors(events[0], len(batch)):
                if len(vector) != options.dimension:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {options.dimension}, got {len(vector)}"
                    )
                vectors.append(vector)
        return vectors


def _ordered_vectors(payload: Mapping[str, Any], expected: int) -> list[list[float]]:
    """Place each returned vector at its input position and require all of them.

    The response carries each vector's input index. A missing, repeated, or
    out-of-range index would silently attach one text's vector to another, so
    any of them rejects the batch.
    """

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Snowflake Cortex embedding response has no data list")
    if len(data) != expected:
        raise ValueError(
            "Snowflake Cortex embedding result count mismatch: "
            f"expected {expected}, got {len(data)}"
        )
    ordered: list[list[float] | None] = [None] * expected
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("Snowflake Cortex embedding returned a malformed item")
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < expected:
            raise ValueError(f"Snowflake Cortex embedding returned an unexpected index: {index}")
        if ordered[index] is not None:
            raise ValueError(f"Snowflake Cortex embedding returned index {index} twice")
        ordered[index] = _normalize_vector(item.get("embedding"))
    return [vector for vector in ordered if vector is not None]


def _normalize_vector(value: object) -> list[float]:
    """Read one vector, which Snowflake documents wrapped in a one-element list."""

    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes | bytearray | dict):
        vector: list[float] = []
        for item in value:
            if not isinstance(item, int | float) or isinstance(item, bool):
                raise ValueError("Snowflake Cortex embedding returned a non-numeric vector value")
            vector.append(float(item))
        return vector
    raise ValueError("Snowflake Cortex embedding returned an unsupported vector value")
