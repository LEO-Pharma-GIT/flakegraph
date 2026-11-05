from __future__ import annotations

import httpx
import pytest
from snowflake_fakes import CONFIG, FakeCortexServer, embedding_reply

from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.ports.embeddings import EMBEDDING_TIMEOUT_SECONDS, EmbedOptions


def test_snowflake_cortex_embeddings_post_texts_and_validate_dimension() -> None:
    server = FakeCortexServer(embedding_reply([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    provider = _provider(server)

    vectors = provider.embed(
        ["Alice", "Acme"],
        EmbedOptions(model="snowflake-arctic-embed-m-v1.5", dimension=3),
    )

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    request = server.requests[0]
    assert str(request.url) == (
        "https://example-account.snowflakecomputing.com/api/v2/cortex/inference:embed"
    )
    assert server.bodies == [{"model": "snowflake-arctic-embed-m-v1.5", "text": ["Alice", "Acme"]}]
    provider.close()
    assert server.credentials.closed


def test_snowflake_cortex_embeddings_accept_unwrapped_vectors() -> None:
    """A flat vector is read the same as the documented one-element wrapper."""

    server = FakeCortexServer(
        httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
    )

    assert _provider(server).embed(["Alice"], EmbedOptions(model="embed", dimension=2)) == [
        [0.1, 0.2]
    ]


def test_snowflake_cortex_embeddings_reject_dimension_mismatch() -> None:
    server = FakeCortexServer(embedding_reply([[0.1, 0.2]]))

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        _provider(server).embed(["Alice"], EmbedOptions(model="embed", dimension=3))


def test_snowflake_cortex_embeddings_enforce_configured_batch_size() -> None:
    server = FakeCortexServer(
        embedding_reply([[0.1], [0.2]]),
        embedding_reply([[0.3], [0.4]]),
        embedding_reply([[0.5]]),
    )

    vectors = _provider(server).embed(
        ["one", "two", "three", "four", "five"],
        EmbedOptions(model="embed", dimension=1, batch_size=2),
    )

    assert vectors == [[0.1], [0.2], [0.3], [0.4], [0.5]]
    assert [body["text"] for body in server.bodies] == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]


def test_snowflake_cortex_embeddings_cap_batches_at_the_endpoint_limit() -> None:
    """A configured batch larger than the endpoint accepts is split to fit it."""

    texts = [f"text {index}" for index in range(1281)]
    server = FakeCortexServer(embedding_reply([[0.1]] * 1280), embedding_reply([[0.2]]))

    vectors = _provider(server).embed(
        texts, EmbedOptions(model="embed", dimension=1, batch_size=5000)
    )

    assert len(vectors) == 1281
    assert [len(body["text"]) for body in server.bodies] == [1280, 1]


def test_snowflake_cortex_embeddings_cut_text_to_the_endpoint_limit() -> None:
    server = FakeCortexServer(embedding_reply([[0.1]]))

    _provider(server).embed(["x" * 5000], EmbedOptions(model="embed", dimension=1))

    assert server.bodies[0]["text"] == ["x" * 4096]


def test_snowflake_cortex_embeddings_bound_every_batch_request() -> None:
    """A stalled embedding request must release the calling worker rather than hold it."""

    server = FakeCortexServer(embedding_reply([[0.1], [0.2]]), embedding_reply([[0.3]]))

    _provider(server).embed(
        ["one", "two", "three"],
        EmbedOptions(model="embed", dimension=1, batch_size=2),
    )

    assert [request.extensions["timeout"]["read"] for request in server.requests] == [
        EMBEDDING_TIMEOUT_SECONDS
    ] * 2


def test_snowflake_cortex_embeddings_reject_missing_vectors() -> None:
    """A partial response must not silently misalign later texts."""

    server = FakeCortexServer(embedding_reply([[0.1]]))

    with pytest.raises(ValueError, match="result count mismatch"):
        _provider(server).embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))


def test_snowflake_cortex_embeddings_place_vectors_by_their_index() -> None:
    """Input order is part of the port contract, so each vector goes where its index says."""

    server = FakeCortexServer(
        httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [[0.2]]},
                    {"index": 0, "embedding": [[0.1]]},
                ]
            },
        )
    )

    vectors = _provider(server).embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))

    assert vectors == [[0.1], [0.2]]


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [0, None]])
def test_snowflake_cortex_embeddings_reject_ambiguous_indices(indices: list[object]) -> None:
    server = FakeCortexServer(
        httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [[0.1]]} for index in indices]},
        )
    )

    with pytest.raises(ValueError, match="index"):
        _provider(server).embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))


def _provider(server: FakeCortexServer) -> SnowflakeCortexEmbeddingProvider:
    return SnowflakeCortexEmbeddingProvider(CONFIG, client=server.client())
