from __future__ import annotations

import math
import os

import pytest

from kg_processor.adapters.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from kg_processor.ports.embeddings import EmbedOptions

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("KG_RUN_SENTENCE_TRANSFORMERS_LIVE") != "1",
        reason=(
            "Set KG_RUN_SENTENCE_TRANSFORMERS_LIVE=1 to run live sentence-transformers "
            "embedding integration checks."
        ),
    ),
]

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_DIMENSION = 384


def test_live_sentence_transformers_embeddings_are_normalized_and_repeatable() -> None:
    pytest.importorskip("sentence_transformers")
    model = os.getenv("KG_SENTENCE_TRANSFORMERS_MODEL", _DEFAULT_MODEL)
    dimension = int(os.getenv("KG_SENTENCE_TRANSFORMERS_DIMENSION", str(_DEFAULT_DIMENSION)))
    device = os.getenv("KG_SENTENCE_TRANSFORMERS_DEVICE", "cpu")

    provider = SentenceTransformersEmbeddingProvider(device=device)
    options = EmbedOptions(model=model, dimension=dimension, batch_size=2)
    vectors = provider.embed(
        [
            "River City Clinic studies public health workflows.",
            "Copenhagen hosts open research teams.",
            "River City Clinic studies public health workflows.",
        ],
        options,
    )

    assert len(vectors) == 3
    assert all(len(vector) == dimension for vector in vectors)
    assert all(math.isclose(_norm(vector), 1.0, rel_tol=1e-4, abs_tol=1e-4) for vector in vectors)
    assert _max_delta(vectors[0], vectors[2]) < 1e-6
    assert _l1_distance(vectors[0], vectors[1]) > 1e-3


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _max_delta(left: list[float], right: list[float]) -> float:
    return max(
        abs(left_value - right_value) for left_value, right_value in zip(left, right, strict=True)
    )


def _l1_distance(left: list[float], right: list[float]) -> float:
    return sum(
        abs(left_value - right_value) for left_value, right_value in zip(left, right, strict=True)
    )
