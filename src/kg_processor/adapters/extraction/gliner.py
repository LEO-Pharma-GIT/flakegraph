# SPDX-License-Identifier: Apache-2.0
"""Optional GLiNER entity extractor implementing the shared entity-stage port."""

from __future__ import annotations

import re
from importlib import import_module
from threading import Lock
from typing import Any

from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    ExtractionWindow,
)
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label

_MAX_WINDOW_WORDS = 320
_WINDOW_OVERLAP_WORDS = 48


class GlinerEntityExtractor:
    """Adapt local GLiNER predictions to the provider-neutral entity port.

    The adapter owns lazy model loading, serializes access to model inference,
    and maps model labels and character spans onto the configured ontology.
    Relation extraction remains independent and can continue using any LLM.
    """

    def __init__(self, model_name: str, threshold: float = 0.5) -> None:
        """Store model selection and defer the load until the first extraction.

        Loading is deliberately deferred until the first extraction request so
        preflight and non-GLiNER commands do not pay model startup cost. Separate
        locks protect one-time loading and model implementations that are not
        safe to invoke concurrently.
        """

        self.model_name = model_name
        self.threshold = threshold
        self._model: Any | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def extract(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
        previous_entities: list[EntityMention] | None = None,
    ) -> EntityExtractionOutcome:
        """Return grounded entity mentions mapped onto ontology entity labels.

        Predictions with invalid offsets, unknown labels, or duplicate mention
        identities are discarded. Accepted records retain exact source spans so
        later relation grounding and evidence inspection do not depend on fuzzy
        string matching.
        """

        del model, timeout_seconds  # Local GLiNER owns its model and runs in-process.
        gliner = self._load_model()
        labels = ontology.entity_type_names()
        previous_keys = {
            (
                normalize_ontology_label(item.name),
                item.type,
                item.source_chunk_id,
                item.start_offset,
            )
            for item in (previous_entities or [])
        }
        mentions: list[EntityMention] = []
        duplicates = 0
        predictions_by_chunk: list[tuple[Chunk, int, list[object]]] = []
        inference_windows = 0
        with self._predict_lock:
            for chunk in window.chunks:
                for base_offset, text in _text_windows(chunk.content):
                    inference_windows += 1
                    predictions_by_chunk.append(
                        (
                            chunk,
                            base_offset,
                            gliner.predict_entities(
                                text,
                                labels,
                                threshold=self.threshold,
                                flat_ner=True,
                            ),
                        )
                    )
        for chunk, base_offset, predictions in predictions_by_chunk:
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    continue
                start = _integer(prediction.get("start"))
                end = _integer(prediction.get("end"))
                label = _ontology_label(str(prediction.get("label", "")), labels)
                if start is None or end is None or label is None or not 0 <= start < end:
                    continue
                start += base_offset
                end += base_offset
                if end > len(chunk.content):
                    continue
                quote = chunk.content[start:end]
                key = (normalize_ontology_label(quote), label, chunk.id, start)
                if key in previous_keys:
                    duplicates += 1
                    continue
                previous_keys.add(key)
                mentions.append(
                    EntityMention(
                        id=stable_id(
                            "entity_mention", window.id, chunk.id, label, start, end, quote
                        ),
                        name=quote,
                        type=label,
                        description=f"{quote} is explicitly mentioned in the source text.",
                        source_chunk_id=chunk.id,
                        quote=quote,
                        confidence=_score(prediction.get("score")),
                        start_offset=start,
                        end_offset=end,
                    )
                )
        mentions, overlapping = _prefer_longest_overlapping_mentions(mentions)
        duplicates += overlapping
        mentions = mentions[:max_entities]
        return EntityExtractionOutcome(
            entities=mentions,
            trace={
                "stage": "entity_extraction",
                "provider": "gliner",
                "model": self.model_name,
                "window_id": window.id,
                "accepted_records": len(mentions),
                "inference_windows": inference_windows,
                "record_actions": {"duplicate": duplicates} if duplicates else {},
            },
        )

    def _load_model(self) -> Any:
        """Load the optional dependency and model once, with a useful install error."""

        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                gliner_class = import_module("gliner").GLiNER
            except (AttributeError, ImportError) as exc:
                raise RuntimeError(
                    "GLiNER extraction requires: uv sync --extra extract-gliner"
                ) from exc
            self._model = gliner_class.from_pretrained(self.model_name)
        return self._model


def _prefer_longest_overlapping_mentions(
    mentions: list[EntityMention],
) -> tuple[list[EntityMention], int]:
    """Keep the longest same-label mention when overlapping windows disagree."""

    accepted: list[EntityMention] = []
    suppressed = 0
    ordered = sorted(
        mentions,
        key=lambda item: (
            item.source_chunk_id,
            item.type,
            _mention_span(item)[0],
            -(_mention_span(item)[1] - _mention_span(item)[0]),
            item.id,
        ),
    )
    for mention in ordered:
        mention_start, mention_end = _mention_span(mention)
        conflict = next(
            (
                existing
                for existing in accepted
                if existing.source_chunk_id == mention.source_chunk_id
                and existing.type == mention.type
                and mention_start < _mention_span(existing)[1]
                and _mention_span(existing)[0] < mention_end
            ),
            None,
        )
        if conflict is None:
            accepted.append(mention)
            continue
        suppressed += 1
        conflict_start, conflict_end = _mention_span(conflict)
        if (mention_end - mention_start) > (conflict_end - conflict_start):
            accepted[accepted.index(conflict)] = mention
    accepted.sort(key=lambda item: (item.source_chunk_id, *_mention_span(item)))
    return accepted, suppressed


def _mention_span(mention: EntityMention) -> tuple[int, int]:
    """Return the required grounded interval for a GLiNER-produced mention."""

    if mention.start_offset is None or mention.end_offset is None:
        raise ValueError("GLiNER mentions must carry source offsets")
    return mention.start_offset, mention.end_offset


def _text_windows(text: str) -> list[tuple[int, str]]:
    """Split long text into overlapping word windows with source offsets.

    GLiNER models commonly truncate inputs around a few hundred tokens. Character
    offsets for each slice allow predictions to map back to the exact chunk, while
    overlap protects entities that straddle a boundary.
    """

    words = list(re.finditer(r"\S+", text))
    if len(words) <= _MAX_WINDOW_WORDS:
        return [(0, text)]
    windows: list[tuple[int, str]] = []
    step = _MAX_WINDOW_WORDS - _WINDOW_OVERLAP_WORDS
    for word_start in range(0, len(words), step):
        word_end = min(word_start + _MAX_WINDOW_WORDS, len(words))
        start = words[word_start].start()
        end = words[word_end - 1].end()
        windows.append((start, text[start:end]))
        if word_end == len(words):
            break
    return windows


def _ontology_label(value: str, labels: list[str]) -> str | None:
    """Resolve a model label to the canonical spelling used by the ontology."""

    normalized = normalize_ontology_label(value)
    return next(
        (label for label in labels if normalize_ontology_label(label) == normalized),
        None,
    )


def _integer(value: object) -> int | None:
    """Accept integral prediction offsets while rejecting booleans, fractions, and strings.

    GLiNER offsets must index Python source text directly.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _score(value: object) -> float:
    """Coerce a model confidence to the normalized range expected by the domain."""

    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(float(value), 1.0))
    return 1.0
