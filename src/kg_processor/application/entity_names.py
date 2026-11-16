# SPDX-License-Identifier: Apache-2.0
"""Conservative entity-name normalization and alias validation.

LLMs often return related concepts in an ``aliases`` field. Relationship,
lineage, or co-occurrence is not identity, so only source-grounded spelling
variants and genuine initialisms are allowed to influence grounding or entity
resolution.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from kg_processor.application.extraction_grounding import surface_occurs

_MIN_INITIALISM_TOKENS = 2
_MIN_REGULAR_PLURAL_TOKEN_LENGTH = 4
_AMBIGUOUS_INITIALISM_WORDS = {
    "an",
    "as",
    "at",
    "be",
    "by",
    "do",
    "go",
    "he",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "no",
    "of",
    "on",
    "or",
    "so",
    "to",
    "up",
    "us",
    "we",
}


@dataclass(frozen=True)
class AliasValidationResult:
    """Report accepted identity aliases and unsafe values discarded during review.

    The rejection count is retained for extraction diagnostics without carrying
    untrusted model-proposed alias text into graph artifacts.
    """

    aliases: list[str]
    rejected_count: int


def validate_identity_aliases(
    name: str,
    proposed_aliases: list[str],
    source: str,
    *,
    is_identifier: Callable[[str], bool] | None = None,
) -> AliasValidationResult:
    """Keep only source-grounded spelling variants, initialisms, or identifiers of an entity.

    Semantically related concepts are not aliases. A candidate must occur in the
    source and pass conservative surface-identity rules before it can influence
    grounding or entity resolution. A code the ontology names an identifier - a
    registry or catalogue number - shares no surface with the name it
    identifies, so it needs only to occur in the source.
    """

    accepted: list[str] = []
    seen = {normalize_entity_surface(name)}
    rejected = 0
    for proposed in proposed_aliases:
        alias = proposed.strip()
        normalized = normalize_entity_surface(alias)
        if not alias or normalized in seen:
            continue
        initialism_alias = _safe_initialism_match(name, alias) or _safe_initialism_match(
            alias, name
        )
        if is_identifier is not None and is_identifier(alias) and surface_occurs(source, [alias]):
            accepted.append(alias)
            seen.add(normalized)
            continue
        if (
            not surface_occurs(source, [alias])
            or not identity_surface_match(name, alias)
            or (initialism_alias and not _case_sensitive_surface_occurs(source, alias))
        ):
            rejected += 1
            continue
        accepted.append(alias)
        seen.add(normalized)
    return AliasValidationResult(aliases=accepted, rejected_count=rejected)


def identity_surface_match(left: str, right: str) -> bool:
    """Return whether two surfaces are spelling, number, or initialism variants.

    This deliberately narrow check prevents relationship, lineage, and category
    terms from being treated as evidence that two mentions share an identity.
    A trailing plural ``s`` is ignored token by token because scientific sources
    routinely alternate names such as ``ResNet``/``ResNets`` and
    ``residual network``/``residual networks`` for the same model family.
    """

    left_key = normalize_entity_surface(left)
    right_key = normalize_entity_surface(right)
    return bool(
        left_key
        and right_key
        and (
            left_key == right_key
            or _singularized_tokens(left_key) == _singularized_tokens(right_key)
            or _single_ocr_token_split(left_key, right_key)
            or _safe_initialism_match(left, right)
            or _safe_initialism_match(right, left)
        )
    )


def _safe_initialism_match(expanded: str, compact: str) -> bool:
    candidate_surface = "".join(character for character in compact if character.isalnum())
    candidate = normalize_entity_surface(candidate_surface)
    return bool(
        candidate_surface
        and candidate_surface == candidate_surface.upper()
        and candidate not in _AMBIGUOUS_INITIALISM_WORDS
        and _initialism(normalize_entity_surface(expanded)) == candidate
    )


def _case_sensitive_surface_occurs(source: str, surface: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", source))


def _single_ocr_token_split(left: str, right: str) -> bool:
    """Recognize one PDF-OCR split inside an otherwise identical surface.

    Multi-column PDF extraction occasionally inserts a space after the first
    letter of a word, for example ``V eda Panneershelvam``. Compact-string
    equality alone would be too permissive because arbitrary word boundaries
    can change meaning. This check accepts exactly one additional boundary, and
    only when one side splits a token into a one-character prefix plus its
    remaining letters while every other token remains identical.
    """

    return _tokens_differ_by_single_prefix_split(left.split(), right.split()) or (
        _tokens_differ_by_single_prefix_split(right.split(), left.split())
    )


def _tokens_differ_by_single_prefix_split(
    compact_tokens: list[str],
    split_tokens: list[str],
) -> bool:
    """Compare token lists where ``split_tokens`` may contain one OCR split."""

    if len(split_tokens) != len(compact_tokens) + 1:
        return False
    for index, token in enumerate(compact_tokens):
        if (
            len(split_tokens[index]) == 1
            and split_tokens[index] + split_tokens[index + 1] == token
            and compact_tokens[index + 1 :] == split_tokens[index + 2 :]
            and compact_tokens[:index] == split_tokens[:index]
        ):
            return True
    return False


def normalize_entity_surface(value: str) -> str:
    """Normalize case and punctuation while preserving identity-bearing tokens.

    Token boundaries remain available for initialism checks, unlike a compacted
    key that could accidentally equate unrelated multi-word names.
    """

    compatible = unicodedata.normalize("NFKC", value)
    return " ".join(re.findall(r"[\w]+", compatible.casefold(), flags=re.UNICODE))


def _initialism(normalized: str) -> str:
    """Return a multi-token surface's initials or an empty non-match marker.

    Single words cannot establish an initialism relationship.
    """

    tokens = normalized.split()
    if len(tokens) < _MIN_INITIALISM_TOKENS:
        return ""
    return "".join(token[0] for token in tokens)


def _singularized_tokens(normalized: str) -> tuple[str, ...]:
    """Return tokens with only an unambiguous regular plural suffix removed.

    The helper deliberately does not attempt general stemming. Removing ``s``
    only from tokens longer than three characters, and never from ``ss`` words,
    covers common named-family plurals without equating unrelated word forms.
    """

    return tuple(
        token[:-1]
        if len(token) >= _MIN_REGULAR_PLURAL_TOKEN_LENGTH
        and token.endswith("s")
        and not token.endswith("ss")
        else token
        for token in normalized.split()
    )
