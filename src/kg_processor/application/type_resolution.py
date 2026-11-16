# SPDX-License-Identifier: Apache-2.0
"""Admit held relations under the types other documents give their entities.

One extraction window reads each entity under one type. A statement whose
relation rejects that reading is held rather than dropped, and finalization -
the first stage that sees every document - admits it when an endpoint's name is
also a node of a type the relation allows, attested in enough documents. Both
engines choose from the rule table built here: the local merge in
``graph_merge``, the Spark finalizer over DataFrames.

An admitted relation is an ordinary edge observation whose evidence method is
``type_resolved``, so it can be judged apart; the rest stay rejected records.
"""

from __future__ import annotations

from dataclasses import dataclass

from kg_processor.domain.ontology import OntologyProfile

TYPE_RESOLVED_METHOD = "type_resolved"

# (stated relation, source type, target type) -> (kept relation, stated relation
# when a rewrite replaced it).
TypeRules = dict[tuple[str, str, str], tuple[str, str | None]]


@dataclass(frozen=True)
class TypeResolution:
    """The rules and the attestation a held relation's admission needs.

    ``min_documents`` is how many distinct documents must type an endpoint's
    name as the type it is admitted under; an endpoint that keeps the type its
    own window gave it needs none.
    """

    rules: TypeRules
    min_documents: int


def type_resolution_rules(ontology: OntologyProfile) -> TypeRules:
    """Every typing under which a restricted relation is kept, and as what.

    A relation is kept under itself where its own rules admit the two types,
    else under the relation an ontology rewrite names for them. The fallback
    relation is not consulted: it keeps any statement, and admitting a held
    statement under it would say nothing its window did not already refuse.
    A relation without type rules is never held and has no rows.
    """

    entity_types = ontology.entity_type_names()
    rules: TypeRules = {}
    for definition in ontology.relation_types:
        if not definition.source_types and not definition.target_types:
            continue
        for source_type in entity_types:
            for target_type in entity_types:
                key = (definition.name, source_type, target_type)
                if definition.admits(source_type, target_type):
                    rules[key] = (definition.name, None)
                elif (
                    rewritten := ontology.rewrite_for(definition.name, source_type, target_type)
                ) is not None:
                    rules[key] = (rewritten.name, definition.name)
    return rules


def type_resolution_metrics(held: int, admitted: int) -> dict[str, int]:
    """How many relations windows held, and how many finalization admitted.

    The held relations it did not admit remain in the rejected records as type
    violations; the admitted ones are edges with ``type_resolved`` evidence.
    """

    return {"held_relations": held, "type_resolved_relations": admitted}
