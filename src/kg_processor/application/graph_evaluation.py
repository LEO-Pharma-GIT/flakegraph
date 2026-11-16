# SPDX-License-Identifier: Apache-2.0
"""Gold-set evaluation and run-to-run stability for extracted graph artifacts."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field, model_validator

from kg_processor.application.graph_quality import as_list, evaluate_graph_quality_rows
from kg_processor.application.inspect import read_local_graph_artifacts
from kg_processor.application.ontology import load_ontology
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label

_MIN_COMMUNITY_SIZE = 2
_MAX_RECOVERY_HOPS = 2
_MIN_FUZZY_EVIDENCE_CHARACTERS = 10
_MIN_FUZZY_EVIDENCE_RATIO = 0.88


class GoldEntity(BaseModel):
    """Define one expected canonical entity and accepted source surface forms.

    Stable fixture IDs let relations reference entities independently from the
    display spelling produced by a particular model.
    """

    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)


class GoldDocument(BaseModel):
    """Identify one source document covered by the gold annotation.

    Document IDs keep relation observations stable when filenames or storage
    locations change, while paths bind the annotation to a reviewable corpus file.
    """

    id: str
    path: str
    checksum: str | None = None
    source_uri: str | None = None
    title: str | None = None
    year: int | None = Field(default=None, ge=1000, le=9999)


class GoldEvidenceObservation(BaseModel):
    """Bind one expected relation to an exact sentence in a corpus document.

    The shorter evidence substring is what generated edge evidence must retain;
    requiring it to occur in the annotated sentence prevents vague endpoint-only
    hints from masquerading as predicate support.
    """

    document: str
    sentence: str
    evidence_contains: str
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def evidence_must_be_specific_and_grounded(self) -> GoldEvidenceObservation:
        """Reject blank or detached evidence hints before benchmark scoring."""

        if not self.document.strip():
            raise ValueError("gold evidence document IDs must not be blank")
        if not self.sentence.strip():
            raise ValueError("gold evidence sentences must not be blank")
        if not self.evidence_contains.strip():
            raise ValueError("gold evidence substrings must not be blank")
        if _normalize(self.evidence_contains) not in _normalize(self.sentence):
            raise ValueError("gold evidence substring must occur in its annotated sentence")
        return self


class GoldRelation(BaseModel):
    """Define one expected directed triple and optional supporting phrases.

    Evidence phrases verify that a matched edge is supported by relevant text,
    rather than merely having the expected endpoint and predicate labels.
    """

    id: str | None = None
    source: str
    target: str
    relation_type: str
    required: bool = True
    evidence_contains: list[str] = Field(default_factory=list)
    observations: list[GoldEvidenceObservation] = Field(default_factory=list)
    # The values the relation states, as written (see Quantity.text).
    quantities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_hints_must_be_nonblank(self) -> GoldRelation:
        """Keep relation-level and observation-level evidence hints meaningful."""

        if any(not value.strip() for value in self.evidence_contains):
            raise ValueError("gold relation evidence substrings must not be blank")
        return self


class EvaluationThresholds(BaseModel):
    """Collect reviewable acceptance thresholds for one curated evaluation corpus.

    Keeping targets in the fixture makes quality expectations versioned alongside
    the facts being measured instead of hidden in benchmark code.
    """

    entity_precision: float = 0.90
    entity_recall: float = 0.80
    triple_precision: float = 0.90
    triple_recall: float = 0.80
    min_evidence_support: float = 0.80
    max_isolate_ratio: float = 0.15
    min_two_hop_recoverability: float = 0.90


class EvaluationScope(BaseModel):
    """Declare which unlisted predictions a gold graph can score as false positives.

    ``exhaustive`` entity coverage means every in-domain entity in the corpus was
    annotated. ``reference`` coverage represents a selected target subgraph, so
    unlisted entities remain unscored rather than becoming false positives.
    ``induced`` relation coverage is exhaustive only among matched reference
    entities; relations touching an unscored entity remain outside the benchmark.
    ``reference`` relation coverage is a selected set of required facts, so every
    other extracted relation remains unscored rather than becoming a false positive.
    """

    entity_coverage: Literal["exhaustive", "reference"] = "exhaustive"
    relation_coverage: Literal["exhaustive", "induced", "reference"] = "exhaustive"


class GoldGraph(BaseModel):
    """Describe a gold graph independent from provider-specific output.

    The fixture contains canonical entities, directed relations, evidence hints,
    and explicit acceptance thresholds used by repeatable local benchmarks.
    """

    schema_version: int = 1
    version: str = "1"
    name: str
    description: str
    license: str | None = None
    ontology_path: str | None = None
    documents: list[GoldDocument] = Field(default_factory=list)
    entities: list[GoldEntity]
    relations: list[GoldRelation]
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)
    evaluation_scope: EvaluationScope = Field(default_factory=EvaluationScope)

    @model_validator(mode="after")
    def relation_endpoints_must_exist(self) -> GoldGraph:
        """Reject gold relations whose stable endpoint IDs are not declared.

        Failing fixture validation early prevents a typo from appearing as false
        extraction recall loss in every benchmark run.
        """

        entity_ids = [entity.id for entity in self.entities]
        if len(set(entity_ids)) != len(entity_ids):
            raise ValueError("gold entity IDs must be unique")
        if any(not entity_id.strip() for entity_id in entity_ids):
            raise ValueError("gold entity IDs must not be blank")

        surfaces: dict[tuple[str, str], str] = {}
        for entity in self.entities:
            for surface in [entity.name, *entity.aliases]:
                normalized = _normalize(surface)
                if not normalized:
                    raise ValueError("gold entity names and aliases must not be blank")
                typed_surface = (normalized, _normalize_label(entity.type))
                owner = surfaces.setdefault(typed_surface, entity.id)
                if owner != entity.id:
                    raise ValueError(
                        f"gold surface {surface!r} with type {entity.type!r} is shared by "
                        f"entity IDs {owner!r} and {entity.id!r}"
                    )

        relation_ids = [relation.id for relation in self.relations if relation.id is not None]
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("gold relation IDs must be unique")
        triples = [
            (relation.source, _normalize_label(relation.relation_type), relation.target)
            for relation in self.relations
        ]
        if len(set(triples)) != len(triples):
            raise ValueError("gold directed relation triples must be unique")

        document_ids = [document.id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("gold document IDs must be unique")
        known_documents = set(document_ids)
        unknown_documents = {
            observation.document
            for relation in self.relations
            for observation in relation.observations
            if observation.document not in known_documents
        }
        if unknown_documents:
            raise ValueError(
                f"gold observations refer to unknown documents: {sorted(unknown_documents)}"
            )

        entity_id_set = set(entity_ids)
        missing = {
            endpoint
            for relation in self.relations
            for endpoint in (relation.source, relation.target)
            if endpoint not in entity_id_set
        }
        if missing:
            raise ValueError(f"Gold relations refer to unknown entities: {sorted(missing)}")
        if (
            self.evaluation_scope.relation_coverage in {"induced", "reference"}
            and self.evaluation_scope.entity_coverage != "reference"
        ):
            raise ValueError(
                "induced and reference relation coverage require reference entity coverage"
            )
        return self


def load_gold_graph(path: Path) -> GoldGraph:
    """Load a JSON gold fixture and validate its cross-record invariants.

    Provider output is never involved, keeping the expected graph reviewable and
    reusable across local, hosted, and Snowflake extraction backends.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldGraph.model_validate(raw)


def evaluate_graph_artifacts(output_path: Path, gold_path: Path) -> dict[str, Any]:
    """Compare one local graph snapshot with a curated gold graph comprehensively.

    The report combines entity and triple precision/recall, evidence support,
    structural diagnostics, two-hop recoverability, and hard quality gates into
    explicit acceptance decisions suitable for benchmark automation.
    """

    gold = load_gold_graph(gold_path)
    ontology = _load_gold_ontology(gold, gold_path)
    artifacts = read_local_graph_artifacts(output_path)
    nodes = artifacts["nodes"].rows
    edges = artifacts["edges"].rows
    evidence = artifacts["evidence"].rows
    documents = artifacts["documents"].rows
    chunks = artifacts["chunks"].rows
    communities = artifacts["communities"].rows

    node_matches, entity_metrics = _entity_metrics(
        nodes,
        gold.entities,
        coverage=gold.evaluation_scope.entity_coverage,
    )
    node_names = {str(node.get("id")): _normalize(str(node.get("name", ""))) for node in nodes}
    triple_metrics = _triple_metrics(
        edges,
        node_matches,
        node_names,
        gold.relations,
        coverage=gold.evaluation_scope.relation_coverage,
    )
    required_relations = [relation for relation in gold.relations if relation.required]
    evidence_metrics = _evidence_metrics(
        edges,
        evidence,
        node_matches,
        required_relations,
        documents=documents,
        gold_documents=gold.documents,
    )
    observed_structural = _structural_metrics(nodes, edges, communities)
    if gold.evaluation_scope.entity_coverage == "reference":
        structural = _reference_structural_metrics(edges, node_matches)
    else:
        structural = observed_structural
    two_hop = _two_hop_recoverability(edges, node_matches, required_relations)
    quantity_metrics = _quantity_metrics(
        edges,
        artifacts["edge_observations"].rows,
        node_matches,
        gold.relations,
    )
    quality = evaluate_graph_quality_rows(
        nodes,
        edges,
        evidence,
        expected_embedding_dimension=None,
        chunks=chunks,
        communities=communities,
        ontology=ontology,
    )
    thresholds = gold.thresholds
    acceptance = {
        "entity_recall": entity_metrics["recall"] >= thresholds.entity_recall,
        "triple_recall": triple_metrics["recall"] >= thresholds.triple_recall,
        "evidence_support": (evidence_metrics["support_rate"] >= thresholds.min_evidence_support),
        "isolate_ratio": structural["isolate_ratio"] <= thresholds.max_isolate_ratio,
        "two_hop_recoverability": two_hop >= thresholds.min_two_hop_recoverability,
        "hard_quality_checks": quality.ok,
    }
    if gold.evaluation_scope.entity_coverage == "exhaustive":
        acceptance["entity_precision"] = entity_metrics["precision"] >= thresholds.entity_precision
    if gold.evaluation_scope.relation_coverage == "exhaustive":
        acceptance["triple_precision"] = triple_metrics["precision"] >= thresholds.triple_precision
    return {
        "gold_name": gold.name,
        "output_path": str(output_path),
        "gold_path": str(gold_path),
        "evaluation_scope": gold.evaluation_scope.model_dump(mode="json"),
        "ok": all(acceptance.values()),
        "acceptance": acceptance,
        "thresholds": thresholds.model_dump(mode="json"),
        "entities": entity_metrics,
        "triples": triple_metrics,
        "evidence": evidence_metrics,
        "structural": structural,
        "observed_structural": observed_structural,
        "two_hop_recoverability": two_hop,
        "quantities": quantity_metrics,
        # MINE-inspired retention treats entity and relation recall equally so
        # an extractor cannot score well merely by returning many entity names.
        "information_retention": round(
            (entity_metrics["recall"] + triple_metrics["recall"]) / 2,
            4,
        ),
        "quality": quality.model_dump(mode="json"),
    }


def _load_gold_ontology(gold: GoldGraph, gold_path: Path) -> OntologyProfile | None:
    """Load the ontology declared by a gold fixture for prediction validation.

    Repository fixtures commonly store a root-relative path, while standalone
    fixtures can use a path relative to their own directory. Trying those two
    deterministic locations keeps the contract portable without searching the
    filesystem or silently ignoring a misspelled ontology path.
    """

    if gold.ontology_path is None:
        return None
    configured = Path(gold.ontology_path)
    candidates = (
        [configured] if configured.is_absolute() else [configured, gold_path.parent / configured]
    )
    ontology_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if ontology_path is None:
        raise FileNotFoundError(f"gold ontology does not exist: {gold.ontology_path}")
    return load_ontology(ontology_path, [], None).profile


def _quantity_metrics(
    edges: list[dict[str, Any]],
    edge_observations: list[dict[str, Any]],
    node_matches: dict[str, str],
    gold_relations: list[GoldRelation],
) -> dict[str, Any]:
    """Score the values relations state against the gold's, on matched edges.

    Recall is the share of gold values found on an edge that matches their gold
    relation; precision is the share of values on those edges the gold lists.
    Values compare by normalized text, so '≤0.5 %' and '≤ 0.5%' agree.
    """

    edges_by_triple: dict[tuple[str, str, str], list[str]] = {}
    for edge in edges:
        source = node_matches.get(str(edge.get("source_node_id")))
        target = node_matches.get(str(edge.get("target_node_id")))
        if source and target:
            key = (source, _normalize_label(str(edge.get("relation_type", ""))), target)
            edges_by_triple.setdefault(key, []).append(str(edge.get("id")))
    observed_by_edge: dict[str, set[str]] = {}
    for observation in edge_observations:
        for quantity in as_list(observation.get("quantities")):
            text = quantity.get("text") if isinstance(quantity, dict) else None
            if text:
                observed_by_edge.setdefault(str(observation.get("edge_id")), set()).add(
                    _normalize_value(str(text))
                )
    expected = found = observed = matched = 0
    for relation in gold_relations:
        if not relation.quantities:
            continue
        key = (relation.source, _normalize_label(relation.relation_type), relation.target)
        values = set().union(
            *(observed_by_edge.get(edge_id, set()) for edge_id in edges_by_triple.get(key, []))
        )
        gold_values = {_normalize_value(value) for value in relation.quantities}
        expected += len(gold_values)
        found += len(gold_values & values)
        observed += len(values)
        matched += len(values & gold_values)
    return {
        "expected": expected,
        "found": found,
        "recall": round(found / expected, 4) if expected else None,
        "observed_on_matched_edges": observed,
        "precision": round(matched / observed, 4) if observed else None,
    }


def _normalize_value(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def graph_signatures(output_path: Path) -> tuple[set[str], set[str]]:
    """Return provider-neutral node and triple signatures for stability comparison.

    Names and labels are normalized so rerun comparisons measure semantic graph
    drift rather than casing or punctuation changes.
    """

    artifacts = read_local_graph_artifacts(output_path)
    nodes = artifacts["nodes"].rows
    edges = artifacts["edges"].rows
    name_by_id = {str(node.get("id")): _normalize(str(node.get("name", ""))) for node in nodes}
    node_signatures = set(name_by_id.values())
    triple_signatures = {
        "|".join(
            [
                name_by_id.get(str(edge.get("source_node_id")), ""),
                _normalize_label(str(edge.get("relation_type", ""))),
                name_by_id.get(str(edge.get("target_node_id")), ""),
            ]
        )
        for edge in edges
    }
    return node_signatures, triple_signatures


def jaccard(left: set[str], right: set[str]) -> float:
    """Measure set overlap while treating two empty outputs as perfectly stable.

    This metric is used for repeated extraction signatures where ordering and
    duplicate observations are intentionally irrelevant.
    """

    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _entity_metrics(
    nodes: list[dict[str, Any]],
    gold_entities: list[GoldEntity],
    *,
    coverage: Literal["exhaustive", "reference"] = "exhaustive",
) -> tuple[dict[str, str], dict[str, Any]]:
    """Match actual nodes to typed gold aliases and compute entity PRF metrics.

    Each expected entity counts once even if extraction emits duplicate canonical
    nodes; duplicate matches are reported separately for diagnosis.
    """

    typed_alias_to_gold = {
        (_normalize(surface), _normalize_label(entity.type)): entity.id
        for entity in gold_entities
        for surface in [entity.name, *entity.aliases]
    }
    node_matches: dict[str, str] = {}
    for node in nodes:
        surfaces = {
            _normalize(str(surface))
            for surface in [node.get("name", ""), *as_list(node.get("aliases"))]
            if str(surface).strip()
        }
        candidates = {
            typed_alias_to_gold[(surface, node_type)]
            for surface in surfaces
            for node_type in _node_types(node)
            if (surface, node_type) in typed_alias_to_gold
        }
        # A canonical node can retain several ontology types after resolution.
        # Match only an unambiguous typed surface instead of depending on map
        # insertion order when two gold concepts intentionally share a name.
        if len(candidates) == 1:
            node_matches[str(node.get("id"))] = candidates.pop()
    matched_gold = set(node_matches.values())
    duplicate_matches = len(node_matches) - len(matched_gold)
    true_positives = len(matched_gold)
    unscored_nodes = [node for node in nodes if str(node.get("id")) not in node_matches]
    false_positives = len(unscored_nodes) if coverage == "exhaustive" else 0
    false_negatives = len(gold_entities) - true_positives
    metrics: dict[str, Any] = _prf(true_positives, false_positives, false_negatives)
    metrics.update(
        {
            "expected": len(gold_entities),
            "actual": true_positives + false_positives,
            "observed_actual": len(nodes),
            "coverage": coverage,
            "duplicate_canonical_matches": duplicate_matches,
            "unscored_count": len(unscored_nodes) if coverage == "reference" else 0,
            "unscored": (
                sorted(str(node.get("name", "")) for node in unscored_nodes)
                if coverage == "reference"
                else []
            ),
            "missing": sorted(
                entity.name for entity in gold_entities if entity.id not in matched_gold
            ),
            "unexpected": sorted(
                str(node.get("name", "")) for node in unscored_nodes if coverage == "exhaustive"
            ),
        }
    )
    if coverage != "exhaustive":
        metrics["precision"] = None
        metrics["f1"] = None
        metrics["precision_applicable"] = False
    else:
        metrics["precision_applicable"] = True
    return node_matches, metrics


def _triple_metrics(
    edges: list[dict[str, Any]],
    node_matches: dict[str, str],
    node_names: dict[str, str],
    gold_relations: list[GoldRelation],
    *,
    coverage: Literal["exhaustive", "induced", "reference"] = "exhaustive",
) -> dict[str, Any]:
    """Compare normalized directed edge signatures against expected gold triples.

    Matched nodes use stable gold IDs while unmatched nodes retain prefixed actual
    names, keeping unexpected triples human-readable in evaluation output.
    """

    required = {
        (relation.source, _normalize_label(relation.relation_type), relation.target)
        for relation in gold_relations
        if relation.required
    }
    accepted = {
        (relation.source, _normalize_label(relation.relation_type), relation.target)
        for relation in gold_relations
    }
    observed_actual: set[tuple[str, str, str]] = set()
    actual: set[tuple[str, str, str]] = set()
    for edge in edges:
        source_id = str(edge.get("source_node_id"))
        target_id = str(edge.get("target_node_id"))
        signature = (
            _matched_or_actual(source_id, node_matches, node_names),
            _normalize_label(str(edge.get("relation_type", ""))),
            _matched_or_actual(target_id, node_matches, node_names),
        )
        observed_actual.add(signature)
        if (
            coverage == "exhaustive"
            or (coverage == "induced" and source_id in node_matches and target_id in node_matches)
            or (coverage == "reference" and signature in accepted)
        ):
            actual.add(signature)
    unscored = observed_actual - actual
    missing = required - actual
    precision = len(accepted & actual) / max(len(actual), 1)
    recall = len(required & actual) / max(len(required), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics: dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
    metrics.update(
        {
            "expected": len(required),
            "accepted_expected": len(accepted),
            "actual": len(actual),
            "observed_actual": len(observed_actual),
            "coverage": coverage,
            "unscored_count": len(unscored),
            "unscored": [_triple_text(item) for item in sorted(unscored)],
            "missing": [_triple_text(item) for item in sorted(missing)],
            # The gold relations behind the missing triples, by id, so a caller
            # can name them without repeating this module's normalization.
            "missing_relation_ids": sorted(
                (
                    relation.id
                    for relation in gold_relations
                    if relation.required
                    and (relation.source, _normalize_label(relation.relation_type), relation.target)
                    in missing
                ),
                key=lambda value: value or "",
            ),
            "missing_by_reason": _missing_triple_reasons(
                missing,
                observed_actual,
                matched_gold_ids=set(node_matches.values()),
            ),
            "unexpected": [_triple_text(item) for item in sorted(actual - accepted)],
        }
    )
    if coverage != "exhaustive":
        metrics["precision"] = None
        metrics["f1"] = None
        metrics["precision_applicable"] = False
    else:
        metrics["precision_applicable"] = True
    return metrics


def _missing_triple_reasons(
    missing: set[tuple[str, str, str]],
    observed: set[tuple[str, str, str]],
    *,
    matched_gold_ids: set[str],
) -> dict[str, int]:
    """Classify recall loss without relaxing exact directed-triple scoring.

    Endpoint failures identify upstream entity extraction or canonicalization
    loss. Once both endpoints are present, reversed and wrong-predicate cases
    distinguish semantic labeling mistakes from an entirely absent edge. The
    categories are mutually exclusive and always emitted, giving benchmark
    tooling a stable schema even when a run has perfect recall.
    """

    reasons = {
        "both_endpoints_missing": 0,
        "source_endpoint_missing": 0,
        "target_endpoint_missing": 0,
        "reversed": 0,
        "wrong_predicate": 0,
        "no_edge": 0,
    }
    predicates_by_endpoints: dict[tuple[str, str], set[str]] = {}
    for source, relation_type, target in observed:
        predicates_by_endpoints.setdefault((source, target), set()).add(relation_type)

    for source, relation_type, target in missing:
        source_present = source in matched_gold_ids
        target_present = target in matched_gold_ids
        if not source_present and not target_present:
            reasons["both_endpoints_missing"] += 1
        elif not source_present:
            reasons["source_endpoint_missing"] += 1
        elif not target_present:
            reasons["target_endpoint_missing"] += 1
        elif relation_type in predicates_by_endpoints.get((target, source), set()):
            reasons["reversed"] += 1
        elif predicates_by_endpoints.get((source, target)):
            reasons["wrong_predicate"] += 1
        else:
            reasons["no_edge"] += 1
    return reasons


def _evidence_metrics(
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    node_matches: dict[str, str],
    gold_relations: list[GoldRelation],
    *,
    documents: list[dict[str, Any]] | None = None,
    gold_documents: list[GoldDocument] | None = None,
) -> dict[str, Any]:
    """Measure whether matched gold relations retain expected evidence phrases.

    The denominator is all gold relations carrying evidence hints, so missing
    relations cannot inflate support by disappearing before evidence evaluation.
    Checksum-derived source identity and the annotated evidence phrase are
    enforced together. Multiple
    observations can describe valid passages for the same triple without allowing
    an unrelated quote from the correct PDF to receive evidence credit.
    """

    expected_by_key = {
        (relation.source, _normalize_label(relation.relation_type), relation.target): relation
        for relation in gold_relations
    }
    actual_document_by_file = {
        str(document.get("file_id") or document.get("id")): gold_document.id
        for document in documents or []
        for gold_document in gold_documents or []
        if str(document.get("checksum") or "")
        and str(gold_document.checksum or "")
        and str(document.get("checksum")) == str(gold_document.checksum)
    }
    evidence_by_subject: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        evidence_by_subject.setdefault(str(item.get("subject_id")), []).append(item)
    expected_with_phrases = sum(bool(relation.observations) for relation in gold_relations)
    matched_keys: set[tuple[str, str, str]] = set()
    supported_keys: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (
            node_matches.get(str(edge.get("source_node_id")), ""),
            _normalize_label(str(edge.get("relation_type", ""))),
            node_matches.get(str(edge.get("target_node_id")), ""),
        )
        relation = expected_by_key.get(key)
        if relation is None or not relation.observations:
            continue
        matched_keys.add(key)
        records = evidence_by_subject.get(str(edge.get("id")), [])
        if _relation_evidence_supported(relation, records, actual_document_by_file):
            supported_keys.add(key)
    return {
        "gold_relations_with_phrases": expected_with_phrases,
        "matched_relations_with_phrases": len(matched_keys),
        "supported": len(supported_keys),
        "support_rate": round(len(supported_keys) / expected_with_phrases, 4)
        if expected_with_phrases
        else 1.0,
    }


def _relation_evidence_supported(
    relation: GoldRelation,
    records: list[dict[str, Any]],
    actual_document_by_file: dict[str, str],
) -> bool:
    """Return whether one edge has source-grounded evidence in a reviewed document.

    Each observation binds an expected phrase to a checksum-matched document.
    Page numbers remain review metadata
    rather than hard constraints because OCR can merge content across page
    boundaries, but a nonempty unrelated quote is never sufficient.
    """

    phrases_by_document: dict[str, set[str]] = {}
    for observation in relation.observations:
        phrases_by_document.setdefault(observation.document, set()).add(
            observation.evidence_contains
        )
    for record in records:
        document_id = actual_document_by_file.get(str(record.get("file_id", "")))
        quote = str(record.get("quote", ""))
        if document_id is None or not quote.strip():
            continue
        if any(
            _evidence_phrase_matches(phrase, quote)
            for phrase in phrases_by_document.get(document_id, set())
            if phrase
        ):
            return True
    return False


def _evidence_phrase_matches(phrase: str, quote: str) -> bool:
    """Match reviewed evidence while tolerating a very small OCR transcription error.

    Exact alphanumeric matching remains the primary rule. The fallback compares
    only same-length token windows and requires a high character similarity, so
    it can recover ligature damage, a transposition such as ``with``/``wtih``, or
    one corrupted surname without treating a paraphrase as source evidence. The
    caller has already matched the directed edge and checksum-bound document,
    which keeps this tolerance local to evidence retention rather than fact
    discovery.
    """

    normalized_phrase = _normalize(phrase)
    normalized_quote = _normalize(quote)
    if not normalized_phrase or not normalized_quote:
        return False
    if normalized_phrase in normalized_quote:
        return True
    phrase_words = _normalized_words(phrase)
    quote_words = _normalized_words(quote)
    if (
        len(normalized_phrase) < _MIN_FUZZY_EVIDENCE_CHARACTERS
        or not phrase_words
        or len(quote_words) < len(phrase_words)
    ):
        return False
    width = len(phrase_words)
    return any(
        SequenceMatcher(
            None,
            normalized_phrase,
            "".join(quote_words[start : start + width]),
            autojunk=False,
        ).ratio()
        >= _MIN_FUZZY_EVIDENCE_RATIO
        for start in range(len(quote_words) - width + 1)
    )


def _normalized_words(value: str) -> list[str]:
    """Return compatibility-normalized word tokens for bounded OCR comparison."""

    compatible = unicodedata.normalize("NFKD", value).casefold()
    # PDF text layers commonly split a word at a visual line-break hyphen. Joining
    # only hyphen-plus-whitespace avoids changing ordinary compound-word tokens.
    compatible = re.sub(r"-\s+", "", compatible)
    return [
        "".join(character for character in token if character.isalnum())
        for token in re.findall(r"[^\W_]+(?:['’][^\W_]+)?", compatible, flags=re.UNICODE)
        if any(character.isalnum() for character in token)
    ]


def _structural_metrics(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    communities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize connectivity, isolates, self-loops, vocabulary, and communities.

    These metrics reveal graph-shape regressions that entity and triple matching
    alone cannot explain.
    """

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(str(node.get("id")) for node in nodes)
    graph.add_edges_from(
        (str(edge.get("source_node_id")), str(edge.get("target_node_id"))) for edge in edges
    )
    isolates = list(nx.isolates(graph))
    components = list(nx.connected_components(graph)) if graph else []
    self_loops = sum(
        str(edge.get("source_node_id")) == str(edge.get("target_node_id")) for edge in edges
    )
    singleton_communities = sum(
        len(as_list(item.get("member_node_ids"))) < _MIN_COMMUNITY_SIZE for item in communities
    )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "isolates": len(isolates),
        "isolate_ratio": round(len(isolates) / max(len(nodes), 1), 4),
        "components": len(components),
        "self_loops": self_loops,
        "relation_types": len(
            {_normalize_label(str(edge.get("relation_type", ""))) for edge in edges}
        ),
        "singleton_communities": singleton_communities,
    }


def _reference_structural_metrics(
    edges: list[dict[str, Any]],
    node_matches: dict[str, str],
) -> dict[str, Any]:
    """Measure reference topology after collapsing accepted surfaces to gold IDs.

    Several actual nodes may legitimately match one reference entity when the gold
    annotation accepts multiple granular surfaces. Collapsing those nodes prevents
    duplicate surface forms from appearing as isolated benchmark entities while the
    whole-output structure remains available in ``observed_structural``.
    """

    nodes = [{"id": gold_id} for gold_id in sorted(set(node_matches.values()))]
    scoped_edges: list[dict[str, Any]] = []
    for edge in edges:
        source = node_matches.get(str(edge.get("source_node_id")))
        target = node_matches.get(str(edge.get("target_node_id")))
        if source is None or target is None:
            continue
        scoped_edges.append(
            {
                "source_node_id": source,
                "target_node_id": target,
                "relation_type": edge.get("relation_type", ""),
            }
        )
    return _structural_metrics(nodes, scoped_edges, [])


def _two_hop_recoverability(
    edges: list[dict[str, Any]],
    node_matches: dict[str, str],
    gold_relations: list[GoldRelation],
) -> float:
    """Measure whether expected endpoints remain connected within two graph hops.

    Recoverability gives partial credit when an exact predicate is missed but the
    extracted graph still preserves a useful short path between gold entities.
    """

    graph: nx.Graph[str] = nx.Graph()
    for edge in edges:
        source = str(edge.get("source_node_id"))
        target = str(edge.get("target_node_id"))
        if source and target:
            # Keep unmatched actual nodes: they can be useful intermediates in a
            # two-hop path between matched gold endpoints.
            graph.add_edge(source, target)
    actual_by_gold: dict[str, set[str]] = {}
    for actual_id, gold_id in node_matches.items():
        actual_by_gold.setdefault(gold_id, set()).add(actual_id)
    recovered = 0
    for relation in gold_relations:
        source_nodes = actual_by_gold.get(relation.source, set())
        target_nodes = actual_by_gold.get(relation.target, set())
        if any(
            _within_recovery_distance(graph, source, target)
            for source in source_nodes
            for target in target_nodes
        ):
            recovered += 1
    return round(recovered / max(len(gold_relations), 1), 4)


def _within_recovery_distance(graph: nx.Graph[str], source: str, target: str) -> bool:
    """Return whether two actual nodes remain connected within the benchmark hop limit."""

    try:
        return nx.shortest_path_length(graph, source, target) <= _MAX_RECOVERY_HOPS
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, float]:
    """Compute rounded precision, recall, and F1 from explicit confusion counts.

    Empty denominators produce finite zero-valued metrics.
    """

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _triple_text(value: tuple[str, str, str]) -> str:
    """Render one normalized directed triple as concise evaluation diagnostics.

    The arrow notation preserves predicate direction visibly.
    """

    return " --".join([value[0], f"{value[1]}--> {value[2]}"])


def _matched_or_actual(
    node_id: str,
    node_matches: dict[str, str],
    node_names: dict[str, str],
) -> str:
    """Return a stable gold ID for matched nodes or a readable actual fallback."""

    matched = node_matches.get(node_id)
    if matched:
        return matched
    return f"actual:{node_names.get(node_id, node_id)}"


def _node_types(node: dict[str, Any]) -> set[str]:
    """Return every normalized node type, including its display-primary type.

    Artifact rows may carry either or both fields, so evaluation accepts the
    complete typed representation rather than relying on one serialization.
    """

    values = [node.get("primary_type", ""), *as_list(node.get("types"))]
    return {_normalize_label(str(value)) for value in values if str(value).strip()}


def _normalize(value: str) -> str:
    """Normalize entity surfaces into alphanumeric, case-insensitive evaluation signatures.

    Punctuation differences therefore do not count as semantic drift.
    """

    # Decomposition handles both compatibility ligatures and OCR streams that
    # encode accents as spacing or combining marks. Non-alphanumeric marks are
    # discarded below, so canonically equivalent typography compares equally.
    compatible = unicodedata.normalize("NFKD", value)
    return "".join(character for character in compatible.casefold() if character.isalnum())


def _normalize_label(value: str) -> str:
    """Normalize relation and type labels into comparable underscore-separated keys.

    Hyphens and repeated spaces receive equivalent treatment.
    """

    return normalize_ontology_label(value)
