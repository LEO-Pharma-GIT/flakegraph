"""Contracts keeping local and Spark finalization identical for the same input.

Both engines merge into the same tables, so a rule that drifts between them mints
different IDs, weights, or ratings for one graph. These tests pin the rules that
must have exactly one definition and the values derived from them.
"""

from __future__ import annotations

from typing import Any

from llm_fakes import RecordingCommunityLlm

from kg_processor.application import spark_finalization
from kg_processor.application.community_reports import (
    CommunitySeed,
    generate_community_reports,
    structural_rating,
)
from kg_processor.application.graph_merge import (
    MAX_RELATION_LABEL_LENGTH,
    normalize_entity_name,
    normalize_relation_type,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import Evidence, GraphEdge, GraphNode
from kg_processor.domain.ontology import normalize_ontology_label

_SSL_KEY = "spark.hadoop.fs.s3a.connection.ssl.enabled"
# Inputs chosen so a rule reimplemented in another language diverges: decomposed
# accents, case folding that changes length, ligatures, non-Latin scripts,
# whitespace runs, punctuation-only values, and oversized labels.
TRICKY_VALUES: tuple[str, ...] = (
    "Cafe\u0301 Latte",
    "Café Latte",
    "  Spaced   Name  ",
    "line\tbreak\nname",
    "STRASSE",
    "Straße",
    "ﬁnance",
    "İstanbul",
    "Next.js",
    "Müller",
    "東京",
    "-- /// --",
    "",
    "Works-At",
    "works at",
    "very-long predicate " * 12,
    "x" * 200,
)


def test_spark_canonical_name_rule_is_the_local_rule() -> None:
    """The distributed engine must key nodes with the reference name rule itself."""

    for value in TRICKY_VALUES:
        assert spark_finalization._normalized_name_value(value) == normalize_entity_name(value)
    assert spark_finalization._normalized_name_value(None) is None


def test_canonical_name_rule_composes_unicode_and_folds_case() -> None:
    """Document the properties a lowercase-and-strip translation cannot reproduce."""

    assert normalize_entity_name("Cafe\u0301 Latte") == normalize_entity_name("Café Latte")
    assert normalize_entity_name("Straße") == normalize_entity_name("STRASSE") == "strasse"
    assert normalize_entity_name("  Spaced   Name  ") == "spacedname"
    assert normalize_entity_name("-- /// --") == "-- /// --"


def test_spark_relation_label_rule_is_the_local_rule() -> None:
    """Canonical edge IDs and ontology rule keys share one bounded label rule."""

    for value in TRICKY_VALUES:
        spark_label = spark_finalization._normalized_relation_type_value(value)
        assert spark_label == normalize_relation_type(value)
    assert spark_finalization._normalized_relation_type_value(None) is None


def test_relation_label_rule_bounds_oversized_predicates() -> None:
    """A predicate longer than the bound still resolves to its own ontology rule."""

    label = "very-long predicate " * 12
    bounded = normalize_relation_type(label)

    assert bounded == normalize_ontology_label(label)[:MAX_RELATION_LABEL_LENGTH]
    assert len(bounded) == MAX_RELATION_LABEL_LENGTH
    assert normalize_relation_type(bounded) == bounded


def test_structural_rating_stays_inside_its_published_range() -> None:
    """Loops and parallel predicates can assert more pairs than a size ceiling allows."""

    rating, explanation = structural_rating(2, 3, 3, 6, 1.0)

    assert rating == 10.0
    assert "density 1.00" in explanation


def test_structural_rating_scores_the_relations_placed_in_the_report() -> None:
    """Coverage compares quotes with the relations that can supply them."""

    fully_supported = structural_rating(10, 5, 4, 4, 0.0)[0]
    additional_quotes_clamped = structural_rating(10, 5, 4, 8, 0.0)[0]
    thinly_supported = structural_rating(10, 5, 40, 4, 0.0)[0]

    assert fully_supported == additional_quotes_clamped
    assert fully_supported > thinly_supported


def test_community_rating_uses_the_relations_the_report_received() -> None:
    """A community rating must not fall because relations were truncated from context."""

    nodes = [
        _node("node_a", "Alpha"),
        _node("node_b", "Beta"),
        _node("node_c", "Gamma"),
    ]
    edges = [
        _edge("edge_ab", "node_a", "node_b", "works_at", weight=9.0),
        _edge("edge_bc", "node_b", "node_c", "located_in", weight=8.0),
        _edge("edge_ac", "node_a", "node_c", "located_in", weight=7.0),
        _edge("edge_ab_alt", "node_a", "node_b", "founded", weight=6.0),
    ]
    evidence = [_evidence("edge_ab"), _evidence("edge_bc")]

    result = generate_community_reports(
        "graph",
        [CommunitySeed(member_ids={"node_a", "node_b", "node_c"})],
        nodes,
        edges,
        RecordingCommunityLlm(),
        max_relations_per_community=2,
        evidence=evidence,
    )

    community = result.communities[0]
    assert community.rating == 10.0
    assert community.rating_explanation == (
        "Structural score from density 1.00, evidence coverage 1.00, and mean edge confidence 1.00."
    )


def test_https_artifact_endpoint_keeps_transport_encryption() -> None:
    """An operator's TLS endpoint must never be silently downgraded to plaintext."""

    assert _artifact_ssl_setting("https://artifacts.example.com") == "true"
    assert _artifact_ssl_setting("HTTPS://artifacts.example.com") == "true"
    assert _artifact_ssl_setting("artifacts.example.com:9000") == "true"


def test_http_artifact_endpoint_disables_transport_encryption() -> None:
    """A plaintext endpoint is the only configuration that turns encryption off."""

    assert _artifact_ssl_setting("http://minio.default.svc:9000") == "false"


def _node(node_id: str, name: str) -> GraphNode:
    """Build one canonical node with the degree used for member ordering."""

    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=normalize_entity_name(name),
        name=name,
        primary_type="ENTITY",
        types=["ENTITY"],
        description=f"{name} appears in the source graph.",
        source_chunk_ids=["chunk_1"],
        degree=2,
        rank=2.0,
    )


def _edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    weight: float,
) -> GraphEdge:
    """Build one canonical edge participating in a community."""

    return GraphEdge(
        id=edge_id,
        graph_id="graph",
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,
        description=f"{source_node_id} {relation_type} {target_node_id}.",
        weight=weight,
        source_file_id="file_1",
        source_chunk_ids=["chunk_1"],
    )


def _evidence(edge_id: str) -> Evidence:
    """Build one exact quote supporting a canonical edge."""

    return Evidence(
        id=f"evidence_{edge_id}",
        graph_id="graph",
        subject_id=edge_id,
        subject_kind="edge",
        file_id="file_1",
        chunk_id="chunk_1",
        page_number=1,
        start_offset=0,
        end_offset=10,
        quote="Grounded source sentence.",
    )


class _RecordingBuilder:
    """Collect Spark configuration entries without starting a session."""

    def __init__(self) -> None:
        """Start with an empty configuration record."""

        self.entries: dict[str, str] = {}

    def config(self, key: str, value: str) -> _RecordingBuilder:
        """Record one configuration entry and continue the builder chain."""

        self.entries[key] = value
        return self


def _artifact_ssl_setting(endpoint_url: str) -> str:
    """Return the S3A transport setting the finalizer derives for one endpoint."""

    overrides: dict[str, Any] = {
        "distributed": {
            "artifact_uri": "s3://artifacts",
            "artifact_endpoint_url": endpoint_url,
        }
    }
    settings = Settings.load(env={}, overrides=overrides)
    builder = _RecordingBuilder()
    spark_finalization.SparkGraphFinalizer(settings)._configure_s3(builder)
    return builder.entries[_SSL_KEY]
