# SPDX-License-Identifier: Apache-2.0
"""Load and fingerprint provider-neutral ontology profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kg_processor.domain.ontology import (
    EntityTypeDefinition,
    OntologyProfile,
    RelationTypeDefinition,
)


@dataclass(frozen=True)
class LoadedOntology:
    """Bundle a validated ontology with immutable source and checksum provenance.

    ``extraction`` is the profile the extraction passes read, and
    ``extraction_checksum`` its fingerprint, which extraction cache keys use:
    what only finalization reads - structural relations, folder exclusions -
    changes neither, so changing it reuses every extraction.
    """

    profile: OntologyProfile
    checksum: str
    source: str
    extraction: OntologyProfile
    extraction_checksum: str


def load_ontology(
    path: Path | None,
    fallback_entity_types: list[str],
    fallback_relation_types: list[str] | None,
    inline: Mapping[str, Any] | None = None,
) -> LoadedOntology:
    """Load a configured YAML ontology or construct a conservative fallback profile.

    Canonical JSON hashing makes semantically identical parsed profiles produce the
    same fingerprint regardless of YAML formatting.

    ``inline`` wins over ``path``: a caller that already holds the resolved profile
    is running somewhere the original file may not exist.
    """

    if inline is not None:
        profile = OntologyProfile.model_validate(dict(inline))
        source = "inline:configuration"
    elif path is None:
        profile = _fallback_profile(fallback_entity_types, fallback_relation_types)
        source = "generated:graph-settings"
    else:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Ontology file must contain a mapping: {path}")
        profile = OntologyProfile.model_validate(raw)
        source = str(path)
    extraction = profile.for_extraction()
    return LoadedOntology(
        profile=profile,
        checksum=_checksum(profile),
        source=source,
        extraction=extraction,
        extraction_checksum=_checksum(extraction),
    )


def _checksum(profile: OntologyProfile) -> str:
    """Fingerprint a profile."""

    canonical_json = json.dumps(
        profile.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def _fallback_profile(
    entity_types: list[str],
    relation_types: list[str] | None,
) -> OntologyProfile:
    """Build an inline ontology from graph label settings.

    Inline labels provide a concise configuration option while following the same
    prompt, validation, provenance, and quality contracts as profile files.
    """

    entities = [
        EntityTypeDefinition(
            name=name,
            description=f"A source-grounded entity categorized as {name}.",
        )
        for name in entity_types
    ]
    configured_relations = ["RELATED_TO"] if relation_types is None else relation_types
    relations = [
        RelationTypeDefinition(
            name=name,
            description=f"A source-grounded directed relationship categorized as {name}.",
        )
        for name in configured_relations
    ]
    return OntologyProfile(
        name="generated-general",
        description="Generated from graph.entity_types and graph.relation_types settings.",
        mode="open" if not relation_types else "closed",
        entity_types=entities,
        relation_types=relations,
    )
