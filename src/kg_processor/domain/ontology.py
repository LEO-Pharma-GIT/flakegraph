# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral ontology contracts for typed graph extraction.

Ontology profiles describe what an entity or relation means without coupling
that definition to a particular LLM, OCR engine, or graph writer.  The same
profile is therefore enforced across local, hosted, and Snowflake providers.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator


class EntityTypeDefinition(BaseModel):
    """Define one allowed entity category and language explaining its scope.

    Aliases and examples guide models while the canonical name remains stable for
    validation and persisted graph typing.
    """

    name: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    document_context: bool = False
    # An entity of this type found by the document-context pass is what the
    # document itself is or describes - a paper's own title, the material a
    # certificate or data sheet covers - so a statement in that document may
    # have it as its implicit source. Requires document_context.
    document_subject: bool = False
    # Names of this type may be codes or numbers, such as a batch or lot number.
    # Where values have a home on relations (see RelationTypeDefinition), only
    # such types may have a name that is just a number or a value.
    identifier: bool = False
    contextual_surfaces: list[str] = Field(default_factory=list)
    # How to tell a named thing of this type from its neighbours, and things
    # that look like it but are not; extraction and verification read both.
    guidance: str = ""
    counter_examples: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def subject_requires_document_context(self) -> EntityTypeDefinition:
        """Only a type the document-context pass looks for can be a document's subject."""

        if self.document_subject and not self.document_context:
            raise ValueError(
                f"ontology entity {self.name} is a document subject without "
                "enabling document_context"
            )
        return self


class RelationTypeDefinition(BaseModel):
    """Define one directed predicate and all structural/evidence constraints.

    Domain/range types, inverses, symmetry, self-loop policy, aliases, examples,
    and grounding cues form a reviewable contract independent from any model.
    """

    name: str
    description: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    inverse: str | None = None
    symmetric: bool = False
    allow_self_loop: bool = False
    examples: list[str] = Field(default_factory=list)
    evidence_cues: list[str] = Field(default_factory=list)
    # The relation may carry the values it states - limits, ranges, results -
    # as quantities, so those values never become entities of their own.
    quantities: bool = False
    # What text states this relation, and wording that looks like it but does
    # not: a signature is not responsibility, a citation is not a description.
    guidance: str = ""
    counter_examples: list[str] = Field(default_factory=list)
    # Edges of this type are stated by how the sources are organized, not by
    # their text: extraction never offers the type to the model, and community
    # detection and graph-shape checks leave the edges out, so a hub such as
    # a folder linked to everything filed in it groups nothing. The edges stay
    # in the graph and its exports.
    structural: bool = False

    def admits(self, source_type: str, target_type: str) -> bool:
        """Return whether endpoint types satisfy this relation's signature.

        An empty source or target type list leaves that side unrestricted.
        """

        source_allowed = not self.source_types or source_type in self.source_types
        target_allowed = not self.target_types or target_type in self.target_types
        return source_allowed and target_allowed


class RelationRewrite(BaseModel):
    """Keep a statement under another relation when it links these entity types.

    A model sometimes names a true statement with a relation whose type rules
    its entities break: "the cream is tested by viscosity" links a product to a
    quality attribute, which HAS_QUALITY_ATTRIBUTE states. A rewrite keeps such
    a statement under ``to`` instead of rejecting it; the stated relation stays
    on its observation. An empty type list matches any type on that side.
    """

    relation: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    to: str

    def matches(self, relation: str, source_type: str, target_type: str) -> bool:
        """Whether a statement of this relation between these types is rewritten."""

        return (
            normalize_ontology_label(relation) == normalize_ontology_label(self.relation)
            and (not self.source_types or source_type in self.source_types)
            and (not self.target_types or target_type in self.target_types)
        )


class IdentifierPattern(BaseModel):
    """A code that identifies an entity and is never an entity itself.

    A registry number or catalogue code names the thing it identifies; as a
    name of its own it would make a node that says nothing. A name that matches
    the whole pattern is refused as an entity and kept as an alias where the
    model gives it as one.
    """

    name: str
    pattern: str

    def matches(self, text: str) -> bool:
        """Whether the whole of ``text`` is such a code."""

        return re.fullmatch(self.pattern, text.strip(), flags=re.IGNORECASE) is not None


DEFAULT_FOLDER_RELATION = "APPEARS_IN"
# The open relation a stated relation that names a structural type is kept
# under, when the profile names no fallback relation.
RESTATED_RELATION = "RELATED_TO"


class FolderEntityDefinition(BaseModel):
    """Tie every document to one entity named by the folder it is filed in.

    Where the folder a file sits in says what the file belongs to - a study,
    a contract, a project - each document gets one entity of ``entity_type``
    for its folder at ``level`` below the source root (1 is the top folder),
    and every entity found in the document is linked to it by
    ``relation_type``. The entity is named by that folder's own name, so
    folders of one name at that level are one entity: the level is the one
    whose folder names identify the thing. A file with no folder at that
    level gets none.

    The exclusions are read when a graph is finalized, so changing them builds
    a new version from the stored extraction: a listed document keeps its
    entities but is tied to no folder. A path is the file's location below the
    source root, folders and file name, as its location page states it.
    """

    entity_type: str
    relation_type: str = DEFAULT_FOLDER_RELATION
    level: int = Field(default=1, ge=1)
    excluded_file_ids: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)

    def folder_at_level(self, folder: str | None) -> str | None:
        """The folder path at this definition's level, or None above it."""

        parts = _path_parts(folder or "")
        if len(parts) < self.level:
            return None
        return "/".join(parts[: self.level])

    def excludes(self, file_id: str, location: str) -> bool:
        """Whether a document, by its file id or its location, is tied to no folder."""

        if file_id in self.excluded_file_ids:
            return True
        parts = _path_parts(location)
        return any(_path_parts(path) == parts for path in self.excluded_paths)

    def excluded_path_folders(self) -> set[str]:
        """The folders the excluded paths are in; a file at the source root is in ``""``."""

        return {"/".join(_path_parts(path)[:-1]) for path in self.excluded_paths}

    def without_exclusions(self) -> FolderEntityDefinition:
        """The definition as extraction reads it: which folder, not which documents."""

        return self.model_copy(update={"excluded_file_ids": [], "excluded_paths": []})


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part]


class OntologyProfile(BaseModel):
    """Represent a complete, reviewable extraction vocabulary and enforcement mode.

    The same profile drives prompt schemas, record validation, graph quality, and
    run provenance across every provider implementation.
    """

    name: str
    description: str
    mode: Literal["closed", "open", "hybrid"] = "hybrid"
    entity_types: list[EntityTypeDefinition]
    relation_types: list[RelationTypeDefinition] = Field(default_factory=list)
    # A relation type that keeps a stated connection whose own type rules the
    # two entities break: "Granulator uses wet granulation" where the ontology
    # lets only products use a process. The edge takes this type, and the
    # label the model gave it is kept on its observation. Unset, such a
    # statement is rejected.
    fallback_relation: str | None = None
    # Statements kept under another relation, by stated relation and entity
    # types; the first rewrite that matches applies.
    relation_rewrites: list[RelationRewrite] = Field(default_factory=list)
    identifier_patterns: list[IdentifierPattern] = Field(default_factory=list)
    folder_entity: FolderEntityDefinition | None = None
    # The labels of the structural relations the extraction view leaves out:
    # extraction keeps a stated relation off them.
    _reserved_relation_labels: frozenset[str] = PrivateAttr(default_factory=frozenset)

    @model_validator(mode="after")
    def folder_entity_must_be_typed_and_linked(self) -> OntologyProfile:
        """A folder entity is of a defined type and linked by a structural relation.

        The relation is added when the profile lacks it: any entity to the
        folder's type. One the profile defines must be that relation already,
        because extraction would otherwise offer it to the model as a relation
        text states. This runs first, so every later rule sees the relation.
        """

        folder = self.folder_entity
        if folder is None:
            return self
        if folder.entity_type not in self.entity_type_names():
            raise ValueError(
                f"ontology folder entity type {folder.entity_type} is not an entity type "
                "of the ontology"
            )
        if not folder.relation_type.strip():
            raise ValueError("ontology folder relation must have a name")
        defined = self.relation(folder.relation_type)
        if defined is None:
            self.relation_types.append(
                RelationTypeDefinition(
                    name=folder.relation_type,
                    description=(
                        f"The entity is found in a document filed in the folder of this "
                        f"{folder.entity_type}."
                    ),
                    target_types=[folder.entity_type],
                    structural=True,
                )
            )
            return self
        if not defined.structural:
            raise ValueError(
                f"ontology folder relation {defined.name} is a relation extraction reads "
                "from text; name another relation or mark it structural"
            )
        if defined.source_types or (
            defined.target_types and folder.entity_type not in defined.target_types
        ):
            raise ValueError(
                f"ontology folder relation {defined.name} must link any entity type to "
                f"{folder.entity_type}"
            )
        return self

    @model_validator(mode="after")
    def definitions_must_be_unique_and_referentially_valid(self) -> OntologyProfile:
        """Reject ambiguous names, broken references, and invalid cues before extraction.

        Early profile validation prevents expensive provider calls under an ontology
        that could never produce internally consistent graph rows.
        """

        entity_names = [definition.name for definition in self.entity_types]
        relation_names = [definition.name for definition in self.relation_types]
        normalized_entity_names = [normalize_ontology_label(name) for name in entity_names]
        normalized_relation_labels = [
            normalize_ontology_label(label)
            for definition in self.relation_types
            for label in [definition.name, *definition.aliases]
        ]
        if len(set(normalized_entity_names)) != len(normalized_entity_names):
            raise ValueError("ontology entity type names must be unique")
        if len(set(normalized_relation_labels)) != len(normalized_relation_labels):
            raise ValueError(
                "ontology relation names and aliases must be unique after normalization"
            )
        known_entities = set(entity_names)
        known_relations = set(relation_names)
        contextual_surface_owners: dict[str, str] = {}
        for entity in self.entity_types:
            if entity.contextual_surfaces and not entity.document_context:
                raise ValueError(
                    f"ontology entity {entity.name} declares contextual surfaces without "
                    "enabling document_context"
                )
            if any(not surface.strip() for surface in entity.contextual_surfaces):
                raise ValueError(
                    f"ontology entity {entity.name} contains a blank contextual surface"
                )
            for surface in entity.contextual_surfaces:
                normalized_surface = " ".join(surface.casefold().split())
                owner = contextual_surface_owners.setdefault(normalized_surface, entity.name)
                if owner != entity.name:
                    raise ValueError(
                        f"ontology contextual surface {surface!r} is shared by document-context "
                        f"entity types {owner} and {entity.name}"
                    )
        for relation in self.relation_types:
            unknown_types = (
                set(relation.source_types) | set(relation.target_types)
            ) - known_entities
            if unknown_types:
                raise ValueError(
                    f"ontology relation {relation.name} refers to unknown entity types: "
                    f"{', '.join(sorted(unknown_types))}"
                )
            if relation.inverse and relation.inverse not in known_relations:
                raise ValueError(
                    f"ontology relation {relation.name} has unknown inverse {relation.inverse}"
                )
            for cue in relation.evidence_cues:
                try:
                    re.compile(cue, flags=re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(
                        f"ontology relation {relation.name} has invalid evidence cue: {exc}"
                    ) from exc
        return self

    @model_validator(mode="after")
    def fallback_relation_must_be_defined(self) -> OntologyProfile:
        """A fallback relation names one of the profile's own relation types."""

        if self.fallback_relation is not None and self.fallback_relation not in {
            relation.name for relation in self.relation_types
        }:
            raise ValueError(f"ontology fallback relation {self.fallback_relation} is not defined")
        return self

    @model_validator(mode="after")
    def rewrites_and_identifiers_must_be_valid(self) -> OntologyProfile:
        """A rewrite joins defined relations whose rules admit it; a pattern compiles."""

        known_entities = {definition.name for definition in self.entity_types}
        for rewrite in self.relation_rewrites:
            stated, kept = self.relation(rewrite.relation), self.relation(rewrite.to)
            if stated is None or kept is None:
                raise ValueError(
                    f"ontology rewrite {rewrite.relation} -> {rewrite.to} "
                    "names an undefined relation"
                )
            unknown = (set(rewrite.source_types) | set(rewrite.target_types)) - known_entities
            if unknown:
                raise ValueError(
                    f"ontology rewrite {rewrite.relation} -> {rewrite.to} refers to unknown "
                    f"entity types: {', '.join(sorted(unknown))}"
                )
            refused = [
                f"{source} -> {target}"
                for source in rewrite.source_types or [""]
                for target in rewrite.target_types or [""]
                if (source and kept.source_types and source not in kept.source_types)
                or (target and kept.target_types and target not in kept.target_types)
            ]
            if refused:
                raise ValueError(
                    f"ontology rewrite {rewrite.relation} -> {rewrite.to} keeps statements "
                    f"{kept.name} does not allow: {', '.join(refused)}"
                )
        for identifier in self.identifier_patterns:
            try:
                re.compile(identifier.pattern)
            except re.error as exc:
                raise ValueError(
                    f"ontology identifier pattern {identifier.name} is invalid: {exc}"
                ) from exc
        return self

    @model_validator(mode="after")
    def structural_relations_are_never_stated(self) -> OntologyProfile:
        """No rule keeps a stated relation under a structural one."""

        structural = {item.name for item in self.relation_types if item.structural}
        if not structural:
            return self
        named = [
            *([self.fallback_relation] if self.fallback_relation else []),
            *(rewrite.to for rewrite in self.relation_rewrites),
            *(item.inverse for item in self.relation_types if item.inverse),
        ]
        stated = sorted(
            {
                definition.name
                for name in named
                if (definition := self.relation(name)) is not None and definition.structural
            }
        )
        if stated:
            raise ValueError(
                "ontology structural relations cannot be a fallback, a rewrite target "
                f"or an inverse: {', '.join(stated)}"
            )
        return self

    def for_extraction(self) -> OntologyProfile:
        """The profile as extraction reads it: no structural relation, no exclusions.

        Structural edges are added when a graph is finalized, so the model is
        never offered their types, and which documents a folder leaves out is
        decided there too.
        """

        view = self.model_copy(
            update={
                "relation_types": [item for item in self.relation_types if not item.structural],
                "folder_entity": (
                    self.folder_entity.without_exclusions() if self.folder_entity else None
                ),
            }
        )
        view._reserved_relation_labels = self.reserved_relation_labels()
        return view

    def reserved_relation_labels(self) -> frozenset[str]:
        """The normalized labels no stated relation may carry: the structural ones."""

        return self.structural_relation_labels() | self._reserved_relation_labels

    def restated_relation(self) -> RelationTypeDefinition | None:
        """The relation a stated relation naming a structural type is kept under.

        Such a label is not the vocabulary's to a statement, so it is handled as
        any label outside the vocabulary is: a closed profile rejects it, and
        another keeps it under its fallback relation, else under an open
        ``RELATED_TO``. None when it is rejected.
        """

        if self.mode == "closed":
            return None
        fallback = self.fallback_definition()
        if fallback is not None:
            return fallback
        if normalize_ontology_label(RESTATED_RELATION) in self.reserved_relation_labels():
            return None
        return self.relation(RESTATED_RELATION) or RelationTypeDefinition(
            name=RESTATED_RELATION,
            description="Source-grounded relation discovered outside the declared vocabulary.",
        )

    def structural_relation_labels(self) -> frozenset[str]:
        """Every normalized label, name or alias, of a structural relation type."""

        return frozenset(
            normalize_ontology_label(label)
            for item in self.relation_types
            if item.structural
            for label in [item.name, *item.aliases]
        )

    def rewrite_for(
        self, relation: str, source_type: str, target_type: str
    ) -> RelationTypeDefinition | None:
        """The relation a statement is kept under instead of the one it was stated as."""

        for rewrite in self.relation_rewrites:
            if rewrite.matches(relation, source_type, target_type):
                kept = self.relation(rewrite.to)
                if kept is not None and kept.admits(source_type, target_type):
                    return kept
        return None

    def identifier_pattern(self, text: str) -> IdentifierPattern | None:
        """The identifier pattern ``text`` wholly matches, if any."""

        return next((item for item in self.identifier_patterns if item.matches(text)), None)

    def fallback_definition(self) -> RelationTypeDefinition | None:
        """The relation type a statement falls back to when its own rules reject it."""

        return next(
            (item for item in self.relation_types if item.name == self.fallback_relation), None
        )

    def carries_quantities(self) -> bool:
        """Whether any relation type gives stated values a home as quantities."""

        return any(definition.quantities for definition in self.relation_types)

    def entity_type_names(self) -> list[str]:
        """Return canonical entity labels in their original configured prompt and schema order."""

        return [definition.name for definition in self.entity_types]

    def relation_type_names(self) -> list[str]:
        """Return canonical relation labels in their original configured prompt and schema order."""

        return [definition.name for definition in self.relation_types]

    def relation(self, name: str) -> RelationTypeDefinition | None:
        """Resolve a relation by normalized canonical name or configured alias.

        Returning the canonical definition centralizes direction, type, loop, and
        evidence policy for extraction and quality checks.
        """

        normalized = normalize_ontology_label(name)
        for definition in self.relation_types:
            labels = [definition.name, *definition.aliases]
            if normalized in {normalize_ontology_label(label) for label in labels}:
                return definition
        return None


def normalize_ontology_label(value: str) -> str:
    """Normalize ontology labels across case, hyphen, and whitespace variations.

    This is the single relation-label identity rule used by ontology lookup,
    canonical edge assembly, and graph quality checks. Keeping it public avoids
    subtle differences where one stage treats ``works-at`` and ``works at`` as
    equal while another persists them as separate predicates.
    """

    return "_".join(value.casefold().replace("-", " ").split())
