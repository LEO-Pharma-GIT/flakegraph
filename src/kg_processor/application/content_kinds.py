# SPDX-License-Identifier: Apache-2.0
"""What a source file is - a document, a dataset, an image - and what that means.

A document is read for its statements. A dataset (a spreadsheet or delimited
table) is profiled once, and its kind decides how much of it is extracted: a
catalogue lists things, and its columns say how each row's thing relates to
the things and values beside it; measurements record values for things; and
anything else - calculations, templates, schedules - is described rather than
extracted. An image is kept as an asset of the graph even when it has no text
to read.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import ColumnRelation
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ontology import OntologyProfile

ContentKind = Literal["document", "dataset", "image"]
DatasetKind = Literal["catalogue", "measurements", "other"]
DATASET_KINDS: tuple[DatasetKind, ...] = ("catalogue", "measurements", "other")
DATASET_PROFILE_STAGE = "dataset_profile"

_DATASET_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv"})
_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic"}
)
# Which extraction passes each dataset kind runs: (entities, relations). A
# catalogue's relation pass reads its rows by their column relations, so it runs
# only when the profile kept at least one.
_DATASET_WINDOWS: dict[str, tuple[bool, bool]] = {
    "catalogue": (True, True),
    "measurements": (True, True),
    "other": (True, False),
}
# What a dataset profile may say a catalogue column is; "none" is not kept.
COLUMN_ROLES = ("subject", "relation", "value", "alias", "none")
# A file name token that identifies something - a batch, lot, or order number -
# has a digit and at least this many characters.
_MIN_IDENTIFIER_TOKEN = 4
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Related files are the same folder's documents that share an identifier with
# the file, or the whole folder when it is this small.
_SMALL_FOLDER = 12
_MAX_RELATED_FILES = 10


# A location outside every known root keeps this many trailing folders.
_FALLBACK_LOCATION_PARTS = 4
LOCATION_PAGE_NUMBER = 0
# The lines of a location page, as location_text writes them.
_LOCATION_PAGE_RE = re.compile(r"Location: (?P<location>.*)\nFolder: .*\nFile: (?P<file>.*)")
_LOCATION_LINE_LABELS = ("location:", "folder:", "file:")


def relative_location(source_uri: str, roots: Iterable[str]) -> str:
    """A file's path relative to its source root: folders and file name.

    The root is the local input folder, a manifest root, or a bucket or stage
    prefix. A file outside every root keeps its last few folders.
    """

    path = _path(source_uri).replace("\\", "/")
    for root in roots:
        prefix = _path(root).replace("\\", "/").rstrip("/")
        if prefix and path.startswith(prefix + "/"):
            return path[len(prefix) + 1 :]
    return "/".join(PurePosixPath(path).parts[-_FALLBACK_LOCATION_PARTS:])


def source_roots(settings: Settings) -> list[str]:
    """Every root a source file's location may be relative to."""

    files = settings.files
    roots = [str(files.input_path), files.stage_prefix or ""]
    if files.manifest_root is not None:
        roots.append(str(files.manifest_root))
    if files.manifest_path is not None:
        roots.append(str(files.manifest_path.parent))
    # An S3 bucket is its URI's host, an Azure container the first folder
    # of the path; only the prefix after them is a folder of the source.
    if settings.s3.bucket and settings.s3.prefix:
        roots.append(f"s3://{settings.s3.bucket}/{settings.s3.prefix}")
    if settings.azure_blob.container:
        container = settings.azure_blob.container
        prefix = settings.azure_blob.prefix or ""
        roots.append(f"azblob://account/{container}/{prefix}".rstrip("/"))
    roots = [root for root in roots if root]
    # The resolved form too, so a relative input path matches absolute URIs.
    return [*roots, *(str(Path(root).resolve()) for root in roots if "://" not in root)]


def location_text(location: str) -> str:
    """The text of a document's location page."""

    folder, _, name = location.rpartition("/")
    return f"Location: {location}\nFolder: {folder or '.'}\nFile: {name}"


def location_page_name(name: str, chunk: Chunk) -> bool:
    """Whether a name read off a location page is where the document sits.

    The file is the document itself, which has a node of its own: its file
    name, a folder path, and a line of the page ("File: report.pdf") name no
    other thing. A folder or file name that names a project or a batch is
    a name, not a path, and is not caught here.
    """

    if chunk.page_number != LOCATION_PAGE_NUMBER:
        return False
    page = _LOCATION_PAGE_RE.fullmatch(chunk.content.strip())
    if page is None:
        return False
    text = " ".join(name.split())
    if text.casefold().startswith(_LOCATION_LINE_LABELS):
        return True
    text = text.replace("\\", "/")
    # A path the model shortened is still a path: ".../Data integrity".
    path = "/" in text
    text = text.strip(" ./…").casefold()
    if text == page.group("file").strip().casefold():
        return True
    return path and bool(text) and text in page.group("location").casefold()


def content_kind(mime_type: str | None, source_uri: str) -> ContentKind:
    """Classify a file by its type: a dataset, an image, or a document."""

    suffix = PurePosixPath(_path(source_uri)).suffix.lower()
    mime = (mime_type or "").lower()
    if suffix in _DATASET_SUFFIXES or mime in {"text/csv", "text/tab-separated-values"}:
        return "dataset"
    if suffix in _IMAGE_SUFFIXES or mime.startswith("image/"):
        return "image"
    return "document"


def dataset_windows(
    profile: Mapping[str, Any] | None, *, catalogue_relations: bool
) -> tuple[bool, bool]:
    """Whether a dataset with this profile gets entity and relation extraction.

    An unprofiled file - including a dataset whose profile failed - is
    extracted in full, as any document would be. A catalogue's rows state
    relations through its column relations; without one, or with
    ``catalogue_relations`` off, it gets its entities only.
    """

    kind = str((profile or {}).get("kind") or "")
    entities, relations = _DATASET_WINDOWS.get(kind, (True, True))
    if kind == "catalogue":
        relations = catalogue_relations and any(
            column.role == "relation" for column in profile_columns(profile)
        )
    return entities, relations


def dataset_profiles(trace: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The dataset profile recorded for each document, by document id."""

    return {
        str(event["document_id"]): event
        for event in trace
        if event.get("stage") == DATASET_PROFILE_STAGE and event.get("document_id")
    }


def window_filter(
    trace: Iterable[dict[str, Any]], *, catalogue_relations: bool
) -> Callable[[str], tuple[bool, bool]]:
    """Map a document id to the extraction passes its profile allows."""

    profiles = dataset_profiles(trace)
    return lambda document_id: dataset_windows(
        profiles.get(document_id), catalogue_relations=catalogue_relations
    )


def profile_columns(profile: Mapping[str, Any] | None) -> list[ColumnRelation]:
    """The column relations a catalogue's profile kept; none for any other kind."""

    if not profile or profile.get("kind") != "catalogue":
        return []
    return [ColumnRelation.model_validate(item) for item in profile.get("columns") or []]


def dataset_columns(trace: Iterable[dict[str, Any]]) -> dict[str, list[ColumnRelation]]:
    """The column relations of each profiled catalogue, by document id."""

    return {
        document_id: columns
        for document_id, profile in dataset_profiles(trace).items()
        if (columns := profile_columns(profile))
    }


def catalogue_columns(
    proposed: Iterable[Any], ontology: OntologyProfile
) -> tuple[list[ColumnRelation], list[dict[str, str]]]:
    """Keep the column relations a profile proposed that the ontology can hold.

    One column names each row's subject, of an ontology entity type. A relation
    column needs an ontology entity type and a relation whose type rules admit
    the subject's type and the column's, in either direction; the direction
    that fits is recorded. A value column may name the relation column whose
    relation carries its value as a quantity. Every column turned away is
    returned with its reason, and without a subject no column is kept. A
    column the profile marked "none" states nothing and is not listed.
    """

    items, rejected = _proposed_columns(proposed)
    entity_types = set(ontology.entity_type_names())
    subject = next(
        (
            ColumnRelation(column=column, role="subject", entity_type=str(item["entity_type"]))
            for column, item in items
            if item["role"] == "subject" and item.get("entity_type") in entity_types
        ),
        None,
    )
    kept: dict[str, ColumnRelation] = {}
    # Values last: a value column names the relation column that carries it.
    for column, item in sorted(items, key=lambda entry: entry[1]["role"] == "value"):
        role = item["role"]
        outcome: ColumnRelation | str
        if subject is None:
            outcome = "unknown_entity_type" if role == "subject" else "no_subject"
        elif role == "subject":
            outcome = (
                subject
                if column == subject.column
                else "second_subject"
                if item.get("entity_type") in entity_types
                else "unknown_entity_type"
            )
        elif role == "alias":
            outcome = ColumnRelation(column=column, role="alias")
        elif role == "relation":
            outcome = _relation_column(column, item, subject, ontology)
        else:
            outcome = _value_column(column, item, kept, ontology)
        if isinstance(outcome, str):
            rejected.append(_rejected_column(item, outcome))
        else:
            kept[column] = outcome
    # In the header's order, the order a row gives its cells in.
    return [kept[column] for column, _item in items if column in kept], rejected


def _proposed_columns(
    proposed: Iterable[Any],
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[dict[str, str]]]:
    """Each proposed column once, by its header, and those that name no column."""

    items: list[tuple[str, Mapping[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for item in proposed:
        if not isinstance(item, Mapping) or item.get("role") == "none":
            continue
        column = _header(item.get("column"))
        if item.get("role") not in COLUMN_ROLES:
            rejected.append(_rejected_column(item, "unknown_role"))
        elif not column:
            rejected.append(_rejected_column(item, "blank_column"))
        elif any(column.casefold() == other.casefold() for other, _item in items):
            rejected.append(_rejected_column(item, "duplicate_column"))
        else:
            items.append((column, item))
    return items, rejected


def _relation_column(
    column: str,
    item: Mapping[str, Any],
    subject: ColumnRelation,
    ontology: OntologyProfile,
) -> ColumnRelation | str:
    """A relation column the ontology admits, or why it does not."""

    entity_type = str(item.get("entity_type") or "")
    subject_type = str(subject.entity_type)
    definition = ontology.relation(str(item.get("relation_type") or ""))
    if entity_type not in ontology.entity_type_names():
        return "unknown_entity_type"
    if definition is None:
        return "unknown_relation"
    forward = definition.admits(subject_type, entity_type)
    if not forward and not definition.admits(entity_type, subject_type):
        return "relation_type_rules"
    return ColumnRelation(
        column=column,
        role="relation",
        entity_type=entity_type,
        relation_type=definition.name,
        source="subject" if forward else "column",
    )


def _value_column(
    column: str,
    item: Mapping[str, Any],
    kept: Mapping[str, ColumnRelation],
    ontology: OntologyProfile,
) -> ColumnRelation | str:
    """A value column, of the relation column that carries it if it names one."""

    of_column = _header(item.get("of_column"))
    if not of_column:
        return ColumnRelation(column=column, role="value")
    carrier = next(
        (
            relation
            for relation in kept.values()
            if relation.role == "relation" and relation.column.casefold() == of_column.casefold()
        ),
        None,
    )
    if carrier is None:
        return "value_of_unmapped_column"
    carried = ontology.relation(str(carrier.relation_type))
    if carried is None or not carried.quantities:
        return "relation_carries_no_quantities"
    return ColumnRelation(column=column, role="value", of_column=carrier.column)


def _header(value: object) -> str:
    return " ".join(str(value or "").split())


def _rejected_column(item: Mapping[str, Any], reason: str) -> dict[str, str]:
    record = {
        "column": str(item.get("column") or ""),
        "role": str(item.get("role") or ""),
        "reason": reason,
    }
    if item.get("relation_type"):
        record["relation_type"] = str(item["relation_type"])
    return record


def annotate_documents(
    rows: list[dict[str, Any]],
    trace: Iterable[dict[str, Any]],
    provenance: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fill each document row's summary, related files and provenance.

    The summary comes from a dataset's profile. Related files are those in the
    same folder that share an identifier token with the file (a batch or order
    number in both names), or every file of a small folder, so an image or a
    data file can be found from the documents beside it. Provenance says how
    this version of the graph came by the document's extraction: ``extracted``
    unless a revision reused, revalidated or re-extracted it.
    """

    profiles = dataset_profiles(trace)
    related = related_file_ids(rows)
    labels = provenance or {}
    return [
        {
            **row,
            "summary": profiles.get(str(row.get("file_id")), {}).get("summary")
            or row.get("summary"),
            "related_file_ids": related.get(str(row.get("file_id")), []),
            "provenance": labels.get(str(row.get("file_id")))
            or row.get("provenance")
            or "extracted",
        }
        for row in rows
    ]


def related_file_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Relate datasets and images to the documents beside them.

    An image in a subfolder of its own (an experiment's "Pictures") belongs
    with the documents one level up when its own folder holds none.
    """

    by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_folder[_folder(row)].append(row)
    related: dict[str, list[str]] = {}
    for folder, folder_rows in by_folder.items():
        for row in folder_rows:
            if row.get("kind") not in {"dataset", "image"}:
                continue
            file_id = str(row.get("file_id"))
            others = [other for other in folder_rows if str(other.get("file_id")) != file_id]
            if not any(other.get("kind") == "document" for other in others):
                others += by_folder.get(str(PurePosixPath(folder).parent), [])
            tokens = _identifier_tokens(row)
            sharing = [other for other in others if tokens & _identifier_tokens(other)]
            chosen = sharing or (others if len(others) < _SMALL_FOLDER else [])
            related[file_id] = sorted(str(other.get("file_id")) for other in chosen)[
                :_MAX_RELATED_FILES
            ]
    return related


def _folder(row: dict[str, Any]) -> str:
    folder = row.get("folder")
    if folder is not None:
        return str(folder)
    return str(PurePosixPath(_path(str(row.get("source_uri", "")))).parent)


def _identifier_tokens(row: dict[str, Any]) -> set[str]:
    name = PurePosixPath(_path(str(row.get("source_uri", "")))).stem
    return {
        token.casefold()
        for token in _IDENTIFIER_TOKEN_RE.findall(name)
        if len(token) >= _MIN_IDENTIFIER_TOKEN and any(character.isdigit() for character in token)
    }


def _path(source_uri: str) -> str:
    parsed = urlparse(source_uri)
    return unquote(parsed.path) if parsed.scheme else source_uri
