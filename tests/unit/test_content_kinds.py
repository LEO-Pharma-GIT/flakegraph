from __future__ import annotations

from typing import Any

from documents import chunk

from kg_processor.application.content_kinds import (
    DATASET_PROFILE_STAGE,
    LOCATION_PAGE_NUMBER,
    annotate_documents,
    content_kind,
    dataset_windows,
    location_page_name,
    location_text,
    related_file_ids,
    relative_location,
    window_filter,
)


def test_files_are_classified_by_type() -> None:
    """Spreadsheets and tables are datasets, pictures are images, the rest documents."""

    assert content_kind("application/vnd.ms-excel", "s3://bucket/plan.XLSX") == "dataset"
    assert content_kind("text/csv", "file:///data/rows.txt") == "dataset"
    assert content_kind("image/tiff", "file:///lab/Batch%209.tif") == "image"
    assert content_kind("application/pdf", "file:///docs/report.pdf") == "document"


def test_a_dataset_profile_decides_its_extraction_passes() -> None:
    """Measurements and mapped catalogues get relation windows; every sheet keeps its entities."""

    columns = [
        {"column": "Material", "role": "subject", "entity_type": "MATERIAL"},
        {
            "column": "Supplier",
            "role": "relation",
            "entity_type": "ORGANIZATION",
            "relation_type": "SUPPLIED_BY",
            "source": "subject",
        },
    ]
    trace: list[dict[str, Any]] = [
        {"stage": DATASET_PROFILE_STAGE, "document_id": "list", "kind": "catalogue"},
        {
            "stage": DATASET_PROFILE_STAGE,
            "document_id": "mapped",
            "kind": "catalogue",
            "columns": columns,
        },
        {"stage": DATASET_PROFILE_STAGE, "document_id": "calc", "kind": "other"},
        {"stage": DATASET_PROFILE_STAGE, "document_id": "assay", "kind": "measurements"},
    ]
    passes = window_filter(trace, catalogue_relations=True)

    assert passes("list") == (True, False)
    assert passes("mapped") == (True, True)
    assert passes("calc") == (True, False)
    assert passes("assay") == (True, True)
    assert passes("report") == (True, True)
    assert window_filter(trace, catalogue_relations=False)("mapped") == (True, False)
    assert dataset_windows(None, catalogue_relations=True) == (True, True)


def test_images_and_datasets_are_related_to_the_documents_beside_them() -> None:
    """Shared identifiers in file names relate files; a small folder relates them all."""

    rows: list[dict[str, Any]] = [
        {"file_id": "coa", "kind": "document", "source_uri": "file:///lot/LT2019-0420%20CoA.pdf"},
        {"file_id": "img", "kind": "image", "source_uri": "file:///lot/LT2019-0420%20photo.jpg"},
        {"file_id": "other", "kind": "document", "source_uri": "file:///lot/unrelated%20memo.pdf"},
        {"file_id": "sheet", "kind": "dataset", "source_uri": "file:///elsewhere/data.xlsx"},
        {"file_id": "near", "kind": "document", "source_uri": "file:///elsewhere/notes.pdf"},
    ]

    related = related_file_ids(rows)

    assert related["img"] == ["coa"]
    assert related["sheet"] == ["near"]
    assert "coa" not in related


def test_documents_get_their_dataset_summary_and_related_files() -> None:
    """Annotation fills what extraction learnt without dropping other columns."""

    rows: list[dict[str, Any]] = [
        {"file_id": "sheet", "kind": "dataset", "source_uri": "file:///d/a.xlsx", "size_bytes": 3},
        {"file_id": "doc", "kind": "document", "source_uri": "file:///d/b.pdf", "summary": None},
    ]
    trace: list[dict[str, Any]] = [
        {"stage": DATASET_PROFILE_STAGE, "document_id": "sheet", "summary": "Excipient list."}
    ]

    annotated = {row["file_id"]: row for row in annotate_documents(rows, trace)}

    assert annotated["sheet"]["summary"] == "Excipient list."
    assert annotated["sheet"]["related_file_ids"] == ["doc"]
    assert annotated["sheet"]["size_bytes"] == 3
    assert annotated["doc"]["summary"] is None


def test_a_location_is_the_path_below_its_source_root() -> None:
    """Folders under the root say where a file belongs; elsewhere its last folders do."""

    uri = "file:///data/uploads/run1/Lab%20A/Trial%203/Pictures/IMG_0392.jpg"
    assert relative_location(uri, ["/data/uploads/run1"]) == "Lab A/Trial 3/Pictures/IMG_0392.jpg"
    assert relative_location("s3://bucket/corpus/a/b.pdf", ["s3://bucket/corpus"]) == "a/b.pdf"
    assert relative_location("file:///x/y/z/w/v/file.pdf", ["/other"]) == "z/w/v/file.pdf"


def test_an_image_in_its_own_subfolder_is_related_to_the_documents_above_it() -> None:
    """A Pictures folder with no documents belongs to its parent folder's documents."""

    rows: list[dict[str, Any]] = [
        {"file_id": "spec", "kind": "document", "folder": "Lab/Coater", "source_uri": "a"},
        {"file_id": "photo", "kind": "image", "folder": "Lab/Coater/Pictures", "source_uri": "b"},
        {"file_id": "far", "kind": "document", "folder": "Other", "source_uri": "c"},
    ]

    assert related_file_ids(rows)["photo"] == ["spec"]


def test_a_location_page_name_is_where_the_document_sits_not_what_a_folder_names() -> None:
    """The file name, a folder path, and a page line are refused; a folder's name is not."""

    page = chunk(
        location_text("Lab A/Trial 3/Batch 42 report.pdf"),
        page_number=LOCATION_PAGE_NUMBER,
    )

    for name in (
        "Batch 42 report.pdf",
        "File: Batch 42 report.pdf",
        "Folder: Lab A/Trial 3",
        "Lab A/Trial 3",
        ".../Trial 3",
    ):
        assert location_page_name(name, page), name
    for name in ("Trial 3", "Lab A", "Batch 42", "Batch 42 report"):
        assert not location_page_name(name, page), name
    # The same words on a page of the document itself are its text.
    assert not location_page_name("Batch 42 report.pdf", chunk("See Batch 42 report.pdf."))
