from __future__ import annotations

from importlib import resources

import pytest

from kg_processor.application.prompt_registry import (
    PROMPT_REVISIONS,
    community_report_prompt,
    entity_description_merge_prompt,
    extraction_prompt,
    get_prompt_template,
    prompt_metadata,
    render_prompt_template,
    structured_output_repair_system,
    two_pass_prompt_fingerprints,
)
from kg_processor.ports.llm import CommunitySummaryRequest, DescriptionMergeRequest


def test_prompt_registry_loads_every_declared_template() -> None:
    """Every declared prompt must resolve to packaged text and stable metadata."""

    for name, revision in PROMPT_REVISIONS.items():
        template = get_prompt_template(name)
        assert template.text
        assert template.revision == revision
        assert len(template.checksum) == 64


def test_prompt_registry_declares_every_packaged_markdown_file() -> None:
    """Prevent an instruction file from bypassing registry versioning and tracing."""

    packaged = {
        item.name.removesuffix(".md")
        for item in resources.files("kg_processor.prompts").iterdir()
        if item.name.endswith(".md")
    }

    assert packaged == set(PROMPT_REVISIONS)


def test_prompt_registry_rejects_unknown_templates() -> None:
    """Unknown names should fail before a provider request is constructed."""

    with pytest.raises(ValueError, match="Unknown prompt template"):
        get_prompt_template("missing_prompt")


def test_prompt_rendering_rejects_unknown_tasks_and_missing_variables() -> None:
    """Fail configuration mistakes before malformed instructions reach a provider."""

    with pytest.raises(ValueError, match="Unknown extraction prompt"):
        extraction_prompt("community_report", {})
    with pytest.raises(KeyError, match="input_json"):
        render_prompt_template("extraction_input")


def test_extraction_prompt_renders_registered_system_and_input_templates() -> None:
    """Extraction builders should contribute data, never inline instruction text."""

    prompt = extraction_prompt("entity_extraction", {"window_id": "window-1"})

    assert prompt.system == get_prompt_template("entity_extraction").text
    assert prompt.user.startswith("Follow the extraction contract exactly.\nINPUT_JSON:\n")
    assert prompt.user.endswith('{"window_id": "window-1"}')
    assert prompt.template.revision == PROMPT_REVISIONS["entity_extraction"]


def test_extraction_fingerprints_cover_shared_and_task_specific_prompts() -> None:
    """Cache provenance must change for any extraction or repair instruction edit."""

    fingerprints = two_pass_prompt_fingerprints()

    assert set(fingerprints) == {
        "revision",
        "document_context_extraction",
        "entity_extraction",
        "relation_extraction",
        "relation_verification",
        "relation_grounding",
        "relation_relabel",
        "entity_grounding",
        "dataset_profile",
        "folder_entity",
        "entity_resolution",
        "input",
        "repair",
    }
    assert all(len(value) == 64 for key, value in fingerprints.items() if key != "revision")


def test_structured_repair_uses_registered_instruction() -> None:
    """Provider retries should append the centrally managed repair contract."""

    repaired = structured_output_repair_system("base contract")

    assert repaired == ("base contract\n\n" + get_prompt_template("structured_output_repair").text)


def test_community_report_prompt_serializes_request_and_metadata() -> None:
    """Community prompts should carry their request and reproducibility metadata."""

    request = CommunitySummaryRequest(
        title_seed="Acme",
        members=["Alice", "Acme"],
        relations=["Alice WORKS_AT Acme"],
        evidence_quotes=["Alice works at Acme."],
    )

    prompt = community_report_prompt(request)

    assert '"title_seed": "Acme"' in prompt.user
    assert '"evidence_quotes": ["Alice works at Acme."]' in prompt.user
    assert prompt_metadata(prompt)["prompt_name"] == "community_report"


def test_description_merge_prompt_serializes_request_and_metadata() -> None:
    """Description prompts should expose all observations to the configured model."""

    request = DescriptionMergeRequest(
        entity_name="Alice",
        entity_type="PERSON",
        descriptions=["Alice is present.", "Alice works at Acme."],
        evidence=["Alice works at Acme."],
    )

    prompt = entity_description_merge_prompt(request)

    assert '"entity_name": "Alice"' in prompt.user
    assert '"descriptions": ["Alice is present.", "Alice works at Acme."]' in prompt.user
    assert prompt_metadata(prompt)["prompt_name"] == "entity_description_merge"
