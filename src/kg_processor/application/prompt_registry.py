# SPDX-License-Identifier: Apache-2.0
"""Versioned prompt loading and prompt fingerprint helpers.

Prompt text is treated like source code: every trace records the prompt revision
and checksum, and extraction cache keys include prompt fingerprints so prompt
edits cannot reuse stale graph output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from string import Template
from typing import Any

from kg_processor.ports.llm import CommunitySummaryRequest, DescriptionMergeRequest

TWO_PASS_PROMPT_REVISION = "two-pass-extraction-v42"
EXTRACTION_SYSTEM_PROMPTS = (
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
)

PROMPT_REVISIONS = {
    "document_context_extraction": TWO_PASS_PROMPT_REVISION,
    "entity_extraction": TWO_PASS_PROMPT_REVISION,
    "relation_extraction": TWO_PASS_PROMPT_REVISION,
    "relation_verification": TWO_PASS_PROMPT_REVISION,
    "relation_grounding": TWO_PASS_PROMPT_REVISION,
    "relation_relabel": TWO_PASS_PROMPT_REVISION,
    "entity_grounding": TWO_PASS_PROMPT_REVISION,
    "dataset_profile": TWO_PASS_PROMPT_REVISION,
    "folder_entity": TWO_PASS_PROMPT_REVISION,
    "entity_resolution": TWO_PASS_PROMPT_REVISION,
    "extraction_input": TWO_PASS_PROMPT_REVISION,
    "structured_output_repair": "structured-output-repair-v1",
    "entity_description_merge": "entity-description-merge-v1",
    "entity_description_batch_merge": "entity-description-batch-merge-v1",
    "community_report": "community-report-v2",
    "community_report_batch": "community-report-batch-v2",
}


@dataclass(frozen=True)
class PromptTemplate:
    """Loaded prompt text plus revision and checksum metadata."""

    name: str
    revision: str
    text: str
    checksum: str


@dataclass(frozen=True)
class GraphPrompt:
    """Provider-ready system/user prompt pair with template provenance."""

    system: str
    user: str
    template: PromptTemplate


@cache
def get_prompt_template(name: str) -> PromptTemplate:
    """Load and cache a named immutable prompt template and its checksum."""

    if name not in PROMPT_REVISIONS:
        raise ValueError(f"Unknown prompt template: {name}")
    text = resources.files("kg_processor.prompts").joinpath(f"{name}.md").read_text().strip()
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(
        name=name,
        revision=PROMPT_REVISIONS[name],
        text=text,
        checksum=checksum,
    )


def render_prompt_template(name: str, **variables: str) -> PromptTemplate:
    """Load and render a text template while retaining reproducibility metadata.

    Templates use :class:`string.Template` placeholders so embedded JSON braces
    remain ordinary data. Missing or misspelled variables fail immediately rather
    than sending an unresolved instruction to an LLM provider.
    """

    template = get_prompt_template(name)
    rendered = Template(template.text).substitute(variables)
    return PromptTemplate(
        name=template.name,
        revision=template.revision,
        text=rendered,
        checksum=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    )


def extraction_prompt(name: str, payload: dict[str, Any]) -> GraphPrompt:
    """Build one extraction prompt from registered instructions and JSON input.

    The system contract and common user envelope both live in the prompt package;
    request builders supply only typed machine data. This makes all natural-language
    model behavior discoverable without searching provider or application modules.

    The payload is serialized in the order the builder wrote it, not sorted.
    A prefix cache covers a prompt only as far as its first differing byte, so
    the parts that repeat across every window - the ontology - belong before
    the parts that do not. Determinism still holds: the builders construct
    these mappings in a fixed order, so the same window renders the same bytes.
    """

    if name not in EXTRACTION_SYSTEM_PROMPTS:
        raise ValueError(f"Unknown extraction prompt: {name}")
    system = get_prompt_template(name)
    user = render_prompt_template(
        "extraction_input",
        input_json=json.dumps(payload, ensure_ascii=False, sort_keys=False),
    )
    return GraphPrompt(system=system.text, user=user.text, template=system)


def structured_output_repair_system(system: str) -> str:
    """Append the registered schema-repair instruction to a system contract."""

    repair = get_prompt_template("structured_output_repair")
    return f"{system}\n\n{repair.text}"


def two_pass_prompt_fingerprints() -> dict[str, str]:
    """Return cache fingerprints for every two-pass extraction instruction.

    Both the shared input envelope and repair behavior are included because either
    can change provider output even when the ontology and source chunks are stable.
    """

    fingerprints = {name: get_prompt_template(name).checksum for name in EXTRACTION_SYSTEM_PROMPTS}
    fingerprints["input"] = get_prompt_template("extraction_input").checksum
    fingerprints["repair"] = get_prompt_template("structured_output_repair").checksum
    return {"revision": TWO_PASS_PROMPT_REVISION, **fingerprints}


def community_report_prompt(request: CommunitySummaryRequest) -> GraphPrompt:
    """Build the prompt that turns a detected community into report rows."""

    template = get_prompt_template("community_report")
    user = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def entity_description_merge_prompt(request: DescriptionMergeRequest) -> GraphPrompt:
    """Build the prompt that synthesizes repeated entity descriptions."""

    template = get_prompt_template("entity_description_merge")
    user = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def entity_description_batch_merge_prompt(records: list[dict[str, object]]) -> GraphPrompt:
    """Build a prompt for several independent description merges in one request."""

    template = get_prompt_template("entity_description_batch_merge")
    user = json.dumps({"records": records}, sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def community_report_batch_prompt(records: list[dict[str, object]]) -> GraphPrompt:
    """Build a prompt for several independent community reports in one request."""

    template = get_prompt_template("community_report_batch")
    user = json.dumps({"records": records}, sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def prompt_metadata(prompt: GraphPrompt) -> dict[str, str]:
    """Return prompt metadata persisted in provider traces."""

    return {
        "prompt_name": prompt.template.name,
        "prompt_revision": prompt.template.revision,
        "prompt_checksum": prompt.template.checksum,
    }
