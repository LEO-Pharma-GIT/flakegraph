# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral helpers for strict JSON-schema completions."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

_NON_IDENTIFIER = re.compile(r"[^a-zA-Z0-9_-]+")


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied schema with every object closed and fully required.

    Provider strict modes commonly require both constraints at every nested object,
    including definitions referenced through ``$ref``.
    """

    normalized = deepcopy(schema)
    _close_objects(normalized)
    return normalized


def response_schema_name(task_name: str) -> str:
    """Convert a trace-friendly task name into a bounded API-safe schema identifier.

    Empty or punctuation-only task names receive a stable descriptive fallback.
    """

    cleaned = _NON_IDENTIFIER.sub("_", task_name).strip("_")
    return (cleaned or "structured_response")[:64]


def _close_objects(value: object) -> None:
    """Recursively apply strict object requirements to dictionaries and array children.

    Every declared property becomes required and undeclared properties are forbidden.
    """

    if isinstance(value, dict):
        # Pydantic emits ``default`` for optional/defaulted fields, but strict
        # structured-output schema subsets reject that annotation. Runtime
        # defaults still apply when the response model validates provider output.
        value.pop("default", None)
        if value.get("type") == "object" or isinstance(value.get("properties"), dict):
            properties = value.get("properties", {})
            if isinstance(properties, dict):
                value["additionalProperties"] = False
                value["required"] = list(properties)
        for child in value.values():
            _close_objects(child)
    elif isinstance(value, list):
        for child in value:
            _close_objects(child)
