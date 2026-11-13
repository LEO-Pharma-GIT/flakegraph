"""Keep request-specific ids out of the schemas a provider compiles.

A provider that enforces a JSON schema by compiling it into a grammar caches one
grammar per distinct schema and compiles only so many a minute; past that it
refuses the request. A caller that enumerates the ids of the text it sent - the
chunks of a window, the mentions a relation may join - sends a schema no earlier
request had, so a busy caller spends that allowance within seconds.

For a model whose gateway entry sets ``model_info.stable_schemas``, every enum
whose values are all opaque ids is removed before the request leaves, and what
remains depends only on the caller's vocabulary: entity and relation types stay
enumerated, so their grammar is compiled once and reused. The caller checks the
ids in a reply against what it sent either way, so an id the grammar no longer
rules out is turned away there. Entries without the setting pass untouched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from litellm.integrations.custom_logger import CustomLogger

# A generated identifier: a lowercase prefix, then a long hexadecimal digest.
OPAQUE_ID = re.compile(r"^[a-z][a-z0-9_]*_[0-9a-f]{12,}$")
COMPLETION_CALLS = frozenset({"completion", "acompletion"})
CONFIG = Path(__file__).with_name("config.yaml")


def stable_schema_models(config: dict[str, Any]) -> frozenset[str]:
    """Return the aliases whose gateway entry asks for stable schemas."""

    return frozenset(
        str(entry["model_name"])
        for entry in config.get("model_list") or []
        if isinstance(entry, dict) and (entry.get("model_info") or {}).get("stable_schemas")
    )


def remove_id_enums(node: object) -> None:
    """Drop, in place, every enum in a JSON schema whose values are all opaque ids."""

    if isinstance(node, dict):
        values = node.get("enum")
        if (
            isinstance(values, list)
            and values
            and all(isinstance(value, str) and OPAQUE_ID.match(value) for value in values)
        ):
            del node["enum"]
        for child in node.values():
            remove_id_enums(child)
    elif isinstance(node, list):
        for child in node:
            remove_id_enums(child)


class StableSchemas(CustomLogger):
    """Make a request's structured-output schema independent of the request."""

    def __init__(self, models: frozenset[str]) -> None:
        self.models = models

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Remove id enums from the schema of a completion bound for a listed model."""

        if call_type not in COMPLETION_CALLS or data.get("model") not in self.models:
            return data
        response_format = data.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            remove_id_enums(response_format.get("json_schema"))
        return data


instance = StableSchemas(stable_schema_models(yaml.safe_load(CONFIG.read_text()) or {}))
