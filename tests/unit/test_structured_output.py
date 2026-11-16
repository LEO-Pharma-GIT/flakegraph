from __future__ import annotations

from pydantic import BaseModel

from kg_processor.application.structured_output import response_schema_name, strict_json_schema


class _NestedRecord(BaseModel):
    """Provide a nested response object used to verify recursive schema closure.

    Its single field must become required.
    """

    name: str


class _Response(BaseModel):
    """Provide a top-level response containing nested records for strict-schema tests.

    Both schema levels should reject extra properties.
    """

    records: list[_NestedRecord]


class _DefaultedResponse(BaseModel):
    """Expose a Pydantic default that strict provider schemas must remove."""

    label: str = "default label"


def test_strict_schema_closes_every_object_and_requires_all_fields() -> None:
    """Require strict schema conversion to close and fully require every nested object."""

    schema = strict_json_schema(_Response.model_json_schema())

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["records"]
    nested = schema["$defs"]["_NestedRecord"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["name"]


def test_response_schema_name_is_provider_safe_and_bounded() -> None:
    """Ensure task names become bounded, provider-safe schema identifiers.

    Spaces, slashes, and excessive length must be normalized.
    """

    name = response_schema_name("entity extraction / pass 1 " + "x" * 100)

    assert "/" not in name
    assert " " not in name
    assert len(name) == 64


def test_strict_schema_removes_provider_unsupported_defaults() -> None:
    """Keep default behavior in Pydantic without emitting unsupported schema keywords."""

    schema = strict_json_schema(_DefaultedResponse.model_json_schema())

    assert "default" not in schema["properties"]["label"]
    assert schema["required"] == ["label"]
