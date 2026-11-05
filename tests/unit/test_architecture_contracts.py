from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path("src/kg_processor")
APPLICATION_SNOWFLAKE_MODULES = {
    "kg_processor.application.snowflake_access",
    "kg_processor.application.snowflake_deployment",
    "kg_processor.application.snowflake_export",
}
# Shared Snowflake transports those modules may use: the connector helpers, and
# the Cortex REST client the access check probes model access through. Neither
# is a provider adapter; both are what the Snowflake adapters themselves build on.
SNOWFLAKE_TRANSPORT_HELPERS = {
    "kg_processor.adapters.snowflake",
    "kg_processor.adapters.snowflake_cortex_rest",
}
CONCRETE_PROVIDER_FAMILIES = (
    "kg_processor.adapters.ocr",
    "kg_processor.adapters.llm",
    "kg_processor.adapters.embeddings",
    "kg_processor.adapters.writers",
)


def test_domain_and_ports_stay_provider_independent() -> None:
    imports_by_module = _production_imports_by_module()
    forbidden_for_domain = (
        "kg_processor.adapters",
        "kg_processor.application",
        "kg_processor.config",
        "kg_processor.factories",
        "kg_processor.ports",
        "kg_processor.cli",
    )
    forbidden_for_ports = (
        "kg_processor.adapters",
        "kg_processor.application",
        "kg_processor.config",
        "kg_processor.factories",
        "kg_processor.cli",
    )

    violations: list[str] = []
    for module_name, imports in imports_by_module.items():
        if module_name.startswith("kg_processor.domain."):
            violations.extend(_forbidden_imports(module_name, imports, forbidden_for_domain))
        if module_name.startswith("kg_processor.ports."):
            violations.extend(_forbidden_imports(module_name, imports, forbidden_for_ports))

    assert violations == []


def test_application_graph_core_does_not_import_provider_adapters() -> None:
    imports_by_module = _production_imports_by_module()
    violations: list[str] = []

    for module_name, imports in imports_by_module.items():
        if not module_name.startswith("kg_processor.application."):
            continue
        for imported in imports:
            if not imported.startswith("kg_processor.adapters"):
                continue
            allowed_snowflake_helper = (
                module_name in APPLICATION_SNOWFLAKE_MODULES
                and imported in SNOWFLAKE_TRANSPORT_HELPERS
            )
            if not allowed_snowflake_helper:
                violations.append(f"{module_name} imports {imported}")

    assert violations == []


def test_concrete_provider_families_are_wired_only_in_adapters_or_factories() -> None:
    imports_by_module = _production_imports_by_module()
    violations: list[str] = []

    for module_name, imports in imports_by_module.items():
        if module_name == "kg_processor.factories" or module_name.startswith(
            "kg_processor.adapters."
        ):
            continue
        for imported in imports:
            if imported.startswith(CONCRETE_PROVIDER_FAMILIES):
                violations.append(f"{module_name} imports {imported}")

    assert violations == []


def _production_imports_by_module() -> dict[str, set[str]]:
    return {
        _module_name(path): _kg_processor_imports(path)
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
    }


def _module_name(path: Path) -> str:
    relative = path.relative_to("src").with_suffix("")
    return ".".join(relative.parts)


def _kg_processor_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("kg_processor")
            )
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("kg_processor")
        ):
            imports.add(node.module)
    return imports


def _forbidden_imports(
    module_name: str,
    imports: set[str],
    forbidden_prefixes: tuple[str, ...],
) -> list[str]:
    return [
        f"{module_name} imports {imported}"
        for imported in imports
        if imported.startswith(forbidden_prefixes)
    ]
