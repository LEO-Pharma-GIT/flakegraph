from __future__ import annotations

import re
import tomllib
from pathlib import Path

import kg_processor

_SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0"


def test_readme_keeps_quick_start_paths_current_and_direct() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "TODO" not in readme
    assert "# FlakeGraph" in readme
    assert "react/public/flakegraph-logo.png" in readme
    assert "docs/assets/flakegraph-pipeline.svg" in readme
    assert "docs/algorithm.md" in readme
    assert "turns documents into evidence-backed knowledge graphs" in readme
    assert "provider interfaces" in readme
    # The quick start must be the no-GPU path, in the order it is run.
    assert "Python 3.12 or newer" in readme
    quick_start = [
        "ollama pull qwen3:8b",
        "uv sync --extra local-embeddings",
        "uv run flakegraph worker --config configs/app-defaults.yaml",
        "uv run flakegraph inspect html --output out/app",
        "cd react && bun install && FLAKEGRAPH_DEMO_IDENTITIES=1 bun run dev",
    ]
    positions = [readme.index(step) for step in quick_start]
    assert positions == sorted(positions)
    assert 'uv tool install --python 3.13 "mineru[pipeline]==3.4.4"' in readme
    assert "https://docs.vllm.ai/en/latest/getting_started/installation.html" in readme
    assert "## Serving Your Own Model" in readme
    assert "## Deploy On Kubernetes" in readme
    assert "## Snowflake" in readme
    assert "The CLI remains available for" in readme
    # Hardware- and organisation-specific material belongs in deploy/, not here.
    assert "NVFP4" not in readme
    assert "GB10" not in readme
    assert "LEO" not in readme
    assert "flakegraph[leiden]" in readme

    required_paths = [
        "configs/app-defaults.yaml",
        "configs/local-vllm-mineru-oss.yaml",
        "configs/local-mineru-oss.yaml",
        "react/README.md",
        "configs/README.md",
        "configs/snowflake-cortex.yaml",
        "data/martial_arts/files/martial-arts-overview.pdf",
        "docs/architecture.md",
        "docs/algorithm.md",
        "docs/kubernetes-fleet.md",
        "docs/snowflake-setup.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
    ]
    for path in required_paths:
        assert path in readme
        assert Path(path).exists()


def test_license_files_carry_the_full_apache_text_and_a_notice() -> None:
    """A public repository must ship the licence itself, not only its appendix."""

    license_text = Path("LICENSE").read_text(encoding="utf-8")
    notice = Path("NOTICE").read_text(encoding="utf-8")
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    for section in range(1, 10):
        assert re.search(rf"^\s*{section}\. ", license_text, flags=re.M), f"section {section}"
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert notice.startswith("FlakeGraph\nCopyright 2026 LEO Pharma A/S")
    assert "THIRD_PARTY_NOTICES.md" in notice
    assert project["license"] == "Apache-2.0"
    assert "NOTICE" in project["license-files"]
    assert project["requires-python"] == ">=3.12"
    assert {"Homepage", "Repository", "Issues"} <= set(project["urls"])
    assert any(
        item.startswith("Programming Language :: Python :: 3.12") for item in project["classifiers"]
    )


def test_every_package_module_carries_an_spdx_header() -> None:
    """The licence identifier travels with each file, the way corporate OSS policies expect.

    The header is the first line, or the second when a shebang comes first, so
    a file copied out of the package still says what it is licensed under.
    """

    missing = []
    for path in sorted(Path("src/kg_processor").rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        head = lines[:2] if lines and lines[0].startswith("#!") else lines[:1]
        if _SPDX_HEADER not in head:
            missing.append(str(path))
    assert missing == []


def test_kubernetes_guides_are_vendor_neutral_and_cross_linked() -> None:
    """The deployment guide must read for any cluster; the fleet is a reference.

    Everything hardware-specific - the Blackwell checkpoint, the unified-memory
    numbers, the k3s bring-up, the drafter - lives in the reference document
    and the example values it points at, and neither names the organisation
    or network the chart was developed on.
    """

    guide = Path("docs/kubernetes-fleet.md").read_text(encoding="utf-8")
    reference = Path("docs/reference-dgx-spark-fleet.md").read_text(encoding="utf-8")
    chart_readme = Path("deploy/helm/flakegraph/README.md").read_text(encoding="utf-8")
    spark_readme = Path("deploy/spark/README.md").read_text(encoding="utf-8")

    assert "reference-dgx-spark-fleet.md" in guide
    assert "kubernetes-fleet.md" in reference
    assert "#verifying-a-fleet" in guide or "### Verifying a fleet" in guide
    for profile in (
        "deploy/examples/minimal-cpu-values.yaml",
        "deploy/examples/external-provider-values.yaml",
        "deploy/examples/dgx-spark-k3s-values.yaml",
    ):
        assert profile in guide or profile.rsplit("/", 1)[1] in guide, profile
        assert Path(profile).exists(), profile
    for hardware_specific in ("NVFP4", "119.2", "dflash", "gpuMemoryUtilization: 0.50"):
        assert hardware_specific not in guide, hardware_specific
        assert hardware_specific in reference or hardware_specific in Path(
            "deploy/examples/dgx-spark-k3s-values.yaml"
        ).read_text(encoding="utf-8"), hardware_specific
    assert "Secrets the chart expects" in chart_readme
    assert "reference profile, not a requirement" in spark_readme
    for text in (guide, reference, chart_readme, spark_readme):
        for identifier in ("10.217", "leo-spark", "platform team", "corporate"):
            assert identifier not in text.lower(), identifier


def test_config_readme_documents_every_yaml_profile() -> None:
    config_readme = Path("configs/README.md").read_text(encoding="utf-8")
    config_files = sorted(Path("configs").glob("*.yaml"))

    assert config_files
    for path in config_files:
        assert f"`{path.name}`" in config_readme


def test_explanatory_guides_keep_their_mermaid_visuals_well_formed() -> None:
    """Protect the diagrams that make the public guides easier to scan.

    Mermaid CLI rendering is performed during documentation review because it
    requires a browser. This focused contract catches missing visuals and
    unclosed Mermaid fences in the fast Python suite without pretending to
    duplicate Mermaid's parser.
    """

    public_markdown = [
        *Path(".").glob("*.md"),
        *Path("configs").rglob("*.md"),
        *Path("data").rglob("*.md"),
        *Path("docs").rglob("*.md"),
    ]
    visual_guides = [
        path
        for path in sorted(set(public_markdown))
        if "```mermaid" in path.read_text(encoding="utf-8")
    ]
    assert visual_guides, "public documentation should contain explanatory diagrams"

    for path in visual_guides:
        guide = path.read_text(encoding="utf-8")
        diagrams = re.findall(r"```mermaid\n(.*?)\n```", guide, flags=re.DOTALL)

        assert diagrams, f"{path} should contain at least one explanatory diagram"
        assert len(diagrams) == guide.count("```mermaid"), (
            f"{path} contains an unclosed Mermaid code fence"
        )
        assert all(
            diagram.lstrip().startswith(("flowchart ", "sequenceDiagram")) for diagram in diagrams
        ), f"{path} contains a Mermaid block without an explicit diagram type"
        reserved_identifiers = {
            "class",
            "click",
            "direction",
            "end",
            "flowchart",
            "graph",
            "linkstyle",
            "style",
            "subgraph",
        }
        declared_identifiers = {
            match.casefold()
            for diagram in diagrams
            for match in re.findall(
                r"(?m)^\s{4}([A-Za-z][A-Za-z0-9_]*)\s*(?=\[|\(|\{|-->|<-->)",
                diagram,
            )
        }
        assert declared_identifiers.isdisjoint(reserved_identifiers), (
            f"{path} uses a Mermaid grammar keyword as a node identifier: "
            f"{sorted(declared_identifiers & reserved_identifiers)}"
        )


def test_readme_pipeline_visual_is_a_self_contained_accessible_svg() -> None:
    """Keep the primary algorithm overview reviewable and GitHub-independent."""

    visual = Path("docs/assets/flakegraph-pipeline.svg").read_text(encoding="utf-8")

    assert visual.startswith("<svg")
    assert "<title>" in visual
    assert "<desc>" in visual
    assert "Documents in. Evidence-backed graph out." in visual
    assert "DOCUMENT-WIDE TWO-PHASE EXTRACTION" in visual
    assert "Document entity inventory barrier" in visual
    assert "CORPUS FINALIZATION" in visual
    assert "LOCAL" in visual
    assert "KUBERNETES" in visual
    assert "SNOWFLAKE" in visual
    assert "<image" not in visual
    assert re.findall(r"https?://[^\"\s]+", visual) == ["http://www.w3.org/2000/svg"]


def test_third_party_notices_cover_runtime_license_boundaries() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    required_phrases = [
        "FlakeGraph source code is licensed under the Apache License 2.0",
        "re-licenses",
        "mineru[pipeline]",
        "MinerU Open Source License",
        "additional terms",
        "opendatalab/PDF-Extract-Kit-1.0",
        "AGPL-3.0",
        "Qwen/Qwen3-Embedding-0.6B",
        "qwen3:8b",
        "Qwen/Qwen3-8B",
        "sentence-transformers/all-MiniLM-L6-v2",
        "urchade/gliner_multi-v2.1",
        "unsloth/Qwen3.8-27B-NVFP4",
        "DFlash2",
        "no public source",
        "Apache-2.0",
        "vLLM",
        "LiteLLM",
        "Envoy",
        "endpoint picker",
        "oauth2-proxy",
        "fastapi",
        "Azure OpenAI",
        "KEDA",
        "MinIO",
        "Snowflake",
        "Cortex",
        "plotly",
        "Plotly.js",
        "tesseract-ocr",
        "poppler-utils",
        "GPL/LGPL/MIT",
        "psycopg",
        "LGPL-3.0",
        "flakegraph[leiden]",
        "GPL-3.0-or-later",
        "react/package.json",
        "react/bun.lock",
    ]
    for phrase in required_phrases:
        assert phrase in notices, phrase
    # Stale components must not linger once they leave the deployment.
    assert "SeaweedFS" not in notices
    assert "Streamlit" not in notices
    assert kg_processor.__version__


def test_third_party_notices_name_every_direct_python_dependency() -> None:
    """Adding a dependency without a notice must fail here, not in legal review."""

    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"])
    for extra, requirements in project["optional-dependencies"].items():
        if extra != "dev":
            declared.extend(requirements)

    missing = []
    for requirement in declared:
        name = re.split(r"[\[><=!~; ]", requirement, maxsplit=1)[0]
        if f"`{name}" not in notices:
            missing.append(name)
    assert missing == []


def test_snowflake_docs_stay_public_and_template_based() -> None:
    setup_notes = Path("docs/snowflake-setup.md").read_text(encoding="utf-8")

    required_phrases = [
        "account-neutral",
        "KG_PROCESSOR_ROLE",
        "CREATE COMPUTE POOL",
        "KG_PROCESSOR_CPU_POOL",
        "KG_*",
        "image repository",
        "KG_RUN_SNOWFLAKE_LIVE",
    ]

    for phrase in required_phrases:
        assert phrase in setup_notes

    forbidden_patterns = [
        r"https://app\.snowflake\.com/[^/\s`]+/[^)\s`]+",
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        r"sha256:[0-9a-f]{64}",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, setup_notes, flags=re.IGNORECASE) is None
