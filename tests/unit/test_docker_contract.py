"""Pin what the container profiles lock and what CI runs them with."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_COMPOSE_PATH = Path(".github/docker-compose.smoke.yaml")
_ANY_ACTION_SHA_RE = re.compile(r"uses: [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}")


def test_locked_container_profiles_use_cpu_torch_on_linux() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lockfile = Path("uv.lock").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")

    sources = pyproject["tool"]["uv"]["sources"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}
    assert sources["torch"] == [{"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}]
    assert "torchvision" not in sources
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True

    assert 'version = "2.13.0+cpu"' in lockfile
    assert 'name = "torchvision"' not in lockfile
    assert 'source = { registry = "https://download.pytorch.org/whl/cpu" }' in lockfile
    assert "nvidia-" not in lockfile
    assert "cuda-toolkit" not in lockfile
    assert 'name = "triton"' not in lockfile
    assert "KG_TORCH_INDEX_URL" not in dockerfile
    assert "KG_TORCH_INDEX_URL" not in compose


def test_compose_smoke_service_is_small_and_self_contained() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"flakegraph"}
    smoke_args = services["flakegraph"]["build"]["args"]
    assert smoke_args["KG_INSTALL_MINERU"] == "false"
    assert smoke_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"
    service = services["flakegraph"]
    assert service["command"] == ["worker", "--config", "/app/smoke.yaml"]
    assert set(service["volumes"]) == {
        "../.github/configs/smoke.yaml:/app/smoke.yaml:ro",
        "../data:/app/data:ro",
        "flakegraph-out:/app/out",
    }
    assert "environment" not in service
    assert "flakegraph-out" in compose["volumes"]


def test_workflows_pin_every_action_to_a_commit_sha() -> None:
    """A tag is a moving target; every action, first- or third-party, is a SHA."""

    for workflow_path in sorted(Path(".github/workflows").glob("*.yml")):
        workflow = workflow_path.read_text(encoding="utf-8")
        action_uses = [
            line.split("#", 1)[0].strip()
            for line in workflow.splitlines()
            if line.strip().startswith("uses: ")
        ]
        assert action_uses, workflow_path
        assert all(_ANY_ACTION_SHA_RE.fullmatch(line) for line in action_uses), (
            workflow_path,
            [line for line in action_uses if not _ANY_ACTION_SHA_RE.fullmatch(line)],
        )


def test_release_tags_publish_both_images_for_both_architectures() -> None:
    """The chart's default image.repository has to resolve for an outsider.

    The publish workflow is what makes it resolve: every v* tag builds the
    application and Spark images for amd64 and arm64 and pushes them under the
    repository owner's GHCR namespace, which is the path values.yaml names.
    """

    workflow = yaml.safe_load(Path(".github/workflows/publish.yml").read_text(encoding="utf-8"))
    values = yaml.safe_load(Path("deploy/helm/flakegraph/values.yaml").read_text(encoding="utf-8"))
    # PyYAML reads the bare `on:` key as boolean True.
    assert workflow[True]["push"]["tags"] == ["v*"]
    assert workflow["permissions"]["packages"] == "write"
    (job,) = workflow["jobs"].values()
    images = {entry["name"]: entry["dockerfile"] for entry in job["strategy"]["matrix"]["image"]}
    assert images == {"flakegraph": "Dockerfile", "flakegraph-spark": "Dockerfile.spark"}
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Build and push"]["with"]["platforms"] == "linux/amd64,linux/arm64"
    assert steps["Build and push"]["with"]["push"] is True
    assert steps["Image metadata"]["with"]["images"] == (
        "ghcr.io/${{ github.repository_owner }}/${{ matrix.image.name }}"
    )
    assert "type=semver,pattern={{version}}" in steps["Image metadata"]["with"]["tags"]
    for repository in (values["image"]["repository"], values["spark"]["image"]["repository"]):
        assert repository.startswith("ghcr.io/")
        assert repository.rsplit("/", 1)[1] in images


def test_image_bases_are_digest_pinned_and_quiet() -> None:
    """Every FROM is byte-exact, and no Hub download carries telemetry."""

    for path in (Path("Dockerfile"), Path("Dockerfile.spark")):
        dockerfile = path.read_text(encoding="utf-8")
        froms = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert froms, path
        for line in froms:
            # The kubectl stage is version-parameterised by build arg, so a
            # digest cannot be pinned to it; every other base carries one.
            if "${KG_KUBECTL_VERSION}" in line:
                continue
            assert "@sha256:" in line, (path, line)
        assert "HF_HUB_DISABLE_TELEMETRY=1" in dockerfile, path


def test_container_project_installs_include_declared_license_materials() -> None:
    """Project wheels and runtime trees carry the declared notices in both images."""

    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    for image in ("Dockerfile", "Dockerfile.spark"):
        dockerfile = Path(image).read_text(encoding="utf-8")
        installed_at = dockerfile.rindex("uv sync --locked")
        for license_file in metadata["project"]["license-files"]:
            copies = [
                line
                for line in dockerfile[:installed_at].splitlines()
                if line.startswith("COPY ") and license_file in line.split()
            ]
            assert copies, (image, license_file)
            assert f"/app/{license_file}" in dockerfile[dockerfile.index("chmod") :]
