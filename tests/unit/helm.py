"""Render the chart the way a cluster would, so tests assert on objects."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

CHART = Path("deploy/helm/flakegraph")
RELEASE = "fg"
NAMESPACE = "fleet"
FULLNAME = f"{RELEASE}-flakegraph"
# The CRD-backed APIs the chart's optional planes hang off - the Prometheus
# operator, KEDA, CloudNativePG, Traefik - declared present so a render
# includes them without a cluster to ask. Each is guarded in the templates on
# exactly these names.
API_VERSIONS = ",".join(
    (
        *(
            f"monitoring.coreos.com/v1/{kind}"
            for kind in ("ServiceMonitor", "PodMonitor", "PrometheusRule")
        ),
        "keda.sh/v1alpha1/ScaledObject",
        "postgresql.cnpg.io/v1/Cluster",
        "traefik.io/v1alpha1/Middleware",
        "traefik.io/v1alpha1/IngressRoute",
    )
)


def helm() -> str:
    """Return the helm binary, or skip the test that needs it."""

    found = shutil.which("helm")
    if found is None:
        pytest.skip("helm is not installed; the rendered-chart contracts need it")
    return found


def _template(
    settings: tuple[str, ...],
    values: tuple[Path, ...],
    api_versions: str,
    release: str = RELEASE,
) -> subprocess.CompletedProcess[str]:
    command = [helm(), "template", release, str(CHART), "--namespace", NAMESPACE]
    if api_versions:
        command += ["--api-versions", api_versions]
    for path in values:
        command += ["--values", str(path)]
    for setting in settings:
        command += ["--set", setting]
    return subprocess.run(command, capture_output=True, text=True, check=False)


@cache
def render(
    settings: tuple[str, ...] = (),
    *,
    values: tuple[Path, ...] = (),
    release: str = RELEASE,
) -> list[dict[str, Any]]:
    """Render the chart offline with ``--set`` overrides and values files."""

    result = _template(settings, values, API_VERSIONS, release)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def fails(
    settings: tuple[str, ...],
    *,
    values: tuple[Path, ...] = (),
    api_versions: str = API_VERSIONS,
) -> str:
    """Render a configuration the chart must refuse, and return what it said.

    ``api_versions=""`` renders against a cluster without the monitoring CRDs.
    """

    result = _template(settings, values, api_versions)
    assert result.returncode != 0, "the chart rendered a configuration it must refuse"
    return result.stderr


@cache
def notes(settings: tuple[str, ...] = ()) -> str:
    """Return the NOTES a release would print, rendered without a cluster.

    ``helm template`` never prints NOTES.txt; a client-side dry-run install
    does, and with no kubeconfig to find it cannot reach for a cluster. That
    holds for Helm 4; Helm 3 refuses the dry run until it has reached one.
    """

    command = [
        helm(),
        "install",
        RELEASE,
        str(CHART),
        "--dry-run=client",
        "--namespace",
        NAMESPACE,
    ]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KUBECONFIG": os.devnull},
    )
    assert result.returncode == 0, result.stderr
    _, _, printed = result.stdout.partition("\nNOTES:\n")
    return printed


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one repository-owned YAML mapping for contract assertions."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def values() -> dict[str, Any]:
    """Return the chart's default values."""

    return load_yaml(CHART / "values.yaml")


def schema() -> dict[str, Any]:
    """Return the chart's values schema."""

    loaded = json.loads((CHART / "values.schema.json").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def one(rendered: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Return exactly one rendered object by kind and name."""

    matches = [doc for doc in rendered if doc["kind"] == kind and doc["metadata"]["name"] == name]
    assert len(matches) == 1, f"expected one {kind}/{name}, found {len(matches)}"
    return matches[0]


def pod(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the pod spec a Deployment, StatefulSet, or Job stamps out."""

    spec: dict[str, Any] = workload["spec"]["template"]["spec"]
    return spec


def container(pod_spec: dict[str, Any], name: str, field: str = "containers") -> dict[str, Any]:
    """Return one container of a pod spec by name."""

    containers: list[dict[str, Any]] = pod_spec[field]
    return next(item for item in containers if item["name"] == name)


def env(container_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a container's environment by variable name."""

    return {entry["name"]: entry for entry in container_spec["env"]}


def args(container_spec: dict[str, Any]) -> dict[str, str]:
    """Pair each ``--flag`` with the argument that follows it.

    A flag followed by another flag is a switch and maps to itself, so a switch
    is looked up through ``in container["args"]`` rather than through this.
    """

    given: list[str] = container_spec["args"]
    paired: dict[str, str] = {}
    for index, arg in enumerate(given):
        if arg.startswith("--"):
            following = given[index + 1] if index + 1 < len(given) else arg
            paired[arg] = arg if following.startswith("--") else following
    return paired
