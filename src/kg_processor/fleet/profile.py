# SPDX-License-Identifier: Apache-2.0
"""The processing profile the fleet's workers actually run.

A worker claims only a run whose semantic configuration hashes to its own, so
a submit host that composes runs from any other profile asks the operator to
reconcile two configurations by hand - and learns of the mismatch only when a
run sits queued with nothing on the page to explain it. The profile is read
from what the worker pools mount, which is the one place it cannot drift.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import yaml

from kg_processor.fleet.kubectl import ClusterTarget, component, config_map, labelled

ENVIRONMENT_PLACEHOLDER = re.compile(r"\$\{[A-Z_][A-Z0-9_]*\}")
PARSING_ENDPOINT_ENVIRONMENT_NAME = "KG_MINERU_API_URL"
REQUIRED_WORKER_COMPONENTS = frozenset({"worker-prepare", "worker-extract", "worker-finalize"})

# Keys inside the compared sections that describe how a worker reaches
# infrastructure rather than what graph it produces. The runtime excludes
# these from the digest workers claim by, so comparing them would report
# differences that never prevent a claim.
DEPLOYMENT_LOCAL_KEYS: Mapping[str, frozenset[str]] = {
    "ocr": frozenset(
        {
            "mineru_api_key",
            "mineru_api_url",
            "mineru_command",
            "mineru_server_url",
            "model_cache_dir",
            "tesseract_command",
            "tesseract_pdf_renderer_command",
            "timeout_seconds",
        }
    ),
    "llm": frozenset({"api_key", "endpoint", "timeout_seconds"}),
    "embedding": frozenset({"api_key", "batch_size", "device", "endpoint"}),
    "graph": frozenset(
        {
            "extraction_parallelism",
            "finalization_provider_concurrency",
            # The vocabulary is the run's, like its ontology profile.
            "entity_types",
            "relation_types",
        }
    ),
}
SEMANTIC_SECTIONS = ("ocr", "llm", "embedding", "graph", "extractors")


def mounted_config_map(resource: Mapping[str, Any], volume_name: str = "config") -> str | None:
    """Return the ConfigMap a worker mounts under one volume name."""

    volumes = resource.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
    for volume in volumes:
        if volume.get("name") != volume_name:
            continue
        name = volume.get("configMap", {}).get("name")
        return str(name) if name else None
    return None


def mounted_config_key(resource: Mapping[str, Any], volume_name: str = "config") -> str:
    """Resolve the worker's ``--config`` path through its ConfigMap volume mount.

    ConfigMap ``items`` can rename a key on disk, and ``subPath`` mounts can
    select one file. The argument identifies the file the worker actually reads
    rather than assuming the chart's default filename.
    """

    pod = resource.get("spec", {}).get("template", {}).get("spec", {})
    volume: Mapping[str, Any] = next(
        (item for item in pod.get("volumes", []) if item.get("name") == volume_name), {}
    )
    mapping = volume.get("configMap", {})
    keys: set[str] = set()
    for container in pod.get("containers", []):
        arguments = [*container.get("command", []), *container.get("args", [])]
        paths = [
            str(arguments[index + 1])
            for index, argument in enumerate(arguments[:-1])
            if argument == "--config"
        ]
        paths.extend(
            str(argument).split("=", 1)[1]
            for argument in arguments
            if str(argument).startswith("--config=")
        )
        for mount in container.get("volumeMounts", []):
            if mount.get("name") != volume_name or mount.get("subPathExpr"):
                continue
            mount_path = PurePosixPath(str(mount.get("mountPath", "")))
            for path in paths:
                try:
                    relative = PurePosixPath(path).relative_to(mount_path)
                except ValueError:
                    continue
                mounted_file = PurePosixPath(str(mount.get("subPath", ""))) / relative
                filename = str(mounted_file)
                items = mapping.get("items")
                if items is None:
                    keys.add(filename)
                else:
                    keys.update(str(item["key"]) for item in items if item.get("path") == filename)
    if len(keys) != 1 or not mapping.get("name"):
        raise RuntimeError("Worker --config must select one file from its processing ConfigMap")
    return next(iter(keys))


def shared_config_key(workers: Sequence[Mapping[str, Any]]) -> str:
    """Require every worker pool to read the same ConfigMap data key."""

    keys = {mounted_config_key(worker) for worker in workers}
    if len(keys) != 1:
        raise RuntimeError("Worker pools must read one shared processing ConfigMap key")
    return next(iter(keys))


def load_deployed_profile(
    config_map: Mapping[str, Any], key: str = "config.yaml"
) -> dict[str, Any]:
    """Parse the mounted worker profile without exposing any Secret resources."""

    text = config_map.get("data", {}).get(key)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Fleet processing ConfigMap does not contain {key}")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"Fleet processing {key} must contain a mapping")
    return dict(loaded)


def worker_deployments(target: ClusterTarget, namespace: str) -> dict[str, dict[str, Any]]:
    """Return the chart's Deployments by component label."""

    return {
        component(item): item
        for item in labelled(target, namespace, "deployments")
        if component(item)
    }


def parsing_endpoint(deployments: Mapping[str, Mapping[str, Any]]) -> str | None:
    """Read the parsing endpoint literal the deployed workers are given.

    It is supplied through the environment, where it also overrides whatever
    the mounted profile says, and it is excluded from the digest: transport,
    not identity. It travels with a run only so the run states the endpoint its
    own preflight requires.
    """

    for name, item in deployments.items():
        if not name.startswith("worker-"):
            continue
        containers = item.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        for container in containers:
            for entry in container.get("env", []):
                if str(entry.get("name")) != PARSING_ENDPOINT_ENVIRONMENT_NAME:
                    continue
                value = str(entry.get("value") or "").strip()
                if value:
                    return value
    return None


def fleet_profile(target: ClusterTarget, namespace: str) -> dict[str, Any]:
    """Describe what the fleet's workers run, for a submit host to compose against.

    Returns the mounted profile, the parsing endpoint, and the ontology the
    workers mount (inline, so a run can carry the same one). Environment
    placeholders in the profile are left as written: the CLI resolves them
    from the pod environment when it loads a run, which is also how the workers
    read them.
    """

    deployments = worker_deployments(target, namespace)
    # Only the pools that claim runs decide the profile; the gate, the gateway
    # and the console mount configuration of their own under the same name.
    workers = [item for name, item in deployments.items() if name in REQUIRED_WORKER_COMPONENTS]
    if len(workers) != len(REQUIRED_WORKER_COMPONENTS):
        missing = sorted(REQUIRED_WORKER_COMPONENTS - set(deployments))
        raise RuntimeError(f"Kubernetes fleet is missing worker deployments: {', '.join(missing)}")
    config_maps = {name for item in workers if (name := mounted_config_map(item)) is not None}
    if len(config_maps) != 1:
        raise RuntimeError("Worker pools must mount one shared FlakeGraph processing ConfigMap")
    config_name = next(iter(config_maps))
    profile = load_deployed_profile(
        config_map(target, namespace, config_name), shared_config_key(workers)
    )
    ontology_name = next(
        (name for item in workers if (name := mounted_config_map(item, "ontology"))), None
    )
    ontology: dict[str, Any] | None = None
    if ontology_name:
        data = config_map(target, namespace, ontology_name).get("data", {})
        text = next((value for value in data.values() if isinstance(value, str)), "")
        loaded = yaml.safe_load(text) or {} if text.strip() else {}
        ontology = dict(loaded) if isinstance(loaded, Mapping) else None
    return {
        "namespace": namespace,
        "config_map": config_name,
        "config": profile,
        "parsing_endpoint": parsing_endpoint(deployments),
        "ontology_config_map": ontology_name,
        "ontology": ontology,
    }


def semantic_mismatches(run: Mapping[str, Any], deployed: Mapping[str, Any]) -> list[str]:
    """Report every digest-relevant field on which a run and the fleet disagree.

    The hash a worker claims by covers whole configuration sections, so the
    effective configuration is compared rather than the fields a form shows.
    Only keys both sides state are compared: a key the fleet omits takes the
    runtime's default on both sides.
    """

    mismatches: list[str] = []
    for section in SEMANTIC_SECTIONS:
        run_section = run.get(section)
        fleet_section = deployed.get(section)
        if not isinstance(run_section, Mapping):
            continue
        excluded = DEPLOYMENT_LOCAL_KEYS.get(section, frozenset())
        fleet_values = fleet_section if isinstance(fleet_section, Mapping) else {}
        for key in sorted(set(run_section) | set(fleet_values)):
            if key in excluded or key not in fleet_values or key not in run_section:
                continue
            selected = run_section.get(key)
            actual = fleet_values.get(key)
            if isinstance(actual, str) and ENVIRONMENT_PLACEHOLDER.fullmatch(actual):
                continue
            if selected != actual:
                mismatches.append(
                    f"{section}.{key} does not match fleet workers: "
                    f"selected {selected!r}, deployed {actual!r}"
                )
    return mismatches
