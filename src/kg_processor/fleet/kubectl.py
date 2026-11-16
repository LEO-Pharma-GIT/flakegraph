# SPDX-License-Identifier: Apache-2.0
"""Bounded kubectl reads against one cluster."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

KUBECTL_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ClusterTarget:
    """Where kubectl should point.

    Empty values mean the ambient kubeconfig and its current context - or,
    inside a pod, the service account - which is right wherever there is one
    cluster to speak of.
    """

    kubeconfig: str = ""
    context: str = ""

    def arguments(self) -> list[str]:
        """Return the kubectl flags that select this cluster."""

        return ["--context", self.context] if self.context else []

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        """Return ``base`` with the kubeconfig this target selects, if any."""

        environment = dict(base)
        if self.kubeconfig:
            environment["KUBECONFIG"] = self.kubeconfig
        return environment


def kubectl_json(
    arguments: Sequence[str],
    *,
    target: ClusterTarget,
    raw: bool = False,
) -> Any:
    """Execute a bounded kubectl call and parse JSON unless raw output is requested.

    A stale VPN route or unreachable API server must not leave the caller
    waiting indefinitely; the timeout applies to each small inventory request.
    """

    command = ["kubectl", *target.arguments(), *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=KUBECTL_TIMEOUT_SECONDS,
            env=target.environment(os.environ),
        )
    except subprocess.TimeoutExpired as exc:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Kubernetes did not respond within {KUBECTL_TIMEOUT_SECONDS} seconds: {rendered}"
        ) from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "kubectl command failed")
    return result.stdout if raw else json.loads(result.stdout)


def current_context(target: ClusterTarget) -> str:
    """Name the context a read resolves to; a pod's service account has none."""

    try:
        return str(kubectl_json(["config", "current-context"], target=target, raw=True)).strip()
    except RuntimeError:
        return target.context or "in-cluster"


def labelled(target: ClusterTarget, namespace: str, resource: str) -> list[dict[str, Any]]:
    """List one resource kind the chart labels as FlakeGraph's."""

    return list(
        kubectl_json(
            [
                "get",
                resource,
                "-n",
                namespace,
                "-l",
                "app.kubernetes.io/name=flakegraph",
                "-o",
                "json",
            ],
            target=target,
        ).get("items", [])
    )


def component(resource: Mapping[str, Any]) -> str:
    """Return the chart component label of a resource, or an empty string."""

    return str(
        resource.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component", "")
    )


def config_map(target: ClusterTarget, namespace: str, name: str) -> dict[str, Any]:
    """Read one ConfigMap."""

    return dict(
        kubectl_json(["get", "configmap", name, "-n", namespace, "-o", "json"], target=target)
    )
