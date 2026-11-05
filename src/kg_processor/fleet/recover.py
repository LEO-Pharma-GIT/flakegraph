# SPDX-License-Identifier: Apache-2.0
"""Nudge the worker pools a queued run is waiting on, and nothing else.

Queued tasks are durable and must not be duplicated, so recovery never
touches task state. It validates the static prerequisites a restart cannot
repair - service account, node placement, rollout and image failures - and
restarts only a pool with no available worker. A healthy pool may be busy
with another graph and is left alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kg_processor.fleet.kubectl import ClusterTarget, component, kubectl_json, labelled

_STAGE_POOLS = {"prepare_document": "worker-prepare", "finalize_graph": "worker-finalize"}
_STARTUP_FAILURES = frozenset({"CrashLoopBackOff", "ErrImagePull", "ImagePullBackOff"})


def queued_worker_components(task_counts: Sequence[Mapping[str, Any]]) -> set[str]:
    """Name the pools able to claim the stages a run has claimable work in.

    Queued work whose dependencies are unmet - a finalizer queued behind its
    documents, say - is not waiting on a pool, and scaling one up for it only
    fights the autoscaler that will scale it back down.
    """

    components: set[str] = set()
    for item in task_counts:
        if str(item.get("status", "")) != "queued" or not int(item.get("ready") or 0):
            continue
        components.add(_STAGE_POOLS.get(str(item.get("stage", "")), "worker-extract"))
    return components


def recover_workers(namespace: str, components: set[str], target: ClusterTarget) -> str:
    """Validate and restart unavailable pools; say what was done, or why nothing was."""

    if not components:
        return "No queued processing stage currently requires infrastructure recovery."
    deployments = labelled(target, namespace, "deployments")
    by_component = {component(item): item for item in deployments if component(item)}
    service_accounts = {
        str(item.get("metadata", {}).get("name", ""))
        for item in kubectl_json(
            ["get", "serviceaccounts", "-n", namespace, "-o", "json"], target=target
        ).get("items", [])
    }
    nodes = kubectl_json(["get", "nodes", "-o", "json"], target=target).get("items", [])
    pods_by_component: dict[str, list[Mapping[str, Any]]] = {}
    for pod in labelled(target, namespace, "pods"):
        pods_by_component.setdefault(component(pod), []).append(pod)

    issues: list[str] = []
    unavailable: list[str] = []
    healthy: list[str] = []
    reconciling: list[str] = []
    for name in sorted(components):
        deployment = by_component.get(name)
        if deployment is None:
            issues.append(f"Kubernetes fleet is missing the {name} deployment")
            continue
        pods = pods_by_component.get(name, [])
        issues.extend(_recovery_issues(deployment, pods, service_accounts, nodes))
        available = int(deployment.get("status", {}).get("availableReplicas", 0) or 0)
        if available > 0:
            healthy.append(name)
        elif any(pod.get("status", {}).get("phase") in {"Pending", "Running"} for pod in pods):
            reconciling.append(name)
        else:
            unavailable.append(name)
    if issues:
        raise RuntimeError(" ".join(dict.fromkeys(issues)))

    for name in unavailable:
        _restart(by_component[name], namespace, target)
    restarted = unavailable
    if restarted:
        return (
            f"Restarted unavailable worker pools: {', '.join(restarted)}. "
            "Queued work remains durable and will resume automatically."
        )
    if reconciling:
        return (
            f"Kubernetes is already reconciling worker pools: {', '.join(reconciling)}. "
            "No tasks or active workers were changed; queued work resumes automatically."
        )
    return (
        f"Required worker pools are healthy: {', '.join(healthy)}. This run is "
        "waiting for shared capacity and will resume automatically."
    )


def _restart(deployment: Mapping[str, Any], namespace: str, target: ClusterTarget) -> None:
    """Bring one pool back: scale a pool at zero up, restart one that is stuck."""

    name = str(deployment.get("metadata", {}).get("name", ""))
    desired = int(deployment.get("spec", {}).get("replicas", 0) or 0)
    arguments = (
        ["scale", "deployment", name, "-n", namespace, "--replicas=1"]
        if desired == 0
        else ["rollout", "restart", f"deployment/{name}", "-n", namespace]
    )
    kubectl_json(arguments, target=target, raw=True)


def _recovery_issues(
    deployment: Mapping[str, Any],
    pods: Sequence[Mapping[str, Any]],
    service_accounts: set[str],
    nodes: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return deterministic rollout blockers that restarting cannot repair."""

    name = str(deployment.get("metadata", {}).get("name", "worker deployment"))
    pod_spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
    service_account = str(pod_spec.get("serviceAccountName") or "default")
    issues = []
    if service_account not in service_accounts:
        issues.append(f"{name} references missing service account {service_account}.")
    selector = pod_spec.get("nodeSelector") or {}
    if selector and not any(
        _node_ready(node)
        and all(
            str(node.get("metadata", {}).get("labels", {}).get(key)) == str(value)
            for key, value in selector.items()
        )
        for node in nodes
    ):
        issues.append(f"{name} has no ready node matching selector {dict(selector)}.")
    for condition in deployment.get("status", {}).get("conditions", []):
        if condition.get("type") == "ReplicaFailure" and condition.get("status") == "True":
            message = str(condition.get("message") or condition.get("reason") or "replica failure")
            issues.append(f"{name} cannot create a worker pod: {message}")
    for pod in pods:
        statuses = [
            *pod.get("status", {}).get("initContainerStatuses", []),
            *pod.get("status", {}).get("containerStatuses", []),
        ]
        for status in statuses:
            waiting = status.get("state", {}).get("waiting") or {}
            reason = str(waiting.get("reason") or "")
            if reason in _STARTUP_FAILURES:
                issues.append(f"{name} pod startup failed: {waiting.get('message') or reason}")
    return issues


def _node_ready(node: Mapping[str, Any]) -> bool:
    return any(
        item.get("type") == "Ready" and item.get("status") == "True"
        for item in node.get("status", {}).get("conditions", [])
    )
