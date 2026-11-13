"""The fleet checks a submit host runs before queueing a run nothing would claim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import yaml
from helm import render as render_chart

from kg_processor.fleet.kubectl import ClusterTarget
from kg_processor.fleet.preflight import fleet_preflight, placeholder_names
from kg_processor.fleet.profile import fleet_profile, mounted_config_key, semantic_mismatches
from kg_processor.fleet.recover import queued_worker_components, recover_workers

_RUN: dict[str, Any] = {
    "ocr": {"provider": "fallback"},
    "llm": {
        "provider": "vllm_local",
        "model": "Qwen/Qwen3-8B",
        "endpoint": "http://localhost:8000/v1",
        "api_key": "${KG_LLM_API_KEY}",
    },
    "embedding": {
        "provider": "sentence_transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


def _use(monkeypatch: pytest.MonkeyPatch, kubectl: object) -> None:
    for module in (
        "kg_processor.fleet.kubectl",
        "kg_processor.fleet.preflight",
        "kg_processor.fleet.profile",
    ):
        monkeypatch.setattr(f"{module}.kubectl_json", kubectl, raising=False)


def _run(**changes: Any) -> dict[str, Any]:
    run = {section: dict(values) for section, values in _RUN.items()}
    for section, values in changes.items():
        run[section] = values
    return run


def _fleet_kubectl_fixture(
    *,
    profile_ocr: str,
    model_servers: int,
    output_credential: str | None = None,
    service_account_available: bool = True,
    ontology: Mapping[str, object] | None = None,
    config_key: str = "config.yaml",
    mounted_filename: str | None = None,
) -> object:
    """Return a deterministic kubectl adapter for fleet-preflight unit tests."""

    deployments = []
    scaled_objects = []
    # The gate mounts configuration of its own under the same volume name; it
    # must not be mistaken for a pool that claims runs.
    deployments.append(
        {
            "metadata": {
                "name": "flakegraph-auth",
                "labels": {"app.kubernetes.io/component": "auth-proxy"},
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "serviceAccountName": "default",
                        "containers": [{"image": "oauth2-proxy:1", "env": []}],
                        "volumes": [{"name": "config", "configMap": {"name": "flakegraph-auth"}}],
                    }
                },
                "status": {"availableReplicas": 1},
            },
        }
    )
    # The gateway reads its own sign-in secret. Only workers read a run's
    # credentials, and the console may read only the Secrets they reference.
    deployments.append(
        {
            "metadata": {
                "name": "flakegraph-litellm",
                "labels": {"app.kubernetes.io/component": "gateway"},
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "serviceAccountName": "default",
                        "containers": [
                            {
                                "image": "litellm:1",
                                "env": [
                                    {
                                        "name": "MICROSOFT_CLIENT_SECRET",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "flakegraph-entra-sso",
                                                "key": "client-secret",
                                            }
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                },
                "status": {"availableReplicas": 1},
            },
        }
    )
    for component in ("worker-prepare", "worker-extract", "worker-finalize"):
        name = f"flakegraph-{component}"
        deployments.append(
            {
                "metadata": {
                    "name": name,
                    "labels": {"app.kubernetes.io/component": component},
                },
                "spec": {
                    "replicas": 0,
                    "template": {
                        "spec": {
                            "serviceAccountName": "flakegraph",
                            "containers": [
                                {
                                    "image": "flakegraph:1.0.0",
                                    "args": [
                                        "distributed",
                                        "worker",
                                        "--config",
                                        f"/etc/flakegraph/{mounted_filename or config_key}",
                                    ],
                                    "volumeMounts": [
                                        {"name": "config", "mountPath": "/etc/flakegraph"}
                                    ],
                                    "env": (
                                        [
                                            {
                                                "name": output_credential,
                                                "valueFrom": {
                                                    "secretKeyRef": {
                                                        "name": "flakegraph-providers",
                                                        "key": output_credential,
                                                    }
                                                },
                                            }
                                        ]
                                        if component == "worker-finalize" and output_credential
                                        else []
                                    ),
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "config",
                                    "configMap": {
                                        "name": "flakegraph-runtime",
                                        **(
                                            {
                                                "items": [
                                                    {"key": config_key, "path": mounted_filename}
                                                ]
                                            }
                                            if mounted_filename
                                            else {}
                                        ),
                                    },
                                },
                                *(
                                    [
                                        {
                                            "name": "ontology",
                                            "configMap": {"name": "flakegraph-ontology"},
                                        }
                                    ]
                                    if ontology
                                    else []
                                ),
                            ],
                        }
                    },
                },
            }
        )
        scaled_objects.append(
            {
                "spec": {"scaleTargetRef": {"name": name}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
        )

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        if arguments == ["config", "current-context"] and raw:
            return "flakegraph-fleet\n"
        if arguments[:2] == ["get", "nodes"]:
            return {
                "items": [
                    {
                        "metadata": {"labels": {"kubernetes.io/arch": "amd64"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        resource = arguments[1]
        if resource == "deployments":
            response: object = {"items": deployments}
        elif resource == "scaledobjects.keda.sh":
            response = {"items": scaled_objects}
        elif resource == "configmap" and arguments[2] == "flakegraph-ontology":
            response = {"data": {"ontology.yaml": yaml.safe_dump(ontology or {})}}
        elif resource == "configmap":
            response = {
                "data": {
                    config_key: yaml.safe_dump(
                        {
                            "ocr": {"provider": profile_ocr},
                            "llm": {
                                "provider": "vllm_local",
                                "model": "Qwen/Qwen3-8B",
                            },
                            "embedding": {
                                "provider": "sentence_transformers",
                                "model": "sentence-transformers/all-MiniLM-L6-v2",
                            },
                        }
                    )
                }
            }
        elif resource == "statefulsets":
            response = {
                "items": (
                    [
                        {
                            "metadata": {
                                "labels": {"app.kubernetes.io/component": "model-serving"}
                            },
                            "spec": {"replicas": model_servers},
                            "status": {"readyReplicas": model_servers},
                        }
                    ]
                    if model_servers
                    else []
                )
            }
        elif resource == "secret":
            # Read by name, never listed: the console's Role grants only that.
            assert arguments[2] == "flakegraph-providers", arguments
            response = {"data": ({output_credential: "encoded"} if output_credential else {})}
        elif resource == "serviceaccounts":
            response = {
                "items": (
                    [{"metadata": {"name": "flakegraph"}}] if service_account_available else []
                )
            }
        else:
            raise AssertionError(f"Unexpected kubectl arguments: {arguments}")
        return response

    return kubectl


def test_fleet_preflight_accepts_keda_workers_and_ready_local_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat scale-to-zero workers as healthy when KEDA and model serving are ready."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is True, result["errors"]
    assert "Chart-managed vLLM model servers are ready: 1/1" in result["checks"]
    assert "Fleet credential is available: KG_LLM_API_KEY" in result["checks"]


def test_fleet_preflight_accepts_a_runs_own_ontology_whatever_the_fleet_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's vocabulary is its own; the fleet's mounted profile is only a default."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            output_credential="KG_LLM_API_KEY",
            ontology={"name": "general", "mode": "strict"},
        ),
    )
    profile = {
        "name": "widgets",
        "description": "Only widgets.",
        "mode": "open",
        "entity_types": [{"name": "WIDGET", "description": "A named widget."}],
    }

    result = fleet_preflight(_run(ontology={"profile": profile}), "flakegraph", ClusterTarget())

    assert result["ok"] is True, result["errors"]
    assert "Selected OCR, LLM, embedding, and ontology match fleet workers" in result["checks"]


def test_fleet_preflight_rejects_an_ontology_that_does_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile that cannot be validated fails here, not after the first document."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(
        _run(ontology={"profile": {"name": "broken", "mode": "hybrid"}}),
        "flakegraph",
        ClusterTarget(),
    )

    assert result["ok"] is False
    assert any("not a valid profile" in item for item in result["errors"])


def test_fleet_preflight_rejects_missing_worker_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a fleet that cannot create autoscaled workers before accepting work."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, service_account_available=False
        ),
    )

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is False
    assert any("references missing service account" in error for error in result["errors"])


def test_fleet_preflight_explains_profile_and_model_serving_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block runs that the homogeneous worker profile cannot execute consistently."""

    _use(monkeypatch, _fleet_kubectl_fixture(profile_ocr="builtin_text", model_servers=0))

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is False
    assert (
        "ocr.provider does not match fleet workers: selected 'fallback', deployed 'builtin_text'"
        in result["errors"]
    )
    assert (
        "Selected vLLM endpoint is node-local, but the fleet has no ready model server (0/0)"
        in result["errors"]
    )


def test_fleet_preflight_requires_output_secret_on_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a Snowflake destination that the finalizer cannot authenticate to."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(
        _run(snowflake={"account": "account", "password": "${SNOWFLAKE_PASSWORD}"}),
        "flakegraph",
        ClusterTarget(),
    )

    assert result["ok"] is False
    assert (
        "Fleet credential is missing from namespace flakegraph: SNOWFLAKE_PASSWORD"
        in result["errors"]
    )
    assert (
        "Finalizer does not map output credential environment variable: SNOWFLAKE_PASSWORD"
        in result["errors"]
    )


def test_semantic_mismatches_compare_only_what_both_sides_state() -> None:
    """A key the fleet omits takes the runtime default on both sides; placeholders are transport."""

    deployed = {
        "ocr": {
            "provider": "fallback",
            "mineru_method": "ocr",
            "mineru_api_url": "${KG_MINERU_API_URL}",
        }
    }

    assert semantic_mismatches(
        {"ocr": {"provider": "fallback", "mineru_method": "auto"}}, deployed
    ) == ["ocr.mineru_method does not match fleet workers: selected 'auto', deployed 'ocr'"]
    assert (
        semantic_mismatches(
            {"ocr": {"provider": "fallback", "mineru_api_url": "http://x"}}, deployed
        )
        == []
    )
    assert semantic_mismatches({"ocr": {"provider": "fallback"}}, deployed) == []


def test_placeholder_names_read_only_credential_fields() -> None:
    run = {
        "llm": {"api_key": "${KG_LLM_API_KEY}", "endpoint": "${KG_LLM_ENDPOINT}"},
        "snowflake": {"password": "${SNOWFLAKE_PASSWORD}", "account": "${SNOWFLAKE_ACCOUNT}"},
    }

    assert placeholder_names(run) == {"KG_LLM_API_KEY", "SNOWFLAKE_PASSWORD"}


def test_fleet_profile_reports_what_the_workers_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submit host composes runs against the mounted profile and ontology."""

    ontology: Mapping[str, object] = {"name": "general", "mode": "hybrid"}
    _use(
        monkeypatch,
        _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1, ontology=ontology),
    )

    profile = fleet_profile(ClusterTarget(), "flakegraph")

    assert profile["config_map"] == "flakegraph-runtime"
    assert profile["config"]["ocr"] == {"provider": "fallback"}
    assert profile["ontology"] == dict(ontology)
    assert profile["parsing_endpoint"] is None


def _recovery_kubectl(
    *,
    available_replicas: int,
    pods: list[dict[str, Any]],
    calls: list[list[str]] | None = None,
    desired: int = 1,
) -> object:
    """Return a kubectl adapter for recovering one finalizer deployment.

    Raw calls are the restarts recovery performs; a test that passes no
    ``calls`` list expects none.
    """

    deployment = {
        "metadata": {
            "name": "flakegraph-finalize",
            "labels": {"app.kubernetes.io/component": "worker-finalize"},
        },
        "spec": {
            "replicas": desired,
            "template": {
                "spec": {
                    "serviceAccountName": "flakegraph-spark",
                    "nodeSelector": {"flakegraph.io/node-class": "gpu-pool"},
                }
            },
        },
        "status": {"availableReplicas": available_replicas},
    }

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        if raw:
            assert calls is not None, f"Unexpected rollout: {arguments}"
            calls.append(arguments)
            return ""
        resource = arguments[1]
        if resource == "deployments":
            return {"items": [deployment]}
        if resource == "serviceaccounts":
            return {"items": [{"metadata": {"name": "flakegraph-spark"}}]}
        if resource == "nodes":
            return {
                "items": [
                    {
                        "metadata": {"labels": {"flakegraph.io/node-class": "gpu-pool"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        if resource == "pods":
            return {"items": pods}
        raise AssertionError(f"Unexpected kubectl arguments: {arguments}")

    return kubectl


def _use_recovery(monkeypatch: pytest.MonkeyPatch, kubectl: object) -> None:
    for module in ("kg_processor.fleet.kubectl", "kg_processor.fleet.recover"):
        monkeypatch.setattr(f"{module}.kubectl_json", kubectl, raising=False)


def test_claimable_stages_name_the_pools_that_claim_them() -> None:
    """A finalizer queued behind its documents is not waiting on a pool."""

    counts = [
        {"stage": "prepare_document", "status": "queued", "count": 2, "ready": 2},
        {"stage": "extract_entity_window", "status": "queued", "count": 4, "ready": 1},
        {"stage": "extract_relation_window", "status": "running", "count": 3, "ready": 0},
        {"stage": "finalize_graph", "status": "queued", "count": 1, "ready": 0},
    ]

    assert queued_worker_components(counts) == {"worker-prepare", "worker-extract"}
    assert recover_workers("flakegraph", set(), ClusterTarget()).startswith("No queued")


def test_recovery_does_not_restart_healthy_shared_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave a healthy finalizer untouched when another graph currently owns it."""

    _use_recovery(monkeypatch, _recovery_kubectl(available_replicas=1, pods=[]))

    message = recover_workers("flakegraph", {"worker-finalize"}, ClusterTarget())

    assert "worker-finalize" in message
    assert "waiting for shared capacity" in message


def test_recovery_does_not_interrupt_reconciling_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let Kubernetes replace an unavailable active pod without forcing a rollout."""

    running_pod = {
        "metadata": {"labels": {"app.kubernetes.io/component": "worker-finalize"}},
        "status": {"phase": "Running", "containerStatuses": []},
    }
    calls: list[list[str]] = []
    _use_recovery(
        monkeypatch, _recovery_kubectl(available_replicas=0, pods=[running_pod], calls=calls)
    )

    message = recover_workers("flakegraph", {"worker-finalize"}, ClusterTarget())

    assert "already reconciling" in message
    assert calls == []


def test_recovery_restarts_a_pool_with_nothing_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool at zero is scaled up; a stuck one is restarted."""

    scaled: list[list[str]] = []
    _use_recovery(
        monkeypatch, _recovery_kubectl(available_replicas=0, pods=[], calls=scaled, desired=0)
    )
    assert "Restarted unavailable worker pools: worker-finalize" in recover_workers(
        "flakegraph", {"worker-finalize"}, ClusterTarget()
    )
    assert scaled == [
        ["scale", "deployment", "flakegraph-finalize", "-n", "flakegraph", "--replicas=1"]
    ]

    restarted: list[list[str]] = []
    _use_recovery(
        monkeypatch, _recovery_kubectl(available_replicas=0, pods=[], calls=restarted, desired=1)
    )
    recover_workers("flakegraph", {"worker-finalize"}, ClusterTarget())
    assert restarted == [
        ["rollout", "restart", "deployment/flakegraph-finalize", "-n", "flakegraph"]
    ]


def test_recovery_reports_a_pod_that_cannot_start_instead_of_restarting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken_pod = {
        "metadata": {"labels": {"app.kubernetes.io/component": "worker-finalize"}},
        "status": {
            "phase": "Pending",
            "containerStatuses": [
                {
                    "state": {
                        "waiting": {"reason": "ImagePullBackOff", "message": "manifest unknown"}
                    }
                }
            ],
        },
    }
    _use_recovery(monkeypatch, _recovery_kubectl(available_replicas=0, pods=[broken_pod]))

    with pytest.raises(RuntimeError, match="pod startup failed: manifest unknown"):
        recover_workers("flakegraph", {"worker-finalize"}, ClusterTarget())


@pytest.mark.parametrize("filename", [None, "runtime.yaml", "nested/runtime.yaml"])
def test_fleet_profile_and_preflight_follow_the_workers_selected_config_key(
    monkeypatch: pytest.MonkeyPatch, filename: str | None
) -> None:
    """Custom chart keys and ConfigMap items renames select the mounted worker profile."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            output_credential="KG_LLM_API_KEY",
            config_key="processing.yaml",
            mounted_filename=filename,
        ),
    )
    profile = fleet_profile(ClusterTarget(), "flakegraph")
    assert profile["config"]["ocr"]["provider"] == "fallback"
    assert fleet_preflight(_run(), "flakegraph", ClusterTarget())["ok"]


def test_profile_discovery_requires_worker_pools_to_agree_on_the_config_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filename mismatch cannot make submission compare against an arbitrary pool's profile."""

    fixture = _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1)

    def kubectl(arguments: list[str], **kwargs: Any) -> Any:
        result = fixture(arguments, **kwargs)  # type: ignore[operator]
        if arguments[:2] == ["get", "deployments"]:
            result["items"][-1]["spec"]["template"]["spec"]["containers"][0]["args"][-1] = (
                "/etc/flakegraph/other.yaml"
            )
        return result

    _use(monkeypatch, kubectl)
    with pytest.raises(RuntimeError, match="shared processing ConfigMap key"):
        fleet_profile(ClusterTarget(), "flakegraph")
    preflight = fleet_preflight(_run(), "flakegraph", ClusterTarget())
    assert not preflight["ok"]
    assert any("shared processing ConfigMap key" in error for error in preflight["errors"])


def test_mounted_profile_resolves_a_configmap_items_key_through_a_subpath_mount() -> None:
    """A file mount maps the on-disk config path back to its ConfigMap key."""

    deployment = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "args": ["--config=/etc/flakegraph/runtime.yaml"],
                            "volumeMounts": [
                                {
                                    "name": "config",
                                    "mountPath": "/etc/flakegraph/runtime.yaml",
                                    "subPath": "selected.yaml",
                                }
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "config",
                            "configMap": {
                                "name": "profile",
                                "items": [{"key": "processing.yaml", "path": "selected.yaml"}],
                            },
                        }
                    ],
                }
            }
        }
    }
    assert mounted_config_key(deployment) == "processing.yaml"


@pytest.mark.helm
def test_fleet_profile_reads_the_custom_filename_rendered_by_the_chart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helm's config.key override remains discoverable through real worker args and mounts."""

    resources = render_chart(("config.key=processing.yaml",))
    deployments = [item for item in resources if item["kind"] == "Deployment"]
    config_maps = {
        item["metadata"]["name"]: item for item in resources if item["kind"] == "ConfigMap"
    }

    def kubectl(arguments: list[str], **_kwargs: Any) -> dict[str, Any]:
        if arguments[:2] == ["get", "deployments"]:
            return {"items": deployments}
        assert arguments[:2] == ["get", "configmap"]
        return config_maps[arguments[2]]

    _use(monkeypatch, kubectl)
    result = fleet_profile(ClusterTarget(), "fleet")
    assert "processing.yaml" in config_maps[result["config_map"]]["data"]
    assert result["config"]["llm"]["provider"]
