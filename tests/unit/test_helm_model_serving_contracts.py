"""Protect the Kubernetes-managed serving and document-parsing contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from helm import FULLNAME as _FULLNAME
from helm import NAMESPACE
from helm import args as _args
from helm import container as _container
from helm import env as _env
from helm import fails as _fails
from helm import load_yaml as _load_yaml
from helm import notes as _notes
from helm import one as _one
from helm import pod as _pod
from helm import render as _render
from helm import schema as _schema
from helm import values as _values

from kg_processor.adapters.distributed.postgres import _SCHEMA_STATEMENTS
from kg_processor.ports.embeddings import EMBEDDING_TIMEOUT_SECONDS
from kg_processor.serving.balancer import LANE_BATCH, LANE_LABEL, LANE_PEOPLE
from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    compute_sizing,
)

_EXAMPLES = Path("deploy/examples")
_SPARK_EXAMPLE = _EXAMPLES / "dgx-spark-k3s-values.yaml"
_EXTERNAL_EXAMPLE = _EXAMPLES / "external-provider-values.yaml"
_MINIMAL_EXAMPLE = _EXAMPLES / "minimal-cpu-values.yaml"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# The serving plane is opt-in; the portable defaults need nothing else.
_SERVING = ("modelServing.enabled=true",)
_SPARK = (*_SERVING, "spark.enabled=true")
# Queue-driven scaling is opt-in too: it renders KEDA objects.
_AUTOSCALING = ("autoscaling.enabled=true",)
# The gate on a Traefik cluster, which is where every one of its objects lives.
_ISOLATED = ("controlPlane.enabled=true", "controlPlane.auth.singleOperator=operator@example.test")
_GATED = (
    "controlPlane.enabled=true",
    "ingress.enabled=true",
    "ingress.domain=example.test",
    "ingress.controller=traefik",
    "ingress.authProxy.enabled=true",
    "controlPlane.networkPolicy.enabled=true",
)
_EXTERNAL = (
    "gateway.litellm.upstream.type=external",
    "gateway.litellm.upstream.apiBase=https://inference.example/v1",
    "gateway.litellm.upstream.apiKeySecret.name=upstream",
    "gateway.litellm.upstream.apiKeySecret.key=api-key",
    "gateway.litellm.upstream.models[0]=hosted_vllm/Qwen/Qwen3-8B",
)
_VLLM = f"{_FULLNAME}-vllm"
_ROUTER = f"{_FULLNAME}-inference-router"
_WORKER_POOLS = ("prepare", "extract", "finalize")
_CONSUMER_ENV = (
    "KG_LLM_ENDPOINT",
    "KG_LLM_MODEL",
    "KG_LLM_API_KEY",
    "KG_MINERU_API_URL",
    "KG_MINERU_API_KEY",
)


def test_model_serving_defaults_are_pinned_and_portable() -> None:
    """Keep opt-in inference reproducible and runnable on any recent GPU.

    The defaults are what an outsider gets on whatever GPU they have, so
    nothing here may depend on a vendor build, a quantization kernel, a
    RuntimeClass, or a drafter nobody can download: a public bf16 checkpoint
    at a pinned commit, the multi-architecture engine tag, and every
    hardware-specific knob left for an example profile to set.
    """

    values = _values()["modelServing"]
    server = values["server"]

    assert values["enabled"] is False
    assert values["runtime"] == "vllm"
    assert values["image"]["repository"] == "vllm/vllm-openai"
    assert values["image"]["tag"] == "v0.32.0"
    # No digest: the tag is a multi-architecture index, and the one digest a
    # values file can pin is one architecture's manifest. The schema lets the
    # field stay empty, and a site pins the index digest it resolved.
    assert values["image"]["digest"] == ""
    schema = _schema()["properties"]["modelServing"]["properties"]["image"]
    assert re.fullmatch(schema["properties"]["digest"]["pattern"], "")
    assert values["model"]["name"] == "Qwen/Qwen3-8B"
    assert _COMMIT.fullmatch(values["model"]["revision"])
    assert values["huggingFaceTokenSecret"]["name"] == ""
    assert values["persistence"]["enabled"] is True
    assert values["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert values["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # Nothing a build might lack: the engine reads the scheme from the
    # checkpoint and picks its own kernels.
    for key in ("quantization", "attentionBackend", "moeBackend", "loadFormat"):
        assert server[key] == "", key
    assert server["kvCacheDtype"] == "auto"
    # A discrete GPU's default share; the unified-memory figure is an example.
    assert server["gpuMemoryUtilization"] == 0.90
    # No drafter, no remote code, no RuntimeClass, no modality limits: each is
    # a property of a particular checkpoint or cluster.
    assert server["speculativeTokens"] == 0
    assert server["speculativeMethod"] == ""
    assert server["speculativeDraftModel"] == ""
    assert server["trustRemoteCode"] is False
    assert server["limitMultimodalPerPrompt"] == {}
    assert values["runtimeClassName"] == ""
    # The default model's own parsers, as values rather than hard-coded flags.
    assert server["reasoningParser"] == "qwen3"
    assert server["toolCallParser"] == "hermes"
    # Nothing phones home.
    assert server["usageStats"] is False


def test_a_local_draft_checkpoint_is_not_pinned_to_a_hub_revision() -> None:
    """Keep the drafter's path and its revision from being set together.

    A filesystem path has no Hub commit behind it. Naming one anyway is not a
    no-op: the engine tries to resolve the revision against a repository that
    does not exist and fails at startup, long after the chart looked correct.
    The reference profile is where a path-named drafter lives.
    """

    server = _load_yaml(_SPARK_EXAMPLE)["modelServing"]["server"]

    assert server["speculativeDraftModel"] == "/models/dflash2"
    assert "speculativeDraftRevision" not in server


def test_the_weight_budget_counts_the_drafter_that_stays_resident() -> None:
    """Charge the draft model against the KV budget it actually competes with.

    The drafter occupies device memory for the whole life of the process, so a
    weights figure covering only the target checkpoint overstates how many
    sequences fit and walks the engine into KV-pressure preemption. The
    reference profile carries the measured figure; the default has no drafter
    and charges the bf16 checkpoint alone.
    """

    # The engine reports "Model loading took 24.24 GiB" for the target plus the
    # DFlash2 draft; the target alone is 21.81.
    assert _load_yaml(_SPARK_EXAMPLE)["modelServing"]["sizing"]["weightsGiB"] == 24.24
    # Qwen3-8B's safetensors sum to 15.26 GiB.
    assert _values()["modelServing"]["sizing"]["weightsGiB"] == 15.3


def _sizing_verdict(values: dict[str, Any]) -> Any:
    """Run the sidecar's own sizing check over a modelServing values block."""

    sizing = values["sizing"]
    return compute_sizing(
        ModelGeometry(
            kv_heads=sizing["kvHeads"],
            head_dim=sizing["headDim"],
            attention_layers=sizing["attentionLayers"],
            kv_cache_dtype=values["server"]["kvCacheDtype"],
            weights_bytes=int(sizing["weightsGiB"] * BYTES_PER_GIB),
            recurrent_layers=sizing["recurrentLayers"],
            recurrent_state_bytes_per_layer=sizing["recurrentStateBytesPerLayer"],
            recurrent_state_slots_per_sequence=sizing["recurrentStateSlotsPerSequence"],
        ),
        DeviceBudget(
            device_memory_bytes=int(sizing["deviceMemoryGiB"] * BYTES_PER_GIB),
            gpu_memory_utilization=values["server"]["gpuMemoryUtilization"],
            overhead_bytes=int(sizing["overheadGiB"] * BYTES_PER_GIB),
        ),
        sizing["expectedContextTokens"],
        values["server"]["maxNumSeqs"],
    )


def test_the_shipped_sequence_limit_is_one_the_sizing_formula_supports() -> None:
    """Ship a default that keeps KV-pressure eviction of batch work impossible.

    The chart and the sidecar must agree on this or the pods refuse to start,
    so the contract is checked against the formula rather than a magic number,
    for the portable default and for the reference profile alike.
    """

    for values in (_values()["modelServing"], _load_yaml(_SPARK_EXAMPLE)["modelServing"]):
        verdict = _sizing_verdict(values)
        assert verdict.sequence_limit_binds_first, verdict.detail


def test_the_default_sizing_is_the_worked_example_it_claims_to_be() -> None:
    """The sizing comment in values.yaml walks through numbers; keep them true."""

    values = _values()["modelServing"]
    sizing = values["sizing"]

    # Qwen3-8B: 8 KV heads, head_dim 128, 36 dense layers, on one 80 GB device.
    assert (sizing["kvHeads"], sizing["headDim"], sizing["attentionLayers"]) == (8, 128, 36)
    assert sizing["recurrentLayers"] == 0
    assert sizing["deviceMemoryGiB"] == 80
    kv_bytes_per_token = 2 * 8 * 128 * 36 * 2
    bytes_per_sequence = kv_bytes_per_token * sizing["expectedContextTokens"]
    budget = (80 * 0.90 - sizing["weightsGiB"] - sizing["overheadGiB"]) * BYTES_PER_GIB
    assert int(budget // bytes_per_sequence) == 23
    assert values["server"]["maxNumSeqs"] == 16


def test_the_engine_owns_the_complete_model_lifecycle() -> None:
    """Require storage, pinned model loading, probes, spreading, and service discovery."""

    values = _values()["modelServing"]
    rendered = _render(_SERVING)
    engines = _one(rendered, "StatefulSet", _VLLM)
    pod = _pod(engines)
    engine = _container(pod, "vllm")
    args = _args(engine)

    assert (
        _one(rendered, "Service", _VLLM)["spec"]["selector"]
        == engines["spec"]["selector"]["matchLabels"]
    )
    assert engines["spec"]["serviceName"] == f"{_VLLM}-headless"
    assert [claim["metadata"]["name"] for claim in engines["spec"]["volumeClaimTemplates"]] == [
        "models"
    ]
    # One engine per host, and required: two engines cannot share the GPU.
    required = pod["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert required[0]["labelSelector"]["matchLabels"] == engines["spec"]["selector"]["matchLabels"]
    assert args["--revision"] == values["model"]["revision"]
    assert args["--max-num-batched-tokens"] == str(values["server"]["maxNumBatchedTokens"])
    # A text-only default renders no modality limit; a vision profile does.
    assert "--limit-mm-per-prompt" not in engine["args"]
    vision = _container(
        _pod(_one(_render(_SERVING, values=(_SPARK_EXAMPLE,)), "StatefulSet", _VLLM)), "vllm"
    )
    assert _args(vision)["--limit-mm-per-prompt"] == json.dumps(
        _load_yaml(_SPARK_EXAMPLE)["modelServing"]["server"]["limitMultimodalPerPrompt"],
        separators=(",", ":"),
    )
    # Engine tunables that are environment variables go through extraEnv; the
    # default carries none, the reference profile its Marlin flag.
    assert "VLLM_MARLIN_USE_ATOMIC_ADD" not in _env(engine)
    assert _env(vision)["VLLM_MARLIN_USE_ATOMIC_ADD"]["value"] == "1"
    # No RuntimeClass unless the cluster needs one named.
    assert "runtimeClassName" not in pod
    assert (
        _pod(_one(_render(_SERVING, values=(_SPARK_EXAMPLE,)), "StatefulSet", _VLLM))[
            "runtimeClassName"
        ]
        == "nvidia"
    )
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(_container(pod, "sidecar"))


def test_model_specific_engine_flags_are_values_and_empty_omits_them() -> None:
    """Serve a model other than the default without editing a template.

    Remote code, the reasoning and tool-call parsers, and usage reporting are
    all properties of a checkpoint or a site. Each is a value; an empty one
    renders no flag, so a model that speaks neither parser starts cleanly.
    """

    default = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    assert "--trust-remote-code" not in default["args"]
    assert _args(default)["--reasoning-parser"] == "qwen3"
    assert _args(default)["--tool-call-parser"] == "hermes"
    assert "--enable-auto-tool-choice" in default["args"]
    env = _env(default)
    assert env["VLLM_NO_USAGE_STATS"]["value"] == "1"
    assert env["DO_NOT_TRACK"]["value"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"]["value"] == "1"

    bare = _container(
        _pod(
            _one(
                _render(
                    (
                        *_SERVING,
                        "modelServing.server.reasoningParser=",
                        "modelServing.server.toolCallParser=",
                        "modelServing.server.trustRemoteCode=true",
                        "modelServing.server.usageStats=true",
                    )
                ),
                "StatefulSet",
                _VLLM,
            )
        ),
        "vllm",
    )
    for flag in ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice"):
        assert flag not in bare["args"], flag
    assert "--trust-remote-code" in bare["args"]
    assert "VLLM_NO_USAGE_STATS" not in _env(bare)

    # Speculation without a method is refused by name, not by the engine.
    refusal = _fails((*_SERVING, "modelServing.server.speculativeTokens=4"))
    assert "speculativeMethod is empty" in refusal


def test_the_engine_reads_a_private_hub_and_a_named_loader_only_when_told() -> None:
    """A Hub token and a loader are opt-in, and each reaches the engine intact."""

    gated = _render(
        (
            *_SERVING,
            "modelServing.huggingFaceTokenSecret.name=hub-token",
            "modelServing.server.loadFormat=safetensors",
            "modelServing.server.skipMultimodalProfiling=true",
        )
    )
    engine = _container(_pod(_one(gated, "StatefulSet", _VLLM)), "vllm")
    values = _values()["modelServing"]["huggingFaceTokenSecret"]

    assert _env(engine)["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "hub-token",
        "key": values["key"],
        "optional": values["optional"],
    }
    assert _args(engine)["--load-format"] == "safetensors"
    assert "--skip-mm-profiling" in engine["args"]

    default = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    assert "HF_TOKEN" not in _env(default)
    assert "--load-format" not in default["args"]
    assert "--skip-mm-profiling" not in default["args"]


def test_engine_tuning_flags_can_be_configured_away() -> None:
    """Let an empty value omit a flag rather than render an empty argument.

    Every one of these names a backend, a loader, or a scheme that a given vLLM
    build may not carry, and vLLM fails at startup on an unknown name rather
    than falling back. A chart that can only ever assert one is a chart that
    cannot follow the engine across an upgrade.
    """

    flags = {
        "quantization": "--quantization",
        "attentionBackend": "--attention-backend",
        "moeBackend": "--moe-backend",
        "loadFormat": "--load-format",
    }
    cleared = _render((*_SERVING, *(f"modelServing.server.{key}=" for key in flags)))
    engine = _container(_pod(_one(cleared, "StatefulSet", _VLLM)), "vllm")

    for flag in flags.values():
        assert flag not in engine["args"], flag
    assert "" not in engine["args"]


def test_priority_scheduling_cannot_be_configured_away() -> None:
    """Render the flag unconditionally; without it every stamp is silently ignored.

    There is no values key that omits it, because an engine serving FIFO looks
    exactly like an engine honouring priority until someone measures the wait.
    """

    schema = _schema()
    server = schema["properties"]["modelServing"]["properties"]["server"]

    assert server["additionalProperties"] is False
    assert not any("scheduling" in key.lower() for key in server["properties"])
    for rendered in (
        _render(_SERVING),
        _render((*_SERVING, "modelServing.server.quantization=")),
        _render((), values=(_SPARK_EXAMPLE,)),
    ):
        engine = _container(_pod(_one(rendered, "StatefulSet", _VLLM)), "vllm")
        assert _args(engine)["--scheduling-policy"] == "priority"


def test_the_engine_is_reachable_only_through_the_enforcement_floor() -> None:
    """Bind the engine to loopback so no route to inference skips the sidecar."""

    values = _values()["modelServing"]
    rendered = _render(_SERVING)
    pod = _pod(_one(rendered, "StatefulSet", _VLLM))
    engine = _container(pod, "vllm")
    sidecar = _container(pod, "sidecar")
    sidecar_env = _env(sidecar)

    assert _args(engine)["--host"] == "127.0.0.1"
    assert _args(engine)["--port"] == str(values["server"]["enginePort"])
    # The engine's own port is never a container port, so no Service can name it.
    assert values["server"]["enginePort"] not in {
        port["containerPort"] for container in pod["containers"] for port in container["ports"]
    }
    assert sidecar["command"] == ["flakegraph", "serving", "sidecar"]
    assert sidecar_env["FLAKEGRAPH_SIDECAR_UPSTREAM"]["value"] == (
        f"http://127.0.0.1:{values['server']['enginePort']}"
    )
    assert sidecar_env["FLAKEGRAPH_SIDECAR_KEYS_FILE"]["value"].endswith(
        values["sidecar"]["keySecret"]["key"]
    )
    assert (
        json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_BANDS"]["value"]) == values["sidecar"]["bands"]
    )
    # The sidecar refuses to start on a limit the KV budget cannot support, so
    # the numbers it checks must actually reach it.
    geometry = json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_GEOMETRY"]["value"])
    assert geometry["kv_heads"] == values["sizing"]["kvHeads"]
    assert geometry["weights_bytes"] == int(values["sizing"]["weightsGiB"] * BYTES_PER_GIB)
    budget = json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_DEVICE_BUDGET"]["value"])
    assert budget["gpu_memory_utilization"] == values["server"]["gpuMemoryUtilization"]
    assert sidecar_env["FLAKEGRAPH_SIDECAR_MAX_NUM_SEQS"]["value"] == str(
        values["server"]["maxNumSeqs"]
    )
    # Probes belong to the floor: a healthy engine behind an unhealthy sidecar
    # is not a servable replica.
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(sidecar)
    assert not {"startupProbe", "readinessProbe", "livenessProbe"} & set(engine)

    policy = _one(rendered, "NetworkPolicy", _VLLM)
    router = {"app.kubernetes.io/component": "inference-router"}
    admitted = {
        port["port"]
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for port in rule["ports"]
        if router.items() <= peer["podSelector"]["matchLabels"].items()
    }
    assert "http" in admitted
    assert values["server"]["enginePort"] not in admitted


def test_the_engine_pod_accepts_a_site_supplied_trust_store() -> None:
    """Let an operator give the engine the CA bundle its egress requires.

    The pod downloads its own weights. Where TLS is terminated on the way out,
    Hub metadata still resolves while the weight transfer fails verification —
    which presents as a stalled download, not an error, so the escape hatch has
    to exist rather than being discovered under time pressure.
    """

    values = _values()["modelServing"]
    for field in ("extraEnv", "extraVolumes", "extraVolumeMounts"):
        assert values[field] == []

    rendered = _render(
        (
            *_SERVING,
            "modelServing.extraEnv[0].name=SSL_CERT_FILE",
            "modelServing.extraEnv[0].value=/etc/ssl/site/ca.crt",
            "modelServing.extraVolumes[0].name=site-ca",
            "modelServing.extraVolumes[0].secret.secretName=site-ca",
            "modelServing.extraVolumeMounts[0].name=site-ca",
            "modelServing.extraVolumeMounts[0].mountPath=/etc/ssl/site",
        )
    )
    pod = _pod(_one(rendered, "StatefulSet", _VLLM))
    engine = _container(pod, "vllm")

    # The mounts belong to the engine, which is the container that downloads.
    assert _env(engine)["SSL_CERT_FILE"]["value"] == "/etc/ssl/site/ca.crt"
    assert {"name": "site-ca", "mountPath": "/etc/ssl/site"} in engine["volumeMounts"]
    assert {"name": "site-ca", "secret": {"secretName": "site-ca"}} in pod["volumes"]
    sidecar = _container(pod, "sidecar")
    assert "SSL_CERT_FILE" not in _env(sidecar)
    assert "site-ca" not in {mount["name"] for mount in sidecar["volumeMounts"]}


def test_the_gateway_expresses_priority_as_which_alias_a_key_may_call() -> None:
    """Keep the class-to-band mapping out of the request body entirely."""

    values = _values()["gateway"]
    rendered = _render(())
    config = yaml.safe_load(
        _one(rendered, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]
    )
    gateway = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-litellm")), "litellm")
    env = _env(gateway)

    assert set(values["litellm"]["aliases"]) == {"interactive", "dev", "batch"}
    # Each alias must present a *different* upstream key. Sharing one would make
    # every class the same band while still looking correctly configured.
    upstream = values["litellm"]["upstreamKeySecret"]
    served = {entry["model_name"]: entry["litellm_params"] for entry in config["model_list"]}
    assert set(served) == set(values["litellm"]["aliases"].values())
    assert {params["api_key"] for params in served.values()} == {
        f"os.environ/{upstream[field]}" for field in ("interactiveKey", "devKey", "batchKey")
    }
    assert len({params["api_key"] for params in served.values()}) == 3
    for field in ("interactiveKey", "devKey", "batchKey"):
        assert env[upstream[field]]["valueFrom"]["secretKeyRef"] == {
            "name": upstream["name"],
            "key": upstream[field],
        }
    # Body injection would be stripped by the sidecar and would collide with
    # LiteLLM's own reserved field name besides.
    assert "priority" not in json.dumps(config)
    # Pointed at FlakeGraph's own schema, LiteLLM's migration tool baselines
    # instead of migrating and silently creates none of its tables.
    assert values["litellm"]["databaseUrlSuffix"].endswith("schema=litellm")
    assert env["DATABASE_URL"]["value"] == (
        f"$(FLAKEGRAPH_DATABASE_URI){values['litellm']['databaseUrlSuffix']}"
    )
    assert (
        config["general_settings"]["proxy_batch_write_at"]
        == (values["litellm"]["spendLogging"]["batchWriteIntervalSeconds"])
    )


def test_placement_routes_on_the_pickers_choice_and_needs_no_crds() -> None:
    """Keep the chart installable where no Inference Extension CRDs exist."""

    rendered = _render(_SERVING)
    router = _one(rendered, "ConfigMap", _ROUTER)["data"]
    envoy = yaml.safe_load(router["envoy.yaml"])
    plugins = yaml.safe_load(router["default-plugins.yaml"])
    picker = _container(_pod(_one(rendered, "Deployment", _ROUTER)), "endpoint-picker")

    clusters = {cluster["name"]: cluster for cluster in envoy["static_resources"]["clusters"]}
    assert clusters["original_destination"]["type"] == "ORIGINAL_DST"
    assert clusters["original_destination"]["original_dst_lb_config"] == {
        "use_http_header": True,
        "http_header_name": "x-gateway-destination-endpoint",
    }
    listener = next(
        listener
        for listener in envoy["static_resources"]["listeners"]
        if listener["name"] == "inference"
    )
    http_filters = listener["filter_chains"][0]["filters"][0]["typed_config"]["http_filters"]
    assert "envoy.filters.http.ext_proc" in {http_filter["name"] for http_filter in http_filters}
    assert _args(picker)["--endpoint-selector"].endswith(
        "app.kubernetes.io/component=model-serving"
    )
    assert "prefix-cache-scorer" in {plugin["type"] for plugin in plugins["plugins"]}
    # An InferencePool would make a CRD a precondition for installing at all.
    assert not [doc for doc in rendered if doc["kind"] == "InferencePool"]
    role = _one(rendered, "Role", _ROUTER)
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "watch", "list"]}
    ]


def test_the_console_runs_the_fleet_the_way_the_workers_do() -> None:
    """The console is the Node server, and its CLI sees the workers' fleet.

    The CLI the console shells out to submits to the same queue, reads the same
    artifact store, and resolves the same provider placeholders as the workers
    - a run it writes is one they claim, a graph they publish is one it reads.
    Behind the gate it trusts the gate's identity headers; without one it does
    not.
    """

    values = _values()
    gated = _render(
        (
            *_SERVING,
            *_GATED,
            "spark.enabled=true",
            "artifactStorage.uri=s3://artifacts",
            "artifactStorage.existingSecret=artifacts",
            "controlPlane.ask.secretName=ask",
        )
    )
    pod = _pod(_one(gated, "Deployment", f"{_FULLNAME}-app"))
    console = _container(pod, "control-plane")
    env = _env(console)
    worker = _env(_worker(gated, "extract"))

    assert console["command"] == ["node", "server.js"]
    assert console["workingDir"] == "/app/react"
    assert env["PORT"]["value"] == str(values["controlPlane"]["service"]["port"])
    assert env["FLAKEGRAPH_CLI"]["value"] == "flakegraph"
    assert env["FLAKEGRAPH_APP_DEFAULT_RUNTIME"]["value"] == "kubernetes"
    assert env["FLAKEGRAPH_TRUST_IDENTITY_HEADERS"]["value"] == "true"
    for name in (
        "KG_DISTRIBUTED_DATABASE_URL",
        "KG_DISTRIBUTED_ARTIFACT_URI",
        "KG_LLM_ENDPOINT",
        "KG_MINERU_API_URL",
    ):
        assert env[name] == worker[name], name
    assert env["DATABASE_URL"]["valueFrom"] == env["KG_DISTRIBUTED_DATABASE_URL"]["valueFrom"]
    assert env["FLAKEGRAPH_ASK_BASE_URL"]["value"] == env["KG_LLM_ENDPOINT"]["value"]
    assert env["FLAKEGRAPH_ASK_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "ask",
        "key": values["controlPlane"]["ask"]["secretKey"],
    }
    # Helper calls (planning, scoring, suggestions) ask the reasoning model
    # for no reasoning: measured 15 s against 2.7 s for one plan.
    assert env["FLAKEGRAPH_ASK_STRUCTURED_REASONING"]["value"] == "none"
    assert {
        probe["httpGet"]["path"]
        for probe in (console["startupProbe"], console["readinessProbe"], console["livenessProbe"])
    } == {"/api/health"}
    assert {mount["mountPath"] for mount in console["volumeMounts"]} >= {
        "/app/react/.next/cache",
        "/tmp",
        values["controlPlane"]["persistence"]["stateRoot"],
    }

    ungated = _env(
        _container(
            _pod(_one(_render(_ISOLATED), "Deployment", f"{_FULLNAME}-app")), "control-plane"
        )
    )
    assert "FLAKEGRAPH_TRUST_IDENTITY_HEADERS" not in ungated
    assert "FLAKEGRAPH_ASK_BASE_URL" not in ungated


def test_the_control_plane_can_see_what_preflight_asks_about() -> None:
    """A pool scaled to zero on purpose is told apart by its ScaledObject."""

    role = _one(_render(_ISOLATED), "Role", f"{_FULLNAME}-app")
    granted = {
        (group, resource)
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
        if "list" in rule["verbs"]
    }
    assert ("keda.sh", "scaledobjects") in granted
    assert ("apps", "statefulsets") in granted
    # Secrets are read by name and never listed: a console compromise must
    # not be a dump of every credential in the namespace.
    assert ("", "secrets") not in granted
    (secret_rule,) = [rule for rule in role["rules"] if rule["resources"] == ["secrets"]]
    assert secret_rule["verbs"] == ["get"]
    values = _values()
    assert set(secret_rule["resourceNames"]) == {
        values["database"]["secretName"],
        values["providerSecret"]["name"],
        values["gateway"]["litellm"]["virtualKeySecret"]["name"],
        values["gateway"]["litellm"]["upstreamKeySecret"]["name"],
        values["documentParsing"]["shim"]["keySecret"]["name"],
    }
    # Recovery may restart the worker pools and nothing else: the write grant
    # names them.
    writes = [
        rule for rule in role["rules"] if "patch" in rule["verbs"] or "update" in rule["verbs"]
    ]
    assert len(writes) == 1
    assert set(writes[0]["resources"]) == {"deployments", "deployments/scale"}
    assert set(writes[0]["resourceNames"]) == {
        f"{_FULLNAME}-prepare",
        f"{_FULLNAME}-extract",
        f"{_FULLNAME}-finalize",
    }


def test_document_parsing_holds_work_rather_than_letting_it_fail() -> None:
    """Give the shim a resolvable pool and the capacity it must not exceed."""

    values = _values()
    parsing = values["documentParsing"]
    rendered = _render(())
    pool = _one(rendered, "StatefulSet", f"{_FULLNAME}-mineru")
    parser = _container(_pod(pool), "mineru")
    shim = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-ocr")), "ocr-shim")
    shim_env = _env(shim)

    # MinerU's own default of three is too low to keep a node busy.
    assert parsing["mineru"]["maxConcurrentRequests"] > 3
    assert shim["command"] == ["flakegraph", "serving", "ocr-shim"]
    assert shim_env["FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY"]["value"] == str(
        parsing["mineru"]["maxConcurrentRequests"]
    )
    assert shim_env["FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"]["value"] == f"{_FULLNAME}-mineru"
    assert shim_env["FLAKEGRAPH_OCR_SHIM_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": values["database"]["secretName"],
        "key": values["database"]["secretKey"],
    }
    assert _env(parser)["MINERU_API_MAX_CONCURRENT_REQUESTS"]["value"] == str(
        parsing["mineru"]["maxConcurrentRequests"]
    )
    assert _args(parser)["--host"] == "0.0.0.0"
    # Headless: a load-balanced ClusterIP would hide the per-replica load that
    # admission control has to count.
    assert _one(rendered, "Service", f"{_FULLNAME}-mineru")["spec"]["clusterIP"] == "None"
    # The pool authenticates nobody and answers 409 to whatever skips the
    # queue, so the shim is the only pod allowed to reach it.
    policy = _one(rendered, "NetworkPolicy", f"{_FULLNAME}-mineru")
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    assert [
        (peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"], port["port"])
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for port in rule["ports"]
    ] == [("ocr-shim", "http")]
    # The parsing pool runs the same image as everything else. MinerU is already
    # installed in it and exposes an entry point, so a second image would only
    # add another artefact to keep on the right architecture.
    assert parser["command"] == ["mineru-api"]
    assert "image" not in parsing["mineru"]
    worker = _worker(rendered, "extract")
    assert parser["image"] == worker["image"]
    assert shim["image"] == worker["image"]
    # Every path the parser writes to has to be one of the writable mounts; the
    # image's defaults sit inside the read-only layer.
    writable = tuple(mount["mountPath"] for mount in parser["volumeMounts"])
    for variable in ("MINERU_API_OUTPUT_ROOT", "XDG_CACHE_HOME", "HF_HOME", "HOME"):
        assert _env(parser)[variable]["value"].startswith(writable), variable
    # The worker waits at least as long as the shim holds a document. Giving up
    # sooner leaves the pool parsing for nobody and retries the parse from the
    # start; the workers' grace period then covers the parse it is finishing.
    config = yaml.safe_load(values["config"]["content"])
    assert config["ocr"]["timeout_seconds"] >= parsing["shim"]["requestTimeoutSeconds"]
    assert values["terminationGracePeriodSeconds"] > config["ocr"]["timeout_seconds"]


def test_the_sidecar_can_hold_its_tag_while_the_application_moves() -> None:
    """A release of the application must not have to roll every engine.

    The sidecar is the engine pod's one container built from the application
    image, and it rarely changes. Pinned, the engines stay up through a worker
    release; unpinned, it follows the application tag.
    """

    following = _render((*_SERVING, "image.tag=0.3.11"))
    pinned = _render((*_SERVING, "image.tag=0.3.11", "modelServing.sidecar.imageTag=0.3.10"))

    for rendered, tag in ((following, "0.3.11"), (pinned, "0.3.10")):
        pod = _pod(_one(rendered, "StatefulSet", _VLLM))
        assert _container(pod, "sidecar")["image"].endswith(f"/flakegraph:{tag}")
        assert _worker(rendered, "extract")["image"].endswith("/flakegraph:0.3.11")


def test_document_parsing_claims_a_gpu_unless_told_to_share_one() -> None:
    """On the GPU the parser claims a whole device by default.

    A claim is what makes the scheduler place the replica on a node with a
    free GPU and account for it. The one case that cannot claim - a single-GPU
    node whose engine already holds the device - opts out with claimGpu=false
    and is handed every device on the node through a RuntimeClass instead.
    Off the GPU nothing about the pod mentions a device.
    """

    values = _values()["documentParsing"]["mineru"]
    cpu = _pod(_one(_render(()), "StatefulSet", f"{_FULLNAME}-mineru"))
    claimed = _pod(
        _one(
            _render(("documentParsing.mineru.device.mode=cuda",)),
            "StatefulSet",
            f"{_FULLNAME}-mineru",
        )
    )
    shared = _pod(_one(_render((), values=(_SPARK_EXAMPLE,)), "StatefulSet", f"{_FULLNAME}-mineru"))

    assert "runtimeClassName" not in cpu
    assert _env(_container(cpu, "mineru"))["MINERU_DEVICE_MODE"]["value"] == "cpu"
    assert "NVIDIA_VISIBLE_DEVICES" not in _env(_container(cpu, "mineru"))
    assert "nvidia.com/gpu" not in _container(cpu, "mineru")["resources"]["limits"]

    assert values["device"]["claimGpu"] is True
    assert "runtimeClassName" not in claimed
    parser = _container(claimed, "mineru")
    assert _env(parser)["MINERU_DEVICE_MODE"]["value"] == "cuda"
    assert "NVIDIA_VISIBLE_DEVICES" not in _env(parser)
    assert parser["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert parser["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # The rest of the resource block is untouched by the claim.
    assert parser["resources"]["limits"]["memory"] == values["resources"]["limits"]["memory"]
    assert _env(parser)["MINERU_VIRTUAL_VRAM_SIZE"]["value"] == str(
        values["device"]["virtualVramGiB"]
    )

    assert shared["runtimeClassName"] == "nvidia"
    sharer = _container(shared, "mineru")
    assert _env(sharer)["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"
    assert "nvidia.com/gpu" not in sharer["resources"]["limits"]


def test_document_parsing_replicas_spread_across_hosts_without_requiring_it() -> None:
    """One parser per host where the fleet allows it, and still schedulable where not.

    The engines are forced apart by their GPU request; a parsing replica holds
    no GPU, so without a constraint four of them can land on one host and leave
    three idle. The constraint is soft on purpose: a pool larger than the fleet
    doubles up rather than pending forever.
    """

    values = _values()["documentParsing"]["mineru"]
    pool = _one(_render(()), "StatefulSet", f"{_FULLNAME}-mineru")

    assert _pod(pool)["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": values["topologyKey"],
            "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {"matchLabels": pool["spec"]["selector"]["matchLabels"]},
        }
    ]
    assert values["topologyKey"] == "kubernetes.io/hostname"


def test_spark_executor_spreading_degrades_gracefully() -> None:
    """Prefer fleet-wide placement without deadlocking finalization on node loss."""

    values = _values()["spark"]
    executor = _executor_template(_render(_SPARK))
    (container,) = executor["spec"]["containers"]

    preferred = executor["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert preferred == [
        {
            "weight": 100,
            "podAffinityTerm": {
                "topologyKey": values["topologyKey"],
                "labelSelector": {"matchLabels": {"spark-role": "executor"}},
            },
        }
    ]
    assert (
        "requiredDuringSchedulingIgnoredDuringExecution"
        not in executor["spec"]["affinity"]["podAntiAffinity"]
    )
    assert executor["spec"]["securityContext"] == values["executorPodSecurityContext"]
    assert container["securityContext"] == values["executorContainerSecurityContext"]
    assert values["executorPodSecurityContext"]["runAsUser"] == 185
    assert values["executorContainerSecurityContext"]["allowPrivilegeEscalation"] is False


def test_the_pipeline_is_a_metered_consumer_like_any_other() -> None:
    """Send batch traffic through the same gateway, holding a batch virtual key.

    Routing workers through the gateway keeps fleet consumption visible and
    lets the endpoint picker distribute work across engines independently of
    worker placement.
    """

    values = _values()
    litellm = values["gateway"]["litellm"]
    rendered = _render(
        (
            "workers.extract.topologySpreadConstraints[0].maxSkew=1",
            "workers.extract.topologySpreadConstraints[0].topologyKey=kubernetes.io/hostname",
            "workers.extract.topologySpreadConstraints[0].whenUnsatisfiable=ScheduleAnyway",
        )
    )

    for pool in _WORKER_POOLS:
        env = _env(_worker(rendered, pool))
        assert env["KG_LLM_ENDPOINT"]["value"] == (
            f"http://{_FULLNAME}-litellm:{litellm['service']['port']}/v1"
        )
        assert env["KG_LLM_MODEL"]["value"] == litellm["aliases"]["batch"]
        assert env["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": litellm["virtualKeySecret"]["name"],
            "key": litellm["virtualKeySecret"]["batchKey"],
        }
        assert env["KG_MINERU_API_URL"]["value"] == (
            f"http://{_FULLNAME}-ocr:{values['documentParsing']['shim']['service']['port']}"
        )
        # Embeddings stay an independently configured provider.
        assert "KG_EMBED_ENDPOINT" not in env
        assert "KG_EMBED_MODEL" not in env
    extract = _pod(_one(rendered, "Deployment", f"{_FULLNAME}-extract"))
    assert extract["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "ScheduleAnyway",
        }
    ]


def test_the_llm_credential_is_declared_exactly_once_per_container() -> None:
    """Never let two sources define KG_LLM_API_KEY in one pod spec.

    Kubernetes resolves a duplicated variable by ordering, not by intent, and an
    optional key that is absent can blank a value another source just set. With
    the gateway on, its virtual key is the credential and the provider Secret's
    mapping must stand aside.
    """

    litellm = _values()["gateway"]["litellm"]
    with_gateway = _render(("providerSecret.name=provider",))
    without_gateway = _render(("providerSecret.name=provider", "gateway.enabled=false"))

    consumers = [(pool, _worker) for pool in _WORKER_POOLS] + [
        ("bootstrap", lambda r, _: _bootstrap(r))
    ]
    for pool, container in consumers:
        gateway_consumer = container(with_gateway, pool)
        names = [entry["name"] for entry in gateway_consumer["env"]]
        assert len(names) == len(set(names)), pool
        assert (
            _env(gateway_consumer)["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"]
            == (litellm["virtualKeySecret"]["name"])
        )
        provider_consumer = container(without_gateway, pool)
        assert _env(provider_consumer)["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == (
            "provider"
        )


def test_the_validator_is_configured_exactly_like_what_it_validates() -> None:
    """Emit one env block for workers and for the Job that preflights them.

    The bootstrap Job runs preflight against the same profile the workers run.
    Configured separately, the two have drifted in both directions: the Job has
    passed a profile the workers could not execute, and failed one they could.
    """

    rendered = _render(_SERVING)
    bootstrap = _first_declared(_bootstrap(rendered))

    for pool in _WORKER_POOLS:
        worker = _first_declared(_worker(rendered, pool))
        for name in _CONSUMER_ENV:
            assert worker[name] == bootstrap[name], (pool, name)
    # The Job must not be left pointing at an engine the consumers do not use.
    for entry in bootstrap.values():
        assert _VLLM not in entry.get("value", "")


def test_no_consumer_can_still_address_an_engine_directly() -> None:
    """Leave one endpoint key, so the placement layer stays replaceable."""

    schema = _schema()
    rendered = _render((), values=(_SPARK_EXAMPLE,))

    assert schema["additionalProperties"] is False
    assert "localModelServing" not in schema["properties"]
    assert "localModelServing" not in _load_yaml(_SPARK_EXAMPLE)
    assert not any(
        "internalTrafficPolicy" in doc["spec"] for doc in rendered if doc["kind"] == "Service"
    )
    consumers = [_worker(rendered, pool) for pool in _WORKER_POOLS] + [_bootstrap(rendered)]
    for consumer in consumers:
        for entry in consumer["env"]:
            assert _VLLM not in entry.get("value", ""), (consumer["name"], entry["name"])


def test_provider_secret_import_is_an_explicit_credential_allowlist() -> None:
    """Prevent unrelated Secret keys from silently replacing reviewed provider settings."""

    values = _values()
    rendered = _render((*_ISOLATED, "providerSecret.name=provider"))

    assert values["providerSecret"]["env"] == [
        {"name": "KG_LLM_API_KEY", "key": "KG_LLM_API_KEY", "optional": True},
        {"name": "KG_EMBED_API_KEY", "key": "KG_EMBED_API_KEY", "optional": True},
        {
            "name": "KG_SNOWFLAKE_PASSWORD",
            "key": "KG_SNOWFLAKE_PASSWORD",
            "optional": True,
        },
        {
            "name": "KG_SNOWFLAKE_OAUTH_TOKEN",
            "key": "KG_SNOWFLAKE_OAUTH_TOKEN",
            "optional": True,
        },
        {"name": "AWS_ACCESS_KEY_ID", "key": "AWS_ACCESS_KEY_ID", "optional": True},
        {"name": "AWS_SECRET_ACCESS_KEY", "key": "AWS_SECRET_ACCESS_KEY", "optional": True},
    ]
    # A mapping without `optional` would render null and be applied as a hard
    # requirement, so the item must spell all three keys.
    assert "missing property 'optional'" in _fails(
        ("providerSecret.env[0].name=KG_X", "providerSecret.env[0].key=KG_X")
    )
    allowlisted = {mapping["name"] for mapping in values["providerSecret"]["env"]}
    # The control plane lists sources and preflights with the same credentials
    # the workers will run with; a bucket it cannot see, they cannot read.
    control_plane = _container(
        _pod(_one(rendered, "Deployment", f"{_FULLNAME}-app")), "control-plane"
    )
    consumers = [_worker(rendered, pool) for pool in _WORKER_POOLS] + [
        _bootstrap(rendered),
        control_plane,
    ]
    for consumer in consumers:
        assert "envFrom" not in consumer, consumer["name"]
        imported = {
            entry["name"]
            for entry in consumer["env"]
            if entry.get("valueFrom", {}).get("secretKeyRef", {}).get("name") == "provider"
        }
        assert imported, consumer["name"]
        assert imported <= allowlisted, consumer["name"]


def test_ontology_is_a_portable_chart_managed_deployment_input() -> None:
    """Mount ontology content without baking repository datasets into worker images."""

    values = _values()
    inline = _render(("ontology.content=entities: []",))
    external = _render(("ontology.existingConfigMap=site-ontology",))

    assert values["ontology"] == {
        "existingConfigMap": "",
        "key": "ontology.yaml",
        "content": "",
    }
    assert _one(inline, "ConfigMap", f"{_FULLNAME}-ontology")["data"] == {
        "ontology.yaml": "entities: []\n"
    }
    assert not [doc for doc in external if doc["metadata"]["name"] == f"{_FULLNAME}-ontology"]
    for rendered, configmap in ((inline, f"{_FULLNAME}-ontology"), (external, "site-ontology")):
        for consumer in [_worker(rendered, pool) for pool in _WORKER_POOLS] + [
            _bootstrap(rendered)
        ]:
            assert _env(consumer)["KG_ONTOLOGY_PROFILE"]["value"] == (
                "/etc/flakegraph-ontology/ontology.yaml"
            )
        for pool in _WORKER_POOLS:
            deployment = _one(rendered, "Deployment", f"{_FULLNAME}-{pool}")
            assert "checksum/ontology" in deployment["spec"]["template"]["metadata"]["annotations"]
            volumes = {volume["name"]: volume for volume in _pod(deployment)["volumes"]}
            assert volumes["ontology"]["configMap"]["name"] == configmap


def test_worker_stage_names_follow_the_distributed_dag() -> None:
    """Keep typed Helm values aligned with role-specific worker pools."""

    values = _values()
    schema = _schema()
    stage_enum = schema["$defs"]["worker"]["properties"]["stages"]["items"]["enum"]
    rendered = _render(())

    assert values["workers"]["prepare"]["stages"] == ["prepare_document"]
    assert values["workers"]["extract"]["stages"] == [
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "revalidate_document_context",
        "revalidate_entity_window",
        "revalidate_relation_window",
        "revalidate_document",
    ]
    assert values["workers"]["finalize"]["stages"] == ["finalize_graph"]
    assert stage_enum == [
        "prepare_document",
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "revalidate_document_context",
        "revalidate_entity_window",
        "revalidate_relation_window",
        "revalidate_document",
        "finalize_graph",
    ]
    # Every stage is served by exactly one pool, and no pool serves a stage the
    # schema does not know.
    served = [stage for pool in _WORKER_POOLS for stage in _stages(_worker(rendered, pool))]
    assert sorted(served) == sorted(stage_enum)


def test_images_default_to_the_published_release() -> None:
    """An install from a release tag pulls the images that release published.

    The publish workflow pushes both images to GHCR for every v* tag, so the
    chart names that registry and follows its own appVersion. Release tags are
    immutable, so nodes may keep what they pulled; a fork overrides the
    repository, a production site adds a digest.
    """

    values = _values()
    rendered = _render(_SPARK)
    version = yaml.safe_load(Path("deploy/helm/flakegraph/Chart.yaml").read_text())["appVersion"]

    assert values["image"]["repository"] == "ghcr.io/leo-pharma-r-d-data-analytics/flakegraph"
    assert values["spark"]["image"]["repository"] == (
        "ghcr.io/leo-pharma-r-d-data-analytics/flakegraph-spark"
    )
    assert values["image"]["tag"] == ""
    assert values["spark"]["image"]["tag"] == ""
    for block in (values["image"], values["spark"]["image"], values["modelServing"]["image"]):
        assert block["pullPolicy"] == "IfNotPresent"
    assert _worker(rendered, "extract")["image"] == f"{values['image']['repository']}:{version}"
    assert _worker(rendered, "finalize")["image"] == (
        f"{values['spark']['image']['repository']}:{version}"
    )
    assert _env(_worker(rendered, "finalize"))["KG_DISTRIBUTED_SPARK_IMAGE"]["value"] == (
        f"{values['spark']['image']['repository']}:{version}"
    )


def test_workers_seed_preloaded_models_into_their_writable_cache() -> None:
    """Prevent the read-only runtime cache mount from hiding baked checkpoints."""

    rendered = _render(_SPARK)

    for pool in _WORKER_POOLS:
        deployment = _one(rendered, "Deployment", f"{_FULLNAME}-{pool}")
        seed = _container(_pod(deployment), "seed-provider-cache", "initContainers")
        (script,) = seed["args"]
        # The init container runs as an unprivileged image user. Recursive copy
        # is sufficient; archive mode would also preserve timestamps/ownership
        # and can fail against an EmptyDir prepared by Kubernetes securityContext
        # settings.
        assert "cp -R " in script
        assert "cp -a " not in script
        assert seed["resources"]["limits"]["memory"] == "1Gi"
        assert seed["volumeMounts"] == [{"name": "cache", "mountPath": "/cache"}]
        worker = _worker(rendered, pool)
        mounts = {mount["name"]: mount["mountPath"] for mount in worker["volumeMounts"]}
        if pool == "finalize":
            # The Spark image's baked cache belongs to executor UID 185, so the
            # driver reads its copy through a neutral mount, not another home.
            assert "/home/spark/.cache/." in script
            assert mounts["cache"] == "/cache"
            assert _env(worker)["HF_HOME"]["value"] == "/cache/huggingface"
            assert _env(worker)["SENTENCE_TRANSFORMERS_HOME"]["value"] == (
                "/cache/sentence_transformers"
            )
        else:
            assert "/home/kgprocessor/.cache/." in script
            assert mounts["cache"] == "/home/kgprocessor/.cache"


def test_spark_workers_reserve_memory_for_python_provider_processes() -> None:
    """Keep executor heap and non-heap memory explicit across Helm boundaries."""

    finalize = _env(_worker(_render(_SPARK), "finalize"))

    assert finalize["KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY"]["value"] == "8g"
    assert finalize["KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD"]["value"] == "8g"


def test_spark_provider_concurrency_reaches_the_driver_only_when_set() -> None:
    """Unset, the mounted configuration decides; set, the chart's figure wins."""

    default = _env(_worker(_render(_SPARK), "finalize"))
    sized = _env(_worker(_render((*_SPARK, "spark.providerConcurrency=64")), "finalize"))

    assert "KG_GRAPH_FINALIZATION_PROVIDER_CONCURRENCY" not in default
    assert sized["KG_GRAPH_FINALIZATION_PROVIDER_CONCURRENCY"]["value"] == "64"
    assert "providerConcurrency" in _fails((*_SPARK, "spark.providerConcurrency=0"))


def test_the_reference_fleet_example_renders_against_the_charts_own_schema() -> None:
    """The example is the documented path onto a DGX Spark fleet, so it has to install.

    It also has to carry, explicitly, everything that fleet relies on and the
    chart's default leaves out: the NVFP4 checkpoint, the arm64 engine digest,
    the drafter, the RuntimeClass, the unified-memory sizing.
    """

    example = _load_yaml(_SPARK_EXAMPLE)
    rendered = _render((), values=(_SPARK_EXAMPLE,))

    engines = _one(rendered, "StatefulSet", _VLLM)
    assert engines["spec"]["replicas"] == 4
    engine = _container(_pod(engines), "vllm")
    assert engine["image"].endswith(
        "@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc"
    )
    args = _args(engine)
    assert args["--quantization"] == "compressed-tensors"
    assert args["--kv-cache-dtype"] == "fp8"
    assert args["--attention-backend"] == "flashinfer"
    assert args["--moe-backend"] == "marlin"
    assert args["--tool-call-parser"] == "qwen3_xml"
    assert "--trust-remote-code" in engine["args"]
    speculative = json.loads(args["--speculative-config"])
    assert speculative == {
        "method": "dflash",
        "num_speculative_tokens": 8,
        "model": "/models/dflash2",
    }
    seeder = _container(_pod(engines), "seed-draft-model", "initContainers")
    assert seeder["image"] == example["modelServing"]["server"]["draftModelSeed"]["image"]
    assert example["modelServing"]["model"]["name"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert example["modelServing"]["server"]["gpuMemoryUtilization"] == 0.50
    assert example["modelServing"]["sizing"]["deviceMemoryGiB"] == 119.2
    # Every GPU node the bootstrap script joins carries the class the example
    # selects on, the finalize pool included: a label no script applies would
    # leave that pool Pending.
    bootstrap = Path("deploy/spark/bootstrap-node.sh").read_text(encoding="utf-8")
    for pool in ("prepare", "finalize"):
        selector = example["workers"][pool]["nodeSelector"]["flakegraph.io/node-class"]
        assert f"FLAKEGRAPH_NODE_CLASS:-{selector}" in bootstrap, pool
    env = _env(_worker(rendered, "finalize"))
    assert env["KG_DISTRIBUTED_FINALIZATION_ENGINE"]["value"] == "spark"
    # Two provider calls per executor core fill the engines' sequences.
    spark = example["spark"]
    serving = example["modelServing"]
    assert spark["providerConcurrency"] == 2 * spark["executorInstances"] * spark["executorCores"]
    assert spark["providerConcurrency"] == serving["replicas"] * serving["server"]["maxNumSeqs"]
    assert env["KG_GRAPH_FINALIZATION_PROVIDER_CONCURRENCY"]["value"] == "32"
    assert env["KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"] == {
        "name": "flakegraph-artifacts",
        "key": "access-key-id",
    }
    assert [doc["kind"] for doc in rendered if doc["kind"] == "ScaledObject"]
    assert _one(rendered, "Ingress", _FULLNAME)["metadata"]["annotations"][
        "traefik.ingress.kubernetes.io/router.middlewares"
    ]


def test_the_gateway_can_forward_to_an_external_provider_with_no_engine() -> None:
    """Serve a corpus with no GPU in the cluster at all.

    Every alias resolves to the named provider models with one key from a
    Secret; the placement router, its RBAC and the sidecar keys are not
    rendered, because there is nothing to place across; and the pre-install
    check asks for the provider key instead of the serving keys.
    """

    rendered = _render(_EXTERNAL)
    config = yaml.safe_load(
        _one(rendered, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]
    )
    aliases = _values()["gateway"]["litellm"]["aliases"]

    assert {entry["model_name"] for entry in config["model_list"]} == set(aliases.values())
    for entry in config["model_list"]:
        assert entry["litellm_params"] == {
            "model": "hosted_vllm/Qwen/Qwen3-8B",
            "api_base": "https://inference.example/v1",
            "api_key": "os.environ/FLAKEGRAPH_UPSTREAM_API_KEY",
        }
    gateway = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-litellm")), "litellm")
    assert _env(gateway)["FLAKEGRAPH_UPSTREAM_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "upstream",
        "key": "api-key",
    }
    assert "SIDECAR_KEY_BATCH" not in _env(gateway)
    assert not [doc for doc in rendered if doc["metadata"]["name"] == _ROUTER]
    check = _container(_pod(_one(rendered, "Job", f"{_FULLNAME}-secrets-check")), "check")
    (script,) = check["args"]
    assert 'check "upstream" "api-key"' in script
    assert "flakegraph-serving-keys" not in script
    # The workers still speak to the gateway by alias, unchanged.
    assert _env(_worker(rendered, "extract"))["KG_LLM_MODEL"]["value"] == aliases["batch"]

    # Engines nobody routes to are refused; an empty model list is refused.
    assert "never forwards to" in _fails((*_EXTERNAL, *_SERVING))
    assert "at least one entry" in _fails(("gateway.litellm.upstream.type=external",))

    # Extra entries are appended verbatim, so an embedding model can share the door.
    extra = _render(
        (
            *_EXTERNAL,
            "gateway.litellm.extraModelList[0].model_name=embeddings",
            "gateway.litellm.extraModelList[0].litellm_params.model=openai/text-embedding-3-small",
        )
    )
    entries = yaml.safe_load(
        _one(extra, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]
    )["model_list"]
    assert {
        "model_name": "embeddings",
        "litellm_params": {"model": "openai/text-embedding-3-small"},
    } in entries


def test_every_example_profile_renders() -> None:
    """The three documented starting points install, each in its own shape."""

    external = _render((), values=(_EXTERNAL_EXAMPLE,))
    assert not [
        doc
        for doc in external
        if doc["kind"] in {"StatefulSet"} and doc["metadata"]["name"] == _VLLM
    ]
    assert not [doc for doc in external if doc["metadata"]["name"] == _ROUTER]
    assert not [doc for doc in external if doc["kind"] == "ScaledObject"]
    assert _one(external, "StatefulSet", f"{_FULLNAME}-mineru")

    minimal = _render((), values=(_MINIMAL_EXAMPLE,))
    kinds = {(doc["kind"], doc["metadata"]["name"]) for doc in minimal}
    assert ("Deployment", f"{_FULLNAME}-litellm") in kinds
    assert ("StatefulSet", f"{_FULLNAME}-mineru") not in kinds
    assert not [name for kind, name in kinds if kind in {"ScaledObject", "Cluster", "Middleware"}]
    for pool in _WORKER_POOLS:
        assert _one(minimal, "Deployment", f"{_FULLNAME}-{pool}")["spec"]["replicas"] == 1
    config = yaml.safe_load(_load_yaml(_MINIMAL_EXAMPLE)["config"]["content"])
    assert config["ocr"]["provider"] == "builtin_text"


def test_worker_pools_autoscale_from_dependency_aware_postgres_demand() -> None:
    """Release drained worker resources without scaling away active task leases."""

    values = _values()
    external_database = _render(_AUTOSCALING)
    bundled_database = _render((*_AUTOSCALING, "database.cloudNativePG.enabled=true"))
    demand_view = next(
        statement
        for statement in _SCHEMA_STATEMENTS
        if "VIEW flakegraph_worker_demand" in statement
    )

    # Off by default: it renders KEDA objects, which need KEDA's CRDs.
    assert values["autoscaling"]["enabled"] is False
    assert "keda.sh/v1alpha1/ScaledObject" in _fails(_AUTOSCALING, api_versions="")
    assert not [doc for doc in _render(()) if doc["kind"] == "ScaledObject"]
    for pool in _WORKER_POOLS:
        fixed = _one(_render(()), "Deployment", f"{_FULLNAME}-{pool}")
        assert fixed["spec"]["replicas"] == values["workers"][pool]["replicas"]
    assert values["autoscaling"]["pollingIntervalSeconds"] == 5
    assert values["autoscaling"]["cooldownPeriodSeconds"] == 30
    assert values["workers"]["prepare"]["autoscaling"]["minReplicas"] == 0
    assert values["distributed"]["leaseSeconds"] == 300
    # Without engines of its own the chart sizes extraction from the provider's
    # declared sequences: 64 x 1.5 / 8 slots.
    ceilings = {"prepare": 4, "extract": 12, "finalize": 1}
    slots = {"prepare": 2, "extract": 8, "finalize": 1}
    for pool in _WORKER_POOLS:
        assert values["workers"][pool]["slots"] == slots[pool]
        scaler = _one(external_database, "ScaledObject", f"{_FULLNAME}-{pool}")
        pool_values = values["workers"][pool]["autoscaling"]
        assert scaler["spec"]["scaleTargetRef"]["name"] == f"{_FULLNAME}-{pool}"
        assert scaler["spec"]["minReplicaCount"] == pool_values["minReplicas"]
        assert scaler["spec"]["maxReplicaCount"] == ceilings[pool]
        assert scaler["spec"]["advanced"]["restoreToOriginalReplicaCount"] is True
        for trigger in scaler["spec"]["triggers"]:
            assert trigger["type"] == "postgresql"
            assert trigger["metadata"]["connectionFromEnv"] == "KG_DISTRIBUTED_DATABASE_URL"
            # A pod runs `slots` tasks, so it is asked for once per that many.
            assert trigger["metadata"]["targetQueryValue"] == str(slots[pool])
            query = " ".join(trigger["metadata"]["query"].split())
            assert "FROM flakegraph_worker_demand" in query
            # Bounded per pool, so a million queued rows ask for maxReplicas, not
            # a million workers.
            assert "SUM(LEAST(" in query
            assert f"LEAST( desired_workers, {ceilings[pool] * slots[pool]} )" in query
            for stage in _stages(_worker(external_database, pool)):
                assert f"'{stage}'" in query
        worker = _worker(external_database, pool)
        assert _env(worker)["KG_DISTRIBUTED_WORKER_SLOTS"]["value"] == str(slots[pool])
        # KEDA runs outside the namespace, so the bundled database is addressed
        # by its fully qualified service and a password rather than the URI.
        bundled = _one(bundled_database, "ScaledObject", f"{_FULLNAME}-{pool}")
        for trigger in bundled["spec"]["triggers"]:
            assert trigger["metadata"]["passwordFromEnv"] == "KG_DISTRIBUTED_DATABASE_PASSWORD"
            assert "connectionFromEnv" not in trigger["metadata"]
        # Offline there is no existing Deployment to look up, so a fresh install
        # seeds each pool at its declared minimum rather than Kubernetes' one.
        deployment = _one(external_database, "Deployment", f"{_FULLNAME}-{pool}")
        assert deployment["spec"]["replicas"] == pool_values["minReplicas"]
    # The view the scaler reads exposes exactly the columns the query names.
    for column in ("stage", "desired_workers", "priority_band"):
        assert column in demand_view


def test_the_extraction_ceiling_oversubscribes_the_model_sequences_it_feeds() -> None:
    """Size the extract pool from engine sequences, so its slots keep every engine busy.

    Slots all together are the sequences times the oversubscription, rounded up
    to whole pods. An explicit ceiling wins; without engines the provider's
    declared sequences stand in; with neither the chart refuses to guess.
    """

    def ceiling(*settings: str) -> int:
        rendered = _render((*_AUTOSCALING, *settings))
        scaler = _one(rendered, "ScaledObject", f"{_FULLNAME}-extract")
        return int(scaler["spec"]["maxReplicaCount"])

    engines = (
        *_SERVING,
        "modelServing.replicas=6",
        "modelServing.server.maxNumSeqs=10",
    )
    # 6 engines x 10 sequences x 1.5 / 8 slots = 11.25, so twelve pods.
    assert ceiling(*engines) == 12
    assert ceiling(*engines, "workers.extract.autoscaling.sequenceOversubscription=1") == 8
    assert ceiling(*engines, "workers.extract.slots=4") == 23
    assert ceiling(*engines, "workers.extract.autoscaling.maxReplicas=40") == 40
    assert ceiling("workers.extract.autoscaling.providerSequences=100") == 19
    refused = _fails((*_AUTOSCALING, "workers.extract.autoscaling.providerSequences=null"))
    assert "workers.extract.autoscaling.maxReplicas is empty" in refused

    # The DGX profile takes the chart's ceiling for its four eight-sequence engines.
    example = _render(_AUTOSCALING, values=(_SPARK_EXAMPLE,))
    extract = _one(example, "ScaledObject", f"{_FULLNAME}-extract")
    assert extract["spec"]["maxReplicaCount"] == 6
    assert {t["metadata"]["targetQueryValue"] for t in extract["spec"]["triggers"]} == {"8"}


def test_each_pool_scales_on_total_demand_across_priority_bands() -> None:
    """Size shared capacity from all tasks; claim ordering selects urgent work."""

    rendered = _render(_AUTOSCALING)
    for pool in _WORKER_POOLS:
        triggers = _one(rendered, "ScaledObject", f"{_FULLNAME}-{pool}")["spec"]["triggers"]
        assert len(triggers) == 1
        (trigger,) = triggers
        assert trigger["name"] == f"{pool}-total"
        query = " ".join(trigger["metadata"]["query"].split())
        assert "SUM(LEAST(" in query
        assert "priority_band =" not in query
        assert "WHERE stage IN" in query
        assert trigger["metadata"]["targetQueryValue"] == str(_values()["workers"][pool]["slots"])


def test_database_schema_is_bootstrapped_before_a_helm_release_is_ready() -> None:
    """Make KEDA's worker-demand view deterministic on fresh installs and upgrades."""

    values = _values()
    rendered = _render(())
    job = _one(rendered, "Job", f"{_FULLNAME}-database-bootstrap")
    initialize = _bootstrap(rendered)
    (script,) = initialize["args"]

    assert values["database"]["bootstrap"]["activeDeadlineSeconds"] >= 600
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "post-install,post-upgrade"
    assert job["metadata"]["annotations"]["helm.sh/hook-delete-policy"] == (
        "before-hook-creation,hook-succeeded"
    )
    assert (
        job["spec"]["activeDeadlineSeconds"]
        == values["database"]["bootstrap"]["activeDeadlineSeconds"]
    )
    commands = [line.strip() for line in script.splitlines() if not line.strip().startswith("#")]
    assert commands[0].startswith("flakegraph preflight --deployment-worker")
    assert commands[1].startswith("exec flakegraph distributed init")
    # The hook declares what every enabled pool serves under this release: a
    # pool scaled to zero has no worker to do it, and the demand signal counts
    # only work the declared fleet can take, so an idle fleet upgraded without
    # this would never scale up for a run planned under the new digest.
    words = commands[1].split()
    served = {words[i + 1] for i, word in enumerate(words) if word == "--serve-stage"}
    expected = {
        stage for pool in values["workers"].values() if pool["enabled"] for stage in pool["stages"]
    }
    assert served == expected
    assert "finalize_graph" in served
    assert _env(initialize)["KG_DISTRIBUTED_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": values["database"]["secretName"],
        "key": values["database"]["secretKey"],
    }


def test_operator_notes_describe_the_selected_database_bootstrap_mode() -> None:
    """Helm output must not claim a disabled bootstrap hook completed."""

    with_hook = _notes(())
    without_hook = _notes(("database.bootstrap.enabled=false",))

    assert "The database bootstrap hook completed" in with_hook
    assert "Database bootstrap was disabled" not in with_hook
    assert "Database bootstrap was disabled" in without_hook
    assert "The database bootstrap hook completed" not in without_hook


def test_operator_notes_say_what_scales_and_which_secrets_are_needed() -> None:
    """A first install must not have to discover missing Secrets one pod at a time."""

    values = _values()
    notes = _notes(())

    assert "Queue-driven autoscaling is OFF" in notes
    assert "Queue-driven autoscaling is on" not in notes
    for name in (
        values["database"]["secretName"],
        values["providerSecret"]["name"],
        values["gateway"]["litellm"]["virtualKeySecret"]["name"],
        values["gateway"]["litellm"]["upstreamKeySecret"]["name"],
        values["documentParsing"]["shim"]["keySecret"]["name"],
    ):
        assert f"secret/{name}" in notes, name
    assert values["gateway"]["litellm"]["virtualKeySecret"]["masterKey"] in notes
    # A client-side dry run cannot declare KEDA's API, so the "on" branch is
    # read from the template rather than rendered.
    template = Path("deploy/helm/flakegraph/templates/NOTES.txt").read_text(encoding="utf-8")
    assert "{{- if .Values.autoscaling.enabled }}\n\nQueue-driven autoscaling is on" in template


def test_a_pre_install_hook_checks_every_secret_the_release_mounts() -> None:
    """Refuse the install while a referenced Secret is missing, naming it.

    The check and the console's read grant come from one helper, so the Job
    reads exactly the names the Role allows and the Role names exactly what
    the release mounts.
    """

    values = _values()
    rendered = _render(
        (*_GATED, "ingress.tls.secretName=wildcard-tls", "ingress.authProxy.existingSecret=oidc")
    )
    job = _one(rendered, "Job", f"{_FULLNAME}-secrets-check")
    role = _one(rendered, "Role", f"{_FULLNAME}-secrets-check")
    check = _container(_pod(job), "check")
    (script,) = check["args"]

    assert values["secretsCheck"]["enabled"] is True
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "pre-install,pre-upgrade"
    assert job["metadata"]["annotations"]["helm.sh/hook-delete-policy"] == (
        "before-hook-creation,hook-succeeded"
    )
    assert check["image"] == _worker(rendered, "extract")["image"]
    checked = set(re.findall(r'^check "([^"]+)" "([^"]*)"', script, re.MULTILINE))
    assert (values["database"]["secretName"], values["database"]["secretKey"]) in checked
    assert (values["providerSecret"]["name"], "") in checked
    assert ("oidc", "client-secret") in checked
    assert ("oidc", "cookie-secret") in checked
    assert ("wildcard-tls", "tls.crt") in checked
    (secret_rule,) = role["rules"]
    assert secret_rule["verbs"] == ["get"]
    assert set(secret_rule["resourceNames"]) >= {name for name, _ in checked}
    console_rule = next(
        rule
        for rule in _one(rendered, "Role", f"{_FULLNAME}-app")["rules"]
        if rule["resources"] == ["secrets"]
    )
    assert set(console_rule["resourceNames"]) == set(secret_rule["resourceNames"])
    assert not [
        doc
        for doc in _render(("secretsCheck.enabled=false",))
        if doc["metadata"]["name"].endswith("-secrets-check")
    ]


def test_bundled_database_has_capacity_for_documented_fleet_workers() -> None:
    """Reserve database sessions for workers, autoscaling, and operations."""

    cloud_native_pg = _values()["database"]["cloudNativePG"]
    cluster = _one(
        _render(("database.cloudNativePG.enabled=true",)),
        "Cluster",
        cloud_native_pg["clusterName"],
    )

    assert cloud_native_pg["maxConnections"] >= 300
    assert cluster["spec"]["postgresql"]["parameters"]["max_connections"] == str(
        cloud_native_pg["maxConnections"]
    )


def test_scheduling_priorities_preserve_models_and_release_workers_for_spark() -> None:
    """Make the extraction-to-finalization handoff resilient to autoscaler delay."""

    values = _values()
    priorities = values["scheduling"]["priorityClasses"]
    rendered = _render(_SPARK)
    prefix = f"fleet-{_FULLNAME}"

    # The serving plane outranks the workers deliberately: it is the path every
    # consumer takes, so leaving it at the default priority lets a queue scale-up
    # preempt the gateway those same workers are trying to reach.
    assert (
        priorities["workerValue"]
        < priorities["sparkValue"]
        < priorities["servingValue"]
        < priorities["modelValue"]
    )
    # Helm prints a value of a million as 1e+06, which the API server reads
    # back as the integer; compare numerically rather than by spelling.
    classes = {
        doc["metadata"]["name"]: float(doc["value"])
        for doc in rendered
        if doc["kind"] == "PriorityClass"
    }
    assert classes == {
        f"{prefix}-worker": priorities["workerValue"],
        f"{prefix}-spark": priorities["sparkValue"],
        f"{prefix}-serving": priorities["servingValue"],
        f"{prefix}-model": priorities["modelValue"],
    }
    placed = {
        name: _pod(_one(rendered, kind, name))["priorityClassName"]
        for kind, name in (
            ("Deployment", f"{_FULLNAME}-litellm"),
            ("Deployment", _ROUTER),
            ("Deployment", f"{_FULLNAME}-ocr"),
            ("StatefulSet", f"{_FULLNAME}-mineru"),
            ("Deployment", f"{_FULLNAME}-prepare"),
            ("Deployment", f"{_FULLNAME}-extract"),
            ("Deployment", f"{_FULLNAME}-finalize"),
            ("StatefulSet", _VLLM),
        )
    }
    assert placed == {
        f"{_FULLNAME}-litellm": f"{prefix}-serving",
        _ROUTER: f"{prefix}-serving",
        f"{_FULLNAME}-ocr": f"{prefix}-serving",
        f"{_FULLNAME}-mineru": f"{prefix}-serving",
        f"{_FULLNAME}-prepare": f"{prefix}-worker",
        f"{_FULLNAME}-extract": f"{prefix}-worker",
        f"{_FULLNAME}-finalize": f"{prefix}-spark",
        _VLLM: f"{prefix}-model",
    }
    assert _executor_template(rendered)["spec"]["priorityClassName"] == f"{prefix}-spark"
    assert values["terminationGracePeriodSeconds"] > 3600


def test_spark_role_allows_native_executor_creation_and_cleanup() -> None:
    """Permit Spark to remove executor collections when a driver terminates."""

    (rule,) = _one(_render(_SPARK), "Role", f"{_FULLNAME}-spark")["rules"]

    assert {"pods", "services", "configmaps", "persistentvolumeclaims"} <= set(rule["resources"])
    assert {"create", "delete", "deletecollection"} <= set(rule["verbs"])


def test_object_storage_cannot_silently_select_an_unavailable_spark_runtime() -> None:
    """Reject a fleet configuration that would fail only after extraction ends."""

    refusal = _fails(("artifactStorage.uri=s3://artifacts",))

    assert "spark.enabled must be true when artifactStorage.uri is configured" in refusal
    _render(("artifactStorage.uri=s3://artifacts", "spark.enabled=true"))


def test_a_draft_model_path_must_have_somewhere_to_come_from() -> None:
    """Naming a path is not obtaining the file at it.

    A replica scheduled onto a node where nobody placed the draft model starts,
    fails to load, and crash-loops with an error about an invalid repository id -
    which reads like a typo rather than a missing artifact. The chart refuses to
    render that, and can seed the volume itself.
    """

    values = _values()["modelServing"]["server"]
    seed = values["draftModelSeed"]
    drafted = (
        *_SERVING,
        "modelServing.server.speculativeTokens=8",
        "modelServing.server.speculativeMethod=dflash",
        "modelServing.server.speculativeDraftModel=/models/dflash2",
    )

    # The default drafts nothing and so names no path; the seed block is idle.
    assert values["speculativeDraftModel"] == ""
    assert seed["image"] == ""
    assert seed["providedExternally"] is False

    # Refuses to render a path nothing supplies...
    refusal = _fails(drafted)
    assert "modelServing.server.speculativeDraftModel is the path" in refusal
    assert "draftModelSeed.image" in refusal
    assert "draftModelSeed.providedExternally=true" in refusal
    _render((*drafted, "modelServing.server.draftModelSeed.providedExternally=true"))

    # ...and seeds the volume when an image carries the model.
    seeded = _render(
        (
            *drafted,
            "modelServing.server.draftModelSeed.image=example/dflash2:1",
            "modelServing.server.draftModelSeed.sourcePath=/dflash2",
        )
    )
    pod = _pod(_one(seeded, "StatefulSet", _VLLM))
    seeder = _container(pod, "seed-draft-model", "initContainers")
    (script,) = seeder["args"]
    commands = [line.strip() for line in script.splitlines() if not line.strip().startswith("#")]
    assert seeder["image"] == "example/dflash2:1"
    assert seeder["volumeMounts"] == [{"name": "models", "mountPath": "/models"}]
    assert 'target="/models/dflash2"' in commands
    # The copy lands beside the target and is renamed last, so an interrupted
    # copy is never mistaken for a complete one.
    assert 'cp -r "/dflash2" "$target.partial"' in commands
    assert commands[-1] == 'mv "$target.partial" "$target"'
    assert "initContainers" not in _pod(_one(_render(_SERVING), "StatefulSet", _VLLM))


def test_the_draft_model_seed_is_constrained_by_the_schema() -> None:
    """An operator mistyping this should hear about it at install, not at load."""

    schema = _schema()
    seed = schema["properties"]["modelServing"]["properties"]["server"]["properties"][
        "draftModelSeed"
    ]

    assert seed["additionalProperties"] is False
    assert set(seed["properties"]) == {
        "image",
        "pullPolicy",
        "sourcePath",
        "providedExternally",
    }


def test_cache_events_are_published_on_a_topic_the_picker_subscribes_to() -> None:
    """Publishing on the default empty topic reaches nobody.

    The picker subscribes with the ZMQ prefix filter "kv@" and parses the model
    name out of ``kv@<pod>@<model>``. vLLM's own default topic is the empty
    string, which that filter discards - so a deployment can look configured,
    publish continuously, and have every event dropped before it is read.
    """

    values = _values()["modelServing"]
    engine = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    events = json.loads(_args(engine)["--kv-events-config"])

    assert events["enable_kv_cache_events"] is True
    assert events["topic"] == f"kv@$(POD_IP):{values['kvEvents']['port']}@{values['model']['name']}"
    # The topic interpolates POD_IP, which Kubernetes expands into args only when
    # the same container declares it.
    assert _env(engine)["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"


def test_exact_prefix_routing_names_its_producer_and_matches_block_size() -> None:
    """Both halves are silent when wrong, so both are pinned here.

    A prefix-cache-scorer with no producer named gets the approximate producer
    auto-created underneath it, and reports confident scores about a cache it has
    never seen. A block size that disagrees with the engine's hashes differently
    sized blocks, which agree about nothing.
    """

    values = _values()["modelServing"]
    router = _one(_render(_SERVING), "ConfigMap", _ROUTER)["data"]
    config = yaml.safe_load(router["default-plugins.yaml"])
    plugins = {plugin["type"]: plugin for plugin in config["plugins"]}

    assert plugins["prefix-cache-scorer"]["parameters"] == {
        "prefixMatchInfoProducerName": "precise-prefix-cache-producer"
    }
    assert plugins["token-producer"]["parameters"]["modelName"] == values["model"]["name"]
    assert "endpoint-notification-source" in plugins
    # Without this wiring the producer never opens a subscriber to any pod.
    assert {
        "pluginRef": "endpoint-notification-source",
        "extractors": [{"pluginRef": "precise-prefix-cache-producer"}],
    } in config["dataLayer"]["sources"]
    producer = plugins["precise-prefix-cache-producer"]["parameters"]
    assert (
        producer["tokenProcessorConfig"]["blockSizeTokens"] == values["kvEvents"]["blockSizeTokens"]
    )
    # vLLM's default block size is 16; the picker must hash the same width.
    assert values["kvEvents"]["blockSizeTokens"] == 16


def _first_declared(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a container's environment by the first declaration of each name.

    The shared consumer block is emitted before any provider Secret mapping,
    so the first declaration is the one that block owns.
    """

    declared: dict[str, dict[str, Any]] = {}
    for entry in container["env"]:
        declared.setdefault(entry["name"], entry)
    return declared


def _stages(worker: dict[str, Any]) -> list[str]:
    """Return the stages a worker container is started for."""

    args: list[str] = worker["args"]
    return [args[index + 1] for index, arg in enumerate(args) if arg == "--stage"]


def _worker(rendered: list[dict[str, Any]], pool: str) -> dict[str, Any]:
    """Return one worker pool's worker container."""

    return _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-{pool}")), "worker")


def _bootstrap(rendered: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the database bootstrap Job's only container."""

    return _container(_pod(_one(rendered, "Job", f"{_FULLNAME}-database-bootstrap")), "initialize")


def _executor_template(rendered: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the Pod template the Spark driver hands to its executors."""

    configmap = _one(rendered, "ConfigMap", f"{_FULLNAME}-spark-executor-template")
    template: dict[str, Any] = yaml.safe_load(configmap["data"]["executor-pod-template.yaml"])
    return template


def test_the_front_door_compresses_what_it_sends_the_browser() -> None:
    """The console's own gzip skips its API routes, so the edge compresses.

    Next.js hands its compression filter a Content-Type list for route
    handler responses, which the filter rejects, and a graph payload is tens
    of megabytes of JSON. The chain keeps the identity strip first, then the
    gate, and compression nearest the service; the Middleware is opt-in and
    Traefik-only, and the default Ingress carries nothing at all.
    """

    gated = _render((*_GATED, "ingress.compression.enabled=true"))
    middleware = _one(gated, "Middleware", f"{_FULLNAME}-compress")
    assert middleware["spec"] == {"compress": {"minResponseBodyBytes": 1024}}
    chain = _one(gated, "Ingress", _FULLNAME)["metadata"]["annotations"][
        "traefik.ingress.kubernetes.io/router.middlewares"
    ].split(",")
    assert chain == [
        f"{NAMESPACE}-{_FULLNAME}-strip-identity@kubernetescrd",
        f"{NAMESPACE}-{_FULLNAME}-auth-errors@kubernetescrd",
        f"{NAMESPACE}-{_FULLNAME}-auth-auth@kubernetescrd",
        f"{NAMESPACE}-{_FULLNAME}-compress@kubernetescrd",
    ]

    ungated = _render(
        (
            "ingress.enabled=true",
            "ingress.domain=example.test",
            "ingress.controller=traefik",
            "ingress.compression.enabled=true",
        )
    )
    assert (
        _one(ungated, "Ingress", _FULLNAME)["metadata"]["annotations"][
            "traefik.ingress.kubernetes.io/router.middlewares"
        ]
        == f"{NAMESPACE}-{_FULLNAME}-compress@kubernetescrd"
    )

    assert _values()["ingress"]["compression"]["enabled"] is False
    plain = _render(("ingress.enabled=true", "ingress.domain=example.test"))
    assert not [doc for doc in plain if doc["kind"] == "Middleware"]
    assert "annotations" not in _one(plain, "Ingress", _FULLNAME)["metadata"]
    # A Middleware nothing can apply is an install error, so it is refused
    # rather than rendered where the controller cannot take it.
    assert "cannot express" in _fails(
        ("ingress.enabled=true", "ingress.domain=example.test", "ingress.compression.enabled=true")
    )
    assert "traefik.io/v1alpha1/Middleware" in _fails(
        (*_GATED, "ingress.compression.enabled=true"), api_versions=""
    )


def test_the_gate_is_expressed_per_controller_and_refused_where_it_cannot_be() -> None:
    """A plain Ingress has no authentication hook, so it must not pretend to.

    On Traefik the gate is a forwardAuth Middleware; on ingress-nginx it is the
    auth-url annotations, with the identity headers replaced from the gate's
    answer on every forwarded request. On anything else an Ingress that
    rendered without its gate would publish the console open, so the render
    is refused instead.
    """

    assert _values()["ingress"]["controller"] == "none"
    refusal = _fails(
        (
            "ingress.enabled=true",
            "ingress.domain=example.test",
            "ingress.authProxy.enabled=true",
            "controlPlane.networkPolicy.enabled=true",
        )
    )
    assert "traefik or nginx" in refusal

    nginx = _render(
        (
            "ingress.enabled=true",
            "ingress.domain=example.test",
            "ingress.controller=nginx",
            "ingress.authProxy.enabled=true",
            "controlPlane.networkPolicy.enabled=true",
        )
    )
    annotations = _one(nginx, "Ingress", _FULLNAME)["metadata"]["annotations"]
    assert annotations["nginx.ingress.kubernetes.io/auth-url"] == (
        f"http://{_FULLNAME}-auth.{NAMESPACE}.svc.cluster.local:4180/oauth2/auth"
    )
    assert annotations["nginx.ingress.kubernetes.io/auth-signin"].startswith(
        "https://auth.example.test/oauth2/start?rd="
    )
    assert annotations["nginx.ingress.kubernetes.io/auth-response-headers"].split(",") == [
        "X-Auth-Request-User",
        "X-Auth-Request-Email",
        "X-Auth-Request-Preferred-Username",
        "X-Auth-Request-Groups",
    ]
    assert "traefik.ingress.kubernetes.io/router.middlewares" not in annotations
    assert not [doc for doc in nginx if doc["kind"] in {"Middleware", "IngressRoute"}]
    # The gate itself is rendered the same way on both.
    assert _one(nginx, "Deployment", f"{_FULLNAME}-auth")


def test_client_supplied_identity_headers_are_removed_before_the_gate_runs() -> None:
    """A signed-in viewer must not become somebody else by adding a header.

    forwardAuth copies only the headers it names from the gate's answer and
    never touches the X-Forwarded-* family, so a client-supplied one would
    reach the console intact. The strip Middleware stands first in the gated
    chain and blanks every identity-shaped header; the gate then sets the
    real ones, groups included, which is what the console maps to roles.
    """

    rendered = _render(_GATED)
    strip = _one(rendered, "Middleware", f"{_FULLNAME}-strip-identity")["spec"]["headers"][
        "customRequestHeaders"
    ]
    assert set(strip) >= {
        "X-Auth-Request-User",
        "X-Auth-Request-Email",
        "X-Auth-Request-Preferred-Username",
        "X-Auth-Request-Groups",
        "X-Auth-Request-Access-Token",
        "X-Forwarded-User",
        "X-Forwarded-Email",
        "X-Forwarded-Preferred-Username",
        "X-Forwarded-Groups",
        "X-Forwarded-Access-Token",
    }
    assert all(value == "" for value in strip.values())
    chain = _one(rendered, "Ingress", _FULLNAME)["metadata"]["annotations"][
        "traefik.ingress.kubernetes.io/router.middlewares"
    ].split(",")
    assert chain[0] == f"{NAMESPACE}-{_FULLNAME}-strip-identity@kubernetescrd"
    assert chain.index(f"{NAMESPACE}-{_FULLNAME}-auth-auth@kubernetescrd") > 0
    forward_auth = _one(rendered, "Middleware", f"{_FULLNAME}-auth-auth")["spec"]["forwardAuth"]
    assert forward_auth["authResponseHeaders"] == [
        "X-Auth-Request-User",
        "X-Auth-Request-Email",
        "X-Auth-Request-Preferred-Username",
        "X-Auth-Request-Groups",
        "Set-Cookie",
    ]


def test_only_the_gates_401_becomes_a_sign_in_redirect() -> None:
    """The console's own refusals reach the page that asked.

    The errors Middleware wraps the service as well as the gate, so any
    status it lists is replaced by the sign-in page. The console answers 403
    when a signed-in person opens a graph that is not shared with them; that
    answer is what the page shows, so only the gate's 401 is rewritten.
    """

    errors = _one(_render(_GATED), "Middleware", f"{_FULLNAME}-auth-errors")["spec"]["errors"]
    assert errors["status"] == ["401"]
    assert errors["statusRewrites"] == {"401": 302}


def test_a_machine_with_a_console_key_reaches_the_api_past_the_gate_as_nobody() -> None:
    """A key holder skips the sign-in, loses every identity header, and hits the console.

    The route matches only requests presenting one of the console's keys, on
    the console's host and API paths, ahead of the gated catch-all; the
    middleware chain strips what the gate would have vouched for, then
    compresses like the gated path does.
    """

    rendered = _render(
        (*_GATED, "ingress.tls.secretName=wildcard-tls", "ingress.compression.enabled=true")
    )
    route = _one(rendered, "IngressRoute", f"{_FULLNAME}-machine-api")
    (rule,) = route["spec"]["routes"]
    assert "Host(`flakegraph.example.test`)" in rule["match"]
    assert "PathPrefix(`/api/`)" in rule["match"]
    assert "HeaderRegexp(`Authorization`, `^Bearer fg_`)" in rule["match"]
    assert "HeaderRegexp(`X-Flakegraph-Api-Key`, `^fg_`)" in rule["match"]
    assert rule["priority"] > 0
    # The same strip the gated chain leads with, so a key holder is nobody.
    assert [item["name"] for item in rule["middlewares"]] == [
        f"{_FULLNAME}-strip-identity",
        f"{_FULLNAME}-compress",
    ]
    assert rule["services"] == [{"name": f"{_FULLNAME}-app", "port": 3000}]
    assert route["spec"]["tls"] == {"secretName": "wildcard-tls"}

    without = _render((*_GATED, "ingress.authProxy.machineKeys.enabled=false"))
    assert not [doc for doc in without if doc["kind"] == "IngressRoute"]
    assert not [doc for doc in _render(()) if doc["kind"] == "IngressRoute"]


def test_the_gate_bounds_its_csrf_cookies() -> None:
    """A sign-in loop must not grow the request header until the edge answers 431."""

    rendered = _render(_GATED)
    gate = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-auth")), "auth-proxy")
    flags = dict(arg.split("=", 1) for arg in gate["args"] if arg.startswith("--") and "=" in arg)
    assert flags["--cookie-csrf-per-request"] == "true"
    assert int(flags["--cookie-csrf-per-request-limit"]) <= 16


def test_the_balancer_can_reach_the_engines_it_is_meant_to_reserve() -> None:
    """A balancer the network policy blocks is a feature that silently does nothing.

    With only the router admitted to the sidecar's port, it gets a connection
    error from every engine. The rule is part of the feature, so it is pinned
    here — and only the sidecar's port, never the engine's.
    """

    rendered = _render((*_SERVING, "modelServing.balancer.enabled=true"))
    policy = _one(rendered, "NetworkPolicy", _VLLM)
    balancer = {"app.kubernetes.io/component": "serving-balancer"}
    admitted = {
        port["port"]
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for port in rule["ports"]
        if balancer.items() <= (peer.get("podSelector", {}).get("matchLabels", {})).items()
    }
    assert admitted == {"http"}

    container = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-balancer")), "balancer")
    assert container["command"] == ["flakegraph", "serving", "balancer"]
    environment = _env(container)
    # It authenticates as an interactive consumer: a batch key must not be
    # able to lift the cap that is holding batch back.
    key = environment["FLAKEGRAPH_BALANCER_API_KEY"]["valueFrom"]["secretKeyRef"]
    assert key["key"] == "SIDECAR_KEY_INTERACTIVE"
    assert environment["FLAKEGRAPH_BALANCER_ENGINE_PORT"]["value"] == "8000"

    # And the whole thing is absent unless asked for.
    names = {str(item.get("metadata", {}).get("name", "")) for item in _render(_SERVING)}
    assert not [name for name in names if "balancer" in name]


def test_the_engine_charges_unified_memory_to_the_scheduler() -> None:
    """A device reservation the kubelet cannot see is a node the scheduler overfills.

    On the reference fleet an engine holds about 69 GiB of the node's own RAM as
    device memory. Requesting less than that, it leaves the scheduler reading a
    full machine as nearly empty and packing workers and Spark executors onto
    it. The kernel resolves the overcommit by global OOM, which can take a kueue
    webhook with it, stall the API server, flip nodes NotReady and kill an
    in-flight finalization. On a discrete GPU none of that applies, so the
    default accounting must stay untouched.
    """

    discrete = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    assert discrete["resources"] == _values()["modelServing"]["resources"]

    pool = 100
    unified = _render((*_SERVING, f"modelServing.server.unifiedMemoryPerDeviceGi={pool}"))
    resources = _container(_pod(_one(unified, "StatefulSet", _VLLM)), "vllm")["resources"]
    declared = _values()["modelServing"]["resources"]
    charged = round(pool * _values()["modelServing"]["server"]["gpuMemoryUtilization"])

    for section in ("requests", "limits"):
        host = int(declared[section]["memory"].removesuffix("Gi"))
        assert resources[section]["memory"] == f"{host + charged}Gi", section
        # Everything else in the block is the operator's, untouched.
        assert {k: v for k, v in resources[section].items() if k != "memory"} == {
            k: v for k, v in declared[section].items() if k != "memory"
        }

    # A limit below the pod's own request is rejected by the API server, so the
    # same reservation is charged to both.
    request = int(resources["requests"]["memory"].removesuffix("Gi"))
    assert int(resources["limits"]["memory"].removesuffix("Gi")) >= request

    # Each device draws from the pool, so the charge follows the split.
    across_two = _render(
        (
            *_SERVING,
            f"modelServing.server.unifiedMemoryPerDeviceGi={pool}",
            "modelServing.server.tensorParallelSize=2",
        )
    )
    pair = _container(_pod(_one(across_two, "StatefulSet", _VLLM)), "vllm")["resources"]
    host = int(declared["requests"]["memory"].removesuffix("Gi"))
    assert pair["requests"]["memory"] == f"{host + 2 * charged}Gi"


def test_a_memory_quantity_the_chart_cannot_charge_is_refused() -> None:
    """Silently mis-charging a reservation is worse than refusing to render."""

    message = _fails(
        (
            *_SERVING,
            "modelServing.server.unifiedMemoryPerDeviceGi=100",
            "modelServing.resources.requests.memory=lots",
        )
    )
    assert "not a memory quantity" in message


def test_the_balancer_can_be_moved_without_reloading_every_checkpoint() -> None:
    """The balancer shares the sidecar's code, not its restart cost.

    Moving the sidecar pin reloads a 27B checkpoint on every engine in turn -
    the better part of two hours on the reference fleet - while the balancer
    is one stateless Deployment that restarts in seconds, and the caps it has
    set expire on their own meanwhile. So a balancer-only fix must not be a
    fleet-wide engine roll. Unset, it follows the sidecar: that is right when
    both are moving together.
    """

    settings = (*_SERVING, "modelServing.balancer.enabled=true")
    together = _render((*settings, "modelServing.sidecar.imageTag=1.2.3"))
    sidecar = _container(_pod(_one(together, "StatefulSet", _VLLM)), "sidecar")["image"]
    balancer = _container(_pod(_one(together, "Deployment", f"{_FULLNAME}-balancer")), "balancer")[
        "image"
    ]
    assert sidecar.endswith(":1.2.3")
    assert balancer == sidecar

    apart = _render(
        (
            *settings,
            "modelServing.sidecar.imageTag=1.2.3",
            "modelServing.balancer.imageTag=1.2.4",
        )
    )
    assert _container(_pod(_one(apart, "StatefulSet", _VLLM)), "sidecar")["image"] == sidecar
    moved = _container(_pod(_one(apart, "Deployment", f"{_FULLNAME}-balancer")), "balancer")
    assert moved["image"].endswith(":1.2.4")

    # And the renewal window the standby depends on reaches the process.
    assert _env(moved)["FLAKEGRAPH_BALANCER_RENEW_BEFORE_SECONDS"]["value"] == "120"


def test_the_spark_driver_pool_is_replaced_rather_than_surged() -> None:
    """Two finalization drivers at once is two fleets' worth of executors.

    The finalize pod drives a Spark application whose executors span every
    node, and it carries a long grace period so an in-flight finalization can
    finish. A rolling update counts a terminating pod as already gone, so it
    starts the replacement at once - which claims the next run while the old
    driver finishes its own, under maxSurge 1 and maxSurge 0 alike, and the two
    end in global kernel OOMs. Only Recreate waits for the old pod to be gone.
    The pools that do not drive Spark are ordinary stateless workers and keep
    surging, which is what makes their upgrades seamless.
    """

    rendered = _render((*_SPARK,))
    finalize = _one(rendered, "Deployment", f"{_FULLNAME}-finalize")["spec"]["strategy"]
    assert finalize == {"type": "Recreate"}
    for pool in ("prepare", "extract"):
        strategy = _one(rendered, "Deployment", f"{_FULLNAME}-{pool}")["spec"]["strategy"]
        assert strategy["type"] == "RollingUpdate", pool
        assert strategy["rollingUpdate"]["maxSurge"] == 1, pool

    # Without Spark the finalize pool is an ordinary worker and surges like one.
    ordinary = _one(_render(_SERVING), "Deployment", f"{_FULLNAME}-finalize")
    assert ordinary["spec"]["strategy"]["rollingUpdate"]["maxSurge"] == 1


_EMBEDDING = (*_SERVING, "embeddingServing.enabled=true")
_EMBEDDING_NAME = f"{_FULLNAME}-embedding"


def test_embedding_serving_is_off_unless_asked_for() -> None:
    """Off, every process embeds locally."""

    rendered = _render(_SERVING)
    assert not [doc for doc in rendered if doc["metadata"]["name"] == _EMBEDDING_NAME]
    for pool in ("prepare", "extract", "finalize"):
        worker = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-{pool}")), "worker")
        assert not [name for name in _env(worker) if name.startswith("KG_EMBEDDING")], pool


def test_embedding_serving_moves_every_consumer_onto_the_gpu_through_the_gateway() -> None:
    """The finalizer's executors cannot embed a real corpus on one CPU thread.

    Measured on the reference fleet: a 500-document run's chunks are hours of
    work per executor on the local path, and a batch of its longest chunks took
    the Python worker to 11.8 GiB beside a JVM in a 14 GiB executor. The same
    chunks took 50 seconds through a pooling server, with vectors matching the
    local path to a cosine of 0.9998. Every consumer must use it - a worker
    left embedding locally would still produce the slow path, and the digest.
    """

    rendered = _render(_EMBEDDING)
    for pool in ("prepare", "extract", "finalize"):
        worker = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-{pool}")), "worker")
        environment = _env(worker)
        assert environment["KG_EMBEDDING_PROVIDER"]["value"] == "openai_compatible", pool
        assert environment["KG_EMBEDDING_ENDPOINT"]["value"].endswith(":4000/v1"), pool
        # The checkpoint's own name, so a graph records what embedded it.
        assert environment["KG_EMBEDDING_MODEL"]["value"] == "Qwen/Qwen3-Embedding-0.6B", pool
        key = environment["KG_EMBEDDING_API_KEY"]["valueFrom"]["secretKeyRef"]
        assert key == environment["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"], pool

    config = next(
        document["data"]
        for document in rendered
        if document["kind"] == "ConfigMap" and "mode: embedding" in str(document.get("data"))
    )
    model_list = yaml.safe_load(next(value for value in config.values() if "model_list" in value))
    model = "Qwen/Qwen3-Embedding-0.6B"
    entry = next(item for item in model_list["model_list"] if item["model_name"] == model)
    assert entry["litellm_params"]["api_base"] == f"http://{_EMBEDDING_NAME}:8000/v1"
    assert entry["model_info"]["mode"] == "embedding"
    # Without it the proxy asks for base64, the pooling server answers 400, and
    # every finalization fails at its first embedding.
    assert entry["litellm_params"]["encoding_format"] == "float"

    pod = _pod(_one(rendered, "Deployment", _EMBEDDING_NAME))
    server = _container(pod, "vllm")
    assert server["args"][:3] == ["Qwen/Qwen3-Embedding-0.6B", "--runner", "pooling"]
    assert _env(server)["HF_HUB_OFFLINE"]["value"] == "1"
    # Seeded from the application image, which carries the very weights the
    # local path loads, so the vectors stay interchangeable.
    seed = pod["initContainers"][0]
    assert (
        seed["image"]
        == _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-extract")), "worker")["image"]
    )
    assert "models--Qwen--Qwen3-Embedding-0.6B" in seed["command"][2]
    # A whole device by default; sharing is the site's explicit choice.
    assert server["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_the_embedding_server_is_reachable_only_through_the_gateway() -> None:
    """It checks no key, so the network policy is the whole of its access control."""

    policy = _one(_render(_EMBEDDING), "NetworkPolicy", _EMBEDDING_NAME)
    peers = [peer for rule in policy["spec"]["ingress"] for peer in rule["from"]]
    components = {
        peer.get("podSelector", {}).get("matchLabels", {}).get("app.kubernetes.io/component")
        for peer in peers
    } - {None}
    assert components == {"gateway"}


def test_the_gateway_sends_each_embedding_call_once_and_gives_up_before_its_caller() -> None:
    """A retried or abandoned call behind a busy server only lengthens its queue."""

    config = next(
        document["data"]
        for document in _render(_EMBEDDING)
        if document["kind"] == "ConfigMap" and "mode: embedding" in str(document.get("data"))
    )
    gateway = yaml.safe_load(next(value for value in config.values() if "model_list" in value))
    model = "Qwen/Qwen3-Embedding-0.6B"
    entry = next(item for item in gateway["model_list"] if item["model_name"] == model)
    assert entry["litellm_params"]["timeout"] < EMBEDDING_TIMEOUT_SECONDS
    policy = gateway["router_settings"]["model_group_retry_policy"]
    assert policy == {model: {"TimeoutErrorRetries": 0, "InternalServerErrorRetries": 0}}


def test_embedding_replicas_are_placed_where_the_site_says(tmp_path: Path) -> None:
    """The scheduler cannot see unified memory an engine already holds."""

    affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {"matchExpressions": [{"key": "k", "operator": "In", "values": ["a", "b"]}]}
                ]
            }
        }
    }
    spread = [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "DoNotSchedule",
        }
    ]
    values = tmp_path / "placement.yaml"
    values.write_text(
        yaml.safe_dump(
            {"embeddingServing": {"affinity": affinity, "topologySpreadConstraints": spread}}
        )
    )

    pod = _pod(_one(_render(_EMBEDDING, values=(values,)), "Deployment", _EMBEDDING_NAME))

    assert pod["affinity"] == affinity
    assert pod["topologySpreadConstraints"] == spread
    assert "affinity" not in _pod(_one(_render(_EMBEDDING), "Deployment", _EMBEDDING_NAME))


def test_a_shared_embedding_device_claims_nothing_and_names_the_runtime() -> None:
    rendered = _render(
        (
            *_EMBEDDING,
            "embeddingServing.device.claimGpu=false",
            "embeddingServing.device.runtimeClassName=nvidia",
        )
    )
    pod = _pod(_one(rendered, "Deployment", _EMBEDDING_NAME))
    server = _container(pod, "vllm")
    assert pod["runtimeClassName"] == "nvidia"
    assert "nvidia.com/gpu" not in str(server["resources"])
    assert _env(server)["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"


def test_turning_embedding_serving_on_does_not_restart_the_engines() -> None:
    """A new component must not ride the engines' pod-template hash.

    Every key under modelServing feeds the engines' checksum annotation, so a
    setting placed there rolls six engines for the better part of two hours.
    This one lives beside modelServing rather than inside it for that reason.
    """

    # The whole template, annotations included: the checksum that rolls the
    # engines lives in the template's metadata, not in the pod spec.
    before = _one(_render(_SERVING), "StatefulSet", _VLLM)["spec"]["template"]
    after = _one(_render(_EMBEDDING), "StatefulSet", _VLLM)["spec"]["template"]
    assert "checksum/model-serving" in before["metadata"]["annotations"]
    assert before == after


def test_embedding_serving_needs_the_gateway() -> None:
    message = _fails((*_EMBEDDING, "gateway.enabled=false"))
    assert "enable gateway" in message


_LANES = (*_SERVING, "modelServing.balancer.enabled=true")


def _picker_config(rendered: list[dict[str, Any]]) -> dict[str, Any]:
    router = next(
        document
        for document in rendered
        if document["kind"] == "ConfigMap" and "default-plugins.yaml" in document.get("data", {})
    )
    config: dict[str, Any] = yaml.safe_load(router["data"]["default-plugins.yaml"])
    return config


def test_lanes_route_people_and_batch_to_the_engines_labelled_for_them() -> None:
    """The reservation only helps if routing sends people to the reserved machine.

    Without lanes, chat requests land on engines other than the standby, and the
    standby - still taking batch through the sidecar's trickle - becomes the
    busiest engine. The lane label the balancer writes, the header each alias
    stamps and the scorer that matches them must all name the same things.
    """

    rendered = _render(_LANES)
    config = _picker_config(rendered)
    scorer = next(plugin for plugin in config["plugins"] if plugin.get("name") == "lane-affinity")
    # The exact upstream plugin and parameter names: an upstream rename fails
    # here rather than as a picker that will not start on the fleet.
    assert scorer["type"] == "header-label-affinity-scorer"
    assert scorer["parameters"]["labelKey"] == LANE_LABEL
    header = scorer["parameters"]["headerName"]
    assert header == header.lower()

    gateway = next(
        document
        for document in rendered
        if document["kind"] == "ConfigMap" and "model_list" in str(document.get("data"))
    )
    model_list = yaml.safe_load(
        next(value for value in gateway["data"].values() if "model_list" in value)
    )["model_list"]
    lanes = {
        entry["model_name"]: entry["litellm_params"]["extra_headers"][header]
        for entry in model_list
    }
    assert lanes == {
        "llm-interactive": LANE_PEOPLE,
        "llm-dev": LANE_PEOPLE,
        "llm-batch": LANE_BATCH,
    }

    picker = _container(_pod(_one(rendered, "Deployment", _ROUTER)), "endpoint-picker")
    assert "--allow-experimental-plugins" in picker["args"]
    balancer = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-balancer")), "balancer")
    assert _env(balancer)["FLAKEGRAPH_BALANCER_SERVING_LANE"]["value"] == "people"


def test_the_lane_outweighs_every_other_scorer_together_whatever_they_are_set_to() -> None:
    """Each scorer contributes 0 to 1, so the lane must weigh more than the rest combined.

    Then an engine in the right lane always outscores one in the wrong lane,
    and where no engine is in the right lane the others still decide - no
    request is refused for want of a label. The weight is derived so that
    re-tuning the others cannot quietly demote it.
    """

    for weights in ((2, 2, 3), (2, 2, 5), (7, 1, 9)):
        queue, kv, prefix = weights
        profile = _picker_config(
            _render(
                (
                    *_LANES,
                    f"gateway.placement.endpointPicker.scorerWeights.queue={queue}",
                    f"gateway.placement.endpointPicker.scorerWeights.kvCacheUtilization={kv}",
                    f"gateway.placement.endpointPicker.scorerWeights.prefixCache={prefix}",
                )
            )
        )["schedulingProfiles"][0]["plugins"]
        weight = {entry["pluginRef"]: entry["weight"] for entry in profile}
        others = sum(value for name, value in weight.items() if name != "lane-affinity")
        assert weight["lane-affinity"] > others, weights


def test_without_the_balancer_routing_is_unchanged_and_nothing_experimental_loads() -> None:
    rendered = _render(_SERVING)
    config = _picker_config(rendered)
    assert not [plugin for plugin in config["plugins"] if plugin.get("name") == "lane-affinity"]
    picker = _container(_pod(_one(rendered, "Deployment", _ROUTER)), "endpoint-picker")
    assert "--allow-experimental-plugins" not in picker["args"]

    # And lanes can be turned off on their own, leaving the balancer in place.
    off = _render((*_LANES, "modelServing.balancer.lanes.enabled=false"))
    picker = _container(_pod(_one(off, "Deployment", _ROUTER)), "endpoint-picker")
    assert "--allow-experimental-plugins" not in picker["args"]


_PLACED = (
    *_SERVING,
    "spark.enabled=true",
    "documentParsing.enabled=true",
    "embeddingServing.enabled=true",
    "modelServing.balancer.enabled=true",
)


def test_batch_work_that_shares_a_gpu_or_memory_stays_off_the_engines_people_use() -> None:
    """A clear engine is only as fast as its node is quiet.

    Measured on one clear engine with nothing else on it: memory traffic on
    the same node cost up to 28 % of its speed, GPU work 29 %, both together
    65 %. Spark executors are the first, document parsing the second. New
    executors prefer any other node, parsing is sent only to replicas in the
    batch lane, and the balancer reserves the standby where the least already
    runs. The label vocabulary must be the balancer's own.
    """

    rendered = _render(_PLACED)

    terms = _executor_template(rendered)["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    people = next(
        term
        for term in terms
        if LANE_LABEL in term["podAffinityTerm"]["labelSelector"]["matchLabels"]
    )
    spread = next(
        term
        for term in terms
        if term["podAffinityTerm"]["labelSelector"]["matchLabels"] == {"spark-role": "executor"}
    )
    assert people["podAffinityTerm"]["labelSelector"]["matchLabels"][LANE_LABEL] == LANE_PEOPLE
    # Preferred, never required: when every engine serves someone the job must
    # still run. And stronger than spreading, so doubling up is chosen first.
    assert people["weight"] > spread["weight"]

    parsing = _one(rendered, "StatefulSet", f"{_FULLNAME}-mineru")
    assert parsing["spec"]["template"]["metadata"]["labels"][LANE_LABEL] == LANE_BATCH
    batch_only = _one(rendered, "Service", f"{_FULLNAME}-mineru-batch")
    assert batch_only["spec"]["clusterIP"] == "None"
    assert batch_only["spec"]["selector"][LANE_LABEL] == LANE_BATCH
    shim = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-ocr")), "ocr-shim")
    assert _env(shim)["FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"]["value"] == f"{_FULLNAME}-mineru-batch"

    balancer = _env(
        _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-balancer")), "balancer")
    )
    neighbours = balancer["FLAKEGRAPH_BALANCER_NEIGHBOUR_SELECTORS"]["value"].split(";")
    assert "spark-role=executor" in neighbours
    assert any("component=embedding-serving" in selector for selector in neighbours)
    assert "component=document-parsing" in balancer["FLAKEGRAPH_BALANCER_OCR_SELECTOR"]["value"]


def test_without_lanes_placement_is_exactly_as_before() -> None:
    rendered = _render((*_SERVING, "spark.enabled=true", "documentParsing.enabled=true"))
    terms = _executor_template(rendered)["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert [term["weight"] for term in terms] == [100]
    assert not [doc for doc in rendered if doc["metadata"]["name"] == f"{_FULLNAME}-mineru-batch"]
    shim = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-ocr")), "ocr-shim")
    assert _env(shim)["FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"]["value"] == f"{_FULLNAME}-mineru"
    parsing = _one(rendered, "StatefulSet", f"{_FULLNAME}-mineru")
    assert LANE_LABEL not in parsing["spec"]["template"]["metadata"]["labels"]


def test_the_console_can_move_without_restarting_the_pipeline() -> None:
    """A web page change must not end an in-flight finalization.

    The console ships in the application image, and the workers, the parsing
    shim, the embedding seed and the finalizer's cache seed share that tag -
    so a console-only release would restart the finalizer, which on a real
    corpus is the better part of three hours of LLM work. Unset, the console
    follows the shared tag, which is right when both move together.
    """

    settings = (*_ISOLATED, *_SERVING, "spark.enabled=true", "image.tag=1.2.3")
    together = _render(settings)
    console = _container(_pod(_one(together, "Deployment", f"{_FULLNAME}-app")), "control-plane")
    assert console["image"].endswith(":1.2.3")

    apart = _render((*settings, "controlPlane.imageTag=1.2.4"))
    moved = _container(_pod(_one(apart, "Deployment", f"{_FULLNAME}-app")), "control-plane")
    assert moved["image"].endswith(":1.2.4")
    # Every other workload is byte-identical, so nothing else restarts.
    for kind, name in (
        ("Deployment", f"{_FULLNAME}-extract"),
        ("Deployment", f"{_FULLNAME}-finalize"),
    ):
        assert (
            _one(together, kind, name)["spec"]["template"]
            == _one(apart, kind, name)["spec"]["template"]
        )


def test_every_request_path_and_the_database_can_be_spread_across_failure_domains(
    tmp_path: Path,
) -> None:
    """Place the proxy, the router, and the database instances by the site's domains."""

    spread = {
        "maxSkew": 1,
        "topologyKey": "example.com/power-domain",
        "whenUnsatisfiable": "DoNotSchedule",
    }
    avoid = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "NotIn",
                                "values": ["node-a"],
                            }
                        ]
                    }
                ]
            }
        }
    }
    tolerate = {"key": "example.com/maintenance", "operator": "Exists", "effect": "NoSchedule"}
    placement = {
        "nodeSelector": {"example.com/pool": "front"},
        "tolerations": [tolerate],
        "affinity": avoid,
        "topologySpreadConstraints": [spread],
    }
    path = tmp_path / "placement.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "gateway": {"litellm": placement, "placement": placement},
                "database": {
                    "cloudNativePG": {
                        "podAntiAffinityType": "required",
                        "topologySpreadConstraints": [spread],
                    }
                },
            }
        )
    )
    settings = (*_SERVING, "database.cloudNativePG.enabled=true")
    rendered = _render(settings, values=(path,))
    default = _render(settings)

    for name in (f"{_FULLNAME}-litellm", _ROUTER):
        placed = _pod(_one(rendered, "Deployment", name))
        assert placed["nodeSelector"] == placement["nodeSelector"], name
        assert placed["tolerations"] == [tolerate], name
        assert placed["affinity"] == avoid, name
        assert placed["topologySpreadConstraints"] == [spread], name
        unplaced = _pod(_one(default, "Deployment", name))
        for field in placement:
            assert field not in unplaced, (name, field)

    cluster = _one(rendered, "Cluster", "flakegraph-postgres")["spec"]
    assert cluster["affinity"]["podAntiAffinityType"] == "required"
    assert cluster["topologySpreadConstraints"] == [spread]
    unspread = _one(default, "Cluster", "flakegraph-postgres")["spec"]
    # The operator's own default, stated rather than inherited.
    assert unspread["affinity"]["podAntiAffinityType"] == "preferred"
    assert "topologySpreadConstraints" not in unspread


def test_the_database_image_can_be_pinned_apart_from_the_operator() -> None:
    """Let a deployment take a PostgreSQL security release without an operator upgrade."""

    settings = (*_SERVING, "database.cloudNativePG.enabled=true")
    pinned = _render((*settings, "database.cloudNativePG.imageName=example.com/postgresql:17.11"))
    assert _one(pinned, "Cluster", "flakegraph-postgres")["spec"]["imageName"] == (
        "example.com/postgresql:17.11"
    )
    assert "imageName" not in _one(_render(settings), "Cluster", "flakegraph-postgres")["spec"]


def test_console_admission_requires_gate_or_explicit_isolated_operator() -> None:
    """A fresh console has an initial named principal without trusting direct headers."""

    assert _values()["controlPlane"]["enabled"] is False
    assert _values()["controlPlane"]["auth"]["singleOperator"] == ""
    assert "requires a protected" in _fails(("controlPlane.enabled=true",))
    for setting in ("ingress.enabled=true", "controlPlane.auth.required=true"):
        assert "requires ingress.enabled=false" in _fails((*_ISOLATED, setting))
    isolated = _env(
        _container(
            _pod(_one(_render(_ISOLATED), "Deployment", f"{_FULLNAME}-app")), "control-plane"
        )
    )
    assert isolated["FLAKEGRAPH_SINGLE_OPERATOR"]["value"] == "operator@example.test"
    assert "FLAKEGRAPH_DEMO_IDENTITIES" not in isolated
    assert "FLAKEGRAPH_TRUST_IDENTITY_HEADERS" not in isolated
    for profile in (_MINIMAL_EXAMPLE, _EXTERNAL_EXAMPLE):
        assert not [
            doc
            for doc in _render(values=(profile,))
            if doc["kind"] == "Deployment" and doc["metadata"]["name"] == f"{_FULLNAME}-app"
        ]
