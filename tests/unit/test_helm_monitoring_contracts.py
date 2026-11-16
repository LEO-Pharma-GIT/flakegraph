"""Protect what the chart declares to a kube-prometheus-stack beside it."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from helm import FULLNAME as _FULLNAME
from helm import NAMESPACE as _NAMESPACE
from helm import container as _container
from helm import env as _env
from helm import fails as _fails
from helm import one as _one
from helm import pod as _pod
from helm import render as _render
from helm import schema as _schema
from helm import values as _values

from kg_processor.adapters.distributed.postgres import _SCHEMA_STATEMENTS

_DDL = "\n".join(_SCHEMA_STATEMENTS)

# Everything the monitoring objects hang off: the database they read, the
# planes they scrape, and the ingress they publish Grafana through.
_ENABLED_SETTINGS = (
    "monitoring.enabled=true",
    "database.cloudNativePG.enabled=true",
    "modelServing.enabled=true",
    "ingress.enabled=true",
    "ingress.domain=example.test",
    "ingress.controller=traefik",
    "ingress.tls.secretName=wildcard-tls",
    "ingress.authProxy.enabled=true",
    "ingress.authProxy.existingSecret=oidc",
    "controlPlane.networkPolicy.enabled=true",
)
_ELSEWHERE = (*_ENABLED_SETTINGS, "monitoring.namespace=observability")
_MONITORING_KINDS = {"ServiceMonitor", "PodMonitor", "PrometheusRule"}


def test_monitoring_is_opt_in_and_leaves_no_trace_when_off() -> None:
    """Keep a cluster without the operator's CRDs installable by default.

    The objects need CRDs the chart does not ship, so the block is off, and
    off has to mean absent: not a ServiceMonitor, not a datasource, not the
    per-request callback in the gateway's config.
    """

    assert _values()["monitoring"]["database"]["role"] == "flakegraph_metrics"
    monitoring_schema = _schema()["properties"]["monitoring"]
    assert monitoring_schema["additionalProperties"] is False
    thresholds_schema = monitoring_schema["properties"]["rules"]["properties"]["thresholds"]
    assert thresholds_schema["additionalProperties"] is False

    rendered = _render(())
    assert not [doc for doc in rendered if doc["kind"] in _MONITORING_KINDS]
    assert not [
        doc for doc in rendered if doc["metadata"].get("labels", {}).get("grafana_datasource")
    ]
    assert "prometheus" not in _gateway_config(rendered)


def test_monitoring_refuses_to_render_where_it_cannot_be_applied() -> None:
    """Name the missing prerequisite once, first, rather than per object at apply."""

    without_crds = _fails(("monitoring.enabled=true",), api_versions="")
    assert "monitoring.coreos.com/v1" in without_crds

    without_bundled_database = _fails(("monitoring.enabled=true",))
    assert "database.cloudNativePG.enabled" in without_bundled_database

    # The same shape of refusal for the other CRD-backed planes.
    without_cnpg = _fails(("database.cloudNativePG.enabled=true",), api_versions="")
    assert "postgresql.cnpg.io/v1/Cluster" in without_cnpg
    assert "CloudNativePG" in without_cnpg


def test_every_plane_is_scraped_through_its_own_service() -> None:
    """Scrape each engine once, by name, and the picker with a token it can check."""

    rendered = _render(_ENABLED_SETTINGS)
    monitors = {doc["metadata"]["name"]: doc for doc in rendered if doc["kind"] == "ServiceMonitor"}
    interval = _values()["monitoring"]["scrapeInterval"]

    engines = monitors[f"{_FULLNAME}-vllm"]
    assert engines["spec"]["selector"]["matchLabels"]["flakegraph.io/scrape"] == "true"
    headless = _one(rendered, "Service", f"{_FULLNAME}-vllm-headless")
    balanced = _one(rendered, "Service", f"{_FULLNAME}-vllm")
    assert headless["metadata"]["labels"]["flakegraph.io/scrape"] == "true"
    assert "flakegraph.io/scrape" not in balanced["metadata"]["labels"]
    (endpoint,) = engines["spec"]["endpoints"]
    assert endpoint["port"] == "http"
    assert endpoint["path"] == "/metrics"
    assert endpoint["interval"] == interval
    relabelings = {rule["targetLabel"]: rule["sourceLabels"] for rule in endpoint["relabelings"]}
    assert relabelings["engine"] == ["__meta_kubernetes_pod_name"]
    assert relabelings["node"] == ["__meta_kubernetes_pod_node_name"]

    for name in (f"{_FULLNAME}-litellm", f"{_FULLNAME}-ocr"):
        (endpoint,) = monitors[name]["spec"]["endpoints"]
        assert endpoint["port"] == "http"
        assert endpoint["path"] == "/metrics"
        assert endpoint["interval"] == interval
        assert {"sourceLabels": ["__meta_kubernetes_pod_node_name"], "targetLabel": "node"} in (
            endpoint["relabelings"]
        )

    (endpoint,) = monitors[f"{_FULLNAME}-inference-router"]["spec"]["endpoints"]
    assert endpoint["port"] == "metrics"
    assert endpoint["bearerTokenFile"] == "/var/run/secrets/kubernetes.io/serviceaccount/token"
    router = _one(rendered, "Service", f"{_FULLNAME}-inference-router")
    ports = {port["name"]: port for port in router["spec"]["ports"]}
    assert ports["metrics"]["port"] == 9090
    assert ports["metrics"]["targetPort"] == "metrics"
    # Always on the Service: the endpoint authenticates, so it hands nobody
    # anything, and a ServiceMonitor can only name a Service port.
    assert "metrics" in {
        port["name"]
        for port in _one(_render(()), "Service", f"{_FULLNAME}-inference-router")["spec"]["ports"]
    }


def test_the_picker_is_allowed_to_evaluate_a_scrape() -> None:
    """Grant the two cluster-scoped reviews the metrics filter performs per request."""

    rendered = _render(_ENABLED_SETTINGS)
    name = f"{_FULLNAME}-inference-router-authn"

    role = _one(rendered, "ClusterRole", name)
    granted = {
        (group, resource)
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
        if "create" in rule["verbs"]
    }
    assert ("authentication.k8s.io", "tokenreviews") in granted
    assert ("authorization.k8s.io", "subjectaccessreviews") in granted
    binding = _one(rendered, "ClusterRoleBinding", name)
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "ClusterRole",
        "name": name,
    }
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": f"{_FULLNAME}-inference-router", "namespace": _NAMESPACE}
    ]
    assert not [
        doc
        for doc in _render(())
        if doc["kind"] == "ClusterRole" and doc["metadata"]["name"] == name
    ]


def test_prometheus_may_reach_the_engines_and_nothing_else_changes() -> None:
    """Admit the stack's Prometheus to the sidecar's port without widening the floor."""

    policy = _one(_render(_ENABLED_SETTINGS), "NetworkPolicy", f"{_FULLNAME}-vllm")
    prometheus_rules = [
        rule
        for rule in policy["spec"]["ingress"]
        if {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}}}
        in rule["from"]
    ]

    assert len(prometheus_rules) == 1
    assert prometheus_rules[0]["ports"] == [{"port": "http", "protocol": "TCP"}]
    assert all(
        "namespaceSelector" not in peer
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
    )
    # A stack in another namespace is admitted from that namespace and no other.
    elsewhere = _one(_render(_ELSEWHERE), "NetworkPolicy", f"{_FULLNAME}-vllm")
    assert {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "observability"}},
        "podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}},
    } in [peer for rule in elsewhere["spec"]["ingress"] for peer in rule["from"]]
    # A site with its own scraper or client widens the floor itself, per rule.
    widened = _one(
        _render(
            (
                *_ENABLED_SETTINGS,
                "modelServing.networkPolicy.extraIngress[0].from[0].podSelector.matchLabels.app=site",
                "modelServing.networkPolicy.extraIngress[0].ports[0].port=http",
            )
        ),
        "NetworkPolicy",
        f"{_FULLNAME}-vllm",
    )
    assert {
        "from": [{"podSelector": {"matchLabels": {"app": "site"}}}],
        "ports": [{"port": "http"}],
    } in widened["spec"]["ingress"]


def test_the_gateway_exports_metrics_and_restarts_when_its_config_changes() -> None:
    """Turn the callback on with monitoring, through a change the pods actually see."""

    on = _render(_ENABLED_SETTINGS)
    off = _render(())

    config = yaml.safe_load(_gateway_config(on))
    assert config["litellm_settings"]["callbacks"] == [
        "message_order.instance",
        "stable_schemas.instance",
        "prometheus",
    ]
    # The scraper presents no key, and the proxy otherwise refuses it.
    assert config["litellm_settings"]["require_auth_for_metrics_endpoint"] is False
    assert config["litellm_settings"]["drop_params"] is False
    checksums = {
        state: _one(rendered, "Deployment", f"{_FULLNAME}-litellm")["spec"]["template"]["metadata"][
            "annotations"
        ]["checksum/config"]
        for state, rendered in (("on", on), ("off", off))
    }
    assert checksums["on"] != checksums["off"]


def test_the_database_exposes_its_queues_through_a_kept_read_only_role() -> None:
    """Declare the exporter, the role Grafana reads as, and a password that survives upgrades."""

    rendered = _render(_ENABLED_SETTINGS)
    values = _values()
    role = values["monitoring"]["database"]["role"]
    secret_name = values["monitoring"]["database"]["secretName"]

    cluster = _one(rendered, "Cluster", values["database"]["cloudNativePG"]["clusterName"])
    # The chart scrapes the instances itself; the operator's PodMonitor is deprecated.
    assert cluster["spec"]["monitoring"]["enablePodMonitor"] is False
    monitor = _one(rendered, "PodMonitor", f"{cluster['metadata']['name']}-instances")
    assert monitor["spec"]["selector"]["matchLabels"] == {
        "cnpg.io/cluster": cluster["metadata"]["name"],
        "cnpg.io/podRole": "instance",
    }
    assert monitor["spec"]["podMetricsEndpoints"] == [{"port": "metrics"}]
    assert cluster["spec"]["monitoring"]["customQueriesConfigMap"] == [
        {"name": f"{_FULLNAME}-postgres-queries", "key": "queries.yaml"}
    ]
    (managed_role,) = cluster["spec"]["managed"]["roles"]
    assert managed_role["name"] == role
    assert managed_role["login"] is True
    assert managed_role["inRoles"] == ["pg_read_all_data"]
    assert managed_role["passwordSecret"] == {"name": secret_name}
    assert managed_role["ensure"] == "present"

    credential = _one(rendered, "Secret", secret_name)
    assert credential["type"] == "kubernetes.io/basic-auth"
    assert credential["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"
    assert credential["stringData"]["username"] == role
    password = credential["stringData"]["password"]
    assert len(password) >= 32

    datasource_secret = _one(rendered, "Secret", f"{_FULLNAME}-grafana-datasource")
    assert datasource_secret["metadata"]["labels"]["grafana_datasource"] == "1"
    (datasource,) = yaml.safe_load(datasource_secret["stringData"]["flakegraph-postgres.yaml"])[
        "datasources"
    ]
    assert datasource["uid"] == "flakegraph-postgres"
    assert datasource["type"] == "grafana-postgresql-datasource"
    assert datasource["url"] == "flakegraph-postgres-rw:5432"
    assert datasource["database"] == values["database"]["cloudNativePG"]["database"]
    assert datasource["user"] == role
    assert datasource["secureJsonData"]["password"] == password
    assert datasource["jsonData"]["sslmode"] == "require"
    assert datasource["editable"] is False
    # The credential belongs in Secrets and nowhere a ConfigMap reader can see.
    for doc in rendered:
        if doc["kind"] == "ConfigMap":
            assert password not in json.dumps(doc)
    # The datasource and dashboards live where the stack's sidecars look.
    assert datasource_secret["metadata"]["namespace"] == _NAMESPACE
    elsewhere = _render(_ELSEWHERE)
    assert _one(elsewhere, "Secret", f"{_FULLNAME}-grafana-datasource")["metadata"][
        "namespace"
    ] == ("observability")
    dashboards = [
        doc for doc in elsewhere if doc["metadata"].get("labels", {}).get("grafana_dashboard")
    ]
    assert dashboards
    assert {doc["metadata"]["namespace"] for doc in dashboards} == {"observability"}


def test_grafana_shares_the_fleets_front_door() -> None:
    """Publish Grafana behind the same gate, and let only the edge talk to it."""

    rendered = _render((*_ENABLED_SETTINGS, "controlPlane.enabled=true"))
    values = _values()
    grafana_host = f"{values['monitoring']['grafana']['host']}.example.test"
    middleware = "traefik.ingress.kubernetes.io/router.middlewares"

    main = _one(rendered, "Ingress", _FULLNAME)
    grafana = _one(rendered, "Ingress", f"{_FULLNAME}-grafana")
    assert (
        grafana["metadata"]["annotations"][middleware]
        == main["metadata"]["annotations"][middleware]
    )
    assert grafana["spec"]["tls"] == [{"secretName": "wildcard-tls", "hosts": [grafana_host]}]
    (rule,) = grafana["spec"]["rules"]
    assert rule["host"] == grafana_host
    (path,) = rule["http"]["paths"]
    assert path["backend"]["service"] == {
        "name": f"{values['monitoring']['release']}-grafana",
        "port": {"number": values["monitoring"]["grafana"]["servicePort"]},
    }

    auth = _one(rendered, "Ingress", f"{_FULLNAME}-auth")
    oauth_hosts = {
        rule["host"]
        for rule in auth["spec"]["rules"]
        if any(path["path"] == "/oauth2/" for path in rule["http"]["paths"])
    }
    assert grafana_host in oauth_hosts
    assert grafana_host in auth["spec"]["tls"][0]["hosts"]

    policy = _one(rendered, "NetworkPolicy", f"{_FULLNAME}-grafana")
    assert policy["spec"]["podSelector"] == {"matchLabels": {"app.kubernetes.io/name": "grafana"}}
    (ingress_rule,) = policy["spec"]["ingress"]
    assert ingress_rule["from"] == [
        *values["controlPlane"]["networkPolicy"]["from"],
        {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "prometheus"}}},
    ]
    # An Ingress backs only onto a Service in its own namespace, so a stack
    # elsewhere publishes Grafana itself; the chart renders neither its
    # Ingress nor a policy over pods it cannot select.
    elsewhere = _render(_ELSEWHERE)
    assert not [
        doc
        for doc in elsewhere
        if doc["kind"] in {"Ingress", "NetworkPolicy"}
        and doc["metadata"]["name"] == f"{_FULLNAME}-grafana"
    ]

    # The console's fleet page links to the dashboards by that same host, and
    # only when they exist to link to.
    console = _env(
        _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-app")), "control-plane")
    )
    assert console["FLAKEGRAPH_APP_GRAFANA_URL"]["value"] == f"https://{grafana_host}"
    isolated = _render(
        ("controlPlane.enabled=true", "controlPlane.auth.singleOperator=operator@example.test")
    )
    without = _env(
        _container(_pod(_one(isolated, "Deployment", f"{_FULLNAME}-app")), "control-plane")
    )
    assert "FLAKEGRAPH_APP_GRAFANA_URL" not in without


def test_alerts_cover_each_plane_and_take_every_threshold_from_values(tmp_path: Path) -> None:
    """Keep every alert explained, actionable, and tunable without editing PromQL."""

    rendered = _render(_ENABLED_SETTINGS)
    values = _values()

    prometheus_rule = _one(rendered, "PrometheusRule", _FULLNAME)
    groups = {group["name"]: group["rules"] for group in prometheus_rule["spec"]["groups"]}
    assert set(groups) == {"flakegraph.serving", "flakegraph.documents", "flakegraph.pipeline"}
    alerts = {rule["alert"]: rule for group in groups.values() for rule in group}
    expected = {
        "FlakeGraphEnginePreempted",
        "FlakeGraphEngineDown",
        "FlakeGraphKvCacheSaturated",
        "FlakeGraphTimeToFirstTokenSlow",
        "FlakeGraphGatewayDown",
        "FlakeGraphOcrShimDown",
        "FlakeGraphOcrQueueStalled",
        "FlakeGraphOcrUpstreamFailing",
        "FlakeGraphTasksFailing",
        "FlakeGraphTasksNearAttemptLimit",
        "FlakeGraphTaskLeasesExpired",
        "FlakeGraphWorkerDemandUnmet",
    }
    assert expected <= set(alerts)
    for rule in (rule for group in groups.values() for rule in group):
        assert rule["labels"]["severity"] in {"critical", "warning"}
        assert rule["annotations"]["summary"]
        assert rule["annotations"]["description"]
        assert rule["annotations"]["runbook"] == "docs/kubernetes-fleet.md#verifying-a-fleet"
        # Helm must hand Prometheus's own templates through untouched.
        assert "`" not in json.dumps(rule)

    assert alerts["FlakeGraphEnginePreempted"]["labels"]["severity"] == "critical"
    assert alerts["FlakeGraphEnginePreempted"]["expr"].startswith(
        "increase(vllm:num_preemptions_total"
    )
    assert alerts["FlakeGraphEngineDown"]["expr"] == (
        f'up{{job="{_FULLNAME}-vllm-headless", namespace="{_NAMESPACE}"}} == 0'
    )
    assert alerts["FlakeGraphGatewayDown"]["expr"].startswith(f'up{{job="{_FULLNAME}-litellm"')
    assert alerts["FlakeGraphOcrShimDown"]["expr"].startswith(f'up{{job="{_FULLNAME}-ocr"')
    assert "{{ $labels.engine }}" in alerts["FlakeGraphEnginePreempted"]["annotations"]["summary"]
    assert "{{ $value" in alerts["FlakeGraphKvCacheSaturated"]["annotations"]["description"]

    thresholds = values["monitoring"]["rules"]["thresholds"]
    # Every declared threshold moves some rule: set each to a value nothing
    # else renders and look for it in the rules that come out.
    sentinels = {
        key: f"{731 + index}m" if isinstance(default, str) else round(0.5 + index / 100, 2)
        for index, (key, default) in enumerate(thresholds.items())
    }
    overrides = tmp_path / "thresholds.yaml"
    overrides.write_text(
        yaml.safe_dump({"monitoring": {"rules": {"thresholds": sentinels}}}), encoding="utf-8"
    )
    tuned = json.dumps(
        _one(_render(_ENABLED_SETTINGS, values=(overrides,)), "PrometheusRule", _FULLNAME)
    )
    for key, value in sentinels.items():
        assert str(value) in tuned, f"threshold {key} is declared but no rule uses it"
    assert alerts["FlakeGraphKvCacheSaturated"]["for"] == thresholds["kvCacheUsageFor"]
    assert alerts["FlakeGraphKvCacheSaturated"]["expr"].endswith(f"> {thresholds['kvCacheUsage']}")

    # One demand rule per pool, comparing what KEDA would ask for - tasks per
    # replica, capped at the ceiling - rather than raw demand.
    demand_rules = [
        rule
        for group in groups.values()
        for rule in group
        if rule["alert"] == "FlakeGraphWorkerDemandUnmet"
    ]
    assert {rule["labels"]["pool"] for rule in demand_rules} == set(values["workers"])
    # The ceilings the ScaledObjects render: extraction's is derived from the
    # one sixteen-sequence engine, 16 x 1.5 / 8 slots.
    ceilings = {"prepare": 4, "extract": 3, "finalize": 1}
    for rule in demand_rules:
        pool = values["workers"][rule["labels"]["pool"]]
        assert f") / {pool['slots']})," in rule["expr"]
        assert f"{ceilings[rule['labels']['pool']]}\n" in rule["expr"] + "\n"
        assert f'deployment="{_FULLNAME}-{rule["labels"]["pool"]}"' in rule["expr"]
        assert rule["for"] == thresholds["workerDemandUnmetFor"]


def test_postgres_queries_count_queued_work_only_for_runs_that_can_take_it() -> None:
    """A stopped run's unstarted tasks stay queued for retry but are not queue.

    retry_run requeues only failed or cancelled tasks and resumes the rest, so a
    failed run keeps its untouched tasks queued. No worker claims them, and a
    queue panel, an attempt-limit alert or an oldest-task age that counted them
    would report thousands of tasks waiting on an idle fleet.
    """

    queries = yaml.safe_load(
        _one(_render(_ENABLED_SETTINGS), "ConfigMap", f"{_FULLNAME}-postgres-queries")["data"][
            "queries.yaml"
        ]
    )
    configmap = _one(_render(_ENABLED_SETTINGS), "ConfigMap", f"{_FULLNAME}-postgres-queries")
    # Without the label an edited query is served only after a database restart.
    assert configmap["metadata"]["labels"]["cnpg.io/reload"] == "true"
    runnable = "run.status IN ('queued', 'running')"
    tasks = " ".join(queries["flakegraph_tasks"]["query"].split())
    assert "'parked'" in tasks
    assert (
        "WHEN task.status = 'queued' AND run.status NOT IN ('queued', 'running') THEN 'parked'"
        in tasks
    )
    for name in ("flakegraph_tasks_near_attempt_limit", "flakegraph_tasks_oldest_queued"):
        assert runnable in " ".join(queries[name]["query"].split()), name


def test_postgres_queries_are_well_formed_and_name_only_real_columns() -> None:
    """Keep the exporter's SQL runnable against the schema the adapter actually creates.

    A query naming a column the DDL does not have fails on every scrape, and
    the failure surfaces as a missing series rather than as an error anyone
    reads. So the columns are checked against the adapter's own statements.
    """

    rendered = _render(_ENABLED_SETTINGS)
    values = _values()
    queries = yaml.safe_load(
        _one(rendered, "ConfigMap", f"{_FULLNAME}-postgres-queries")["data"]["queries.yaml"]
    )
    columns = _schema_columns()
    stages = {stage for pool in values["workers"].values() for stage in pool["stages"]}

    expected = {
        "flakegraph_tasks",
        "flakegraph_tasks_failed_recent",
        "flakegraph_tasks_near_attempt_limit",
        "flakegraph_tasks_lease_expired",
        "flakegraph_tasks_oldest_queued",
        "flakegraph_worker_demand",
        "flakegraph_runs",
        "flakegraph_publications",
    }
    assert set(queries) == expected
    for name, query in queries.items():
        assert query["target_databases"] == [values["database"]["cloudNativePG"]["database"]], name
        assert query["primary"] is True, name
        for metric in query["metrics"]:
            ((column, spec),) = metric.items()
            assert spec["usage"] in {"LABEL", "GAUGE"}, (name, column)
            assert spec["description"], (name, column)
        _assert_query_names_real_columns(name, query["query"], columns)

    # The band has to be the one the demand view and KEDA scale on: the run's.
    assert "SELECT task.stage, run.priority_band" in _DDL
    for name in ("flakegraph_tasks", "flakegraph_tasks_oldest_queued"):
        assert "run.priority_band AS" in queries[name]["query"]
        assert "task.priority" not in queries[name]["query"]
    # Every stage the DDL admits reports a zero rather than vanishing.
    for name in (
        "flakegraph_tasks_failed_recent",
        "flakegraph_tasks_near_attempt_limit",
        "flakegraph_tasks_lease_expired",
    ):
        assert stages == set(re.findall(r"'([a-z_]+)'", queries[name]["query"].split("]")[0])), name


_SQL_WORDS = {
    "and",
    "array",
    "as",
    "by",
    "case",
    "coalesce",
    "cross",
    "count",
    "else",
    "end",
    "epoch",
    "exists",
    "extract",
    "float",
    "from",
    "group",
    "hour",
    "in",
    "interval",
    "join",
    "left",
    "min",
    "not",
    "now",
    "on",
    "or",
    "select",
    "then",
    "unnest",
    "when",
    "where",
}


def _assert_query_names_real_columns(name: str, sql: str, columns: dict[str, set[str]]) -> None:
    """Fail on the first identifier that is neither SQL, an alias, nor a real column."""

    without_literals = re.sub(r"'[^']*'", "''", sql)
    # The exporter pins its search_path, so an unqualified table is not found.
    assert not re.search(r"\b(?:FROM|JOIN)\s+flakegraph_", without_literals), name
    tables = set(re.findall(r"\b(?:FROM|JOIN)\s+public\.(flakegraph_[a-z_]+)", without_literals))
    assert tables, name
    aliases = set(re.findall(r"\bAS\s+([a-z_]+)", without_literals))
    aliases |= {
        match for match in re.findall(r"\b([a-z_]+)\(", without_literals) if match not in _SQL_WORDS
    }
    aliases |= set(re.findall(r"\bAS\s+[a-z_]+\(([a-z_]+)\)", without_literals))
    known = set().union(*(columns[table] for table in tables))
    for word in set(re.findall(r"\b[a-z_]+\b", without_literals.lower())):
        if word in _SQL_WORDS or word in tables or word in aliases or word == "public":
            continue
        assert word in known, f"{name} names {word!r}, which is not a column of {sorted(tables)}"


@cache
def _schema_columns() -> dict[str, set[str]]:
    """Read table and view columns out of the adapter's own DDL strings."""

    columns: dict[str, set[str]] = {}
    for match in re.finditer(
        r"CREATE TABLE IF NOT EXISTS (flakegraph_[a-z_]+) \((.*?)\n    \)", _DDL, re.DOTALL
    ):
        table, body = match.groups()
        columns[table] = set(
            re.findall(
                r"^\s*([a-z_]+)\s+(?:TEXT|INTEGER|TIMESTAMPTZ|JSONB|BIGSERIAL|BIGINT)",
                body,
                re.MULTILINE,
            )
        )
    view = re.search(
        r"CREATE OR REPLACE VIEW (flakegraph_[a-z_]+) AS\n    SELECT (.*?)\n    FROM",
        _DDL,
        re.DOTALL,
    )
    assert view is not None
    # Output columns are either aliased or a bare `demand.<column>,` entry.
    columns[view.group(1)] = set(re.findall(r"\bAS ([a-z_]+)", view.group(2))) | set(
        re.findall(r"\bdemand\.([a-z_]+),", view.group(2))
    )
    assert {
        "stage",
        "status",
        "priority",
        "attempts",
        "max_attempts",
        "lease_expires_at",
    } <= columns["flakegraph_task"]
    assert columns["flakegraph_worker_demand"] == {"stage", "priority_band", "desired_workers"}
    return columns


def _gateway_config(rendered: list[dict[str, Any]]) -> str:
    config: str = _one(rendered, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]
    return config
