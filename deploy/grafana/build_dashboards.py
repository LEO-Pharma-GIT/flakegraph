"""Generate the FlakeGraph Grafana dashboards as checked-in JSON.

Grafana dashboards are verbose JSON with hand-maintained panel ids and grid
coordinates; editing them by hand is where drift and layout bugs come from.
This script is the source: run it with ``uv run python deploy/grafana/build_dashboards.py``
and commit the files it writes to ``deploy/helm/flakegraph/dashboards/``, where
the chart wraps each one in a ConfigMap for the stack's Grafana sidecar. The unit
test regenerates and compares, so a dashboard edited in place fails CI until the
generator is updated too.

Only the standard library is used, so the script runs anywhere the repo does.

Metric names follow what a running deployment exposes: the vLLM exposition,
NVIDIA's DCGM exporter, the CloudNativePG exporter, LiteLLM's label tables, and
the LiteLLM and FlakeGraph table definitions in Postgres. The endpoint picker's
names come from the llm-d-router metrics reference, since its metrics port only
answers to a service-account token.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

Panel = dict[str, Any]
Target = dict[str, Any]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "helm" / "flakegraph" / "dashboards"

# Datasources are addressed by uid, never by name: the stack provisions its
# Prometheus with uid `prometheus` and the chart provisions the read-only
# Postgres datasource with uid `flakegraph-postgres`. A rename of either in the
# UI leaves every panel here working.
PROMETHEUS = {"type": "prometheus", "uid": "prometheus"}
POSTGRES = {"type": "grafana-postgresql-datasource", "uid": "flakegraph-postgres"}

TAG = "flakegraph"
GRID_COLUMNS = 24
SCHEMA_VERSION = 39

# Label selectors shared by most queries. `$namespace` is a single value, the
# rest are multi-value with All, so they always use regex matches.
NS = 'namespace="$namespace"'
ENGINE = f'{NS}, engine=~"$engine"'
CLASS = 'consumer_class=~"$consumer_class"'
WORKER_POOLS = "prepare|extract|finalize"

# Sidecar histogram quantiles rendered from Prometheus histograms.
P50, P95, P99 = "0.5", "0.95", "0.99"

# Nodes pinned to a colour scheme; the stat helpers take these as thresholds.
GREEN, YELLOW, ORANGE, RED, BLUE = "green", "yellow", "orange", "red", "blue"


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def prom(expr: str, legend: str = "__auto", *, instant: bool = False) -> Target:
    """Return a Prometheus query target.

    ``instant`` is for stat and table panels that want the latest value only;
    everything else is a range query so it draws over the dashboard's window.
    """

    return {
        "datasource": PROMETHEUS,
        "editorMode": "code",
        "expr": expr,
        "instant": instant,
        "range": not instant,
        "legendFormat": legend,
    }


def sql(query: str, *, table: bool = False) -> Target:
    """Return a Postgres query target.

    Time-series queries must select a ``time`` column (``$__timeGroupAlias``), a
    string ``metric`` column and one numeric ``value`` column; Grafana turns each
    distinct metric into a series. ``table`` keeps the rows as they are.
    """

    return {
        "datasource": POSTGRES,
        "editorMode": "code",
        "format": "table" if table else "time_series",
        "rawQuery": True,
        "rawSql": query.strip(),
    }


def quantile(q: str, histogram: str, selector: str, by: str = "") -> str:
    """Render a ``histogram_quantile`` over a Prometheus histogram's buckets."""

    labels = f"le, {by}" if by else "le"
    return (
        f"histogram_quantile({q}, sum by ({labels}) "
        f"(rate({histogram}_bucket{{{selector}}}[$__rate_interval])))"
    )


# ---------------------------------------------------------------------------
# Field configuration
# ---------------------------------------------------------------------------


def thresholds(base: str, *steps: tuple[float, str]) -> dict[str, Any]:
    """Return absolute thresholds: ``base`` colour below the first step."""

    return {
        "mode": "absolute",
        "steps": [
            {"color": base, "value": None},
            *({"color": colour, "value": value} for value, colour in steps),
        ],
    }


def _field_defaults(
    unit: str | None,
    *,
    limits: tuple[float | None, float | None] = (None, None),
    decimals: int | None = None,
    steps: dict[str, Any] | None = None,
    color: dict[str, Any] | None = None,
    custom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "color": color or {"mode": "palette-classic"},
        "thresholds": steps or thresholds(GREEN, (80, RED)),
    }
    if unit:
        defaults["unit"] = unit
    low, high = limits
    if low is not None:
        defaults["min"] = low
    if high is not None:
        defaults["max"] = high
    if decimals is not None:
        defaults["decimals"] = decimals
    if custom:
        defaults["custom"] = custom
    return defaults


def _panel(
    kind: str,
    title: str,
    targets: Sequence[Target],
    *,
    description: str,
    w: int,
    h: int,
    defaults: dict[str, Any],
    options: dict[str, Any],
    overrides: Sequence[dict[str, Any]] = (),
    transformations: Sequence[dict[str, Any]] = (),
) -> Panel:
    if not 1 <= w <= GRID_COLUMNS:
        raise ValueError(f"panel {title!r} width {w} is outside the {GRID_COLUMNS}-column grid")
    panel: Panel = {
        "type": kind,
        "title": title,
        "description": description,
        "datasource": targets[0]["datasource"],
        "gridPos": {"w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": list(overrides)},
        "options": options,
        "targets": [{**target, "refId": chr(ord("A") + i)} for i, target in enumerate(targets)],
    }
    if transformations:
        panel["transformations"] = list(transformations)
    return panel


def timeseries(
    title: str,
    targets: Sequence[Target],
    *,
    description: str = "",
    unit: str = "short",
    w: int = 12,
    h: int = 8,
    stack: bool = False,
    bars: bool = False,
    fill: int = 10,
    limits: tuple[float | None, float | None] = (None, None),
    decimals: int | None = None,
    steps: dict[str, Any] | None = None,
    overrides: Sequence[dict[str, Any]] = (),
) -> Panel:
    """Return a time-series panel; ``stack`` stacks series, ``bars`` draws bars."""

    custom: dict[str, Any] = {
        "drawStyle": "bars" if bars else "line",
        "fillOpacity": fill,
        "lineWidth": 1,
        "showPoints": "never",
        "spanNulls": False,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
    }
    return _panel(
        "timeseries",
        title,
        targets,
        description=description,
        w=w,
        h=h,
        defaults=_field_defaults(
            unit, limits=limits, decimals=decimals, steps=steps, custom=custom
        ),
        options={
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        overrides=overrides,
    )


def stat(
    title: str,
    targets: Sequence[Target],
    *,
    description: str = "",
    unit: str = "short",
    w: int = 4,
    h: int = 4,
    steps: dict[str, Any] | None = None,
    decimals: int | None = None,
    graph: bool = False,
) -> Panel:
    """Return a stat tile coloured by ``steps``; ``graph`` adds a sparkline."""

    return _panel(
        "stat",
        title,
        targets,
        description=description,
        w=w,
        h=h,
        defaults=_field_defaults(
            unit, decimals=decimals, steps=steps or thresholds(GREEN), color={"mode": "thresholds"}
        ),
        options={
            "colorMode": "value",
            "graphMode": "area" if graph else "none",
            "justifyMode": "auto",
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "textMode": "auto",
            "wideLayout": True,
        },
    )


def gauge(
    title: str,
    targets: Sequence[Target],
    *,
    description: str = "",
    unit: str = "percentunit",
    w: int = 4,
    h: int = 4,
    limits: tuple[float | None, float | None] = (0, 1),
    steps: dict[str, Any] | None = None,
) -> Panel:
    """Return a radial gauge, for a single bounded ratio."""

    return _panel(
        "gauge",
        title,
        targets,
        description=description,
        w=w,
        h=h,
        defaults=_field_defaults(
            unit,
            limits=limits,
            steps=steps or thresholds(GREEN, (0.7, ORANGE), (0.9, RED)),
            color={"mode": "thresholds"},
        ),
        options={
            "orientation": "auto",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showThresholdLabels": False,
            "showThresholdMarkers": True,
        },
    )


def bargauge(
    title: str,
    targets: Sequence[Target],
    *,
    description: str = "",
    unit: str = "percentunit",
    w: int = 8,
    h: int = 8,
    limits: tuple[float | None, float | None] = (0, 1),
    steps: dict[str, Any] | None = None,
    transformations: Sequence[dict[str, Any]] = (),
) -> Panel:
    """Return a horizontal bar gauge, one bar per series."""

    return _panel(
        "bargauge",
        title,
        targets,
        description=description,
        w=w,
        h=h,
        defaults=_field_defaults(
            unit,
            limits=limits,
            steps=steps or thresholds(GREEN, (0.7, ORANGE), (0.9, RED)),
            color={"mode": "thresholds"},
        ),
        options={
            "displayMode": "gradient",
            "orientation": "horizontal",
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "showUnfilled": True,
            "valueMode": "color",
        },
        transformations=transformations,
    )


def table(
    title: str,
    targets: Sequence[Target],
    *,
    description: str = "",
    w: int = 24,
    h: int = 8,
    sort_by: str | None = None,
    overrides: Sequence[dict[str, Any]] = (),
    transformations: Sequence[dict[str, Any]] = (),
) -> Panel:
    """Return a table panel, optionally pre-sorted descending on ``sort_by``."""

    options: dict[str, Any] = {"cellHeight": "sm", "showHeader": True}
    if sort_by:
        options["sortBy"] = [{"displayName": sort_by, "desc": True}]
    return _panel(
        "table",
        title,
        targets,
        description=description,
        w=w,
        h=h,
        defaults=_field_defaults(None, custom={"align": "auto", "filterable": True}),
        options=options,
        overrides=overrides,
        transformations=transformations,
    )


def text(title: str, markdown: str, *, w: int = 8, h: int = 4) -> Panel:
    """Return a markdown panel, for a note that a row of numbers cannot carry."""

    return {
        "type": "text",
        "title": title,
        "gridPos": {"w": w, "h": h},
        "options": {"mode": "markdown", "content": markdown.strip()},
    }


def unit_override(name: str, unit: str) -> dict[str, Any]:
    """Give one series (by display name) a different unit from the panel."""

    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "unit", "value": unit}],
    }


def hide_fields(*names: str) -> dict[str, Any]:
    """Return an organize transformation that drops the named columns."""

    return {"id": "organize", "options": {"excludeByName": dict.fromkeys(names, True)}}


def row(
    title: str, panels: Sequence[Panel], *, collapsed: bool = False, repeat: str | None = None
) -> Panel:
    """Return a row that groups ``panels``; collapsed rows load lazily."""

    row_panel: Panel = {
        "type": "row",
        "title": title,
        "collapsed": collapsed,
        "gridPos": {"w": GRID_COLUMNS, "h": 1},
        "panels": list(panels),
    }
    if repeat:
        row_panel["repeat"] = repeat
    return row_panel


# ---------------------------------------------------------------------------
# Layout and dashboard assembly
# ---------------------------------------------------------------------------


def _layout(rows: Sequence[Panel]) -> list[Panel]:
    """Assign ids and grid coordinates, filling each row left to right.

    Panels wrap to a new line when the next one would cross the right edge, so
    a row's panels only need widths and heights. Collapsed rows keep their
    panels nested, which is how Grafana expects them; expanded rows are
    flattened to top level after the row header.
    """

    placed: list[Panel] = []
    next_id = 1
    y = 0
    for row_panel in rows:
        header = {
            **row_panel,
            "id": next_id,
            "gridPos": {"x": 0, "y": y, "w": GRID_COLUMNS, "h": 1},
        }
        next_id += 1
        y += 1
        x = 0
        line_height = 0
        children: list[Panel] = []
        for panel in row_panel["panels"]:
            w, h = panel["gridPos"]["w"], panel["gridPos"]["h"]
            if x + w > GRID_COLUMNS:
                x = 0
                y += line_height
                line_height = 0
            children.append({**panel, "id": next_id, "gridPos": {"x": x, "y": y, "w": w, "h": h}})
            next_id += 1
            x += w
            line_height = max(line_height, h)
        y += line_height
        if header["collapsed"]:
            header["panels"] = children
            placed.append(header)
        else:
            header["panels"] = []
            placed.extend([header, *children])
    return placed


def variable(
    name: str,
    query: str,
    *,
    label: str,
    multi: bool = True,
    current: str | None = None,
) -> dict[str, Any]:
    """Return a Prometheus ``label_values`` template variable.

    Multi-value variables include All; a single-value one takes ``current`` as
    its default so a fresh Grafana opens on the right namespace. All is spelled
    as a match-anything regex rather than the option list, because a list
    Grafana has not populated yet (a class nobody has used, a dashboard opened
    before the first scrape) would otherwise expand to a filter nothing matches.
    """

    selected = (
        {"selected": True, "text": ["All"], "value": ["$__all"]}
        if multi
        else {"selected": True, "text": current, "value": current}
    )
    return {
        "type": "query",
        "name": name,
        "label": label,
        "datasource": PROMETHEUS,
        "definition": query,
        "query": {"qryType": 1, "query": query, "refId": "PrometheusVariableQueryEditor"},
        "refresh": 2,
        "sort": 1,
        "multi": multi,
        "includeAll": multi,
        **({"allValue": ".*"} if multi else {}),
        "current": selected,
        "options": [],
    }


def namespace_variable() -> dict[str, Any]:
    """The namespace the release lives in; defaults to the one the engines use."""

    return variable(
        "namespace",
        "label_values(vllm:num_requests_running, namespace)",
        label="Namespace",
        multi=False,
        current="flakegraph",
    )


def node_variable() -> dict[str, Any]:
    """Cluster nodes, from kube-state-metrics."""

    return variable("node", "label_values(kube_node_info, node)", label="Node")


def dcgm_namespace_variable() -> dict[str, Any]:
    """The namespace the DCGM exporter runs in, read from its own series.

    The exporter is a prerequisite the chart does not install (the GPU
    Operator brings one, deploy/spark/install-cluster.sh installs one into
    ``gpu-monitoring``), so where it lives is the cluster's business. The GPU
    panels join its series to kube-state-metrics by pod name inside that
    namespace, and the variable resolves to whichever namespace is scraping
    ``DCGM_FI_DEV_GPU_UTIL``; with no exporter the options are empty and the
    panels say so rather than silently showing nothing.
    """

    return variable(
        "dcgm_namespace",
        "label_values(DCGM_FI_DEV_GPU_UTIL, namespace)",
        label="DCGM exporter namespace",
        multi=False,
        current="gpu-monitoring",
    )


def engine_variable() -> dict[str, Any]:
    """Engine pods, from the pod-name label the ServiceMonitor stamps on vLLM."""

    return variable(
        "engine",
        f"label_values(vllm:num_requests_running{{{NS}}}, engine)",
        label="Engine",
    )


def consumer_class_variable(metric: str) -> dict[str, Any]:
    """Consumer classes seen by ``metric`` (sidecar or OCR shim counters)."""

    return variable(
        "consumer_class",
        f"label_values({metric}{{{NS}}}, consumer_class)",
        label="Consumer class",
    )


def dashboard(
    uid: str,
    title: str,
    *,
    description: str,
    variables: Sequence[dict[str, Any]],
    rows: Sequence[Panel],
) -> dict[str, Any]:
    """Assemble a complete dashboard from rows of panels."""

    return {
        "uid": uid,
        "title": title,
        "description": description,
        "tags": [TAG],
        "timezone": "browser",
        "editable": True,
        "graphTooltip": 1,
        "schemaVersion": SCHEMA_VERSION,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "timepicker": {},
        "links": [
            {
                "type": "dashboards",
                "title": "FlakeGraph",
                "tags": [TAG],
                "asDropdown": True,
                "includeVars": True,
                "keepTime": True,
            }
        ],
        "templating": {"list": list(variables)},
        "annotations": {"list": []},
        "panels": _layout(rows),
    }


# ---------------------------------------------------------------------------
# Dashboards
# ---------------------------------------------------------------------------


def _node_join(expr: str) -> str:
    """Attach the node name to a node-exporter expression.

    The stack's node-exporter ServiceMonitor adds no ``node`` label, so its
    series only carry ``instance`` (pod IP and port). ``node_uname_info`` shares
    that instance and names the host, which is what the ``$node`` variable
    holds.
    """

    return f'({expr}) * on (instance) group_left (nodename) node_uname_info{{nodename=~"$node"}}'


def _gpu(metric: str) -> str:
    """Attach the node name to a DCGM series.

    The exporter's own ``hostname`` label is its DaemonSet pod's name, so the
    node comes from kube-state-metrics via the ``pod`` label Prometheus adds to
    every scraped series, looked up in the exporter's namespace.
    """

    return (
        f"{metric} * on (pod) group_left (node) "
        f'kube_pod_info{{namespace="$dcgm_namespace", node=~"$node"}}'
    )


def fleet_overview() -> dict[str, Any]:
    """The landing page: is the fleet healthy, and how busy is it."""

    counts = thresholds(GREEN)
    zero_is_good = thresholds(GREEN, (1, RED))
    return dashboard(
        "flakegraph-fleet",
        "Fleet Overview",
        description="Health and load of the whole FlakeGraph fleet at a glance.",
        variables=[
            namespace_variable(),
            node_variable(),
            dcgm_namespace_variable(),
            engine_variable(),
            consumer_class_variable("flakegraph_sidecar_requests_total"),
        ],
        rows=[
            row(
                "Right now",
                [
                    stat(
                        "Engines ready",
                        [
                            prom(
                                "sum(kube_statefulset_status_replicas_ready"
                                f'{{{NS}, statefulset=~".*-vllm"}})',
                                "ready",
                                instant=True,
                            ),
                            prom(
                                f'sum(kube_statefulset_replicas{{{NS}, statefulset=~".*-vllm"}})',
                                "desired",
                                instant=True,
                            ),
                        ],
                        description="vLLM pods passing readiness against the number the "
                        "StatefulSet wants. Fewer ready than desired is a rollout or a lost "
                        "engine; the engine-down alert fires after five minutes.",
                        steps=thresholds(RED, (1, ORANGE), (4, GREEN)),
                    ),
                    stat(
                        "Requests running",
                        [prom(f"sum(vllm:num_requests_running{{{ENGINE}}})", instant=True)],
                        description="Requests being decoded across the selected engines.",
                        steps=counts,
                    ),
                    stat(
                        "Requests waiting",
                        [prom(f"sum(vllm:num_requests_waiting{{{ENGINE}}})", instant=True)],
                        description="Requests the engines have accepted but not started. "
                        "Anything sustained above zero means callers are queueing.",
                        steps=zero_is_good,
                    ),
                    stat(
                        "Generation tokens/s",
                        [
                            prom(
                                "sum(rate(vllm:generation_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                instant=True,
                            )
                        ],
                        description="Output tokens per second across the selected engines.",
                        unit="suffix: tok/s",
                        steps=thresholds(BLUE),
                        decimals=0,
                    ),
                    stat(
                        "Prompt tokens/s",
                        [
                            prom(
                                f"sum(rate(vllm:prompt_tokens_total{{{ENGINE}}}[$__rate_interval]))",
                                instant=True,
                            )
                        ],
                        description="Prompt tokens per second, including tokens served from the "
                        "prefix cache.",
                        unit="suffix: tok/s",
                        steps=thresholds(BLUE),
                        decimals=0,
                    ),
                    stat(
                        "Preemptions, 24h",
                        [
                            prom(
                                f"sum(increase(vllm:num_preemptions_total{{{ENGINE}}}[24h]))",
                                instant=True,
                            )
                        ],
                        description="Running requests evicted from the KV cache in the last day. "
                        "The serving design exists to keep this at zero; any value means an "
                        "engine was oversubscribed.",
                        steps=zero_is_good,
                        decimals=0,
                    ),
                    stat(
                        "OCR queued",
                        [
                            prom(
                                f'max(flakegraph_ocr_queue_depth{{{NS}, status="waiting"}})',
                                instant=True,
                            )
                        ],
                        description="Documents waiting for a parsing replica. Every shim replica "
                        "reports the same shared queue, hence max rather than sum.",
                        steps=thresholds(GREEN, (1, YELLOW), (10, RED)),
                    ),
                    stat(
                        "OCR in flight",
                        [prom(f"sum(flakegraph_ocr_in_flight{{{NS}}})", instant=True)],
                        description="Parses dispatched to MinerU replicas right now.",
                        steps=counts,
                    ),
                    stat(
                        "Tasks queued",
                        [
                            prom(
                                f'sum(cnpg_flakegraph_tasks_count{{{NS}, status="queued"}})',
                                instant=True,
                            )
                        ],
                        description=(
                            "Pipeline tasks waiting for a worker, all stages, in queued or "
                            "running runs. A failed or cancelled run keeps its unstarted "
                            "tasks for a retry; those are reported as parked, not counted here."
                        ),
                        steps=counts,
                    ),
                    stat(
                        "Tasks running",
                        [
                            prom(
                                f'sum(cnpg_flakegraph_tasks_count{{{NS}, status="running"}})',
                                instant=True,
                            )
                        ],
                        description="Pipeline tasks under lease by a worker.",
                        steps=counts,
                    ),
                    stat(
                        "Worker pods",
                        [
                            prom(
                                "sum by (deployment) (kube_deployment_status_replicas_available"
                                f'{{{NS}, deployment=~".*-({WORKER_POOLS})$"}})',
                                "{{deployment}}",
                                instant=True,
                            )
                        ],
                        description="Available replicas per worker pool. KEDA scales each pool "
                        "to zero when its stages have no work.",
                        steps=counts,
                    ),
                    stat(
                        "Runs active",
                        [
                            prom(
                                "sum(cnpg_flakegraph_runs_count"
                                f'{{{NS}, status=~"planning|queued|running"}})',
                                instant=True,
                            )
                        ],
                        description="Graph builds that have not finished.",
                        steps=counts,
                    ),
                ],
            ),
            row(
                "Engines",
                [
                    bargauge(
                        "KV-cache usage",
                        [prom(f"vllm:kv_cache_usage_perc{{{ENGINE}}}", "{{engine}}")],
                        description="Share of each engine's KV cache in use. Above 0.9 the next "
                        "burst preempts a running request.",
                        w=6,
                    ),
                    timeseries(
                        "Running and waiting per engine",
                        [
                            prom(f"vllm:num_requests_running{{{ENGINE}}}", "{{engine}} running"),
                            prom(f"vllm:num_requests_waiting{{{ENGINE}}}", "{{engine}} waiting"),
                        ],
                        description="Stacked: the waiting share should be a sliver.",
                        w=9,
                        stack=True,
                    ),
                    timeseries(
                        "Generation tokens/s per engine",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:generation_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Decode throughput per engine. Engines should track each "
                        "other; one far below the rest is misrouted or unhealthy.",
                        unit="suffix: tok/s",
                        w=9,
                    ),
                    timeseries(
                        "Time to first token p95 per engine",
                        [
                            prom(
                                quantile(P95, "vllm:time_to_first_token_seconds", ENGINE, "engine"),
                                "{{engine}}",
                            )
                        ],
                        description="How long a caller waits for the first token. Interactive "
                        "callers give up around ten seconds.",
                        unit="s",
                        w=12,
                        steps=thresholds(GREEN, (5, ORANGE), (10, RED)),
                    ),
                    timeseries(
                        "Queue time p95 per engine",
                        [
                            prom(
                                quantile(P95, "vllm:request_queue_time_seconds", ENGINE, "engine"),
                                "{{engine}}",
                            )
                        ],
                        description="Time a request sat in the engine's queue before its first "
                        "scheduler step. Rises before TTFT does.",
                        unit="s",
                        w=12,
                    ),
                ],
            ),
            row(
                "Host $node",
                [
                    timeseries(
                        "GPU utilisation",
                        [prom(_gpu("DCGM_FI_DEV_GPU_UTIL"), "{{node}} gpu{{gpu}}")],
                        description="SM utilisation per GPU, as the DCGM exporter reports it. "
                        "Empty until a DCGM exporter is scraped; see the DCGM namespace variable.",
                        unit="percent",
                        w=6,
                        h=7,
                        limits=(0, 100),
                    ),
                    timeseries(
                        "GPU power",
                        [prom(_gpu("DCGM_FI_DEV_POWER_USAGE"), "{{node}} gpu{{gpu}}")],
                        description="Board power draw.",
                        unit="watt",
                        w=6,
                        h=7,
                    ),
                    timeseries(
                        "GPU temperature",
                        [prom(_gpu("DCGM_FI_DEV_GPU_TEMP"), "{{node}} gpu{{gpu}}")],
                        description="Die temperature. Thermal throttling shows up as clocks "
                        "dropping while this stays flat at the limit.",
                        unit="celsius",
                        w=6,
                        h=7,
                    ),
                    timeseries(
                        "CPU utilisation",
                        [
                            prom(
                                _node_join(
                                    "1 - avg by (instance) (rate(node_cpu_seconds_total"
                                    '{mode="idle"}[$__rate_interval]))'
                                ),
                                "{{nodename}}",
                            )
                        ],
                        description="Share of all cores busy. Parsing and worker pods are the "
                        "CPU consumers; the engines barely register.",
                        unit="percentunit",
                        w=6,
                        h=7,
                        limits=(0, 1),
                    ),
                    timeseries(
                        "Memory used",
                        [
                            prom(
                                _node_join(
                                    "1 - node_memory_MemAvailable_bytes / "
                                    "node_memory_MemTotal_bytes"
                                ),
                                "{{nodename}}",
                            )
                        ],
                        description="Share of the node's memory in use. On a unified-memory part "
                        "(a DGX Spark, for one) this is also the GPU memory: model weights and "
                        "the KV cache live here, so a full node starves the engine as well as "
                        "the pods.",
                        unit="percentunit",
                        w=8,
                        h=7,
                        limits=(0, 1),
                        steps=thresholds(GREEN, (0.85, ORANGE), (0.95, RED)),
                    ),
                    timeseries(
                        "Root disk used",
                        [
                            prom(
                                _node_join(
                                    "1 - node_filesystem_avail_bytes"
                                    '{mountpoint="/", fstype!="rootfs"} / '
                                    'node_filesystem_size_bytes{mountpoint="/", fstype!="rootfs"}'
                                ),
                                "{{nodename}}",
                            )
                        ],
                        description="The root filesystem holds container images, the local-path "
                        "PVCs and the model cache. Full means pods stop scheduling.",
                        unit="percentunit",
                        w=8,
                        h=7,
                        limits=(0, 1),
                        steps=thresholds(GREEN, (0.8, ORANGE), (0.9, RED)),
                    ),
                    timeseries(
                        "Network",
                        [
                            prom(
                                _node_join(
                                    "sum by (instance) (rate(node_network_receive_bytes_total"
                                    '{device!~"lo|veth.*|cni.*|flannel.*|docker.*"}'
                                    "[$__rate_interval]))"
                                ),
                                "{{nodename}} rx",
                            ),
                            prom(
                                _node_join(
                                    "sum by (instance) (rate(node_network_transmit_bytes_total"
                                    '{device!~"lo|veth.*|cni.*|flannel.*|docker.*"}'
                                    "[$__rate_interval]))"
                                ),
                                "{{nodename}} tx",
                            ),
                        ],
                        description="Physical interfaces only; pod and CNI interfaces are "
                        "excluded so traffic is not counted twice.",
                        unit="Bps",
                        w=8,
                        h=7,
                    ),
                ],
                repeat="node",
            ),
            row(
                "Throughput history",
                [
                    timeseries(
                        "Sidecar requests/s by consumer class",
                        [
                            prom(
                                "sum by (consumer_class) (rate(flakegraph_sidecar_requests_total"
                                f"{{{ENGINE}, {CLASS}}}[$__rate_interval]))",
                                "{{consumer_class}}",
                            )
                        ],
                        description="Requests reaching the engines' sidecars, by the class the "
                        "presented key resolved to. `unauthenticated` requests were rejected.",
                        unit="reqps",
                        w=8,
                    ),
                    timeseries(
                        "Tokens/s by engine",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:generation_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Generation throughput stacked to a fleet total.",
                        unit="suffix: tok/s",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "OCR parses/min by outcome",
                        [
                            prom(
                                "sum by (outcome) (rate(flakegraph_ocr_requests_total"
                                f"{{{NS}, {CLASS}}}[$__rate_interval])) * 60",
                                "{{outcome}}",
                            )
                        ],
                        description="Documents finished per minute, by how they ended. "
                        "`upstream_timeout` and `upstream_error` are the parser failing; "
                        "`rejected` is a bad key or a full queue.",
                        unit="cpm",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Tasks completed/min by stage",
                        [
                            prom(
                                "sum by (stage) (clamp_min(increase(cnpg_flakegraph_tasks_count"
                                f'{{{NS}, status="succeeded"}}[5m]), 0)) / 5',
                                "{{stage}}",
                            )
                        ],
                        description="Derived from the succeeded-task gauge, so a run being "
                        "deleted reads as a dip (clamped at zero) rather than as work. The "
                        "Pipeline dashboard has the exact history from Postgres.",
                        unit="cpm",
                        w=12,
                        stack=True,
                    ),
                    timeseries(
                        "LiteLLM requests/s by key alias",
                        [
                            prom(
                                "sum by (api_key_alias) (rate("
                                f"litellm_proxy_total_requests_metric_total{{{NS}}}"
                                "[$__rate_interval]))",
                                "{{api_key_alias}}",
                            )
                        ],
                        description="Requests through the gateway, by the alias of the virtual "
                        "key that made them.",
                        unit="reqps",
                        w=12,
                    ),
                ],
            ),
            row(
                "Alerts",
                [
                    table(
                        "Firing alerts",
                        [
                            prom(
                                'ALERTS{alertstate="firing", alertname!="Watchdog"}',
                                instant=True,
                            )
                        ],
                        description="Every Prometheus alert firing in the cluster, from the "
                        "chart's rules and the stack's bundled ones. Watchdog always fires "
                        "to prove the pipeline works and is hidden.",
                        transformations=[
                            {"id": "labelsToFields", "options": {}},
                            hide_fields("Time", "Value", "__name__", "alertstate"),
                        ],
                    )
                ],
            ),
        ],
    )


def llm_serving() -> dict[str, Any]:
    """Everything about the engines and the router in front of them."""

    return dashboard(
        "flakegraph-llm-serving",
        "LLM Serving",
        description="vLLM engines, the priority sidecars and the endpoint picker.",
        variables=[
            namespace_variable(),
            engine_variable(),
            consumer_class_variable("flakegraph_sidecar_requests_total"),
        ],
        rows=[
            row(
                "Load",
                [
                    timeseries(
                        "Running and waiting per engine",
                        [
                            prom(f"vllm:num_requests_running{{{ENGINE}}}", "{{engine}} running"),
                            prom(f"vllm:num_requests_waiting{{{ENGINE}}}", "{{engine}} waiting"),
                        ],
                        description="Requests in decode and requests queued, per engine.",
                        w=8,
                    ),
                    timeseries(
                        "Waiting by reason",
                        [
                            prom(
                                "sum by (engine, reason) (vllm:num_requests_waiting_by_reason"
                                f"{{{ENGINE}}})",
                                "{{engine}} {{reason}}",
                            )
                        ],
                        description="`capacity` is a full KV cache or batch; `deferred` is a "
                        "request the engine chose to hold, which is what priority does.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "KV-cache usage per engine",
                        [prom(f"vllm:kv_cache_usage_perc{{{ENGINE}}}", "{{engine}}")],
                        description="Share of KV-cache blocks in use over time; the alert "
                        "fires when it stays above the configured fraction.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                        steps=thresholds(GREEN, (0.9, RED)),
                    ),
                    timeseries(
                        "Requests in flight by consumer class",
                        [
                            prom(
                                "sum by (consumer_class) (flakegraph_sidecar_requests_in_flight"
                                f"{{{ENGINE}, {CLASS}}})",
                                "{{consumer_class}}",
                            )
                        ],
                        description="Requests the sidecars have accepted and not finished "
                        "answering, including time spent streaming.",
                        w=12,
                        stack=True,
                    ),
                    timeseries(
                        "Priority stamped per band",
                        [
                            prom(
                                "sum by (consumer_class, priority) (rate("
                                f"flakegraph_sidecar_priority_stamped_total{{{ENGINE}, {CLASS}}}"
                                "[$__rate_interval]))",
                                "{{consumer_class}} (priority {{priority}})",
                            )
                        ],
                        description="Generation requests per second the sidecars stamped with "
                        "each priority band before forwarding. A class landing on the wrong "
                        "band is a keyring misconfiguration.",
                        unit="reqps",
                        w=12,
                        stack=True,
                    ),
                ],
            ),
            row(
                "Latency",
                [
                    timeseries(
                        "Time to first token",
                        [
                            prom(quantile(q, "vllm:time_to_first_token_seconds", ENGINE), label)
                            for q, label in ((P50, "p50"), (P95, "p95"), (P99, "p99"))
                        ],
                        description="Across the selected engines. Includes queue time and prefill.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Inter-token latency",
                        [
                            prom(quantile(q, "vllm:inter_token_latency_seconds", ENGINE), label)
                            for q, label in ((P50, "p50"), (P95, "p95"))
                        ],
                        description="Gap between consecutive output tokens. Rises with batch "
                        "size; a step up means the engines are saturated.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "End-to-end request latency",
                        [
                            prom(quantile(q, "vllm:e2e_request_latency_seconds", ENGINE), label)
                            for q, label in ((P50, "p50"), (P95, "p95"), (P99, "p99"))
                        ],
                        description="Arrival to last token. Dominated by output length, so "
                        "compare against tokens per request before reading it as slowness.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Queue time p95",
                        [
                            prom(
                                quantile(P95, "vllm:request_queue_time_seconds", ENGINE, "engine"),
                                "{{engine}}",
                            )
                        ],
                        description="Time in the engine's waiting queue before the first "
                        "scheduler step.",
                        unit="s",
                        w=6,
                    ),
                    timeseries(
                        "Prefill time p95",
                        [
                            prom(
                                quantile(
                                    P95, "vllm:request_prefill_time_seconds", ENGINE, "engine"
                                ),
                                "{{engine}}",
                            )
                        ],
                        description="Time spent computing the prompt. Long prompts that miss "
                        "the prefix cache land here.",
                        unit="s",
                        w=6,
                    ),
                    timeseries(
                        "Decode time p95",
                        [
                            prom(
                                quantile(P95, "vllm:request_decode_time_seconds", ENGINE, "engine"),
                                "{{engine}}",
                            )
                        ],
                        description="Time spent generating output tokens.",
                        unit="s",
                        w=6,
                    ),
                    timeseries(
                        "Sidecar duration p95 by route",
                        [
                            prom(
                                quantile(
                                    P95,
                                    "flakegraph_sidecar_request_duration_seconds",
                                    f"{ENGINE}, {CLASS}",
                                    "route",
                                ),
                                "{{route}}",
                            )
                        ],
                        description="Receive to last byte, as the sidecar sees it, per API "
                        "route. Streaming routes include the whole stream.",
                        unit="s",
                        w=6,
                    ),
                ],
            ),
            row(
                "Tokens",
                [
                    timeseries(
                        "Prompt tokens/s",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:prompt_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Prompt tokens processed per second, per engine.",
                        unit="suffix: tok/s",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Generation tokens/s",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:generation_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Output tokens generated per second, per engine.",
                        unit="suffix: tok/s",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Tokens/s per running request",
                        [
                            prom(
                                "sum(rate(vllm:generation_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval])) / "
                                f"(sum(vllm:num_requests_running{{{ENGINE}}}) > 0)",
                                "per request",
                            )
                        ],
                        description="Generation throughput divided by concurrency: the speed "
                        "one caller experiences. Falls as the batch grows.",
                        unit="suffix: tok/s",
                        w=8,
                    ),
                    timeseries(
                        "Prompt tokens per request",
                        [
                            prom(quantile(q, "vllm:request_prompt_tokens", ENGINE), label)
                            for q, label in ((P50, "p50"), (P95, "p95"))
                        ],
                        description="Prompt length distribution of finished requests.",
                        w=12,
                    ),
                    timeseries(
                        "Generation tokens per request",
                        [
                            prom(quantile(q, "vllm:request_generation_tokens", ENGINE), label)
                            for q, label in ((P50, "p50"), (P95, "p95"))
                        ],
                        description="Output length distribution of finished requests.",
                        w=12,
                    ),
                ],
            ),
            row(
                "Cache & speculation",
                [
                    timeseries(
                        "Prefix-cache hit ratio",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:prefix_cache_hits_total"
                                f"{{{ENGINE}}}[$__rate_interval])) / "
                                "sum by (engine) (rate(vllm:prefix_cache_queries_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Share of prompt tokens served from the local prefix cache. "
                        "Prefix-aware routing should keep this high for repeated system prompts.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                    ),
                    timeseries(
                        "External prefix-cache hit ratio",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:external_prefix_cache_hits_total"
                                f"{{{ENGINE}}}[$__rate_interval])) / "
                                "sum by (engine) (rate(vllm:external_prefix_cache_queries_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Hits against a KV connector's external cache. Absent or "
                        "zero unless KV offloading or transfer is configured.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                    ),
                    timeseries(
                        "Prompt tokens by source",
                        [
                            prom(
                                "sum by (source) (rate(vllm:prompt_tokens_by_source_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{source}}",
                            )
                        ],
                        description="Where prompt tokens came from: computed locally, hit in the "
                        "local cache, or transferred from another engine's KV.",
                        unit="suffix: tok/s",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Speculative decoding acceptance rate",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:spec_decode_num_accepted_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval])) / "
                                "sum by (engine) (rate(vllm:spec_decode_num_draft_tokens_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Draft tokens the target model accepted. Zero or absent when "
                        "speculation is off; well below the draft model's usual rate means the "
                        "draft and target have drifted apart.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                    ),
                    timeseries(
                        "Accepted draft tokens by position",
                        [
                            prom(
                                "sum by (position) (rate("
                                "vllm:spec_decode_num_accepted_tokens_per_pos_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "position {{position}}",
                            )
                        ],
                        description="Acceptance falls off with draft position; a steep drop "
                        "means the speculative length is set too high.",
                        unit="suffix: tok/s",
                        w=8,
                    ),
                    timeseries(
                        "Multimodal cache hit ratio",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:mm_cache_hits_total"
                                f"{{{ENGINE}}}[$__rate_interval])) / "
                                "sum by (engine) (rate(vllm:mm_cache_queries_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Cache hits for encoded images and other non-text inputs.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                    ),
                ],
                collapsed=True,
            ),
            row(
                "Outcomes",
                [
                    timeseries(
                        "Finished requests/s by reason",
                        [
                            prom(
                                "sum by (finished_reason) (rate(vllm:request_success_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{finished_reason}}",
                            )
                        ],
                        description="`stop` is a natural end, `length` hit max_tokens, `abort` "
                        "is the caller hanging up, `error` and `repetition` are the engine "
                        "giving up.",
                        unit="reqps",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Sidecar responses/s by class and status",
                        [
                            prom(
                                "sum by (consumer_class, class) (label_replace(rate("
                                f"flakegraph_sidecar_requests_total{{{ENGINE}, {CLASS}}}"
                                '[$__rate_interval]), "class", "$1xx", "status", '
                                '"([0-9])[0-9][0-9]"))',
                                "{{consumer_class}} {{class}}",
                            )
                        ],
                        description="HTTP status the caller received, grouped to its hundred. "
                        "4xx from `unauthenticated` is expected; 5xx from anyone is not.",
                        unit="reqps",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Preemptions/s",
                        [
                            prom(
                                "sum by (engine) (rate(vllm:num_preemptions_total"
                                f"{{{ENGINE}}}[$__rate_interval]))",
                                "{{engine}}",
                            )
                        ],
                        description="Must be zero. A preempted request loses its KV cache and "
                        "starts over, which is the eviction the priority design prevents.",
                        w=8,
                        steps=thresholds(GREEN, (0.001, RED)),
                    ),
                    timeseries(
                        "Engine sleep state",
                        [
                            prom(
                                "sum by (engine, sleep_state) "
                                f"(vllm:engine_sleep_state{{{ENGINE}}})",
                                "{{engine}} {{sleep_state}}",
                            )
                        ],
                        description="1 for the state each engine is in. `awake` is the only "
                        "state that serves; the others have offloaded or dropped weights.",
                        w=24,
                        h=6,
                        limits=(0, 1),
                    ),
                ],
            ),
            row(
                "Router (endpoint picker)",
                [
                    text(
                        "About these names",
                        "Metric names follow the llm-d-router **v0.10.0** metrics reference "
                        "(`docs/metrics.md`, `llm_d_epp_*` prefix). The picker's metrics port "
                        "authenticates every scrape, so they could not be checked against a "
                        "live exposition when this dashboard was written; an empty panel here "
                        "means the ServiceMonitor's token is not being accepted or a name "
                        "changed in a newer image.",
                        w=6,
                        h=8,
                    ),
                    stat(
                        "Ready endpoints",
                        [prom(f"max(llm_d_epp_ready_endpoints{{{NS}}})", instant=True)],
                        description="Engines the picker considers routable. Should equal the "
                        "engines-ready tile on the fleet overview.",
                        w=6,
                        h=4,
                        steps=thresholds(RED, (1, ORANGE), (4, GREEN)),
                    ),
                    stat(
                        "Pool KV-cache utilisation",
                        [
                            prom(
                                f"max(llm_d_epp_average_kv_cache_utilization{{{NS}}})",
                                instant=True,
                            )
                        ],
                        description="Mean KV usage across the pool as the picker sees it, "
                        "sampled from each engine's metrics.",
                        unit="percentunit",
                        w=6,
                        h=4,
                        steps=thresholds(GREEN, (0.7, ORANGE), (0.9, RED)),
                    ),
                    stat(
                        "Pool queue size",
                        [prom(f"max(llm_d_epp_average_queue_size{{{NS}}})", instant=True)],
                        description="Mean waiting requests per engine as the picker sees it.",
                        w=6,
                        h=4,
                        steps=thresholds(GREEN, (1, ORANGE), (5, RED)),
                    ),
                    timeseries(
                        "Router requests/s by model",
                        [
                            prom(
                                "sum by (model_name) (rate(llm_d_epp_request_total"
                                f"{{{NS}}}[$__rate_interval]))",
                                "{{model_name}}",
                            )
                        ],
                        description="Requests the picker scheduled, by requested model name.",
                        unit="reqps",
                        w=6,
                        h=4,
                    ),
                    timeseries(
                        "Router errors/s by code",
                        [
                            prom(
                                "sum by (error_code) (rate(llm_d_epp_request_error_total"
                                f"{{{NS}}}[$__rate_interval]))",
                                "{{error_code}}",
                            )
                        ],
                        description="Requests the picker could not place or that failed "
                        "upstream, by error code.",
                        unit="reqps",
                        w=6,
                        h=4,
                    ),
                    timeseries(
                        "Per-endpoint queue size",
                        [
                            prom(
                                f"llm_d_epp_per_endpoint_queue_size{{{NS}}}",
                                "{{model_server_endpoint}}",
                            )
                        ],
                        description="Waiting requests per engine as sampled by the picker. "
                        "Uneven queues mean the scorers are not balancing.",
                        w=8,
                    ),
                    timeseries(
                        "Router request duration",
                        [
                            prom(quantile(q, "llm_d_epp_request_duration_seconds", NS), label)
                            for q, label in ((P50, "p50"), (P95, "p95"), (P99, "p99"))
                        ],
                        description="End-to-end latency through the router, including the "
                        "engine's time.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Scheduling and plugin time p95",
                        [
                            prom(
                                quantile(P95, "llm_d_epp_scheduler_e2e_duration_seconds", NS),
                                "scheduler e2e",
                            ),
                            prom(
                                quantile(
                                    P95, "llm_d_epp_plugin_duration_seconds", NS, "plugin_name"
                                ),
                                "{{plugin_name}}",
                            ),
                        ],
                        description="Time the picker itself spends choosing an engine, in total "
                        "and per plugin. Milliseconds; anything more is the picker, not the "
                        "model, adding latency.",
                        unit="s",
                        w=8,
                    ),
                ],
            ),
        ],
    )


# Postgres fragments shared by the gateway's history panels. The key alias
# comes from the token row when the key still exists and from the request's
# metadata otherwise, so a deleted key keeps its history under its name.
_SPEND_ALIAS = (
    "COALESCE(NULLIF(t.key_alias, ''), NULLIF(s.metadata->>'user_api_key_alias', ''), 'unknown')"
)
_SPEND_FROM = (
    'FROM litellm."LiteLLM_SpendLogs" AS s\n'
    'LEFT JOIN litellm."LiteLLM_VerificationToken" AS t ON t.token = s.api_key\n'
    'WHERE $__timeFilter(s."startTime")'
)


def gateway() -> dict[str, Any]:
    """LiteLLM: who is calling, how much, and how it is going for them."""

    return dashboard(
        "flakegraph-gateway",
        "Gateway & Consumers",
        description="LiteLLM traffic per virtual key, live from Prometheus and historically "
        "from its spend log in Postgres.",
        variables=[namespace_variable()],
        rows=[
            row(
                "Live (Prometheus)",
                [
                    timeseries(
                        "Requests/s by key alias",
                        [
                            prom(
                                "sum by (api_key_alias) (rate("
                                f"litellm_proxy_total_requests_metric_total{{{NS}}}"
                                "[$__rate_interval]))",
                                "{{api_key_alias}}",
                            )
                        ],
                        description="Requests reaching the proxy, by virtual-key alias.",
                        unit="reqps",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Failures/s by key alias and status",
                        [
                            prom(
                                "sum by (api_key_alias, exception_status) (rate("
                                f"litellm_proxy_failed_requests_metric_total{{{NS}}}"
                                "[$__rate_interval]))",
                                "{{api_key_alias}} {{exception_status}}",
                            )
                        ],
                        description="Requests the caller did not get a success for, with the "
                        "status LiteLLM returned. 429 is a key over its limit; 5xx is upstream.",
                        unit="reqps",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Deployment state",
                        [
                            prom(
                                f"litellm_deployment_state{{{NS}}}",
                                "{{litellm_model_name}} @ {{api_base}}",
                            )
                        ],
                        description="0 healthy, 1 partial outage, 2 complete outage, per "
                        "configured upstream deployment.",
                        w=8,
                        limits=(0, 2),
                        steps=thresholds(GREEN, (1, ORANGE), (2, RED)),
                    ),
                    timeseries(
                        "Total latency p95 by model",
                        [
                            prom(
                                quantile(P95, "litellm_request_total_latency_metric", NS, "model"),
                                "{{model}}",
                            )
                        ],
                        description="Caller-observed latency through the proxy.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "LLM API latency p95 by model",
                        [
                            prom(
                                quantile(P95, "litellm_llm_api_latency_metric", NS, "model"),
                                "{{model}}",
                            )
                        ],
                        description="Time LiteLLM waited on the router. The gap to total "
                        "latency is the proxy's own overhead.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Time to first token p95 by model",
                        [
                            prom(
                                quantile(
                                    P95,
                                    "litellm_llm_api_time_to_first_token_metric",
                                    NS,
                                    "model",
                                ),
                                "{{model}}",
                            )
                        ],
                        description="Streaming requests only.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Input tokens/s by key alias",
                        [
                            prom(
                                "sum by (api_key_alias) (rate("
                                f"litellm_input_tokens_metric_total{{{NS}}}[$__rate_interval]))",
                                "{{api_key_alias}}",
                            )
                        ],
                        description="Prompt tokens per second, by virtual-key alias.",
                        unit="suffix: tok/s",
                        w=12,
                        stack=True,
                    ),
                    timeseries(
                        "Output tokens/s by key alias",
                        [
                            prom(
                                "sum by (api_key_alias) (rate("
                                f"litellm_output_tokens_metric_total{{{NS}}}[$__rate_interval]))",
                                "{{api_key_alias}}",
                            )
                        ],
                        description="Completion tokens per second, by virtual-key alias.",
                        unit="suffix: tok/s",
                        w=12,
                        stack=True,
                    ),
                ],
            ),
            row(
                "History (Postgres spend log)",
                [
                    timeseries(
                        "Tokens by key alias",
                        [
                            sql(
                                f'SELECT $__timeGroupAlias(s."startTime", $__interval),\n'
                                f"       {_SPEND_ALIAS} AS metric,\n"
                                f"       sum(s.total_tokens) AS value\n"
                                f"{_SPEND_FROM}\n"
                                f"GROUP BY 1, 2\nORDER BY 1"
                            )
                        ],
                        description="Prompt plus completion tokens per interval, summed from "
                        "LiteLLM's spend log. Survives proxy restarts, unlike the counters "
                        "above.",
                        w=12,
                        bars=True,
                        fill=60,
                        stack=True,
                    ),
                    timeseries(
                        "Requests by key alias",
                        [
                            sql(
                                f'SELECT $__timeGroupAlias(s."startTime", $__interval),\n'
                                f"       {_SPEND_ALIAS} AS metric,\n"
                                f"       count(*) AS value\n"
                                f"{_SPEND_FROM}\n"
                                f"GROUP BY 1, 2\nORDER BY 1"
                            )
                        ],
                        description="Logged requests per interval, by virtual-key alias.",
                        w=12,
                        bars=True,
                        fill=60,
                        stack=True,
                    ),
                    table(
                        "Top consumers in range",
                        [
                            sql(
                                f"SELECT {_SPEND_ALIAS} AS key_alias,\n"
                                "       count(*) AS requests,\n"
                                "       count(*) FILTER (WHERE s.status = 'failure') AS failures,\n"
                                "       sum(s.prompt_tokens) AS prompt_tokens,\n"
                                "       sum(s.completion_tokens) AS completion_tokens,\n"
                                "       sum(s.total_tokens) AS total_tokens,\n"
                                "       percentile_cont(0.95) WITHIN GROUP (ORDER BY\n"
                                '           extract(epoch FROM s."endTime" - s."startTime"))\n'
                                "           AS p95_latency_seconds\n"
                                f"{_SPEND_FROM}\n"
                                "GROUP BY 1\nORDER BY total_tokens DESC\nLIMIT 25",
                                table=True,
                            )
                        ],
                        description="Who used the gateway most over the selected range. Latency "
                        "is start to end of each logged request.",
                        w=14,
                        h=10,
                        sort_by="total_tokens",
                        overrides=[unit_override("p95_latency_seconds", "s")],
                    ),
                    timeseries(
                        "Requests by model",
                        [
                            sql(
                                f'SELECT $__timeGroupAlias(s."startTime", $__interval),\n'
                                "       COALESCE(NULLIF(s.model_group, ''), NULLIF(s.model, ''), "
                                "'unknown') AS metric,\n"
                                "       count(*) AS value\n"
                                f"{_SPEND_FROM}\n"
                                "GROUP BY 1, 2\nORDER BY 1"
                            )
                        ],
                        description="By the public model name the caller asked for "
                        "(`model_group`), falling back to the upstream model.",
                        w=10,
                        h=10,
                        bars=True,
                        fill=60,
                        stack=True,
                    ),
                    timeseries(
                        "Daily tokens",
                        [
                            sql(
                                "SELECT $__timeGroupAlias(s.\"startTime\", '1d'),\n"
                                "       'prompt' AS metric, sum(s.prompt_tokens) AS value\n"
                                f"{_SPEND_FROM}\nGROUP BY 1\n"
                                "UNION ALL\n"
                                "SELECT $__timeGroupAlias(s.\"startTime\", '1d'),\n"
                                "       'completion' AS metric, sum(s.completion_tokens) AS value\n"
                                f"{_SPEND_FROM}\nGROUP BY 1\n"
                                "ORDER BY 1"
                            )
                        ],
                        description="Prompt and completion tokens per day. Widen the time range "
                        "to see a month.",
                        w=24,
                        h=7,
                        bars=True,
                        fill=60,
                        stack=True,
                    ),
                ],
            ),
        ],
    )


def documents() -> dict[str, Any]:
    """The OCR shim's admission queue and the MinerU replicas behind it."""

    return dashboard(
        "flakegraph-documents",
        "Document Parsing (OCR)",
        description="The OCR shim's shared queue, the parsing replicas it feeds and their cost.",
        variables=[
            namespace_variable(),
            consumer_class_variable("flakegraph_ocr_requests_total"),
        ],
        rows=[
            row(
                "Queue",
                [
                    timeseries(
                        "Queue depth by class and status",
                        [
                            prom(
                                "max by (consumer_class, status) (flakegraph_ocr_queue_depth"
                                f"{{{NS}, {CLASS}}})",
                                "{{consumer_class}} {{status}}",
                            )
                        ],
                        description="`waiting` rows are callers blocked on a replica; "
                        "`dispatched` rows are parses under way. Every shim replica reports "
                        "the same shared queue, so this takes the max across them.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Oldest waiting request",
                        [
                            prom(
                                "max by (consumer_class) (flakegraph_ocr_oldest_waiting_seconds"
                                f"{{{NS}, {CLASS}}})",
                                "{{consumer_class}}",
                            )
                        ],
                        description="Age of the longest-waiting request per class. Interactive "
                        "work should never wait long; the alert fires on the configured bound.",
                        unit="s",
                        w=8,
                        steps=thresholds(GREEN, (60, ORANGE), (300, RED)),
                    ),
                    timeseries(
                        "Queue wait by class",
                        [
                            prom(
                                quantile(
                                    P50,
                                    "flakegraph_ocr_queue_wait_seconds",
                                    f"{NS}, {CLASS}",
                                    "consumer_class",
                                ),
                                "{{consumer_class}} p50",
                            ),
                            prom(
                                quantile(
                                    P95,
                                    "flakegraph_ocr_queue_wait_seconds",
                                    f"{NS}, {CLASS}",
                                    "consumer_class",
                                ),
                                "{{consumer_class}} p95",
                            ),
                        ],
                        description="Queued to dispatched, for requests that were dispatched.",
                        unit="s",
                        w=8,
                    ),
                ],
            ),
            row(
                "Replicas",
                [
                    stat(
                        "Upstream replicas",
                        [prom(f"max(flakegraph_ocr_upstream_replicas{{{NS}}})", instant=True)],
                        description="MinerU replicas the shim currently resolves. Below the "
                        "StatefulSet's size means a replica is not ready.",
                        w=4,
                        h=8,
                        steps=thresholds(RED, (1, ORANGE), (4, GREEN)),
                    ),
                    bargauge(
                        "Replica saturation",
                        [
                            prom(
                                f"sum by (replica) (flakegraph_ocr_in_flight{{{NS}}}) / "
                                f"sum by (replica) (flakegraph_ocr_replica_capacity{{{NS}}})",
                                "{{replica}}",
                            )
                        ],
                        description="In-flight parses over the concurrency the shim allows each "
                        "replica. Every bar full and a queue forming means more replicas, not "
                        "a bigger cap.",
                        w=8,
                    ),
                    timeseries(
                        "In flight per replica",
                        [
                            prom(
                                f"sum by (replica) (flakegraph_ocr_in_flight{{{NS}}})",
                                "{{replica}}",
                            )
                        ],
                        description="Parses dispatched to each replica, stacked.",
                        w=12,
                        stack=True,
                    ),
                    timeseries(
                        "Parse duration per replica",
                        [
                            prom(
                                quantile(
                                    P50, "flakegraph_ocr_parse_duration_seconds", NS, "replica"
                                ),
                                "{{replica}} p50",
                            ),
                            prom(
                                quantile(
                                    P95, "flakegraph_ocr_parse_duration_seconds", NS, "replica"
                                ),
                                "{{replica}} p95",
                            ),
                        ],
                        description="Dispatch to last byte of the parser's answer. One replica "
                        "consistently slower than the rest is sharing its node with something.",
                        unit="s",
                        w=8,
                    ),
                    timeseries(
                        "Parses/min by outcome",
                        [
                            prom(
                                "sum by (outcome) (rate(flakegraph_ocr_requests_total"
                                f"{{{NS}, {CLASS}}}[$__rate_interval])) * 60",
                                "{{outcome}}",
                            )
                        ],
                        description="How requests ended. `client_gone` is the caller giving up "
                        "while queued, `queue_error` is the shared queue's database failing.",
                        unit="cpm",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Error ratio",
                        [
                            prom(
                                "sum(rate(flakegraph_ocr_requests_total"
                                f'{{{NS}, {CLASS}, outcome!="succeeded"}}[$__rate_interval])) / '
                                f"sum(rate(flakegraph_ocr_requests_total{{{NS}, {CLASS}}}"
                                "[$__rate_interval]))",
                                "not succeeded",
                            ),
                            prom(
                                "sum(rate(flakegraph_ocr_requests_total"
                                f'{{{NS}, {CLASS}, outcome=~"upstream_.*"}}[$__rate_interval])) / '
                                f"sum(rate(flakegraph_ocr_requests_total{{{NS}, {CLASS}}}"
                                "[$__rate_interval]))",
                                "upstream failures",
                            ),
                        ],
                        description="Share of requests that did not succeed, and the share the "
                        "parser itself failed or timed out. Rejections and callers leaving "
                        "are in the first line only.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                        steps=thresholds(GREEN, (0.05, ORANGE), (0.2, RED)),
                    ),
                ],
            ),
            row(
                "Resources",
                [
                    timeseries(
                        "MinerU CPU",
                        [
                            prom(
                                "sum by (pod, node) (rate(container_cpu_usage_seconds_total"
                                f'{{{NS}, container="mineru"}}[$__rate_interval]))',
                                "{{pod}} ({{node}})",
                            )
                        ],
                        description="Cores busy per parsing replica. MinerU runs on CPU here.",
                        unit="short",
                        w=12,
                    ),
                    timeseries(
                        "MinerU memory working set",
                        [
                            prom(
                                "sum by (pod, node) (container_memory_working_set_bytes"
                                f'{{{NS}, container="mineru"}})',
                                "{{pod}} ({{node}})",
                            )
                        ],
                        description="Memory each parsing replica holds. This is charged against "
                        "the node's unified memory, alongside the engine on the same node.",
                        unit="bytes",
                        w=12,
                    ),
                ],
            ),
            row(
                "Waiting now (Postgres)",
                [
                    table(
                        "Waiting requests",
                        [
                            sql(
                                "SELECT consumer_class, priority, shim_owner,\n"
                                "       extract(epoch FROM now() - created_at) AS age_seconds,\n"
                                "       extract(epoch FROM now() - heartbeat_at) "
                                "AS since_heartbeat_seconds\n"
                                "FROM flakegraph_ocr_request\n"
                                "WHERE status = 'waiting'\n"
                                "ORDER BY priority, created_at\nLIMIT 100",
                                table=True,
                            )
                        ],
                        description="Live rows of the shared queue in dispatch order (lower "
                        "priority value first). A stale heartbeat means the owning shim "
                        "replica has gone; its rows are reaped on the next sweep.",
                        overrides=[
                            unit_override("age_seconds", "s"),
                            unit_override("since_heartbeat_seconds", "s"),
                        ],
                    )
                ],
            ),
        ],
    )


def _pool_of(expr: str) -> str:
    """Relabel a kube-state-metrics deployment series with its worker pool."""

    return f'label_replace({expr}, "pool", "$1", "deployment", ".*-({WORKER_POOLS})$")'


def pipeline() -> dict[str, Any]:
    """The task queue, the workers draining it and what they cost."""

    task_history_from = "FROM flakegraph_task\nWHERE $__timeFilter(completed_at)"
    return dashboard(
        "flakegraph-pipeline",
        "Pipeline Workers",
        description="Task queue and worker pools: live from the CloudNativePG exporter, "
        "history from the task table.",
        variables=[namespace_variable()],
        rows=[
            row(
                "Queue (live)",
                [
                    timeseries(
                        "Open tasks by stage",
                        [
                            prom(
                                "sum by (stage, status) (cnpg_flakegraph_tasks_count"
                                f'{{{NS}, status=~"queued|running"}})',
                                "{{stage}} {{status}}",
                            )
                        ],
                        description="Queued and running tasks per stage. Extraction stages "
                        "fan out per window, so they dominate the count.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Open tasks by priority band",
                        [
                            prom(
                                "sum by (priority_band, status) (cnpg_flakegraph_tasks_count"
                                f'{{{NS}, status=~"queued|running"}})',
                                "{{priority_band}} {{status}}",
                            )
                        ],
                        description="`interactive` is a run submitted with `--priority "
                        "interactive`; it is claimed ahead of `bulk` and scaled for separately.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Oldest queued task",
                        [
                            prom(
                                "max by (stage, priority_band) ("
                                f"cnpg_flakegraph_tasks_oldest_queued_seconds{{{NS}}})",
                                "{{stage}} {{priority_band}}",
                            )
                        ],
                        description="How long the oldest claimable task in each stage and band "
                        "has waited. Growing while pods are available means workers are "
                        "not claiming: check the fleet digest and lease-expired panels.",
                        unit="s",
                        w=8,
                        steps=thresholds(GREEN, (300, ORANGE), (900, RED)),
                    ),
                    timeseries(
                        "Worker demand vs available replicas",
                        [
                            prom(
                                "sum by (pool) (cnpg_flakegraph_worker_demand_desired_workers"
                                f"{{{NS}}})",
                                "{{pool}} demand",
                            ),
                            prom(
                                "sum by (pool) ("
                                + _pool_of(
                                    "kube_deployment_status_replicas_available"
                                    f'{{{NS}, deployment=~".*-({WORKER_POOLS})$"}}'
                                )
                                + ")",
                                "{{pool}} available",
                            ),
                        ],
                        description="Demand is what KEDA scales on (ready tasks, active leases "
                        "and pending publications); available is what the Deployment has. "
                        "Demand above available for long is KEDA capped at maxReplicas or "
                        "pods failing to schedule.",
                        w=12,
                    ),
                    timeseries(
                        "KEDA scaler metric values",
                        [
                            prom(
                                'keda_scaler_metrics_value{namespace="$namespace"}',
                                "{{scaledObject}} {{metric}}",
                            )
                        ],
                        description="The demand each trigger reported to KEDA. Only present "
                        "when the KEDA release exposes its operator metrics "
                        "(prometheus.operator.enabled), which the install script turns on.",
                        w=6,
                    ),
                    timeseries(
                        "KEDA scaler errors/s",
                        [
                            prom(
                                "sum by (scaledObject) (rate(keda_scaler_errors_total"
                                '{namespace="$namespace"}[$__rate_interval]))',
                                "{{scaledObject}}",
                            )
                        ],
                        description="Trigger queries failing, usually the database connection. "
                        "KEDA falls back to the configured replica count after three.",
                        unit="ops",
                        w=6,
                        steps=thresholds(GREEN, (0.001, RED)),
                    ),
                    stat(
                        "Leases expired",
                        [
                            prom(
                                "sum by (stage) (cnpg_flakegraph_tasks_lease_expired_count"
                                f"{{{NS}}})",
                                "{{stage}}",
                                instant=True,
                            )
                        ],
                        description="Running tasks whose worker stopped heartbeating and that "
                        "nobody has reclaimed yet. A worker died mid-task.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, RED)),
                    ),
                    stat(
                        "Near attempt limit",
                        [
                            prom(
                                "sum by (stage) ("
                                f"cnpg_flakegraph_tasks_near_attempt_limit_count{{{NS}}})",
                                "{{stage}}",
                                instant=True,
                            )
                        ],
                        description="Open tasks on their last permitted attempt: one more "
                        "failure and the run fails.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, ORANGE)),
                    ),
                    stat(
                        "Failed in the last hour",
                        [
                            prom(
                                "sum by (stage) (cnpg_flakegraph_tasks_failed_recent_count"
                                f"{{{NS}}})",
                                "{{stage}}",
                                instant=True,
                            )
                        ],
                        description="Tasks that reached failed status in the last hour, per stage.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, ORANGE), (10, RED)),
                    ),
                    stat(
                        "Runs and publications",
                        [
                            prom(
                                f"sum by (status) (cnpg_flakegraph_runs_count{{{NS}}})",
                                "runs {{status}}",
                                instant=True,
                            ),
                            prom(
                                f"sum by (status) (cnpg_flakegraph_publications_count{{{NS}}})",
                                "publications {{status}}",
                                instant=True,
                            ),
                        ],
                        description="Graph builds and the publications that write their "
                        "result, by status.",
                        w=6,
                        h=5,
                    ),
                ],
            ),
            row(
                "History (Postgres)",
                [
                    timeseries(
                        "Completions by stage",
                        [
                            sql(
                                "SELECT $__timeGroupAlias(completed_at, $__interval),\n"
                                "       stage AS metric, count(*) AS value\n"
                                f"{task_history_from} AND status = 'succeeded'\n"
                                "GROUP BY 1, 2\nORDER BY 1"
                            )
                        ],
                        description="Tasks that succeeded per interval, by stage.",
                        w=12,
                        bars=True,
                        fill=60,
                        stack=True,
                    ),
                    timeseries(
                        "Task duration by stage",
                        [
                            sql(
                                "SELECT $__timeGroupAlias(completed_at, $__interval),\n"
                                "       stage || ' p50' AS metric,\n"
                                "       percentile_cont(0.5) WITHIN GROUP (ORDER BY\n"
                                "           extract(epoch FROM completed_at - started_at))\n"
                                "           AS value\n"
                                f"{task_history_from} AND status = 'succeeded'\n"
                                "  AND started_at IS NOT NULL\n"
                                "GROUP BY 1, 2\nORDER BY 1"
                            ),
                            sql(
                                "SELECT $__timeGroupAlias(completed_at, $__interval),\n"
                                "       stage || ' p95' AS metric,\n"
                                "       percentile_cont(0.95) WITHIN GROUP (ORDER BY\n"
                                "           extract(epoch FROM completed_at - started_at))\n"
                                "           AS value\n"
                                f"{task_history_from} AND status = 'succeeded'\n"
                                "  AND started_at IS NOT NULL\n"
                                "GROUP BY 1, 2\nORDER BY 1"
                            ),
                        ],
                        description="Claim to completion of the successful attempt, p50 and "
                        "p95 per stage and interval.",
                        unit="s",
                        w=12,
                    ),
                    table(
                        "Failed tasks",
                        [
                            sql(
                                "SELECT updated_at, id, run_id, stage, attempts, max_attempts,\n"
                                "       last_error_json->>'error_type' AS error_type,\n"
                                "       left(last_error_json->>'error_message', 200) "
                                "AS error_message\n"
                                "FROM flakegraph_task\n"
                                "WHERE status = 'failed' AND $__timeFilter(updated_at)\n"
                                "ORDER BY updated_at DESC\nLIMIT 100",
                                table=True,
                            )
                        ],
                        description="Tasks that exhausted their attempts in the time range, "
                        "newest first, with the last error the worker recorded.",
                        w=24,
                        h=8,
                    ),
                    table(
                        "Runs",
                        [
                            sql(
                                "SELECT r.id, r.graph_id, r.status, r.created_at, r.updated_at,\n"
                                "       count(t.id) AS tasks,\n"
                                "       count(t.id) FILTER (WHERE t.status = 'succeeded') "
                                "AS succeeded,\n"
                                "       count(t.id) FILTER (WHERE t.status = 'running') "
                                "AS running,\n"
                                "       count(t.id) FILTER (WHERE t.status = 'queued') "
                                "AS queued,\n"
                                "       count(t.id) FILTER (WHERE t.status = 'failed') AS failed\n"
                                "FROM flakegraph_run AS r\n"
                                "LEFT JOIN flakegraph_task AS t ON t.run_id = r.id\n"
                                "WHERE $__timeFilter(r.updated_at)\n"
                                "GROUP BY r.id\nORDER BY r.updated_at DESC\nLIMIT 50",
                                table=True,
                            )
                        ],
                        description="Runs touched in the time range with their task counts.",
                        w=16,
                        h=9,
                    ),
                    bargauge(
                        "Attempts per task",
                        [
                            sql(
                                "SELECT attempts::text AS attempts, count(*) AS tasks\n"
                                "FROM flakegraph_task\n"
                                "WHERE $__timeFilter(updated_at)\n"
                                "GROUP BY attempts\nORDER BY attempts",
                                table=True,
                            )
                        ],
                        description="How many attempts tasks in the range needed. Anything "
                        "past 1 is a retry; a tail toward max_attempts is a flaky stage.",
                        unit="short",
                        w=8,
                        h=9,
                        limits=(0, None),
                        steps=thresholds(BLUE),
                        transformations=[
                            {
                                "id": "rowsToFields",
                                "options": {
                                    "mappings": [
                                        {"fieldName": "attempts", "handlerKey": "field.name"},
                                        {"fieldName": "tasks", "handlerKey": "field.value"},
                                    ]
                                },
                            }
                        ],
                    ),
                ],
            ),
            row(
                "Resources",
                [
                    timeseries(
                        "Worker CPU by pool",
                        [
                            prom(
                                "sum by (pool) (label_replace(rate("
                                "container_cpu_usage_seconds_total"
                                f'{{{NS}, container="worker"}}[$__rate_interval]), '
                                f'"pool", "$1", "pod", ".*-({WORKER_POOLS})-[^-]+-[^-]+$"))',
                                "{{pool}}",
                            )
                        ],
                        description="Cores busy across each pool's pods.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Worker memory by pool",
                        [
                            prom(
                                "sum by (pool) (label_replace("
                                "container_memory_working_set_bytes"
                                f'{{{NS}, container="worker"}}, '
                                f'"pool", "$1", "pod", ".*-({WORKER_POOLS})-[^-]+-[^-]+$"))',
                                "{{pool}}",
                            )
                        ],
                        description="Working set across each pool's pods. The extract pool "
                        "holds the largest model context and grows with window size.",
                        unit="bytes",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Worker container restarts",
                        [
                            prom(
                                "sum by (pod) (increase(kube_pod_container_status_restarts_total"
                                f'{{{NS}, container="worker"}}[1h])) > 0',
                                "{{pod}}",
                            )
                        ],
                        description="Restarts in the trailing hour, per pod. A worker OOM-killed "
                        "mid-task leaves a lease to expire and an attempt consumed.",
                        w=8,
                        bars=True,
                        fill=60,
                        steps=thresholds(GREEN, (1, RED)),
                    ),
                ],
            ),
        ],
    )


def platform() -> dict[str, Any]:
    """The database, the volumes and the cluster underneath everything."""

    db = f'{NS}, datname="flakegraph"'
    return dashboard(
        "flakegraph-platform",
        "Database & Storage",
        description="CloudNativePG, persistent volumes and the Kubernetes objects "
        "the release depends on.",
        variables=[namespace_variable(), node_variable()],
        rows=[
            row(
                "Database (CloudNativePG)",
                [
                    timeseries(
                        "Backends by state",
                        [
                            prom(
                                f"sum by (state) (cnpg_backends_total{{{NS}}})",
                                "{{state}}",
                            )
                        ],
                        description="Server processes by state. Many `idle` backends are the "
                        "workers' and the gateway's pools; `active` is real work.",
                        w=8,
                        stack=True,
                    ),
                    timeseries(
                        "Connections vs max_connections",
                        [
                            prom(f"sum(cnpg_backends_total{{{NS}}})", "connections"),
                            prom(
                                f'cnpg_pg_settings_setting{{{NS}, name="max_connections"}}',
                                "max_connections",
                            ),
                        ],
                        description="Every scaled-out worker, shim and gateway replica opens "
                        "its own pool; KEDA can bring dozens of workers up at once.",
                        w=8,
                    ),
                    timeseries(
                        "Database size",
                        [
                            prom(
                                f'cnpg_pg_database_size_bytes{{{NS}, datname!~"template.*"}}',
                                "{{datname}}",
                            )
                        ],
                        description="On-disk size per database. `flakegraph` holds the task "
                        "queue, the OCR queue and LiteLLM's spend log.",
                        unit="bytes",
                        w=8,
                    ),
                    timeseries(
                        "Transactions/s",
                        [
                            prom(
                                f"rate(cnpg_pg_stat_database_xact_commit{{{db}}}[$__rate_interval])",
                                "commit",
                            ),
                            prom(
                                "rate(cnpg_pg_stat_database_xact_rollback"
                                f"{{{db}}}[$__rate_interval])",
                                "rollback",
                            ),
                        ],
                        description="Commits and rollbacks per second in the application "
                        "database. Workers poll every second, so this never reaches zero.",
                        unit="ops",
                        w=8,
                    ),
                    timeseries(
                        "Buffer cache hit ratio",
                        [
                            prom(
                                f"rate(cnpg_pg_stat_database_blks_hit{{{db}}}[$__rate_interval])"
                                " / (rate(cnpg_pg_stat_database_blks_hit"
                                f"{{{db}}}[$__rate_interval]) + rate("
                                f"cnpg_pg_stat_database_blks_read{{{db}}}[$__rate_interval]))",
                                "hit ratio",
                            )
                        ],
                        description="Reads served from shared_buffers. Below 0.99 on a queue "
                        "this small means shared_buffers is undersized or a scan is running.",
                        unit="percentunit",
                        w=8,
                        limits=(0, 1),
                        steps=thresholds(RED, (0.9, ORANGE), (0.99, GREEN)),
                    ),
                    timeseries(
                        "Deadlocks and waiting backends",
                        [
                            prom(
                                "rate(cnpg_pg_stat_database_deadlocks"
                                f"{{{db}}}[$__rate_interval]) * 60",
                                "deadlocks/min",
                            ),
                            prom(f"cnpg_backends_waiting_total{{{NS}}}", "backends waiting"),
                            prom(
                                f"cnpg_backends_max_tx_duration_seconds{{{NS}}}",
                                "longest transaction",
                            ),
                        ],
                        description="Lock contention: backends blocked on a lock, deadlocks "
                        "resolved by abort, and the longest open transaction (seconds). "
                        "Task claiming uses SKIP LOCKED, so waits should be rare.",
                        w=8,
                        overrides=[unit_override("longest transaction", "s")],
                    ),
                    timeseries(
                        "Checkpoints",
                        [
                            prom(
                                "rate(cnpg_pg_stat_checkpointer_checkpoints_timed"
                                f"{{{NS}}}[$__rate_interval]) * 3600",
                                "timed/h",
                            ),
                            prom(
                                "rate(cnpg_pg_stat_checkpointer_checkpoints_req"
                                f"{{{NS}}}[$__rate_interval]) * 3600",
                                "requested/h",
                            ),
                        ],
                        description="Requested checkpoints outnumbering timed ones means WAL is "
                        "filling faster than checkpoint_timeout: raise max_wal_size.",
                        w=8,
                    ),
                    timeseries(
                        "WAL",
                        [
                            prom(
                                f'cnpg_collector_pg_wal{{{NS}, value="size"}}',
                                "segments on disk",
                            ),
                            prom(
                                f"rate(cnpg_collector_wal_bytes{{{NS}}}[$__rate_interval])",
                                "written/s",
                            ),
                        ],
                        description="WAL held in pg_wal and the rate it is written at.",
                        unit="bytes",
                        w=8,
                        overrides=[unit_override("written/s", "Bps")],
                    ),
                    stat(
                        "Replication",
                        [
                            prom(
                                f"cnpg_pg_replication_streaming_replicas{{{NS}}}",
                                "streaming replicas",
                                instant=True,
                            ),
                            prom(
                                f"cnpg_pg_replication_lag{{{NS}}}",
                                "lag",
                                instant=True,
                            ),
                            prom(
                                f"cnpg_collector_last_collection_error{{{NS}}}",
                                "collector error",
                                instant=True,
                            ),
                        ],
                        description="The cluster runs a single instance, so zero replicas and "
                        "zero lag is the expected reading; a collector error of 1 means the "
                        "custom queries (task and OCR gauges) are failing.",
                        w=8,
                        steps=thresholds(GREEN),
                    ),
                ],
            ),
            row(
                "Storage",
                [
                    bargauge(
                        "Persistent volume usage",
                        [
                            prom(
                                f"kubelet_volume_stats_used_bytes{{{NS}}} / "
                                f"kubelet_volume_stats_capacity_bytes{{{NS}}}",
                                "{{persistentvolumeclaim}}",
                            )
                        ],
                        description="Every PVC in the namespace: Postgres, Prometheus, "
                        "Alertmanager, Grafana, MinIO and the model caches. A full Postgres "
                        "volume stops the whole pipeline.",
                        w=12,
                        h=9,
                        steps=thresholds(GREEN, (0.75, ORANGE), (0.9, RED)),
                    ),
                    timeseries(
                        "Persistent volume used",
                        [
                            prom(
                                f"kubelet_volume_stats_used_bytes{{{NS}}}",
                                "{{persistentvolumeclaim}}",
                            )
                        ],
                        description="Bytes used per PVC over time; the slope says when it fills.",
                        unit="bytes",
                        w=12,
                        h=9,
                    ),
                    timeseries(
                        "Node root disk used",
                        [
                            prom(
                                _node_join(
                                    "1 - node_filesystem_avail_bytes"
                                    '{mountpoint="/", fstype!="rootfs"} / '
                                    'node_filesystem_size_bytes{mountpoint="/", fstype!="rootfs"}'
                                ),
                                "{{nodename}}",
                            )
                        ],
                        description="local-path volumes and container images share the root "
                        "filesystem, so this is the real ceiling for every PVC above.",
                        unit="percentunit",
                        w=16,
                        h=7,
                        limits=(0, 1),
                        steps=thresholds(GREEN, (0.8, ORANGE), (0.9, RED)),
                    ),
                    text(
                        "Object storage (MinIO)",
                        "MinIO's `/minio/v2/metrics/cluster` endpoint needs a bearer token "
                        "minted from the root credential, which nothing in the chart holds, "
                        "so it is **not scraped**. Its volume appears in the PVC panels; "
                        "bucket sizes and request rates are in the MinIO console.",
                        w=8,
                        h=7,
                    ),
                ],
            ),
            row(
                "Cluster",
                [
                    stat(
                        "Pods not ready",
                        [
                            prom(
                                f'count(kube_pod_status_ready{{{NS}, condition="true"}} == 0 '
                                f'and on (pod) kube_pod_status_phase{{{NS}, phase="Running"}} '
                                "== 1)",
                                instant=True,
                            )
                        ],
                        description="Running pods in the namespace that fail readiness. "
                        "Completed job pods are excluded.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, RED)),
                    ),
                    stat(
                        "Pods pending or failed",
                        [
                            prom(
                                "sum(kube_pod_status_phase"
                                f'{{{NS}, phase=~"Pending|Failed|Unknown"}})',
                                instant=True,
                            )
                        ],
                        description="Pods that could not be scheduled or crashed out. Pending "
                        "on this fleet is usually memory: the engines reserve most of it.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, ORANGE)),
                    ),
                    stat(
                        "Nodes ready",
                        [
                            prom(
                                'sum(kube_node_status_condition{condition="Ready", status="true"})',
                                "ready",
                                instant=True,
                            ),
                            prom(
                                "sum(kube_node_status_condition{condition=~"
                                '"MemoryPressure|DiskPressure|PIDPressure", status="true"})',
                                "under pressure",
                                instant=True,
                            ),
                        ],
                        description="Nodes reporting Ready, and nodes reporting any pressure "
                        "condition (which evicts pods).",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN),
                    ),
                    stat(
                        "Container restarts, 1h",
                        [
                            prom(
                                f"sum(increase(kube_pod_container_status_restarts_total{{{NS}}}"
                                "[1h]))",
                                instant=True,
                            )
                        ],
                        description="Restarts across every container in the namespace over "
                        "the trailing hour.",
                        w=6,
                        h=5,
                        steps=thresholds(GREEN, (1, ORANGE), (5, RED)),
                        decimals=0,
                    ),
                    timeseries(
                        "Container restarts by pod",
                        [
                            prom(
                                "sum by (pod) (increase(kube_pod_container_status_restarts_total"
                                f"{{{NS}}}[1h])) > 0",
                                "{{pod}}",
                            )
                        ],
                        description="Which pods are restarting. Only pods with at least one "
                        "restart in the trailing hour are drawn.",
                        w=12,
                        bars=True,
                        fill=60,
                    ),
                    timeseries(
                        "Pods not ready over time",
                        [
                            prom(
                                f'sum by (pod) (kube_pod_status_ready{{{NS}, condition="false"}} '
                                f'and on (pod) kube_pod_status_phase{{{NS}, phase="Running"}} '
                                "== 1)",
                                "{{pod}}",
                            )
                        ],
                        description="1 while a running pod fails readiness. Long bars on an "
                        "engine are weight loading; on anything else they are a problem.",
                        w=12,
                        limits=(0, 1),
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

BUILDERS = (fleet_overview, llm_serving, gateway, documents, pipeline, platform)


def build_all() -> dict[str, dict[str, Any]]:
    """Return every dashboard keyed by its output file name."""

    dashboards = {f"{built['uid']}.json": built for built in (build() for build in BUILDERS)}
    if len(dashboards) != len(BUILDERS):
        raise ValueError("dashboard uids must be unique")
    return dashboards


def render(built: dict[str, Any]) -> str:
    """Serialise a dashboard the way the generator writes it, for comparison."""

    return json.dumps(built, indent=2, sort_keys=True) + "\n"


def write_all(output_dir: Path) -> Iterable[Path]:
    """Write every dashboard into ``output_dir`` and yield the paths written."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, built in build_all().items():
        path = output_dir / name
        path.write_text(render(built), encoding="utf-8")
        yield path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the dashboards, or with ``--check`` report whether they are current."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the files on disk differ from what would be written",
    )
    args = parser.parse_args(argv)
    if args.check:
        stale = [
            name
            for name, built in build_all().items()
            if not (args.output_dir / name).is_file()
            or (args.output_dir / name).read_text(encoding="utf-8") != render(built)
        ]
        for name in stale:
            print(f"stale: {args.output_dir / name}", file=sys.stderr)
        return 1 if stale else 0
    for path in write_all(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
