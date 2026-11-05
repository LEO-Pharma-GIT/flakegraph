import type { ClusterSnapshot, NodeStatus, WorkloadStatus } from "@/server/protocol/schema";

/** Rows shown before "Show more" - a namespace holds hundreds of pods, a screen does not. */
export const FLEET_PAGE_SIZE = 50;

/** Past this many nodes a grid of cards runs off the screen; a table of rows does not. */
export const NODE_TABLE_THRESHOLD = 8;

/** Leased tasks listed on a node before "N more"; a busy node holds dozens. */
export const NODE_ASSIGNMENT_LIMIT = 20;

/**
 * The plane a pod belongs to, read from its component label. Helm stamps
 * `app.kubernetes.io/component` per template (model-serving, inference-router,
 * document-parsing, ocr-shim, worker-<pool>, gateway, auth-proxy, control-plane,
 * monitoring, database-bootstrap); the filter groups these into the handful of
 * families an operator thinks in, and the row still shows the raw label.
 */
export type WorkloadFamily =
  | "model-serving"
  | "document-parsing"
  | "workers"
  | "gateway"
  | "control-plane"
  | "monitoring"
  | "other";

export const WORKLOAD_FAMILIES: ReadonlyArray<{ id: WorkloadFamily; label: string }> = [
  { id: "model-serving", label: "Model serving" },
  { id: "document-parsing", label: "Document parsing" },
  { id: "workers", label: "Workers" },
  { id: "gateway", label: "Gateway" },
  { id: "control-plane", label: "Control plane" },
  { id: "monitoring", label: "Monitoring" },
  { id: "other", label: "Other" },
];

export function workloadFamily(component: string): WorkloadFamily {
  const normalized = component.toLowerCase();
  if (normalized.startsWith("worker")) {
    return "workers";
  }
  if (
    normalized === "model-serving" ||
    normalized === "inference-router" ||
    normalized.includes("vllm") ||
    normalized.includes("model")
  ) {
    return "model-serving";
  }
  if (normalized.includes("parsing") || normalized.includes("ocr") || normalized.includes("mineru")) {
    return "document-parsing";
  }
  if (normalized === "gateway" || normalized === "auth-proxy" || normalized.includes("ingress")) {
    return "gateway";
  }
  if (normalized === "control-plane" || normalized === "database-bootstrap" || normalized === "console") {
    return "control-plane";
  }
  if (normalized === "monitoring") {
    return "monitoring";
  }
  return "other";
}

/** A pod's Kubernetes phase folded to the five kinds the console reasons about. */
export type PhaseKind = "failed" | "pending" | "unknown" | "running" | "succeeded";

/** Kinds in the order a reader wants them: trouble first, finished work last. */
export const PHASE_KINDS: ReadonlyArray<{ id: PhaseKind; label: string }> = [
  { id: "failed", label: "Failed" },
  { id: "pending", label: "Pending" },
  { id: "unknown", label: "Unknown" },
  { id: "running", label: "Running" },
  { id: "succeeded", label: "Succeeded" },
];

export function phaseKind(phase: string): PhaseKind {
  const normalized = phase.trim().toLowerCase();
  if (normalized === "failed" || normalized === "error") {
    return "failed";
  }
  if (normalized === "pending" || normalized === "queued") {
    return "pending";
  }
  if (normalized === "running") {
    return "running";
  }
  if (normalized === "succeeded" || normalized === "completed") {
    return "succeeded";
  }
  return "unknown";
}

/**
 * Sort key with problems first: failed, pending, running but not ready, unknown,
 * then healthy running pods, then finished ones. Within a rank rows sort by
 * name so the order is stable between refreshes.
 */
export function workloadRank(workload: Pick<WorkloadStatus, "phase" | "ready">): number {
  switch (phaseKind(workload.phase)) {
    case "failed":
      return 0;
    case "pending":
      return 1;
    case "running":
      return workload.ready ? 4 : 2;
    case "unknown":
      return 3;
    case "succeeded":
      return 5;
  }
}

export function compareWorkloads(left: WorkloadStatus, right: WorkloadStatus): number {
  return workloadRank(left) - workloadRank(right) || left.name.localeCompare(right.name);
}

/** A pod that wants a person's attention: it is not running, or runs without being ready. */
export function workloadNeedsAttention(workload: Pick<WorkloadStatus, "phase" | "ready">): boolean {
  return workloadRank(workload) <= 3;
}

export type PhaseFilter = PhaseKind | "active" | "all";
export type FamilyFilter = WorkloadFamily | "all";

export type WorkloadFilter = {
  search: string;
  family: FamilyFilter;
  phase: PhaseFilter;
  node: string | null;
};

/**
 * The phase filter's default. A Succeeded pod is a finished one-off Job
 * (a benchmark export, a probe, a bootstrap) that holds no GPU or CPU and
 * needs no decision; it lingers only until its Job's TTL reaps it. Showing
 * those beside the serving plane by default buries the pods that matter, so
 * the list opens on everything still active and the phase filter says how
 * many finished pods it is hiding.
 */
export const DEFAULT_WORKLOAD_FILTER: WorkloadFilter = {
  search: "",
  family: "all",
  phase: "active",
  node: null,
};

export function matchesPhaseFilter(workload: Pick<WorkloadStatus, "phase">, phase: PhaseFilter): boolean {
  if (phase === "all") {
    return true;
  }
  const kind = phaseKind(workload.phase);
  if (phase === "active") {
    return kind !== "succeeded";
  }
  return kind === phase;
}

/** The rows a filter keeps, problems first. Search matches the pod name or its node. */
export function filterWorkloads(workloads: readonly WorkloadStatus[], filter: WorkloadFilter): WorkloadStatus[] {
  const needle = filter.search.trim().toLowerCase();
  return workloads
    .filter((workload) => {
      if (!matchesPhaseFilter(workload, filter.phase)) {
        return false;
      }
      if (filter.family !== "all" && workloadFamily(workload.component) !== filter.family) {
        return false;
      }
      if (filter.node && workload.node !== filter.node) {
        return false;
      }
      return !needle || `${workload.name} ${workload.node ?? ""}`.toLowerCase().includes(needle);
    })
    .sort(compareWorkloads);
}

export function familyCounts(workloads: readonly WorkloadStatus[]): Array<{ id: WorkloadFamily; label: string; count: number }> {
  const counts = new Map<WorkloadFamily, number>();
  for (const workload of workloads) {
    const family = workloadFamily(workload.component);
    counts.set(family, (counts.get(family) ?? 0) + 1);
  }
  return WORKLOAD_FAMILIES.flatMap((family) => {
    const count = counts.get(family.id) ?? 0;
    return count > 0 ? [{ ...family, count }] : [];
  });
}

export function phaseCounts(workloads: readonly WorkloadStatus[]): Array<{ id: PhaseKind; label: string; count: number }> {
  const counts = new Map<PhaseKind, number>();
  for (const workload of workloads) {
    const kind = phaseKind(workload.phase);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  }
  return PHASE_KINDS.flatMap((kind) => {
    const count = counts.get(kind.id) ?? 0;
    return count > 0 ? [{ ...kind, count }] : [];
  });
}

/**
 * What the status column says about a pod. Kubernetes reports a finished Job's
 * pod as Succeeded with no ready container, which "not ready" would make look
 * like a fault; it has completed.
 */
export function workloadDetail(
  workload: Pick<WorkloadStatus, "phase" | "ready" | "restarts">,
  nodesReady: number,
  nodesTotal: number,
): string {
  const restarts = workload.restarts > 0 ? ` · ${workload.restarts} restart${workload.restarts === 1 ? "" : "s"}` : "";
  switch (phaseKind(workload.phase)) {
    case "succeeded":
      return "Completed";
    case "failed":
      return `Failed${restarts}`;
    case "pending":
      return nodesReady < nodesTotal
        ? `Waiting for node capacity (${nodesReady}/${nodesTotal} ready)`
        : "Waiting for GPU quota or image pull";
    case "running":
      return workload.ready ? (restarts ? `Ready${restarts}` : "Ready") : `Not ready${restarts}`;
    case "unknown":
      return `${workload.phase.trim() || "Unknown"} · not ready`;
  }
}

export type FleetSummary = {
  nodesReady: number;
  nodesTotal: number;
  gpus: number;
  modelServersReady: number;
  workersRunning: number;
  pending: number;
  failed: number;
  notReady: number;
  completed: number;
  active: number;
  total: number;
};

/** The counts at the top of the page: they stay four numbers however large the fleet grows. */
export function fleetSummary(snapshot: Pick<ClusterSnapshot, "nodes" | "workloads" | "nodesReady" | "nodesTotal">): FleetSummary {
  const summary: FleetSummary = {
    nodesReady: snapshot.nodesReady,
    nodesTotal: snapshot.nodesTotal,
    gpus: snapshot.nodes.reduce((sum, node) => sum + node.gpuCount, 0),
    modelServersReady: snapshot.nodes.filter((node) => node.modelServerReady).length,
    workersRunning: 0,
    pending: 0,
    failed: 0,
    notReady: 0,
    completed: 0,
    active: 0,
    total: snapshot.workloads.length,
  };
  for (const workload of snapshot.workloads) {
    const kind = phaseKind(workload.phase);
    if (kind === "succeeded") {
      summary.completed += 1;
      continue;
    }
    summary.active += 1;
    if (kind === "pending") {
      summary.pending += 1;
    } else if (kind === "failed") {
      summary.failed += 1;
    } else if (kind === "running") {
      if (!workload.ready) {
        summary.notReady += 1;
      }
      if (workloadFamily(workload.component) === "workers") {
        summary.workersRunning += 1;
      }
    }
  }
  return summary;
}

export type NodeSummary = {
  node: NodeStatus;
  /** Pods on the node that are still running or trying to; finished Jobs are not counted. */
  activeWorkloads: number;
  workersRunning: number;
  attention: number;
};

/** One row per node with the counts the card and the table both show. */
export function nodeSummaries(snapshot: Pick<ClusterSnapshot, "nodes" | "workloads">): NodeSummary[] {
  const byNode = new Map<string, WorkloadStatus[]>();
  for (const workload of snapshot.workloads) {
    if (!workload.node) {
      continue;
    }
    const list = byNode.get(workload.node) ?? [];
    list.push(workload);
    byNode.set(workload.node, list);
  }
  return snapshot.nodes.map((node) => {
    const onNode = byNode.get(node.name) ?? [];
    const active = onNode.filter((workload) => phaseKind(workload.phase) !== "succeeded");
    return {
      node,
      activeWorkloads: active.length,
      workersRunning: active.filter(
        (workload) => phaseKind(workload.phase) === "running" && workloadFamily(workload.component) === "workers",
      ).length,
      attention: active.filter(workloadNeedsAttention).length,
    };
  });
}
