import { describe, expect, it } from "vitest";
import {
  DEFAULT_WORKLOAD_FILTER,
  familyCounts,
  filterWorkloads,
  fleetSummary,
  nodeSummaries,
  phaseCounts,
  phaseKind,
  workloadDetail,
  workloadFamily,
} from "./fleet-workloads";
import type { NodeStatus, WorkloadStatus } from "@/server/protocol/schema";

function workload(name: string, component: string, overrides: Partial<WorkloadStatus> = {}): WorkloadStatus {
  return {
    name,
    component,
    phase: "Running",
    node: "gpu-a",
    ready: true,
    restarts: 0,
    cpu: null,
    memory: null,
    image: null,
    model: null,
    ...overrides,
  };
}

function node(name: string, overrides: Partial<NodeStatus> = {}): NodeStatus {
  return {
    name,
    ready: true,
    nodeClass: "gpu",
    gpuCount: 1,
    gpuModel: "NVIDIA A10",
    cpuCapacity: null,
    memoryCapacity: null,
    gpuPercent: null,
    cpuUsage: null,
    cpuPercent: null,
    memoryUsage: null,
    memoryPercent: null,
    workloadCount: 0,
    workerCount: 0,
    modelServerReady: false,
    model: null,
    modelImage: null,
    ...overrides,
  };
}

const fleet: WorkloadStatus[] = [
  workload("vllm-a", "model-serving", { model: "qwen" }),
  workload("router-0", "inference-router"),
  workload("mineru-0", "document-parsing", { node: "gpu-b" }),
  workload("ocr-shim-0", "ocr-shim", { node: "gpu-b", ready: false, restarts: 4 }),
  workload("gateway-0", "gateway"),
  workload("auth-proxy-0", "auth-proxy"),
  workload("console-0", "control-plane"),
  workload("grafana-0", "monitoring", { node: "gpu-b" }),
  workload("spark-exec-0", "workload", { node: "gpu-b" }),
  workload("worker-extract-00", "worker-extract"),
  workload("worker-prepare-00", "worker-prepare", { node: "gpu-b" }),
  workload("worker-extract-01", "worker-extract", { node: null, phase: "Pending", ready: false }),
  workload("worker-extract-oom", "worker-extract", { phase: "Failed", ready: false, restarts: 3 }),
  workload("bench-export", "workload", { phase: "Succeeded", ready: false }),
  workload("vllm28-probe", "model-serving", { phase: "Succeeded", ready: false, node: "gpu-b" }),
];

describe("workloadFamily", () => {
  it("folds Helm component labels into the families an operator thinks in", () => {
    expect(workloadFamily("model-serving")).toBe("model-serving");
    expect(workloadFamily("inference-router")).toBe("model-serving");
    expect(workloadFamily("model-server")).toBe("model-serving");
    expect(workloadFamily("document-parsing")).toBe("document-parsing");
    expect(workloadFamily("ocr-shim")).toBe("document-parsing");
    expect(workloadFamily("worker-extract")).toBe("workers");
    expect(workloadFamily("worker-prepare")).toBe("workers");
    expect(workloadFamily("gateway")).toBe("gateway");
    expect(workloadFamily("auth-proxy")).toBe("gateway");
    expect(workloadFamily("control-plane")).toBe("control-plane");
    expect(workloadFamily("database-bootstrap")).toBe("control-plane");
    expect(workloadFamily("monitoring")).toBe("monitoring");
    expect(workloadFamily("workload")).toBe("other");
  });
});

describe("phaseKind", () => {
  it("reads Kubernetes phases case-insensitively and treats the rest as unknown", () => {
    expect(phaseKind("Running")).toBe("running");
    expect(phaseKind("pending")).toBe("pending");
    expect(phaseKind("queued")).toBe("pending");
    expect(phaseKind("Succeeded")).toBe("succeeded");
    expect(phaseKind("Failed")).toBe("failed");
    expect(phaseKind("")).toBe("unknown");
    expect(phaseKind("Terminating")).toBe("unknown");
  });
});

describe("filterWorkloads", () => {
  it("hides finished Jobs by default and sorts problems first", () => {
    const shown = filterWorkloads(fleet, DEFAULT_WORKLOAD_FILTER);
    expect(shown).toHaveLength(13);
    expect(shown.map((item) => item.name).slice(0, 3)).toEqual(["worker-extract-oom", "worker-extract-01", "ocr-shim-0"]);
    expect(shown.some((item) => item.phase === "Succeeded")).toBe(false);
    // Healthy pods follow in name order, so the list is stable between refreshes.
    const healthy = shown.slice(3).map((item) => item.name);
    expect(healthy).toEqual([...healthy].sort((left, right) => left.localeCompare(right)));
  });

  it("shows only finished Jobs when asked, and everything under all", () => {
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, phase: "succeeded" }).map((item) => item.name)).toEqual([
      "bench-export",
      "vllm28-probe",
    ]);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, phase: "all" })).toHaveLength(fleet.length);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, phase: "failed" }).map((item) => item.name)).toEqual([
      "worker-extract-oom",
    ]);
  });

  it("narrows by component family, node and a name or node search", () => {
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, family: "workers" }).map((item) => item.name)).toEqual([
      "worker-extract-oom",
      "worker-extract-01",
      "worker-extract-00",
      "worker-prepare-00",
    ]);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, family: "gateway" })).toHaveLength(2);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, node: "gpu-b" }).map((item) => item.name)).toEqual([
      "ocr-shim-0",
      "grafana-0",
      "mineru-0",
      "spark-exec-0",
      "worker-prepare-00",
    ]);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, search: "VLLM" }).map((item) => item.name)).toEqual(["vllm-a"]);
    expect(filterWorkloads(fleet, { ...DEFAULT_WORKLOAD_FILTER, search: "gpu-b", family: "monitoring" }).map((item) => item.name)).toEqual([
      "grafana-0",
    ]);
  });
});

describe("counts", () => {
  it("counts families and phases in a fixed order, leaving out empty ones", () => {
    expect(familyCounts(fleet)).toEqual([
      { id: "model-serving", label: "Model serving", count: 3 },
      { id: "document-parsing", label: "Document parsing", count: 2 },
      { id: "workers", label: "Workers", count: 4 },
      { id: "gateway", label: "Gateway", count: 2 },
      { id: "control-plane", label: "Control plane", count: 1 },
      { id: "monitoring", label: "Monitoring", count: 1 },
      { id: "other", label: "Other", count: 2 },
    ]);
    expect(phaseCounts(fleet)).toEqual([
      { id: "failed", label: "Failed", count: 1 },
      { id: "pending", label: "Pending", count: 1 },
      { id: "running", label: "Running", count: 11 },
      { id: "succeeded", label: "Succeeded", count: 2 },
    ]);
  });
});

describe("workloadDetail", () => {
  it("calls a finished Job completed instead of not ready", () => {
    expect(workloadDetail({ phase: "Succeeded", ready: false, restarts: 0 }, 2, 2)).toBe("Completed");
  });

  it("explains pending, failed and unready pods", () => {
    expect(workloadDetail({ phase: "Pending", ready: false, restarts: 0 }, 1, 2)).toBe("Waiting for node capacity (1/2 ready)");
    expect(workloadDetail({ phase: "Pending", ready: false, restarts: 0 }, 2, 2)).toBe("Waiting for GPU quota or image pull");
    expect(workloadDetail({ phase: "Failed", ready: false, restarts: 3 }, 2, 2)).toBe("Failed · 3 restarts");
    expect(workloadDetail({ phase: "Running", ready: false, restarts: 1 }, 2, 2)).toBe("Not ready · 1 restart");
    expect(workloadDetail({ phase: "Running", ready: true, restarts: 0 }, 2, 2)).toBe("Ready");
    expect(workloadDetail({ phase: "", ready: false, restarts: 0 }, 2, 2)).toBe("Unknown · not ready");
  });
});

describe("fleetSummary", () => {
  it("reduces the fleet to the counts on the summary strip", () => {
    const summary = fleetSummary({
      nodesReady: 1,
      nodesTotal: 2,
      nodes: [node("gpu-a", { modelServerReady: true, model: "qwen" }), node("gpu-b", { ready: false })],
      workloads: fleet,
    });
    expect(summary).toEqual({
      nodesReady: 1,
      nodesTotal: 2,
      gpus: 2,
      modelServersReady: 1,
      workersRunning: 2,
      pending: 1,
      failed: 1,
      notReady: 1,
      completed: 2,
      active: 13,
      total: 15,
    });
  });
});

describe("nodeSummaries", () => {
  it("counts active pods, running workers and trouble per node, ignoring finished Jobs", () => {
    const rows = nodeSummaries({ nodes: [node("gpu-a"), node("gpu-b"), node("gpu-c")], workloads: fleet });
    expect(rows.map((row) => [row.node.name, row.activeWorkloads, row.workersRunning, row.attention])).toEqual([
      ["gpu-a", 7, 1, 1],
      ["gpu-b", 5, 1, 1],
      ["gpu-c", 0, 0, 0],
    ]);
  });
});
