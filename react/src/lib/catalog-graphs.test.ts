import { describe, expect, it } from "vitest";
import type { RunSnapshot } from "@/server/protocol/schema";
import { askableGraphs, catalogGraphs } from "./catalog-graphs";

const run = (runId: string, graphId: string, status: string) => ({ runId, graphId, status }) as RunSnapshot;

describe("catalogGraphs", () => {
  it("makes one row per graph, showing a run under way, else the newest success, else the newest attempt", () => {
    const graphs = catalogGraphs([
      run("r5", "g1", "failed"),
      run("r4", "g2", "running"),
      run("r3", "g1", "succeeded"),
      run("r2", "g2", "succeeded"),
      run("r1", "g1", "succeeded"),
      run("r0", "g3", "cancelled"),
    ]);
    expect(graphs.map((graph) => [graph.graphId, graph.run.runId, graph.runIds])).toEqual([
      ["g1", "r3", ["r5", "r3", "r1"]],
      ["g2", "r4", ["r4", "r2"]],
      ["g3", "r0", ["r0"]],
    ]);
  });
});

describe("askableGraphs", () => {
  it("offers each graph's newest successful version, and no graph without one", () => {
    const graphs = askableGraphs([
      run("r5", "g1", "failed"),
      run("r4", "g2", "running"),
      run("r3", "g1", "succeeded"),
      run("r2", "g2", "succeeded"),
      run("r1", "g1", "succeeded"),
      run("r0", "g3", "cancelled"),
    ]);
    expect(graphs.map((graph) => [graph.graphId, graph.runId])).toEqual([
      ["g1", "r3"],
      ["g2", "r2"],
    ]);
  });
});
