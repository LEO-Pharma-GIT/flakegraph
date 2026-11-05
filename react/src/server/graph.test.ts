import path from "node:path";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { edgeStatedTypes, loadLocalGraph, reviewEvidence } from "./graph";
import { LocalRuntime } from "./runtime/local";
import { Effect } from "effect";
import { snapshotFromRecord } from "./catalog";
import { searchCommunities } from "./ask/retrieve";

const FIXTURE = path.join(import.meta.dirname, "__fixtures__", "graph");

describe("loadLocalGraph", () => {
  it("joins separate community findings for retrieval", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fg-findings-"));
    try {
      for (const [table, rows] of Object.entries({ nodes: [{ id: "n" }], edges: [], communities: [{ id: "c", level: 0, title: "Group", summary: "Overview" }], community_findings: [{ id: "f", community_id: "c", summary: "QUARTZ_SENTINEL", explanation: "Grounded detail" }] })) {
        await writeFile(path.join(directory, `${table}.json`), JSON.stringify(rows));
      }
      const dataset = await loadLocalGraph(directory, { purpose: "retrieval" });
      expect(dataset.communities[0].findings).toEqual([expect.objectContaining({ id: "f" })]);
      expect(searchCommunities(dataset, "QUARTZ_SENTINEL")).toHaveLength(1);
      expect(searchCommunities(dataset, "QUARTZ_SENTINEL")[0].findings).toEqual([expect.objectContaining({ summary: "QUARTZ_SENTINEL", explanation: "Grounded detail" })]);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  it("keeps browser limits out of server retrieval", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fg-retrieval-"));
    try {
      const rows = {
        nodes: Array.from({ length: 10_001 }, (_, i) => ({ id: `n${i}`, name: `Node ${i}` })),
        edges: Array.from({ length: 25_001 }, (_, i) => ({ id: `e${i}`, source_node_id: "n0", target_node_id: "n1" })),
        communities: Array.from({ length: 2_001 }, (_, i) => ({ id: `c${i}` })),
        evidence: Array.from({ length: 10_001 }, (_, i) => ({ id: `q${i}`, entity_id: "n0", quote: `Quote ${i}` })),
      };
      await Promise.all(Object.entries(rows).map(([name, data]) => writeFile(path.join(directory, `${name}.json`), JSON.stringify(data))));
      const review = await loadLocalGraph(directory);
      expect([review.nodes.length, review.edges.length, review.communities.length, review.evidence.length]).toEqual([10_000, 25_000, 2_000, 10_000]);
      await writeFile(path.join(directory, "run_report.json"), JSON.stringify({ job_id: "r", graph_id: "g" }));
      const runtime = new LocalRuntime(directory, directory);
      const snapshot = snapshotFromRecord({ runId: "r", graphId: "g", runtime: "local", status: "succeeded", outputPath: directory });
      const full = await Effect.runPromise(runtime.loadRunGraph(snapshot, { purpose: "retrieval" }));
      expect([full.nodes.length, full.edges.length, full.communities.length, full.evidence.length]).toEqual([10_001, 25_001, 2_001, 10_001]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
  it("reads parquet artifacts without their vector columns", async () => {
    const dataset = await loadLocalGraph(FIXTURE);
    expect(dataset.graphId).toBe("fixture-graph");
    expect(dataset.nodes.map((node) => node.id)).toEqual(["n1", "n2", "n3", "n4"]);
    expect(dataset.nodes.every((node) => !("embedding" in node))).toBe(true);
    expect(dataset.nodes[0]).toMatchObject({ name: "Judo", primary_type: "ART" });
    expect(dataset.edges).toHaveLength(3);
    expect(dataset.edges.every((edge) => !("description_embedding" in edge))).toBe(true);
    expect(dataset.edges[0]).toMatchObject({ source_node_id: "n1", target_node_id: "n2", confidence: 0.95 });
    expect(dataset.runReport.app_full_counts).toMatchObject({ node: 4, edge: 3, community: 0, evidence: 0 });
  });
});

describe("edgeStatedTypes", () => {
  it("keeps each type the model stated for an edge a fallback relation kept, once", () => {
    const stated = edgeStatedTypes([
      { edge_id: "e1", proposed_relation_type: "USES_EQUIPMENT" },
      { edge_id: "e1", proposed_relation_type: "USES_EQUIPMENT" },
      { edge_id: "e1", proposed_relation_type: "USES_PROCESS" },
      { edge_id: "e2", proposed_relation_type: null },
    ]);
    expect(stated.get("e1")).toEqual(["USES_EQUIPMENT", "USES_PROCESS"]);
    expect(stated.has("e2")).toBe(false);
  });
});


describe("bounded review evidence", () => {
  it("retains grounding for selected edges after a large entity evidence prefix", () => {
    const nodeQuotes = Array.from({ length: 10_001 }, (_, i) => ({ id: `nq${i}`, subject_kind: "node", subject_id: "n", quote: `Node quote ${i}` }));
    const edgeQuote = { id: "eq", subject_kind: "edge", subject_id: "e", quote: "Grounded relation", document_id: "d" };
    const unrelated = { id: "other", relation_id: "not-kept", quote: "Other relation" };
    const result = reviewEvidence([...nodeQuotes, unrelated, edgeQuote], [{ id: "n" }], [{ id: "e" }], 10_000);
    expect(result).toHaveLength(10_000);
    expect(result[0]).toEqual(edgeQuote);
    expect(result).not.toContain(unrelated);
    expect(reviewEvidence([edgeQuote], [], [{ id: "e" }], 0)).toEqual([]);
  });
});
