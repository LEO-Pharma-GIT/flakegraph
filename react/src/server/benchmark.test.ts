import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { benchmarkGraph, benchmarkReport, packBaselines } from "./benchmark";
import { resetAppEnv } from "./env";
import { goldToDataset, writeDataset, type GoldGraph } from "./gold";
import { saveUploadedGold } from "./gold-store";
import { qualityForGraph } from "./quality";
import type { RunSnapshot } from "./protocol/schema";

/** What the evaluator prints, shaped as `flakegraph inspect evaluate` prints it. */
const EVALUATION = {
  gold_name: "clinic-v1",
  evaluation_scope: { entity_coverage: "exhaustive", relation_coverage: "reference" },
  ok: false,
  acceptance: { entity_recall: true, entity_precision: true, triple_recall: false },
  thresholds: { entity_recall: 0.8, entity_precision: 0.7, triple_recall: 0.9 },
  entities: { precision: 0.75, recall: 1, f1: 0.8571, expected: 3, missing: [] },
  triples: {
    precision: null,
    recall: 0.5,
    f1: null,
    expected: 2,
    missing: ["drug treats psoriasis"],
    missing_relation_ids: ["r2"],
    missing_by_reason: { wrong_predicate: 1, no_edge: 0 },
  },
  evidence: { support_rate: 0.5 },
  two_hop_recoverability: 0,
  information_retention: 0.75,
};

const GOLD: GoldGraph = {
  name: "clinic-v1",
  description: "A small clinic graph.",
  entities: [
    { id: "a", name: "Adalimumab", type: "DRUG" },
    { id: "p", name: "Psoriasis", type: "DISEASE" },
    { id: "t", name: "TNF", type: "TARGET" },
  ],
  relations: [
    { id: "r1", source: "a", target: "t", relation_type: "INHIBITS" },
    { id: "r2", source: "a", target: "p", relation_type: "TREATS" },
  ],
};

let root: string;
let calls: string;

beforeEach(async () => {
  root = await mkdtemp(path.join(tmpdir(), "benchmark-"));
  calls = path.join(root, "calls.log");
  const stub = path.join(root, "evaluate.mjs");
  await writeFile(
    stub,
    `import { appendFileSync } from "node:fs";
appendFileSync(${JSON.stringify(calls)}, process.argv.slice(2).join(" ") + "\\n");
console.log("progress line that is not the result");
console.log(${JSON.stringify(JSON.stringify(EVALUATION))});
`,
  );
  process.env.FLAKEGRAPH_CLI = `node ${stub}`;
  resetAppEnv();
});

afterEach(() => {
  delete process.env.FLAKEGRAPH_CLI;
  resetAppEnv();
});

async function callCount(): Promise<number> {
  return (await readFile(calls, "utf8").catch(() => "")).split("\n").filter(Boolean).length;
}

describe("the Quality tab's benchmark", () => {
  it("keeps the evaluator's scores, with precision left n/a where the gold cannot judge it", () => {
    const report = benchmarkReport(EVALUATION);
    expect(report.entities).toMatchObject({ precision: 0.75, recall: 1, f1: 0.8571, expected: 3, found: 3 });
    expect(report.triples).toMatchObject({ precision: null, f1: null, recall: 0.5, found: 1 });
    expect(report.triples.missingByReason).toEqual({ wrong_predicate: 1 });
    expect(report.coverage).toEqual({ entities: "exhaustive", relations: "reference" });
    expect(report.gates).toContainEqual({ gate: "triple_recall", passed: false });
  });

  it("scores a graph once, and again only when its files or its gold change", async () => {
    const graph = path.join(root, "graph");
    await writeDataset(graph, goldToDataset(GOLD, "graph_clinic"));
    const goldPath = path.join(root, "gold.json");
    await writeFile(goldPath, JSON.stringify(GOLD));
    const options = {
      graphDirectory: graph,
      goldPath,
      cacheFile: path.join(root, "cache", "run_1.json"),
      repositoryRoot: root,
    };
    const first = await benchmarkGraph(options);
    expect(first.goldName).toBe("clinic-v1");
    await benchmarkGraph(options);
    expect(await callCount()).toBe(1);
    expect(await readFile(calls, "utf8")).toContain(`inspect evaluate --gold ${goldPath} --output ${graph} --no-fail`);

    await writeFile(path.join(graph, "edges.json"), "[]");
    await benchmarkGraph(options);
    expect(await callCount()).toBe(2);
    await writeFile(goldPath, JSON.stringify({ ...GOLD, description: "Revised." }));
    await benchmarkGraph(options);
    expect(await callCount()).toBe(3);
  });

  it("lists as missing exactly the required relations the evaluator missed", async () => {
    const stateRoot = path.join(root, "state");
    const graph = path.join(root, "graph");
    const dataset = goldToDataset(GOLD, "graph_clinic");
    await writeDataset(graph, dataset);
    await saveUploadedGold(stateRoot, "graph_clinic", GOLD);
    const snapshot = { runId: "run_1", graphId: "graph_clinic", raw: {} } as unknown as RunSnapshot;
    const quality = await qualityForGraph({ snapshot, dataset, repositoryRoot: root, stateRoot, graphDirectory: graph });
    // By name both ends of r2 are present; the evaluator says the edge is not.
    expect(quality.benchmark?.triples.recall).toBe(0.5);
    expect(quality.gold?.missingRequired).toEqual([
      { id: "r2", source: "Adalimumab", target: "Psoriasis", relationType: "TREATS" },
    ]);
    expect(quality.gold?.matchedRequired).toBe(1);
  });

  it("says why when the evaluator cannot score, and still reports the rest", async () => {
    const failing = path.join(root, "fail.mjs");
    await writeFile(failing, "process.exit(3);\n");
    process.env.FLAKEGRAPH_CLI = `node ${failing}`;
    resetAppEnv();
    const stateRoot = path.join(root, "state");
    const dataset = goldToDataset(GOLD, "graph_clinic");
    await writeDataset(path.join(root, "graph"), dataset);
    await saveUploadedGold(stateRoot, "graph_clinic", GOLD);
    const snapshot = { runId: "run_1", graphId: "graph_clinic", raw: {} } as unknown as RunSnapshot;
    const quality = await qualityForGraph({
      snapshot,
      dataset,
      repositoryRoot: root,
      stateRoot,
      graphDirectory: path.join(root, "graph"),
    });
    expect(quality.benchmark).toBeNull();
    expect(quality.benchmarkError).toBe("The evaluator returned no result");
    expect(quality.gold?.requiredTotal).toBe(2);
  });

  it("offers a sample pack's recorded results on this gold and version as baselines, newest first", async () => {
    const pack = path.join(root, "pack");
    await mkdir(path.join(pack, "results"), { recursive: true });
    await writeFile(path.join(pack, "gold.json"), JSON.stringify({ ...GOLD, version: "2.0.0" }));
    const result = (id: string, version: string, at: string, model: string) => ({
      result_id: id,
      measured_at_utc: at,
      dataset: { id: "clinic-v1", version },
      models: { llm: { name: model } },
      measurements: { quality: { entity_recall: 0.9, entity_precision: 0.8, entity_f1: 0.847, triple_recall: 0.5 } },
    });
    await writeFile(path.join(pack, "results", "a.json"), JSON.stringify(result("a", "2.0.0", "2026-09-01T00:00:00Z", "older")));
    await writeFile(path.join(pack, "results", "b.json"), JSON.stringify(result("b", "2.0.0", "2026-09-20T00:00:00Z", "newer")));
    await writeFile(path.join(pack, "results", "c.json"), JSON.stringify(result("c", "1.0.0", "2026-09-21T00:00:00Z", "stale")));
    await writeFile(
      path.join(pack, "results", "d.json"),
      JSON.stringify({ ...result("d", "2.0.0", "2026-09-22T00:00:00Z", "elsewhere"), dataset: { id: "other", version: "2.0.0" } }),
    );
    const baselines = await packBaselines(path.join(pack, "gold.json"));
    expect(baselines.map((item) => item.model)).toEqual(["newer", "older"]);
    expect(baselines[0]?.entities).toEqual({ precision: 0.8, recall: 0.9, f1: 0.847 });
    expect(await packBaselines(path.join(root, "gold.json"))).toEqual([]);
  });
});
