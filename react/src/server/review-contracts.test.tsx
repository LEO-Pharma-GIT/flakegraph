import { mkdtemp, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ConsumptionPanel } from "@/components/console/consumption-panel";
import { actualUsdFromConsumption } from "./estimate";
import { compareToGold, goldToDataset, type GoldGraph } from "./gold";
import { goldTemplate, validateGold } from "./gold-store";
import { changeGraphNames, readGraphNames, renameGraph } from "./catalog";
import { sourceSettings } from "./config";

const gold: GoldGraph = {
  name: "relations", description: "Required directed relationship",
  entities: [{ id: "a", name: "Alpha", type: "THING" }, { id: "b", name: "Beta", type: "THING" }],
  relations: [{ id: "r", source: "a", target: "b", relation_type: "USES", required: true }],
};

describe("gold review contracts", () => {
  it("requires a directed edge with the expected normalized predicate", () => {
    const graph = goldToDataset(gold, "g");
    expect(compareToGold({ ...graph, edges: [{ ...graph.edges[0], relation_type: "belongs-to" }] }, { ...gold, relations: [{ ...gold.relations![0], relation_type: "BELONGS_TO" }] }).matchedRequired).toBe(1);
    expect(compareToGold({ ...graph, edges: [] }, gold).matchedRequired).toBe(0);
    expect(compareToGold({ ...graph, edges: [{ ...graph.edges[0], relation_type: "RELATED_TO" }] }, gold).matchedRequired).toBe(0);
    expect(compareToGold({ ...graph, edges: [{ ...graph.edges[0], source_node_id: "b", target_node_id: "a" }] }, gold).matchedRequired).toBe(0);
    expect(compareToGold({ ...graph, edges: [{ ...graph.edges[0], relation_type: " uses " }] }, gold).matchedRequired).toBe(1);
  });
  it("exports document paths and omits documents without representable provenance", () => {
    const graph = goldToDataset(gold, "g");
    const template = goldTemplate({ ...graph, documents: [{ id: "d", source_uri: "s3://fixtures/notes.pdf" }, { id: "missing" }] }, "Example");
    expect(template.documents).toEqual([{ id: "d", path: "s3://fixtures/notes.pdf" }]);
    expect(validateGold(template)).toEqual(template);
    expect(() => validateGold({ ...gold, documents: [{ id: "d" }] })).toThrow(/path/);
  });
});

describe("consumption presentation", () => {
  it("renders billed and avoided totals and nested token usage", () => {
    const consumption = { totals: { billed_usd: 1.25, avoided_usd: 2.5, unpriced_calls: 0 }, events: [{ stage: "extraction", model: "fixture", usage: { total_tokens: 321 }, locality: "hosted" }] };
    const html = renderToStaticMarkup(<ConsumptionPanel consumption={consumption} estimate={null} />);
    expect(html).toContain("Billed cost 1.25 usd");
    expect(html).toContain("Avoided cost 2.5 usd");
    expect(html).toContain(">321<");
    expect(actualUsdFromConsumption(consumption)).toBe(3.75);
    expect(actualUsdFromConsumption({ totals: { billed_usd: 0, avoided_usd: 0 } })).toBe(0);
    expect(actualUsdFromConsumption({ totals: { billed_usd: 1, unpriced_calls: 1 } })).toBeNull();
    expect(actualUsdFromConsumption({ totals: {} })).toBeNull();
  });
});

describe("source request boundaries", () => {
  it("rejects interpolation recursively before creating provider settings", () => {
    for (const kind of ["s3", "azure_blob", "snowflake_stage"] as const) {
      expect(() => sourceSettings(kind, { prefix: "${SYNTHETIC_SETTING}", nested: { value: "safe" } })).toThrow(/operator profiles/);
    }
    expect(() => sourceSettings("s3", { bucket: "fixtures", nested: ["${SYNTHETIC_SETTING}"] })).toThrow(/operator profiles/);
  });
});

describe("shared graph names", () => {
  it("serializes updates across graphs and canonical state-root aliases", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "fg-names-"));
    const alias = `${root}-alias`;
    try {
      await symlink(root, alias, "dir");
      await Promise.all(Array.from({ length: 20 }, (_, index) => renameGraph(index % 2 ? root : alias, `graph_${index}`, `Name ${index}`)));
      expect(Object.keys(await readGraphNames(root))).toHaveLength(20);
      await Promise.all([changeGraphNames(alias, (names) => { delete names.graph_0; }), renameGraph(root, "external_graph", "External")]);
      expect(await readGraphNames(root)).toMatchObject({ external_graph: "External", graph_19: "Name 19" });
      expect((await readGraphNames(root)).graph_0).toBeUndefined();
    } finally { await rm(alias, { force: true }); await rm(root, { recursive: true, force: true }); }
  });
});
