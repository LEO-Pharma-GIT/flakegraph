import { describe, expect, it } from "vitest";
import { askGraph } from "./ask";
import { goldToDataset } from "./gold";
import { scanText } from "./pii";
import { estimateConsumption, compareEstimateToActual } from "./estimate";
import { documentStatusesFromEvents } from "./documents";
import { inspectFromDataset, inspectHtml } from "./inspect-report";
import { qualityForGraph } from "./quality";
import { ontologyCoverage } from "./gold";
import { sampleHighConfidenceReviews } from "./workspace";

const dataset = goldToDataset(
  {
    name: "martial",
    entities: [{ id: "n1", name: "Jigoro Kano", type: "PERSON" }],
    relations: [{ id: "r1", source: "n1", target: "n1", relation_type: "RELATED_TO" }],
  },
  "graph_martial_arts",
);

describe("askGraph", () => {
  it("answers a local name question with quotes rather than a community citation", () => {
    const answer = askGraph(dataset, "Who is Jigoro Kano?", "local");
    expect(answer.entityIds).toContain("n1");
    expect(answer.usedCommunityReport).toBe(false);
    expect(answer.queryConsumption.retrievedEntities).toBeGreaterThan(0);
  });
});

describe("pii", () => {
  it("flags an email before persist", () => {
    const hits = scanText("Contact jane@example.com", "doc.md");
    expect(hits[0]?.kind).toBe("email");
  });
});

describe("estimate", () => {
  it("returns a band not a point", () => {
    const estimate = estimateConsumption({ objectCount: 10, runtime: "local", ocrProvider: "built_in_text", parallelism: 4 });
    expect(estimate.usdHigh).toBeGreaterThan(estimate.usdLow);
    expect(estimate.preset).toBe("fast");
  });

  it("compares estimate to actual", () => {
    const estimate = estimateConsumption({ objectCount: 10, runtime: "local", ocrProvider: "builtin_text", parallelism: 4 });
    const comparison = compareEstimateToActual(estimate, 1.5);
    expect(comparison?.label).toMatch(/usd/i);
  });
});

describe("documents", () => {
  it("classifies a failed OCR file as ocr-failed", () => {
    const rows = documentStatusesFromEvents([
      {
        timestamp: "t",
        stage: "ocr",
        status: "failed",
        fileId: "doc_poison_scan",
        message: "bad pdf",
        elapsedMs: 1,
        counts: {},
        metadata: {},
      },
    ]);
    expect(rows[0]?.phase).toBe("ocr-failed");
  });
});

describe("inspect", () => {
  it("builds table counts from the same dataset quality uses", async () => {
    const quality = await qualityForGraph({
      snapshot: {
        runId: "run",
        graphId: "graph_martial_arts",
        status: "succeeded",
        startedAt: null,
        updatedAt: null,
        graphName: "Martial arts history",
        stages: [],
        events: [],
        documentsTotal: 1,
        documentsCompleted: 1,
        documentsFailed: 0,
        outputPath: null,
        storageKind: "local_files",
        storageLocation: null,
        warnings: [],
        error: null,
        listingWarning: null,
        raw: {},
      },
      dataset,
      repositoryRoot: process.cwd(),
      stateRoot: process.cwd(),
    });
    const report = inspectFromDataset({ graphId: "graph_martial_arts", graphName: "Martial arts history" }, dataset, quality);
    expect(report.tables.nodes).toBe(1);
    expect(inspectHtml(report)).toContain("Gold compare");
  });
});

describe("ontology coverage", () => {
  it("flags gold types the proposal dropped", () => {
    const coverage = ontologyCoverage(["PERSON"], { entities: [{ id: "1", name: "A", type: "PERSON" }, { id: "2", name: "B", type: "SCHOOL" }] });
    expect(coverage.missing).toEqual(["SCHOOL"]);
  });
});

describe("HITL sample", () => {
  it("samples about one percent of high-confidence edges", () => {
    const edges = Array.from({ length: 200 }, (_, index) => ({ id: `e${index}`, confidence: 0.95 }));
    const sampled = sampleHighConfidenceReviews("graph_martial_arts", edges);
    expect(sampled.length).toBeGreaterThan(0);
    expect(sampled.length).toBeLessThanOrEqual(3);
  });

  it("names the triple a reviewer is asked about", () => {
    const [item] = sampleHighConfidenceReviews(
      "g",
      [{ id: "r1", source_node_id: "n1", target_node_id: "n2", relation_type: "developed_by", confidence: 0.95 }],
      [
        { id: "n1", name: "Judo" },
        { id: "n2", name: "Jigoro Kano" },
      ],
    );
    expect(item?.triple).toEqual({ source: "Judo", relation: "developed_by", target: "Jigoro Kano" });
  });
});
