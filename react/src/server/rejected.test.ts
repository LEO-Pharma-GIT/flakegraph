import { describe, expect, it } from "vitest";
import type { GraphDataset } from "./protocol/schema";
import { BROKEN_RULES_LIMIT, rejectedRecords } from "./rejected";

function dataset(rejected: Record<string, unknown>[]): GraphDataset {
  return {
    graphId: "g",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents: [
      { id: "f1", file_id: "f1", source_uri: "file:///in/Lab%20A/brochure.pdf" },
      { id: "f2", file_id: "f2", source_uri: "file:///in/notes.docx" },
    ],
    chunks: [],
    discardedWindows: [],
    rejectedRecords: rejected,
    failedDocuments: [],
    runReport: {},
  } as unknown as GraphDataset;
}

const violation = (sourceType: string, quote: string) => ({
  id: `r-${sourceType}-${quote}`,
  file_id: "f1",
  page_number: 3,
  kind: "relation",
  reason: "domain_or_range_violation",
  name: "USES_PROCESS",
  source: "Granulator G-10",
  source_type: sourceType,
  target: "wet granulation",
  target_type: "PROCESS_STEP",
  quote,
});

describe("rejectedRecords", () => {
  it("lists every record readably and counts reasons by kind", () => {
    const report = rejectedRecords(
      dataset([
        violation("EQUIPMENT", "The G-10 performs wet granulation."),
        { id: "e1", file_id: "f2", page_number: 1, kind: "entity", reason: "ungrounded_quote", name: "Acme", type: "ORGANIZATION", quote: "Acmes" },
      ]),
    );
    expect(report.records).toBe(2);
    expect([report.entities, report.relations, report.documents, report.documentsTotal]).toEqual([1, 1, 2, 2]);
    expect(report.reasons).toEqual({ "relation:domain_or_range_violation": 1, "entity:ungrounded_quote": 1 });
    expect(report.rows.map((row) => [row.document, row.statement])).toEqual([
      ["brochure.pdf", "Granulator G-10 (EQUIPMENT) USES_PROCESS wet granulation (PROCESS_STEP)"],
      ["notes.docx", "Acme (ORGANIZATION)"],
    ]);
  });

  it("names the type rules that turned the most relations away, with an example", () => {
    const report = rejectedRecords(
      dataset([
        violation("EQUIPMENT", "first"),
        violation("EQUIPMENT", "second"),
        violation("FACILITY", "third"),
      ]),
    );
    expect(report.brokenRules).toEqual([
      { sourceType: "EQUIPMENT", relationType: "USES_PROCESS", targetType: "PROCESS_STEP", count: 2, example: "first" },
      { sourceType: "FACILITY", relationType: "USES_PROCESS", targetType: "PROCESS_STEP", count: 1, example: "third" },
    ]);
    expect(BROKEN_RULES_LIMIT).toBeGreaterThan(0);
  });
});
