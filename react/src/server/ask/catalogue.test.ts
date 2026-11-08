import { describe, expect, it } from "vitest";
import type { GraphDataset } from "../protocol/schema";
import { catalogueCards, findDocuments, renderCards } from "./catalogue";

function dataset(documents: Record<string, unknown>[]): GraphDataset {
  return {
    graphId: "graph",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents,
    chunks: [],
    discardedWindows: [],
    rejectedRecords: [],
    failedDocuments: [],
    runReport: {},
    graphMetrics: {},
  } as unknown as GraphDataset;
}

const graph = dataset([
  {
    id: "doc_orchard",
    source_uri: "s3://bucket/farms/orchard-trial.pdf",
    folder: "farms",
    summary: "Title: Orchard trial\nPlaces: North valley\nKey findings:\n- Irrigation raised yield 12%.",
  },
  {
    id: "doc_soil",
    source_uri: "s3://bucket/farms/soil-survey.pdf",
    folder: "farms",
    summary: "Title: Soil survey\nPlaces: South plain\nKey findings:\n- Soil pH ranged from 5.5 to 7.",
  },
  { id: "doc_blank", source_uri: "s3://bucket/misc/notes.txt", summary: "" },
]);

describe("findDocuments", () => {
  it("lists every document with its card when there is no query", () => {
    const result = findDocuments(graph);

    expect(result.total).toBe(3);
    expect(result.cards.map((card) => card.documentId)).toEqual(["doc_orchard", "doc_soil", "doc_blank"]);
    expect(catalogueCards(graph)[2]?.card).toBeNull();
  });

  it("ranks cards by the words of the query and honours a document scope", () => {
    expect(findDocuments(graph, { query: "irrigation yield" }).cards.map((card) => card.documentId)).toEqual([
      "doc_orchard",
    ]);
    expect(
      findDocuments(graph, { query: "farms", documentIds: ["doc_soil"] }).cards.map((card) => card.documentId),
    ).toEqual(["doc_soil"]);
  });
});

describe("renderCards", () => {
  it("writes a heading per document and marks one without a card", () => {
    const text = renderCards(findDocuments(graph, { documentIds: ["doc_soil", "doc_blank"] }));

    expect(text).toContain("2 of 3 documents match; showing 2.");
    expect(text).toContain("## soil-survey.pdf (doc_soil)\nLocation: farms\nTitle: Soil survey");
    expect(text).toContain("(doc_blank)");
    expect(text).toContain("(no card)");
  });
});
