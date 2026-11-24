import { describe, expect, it } from "vitest";
import type { GraphDataset } from "../protocol/schema";
import { parseRecord, renderRecord } from "./card-record";
import { queryCatalogue, renderCatalogueQuery } from "./catalogue";
import { renderFindings } from "./per-document";

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

const card = (fields: Record<string, unknown>) => JSON.stringify({ title: "", document_type: "report", ...fields });

const graph = dataset([
  {
    id: "doc_orchard",
    source_uri: "s3://bucket/farms/orchard-trial.pdf",
    folder: "farms/trial",
    summary: "Title: Orchard trial",
    card: card({
      date: "2020-09",
      places: ["North valley"],
      organizations: [
        { name: "Field Lab Ltd", role: "author" },
        { name: "Valley Growers", role: "commissioner" },
      ],
      samples: [{ group: "orchard owners", count: 12 }],
    }),
  },
  {
    id: "doc_soil",
    source_uri: "s3://bucket/farms/soil-survey.pdf",
    folder: "farms/soil",
    summary: "Title: Soil survey",
    card: card({
      date: "2021",
      places: ["South plain"],
      organizations: [
        { name: "Field Lab", role: "author" },
        { name: "Seed Bank", role: "source" },
      ],
      samples: [{ group: "farm plots", count: null }],
    }),
  },
  {
    id: "doc_review",
    source_uri: "s3://bucket/farms/review.pdf",
    folder: "farms/review",
    summary: "Title: Review",
    card: card({ date: "2020-03-01", organizations: [{ name: "Seed Bank", role: "author" }] }),
  },
  { id: "doc_old", source_uri: "s3://bucket/misc/notes.txt", summary: "Title: Notes from Field Lab" },
]);

describe("parseRecord", () => {
  it("reads a card's typed fields from JSON text or an object, and nothing from anything else", () => {
    const record = parseRecord('{"date": "2020-09-14T00:00", "organizations": [{"name": "Field Lab", "role": "author"}, "Old Style"], "samples": [{"group": "farmers", "count": 4}]}');
    expect(record?.date).toBe("2020-09-14");
    expect(record?.organizations).toEqual([
      { name: "Field Lab", role: "author" },
      { name: "Old Style", role: "other" },
    ]);
    expect(renderRecord(record!)).toBe("dated 2020-09-14; prepared by Field Lab; sample: farmers (4)");
    expect(parseRecord({ date: "September 2020" })?.date).toBe("");
    expect(parseRecord("not json")).toBeNull();
    expect(parseRecord(null)).toBeNull();
  });
});

describe("queryCatalogue", () => {
  it("counts an organization named with and without its descriptor once", () => {
    const named = (id: string, name: string) => ({ id, source_uri: `s3://bucket/${id}.pdf`, card: card({ organizations: [{ name, role: "author" }] }) });
    const byAuthor = queryCatalogue(dataset([named("a", "Acme"), named("b", "Acme Research"), named("c", "Acme Research"), named("d", "Bolt")]), {}, "author");
    expect(byAuthor.groups?.map((group) => [group.value, group.documents.length])).toEqual([
      ["Acme Research / Acme", 3],
      ["Bolt", 1],
    ]);
  });

  it("counts the documents an organization prepared, not those that only cite it", () => {
    const result = queryCatalogue(graph, { organizations: ["Seed Bank"], role: "author" });
    expect(result.matches.map((match) => match.card.documentId)).toEqual(["doc_review"]);
    expect(result.withoutRecord).toBe(1);
  });

  it("groups by author with spellings of one name together, and filters by year, place and sample", () => {
    const byAuthor = queryCatalogue(graph, {}, "author");
    expect(byAuthor.groups?.map((group) => [group.value, group.documents.length])).toEqual([
      ["Field Lab Ltd", 2],
      ["Seed Bank", 1],
    ]);
    expect(queryCatalogue(graph, { fromYear: 2020, toYear: 2020 }).matches).toHaveLength(2);
    expect(queryCatalogue(graph, { places: ["north"] }).matches.map((match) => match.card.documentId)).toEqual(["doc_orchard"]);
    expect(queryCatalogue(graph, { samples: ["plots"] }).matches.map((match) => match.card.documentId)).toEqual(["doc_soil"]);
    // A document without a record matches on its text only.
    expect(queryCatalogue(graph, { text: "Field Lab notes" }).matches.map((match) => match.card.documentId)).toContain("doc_old");
  });

  it("renders the counts first, then each document's record", () => {
    const text = renderCatalogueQuery(queryCatalogue(graph, { fromYear: 2020, toYear: 2020 }, "author"), "author");
    expect(text.split("\n")[0]).toBe("2 of 4 documents match, in 2 folders. 1 documents have no record and match only by text.");
    expect(text).toContain("- Field Lab Ltd: 1 (orchard-trial.pdf)");
    expect(text).toContain("dated 2020-09; report; prepared by Field Lab Ltd; for Valley Growers; covers North valley; sample: orchard owners (12)");
  });
});

describe("renderFindings with records", () => {
  it("puts each document's record under its heading", () => {
    const text = renderFindings([
      {
        documentId: "doc_orchard",
        documentName: "Orchard trial",
        record: "dated 2020-09; prepared by Field Lab",
        relevant: true,
        answer: "Yield rose 12%.",
        facts: [],
      },
    ]);
    expect(text).toContain("## Orchard trial (doc_orchard)\nRecord: dated 2020-09; prepared by Field Lab\nYield rose 12%.");
  });
});
