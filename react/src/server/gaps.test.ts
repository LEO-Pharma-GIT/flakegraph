import { describe, expect, it } from "vitest";
import { GAP_WINDOWS_PER_DOCUMENT, extractionGaps, pageRanges } from "./gaps";
import type { GraphDataset } from "./protocol/schema";

function dataset(discardedWindows: Record<string, unknown>[], documents = 3): GraphDataset {
  return {
    graphId: "graph_1",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents: Array.from({ length: documents }, (_, index) => ({
      id: `doc_${index + 1}`,
      file_id: `file_${index + 1}`,
      source_uri: `s3://corpus/batch-1/record-${index + 1}.pdf`,
    })),
    chunks: [],
    discardedWindows,
    rejectedRecords: [],
    failedDocuments: [],
    runReport: { app_full_counts: { document: documents } },
    graphMetrics: {},
  };
}

function window(fileId: string, stage: string, page: number, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: `w_${fileId}_${stage}_${page}`,
    file_id: fileId,
    document_id: fileId,
    stage,
    window_id: `window_${page}`,
    chunk_ids: [`chunk_${page}`],
    page_start: page,
    page_end: page,
    extracted_records: 2,
    record_actions: { ungrounded_quote: 2 },
    preview: `Page ${page} text`,
    ...extra,
  };
}

describe("extractionGaps", () => {
  it("is empty and says so for a graph without gaps", () => {
    const gaps = extractionGaps(dataset([]));
    expect(gaps).toMatchObject({ windows: 0, documents: 0, documentsTotal: 3, records: 0, rows: [] });
  });

  it("groups windows by document, names the document, and puts the biggest loss first", () => {
    const gaps = extractionGaps(
      dataset([
        window("file_2", "entities", 4),
        window("file_1", "relations", 7, { record_actions: { domain_or_range_violation: 3 }, extracted_records: 3 }),
        window("file_2", "entities", 5, { page_end: 6 }),
        window("file_2", "relations", 9),
      ]),
    );
    expect(gaps.windows).toBe(4);
    expect(gaps.documents).toBe(2);
    expect(gaps.documentsTotal).toBe(3);
    expect(gaps.records).toBe(9);
    expect(gaps.entityWindows).toBe(2);
    expect(gaps.relationWindows).toBe(2);
    expect(gaps.reasons).toEqual({ domain_or_range_violation: 3, ungrounded_quote: 6 });
    expect(gaps.rows.map((row) => row.name)).toEqual(["record-2.pdf", "record-1.pdf"]);
    const [record2] = gaps.rows;
    expect(record2).toMatchObject({ fileId: "file_2", windows: 3, entityWindows: 2, relationWindows: 1, records: 6, pages: "4–6, 9" });
    expect(record2.detail.map((item) => item.pageStart)).toEqual([4, 5, 9]);
  });

  it("bounds the per-document detail while keeping the totals exact", () => {
    const many = Array.from({ length: GAP_WINDOWS_PER_DOCUMENT + 25 }, (_, index) => window("file_1", "entities", index + 1));
    const gaps = extractionGaps(dataset(many));
    expect(gaps.windows).toBe(GAP_WINDOWS_PER_DOCUMENT + 25);
    expect(gaps.rows[0].windows).toBe(GAP_WINDOWS_PER_DOCUMENT + 25);
    expect(gaps.rows[0].detail).toHaveLength(GAP_WINDOWS_PER_DOCUMENT);
    expect(gaps.rows[0].detailOmitted).toBe(25);
  });

  it("reads the reasons a Parquet row carries as a JSON string", () => {
    const gaps = extractionGaps(dataset([window("file_1", "entities", 2, { record_actions: '{"duplicate": 4}' })]));
    expect(gaps.reasons).toEqual({ duplicate: 4 });
  });
});

describe("pageRanges", () => {
  it("folds consecutive pages into ranges", () => {
    expect(
      pageRanges([
        { pageStart: 2, pageEnd: 3 },
        { pageStart: 3, pageEnd: 3 },
        { pageStart: 7, pageEnd: null },
        { pageStart: 12, pageEnd: 14 },
        { pageStart: null, pageEnd: null },
      ]),
    ).toBe("2–3, 7, 12–14");
  });
});
