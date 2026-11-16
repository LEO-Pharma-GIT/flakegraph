import { describe, expect, it } from "vitest";
import { failedDocuments } from "./failed-documents";
import type { GraphDataset } from "./protocol/schema";

function dataset(failed: Record<string, unknown>[], documents = 4): GraphDataset {
  return {
    graphId: "graph_1",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents: Array.from({ length: documents }, (_, index) => ({ id: `file_${index + 1}`, file_id: `file_${index + 1}` })),
    chunks: [],
    discardedWindows: [],
    rejectedRecords: [],
    failedDocuments: failed,
    runReport: { app_full_counts: { document: documents } },
    graphMetrics: {},
  };
}

function failure(fileId: string, uri: string, errorType: string, message = ""): Record<string, unknown> {
  return {
    id: `failed_${fileId}`,
    file_id: fileId,
    document_id: fileId,
    source_uri: uri,
    mime_type: "application/pdf",
    size_bytes: 18,
    provider: "fallback",
    error_type: errorType,
    error_message: message,
  };
}

describe("failedDocuments", () => {
  it("names each unreadable file by its filename, with the error it gave", () => {
    const report = failedDocuments(
      dataset([
        failure("f2", "s3://corpus/batch%201/z-scan.pdf", "RuntimeError", "Unsupported file type: txt"),
        failure("f1", "s3://corpus/broken.pdf", "PdfReadError", "EOF marker not found"),
        failure("f3", "s3://corpus/empty.pdf", "RuntimeError"),
      ]),
    );

    expect(report.documents).toBe(3);
    // The graph's four documents and the three that never made it in.
    expect(report.documentsTotal).toBe(7);
    expect(report.rows.map((row) => row.name)).toEqual(["broken.pdf", "empty.pdf", "z-scan.pdf"]);
    expect(report.rows[2]).toMatchObject({ errorType: "RuntimeError", errorMessage: "Unsupported file type: txt", sizeBytes: 18 });
    expect(report.byError).toEqual({ RuntimeError: 2, PdfReadError: 1 });
  });

  it("reports nothing for a graph with every document read, or one with no such table", () => {
    expect(failedDocuments(dataset([]))).toEqual({ documents: 0, documentsTotal: 4, byError: {}, rows: [] });
  });
});
