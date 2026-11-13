import type { GraphDataset } from "./protocol/schema";

export interface FailedDocumentRow {
  fileId: string;
  /** The source's filename, which is what a reader recognises. */
  name: string;
  sourceUri: string;
  mimeType: string | null;
  sizeBytes: number | null;
  /** The parser that refused it. */
  provider: string | null;
  errorType: string;
  errorMessage: string;
}

/**
 * The source documents no parser could read. Each was recorded and the run
 * went on without it, so nothing from these files is in the graph - this is
 * the report that says which files, and why.
 */
export interface FailedDocuments {
  documents: number;
  /** Every document the run was given: the ones in the graph and these. */
  documentsTotal: number;
  /** How many failed with each error type, most common first. */
  byError: Record<string, number>;
  rows: FailedDocumentRow[];
}

export function failedDocuments(dataset: GraphDataset): FailedDocuments {
  const byFile = new Map<string, FailedDocumentRow>();
  for (const raw of dataset.failedDocuments) {
    const fileId = String(raw.file_id ?? raw.document_id ?? "");
    if (!fileId) {
      continue;
    }
    const sourceUri = String(raw.source_uri ?? "");
    byFile.set(fileId, {
      fileId,
      name: fileName(sourceUri) || fileId,
      sourceUri,
      mimeType: textOrNull(raw.mime_type),
      sizeBytes: typeof raw.size_bytes === "number" && Number.isFinite(raw.size_bytes) ? raw.size_bytes : null,
      provider: textOrNull(raw.provider),
      errorType: String(raw.error_type ?? "") || "Unknown",
      errorMessage: String(raw.error_message ?? ""),
    });
  }
  const rows = [...byFile.values()].sort((left, right) => left.name.localeCompare(right.name));
  const byError: Record<string, number> = {};
  for (const row of rows) {
    byError[row.errorType] = (byError[row.errorType] ?? 0) + 1;
  }
  return {
    documents: rows.length,
    documentsTotal: parsedDocumentTotal(dataset) + rows.length,
    byError: Object.fromEntries(
      Object.entries(byError).sort(([leftName, left], [rightName, right]) => right - left || leftName.localeCompare(rightName)),
    ),
    rows,
  };
}

function fileName(sourceUri: string): string {
  const name = sourceUri.split(/[\\/]/).filter(Boolean).pop() ?? "";
  try {
    return decodeURIComponent(name);
  } catch {
    return name;
  }
}

function parsedDocumentTotal(dataset: GraphDataset): number {
  const counts = dataset.runReport.app_full_counts;
  if (counts && typeof counts === "object" && typeof (counts as Record<string, unknown>).document === "number") {
    return (counts as Record<string, number>).document;
  }
  return dataset.documents.length;
}

function textOrNull(value: unknown): string | null {
  const text = value === null || value === undefined ? "" : String(value);
  return text || null;
}
