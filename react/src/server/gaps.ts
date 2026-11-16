import type { GraphDataset } from "./protocol/schema";

/**
 * Windows in one document a reader can open in the report before the list
 * is cut. Totals stay exact; only the per-window detail is bounded, so a
 * document that failed wholesale does not put hundreds of previews on the
 * wire for every quality query.
 */
export const GAP_WINDOWS_PER_DOCUMENT = 100;

export type GapStage = "entities" | "relations";

export interface GapWindow {
  id: string;
  stage: GapStage;
  pageStart: number | null;
  pageEnd: number | null;
  /** Records the model returned for this window, none of which were kept. */
  extracted: number;
  /** Why each record was rejected, by reason. */
  reasons: Record<string, number>;
  /** The start of the window's text, so what was lost can be judged. */
  preview: string;
}

export interface GapDocument {
  fileId: string;
  /** The source's filename when the run recorded one. */
  name: string;
  windows: number;
  entityWindows: number;
  relationWindows: number;
  /** Records rejected across the document's discarded windows. */
  records: number;
  reasons: Record<string, number>;
  /** The pages the gaps fall on, folded into ranges: "2–3, 7". */
  pages: string;
  detail: GapWindow[];
  /** Windows beyond the detail bound; the counts above still include them. */
  detailOmitted: number;
}

/**
 * The graph's known gaps: every window the model read and returned records
 * for that validation rejected in full. What that text says is not in the
 * graph, and this is the report that says so - per document, with the
 * reasons counted, sized for a corpus where hundreds of documents may carry
 * one.
 */
export interface ExtractionGaps {
  windows: number;
  documents: number;
  documentsTotal: number;
  records: number;
  entityWindows: number;
  relationWindows: number;
  reasons: Record<string, number>;
  rows: GapDocument[];
}

export function extractionGaps(dataset: GraphDataset): ExtractionGaps {
  const names = documentNames(dataset);
  const byFile = new Map<string, { windows: GapWindow[]; fileId: string }>();
  for (const raw of dataset.discardedWindows) {
    const fileId = String(raw.file_id ?? raw.document_id ?? "");
    const stage: GapStage = String(raw.stage) === "relations" ? "relations" : "entities";
    const window: GapWindow = {
      id: String(raw.id ?? ""),
      stage,
      pageStart: numberOrNull(raw.page_start),
      pageEnd: numberOrNull(raw.page_end),
      extracted: Number(raw.extracted_records ?? 0),
      reasons: countRecord(raw.record_actions),
      preview: String(raw.preview ?? ""),
    };
    const entry = byFile.get(fileId) ?? { windows: [], fileId };
    entry.windows.push(window);
    byFile.set(fileId, entry);
  }
  const rows: GapDocument[] = [];
  const reasons: Record<string, number> = {};
  let records = 0;
  let entityWindows = 0;
  let relationWindows = 0;
  for (const { fileId, windows } of byFile.values()) {
    windows.sort((left, right) => (left.pageStart ?? 0) - (right.pageStart ?? 0) || left.stage.localeCompare(right.stage));
    const documentReasons: Record<string, number> = {};
    let documentRecords = 0;
    let entities = 0;
    for (const window of windows) {
      documentRecords += window.extracted;
      if (window.stage === "entities") {
        entities += 1;
      }
      for (const [reason, count] of Object.entries(window.reasons)) {
        documentReasons[reason] = (documentReasons[reason] ?? 0) + count;
        reasons[reason] = (reasons[reason] ?? 0) + count;
      }
    }
    records += documentRecords;
    entityWindows += entities;
    relationWindows += windows.length - entities;
    rows.push({
      fileId,
      name: names.get(fileId) ?? fileId,
      windows: windows.length,
      entityWindows: entities,
      relationWindows: windows.length - entities,
      records: documentRecords,
      reasons: documentReasons,
      pages: pageRanges(windows),
      detail: windows.slice(0, GAP_WINDOWS_PER_DOCUMENT),
      detailOmitted: Math.max(0, windows.length - GAP_WINDOWS_PER_DOCUMENT),
    });
  }
  // The documents that lost the most first: that is where a reader looks.
  rows.sort((left, right) => right.windows - left.windows || left.name.localeCompare(right.name));
  return {
    windows: entityWindows + relationWindows,
    documents: rows.length,
    documentsTotal: documentTotal(dataset),
    records,
    entityWindows,
    relationWindows,
    reasons: sortedRecord(reasons),
    rows,
  };
}

/** "2–3, 7, 12–14": the pages a document's gaps fall on, as ranges. */
export function pageRanges(windows: readonly Pick<GapWindow, "pageStart" | "pageEnd">[]): string {
  const pages = new Set<number>();
  for (const window of windows) {
    if (window.pageStart === null) {
      continue;
    }
    for (let page = window.pageStart; page <= (window.pageEnd ?? window.pageStart); page += 1) {
      pages.add(page);
    }
  }
  const sorted = [...pages].sort((left, right) => left - right);
  const ranges: string[] = [];
  let start: number | null = null;
  let previous: number | null = null;
  for (const page of sorted) {
    if (start === null || previous === null || page !== previous + 1) {
      if (start !== null && previous !== null) {
        ranges.push(start === previous ? String(start) : `${start}–${previous}`);
      }
      start = page;
    }
    previous = page;
  }
  if (start !== null && previous !== null) {
    ranges.push(start === previous ? String(start) : `${start}–${previous}`);
  }
  return ranges.join(", ");
}

function documentNames(dataset: GraphDataset): Map<string, string> {
  const names = new Map<string, string>();
  for (const document of dataset.documents) {
    const fileId = String(document.file_id ?? document.id ?? "");
    // A fleet artifact names the source by uri; a gold-derived sample by
    // path or title. The filename is what a reader recognises.
    const located = String(document.source_uri ?? document.path ?? "");
    const name = located.split(/[\\/]/).filter(Boolean).pop() ?? String(document.title ?? "");
    if (fileId && name) {
      names.set(fileId, decodeURIComponentSafe(name));
    }
  }
  return names;
}

function documentTotal(dataset: GraphDataset): number {
  const counts = dataset.runReport.app_full_counts;
  if (counts && typeof counts === "object" && typeof (counts as Record<string, unknown>).document === "number") {
    return (counts as Record<string, number>).document;
  }
  return dataset.documents.length;
}

function decodeURIComponentSafe(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function countRecord(value: unknown): Record<string, number> {
  if (typeof value === "string") {
    try {
      return countRecord(JSON.parse(value));
    } catch {
      return {};
    }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  const record: Record<string, number> = {};
  for (const [key, count] of Object.entries(value as Record<string, unknown>)) {
    const number = Number(count);
    if (Number.isFinite(number) && number > 0) {
      record[key] = number;
    }
  }
  return sortedRecord(record);
}

function sortedRecord(record: Record<string, number>): Record<string, number> {
  return Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));
}
