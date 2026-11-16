import type { ProgressEvent } from "./protocol/schema";

export type DocumentPhase =
  | "queued"
  | "running"
  | "scanned"
  | "skipped"
  | "ocr-failed"
  | "extracted-0"
  | "indexed"
  | "poison"
  | "cancelled"
  | "inherited";

export interface DocumentStatus {
  fileId: string;
  /** The source's own filename when the run recorded one. */
  name?: string;
  /** Where the file sits below its source root, folders and file name, when the run recorded it. */
  location?: string;
  phase: DocumentPhase;
  detail: string;
  /**
   * How a revision came by the document's extraction: `reused`,
   * `revalidated`, `reextracted:<stage>`, or `extracted` for one it read itself.
   */
  provenance?: string;
}

/**
 * The durable stages whose scope is one document, in pipeline order. A
 * revision starts a document it reprocesses at a later one of them.
 */
export const DOCUMENT_STAGES = [
  "prepare_document",
  "extract_document_context",
  "revalidate_document_context",
  "compact_entity_inventory",
  "compact_document",
  "revalidate_document",
] as const;

/** The stages whose success leaves a document's extraction complete. */
const FINISHING_STAGES = new Set(["compact_document", "revalidate_document"]);

interface DocumentTask {
  stage: string;
  scopeId: string;
  status: string;
  attempts: number;
  lastError: unknown;
  sourceMetadata: unknown;
  /** Why the document could not be read, when its prepare task recorded that. */
  unreadable?: unknown;
}

/**
 * One status per document from a fleet run's task rows.
 *
 * A document is indexed once its compaction succeeded, poisoned or OCR-failed
 * by whichever of its tasks failed, in progress while one runs, and scanned
 * once parsed but not yet compacted. One no parser could read is OCR-failed
 * too, though its prepare task succeeded: that task recorded the failure and
 * let the run go on. The prepare task's source artifact names the file.
 */
export function documentStatusesFromTasks(tasks: readonly DocumentTask[]): DocumentStatus[] {
  const byFile = new Map<string, DocumentTask[]>();
  for (const task of tasks) {
    const rows = byFile.get(task.scopeId) ?? [];
    rows.push(task);
    byFile.set(task.scopeId, rows);
  }
  const statuses: DocumentStatus[] = [];
  for (const [fileId, rows] of byFile) {
    const source = rows.find((row) => row.stage === "prepare_document")?.sourceMetadata;
    const name = sourceFilename(source);
    const location = sourceLocation(source);
    const failed = rows.find((row) => row.status === "failed");
    const status = (phase: DocumentPhase, detail: string) => ({
      fileId,
      ...(name ? { name } : {}),
      ...(location ? { location } : {}),
      phase,
      detail,
    });
    if (failed) {
      const reason = errorMessage(failed.lastError) || `${titleStage(failed.stage)} failed`;
      statuses.push(
        failed.stage === "prepare_document"
          ? status("ocr-failed", reason)
          : status("poison", `${reason} (attempt ${failed.attempts})`),
      );
      continue;
    }
    const unreadable = rows.find((row) => row.stage === "prepare_document" && row.unreadable)?.unreadable;
    if (unreadable) {
      statuses.push(status("ocr-failed", unreadableReason(unreadable)));
      continue;
    }
    if (rows.some((row) => row.status === "cancelled")) {
      statuses.push(status("cancelled", "Cancelled before it finished"));
      continue;
    }
    const finished = rows.find((row) => FINISHING_STAGES.has(row.stage) && row.status === "succeeded");
    if (finished) {
      statuses.push(
        status(
          "indexed",
          finished.stage === "revalidate_document"
            ? "Revalidated under the current rules"
            : "Extracted and compacted into the graph",
        ),
      );
      continue;
    }
    const running = rows.find((row) => row.status === "running");
    if (running) {
      statuses.push(status("running", titleStage(running.stage)));
      continue;
    }
    if (rows.some((row) => row.stage === "prepare_document" && row.status === "succeeded")) {
      statuses.push(status("scanned", "Extraction pending"));
      continue;
    }
    statuses.push(status("queued", "Waiting for a worker"));
  }
  return statuses.sort((left, right) => (left.name ?? left.fileId).localeCompare(right.name ?? right.fileId));
}

/**
 * Label each document with how the run came by its extraction, as its
 * finalizer's payload records it: documents it inherits are reused, those it
 * reprocessed carry their label, and the rest it extracted itself.
 */
export function withProvenance(
  statuses: readonly DocumentStatus[],
  payload: Record<string, unknown> | null,
): DocumentStatus[] {
  const labels = documentProvenance(payload);
  return statuses.map((item) => ({ ...item, provenance: labels.get(item.fileId) ?? "extracted" }));
}

/** The provenance label of every document a finalizer's payload names. */
export function documentProvenance(payload: Record<string, unknown> | null): Map<string, string> {
  const labels = new Map<string, string>();
  const entries = Array.isArray(payload?.inherit) ? payload.inherit : [];
  for (const entry of entries) {
    const fileIds = (entry as Record<string, unknown> | null)?.file_ids;
    for (const fileId of Array.isArray(fileIds) ? fileIds : []) {
      labels.set(String(fileId), "reused");
    }
  }
  const reprocessed = payload?.reprocessed;
  if (reprocessed && typeof reprocessed === "object" && !Array.isArray(reprocessed)) {
    for (const [label, fileIds] of Object.entries(reprocessed as Record<string, unknown>)) {
      for (const fileId of Array.isArray(fileIds) ? fileIds : []) {
        labels.set(String(fileId), label);
      }
    }
  }
  return labels;
}

/** Mark the files an operator quarantined, whichever runtime reported them. */
export function withSkippedFiles(statuses: readonly DocumentStatus[], skipped: readonly string[]): DocumentStatus[] {
  if (skipped.length === 0) {
    return [...statuses];
  }
  return statuses.map((item) =>
    skipped.includes(item.fileId)
      ? { ...item, phase: "skipped" as const, detail: "Quarantined. Retry will not send this file." }
      : item,
  );
}

function sourceFilename(metadata: unknown): string | undefined {
  const record = inputFile(metadata);
  const name = record?.filename ?? record?.source_uri;
  return typeof name === "string" && name ? name : undefined;
}

function sourceLocation(metadata: unknown): string | undefined {
  const location = inputFile(metadata)?.location;
  return typeof location === "string" && location ? location : undefined;
}

function inputFile(metadata: unknown): Record<string, unknown> | undefined {
  if (!metadata || typeof metadata !== "object") {
    return undefined;
  }
  const input = (metadata as Record<string, unknown>).input_file;
  return input && typeof input === "object" ? (input as Record<string, unknown>) : undefined;
}

function unreadableReason(failure: unknown): string {
  return `Could not be read and was left out of the graph: ${errorMessage(failure) || "no reason recorded"}`;
}

function errorMessage(error: unknown): string {
  if (!error || typeof error !== "object") {
    return typeof error === "string" ? error : "";
  }
  const record = error as Record<string, unknown>;
  const message = record.error_message ?? record.message;
  const type = record.error_type;
  if (typeof message === "string" && message) {
    return typeof type === "string" && type ? `${type}: ${message}` : message;
  }
  return "";
}

function titleStage(stage: string): string {
  return stage.replaceAll("_", " ").replace(/^\w/, (letter) => letter.toUpperCase());
}

export function documentStatusesFromEvents(
  events: readonly ProgressEvent[],
  skipped: readonly string[] = [],
): DocumentStatus[] {
  const byFile = new Map<string, ProgressEvent[]>();
  for (const event of events) {
    if (!event.fileId) {
      continue;
    }
    const rows = byFile.get(event.fileId) ?? [];
    rows.push(event);
    byFile.set(event.fileId, rows);
  }
  const statuses: DocumentStatus[] = [];
  for (const [fileId, rows] of byFile) {
    if (skipped.includes(fileId)) {
      statuses.push({ fileId, phase: "skipped", detail: "Quarantined. Retry will not send this file." });
      continue;
    }
    const failed = rows.find((row) => row.status === "failed" || row.status === "error");
    if (failed) {
      statuses.push({
        fileId,
        phase: failed.stage.includes("ocr") ? "ocr-failed" : "poison",
        detail: failed.message || "Failed during extraction",
      });
      continue;
    }
    const extractedZero = rows.find((row) => /0 entit/i.test(row.message ?? ""));
    if (extractedZero) {
      statuses.push({ fileId, phase: "extracted-0", detail: extractedZero.message || "Extracted 0 entities" });
      continue;
    }
    const extracted = rows.some((row) => row.stage.includes("extract") && row.status === "completed");
    statuses.push({
      fileId,
      phase: extracted ? "indexed" : "scanned",
      detail: extracted ? "Indexed" : "Seen by OCR",
    });
  }
  return statuses.sort((left, right) => left.fileId.localeCompare(right.fileId));
}
