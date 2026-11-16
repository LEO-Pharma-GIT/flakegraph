import { open, stat } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";
import { atomicWriteJson, readJsonFile } from "./catalog";
import {
  PIPELINE_STAGE_ORDER,
  type ProgressEvent,
  type StageProgress,
} from "./protocol/schema";

const TAIL_BLOCK_BYTES = 64 * 1024;
const SUMMARY_VERSION = 2;
const SUMMARY_FILE = "progress-summary.json";
const TERMINAL_SUCCESS = new Set(["completed", "succeeded", "done"]);

export interface LocalProgress {
  events: ProgressEvent[];
  stages: StageProgress[];
  documentsTotal: number | null;
  documentsCompleted: number;
  documentsFailed: number;
  updatedAt: string | null;
}

export async function readJsonlEvents(file: string, limit = 2_000): Promise<ProgressEvent[]> {
  if (!existsSync(file) || limit <= 0) {
    return [];
  }
  const records: ProgressEvent[] = [];
  for (const line of await tailLines(file, limit)) {
    const event = parseProgressLine(line);
    if (event) {
      records.push(event);
    }
  }
  return records;
}

/**
 * Every document's events, read from the whole file rather than its tail.
 *
 * A run over hundreds of documents writes thousands of lines; the tail that
 * serves the stage bars and the recent-events card would lose the first
 * documents entirely. This streams the file once and keeps only the events
 * that name a document, which is what a document list is built from.
 */
export async function readDocumentEvents(file: string): Promise<ProgressEvent[]> {
  if (!existsSync(file)) {
    return [];
  }
  const records: ProgressEvent[] = [];
  const handle = await open(file, "r");
  try {
    for await (const line of handle.readLines({ encoding: "utf8" })) {
      const event = parseProgressLine(line);
      if (event?.fileId) {
        records.push(event);
      }
    }
  } finally {
    await handle.close();
  }
  return records;
}

function parseProgressLine(line: string): ProgressEvent | null {
  if (!line.trim()) {
    return null;
  }
  try {
    const raw = JSON.parse(line) as Record<string, unknown>;
    return raw && raw.event === "kg_processor.progress" ? progressEvent(raw) : null;
  } catch {
    return null;
  }
}

export async function readLocalProgress(file: string, limit = 2_000): Promise<LocalProgress> {
  if (!existsSync(file)) {
    return { events: [], stages: [], documentsTotal: null, documentsCompleted: 0, documentsFailed: 0, updatedAt: null };
  }
  const summaryPath = path.join(path.dirname(file), SUMMARY_FILE);
  const info = await stat(file);
  const sourceIdentity = `${info.dev}:${info.ino}`;
  let summary = (await readJsonFile<Record<string, unknown>>(summaryPath)) ?? {};
  let offset = asInt(summary.offset);
  const rebuilt =
    summary.version !== SUMMARY_VERSION ||
    summary.source_identity !== sourceIdentity ||
    offset > info.size;
  if (rebuilt) {
    summary = emptySummary(sourceIdentity);
    offset = 0;
  }
  const nextOffset = await accumulateNewRecords(file, offset, summary);
  if (rebuilt || nextOffset !== offset) {
    summary.offset = nextOffset;
    await atomicWriteJson(summaryPath, summary);
  }
  const events = await readJsonlEvents(file, limit);
  return {
    events,
    stages: summaryStages(summary),
    documentsTotal: optionalInt(summary.documents_total),
    documentsCompleted: stageFileCount(summary, "completed_files", "ocr"),
    documentsFailed: stageFileCount(summary, "failed_files", "ocr"),
    updatedAt: typeof summary.updated_at === "string" ? summary.updated_at : null,
  };
}

export function progressEvent(raw: Record<string, unknown>): ProgressEvent {
  return {
    timestamp: String(raw.timestamp ?? ""),
    stage: String(raw.stage ?? "unknown"),
    status: String(raw.status ?? "unknown"),
    fileId: raw.file_id ? String(raw.file_id) : raw.fileId ? String(raw.fileId) : null,
    message: raw.message ? String(raw.message) : null,
    elapsedMs: raw.elapsed_ms != null ? Number(raw.elapsed_ms) : raw.elapsedMs != null ? Number(raw.elapsedMs) : null,
    counts: asRecord(raw.counts),
    metadata: asRecord(raw.metadata),
  };
}

export function makeProgressRecord(
  event: Omit<ProgressEvent, "counts" | "metadata"> & {
    counts?: Record<string, unknown>;
    metadata?: Record<string, unknown>;
  },
): Record<string, unknown> {
  return {
    event: "kg_processor.progress",
    timestamp: event.timestamp,
    stage: event.stage,
    status: event.status,
    file_id: event.fileId,
    message: event.message,
    elapsed_ms: event.elapsedMs,
    counts: event.counts ?? {},
    metadata: event.metadata ?? {},
  };
}

function emptySummary(sourceIdentity: string): Record<string, unknown> {
  return {
    version: SUMMARY_VERSION,
    source_identity: sourceIdentity,
    offset: 0,
    latest: {},
    elapsed: {},
    completed_files: {},
    failed_files: {},
    completed_counts: {},
    totals: {},
  };
}

async function accumulateNewRecords(
  file: string,
  offset: number,
  summary: Record<string, unknown>,
): Promise<number> {
  let nextOffset = offset;
  const handle = await open(file, "r");
  try {
    const stream = handle.createReadStream({ start: offset, encoding: "utf8" });
    let leftover = "";
    for await (const chunk of stream) {
      leftover += chunk;
      const lines = leftover.split("\n");
      leftover = lines.pop() ?? "";
      for (const line of lines) {
        nextOffset += Buffer.byteLength(`${line}\n`, "utf8");
        try {
          const raw = JSON.parse(line) as Record<string, unknown>;
          if (raw?.event === "kg_processor.progress") {
            accumulateSummary(summary, raw);
          }
        } catch {
          continue;
        }
      }
    }
  } finally {
    await handle.close();
  }
  return nextOffset;
}

export function accumulateSummary(summary: Record<string, unknown>, raw: Record<string, unknown>): void {
  const event = progressEvent(raw);
  const latest = mapping(summary, "latest");
  const elapsed = mapping(summary, "elapsed");
  const completedFiles = mapping(summary, "completed_files");
  const failedFiles = mapping(summary, "failed_files");
  const completedCounts = mapping(summary, "completed_counts");
  const totals = mapping(summary, "totals");
  latest[event.stage] = eventRecord(event);
  summary.updated_at = event.timestamp;
  if (event.elapsedMs != null && TERMINAL_SUCCESS.has(event.status)) {
    elapsed[event.stage] = asInt(elapsed[event.stage]) + event.elapsedMs;
  }
  if (event.fileId && TERMINAL_SUCCESS.has(event.status)) {
    recordFile(completedFiles, event.stage, event.fileId);
    forgetFile(failedFiles, event.stage, event.fileId);
  }
  if (event.fileId && event.status === "failed") {
    recordFile(failedFiles, event.stage, event.fileId);
  }
  for (const [completedKey, totalKey] of [
    ["batches_completed", "batches_total"],
    ["reports_completed", "reports_total"],
  ] as const) {
    const completedValue = event.counts[completedKey];
    const totalValue = event.counts[totalKey];
    if (typeof completedValue === "number") {
      completedCounts[event.stage] = completedValue;
    }
    if (typeof totalValue === "number") {
      totals[event.stage] = totalValue;
    }
  }
  const filesSeen = event.counts.files_seen;
  if (event.stage === "file_source" && typeof filesSeen === "number") {
    totals[event.stage] = filesSeen;
    summary.documents_total = filesSeen;
    if (TERMINAL_SUCCESS.has(event.status)) {
      completedCounts[event.stage] = filesSeen;
    }
  }
  for (const key of ["files_total", "documents_total", "total"]) {
    const value = event.counts[key];
    if (typeof value === "number") {
      totals[event.stage] = value;
      if (event.stage === "file_source" || event.stage === "ocr") {
        summary.documents_total = value;
      }
      break;
    }
  }
}

function summaryStages(summary: Record<string, unknown>): StageProgress[] {
  const latest = readonlyMapping(summary, "latest");
  const elapsed = readonlyMapping(summary, "elapsed");
  const completedFiles = readonlyMapping(summary, "completed_files");
  const completedCounts = readonlyMapping(summary, "completed_counts");
  const totals = readonlyMapping(summary, "totals");
  const order = new Map<string, number>(PIPELINE_STAGE_ORDER.map((stage, index) => [stage, index]));
  return Object.entries(latest)
    .sort(
      ([left], [right]) =>
        (order.get(left) ?? PIPELINE_STAGE_ORDER.length) - (order.get(right) ?? PIPELINE_STAGE_ORDER.length),
    )
    .flatMap(([stage, raw]) => {
      if (!raw || typeof raw !== "object") {
        return [];
      }
      const event = progressEvent(raw as Record<string, unknown>);
      const total = optionalInt(totals[stage]);
      let completed = Math.max(distinctFiles(completedFiles[stage]), asInt(completedCounts[stage]));
      if (TERMINAL_SUCCESS.has(event.status) && !event.fileId && total != null) {
        completed = total;
      }
      return [
        {
          stage,
          status: event.status,
          completed,
          total,
          elapsedMs: asInt(elapsed[stage]),
          message: event.message,
        },
      ];
    });
}

function recordFile(files: Record<string, unknown>, stage: string, fileId: string): void {
  const seen = asRecord(files[stage]);
  seen[fileId] = true;
  files[stage] = seen;
}

function forgetFile(files: Record<string, unknown>, stage: string, fileId: string): void {
  const seen = files[stage];
  if (seen && typeof seen === "object") {
    delete (seen as Record<string, unknown>)[fileId];
  }
}

function distinctFiles(value: unknown): number {
  return value && typeof value === "object" ? Object.keys(value).length : 0;
}

function stageFileCount(summary: Record<string, unknown>, aggregate: string, stage: string): number {
  return distinctFiles(readonlyMapping(summary, aggregate)[stage]);
}

function mapping(summary: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = summary[key];
  if (value && typeof value === "object") {
    return value as Record<string, unknown>;
  }
  const created: Record<string, unknown> = {};
  summary[key] = created;
  return created;
}

function readonlyMapping(summary: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = summary[key];
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function eventRecord(event: ProgressEvent): Record<string, unknown> {
  return {
    timestamp: event.timestamp,
    stage: event.stage,
    status: event.status,
    file_id: event.fileId,
    message: event.message,
    elapsed_ms: event.elapsedMs,
    counts: event.counts,
    metadata: event.metadata,
  };
}

async function tailLines(file: string, limit: number): Promise<string[]> {
  const handle = await open(file, "r");
  try {
    const info = await handle.stat();
    let position = info.size;
    const chunks: Buffer[] = [];
    let newlineCount = 0;
    while (position > 0 && newlineCount <= limit) {
      const blockSize = Math.min(TAIL_BLOCK_BYTES, position);
      position -= blockSize;
      const buffer = Buffer.alloc(blockSize);
      await handle.read(buffer, 0, blockSize, position);
      chunks.push(buffer);
      newlineCount += buffer.reduce((count, byte) => count + (byte === 10 ? 1 : 0), 0);
    }
    return Buffer.concat(chunks.reverse()).toString("utf8").split(/\r?\n/).slice(-limit);
  } finally {
    await handle.close();
  }
}

function asInt(value: unknown): number {
  return typeof value === "number" && value >= 0 ? value : 0;
}

function optionalInt(value: unknown): number | null {
  return typeof value === "number" && value >= 0 ? value : null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? { ...(value as Record<string, unknown>) } : {};
}
