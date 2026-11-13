import { mkdir, readdir, readFile, realpath, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { validateGraphName, type IngestionRequest, type RunSnapshot, type StorageKind } from "./protocol/schema";
import { assertSafeId } from "./confine";

/**
 * Runs the catalog does not list: revision attempts the planner refused,
 * which it may already have recorded as cancelled rows. They are not
 * versions of their graph. (Removing a graph is a delete, not a hiding.)
 */
const WITHDRAWN_RUNS_FILE = "withdrawn-runs.json";
const GRAPH_NAMES_FILE = "graph-names.json";
const graphNameQueues = new Map<string, Promise<void>>();

/** Serialize the complete shared name-map update, including graph deletion. */
export async function changeGraphNames(stateRoot: string, change: (names: Record<string, string>) => void): Promise<void> {
  await mkdir(stateRoot, { recursive: true });
  const file = path.join(await realpath(stateRoot), GRAPH_NAMES_FILE);
  const next = (graphNameQueues.get(file) ?? Promise.resolve()).catch(() => undefined).then(async () => {
    const names = await readGraphNames(stateRoot);
    change(names);
    await atomicWriteJson(file, names);
  });
  graphNameQueues.set(file, next);
  try { await next; } finally {
    if (graphNameQueues.get(file) === next) graphNameQueues.delete(file);
  }
}

export interface CatalogRecord {
  runId: string;
  graphId: string;
  graphName?: string | null;
  status: string;
  runtime: string;
  startedAt?: string | null;
  updatedAt?: string | null;
  outputPath?: string | null;
  configPath?: string | null;
  storageKind?: StorageKind | null;
  storageLocation?: string | null;
  error?: string | null;
  pid?: number | null;
  processBirth?: string | null;
  cancellationRequestedAt?: string | null;
  documentsTotal?: number | null;
  documentsCompleted?: number | null;
  documentsFailed?: number | null;
  sourceKind?: string | null;
  sourcePath?: string | null;
  source?: Record<string, unknown> | null;
  ocrProvider?: string | null;
  llmProvider?: string | null;
  embeddingProvider?: string | null;
  owner?: string | null;
  /** The run this one revised, when it is a new version of that run's graph. */
  baseRunId?: string | null;
  [key: string]: unknown;
}

export function cloneFieldsFromRequest(request: IngestionRequest) {
  const source = request.source as Record<string, unknown>;
  return {
    sourceKind: request.sourceKind,
    sourcePath: String(source.path ?? source.prefix ?? source.stage ?? ""),
    // The whole source, so a bucket or container can be run again as it
    // was named: kind, bucket, prefix, endpoint. Credentials never travel
    // in it - they come from the environment.
    source: { ...source },
    ocrProvider: request.ocr.provider,
    llmProvider: request.llm.provider,
    embeddingProvider: request.embedding.provider,
  };
}

/**
 * Where a run's record, configuration and events live. The id is checked
 * here as well as at the API edge: every runtime names the directory
 * through this one function, so nothing a caller sends can name a
 * directory anywhere else.
 */
export function runDirectory(stateRoot: string, runId: string): string {
  return path.join(stateRoot, "runs", assertSafeId(runId, "runId"));
}

export async function writeRunRecord(
  directory: string,
  values: Partial<CatalogRecord> & Record<string, unknown>,
): Promise<CatalogRecord> {
  await mkdir(directory, { recursive: true });
  const current = await readRunRecord(directory);
  const overrides = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
  const payload: CatalogRecord = {
    ...current,
    ...overrides,
    runId: String(overrides.runId ?? current.runId ?? path.basename(directory)),
    graphId: String(overrides.graphId ?? current.graphId ?? path.basename(directory)),
    status: String(overrides.status ?? current.status ?? "unknown"),
    runtime: String(overrides.runtime ?? current.runtime ?? "local"),
  };
  if (payload.cancellationRequestedAt) {
    payload.status = "cancelled";
  }
  await atomicWriteJson(path.join(directory, "run.json"), payload);
  return payload;
}

export async function readRunRecord(directory: string): Promise<CatalogRecord> {
  const file = path.join(directory, "run.json");
  const value = await readJsonFile<CatalogRecord>(file);
  if (!value) {
    return emptyRecord(path.basename(directory));
  }
  return normalizeRecord(value);
}

export async function listRunRecords(
  stateRoot: string,
  limit = 100,
  options: { includeWithdrawn?: boolean } = {},
): Promise<Array<{ directory: string; record: CatalogRecord }>> {
  const runsRoot = path.join(stateRoot, "runs");
  if (!existsSync(runsRoot)) {
    return [];
  }
  const withdrawn = options.includeWithdrawn ? new Set<string>() : await withdrawnRunIds(stateRoot);
  const entries = await readdir(runsRoot, { withFileTypes: true });
  // A run's directory is named by its id, so withdrawn runs are left out before
  // the list is cut to size rather than taking places in it.
  const directories = entries
    .filter((entry) => entry.isDirectory() && !withdrawn.has(entry.name))
    .map((entry) => path.join(runsRoot, entry.name));
  const withTimes = await Promise.all(
    directories.map(async (directory) => {
      const file = path.join(directory, "run.json");
      try {
        const info = await stat(file);
        return { directory, mtime: info.mtimeMs };
      } catch {
        return null;
      }
    }),
  );
  return (
    await Promise.all(
      withTimes
        .filter((entry): entry is { directory: string; mtime: number } => entry !== null)
        .sort((left, right) => right.mtime - left.mtime)
        .slice(0, limit)
        .map(async ({ directory }) => ({ directory, record: await readRunRecord(directory) })),
    )
  ).filter((item) => Boolean(item.record.runId) && !withdrawn.has(item.record.runId));
}

export async function withdrawnRunIds(stateRoot: string): Promise<Set<string>> {
  const value = await readJsonFile<unknown>(path.join(stateRoot, WITHDRAWN_RUNS_FILE));
  if (!Array.isArray(value)) {
    return new Set();
  }
  return new Set(value.map((item) => String(item)).filter(Boolean));
}

/** Leave a refused revision attempt out of the catalog. */
export async function withdrawRun(stateRoot: string, runId: string): Promise<void> {
  const normalized = normalizeRunIds([runId]);
  await changeWithdrawn(stateRoot, (withdrawn) => normalized.forEach((id) => withdrawn.add(id)));
}

/** Forget withdrawn runs that no longer exist, once their graph is deleted. */
export async function unwithdrawRuns(stateRoot: string, runIds: readonly string[]): Promise<void> {
  if (runIds.length === 0) {
    return;
  }
  const normalized = normalizeRunIds(runIds);
  await changeWithdrawn(stateRoot, (withdrawn) => normalized.forEach((id) => withdrawn.delete(id)));
}

function normalizeRunIds(runIds: readonly string[]): string[] {
  const normalized = [...new Set(runIds.map((runId) => runId.trim()).filter(Boolean))];
  if (normalized.length === 0) {
    throw new Error("run ID must not be empty");
  }
  return normalized;
}

const withdrawnQueues = new Map<string, Promise<unknown>>();

/** One change to the withdrawn list at a time, so concurrent changes cannot lose each other. */
async function changeWithdrawn(stateRoot: string, change: (withdrawn: Set<string>) => void): Promise<void> {
  const file = path.join(stateRoot, WITHDRAWN_RUNS_FILE);
  const next = (withdrawnQueues.get(file) ?? Promise.resolve())
    .catch(() => undefined)
    .then(async () => {
      await mkdir(stateRoot, { recursive: true });
      const withdrawn = await withdrawnRunIds(stateRoot);
      change(withdrawn);
      await atomicWriteJson(file, [...withdrawn].sort());
    });
  withdrawnQueues.set(file, next);
  try {
    await next;
  } finally {
    if (withdrawnQueues.get(file) === next) {
      withdrawnQueues.delete(file);
    }
  }
}

export function runRecordExists(directory: string): boolean {
  return existsSync(path.join(directory, "run.json"));
}

export async function readGraphNames(stateRoot: string): Promise<Record<string, string>> {
  const value = await readJsonFile<Record<string, unknown>>(path.join(stateRoot, GRAPH_NAMES_FILE));
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) => (typeof item === "string" ? [[key, item] as const] : [])),
  );
}

export async function graphName(stateRoot: string, graphId: string): Promise<string | null> {
  const names = await readGraphNames(stateRoot);
  return names[graphId] ?? null;
}

export async function renameGraph(
  stateRoot: string,
  graphId: string,
  displayName: string,
): Promise<string> {
  const normalizedId = graphId.trim();
  if (!normalizedId) {
    throw new Error("graph ID must not be empty");
  }
  const normalizedName = validateGraphName(displayName);
  await mkdir(stateRoot, { recursive: true });
  await changeGraphNames(stateRoot, (names) => { names[normalizedId] = normalizedName; });
  const rows = await listRunRecords(stateRoot, 1_000, { includeWithdrawn: true });
  for (const { directory, record } of rows) {
    if (record.graphId === normalizedId) {
      await writeRunRecord(directory, { graphName: normalizedName });
    }
  }
  return normalizedName;
}

export async function atomicWriteJson(file: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${randomUUID()}.tmp`);
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, file);
}

export async function readJsonFile<T>(file: string): Promise<T | null> {
  try {
    const text = await readFile(file, "utf8");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

export function snapshotFromRecord(
  record: CatalogRecord,
  extras: Partial<RunSnapshot> = {},
): RunSnapshot {
  return {
    runId: record.runId,
    graphId: record.graphId,
    status: record.status,
    startedAt: record.startedAt ?? null,
    updatedAt: record.updatedAt ?? record.startedAt ?? null,
    graphName: (record.graphName as string | null | undefined) ?? extras.graphName ?? null,
    stages: extras.stages ?? [],
    events: extras.events ?? [],
    documentsTotal: record.documentsTotal ?? extras.documentsTotal ?? null,
    documentsCompleted: record.documentsCompleted ?? extras.documentsCompleted ?? 0,
    documentsFailed: record.documentsFailed ?? extras.documentsFailed ?? 0,
    outputPath: record.outputPath ?? null,
    storageKind: record.storageKind ?? "local_files",
    storageLocation: record.storageLocation ?? record.outputPath ?? null,
    warnings: extras.warnings ?? [],
    error: record.error ?? extras.error ?? null,
    listingWarning: extras.listingWarning ?? null,
    raw: {
      ...((extras.raw as Record<string, unknown> | undefined) ?? {}),
      config_path: record.configPath ?? null,
      nodeCount: record.nodeCount ?? (extras.raw as Record<string, unknown> | undefined)?.nodeCount ?? null,
      sourceKind: record.sourceKind ?? null,
      sourcePath: record.sourcePath ?? null,
      source: record.source ?? (extras.raw as Record<string, unknown> | undefined)?.source ?? null,
      ocrProvider: record.ocrProvider ?? null,
      llmProvider: record.llmProvider ?? null,
      embeddingProvider: record.embeddingProvider ?? null,
      owner: record.owner ?? (extras.raw as Record<string, unknown> | undefined)?.owner ?? null,
      baseRunId: record.baseRunId ?? (extras.raw as Record<string, unknown> | undefined)?.baseRunId ?? null,
    },
  };
}

function emptyRecord(runId: string): CatalogRecord {
  return { runId, graphId: runId, status: "unknown", runtime: "local" };
}

function normalizeRecord(value: CatalogRecord): CatalogRecord {
  return {
    ...value,
    graphName: value.graphName ?? null,
    startedAt: value.startedAt ?? null,
    updatedAt: value.updatedAt ?? null,
    outputPath: value.outputPath ?? null,
    configPath: value.configPath ?? null,
    storageKind: value.storageKind ?? "local_files",
    storageLocation: value.storageLocation ?? null,
    error: value.error ?? null,
    pid: value.pid ?? null,
    cancellationRequestedAt: value.cancellationRequestedAt ?? null,
    documentsTotal: value.documentsTotal ?? null,
    documentsCompleted: value.documentsCompleted ?? 0,
    documentsFailed: value.documentsFailed ?? 0,
    sourceKind: value.sourceKind ?? null,
    sourcePath: value.sourcePath ?? null,
    source: value.source ?? null,
    ocrProvider: value.ocrProvider ?? null,
    llmProvider: value.llmProvider ?? null,
    embeddingProvider: value.embeddingProvider ?? null,
    owner: value.owner ?? null,
  };
}
