import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { Effect } from "effect";
import { documentStatusesFromEvents, type DocumentStatus } from "../documents";
import {
  graphName,
  listRunRecords,
  readRunRecord,
  renameGraph as persistGraphName,
  runDirectory,
  snapshotFromRecord,
  writeRunRecord,
  cloneFieldsFromRequest,
} from "../catalog";
import { lastJsonObject, runFlakegraph, spawnFlakegraph } from "../cli";
import { buildRunConfig, environmentForRequest, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import { graphArtifactsExist, loadLocalGraph } from "../graph";
import { unidentifiedViewer } from "../identity";
import { readDocumentEvents, readLocalProgress } from "../progress";
import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  LOCAL_CAPABILITIES,
  type Capability,
  type ClusterSnapshot,
  type GraphDataset,
  type IngestionRequest,
  type NodeWorkAssignment,
  type PreflightResult,
  type GraphVersion,
  type RunSnapshot,
  type SourceObject,
  type SourceSummary,
  type Viewer,
  isActiveStatus,
  isSuccessStatus,
  validateGraphName,
} from "../protocol/schema";
import {
  ControlPlaneError,
  fromCause,
  invalid,
  notFound,
  notSupported,
} from "../protocol/errors";
import type { ControlPlane, GraphLoadOptions } from "../protocol/runtime";
import { listLocalObjects, listRemoteObjects, REMOTE_SUMMARY_LIMIT, summarizeListing, summarizeLocalObjects } from "../sources";
import { pidIsRunning, processRegistry, processBirth, stopLocalWorker } from "./process-registry";
import { stringify as stringifyYaml } from "yaml";
import { catalogWriterPrincipal } from "../workspace";

const exportsInFlight = new Map<string, Promise<void>>();
const SNOWFLAKE_EXPORT_TIMEOUT_MS = 60 * 60 * 1000;

export class LocalRuntime implements ControlPlane {
  readonly runtime = "local" as const;
  readonly capabilities = new Set<Capability>(LOCAL_CAPABILITIES);

  constructor(
    readonly repositoryRoot: string,
    readonly stateRoot: string,
    /** Who is asking, when a gate or key identified them; the owner of what they submit. */
    readonly currentViewer: Viewer | null = null,
    private readonly sourceProfiles: () => Promise<readonly Record<string, unknown>[]> = async () => [],
  ) {}

  /** The principal recorded as owner: the identified viewer, else the workspace's assumed identity. */
  async writerPrincipal(): Promise<string | null> {
    const name = this.currentViewer?.userName?.trim();
    return name ? name.toUpperCase() : catalogWriterPrincipal(this.stateRoot);
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return Effect.succeed(unidentifiedViewer());
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const kind = String(source.kind ?? "local");
        if (kind === "local" || kind === "upload" || kind === "local_path") {
          const input = this.resolvePath(String(source.path ?? ""));
          return listLocalObjects(input, limit);
        }
        if (kind === "s3" || kind === "azure_blob") {
          return listRemoteObjects(kind, source, {
            cwd: this.repositoryRoot,
            stateRoot: this.stateRoot,
            operatorProfiles: await this.sourceProfiles(),
            limit,
          });
        }
        throw invalid(`Source browser is not available for ${kind} sources on this runtime`);
      },
      catch: (cause) => fromCause(cause, "Unable to list source objects"),
    });
  }

  summarizeSource(source: Record<string, unknown>): Effect.Effect<SourceSummary, ControlPlaneError> {
    const kind = String(source.kind ?? "local");
    if (kind === "local" || kind === "upload" || kind === "local_path") {
      return Effect.tryPromise({
        try: () => summarizeLocalObjects(this.resolvePath(String(source.path ?? ""))),
        catch: (cause) => fromCause(cause, "Unable to count source objects"),
      });
    }
    return this.listSourceObjects(source, REMOTE_SUMMARY_LIMIT).pipe(
      Effect.map((objects) => summarizeListing(objects, REMOTE_SUMMARY_LIMIT)),
    );
  }

  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const configPath = path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml");
        await writeRunConfig(request, configPath);
        const result = await runFlakegraph(["preflight", "--config", configPath], {
          cwd: this.repositoryRoot,
          env: environmentForRequest(request),
        });
        const payload = lastJsonObject(result.stdout) ?? lastJsonObject(result.stderr);
        if (payload) {
          const ok = payload.ok !== false && result.exitCode === 0;
          return {
            ok,
            errors: asStringArray(payload.errors) || (ok ? [] : [result.stderr || "Preflight failed"]),
            warnings: asStringArray(payload.warnings),
            checks: Array.isArray(payload.checks) ? (payload.checks as Record<string, unknown>[]) : [],
          };
        }
        return {
          ok: false,
          errors: [result.stderr.trim() || result.stdout.trim() || "Preflight returned no JSON result"],
          warnings: [],
          checks: [],
        };
      },
      catch: (cause) => fromCause(cause, "Preflight failed"),
    });
  }

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (request.graphName) {
          await persistGraphName(this.stateRoot, request.graphId, validateGraphName(request.graphName));
        }
        const directory = runDirectory(this.stateRoot, request.jobId);
        await mkdir(directory, { recursive: true });
        const configPath = await writeRunConfig(request, path.join(directory, "config.yaml"));
        const eventsPath = path.join(directory, "events.jsonl");
        const stdoutPath = path.join(directory, "stdout.log");
        const startedAt = new Date().toISOString();
        const child = spawnFlakegraph(
          ["worker", "--config", configPath, "--progress", "json"],
          {
            cwd: this.repositoryRoot,
            env: environmentForRequest(request),
            stdoutPath,
            stderrPath: eventsPath,
          },
        );
        await writeRunRecord(directory, {
          runId: request.jobId,
          graphId: request.graphId,
          graphName: request.graphName,
          status: "running",
          runtime: "local",
          startedAt,
          updatedAt: startedAt,
          outputPath: request.output.workspacePath,
          configPath,
          storageKind: request.output.kind,
          storageLocation:
            request.output.kind === "snowflake" && request.output.snowflake
              ? `${request.output.snowflake.database}.${request.output.snowflake.schema}`
              : request.output.workspacePath,
          pid: child.pid ?? null,
          processBirth: await processBirth(child.pid),
          owner: await this.writerPrincipal(),
          ...cloneFieldsFromRequest(request),
        });
        processRegistry.start({
          runId: request.jobId,
          graphId: request.graphId,
          process: child,
          eventsPath,
          outputPath: request.output.workspacePath,
          stdoutPath,
          configPath,
          startedAt,
          storageKind: request.output.kind,
          storageLocation:
            request.output.kind === "snowflake" && request.output.snowflake
              ? `${request.output.snowflake.database}.${request.output.snowflake.schema}`
              : request.output.workspacePath,
        });
        return this.snapshotFromLive(request.jobId);
      },
      catch: (cause) => fromCause(cause, "Unable to submit run"),
    });
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const rows = await listRunRecords(this.stateRoot, limit);
        const snapshots: RunSnapshot[] = [];
        for (const { directory, record } of rows) {
          if (String(record.runtime || "local") !== "local") {
            continue;
          }
          const managed = processRegistry.get(record.runId);
          if (managed && (managed.process.exitCode === null || isActiveStatus(String(record.status)))) {
            snapshots.push(await this.snapshotFromLive(record.runId));
          } else {
            snapshots.push(await this.catalogSnapshot(directory, record, false));
          }
        }
        return snapshots;
      },
      catch: (cause) => fromCause(cause, "Unable to list runs"),
    });
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (processRegistry.get(runId)) {
          return this.snapshotFromLive(runId);
        }
        const directory = runDirectory(this.stateRoot, runId);
        const record = await readRunRecord(directory);
        if (!record.runId || record.status === "unknown") {
          throw notFound(`Unknown local run: ${runId}`);
        }
        // The listing shows only local runs; opening one another runtime owns
        // would read that runtime's record through this one's rules.
        if (String(record.runtime || "local") !== "local") {
          throw notFound(`Run ${runId} belongs to the ${record.runtime} runtime`);
        }
        return this.catalogSnapshot(directory, record, true);
      },
      catch: (cause) => fromCause(cause, "Unable to load run"),
    });
  }

  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    // From the whole events file, not the snapshot's tail: a corpus of
    // hundreds of documents outruns any tail long before it finishes.
    return Effect.flatMap(this.getRun(runId), () =>
      Effect.tryPromise({
        try: async () =>
          documentStatusesFromEvents(await readDocumentEvents(path.join(runDirectory(this.stateRoot, runId), "events.jsonl"))),
        catch: (cause) => fromCause(cause, `Unable to read documents for ${runId}`),
      }),
    );
  }

  versions(_graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return Effect.succeed([]);
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        const managed = processRegistry.get(runId);
        const record = await readRunRecord(directory);
        if (!record.runId) {
          throw notFound(`Unknown local run: ${runId}`);
        }
        await stopLocalWorker(record, managed?.process);
        await writeRunRecord(directory, {
          status: "cancelled",
          pid: null,
          processBirth: null,
          cancellationRequestedAt: new Date().toISOString(),
        });
        return this.catalogSnapshot(directory, await readRunRecord(directory), true);
      },
      catch: (cause) => fromCause(cause, "Unable to cancel run"),
    });
  }

  retry(_runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.fail(notSupported("Retry is available only for durable distributed runs"));
  }

  recover(_runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.fail(notSupported("Infrastructure recovery is available only for Kubernetes runs"));
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: () => persistGraphName(this.stateRoot, graphId, displayName),
      catch: (cause) => fromCause(cause, "Unable to rename graph"),
    });
  }

  loadGraph(location: string, _graphId?: string | null, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return Effect.tryPromise({
      try: () => loadLocalGraph(this.artifactDirectory(location), options),
      catch: (cause) => fromCause(cause, "Unable to load graph"),
    });
  }

  graphDirectory(snapshot: RunSnapshot): string | null {
    if (snapshot.storageKind === "snowflake") {
      return path.join(this.stateRoot, "artifacts", snapshot.runId);
    }
    return snapshot.outputPath ? this.artifactDirectory(snapshot.outputPath) : null;
  }

  loadRunGraph(snapshot: RunSnapshot, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    if (snapshot.storageKind === "snowflake") {
      return Effect.tryPromise({
        try: async () => {
          const directory = this.graphDirectory(snapshot)!;
          if (!graphArtifactsExist(directory) || !existsSync(path.join(directory, "graph_metrics.json")) ||
              !await artifactsBelongToRun(directory, snapshot.runId, snapshot.graphId)) {
            // Only a finished run's Snowflake graph is complete enough to keep.
            if (!isSuccessStatus(snapshot.status)) {
              throw invalid(`Run ${snapshot.runId} has not finished writing its Snowflake graph`);
            }
            let pending = exportsInFlight.get(directory);
            if (!pending) {
              pending = this.exportSnowflakeGraph(snapshot, directory).finally(() => exportsInFlight.delete(directory));
              exportsInFlight.set(directory, pending);
            }
            await pending;
          }
          if (!await artifactsBelongToRun(directory, snapshot.runId, snapshot.graphId)) {
            throw invalid(`Exported diagnostics do not belong to run ${snapshot.runId}`);
          }
          const dataset = await loadLocalGraph(directory, options);
          if (dataset.runReport.job_id !== snapshot.runId || dataset.graphId !== snapshot.graphId) {
            throw invalid(`Artifacts for ${snapshot.runId} changed while being read`);
          }
          return dataset;
        },
        catch: (cause) => fromCause(cause, "Unable to load Snowflake graph"),
      });
    }
    if (!snapshot.outputPath) {
      return Effect.fail(invalid(`Run ${snapshot.runId} does not record an output directory`));
    }
    return Effect.tryPromise({
      try: async () => {
        const directory = this.artifactDirectory(snapshot.outputPath!);
        if (!await artifactsBelongToRun(directory, snapshot.runId, snapshot.graphId)) {
          throw invalid(`Artifacts for ${snapshot.runId} are unavailable or have been superseded by another run`);
        }
        const dataset = await loadLocalGraph(directory, options);
        if (dataset.runReport.job_id !== snapshot.runId || dataset.graphId !== snapshot.graphId) {
          throw invalid(`Artifacts for ${snapshot.runId} changed while being read`);
        }
        return dataset;
      },
      catch: (cause) => fromCause(cause, "Unable to load run artifacts"),
    });
  }

  private async exportSnowflakeGraph(snapshot: RunSnapshot, directory: string): Promise<void> {
    const record = await readRunRecord(runDirectory(this.stateRoot, snapshot.runId));
    if (!record.configPath) throw invalid(`Run ${snapshot.runId} does not record its Snowflake configuration`);
    const result = await runFlakegraph([
      "snowflake", "export", "--graph-id", snapshot.graphId, "--job-id", snapshot.runId, "--output", directory, "--config", record.configPath,
    ], { cwd: this.repositoryRoot, timeoutMs: SNOWFLAKE_EXPORT_TIMEOUT_MS });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr.trim() || `Exporting the graph of ${snapshot.runId} failed`);
    }
  }

  cluster(_namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.succeed(null);
  }

  nodeAssignments(
    _namespace: string,
    _nodeName: string,
  ): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.succeed([]);
  }


  /**
   * A local graph is its runs' records and the output they wrote under the
   * state root, which the caller removes with the rest of what the console
   * keeps; here the runs are only let go of by the process registry.
   */
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const runs = (await listRunRecords(this.stateRoot, Number.MAX_SAFE_INTEGER, { includeWithdrawn: true }))
          .map((entry) => entry.record)
          .filter((record) => record.graphId === graphId && record.runtime === "local");
        for (const record of runs) {
          processRegistry.forget(record.runId);
        }
        return { runs: runs.length };
      },
      catch: (cause) => fromCause(cause, "Unable to delete graph"),
    });
  }

  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => stringifyYaml(redactedConfig(await buildRunConfig(request)), { sortMapEntries: false }),
      catch: (cause) => fromCause(cause, "Unable to preview configuration"),
    });
  }

  artifactDirectory(location: string): string {
    return path.isAbsolute(location) ? location : path.resolve(this.repositoryRoot, location);
  }

  private async snapshotFromLive(runId: string): Promise<RunSnapshot> {
    const managed = processRegistry.get(runId);
    const directory = runDirectory(this.stateRoot, runId);
    const record = await readRunRecord(directory);
    const progress = await readLocalProgress(path.join(directory, "events.jsonl"));
    const returnCode = managed?.process.exitCode ?? null;
    const signalCode = managed?.process.signalCode ?? null;
    let status = record.status;
    if (record.cancellationRequestedAt) {
      status = managed && returnCode === null && !signalCode ? "cancelling" : "cancelled";
    } else if (signalCode) {
      status = "failed";
    } else if (managed && returnCode === null) {
      status = "running";
    } else if (returnCode === 0) {
      status = "succeeded";
    } else if (returnCode != null) {
      status = "failed";
    }
    let snapshot: RunSnapshot = {
      runId,
      graphId: managed?.graphId ?? record.graphId,
      status,
      startedAt: managed?.startedAt ?? record.startedAt ?? null,
      updatedAt: progress.updatedAt ?? record.updatedAt ?? null,
      graphName: (await graphName(this.stateRoot, managed?.graphId ?? record.graphId)) ?? record.graphName ?? null,
      stages: progress.stages,
      events: progress.events.slice(-200),
      documentsTotal: progress.documentsTotal,
      documentsCompleted: progress.documentsCompleted,
      documentsFailed: progress.documentsFailed,
      outputPath: managed?.outputPath ?? record.outputPath ?? null,
      storageKind: managed?.storageKind ?? record.storageKind ?? "local_files",
      storageLocation: managed?.storageLocation ?? record.storageLocation ?? null,
      warnings: [],
      error: status === "failed" ? failureSummary(progress.events, signalCode ? `Worker terminated by ${signalCode}` : record.error ?? null) : record.error ?? null,
      listingWarning: null,
      raw: {
        config_path: managed?.configPath ?? record.configPath ?? null,
        sourceKind: record.sourceKind ?? null,
        sourcePath: record.sourcePath ?? null,
        ocrProvider: record.ocrProvider ?? null,
        llmProvider: record.llmProvider ?? null,
        embeddingProvider: record.embeddingProvider ?? null,
        owner: record.owner ?? null,
      },
    };
    if (isSuccessStatus(snapshot.status) && snapshot.storageKind === "local_files" && snapshot.outputPath &&
        !await artifactsBelongToRun(this.artifactDirectory(snapshot.outputPath), snapshot.runId, snapshot.graphId)) {
      snapshot = { ...snapshot, status: ARTIFACTS_UNAVAILABLE_STATUS };
    }
    await writeRunRecord(directory, {
      status: snapshot.status,
      updatedAt: snapshot.updatedAt,
      outputPath: snapshot.outputPath,
      error: snapshot.error,
      documentsTotal: snapshot.documentsTotal,
      documentsCompleted: snapshot.documentsCompleted,
      documentsFailed: snapshot.documentsFailed,
    });
    return snapshot;
  }

  private async catalogSnapshot(
    directory: string,
    record: Awaited<ReturnType<typeof readRunRecord>>,
    includeProgress: boolean,
  ): Promise<RunSnapshot> {
    const progress = includeProgress
      ? await readLocalProgress(path.join(directory, "events.jsonl"))
      : { events: [], stages: [], documentsTotal: null, documentsCompleted: 0, documentsFailed: 0, updatedAt: null };
    const outputPath = record.outputPath ?? null;
    const storageKind = record.storageKind ?? "local_files";
    let status = String(record.status || "unknown").toLowerCase();
    if (record.cancellationRequestedAt || status === "cancelled") {
      status = "cancelled";
    } else if (status === "failed") {
      status = "failed";
    } else if (status === "succeeded") {
      status = storageKind === "local_files" && outputPath &&
        !await artifactsBelongToRun(this.artifactDirectory(outputPath), record.runId, record.graphId)
        ? ARTIFACTS_UNAVAILABLE_STATUS : "succeeded";
    } else if (pidIsRunning(record.pid)) {
      status = "running";
    } else if (storageKind === "local_files" && outputPath && graphArtifactsExist(this.artifactDirectory(outputPath)) &&
      await artifactsBelongToRun(this.artifactDirectory(outputPath), record.runId, record.graphId)) {
      status = "succeeded";
    } else if (isActiveStatus(status)) {
      status = "interrupted";
    }
    if (
      status === "succeeded" &&
      storageKind === "local_files" &&
      outputPath &&
      !graphArtifactsExist(this.artifactDirectory(outputPath))
    ) {
      status = ARTIFACTS_UNAVAILABLE_STATUS;
    }
    const name = await graphName(this.stateRoot, record.graphId);
    return snapshotFromRecord(
      { ...record, status },
      {
        graphName: name ?? record.graphName ?? null,
        stages: progress.stages,
        events: progress.events.slice(-200),
        documentsTotal: includeProgress ? progress.documentsTotal : record.documentsTotal ?? null,
        documentsCompleted: includeProgress ? progress.documentsCompleted : record.documentsCompleted ?? 0,
        documentsFailed: includeProgress ? progress.documentsFailed : record.documentsFailed ?? 0,
        error: record.error ?? null,
      },
    );
  }

  private resolvePath(value: string): string {
    return path.isAbsolute(value) ? value : path.resolve(this.repositoryRoot, value);
  }
}

/** Shared graph output recovers a worker only when its committed report names that run. */
async function artifactsBelongToRun(directory: string, runId: string, graphId: string): Promise<boolean> {
  try {
    const report = JSON.parse(await readFile(path.join(directory, "run_report.json"), "utf8"));
    return report?.job_id === runId && report?.graph_id === graphId;
  } catch {
    return false;
  }
}

export function createLocalRuntime(viewer: Viewer | null = null): LocalRuntime {
  const env = appEnv();
  return new LocalRuntime(env.repositoryRoot, env.stateRoot, viewer);
}

function failureSummary(events: RunSnapshot["events"], fallback: string | null): string {
  const failed = [...events].reverse().find((event) => event.status === "failed" && event.message);
  return failed?.message || fallback || "Worker exited with a non-zero status";
}

function asStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (typeof value === "string" && value.trim()) {
    return [value];
  }
  return [];
}
