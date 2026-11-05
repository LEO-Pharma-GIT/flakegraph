import path from "node:path";
import { mkdir, rm } from "node:fs/promises";
import { Effect } from "effect";
import { documentStatusesFromEvents, type DocumentStatus } from "../documents";
import { stringify as stringifyYaml } from "yaml";
import { listRunRecords, readJsonFile, readRunRecord, renameGraph as persistGraphName, runDirectory, runRecordExists, snapshotFromRecord, writeRunRecord, graphName, atomicWriteJson, cloneFieldsFromRequest } from "../catalog";
import { buildRunConfig, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import { loadLocalGraph } from "../graph";
import { unidentifiedViewer } from "../identity";
import {
  SNOWFLAKE_CAPABILITIES,
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
  validateGraphName,
} from "../protocol/schema";
import { ControlPlaneError, fromCause, invalid, notFound, notSupported } from "../protocol/errors";
import type { ControlPlane, GraphLoadOptions } from "../protocol/runtime";
import { listLocalObjects, REMOTE_SUMMARY_LIMIT, summarizeListing, summarizeLocalObjects } from "../sources";

interface SnowflakeGraphRecord {
  graphId: string;
  owner: string | null;
  location: string;
  name?: string | null;
  deleted?: boolean;
}

interface SnowflakeStore {
  graphs: SnowflakeGraphRecord[];
  jobs: Array<Record<string, unknown>>;
}

/**
 * A rehearsal of the Snowflake runtime for a console that is not hosted there:
 * jobs and graphs in a JSON store, with no account behind them. It knows
 * nothing about access, so `controlledPlane` wraps it like any other runtime.
 */
export class SnowflakeStubRuntime implements ControlPlane {
  readonly runtime = "snowflake" as const;
  readonly capabilities = new Set<Capability>(SNOWFLAKE_CAPABILITIES);
  private currentViewer: Viewer = unidentifiedViewer();

  constructor(
    readonly repositoryRoot: string,
    readonly stateRoot: string,
    readonly stubbed: boolean,
    viewer?: Viewer,
  ) {
    if (viewer) {
      this.currentViewer = viewer;
    }
  }

  withViewer(viewer: Viewer): SnowflakeStubRuntime {
    return new SnowflakeStubRuntime(this.repositoryRoot, this.stateRoot, this.stubbed, viewer);
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return Effect.succeed(this.currentViewer);
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const kind = String(source.kind ?? "snowflake_stage");
        if (kind === "local" || kind === "upload" || kind === "local_path") {
          return listLocalObjects(String(source.path ?? ""), limit);
        }
        const store = await this.store();
        const prefix = String(source.prefix ?? "");
        return store.jobs
          .filter((job) => String(job.stagePrefix ?? "").startsWith(prefix) || !prefix)
          .slice(0, limit)
          .map((job) => ({
            uri: String(job.uri ?? job.path ?? job.graphId ?? "stage://object"),
            name: String(job.name ?? job.uri ?? "object"),
            sizeBytes: Number(job.sizeBytes ?? 0),
            modifiedAt: String(job.modifiedAt ?? "") || null,
            checksum: null,
          }));
      },
      catch: (cause) => fromCause(cause, "Unable to list Snowflake stage objects"),
    });
  }

  summarizeSource(source: Record<string, unknown>): Effect.Effect<SourceSummary, ControlPlaneError> {
    const kind = String(source.kind ?? "snowflake_stage");
    if (kind === "local" || kind === "upload" || kind === "local_path") {
      return Effect.tryPromise({
        try: () => summarizeLocalObjects(String(source.path ?? "")),
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
        await writeRunConfig(request, path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml"));
        if (request.embedding.dimension && request.embedding.dimension < 1) {
          return { ok: false, errors: ["Embedding dimension must be positive"], warnings: [], checks: [] };
        }
        return {
          ok: true,
          errors: [],
          warnings: this.stubbed ? ["Snowflake App Runtime stub accepted the job."] : [],
          checks: [{ name: "stage", ok: true }],
        };
      },
      catch: (cause) => fromCause(cause, "Snowflake preflight failed"),
    });
  }

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (request.graphName) {
          await persistGraphName(this.stateRoot, request.graphId, validateGraphName(request.graphName));
        }
        const directory = runDirectory(this.stateRoot, request.jobId);
        const configPath = await writeRunConfig(request, path.join(directory, "config.yaml"));
        const startedAt = new Date().toISOString();
        await writeRunRecord(directory, {
          runId: request.jobId,
          graphId: request.graphId,
          graphName: request.graphName,
          status: "queued",
          runtime: "snowflake",
          startedAt,
          updatedAt: startedAt,
          outputPath: request.output.workspacePath,
          configPath,
          storageKind: "snowflake",
          storageLocation: request.output.snowflake
            ? `${request.output.snowflake.database}.${request.output.snowflake.schema}`
            : request.output.workspacePath,
          ...cloneFieldsFromRequest(request),
          owner: this.currentViewer.userName || null,
        });
        const store = await this.store();
        store.jobs.unshift({
          jobId: request.jobId,
          graphId: request.graphId,
          status: "queued",
          owner: this.currentViewer.userName || null,
        });
        if (!store.graphs.some((graph) => graph.graphId === request.graphId)) {
          store.graphs.push({
            graphId: request.graphId,
            owner: this.currentViewer.userName || null,
            location: request.output.workspacePath,
            name: request.graphName,
          });
        }
        await this.writeStore(store);
        return Effect.runPromise(this.getRun(request.jobId));
      },
      catch: (cause) => fromCause(cause, "Unable to submit Snowflake job"),
    });
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const rows = await listRunRecords(this.stateRoot, limit);
        const store = await this.store();
        const visible = [];
        for (const { record } of rows) {
          if (String(record.runtime || "") !== "snowflake") {
            continue;
          }
          visible.push(
            snapshotFromRecord(record, {
              graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
              raw: {
                owner: record.owner ?? store.graphs.find((graph) => graph.graphId === record.graphId)?.owner ?? null,
              },
            }),
          );
        }
        return visible;
      },
      catch: (cause) => fromCause(cause, "Unable to list Snowflake jobs"),
    });
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        if (!runRecordExists(directory)) {
          throw notFound(`Unknown Snowflake job: ${runId}`);
        }
        const record = await readRunRecord(directory);
        const store = await this.store();
        return snapshotFromRecord(record, {
          graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
          raw: {
            owner: record.owner ?? store.graphs.find((graph) => graph.graphId === record.graphId)?.owner ?? null,
          },
        });
      },
      catch: (cause) => fromCause(cause, "Unable to load Snowflake job"),
    });
  }

  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    return Effect.map(this.getRun(runId), (snapshot) => documentStatusesFromEvents(snapshot.events));
  }

  versions(_graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return Effect.succeed([]);
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.mutateJob(runId, "cancelled");
  }

  retry(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.mutateJob(runId, "queued");
  }

  recover(_runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.fail(notSupported("Infrastructure recovery is available only for Kubernetes runs"));
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const name = await persistGraphName(this.stateRoot, graphId, displayName);
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (graph) {
          graph.name = name;
          await this.writeStore(store);
        }
        return name;
      },
      catch: (cause) => fromCause(cause, "Unable to rename Snowflake graph"),
    });
  }

  loadGraph(location: string, _graphId?: string | null, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        return loadLocalGraph(path.isAbsolute(location) ? location : path.resolve(this.repositoryRoot, location), options);
      },
      catch: (cause) => fromCause(cause, "Unable to load Snowflake graph"),
    });
  }

  graphDirectory(snapshot: RunSnapshot): string | null {
    if (!snapshot.outputPath) {
      return null;
    }
    return path.isAbsolute(snapshot.outputPath) ? snapshot.outputPath : path.resolve(this.repositoryRoot, snapshot.outputPath);
  }

  loadRunGraph(snapshot: RunSnapshot, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    if (!snapshot.outputPath) {
      return Effect.fail(invalid(`Job ${snapshot.runId} does not record an output location`));
    }
    return this.loadGraph(snapshot.outputPath, snapshot.graphId, options);
  }

  cluster(_namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.succeed(null);
  }

  nodeAssignments(): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.succeed([]);
  }


  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (!graph) {
          throw notFound(`Unknown graph: ${graphId}`);
        }
        graph.deleted = true;
        await this.writeStore(store);
        const runs = await listRunRecords(this.stateRoot, 500);
        let removed = 0;
        for (const { record } of runs) {
          if (record.graphId === graphId) {
            await rm(runDirectory(this.stateRoot, record.runId), { recursive: true, force: true });
            removed += 1;
          }
        }
        return { KG_GRAPH: 1, runs: removed };
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

  private mutateJob(runId: string, status: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const snapshot = await Effect.runPromise(this.getRun(runId));
        await writeRunRecord(runDirectory(this.stateRoot, runId), {
          status,
          error: status === "queued" ? null : snapshot.error,
          cancellationRequestedAt: status === "cancelled" ? new Date().toISOString() : null,
        });
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to update Snowflake job"),
    });
  }

  private async store(): Promise<SnowflakeStore> {
    const file = this.storeFile();
    const current = await readJsonFile<SnowflakeStore>(file);
    return current ?? { graphs: [], jobs: [] };
  }

  private async writeStore(store: SnowflakeStore): Promise<void> {
    await mkdir(path.dirname(this.storeFile()), { recursive: true });
    await atomicWriteJson(this.storeFile(), store);
  }

  private storeFile(): string {
    return path.join(this.stateRoot, "snowflake", "store.json");
  }
}

export function createSnowflakeStubRuntime(viewer?: Viewer): SnowflakeStubRuntime {
  const env = appEnv();
  return new SnowflakeStubRuntime(env.repositoryRoot, env.stateRoot, true, viewer);
}

