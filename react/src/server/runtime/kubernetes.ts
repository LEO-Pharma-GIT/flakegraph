import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { Effect } from "effect";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import {
  atomicWriteJson,
  readJsonFile,
  writeRunRecord,
  listRunRecords,
  readRunRecord,
  runDirectory,
  runRecordExists,
  withdrawnRunIds,
  withdrawRun,
  graphName,
  renameGraph as persistGraphName,
  snapshotFromRecord,
  cloneFieldsFromRequest,
} from "../catalog";
import { cliFailure, lastJsonObject, runFlakegraph } from "../cli";
import { buildRunConfig, environmentForRequest, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import { assertSafeId } from "../confine";
import {
  documentCountsByRun,
  documentTasksByRun,
  finalizerPayload,
  graphVersionsByGraph,
  leasedTasks,
  listPostgresRuns,
} from "../db/client";
import { documentStatusesFromEvents, documentStatusesFromTasks, withProvenance, type DocumentStatus } from "../documents";
import { readLocalProgress } from "../progress";
import {
  KUBERNETES_CAPABILITIES,
  type Capability,
  type ClusterSnapshot,
  type GraphDataset,
  type GraphVersion,
  type IngestionRequest,
  type NodeStatus,
  type NodeWorkAssignment,
  type PreflightResult,
  type RunSnapshot,
  isSuccessStatus,
  type SourceObject,
  type SourceSummary,
  type Viewer,
  type WorkloadStatus,
  validateGraphName,
} from "../protocol/schema";
import { ControlPlaneError, fromCause, invalid, notFound, notSupported } from "../protocol/errors";
import type { ControlPlane, GraphLoadOptions, SnowflakePublishTarget } from "../protocol/runtime";
import { LocalRuntime } from "./local";
import { graphArtifactsExist, loadLocalGraph } from "../graph";
import { readFleetProfile, type FleetProfile } from "../fleet";
import { snapshotFromStatus } from "../fleet-status";


export type { SnowflakePublishTarget } from "../protocol/runtime";

/** The `flakegraph distributed` command line that submits or revises this request's run. */
export function submitArguments(request: IngestionRequest, configPath: string): string[] {
  const revision = request.revision;
  const run = ["--config", configPath, "--run-id", request.jobId, "--priority", request.priority];
  return revision
    ? [
        "distributed",
        "revise",
        ...run,
        "--base-run",
        revision.baseRunId,
        ...revision.dropFileIds.flatMap((fileId) => ["--drop-file", fileId]),
        ...(revision.addDocuments ? [] : ["--no-add-documents"]),
        // Folder exclusions change how the kept documents are finalized, not
        // which documents the graph holds.
        ...(revision.folderExclusions ? ["--rebuild"] : []),
      ]
    : ["distributed", "submit", ...run];
}

/** Graph exports under way, by output directory, shared by every caller that needs the same graph. */
const exportsInFlight = new Map<string, Promise<void>>();

export class KubernetesRuntime implements ControlPlane {
  readonly runtime = "kubernetes" as const;
  readonly capabilities = new Set<Capability>(KUBERNETES_CAPABILITIES);
  private readonly local: LocalRuntime;

  constructor(
    readonly repositoryRoot: string,
    readonly stateRoot: string,
    readonly stubbed: boolean,
    viewer: Viewer | null = null,
  ) {
    this.local = new LocalRuntime(repositoryRoot, stateRoot, viewer, async () => {
      const fleet = this.stubbed ? null : await this.fleetProfile();
      return fleet ? [fleet.config] : [];
    });
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return this.local.viewer();
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return this.local.listSourceObjects(source, limit);
  }

  summarizeSource(source: Record<string, unknown>): Effect.Effect<SourceSummary, ControlPlaneError> {
    return this.local.summarizeSource(source);
  }

  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          const configPath = path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml");
          await writeRunConfig(request, configPath);
          return { ok: true, errors: [], warnings: ["Stub fleet accepted the run."], checks: [] };
        }
        const configPath = path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml");
        await writeRunConfig(request, configPath, await this.fleetProfile(), await this.local.writerPrincipal());
        const result = await runFlakegraph(
          ["fleet", "preflight", "--config", configPath, ...this.fleetArguments()],
          { cwd: this.repositoryRoot, env: environmentForRequest(request) },
        );
        const payload = lastJsonObject(result.stdout) ?? lastJsonObject(result.stderr);
        if (!payload) {
          return {
            ok: false,
            errors: [cliFailure(result.stderr, "Fleet preflight returned no JSON result")],
            warnings: [],
            checks: [],
          };
        }
        return {
          ok: payload.ok !== false && result.exitCode === 0,
          errors: asStrings(payload.errors),
          warnings: asStrings(payload.warnings),
          checks: Array.isArray(payload.checks) ? (payload.checks as Record<string, unknown>[]) : [],
        };
      },
      catch: (cause) => fromCause(cause, "Fleet preflight failed"),
    });
  }

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (request.graphName) {
          await persistGraphName(this.stateRoot, request.graphId, validateGraphName(request.graphName));
        }
        const directory = runDirectory(this.stateRoot, request.jobId);
        const revision = request.revision;
        // A revision that adds nothing has no source to discover; the config
        // still needs one to load, so it names the run's own directory, which
        // the planner is told not to read.
        const configured =
          revision && !revision.addDocuments
            ? { ...request, sourceKind: "local_path" as const, source: { kind: "local", path: directory } }
            : request;
        // The fleet records who the graph belongs to with the run itself, so
        // the owner is known wherever the run is read from.
        const configPath = await writeRunConfig(
          configured,
          path.join(directory, "config.yaml"),
          this.stubbed ? null : await this.fleetProfile(),
          await this.local.writerPrincipal(),
        );
        const startedAt = new Date().toISOString();
        if (this.stubbed) {
          await writeRunRecord(directory, {
            runId: request.jobId,
            graphId: request.graphId,
            graphName: request.graphName,
            status: "queued",
            runtime: "kubernetes",
            startedAt,
            updatedAt: startedAt,
            outputPath: request.output.workspacePath,
            configPath,
            storageKind: request.output.kind,
            storageLocation: request.output.workspacePath,
            owner: await this.local.writerPrincipal(),
            baseRunId: revision?.baseRunId ?? null,
            ...cloneFieldsFromRequest(request),
          });
          await appendStubRun(this.stateRoot, {
            runId: request.jobId,
            graphId: request.graphId,
            status: "queued",
          });
        } else {
          const result = await runFlakegraph(submitArguments(request, configPath), {
            cwd: this.repositoryRoot,
            env: environmentForRequest(configured),
          });
          if (revision && result.exitCode === 2) {
            // The planner refused the revision. It may have cancelled a run
            // row it had already created; that attempt is not a version of
            // the graph, so the catalog does not list it.
            await withdrawRun(this.stateRoot, request.jobId);
            throw invalid(cliFailure(result.stderr, "The revision was refused"));
          }
          const payload = lastJsonObject(result.stdout) ?? {};
          await writeRunRecord(directory, {
            runId: request.jobId,
            graphId: request.graphId,
            graphName: request.graphName,
            status: String(payload.status ?? (result.exitCode === 0 ? "queued" : "failed")),
            runtime: "kubernetes",
            startedAt,
            updatedAt: startedAt,
            outputPath: request.output.workspacePath,
            configPath,
            storageKind: request.output.kind,
            storageLocation: request.output.workspacePath,
            error: result.exitCode === 0 ? null : result.stderr,
            owner: await this.local.writerPrincipal(),
            baseRunId: revision?.baseRunId ?? null,
            ...cloneFieldsFromRequest(request),
          });
        }
        return Effect.runPromise(this.getRun(request.jobId));
      },
      catch: (cause) => fromCause(cause, "Unable to submit distributed run"),
    });
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const catalog = await listRunRecords(this.stateRoot, limit);
        const records = new Map(
          catalog
            .filter(({ record }) => String(record.runtime || "") === "kubernetes")
            .map(({ record }) => [record.runId, record] as const),
        );
        const local = await Promise.all(
          [...records.values()].map(async (record) =>
            snapshotFromRecord(record, {
              graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
            }),
          ),
        );
        if (this.stubbed) {
          return local;
        }
        // The coordination store lists every run the fleet holds, including
        // those submitted from elsewhere; the catalog only adds what this
        // console knows about them. Without the store, the catalog is shown
        // with a warning rather than nothing.
        let rows;
        try {
          rows = await listPostgresRuns(limit);
        } catch (error) {
          const reason = error instanceof Error ? error.message : "coordination store unavailable";
          return local.map((snapshot) => ({
            ...snapshot,
            listingWarning: `Showing this console's own record of fleet runs, not the fleet's: ${reason}.`,
          }));
        }
        const withdrawn = await withdrawnRunIds(this.stateRoot);
        const counts = await documentCountsByRun(rows.map((row) => row.id).filter((id) => !withdrawn.has(id)));
        const versions = await graphVersionsByGraph([...new Set(rows.map((row) => row.graphId))]);
        const seen = new Set<string>();
        const merged: RunSnapshot[] = [];
        for (const row of rows) {
          if (withdrawn.has(row.id)) {
            continue;
          }
          seen.add(row.id);
          const record = records.get(row.id);
          const name = await graphName(this.stateRoot, row.graphId);
          const documents = counts.get(row.id) ?? { total: null, completed: 0, failed: 0 };
          const snapshot = record
            ? snapshotFromRecord(
                { ...record, status: row.status, updatedAt: row.updatedAt?.toISOString() ?? record.updatedAt },
                { graphName: name ?? record.graphName ?? null },
              )
            : this.snapshotFromRow(row, name);
          merged.push({
            ...snapshot,
            documentsTotal: documents.total,
            documentsCompleted: documents.completed,
            documentsFailed: documents.failed,
            raw: {
              ...snapshot.raw,
              // The owner the run itself recorded, for runs started anywhere;
              // a console record's own owner comes first where there is one.
              owner: snapshot.raw.owner ?? runOwner(row.configJson),
              priority: row.priorityBand,
              version: versionOf(versions.get(row.graphId) ?? [], row.id),
            },
          });
        }
        for (const snapshot of local) {
          if (!seen.has(snapshot.runId)) {
            merged.push(snapshot);
          }
        }
        return merged;
      },
      catch: (cause) => fromCause(cause, "Unable to list fleet runs"),
    });
  }

  private snapshotFromRow(
    row: { id: string; graphId: string; status: string; createdAt: Date | null; updatedAt: Date | null; errorJson: unknown },
    graphName: string | null,
  ): RunSnapshot {
    const outputPath = this.defaultOutputPath(row.id);
    return {
      runId: row.id,
      graphId: row.graphId,
      status: row.status,
      startedAt: row.createdAt?.toISOString() ?? null,
      updatedAt: row.updatedAt?.toISOString() ?? null,
      graphName,
      stages: [],
      events: [],
      documentsTotal: null,
      documentsCompleted: 0,
      documentsFailed: 0,
      outputPath,
      storageKind: "local_files",
      storageLocation: outputPath,
      warnings: [],
      error: row.errorJson ? JSON.stringify(row.errorJson) : null,
      listingWarning: null,
      raw: {},
    };
  }

  /** Where a fleet run's graph is materialised on first view: under the console's own state. */
  private defaultOutputPath(runId: string): string {
    return path.join(this.stateRoot, "artifacts", assertSafeId(runId, "runId"));
  }

  /** The record the console keeps for a run, or null for one it only knows through the fleet. */
  private async record(runId: string) {
    const directory = runDirectory(this.stateRoot, runId);
    return runRecordExists(directory) ? readRunRecord(directory) : null;
  }

  /** `distributed status` for one run, read with the run's own configuration when the console has it. */
  private async status(runId: string, configPath: string | null | undefined): Promise<Record<string, unknown> | null> {
    const args = ["distributed", "status", "--run-id", runId];
    if (configPath) {
      args.push("--config", configPath);
    }
    const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
    return lastJsonObject(result.stdout);
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (this.stubbed) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          const snapshot = snapshotFromRecord(record, {
            graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
          });
          return { ...snapshot, raw: { ...snapshot.raw, version: versionOf(await this.stubVersions(record.graphId), runId) } };
        }
        const payload = await this.status(runId, record?.configPath);
        if (!payload?.run) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          return snapshotFromRecord(record, {
            graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
            listingWarning: "The fleet's coordination store could not be read; this is the console's own record.",
          });
        }
        const run = payload.run as Record<string, unknown>;
        const graphId = String(run.graph_id ?? record?.graphId ?? runId);
        const name = (await graphName(this.stateRoot, graphId)) ?? record?.graphName ?? null;
        const base = record
          ? snapshotFromRecord(record, { graphName: name })
          : this.snapshotFromRow(
              { id: runId, graphId, status: String(run.status ?? "unknown"), createdAt: null, updatedAt: null, errorJson: null },
              name,
            );
        const snapshot = snapshotFromStatus(payload, base);
        if (record && record.status !== snapshot.status) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), { status: snapshot.status });
        }
        const versions = (await graphVersionsByGraph([graphId])).get(graphId) ?? [];
        return { ...snapshot, raw: { ...snapshot.raw, version: versionOf(versions, runId) } };
      },
      catch: (cause) => fromCause(cause, "Unable to load Kubernetes run"),
    });
  }

  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          // The stub keeps a run's progress beside its record, as a local run does.
          const progress = await readLocalProgress(path.join(runDirectory(this.stateRoot, runId), "events.jsonl"));
          return documentStatusesFromEvents(progress.events);
        }
        const finalizer = await finalizerPayload(runId);
        const own = documentStatusesFromTasks(await documentTasksByRun(runId));
        // A revision's kept documents live in the runs that processed them;
        // the finalizer's payload says which, and they are shown as kept.
        const inherited: DocumentStatus[] = [];
        const entries = inheritedDocuments(finalizer);
        if (entries.length) {
          const graphId = (await this.record(runId))?.graphId ?? null;
          const versions = graphId ? ((await graphVersionsByGraph([graphId])).get(graphId) ?? []) : [];
          for (const entry of entries) {
            const kept = new Set(entry.fileIds);
            const number = versions.findIndex((version) => version.runId === entry.runId) + 1;
            const origin = number ? `version ${number}` : entry.runId;
            for (const status of documentStatusesFromTasks(await documentTasksByRun(entry.runId))) {
              if (kept.has(status.fileId)) {
                inherited.push({ ...status, phase: "inherited", detail: `Kept from ${origin}` });
              }
            }
          }
        }
        return withProvenance([...own, ...inherited], finalizer).sort((left, right) =>
          (left.name ?? left.fileId).localeCompare(right.name ?? right.fileId),
        );
      },
      catch: (cause) => fromCause(cause, "Unable to list the run's documents"),
    });
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (!record && this.stubbed) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed) {
          await this.control("cancel", runId, record?.configPath);
        }
        if (record) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), {
            status: "cancelled",
            cancellationRequestedAt: new Date().toISOString(),
          });
        }
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to cancel Kubernetes run"),
    });
  }

  retry(runId: string, skippedFileIds: readonly string[] = []): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (!record && this.stubbed) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed) {
          await this.control("retry", runId, record?.configPath, skippedFileIds.flatMap((id) => ["--skip-file-id", id]));
        }
        if (record) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), {
            status: "queued",
            error: null,
            cancellationRequestedAt: null,
          });
        }
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to retry Kubernetes run"),
    });
  }

  versions(graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          return (await this.stubVersions(graphId)).map((row, index) => ({
            graphId,
            runId: row.runId,
            number: index + 1,
            createdAt: row.createdAt,
            head: row.head,
          }));
        }
        const rows = (await graphVersionsByGraph([graphId])).get(graphId) ?? [];
        return rows.map((row, index) => ({
          graphId,
          runId: row.runId,
          number: index + 1,
          createdAt: row.createdAt.toISOString(),
          head: row.head,
        }));
      },
      catch: (cause) => fromCause(cause, "Unable to list the graph's versions"),
    });
  }

  recover(runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (this.stubbed) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          await writeRunRecord(runDirectory(this.stateRoot, runId), { status: "queued", cancellationRequestedAt: null });
          return `Reconciled infrastructure for ${runId}`;
        }
        // The CLI restarts only a pool with no available worker, after checking
        // for the failures a restart cannot fix; task state is never touched.
        const args = ["fleet", "recover", "--run-id", runId, ...this.fleetArguments()];
        if (record?.configPath) {
          args.push("--config", record.configPath);
        }
        const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
        const payload = lastJsonObject(result.stdout);
        if (result.exitCode !== 0 || !payload) {
          throw new Error(lastLine(result.stderr) || `Unable to recover the workers of ${runId}`);
        }
        return String(payload.message ?? `Reconciled infrastructure for ${runId}`);
      },
      catch: (cause) => fromCause(cause, "Unable to recover Kubernetes run"),
    });
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return this.local.renameGraph(graphId, displayName);
  }

  loadGraph(location: string, graphId?: string | null, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return this.local.loadGraph(location, graphId, options);
  }

  /** The fleet graph's local copy, which loading the run's graph writes first. */
  graphDirectory(snapshot: RunSnapshot): string | null {
    return snapshot.outputPath ? this.materialisedGraphDirectory(snapshot.runId, snapshot.outputPath) : null;
  }

  loadRunGraph(snapshot: RunSnapshot, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (!snapshot.outputPath) {
          throw invalid(`Run ${snapshot.runId} does not record an output directory`);
        }
        const directory = this.materialisedGraphDirectory(snapshot.runId, snapshot.outputPath);
        // The fleet publishes its graph to the artifact store; the local copy
        // is materialised once, by the CLI, then read like any other graph.
        // A page asks for the graph from several places at once (the explorer,
        // Quality, Inspect); they share one export, since two exports of a
        // large graph at once can exhaust the console's memory.
        if (!this.stubbed && !graphArtifactsExist(directory)) {
          let pending = exportsInFlight.get(directory);
          if (!pending) {
            pending = this.exportGraph(snapshot.runId, directory).finally(() => exportsInFlight.delete(directory));
            exportsInFlight.set(directory, pending);
          }
          await pending;
        }
        return loadLocalGraph(directory, options);
      },
      catch: (cause) => fromCause(cause, "Unable to load fleet graph"),
    });
  }

  private async exportGraph(runId: string, directory: string): Promise<void> {
    const record = await this.record(runId);
    const args = ["distributed", "export", "--run-id", runId, "--output", directory];
    if (record?.configPath) {
      args.push("--config", record.configPath);
    }
    const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
    if (result.exitCode !== 0) {
      const failed = `Exporting the graph of ${runId} failed`;
      const reason = cliFailure(result.stderr, "");
      throw new Error(reason ? `${failed}: ${reason}` : failed);
    }
  }

  /**
   * Where a fleet run's graph is materialised for reading.
   *
   * Runs of one graph submitted by the console share the graph's output path,
   * and each version is its own graph, so a run keeps its export in a
   * directory of its own beneath it. A run submitted with its id as the last
   * element of its path already owns that directory.
   */
  private materialisedGraphDirectory(runId: string, outputPath: string): string {
    const base = this.local.artifactDirectory(outputPath);
    if (path.basename(base) === runId) {
      return base;
    }
    if (this.stubbed && graphArtifactsExist(base)) {
      return base;
    }
    return path.join(base, runId);
  }

  private async control(action: "cancel" | "retry", runId: string, configPath: string | null | undefined, extra: readonly string[] = []): Promise<void> {
    const args = ["distributed", action, "--run-id", runId, ...extra];
    if (configPath) {
      args.push("--config", configPath);
    }
    const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
    if (result.exitCode !== 0) {
      throw new Error(cliFailure(result.stderr, `Unable to ${action} ${runId}`));
    }
  }

  /**
   * The stub fleet's versions: every run that published this graph, oldest
   * first, the latest being the head - the same shape the store keeps.
   */
  private async stubVersions(graphId: string): Promise<Array<{ runId: string; createdAt: string; head: boolean }>> {
    const records = (await listRunRecords(this.stateRoot, 1_000))
      .map((entry) => entry.record)
      .filter((record) => record.graphId === graphId && isSuccessStatus(record.status))
      .sort((left, right) => String(left.startedAt ?? "").localeCompare(String(right.startedAt ?? "")));
    return records.map((record, index) => ({
      runId: record.runId,
      createdAt: record.updatedAt ?? record.startedAt ?? new Date(0).toISOString(),
      head: index === records.length - 1,
    }));
  }

  /**
   * Write a finished graph into a Snowflake schema without running it again.
   *
   * The run's own configuration is reused with the writer pointed at
   * Snowflake: the fleet's account and credential from its profile, the
   * database, schema, warehouse and stage from the request. The CLI reads
   * the run's final artifact and bulk-loads it, which is the finalizer's
   * last step done once more for another destination.
   */
  async publishToSnowflake(runId: string, target: SnowflakePublishTarget): Promise<{ nodes: number; edges: number }> {
    const directory = runDirectory(this.stateRoot, runId);
    const record = await this.record(runId);
    if (!record?.configPath) {
      throw invalid(`Run ${runId} records no configuration to publish with`);
    }
    if (this.stubbed) {
      return { nodes: 0, edges: 0 };
    }
    const fleet = await this.fleetProfile();
    const account = fleet?.config.snowflake;
    if (!account || typeof account !== "object") {
      throw notSupported("This fleet has no Snowflake account in its processing profile");
    }
    const base = parseYaml(await readFile(record.configPath, "utf8")) as Record<string, unknown>;
    const config = {
      ...base,
      writer: { provider: "snowflake_bulk" },
      snowflake: {
        ...(account as Record<string, unknown>),
        database: target.database,
        schema: target.schema,
        warehouse: target.warehouse,
        ...(target.role ? { role: target.role } : {}),
        bulk_stage: target.bulkStage,
      },
    };
    const configPath = path.join(directory, "publish-snowflake.yaml");
    await writeFile(configPath, stringifyYaml(config), "utf8");
    const result = await runFlakegraph(["distributed", "publish", "--run-id", runId, "--config", configPath], {
      cwd: this.repositoryRoot,
      timeoutMs: 60 * 60 * 1000,
    });
    if (result.exitCode !== 0) {
      throw invalid(lastLine(result.stderr) || "Publishing to Snowflake failed");
    }
    const payload = lastJsonObject(result.stdout) ?? {};
    return { nodes: Number(payload.nodes ?? 0), edges: Number(payload.edges ?? 0) };
  }

  /** What the fleet's workers run, or null when the fleet cannot be read. */
  async fleetProfile(): Promise<FleetProfile | null> {
    if (this.stubbed) {
      return null;
    }
    return readFleetProfile(appEnv().kubernetesNamespace, { cwd: this.repositoryRoot, context: null });
  }

  // The fleet is the one kubectl reaches: the service account in the pod,
  // or the current kubeconfig context on a laptop, in the configured
  // namespace. A catalog of contexts to switch between belonged to a time
  // when the console ran beside the fleet rather than in it.
  private fleetArguments(): string[] {
    return ["--namespace", appEnv().kubernetesNamespace];
  }

  cluster(namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          const snapshot = await readJsonFile<ClusterSnapshot>(path.join(this.stateRoot, "kubernetes", "cluster.json"));
          if (snapshot) {
            return { ...snapshot, namespace: snapshot.namespace || namespace };
          }
          return emptyClusterSnapshot(namespace, ["No stub cluster snapshot is present."]);
        }
        return await readLiveCluster(namespace);
      },
      catch: (cause) => fromCause(cause, "Unable to load cluster snapshot"),
    });
  }

  nodeAssignments(
    namespace: string,
    nodeName: string,
  ): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          const snapshot = await readJsonFile<{ assignments?: NodeWorkAssignment[] }>(
            path.join(this.stateRoot, "kubernetes", "assignments.json"),
          );
          return (snapshot?.assignments ?? []).filter(
            (item) => item.scopeId === nodeName || item.workerId.includes(nodeName),
          );
        }
        // A lease names the worker pod that holds it; the cluster says which
        // node that pod runs on.
        const cluster = await Effect.runPromise(this.cluster(namespace));
        const podsOnNode = new Set(
          (cluster?.workloads ?? []).filter((workload) => workload.node === nodeName).map((workload) => workload.name),
        );
        if (podsOnNode.size === 0) {
          return [];
        }
        return (await leasedTasks())
          .filter((row) => podsOnNode.has(row.leaseOwner))
          .map((row) => ({
            workerId: row.leaseOwner,
            runId: row.runId,
            graphId: row.graphId,
            taskId: row.taskId,
            stage: row.stage,
            scopeId: row.scopeId,
            updatedAt: row.updatedAt?.toISOString() ?? null,
          }));
      },
      catch: (cause) => fromCause(cause, "Unable to load node assignments"),
    });
  }

  /**
   * Delete a graph from the fleet: every run of it, their coordination rows
   * and every object they stored, through the pipeline's own `distributed
   * delete`. The console's own records of it are removed afterwards by the
   * caller, with the rest of what the console keeps.
   */
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return Effect.tryPromise({
      try: async (): Promise<Record<string, number>> => {
        if (this.stubbed) {
          return { runs: 0 };
        }
        const records = (await listRunRecords(this.stateRoot, Number.MAX_SAFE_INTEGER, { includeWithdrawn: true }))
          .map((entry) => entry.record)
          .filter((record) => record.graphId === graphId);
        const configPath = records
          .map((record) => record.configPath)
          .find((candidate): candidate is string => Boolean(candidate) && existsSync(candidate!));
        const args = ["distributed", "delete", "--graph-id", graphId];
        if (configPath) {
          args.push("--config", configPath);
        }
        const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
        if (result.exitCode === 2) {
          throw invalid(cliFailure(result.stderr, "The fleet refused to delete the graph"));
        }
        if (result.exitCode !== 0) {
          throw new Error(cliFailure(result.stderr, `Deleting ${graphId} failed`));
        }
        const payload = lastJsonObject(result.stdout) ?? {};
        return {
          runs: Array.isArray(payload.run_ids) ? payload.run_ids.length : 0,
          tasks: Number(payload.tasks ?? 0),
          artifacts: Number(payload.artifacts ?? 0),
          objects: Number(payload.objects ?? 0),
        };
      },
      catch: (cause) => fromCause(cause, "Unable to delete graph"),
    });
  }

  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () =>
        stringifyYaml(redactedConfig(await buildRunConfig(request, await this.fleetProfile())), {
          sortMapEntries: false,
        }),
      catch: (cause) => fromCause(cause, "Unable to preview configuration"),
    });
  }
}

export function createKubernetesRuntime(viewer: Viewer | null = null): KubernetesRuntime {
  const env = appEnv();
  return new KubernetesRuntime(env.repositoryRoot, env.stateRoot, env.stubRuntimes, viewer);
}

async function readLiveCluster(namespace: string): Promise<ClusterSnapshot> {
  return Promise.race([
    readLiveClusterUnbound(namespace),
    new Promise<ClusterSnapshot>((resolve) => {
      setTimeout(() => {
        resolve(emptyClusterSnapshot(namespace, ["Timed out waiting for the Kubernetes API"]));
      }, 4_000);
    }),
  ]);
}

async function readLiveClusterUnbound(namespace: string): Promise<ClusterSnapshot> {
  try {
    const k8s = await import("@kubernetes/client-node");
    const config = new k8s.KubeConfig();
    config.loadFromDefault();
    const core = config.makeApiClient(k8s.CoreV1Api);
    const nodes = await core.listNode();
    const pods = await core.listNamespacedPod({ namespace });
    const nodeStatuses: NodeStatus[] = (nodes.items ?? []).map((node) => {
      const ready = (node.status?.conditions ?? []).some(
        (condition) => condition.type === "Ready" && condition.status === "True",
      );
      const gpu = Number(node.status?.capacity?.["nvidia.com/gpu"] ?? 0);
      const labels = node.metadata?.labels ?? {};
      return {
        name: node.metadata?.name ?? "unknown",
        ready,
        nodeClass: labels["node.kubernetes.io/instance-type"] ?? labels["kubernetes.io/arch"] ?? "unknown",
        gpuCount: Number.isFinite(gpu) ? gpu : 0,
        gpuModel: labels["nvidia.com/gpu.product"] ?? null,
        cpuCapacity: node.status?.capacity?.cpu ?? null,
        memoryCapacity: node.status?.capacity?.memory ?? null,
        gpuPercent: null,
        cpuUsage: null,
        cpuPercent: null,
        memoryUsage: null,
        memoryPercent: null,
        workloadCount: 0,
        workerCount: 0,
        modelServerReady: false,
        model: null,
        modelImage: null,
      };
    });
    const workloads: WorkloadStatus[] = (pods.items ?? []).map((pod) => ({
      name: pod.metadata?.name ?? "unknown",
      component: pod.metadata?.labels?.["app.kubernetes.io/component"] ?? "workload",
      phase: pod.status?.phase ?? "Unknown",
      node: pod.spec?.nodeName ?? null,
      ready: (pod.status?.containerStatuses ?? []).every((status) => status.ready),
      restarts: (pod.status?.containerStatuses ?? []).reduce((sum, status) => sum + (status.restartCount ?? 0), 0),
      cpu: null,
      memory: null,
      image: engineContainer(pod)?.image ?? pod.spec?.containers?.[0]?.image ?? null,
      model: servedModel(pod),
    }));
    const annotatedNodes = nodeStatuses.map((node) => {
      const onNode = workloads.filter((workload) => workload.node === node.name);
      const model = onNode.find((workload) => workload.component.includes("vllm") || workload.component.includes("model"));
      return {
        ...node,
        workloadCount: onNode.length,
        workerCount: onNode.filter((workload) => workload.component.startsWith("worker")).length,
        modelServerReady: Boolean(model?.ready),
        model: model?.model ?? null,
        modelImage: model?.image ?? null,
      };
    });
    return {
      context: config.currentContext ?? "default",
      namespace,
      nodesReady: annotatedNodes.filter((node) => node.ready).length,
      nodesTotal: annotatedNodes.length,
      nodes: annotatedNodes,
      workloads,
      warnings: [],
      observedAt: new Date().toISOString(),
    };
  } catch (error) {
    return emptyClusterSnapshot(namespace, [
      error instanceof Error ? error.message : "Kubernetes API is unavailable",
    ]);
  }
}

type PodLike = {
  metadata?: { labels?: Record<string, string> };
  spec?: { containers?: Array<{ name?: string; image?: string; args?: string[] }> };
};

/** The container that runs the model, on a pod that serves one. */
function engineContainer(pod: PodLike) {
  const containers = pod.spec?.containers ?? [];
  return containers.find((container) => container.name === "vllm") ?? null;
}

/**
 * The model a serving pod answers for: the `--served-model-name` it was
 * started with, else the checkpoint after `serve`. Nothing on the pod's
 * labels says it, so the arguments do.
 */
function servedModel(pod: PodLike): string | null {
  const labelled = pod.metadata?.labels?.["flakegraph/model"];
  if (labelled) {
    return labelled;
  }
  const args = engineContainer(pod)?.args ?? [];
  const named = args.indexOf("--served-model-name");
  if (named >= 0 && args[named + 1]) {
    return args[named + 1] ?? null;
  }
  const serve = args.indexOf("serve");
  if (serve >= 0 && args[serve + 1] && !args[serve + 1]!.startsWith("-")) {
    return args[serve + 1] ?? null;
  }
  return null;
}

/**
 * Where a run stands among its graph's published versions: its number, how
 * many there are, and whether the head points at it. Null for a run that has
 * not published one, whose graph may still have versions from other runs.
 */
function versionOf(
  versions: readonly { runId: string; head: boolean }[],
  runId: string,
): { number: number; count: number; head: boolean } | null {
  const index = versions.findIndex((version) => version.runId === runId);
  if (index < 0) {
    return versions.length ? { number: 0, count: versions.length, head: false } : null;
  }
  return { number: index + 1, count: versions.length, head: versions[index]!.head };
}

/** The documents a finalizer's payload says it kept, per run they came from. */
function inheritedDocuments(payload: Record<string, unknown> | null): Array<{ runId: string; fileIds: string[] }> {
  const entries = payload?.inherit;
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.flatMap((entry) => {
    if (!entry || typeof entry !== "object") {
      return [];
    }
    const record = entry as Record<string, unknown>;
    const fileIds = Array.isArray(record.file_ids) ? record.file_ids.map(String) : [];
    return typeof record.run_id === "string" && fileIds.length ? [{ runId: record.run_id, fileIds }] : [];
  });
}

function emptyClusterSnapshot(namespace: string, warnings: string[]): ClusterSnapshot {
  return {
    context: "unavailable",
    namespace,
    nodesReady: 0,
    nodesTotal: 0,
    nodes: [],
    workloads: [],
    warnings,
    observedAt: new Date().toISOString(),
  };
}

async function appendStubRun(stateRoot: string, run: { runId: string; graphId: string; status: string }) {
  const file = path.join(stateRoot, "kubernetes", "runs.json");
  await mkdir(path.dirname(file), { recursive: true });
  const current = (await readJsonFile<typeof run[]>(file)) ?? [];
  current.unshift(run);
  await atomicWriteJson(file, current);
}

// A CLI failure ends with its cause; the traceback above it is not for the page.
function lastLine(text: string): string {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines[lines.length - 1] ?? "";
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

/** The owner a fleet run recorded in its stored configuration, if it recorded one. */
function runOwner(config: unknown): string | null {
  const owner = (config as { owner?: unknown } | null)?.owner;
  return typeof owner === "string" && owner.trim() ? owner.trim() : null;
}
