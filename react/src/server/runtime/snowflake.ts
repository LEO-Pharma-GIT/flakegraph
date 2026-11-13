import { createHash } from "node:crypto";
import path from "node:path";
import { existsSync } from "node:fs";
import { readFile, realpath } from "node:fs/promises";
import { Effect } from "effect";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import { ACTIVE_SNOWFLAKE_SESSION, buildRunConfig, loadBaseConfig, redactedConfig } from "../config";
import type { DocumentPhase, DocumentStatus } from "../documents";
import { appEnv, type SnowflakeConnectionEnv } from "../env";
import type { FleetProfile } from "../fleet";
import { edgeQuantities, edgeStatedTypes } from "../graph";
import { graphFootprint, removeGraphRecords } from "../graph-cleanup";
import { unidentifiedViewer } from "../identity";
import {
  accessRefusal,
  NO_ACCESS,
  principalKey,
  principalRole,
  roleFor,
  rolePrincipal,
  validPrincipal,
  type AccessLevel,
  type GraphAccessRecord,
  type GraphGrant,
} from "../access/graph-access";
import {
  mayReadStagePath,
  SNOWFLAKE_ACCESS_POLICY,
  viewerStagePrefix,
  visibleGraphPredicate,
} from "../access/snowflake-access";
import { loadAccess } from "../access/store";
import {
  SNOWFLAKE_CAPABILITIES,
  isActiveStatus,
  type Capability,
  type ClusterSnapshot,
  type GraphAccessView,
  type GraphDataset,
  type GraphDeletionPreview,
  type GraphRole,
  type GraphVersion,
  type IngestionRequest,
  type NodeWorkAssignment,
  type PreflightResult,
  type RunSnapshot,
  type SourceObject,
  type SourceSummary,
  type StageProgress,
  type Viewer,
  validateGraphName,
} from "../protocol/schema";
import { ControlPlaneError, forbidden, fromCause, invalid, notFound, notSupported } from "../protocol/errors";
import type { GraphAccessPlane, GraphLoadOptions, ModelChoices, SnowflakePublishing, StageChoice, StageFolder } from "../protocol/runtime";
import { cortexModelChoices, documentStageChoices, readableStageFolders } from "../snowflake/choices";
import { embeddingDimension } from "../providers";
import {
  listLocalObjects,
  REMOTE_SUMMARY_LIMIT,
  resolveUploadFolder,
  summarizeListing,
  summarizeLocalObjects,
  uploadsRoot,
} from "../sources";
import {
  callerAvailableRoles,
  normalizeRoles,
  sharedSnowflakeSql,
  type SnowflakeSql,
  type SqlRow,
} from "../snowflake/client";
import {
  COMMUNITY_FINDING_QUERY,
  COUNTED_TABLES,
  EDGE_DETAIL_QUERY,
  EMBEDDING_TABLES,
  GRAPH_DATA_TABLES,
  GRAPH_METRICS_QUERY,
  REQUIRED_TABLES,
  REVIEW_RECORD_TABLES,
  REVIEW_TABLE_LIMITS,
  RUN_REPORT_QUERY,
  graphCountQuery,
  reviewEdgeQuery,
  reviewProjection,
  reviewRecordQuery,
} from "../snowflake/graph-tables";
import { jobErrorText, liveConsumption, variantList, variantMapping, vectorWidth, withLiveStageProgress } from "../snowflake/progress";
import {
  buildSpcsLaunch,
  containerResources,
  serviceNameForJob,
  sortKeys,
  type SpcsLaunch,
} from "../snowflake/spcs";
import {
  IDENTIFIER,
  UPLOAD_NAMESPACE,
  UPLOAD_STAGE,
  putFolderToStage,
  qualifiedStage,
  sourceObjectFromListRow,
  stageLocation,
  stageRelativePath,
  submissionRow,
  type SubmissionRow,
} from "../snowflake/stage";

/** Queue rows merged per statement; seven binds each keeps a statement well inside bind limits. */
const SUBMISSION_BATCH_ROWS = 1_000;
// Snowflake keeps a job service after it stops. Only these states mean the run
// is over and its name is free; anything else is still holding the compute pool.
const TERMINAL_SERVICE_STATUSES = new Set(["DONE", "FAILED", "CANCELLED", "SUSPENDED", "DELETING", "INTERNAL_ERROR"]);
// Deliberately narrower than the set above. Reusing a name and declaring a run
// dead are different questions, and the second one is destructive: a wrong
// answer marks a working run failed and stores that. SUSPENDED and DELETING are
// states a service can pass through while its work is still expected to
// continue, so only states a service can never do further work from count.
const DEAD_SERVICE_STATUSES = new Set(["DONE", "FAILED", "CANCELLED", "INTERNAL_ERROR"]);
/** Job states that still expect a worker to do something. */
const UNSETTLED_JOB_STATUSES = new Set(["pending", "running"]);
// How long a run may go without its job row changing before the app stops
// assuming a worker is behind it. Comfortably longer than the worker's own
// progress interval, so a busy run is never interrogated.
const QUIET_RUN_SECONDS = 120;
const ROLE_CACHE_MS = 5 * 60_000;
const DISCOVERY_CACHE_MS = 60_000;
/** Objects read from a stage before its folders are offered, to keep the request short. */
const MAX_FOLDER_LISTING = 50_000;
/**
 * The runtime options a viewer may choose. Everything else in the snowflake
 * section (account, namespace, credentials) is the app's to decide: runtime
 * options are merged over it, so an open map would let a request redirect the
 * worker to another database.
 */
const VIEWER_RUNTIME_OPTIONS = new Set([
  "compute_pool",
  "image_repository",
  "image_name",
  "image_digest",
  "service_cpu_request",
  "service_cpu_limit",
  "service_memory_request",
  "service_memory_limit",
]);
const RESOURCE_OPTIONS = ["service_cpu_request", "service_cpu_limit", "service_memory_request", "service_memory_limit"];
const FOREIGN_UPLOADS = "That stage folder holds other users' uploads. Point at your own upload folder, or give a prefix outside it.";
const FOREIGN_UPLOAD_FOLDER = "That upload folder belongs to someone else.";
const SHARE_TARGET = "Enter a Snowflake user name, or role:NAME for everyone holding a role.";
/** Who the console writes as when no person is named. */
const SYSTEM_PRINCIPAL = "SYSTEM";

export interface SnowflakeNamespace {
  database: string;
  schema: string;
  warehouse: string | null;
}

export interface SnowflakeRuntimeOptions {
  repositoryRoot: string;
  stateRoot: string;
  sql: SnowflakeSql;
  connection: SnowflakeConnectionEnv;
  viewer?: Viewer;
  /** The viewer's own request token, present when the app has caller's rights. */
  callerToken?: string | null;
  /** Ask Snowflake for the caller's roles; replaced in tests. */
  callerRoles?: (callerToken: string) => Promise<string[]>;
  /** Move an upload folder into a stage; replaced in tests, or if PUT egress is blocked. */
  putFolder?: (sql: SnowflakeSql, directory: string, stage: string, prefix: string) => Promise<number>;
}

interface PreparedRequest {
  request: IngestionRequest;
  uploadDirectory: string | null;
  namespace: SnowflakeNamespace;
}

interface Cached {
  at: number;
  value: unknown;
}

// Keyed by client so each connection keeps its own answers and a test's fake
// never sees another's.
const discoveryCaches = new WeakMap<SnowflakeSql, Map<string, Cached>>();

/**
 * The control plane inside Snowflake, and its access plane: KG_* tables are the
 * run catalog, a Snowpark Container Services job service is the worker, graph
 * rows are read straight from the tables the worker writes, and KG_GRAPH.OWNER
 * and KG_GRAPH_ACL say who may use them.
 *
 * Elsewhere a console keeps ownership in its own state beside a runtime that
 * knows nothing of it. Here that state is the account's, and the container's
 * disk does not outlive a restart, so nothing wraps this class: every method
 * checks the viewer itself, and there is only this one check.
 *
 * The session runs with the service's own role, so the warehouse returns every
 * row that role can read. A list carries `visibleGraphPredicate` in its query;
 * anything that names one graph or run goes through `roleOrFail`, which decides
 * with the same `roleFor` a console's own store is read with.
 *
 * What stays on disk is what only exists there: an upload folder, with the
 * note of who created it, until its run puts it in a stage; and the people the
 * console has seen, offered when sharing. A restart loses both together.
 */
export class SnowflakeRuntime implements GraphAccessPlane {
  readonly runtime = "snowflake" as const;
  readonly capabilities = new Set<Capability>(SNOWFLAKE_CAPABILITIES);
  private viewerPromise: Promise<Viewer> | null = null;
  private namespacePromise: Promise<SnowflakeNamespace> | null = null;

  constructor(private readonly options: SnowflakeRuntimeOptions) {}

  get repositoryRoot(): string {
    return this.options.repositoryRoot;
  }

  get stateRoot(): string {
    return this.options.stateRoot;
  }

  private get sql(): SnowflakeSql {
    return this.options.sql;
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return this.attempt(() => this.resolvedViewer(), "Unable to identify the viewer");
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return this.attempt(async () => {
      const kind = String(source.kind ?? "snowflake_stage");
      if (kind === "local" || kind === "upload" || kind === "local_path") {
        return listLocalObjects(await this.ownUploadFolder(String(source.path ?? "")), limit);
      }
      const stage = String(source.stage ?? "");
      const prefix = String(source.prefix ?? "");
      await this.requireReadableStage(stage, prefix);
      return this.listStage(stage, prefix, limit);
    }, "Unable to list Snowflake stage objects");
  }

  summarizeSource(source: Record<string, unknown>): Effect.Effect<SourceSummary, ControlPlaneError> {
    return this.attempt(async () => {
      const kind = String(source.kind ?? "snowflake_stage");
      if (kind === "local" || kind === "upload" || kind === "local_path") {
        return summarizeLocalObjects(await this.ownUploadFolder(String(source.path ?? "")));
      }
      const stage = String(source.stage ?? "");
      const prefix = String(source.prefix ?? "");
      await this.requireReadableStage(stage, prefix);
      return summarizeListing(await this.listStage(stage, prefix, REMOTE_SUMMARY_LIMIT), REMOTE_SUMMARY_LIMIT);
    }, "Unable to count source objects");
  }

  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError> {
    return this.attempt(() => this.runPreflight(request), "Snowflake preflight failed");
  }

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(() => this.runSubmit(request), "Unable to submit Snowflake job");
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return this.attempt(() => this.queryRuns(limit), "Unable to list Snowflake jobs");
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(() => this.status(runId), "Unable to load Snowflake job");
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(async () => {
      await this.requireRunRole(runId, "write");
      // The parent first: a worker that sees its job cancelled stops claiming,
      // so no file can be claimed between the two statements and left running.
      await this.sql.query(
        "UPDATE KG_JOB SET STATUS = 'CANCELLED', LEASE_OWNER = NULL, " +
          "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() " +
          "WHERE ID = ? AND STATUS NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')",
        [runId],
      );
      await this.sql.query(
        "UPDATE KG_JOB_FILE SET STATUS = 'CANCELLED', WORKER_ID = NULL, " +
          "LEASE_UNTIL = NULL, UPDATED_AT = CURRENT_TIMESTAMP() " +
          "WHERE JOB_ID = ? AND STATUS IN ('QUEUED', 'CLAIMED')",
        [runId],
      );
      return this.status(runId);
    }, "Unable to cancel Snowflake job");
  }

  /**
   * Requeue a failed run's documents and start a worker to finish them.
   *
   * Documents that already succeeded are left alone, so a retry costs only the
   * work that was lost, and those the viewer chose to skip stay as they ended.
   * The worker's specification is rebuilt from the configuration stored on the
   * job when it was submitted.
   */
  retry(runId: string, skippedFileIds: readonly string[] = []): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(async () => {
      const graphId = await this.requireRunRole(runId, "write");
      const launch = await this.storedLaunch(runId);
      const requeued = await this.requeueFailedDocuments(runId, graphId, skippedFileIds);
      if (requeued > 0) {
        await this.releaseFinishedJobService(launch.serviceIdentifier);
        await this.sql.query(launch.executeSql);
      }
      return this.status(runId);
    }, "Unable to retry Snowflake job");
  }

  recover(_runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.fail(notSupported("Infrastructure recovery is managed by Snowflake"));
  }

  /** Where each of a run's documents stands, from its row in the file queue. */
  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    return this.attempt(async () => {
      await this.requireRunRole(runId, "read");
      const rows = await this.sql.query(
        "SELECT FILE_ID, BLOB_PATH, STATUS, STAGE, ERROR FROM KG_JOB_FILE WHERE JOB_ID = ? ORDER BY BLOB_PATH",
        [runId.trim()],
      );
      return rows.map(documentStatusFromRow);
    }, "Unable to read the run's documents");
  }

  /** The tables hold one current version of a graph. */
  versions(graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return this.attempt(async () => {
      await this.roleOrFail(graphId.trim(), "read");
      return [];
    }, "Unable to read the graph's versions");
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return this.attempt(async () => {
      const name = validateGraphName(displayName);
      const target = graphId.trim();
      if (!target) {
        throw invalid("graph ID must not be empty");
      }
      await this.roleOrFail(target, "write");
      await this.writeDisplayName(target, name);
      return name;
    }, "Unable to rename Snowflake graph");
  }

  /**
   * Load a bounded review projection while retaining full graph counts.
   *
   * A graph is opened by an identifier the client holds, so listing it is no
   * boundary: the viewer's role on it gates the open itself. The location means
   * nothing here, since a graph is its rows rather than a directory.
   */
  loadGraph(_location: string, graphId?: string | null, _options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return this.attempt(async () => {
      const target = graphId?.trim();
      if (!target) {
        throw forbidden("A stored graph is opened through its run, or with its graph id.");
      }
      await this.roleOrFail(target, "read");
      return this.readGraph(target);
    }, "Unable to load Snowflake graph");
  }

  loadRunGraph(snapshot: RunSnapshot, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError> {
    return this.loadGraph("current", snapshot.graphId, options);
  }

  /** A graph here is rows in tables, so a run has no directory of files. */
  graphDirectory(): string | null {
    return null;
  }

  cluster(_namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.succeed(null);
  }

  nodeAssignments(): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.succeed([]);
  }

  fleetProfile(): Promise<FleetProfile | null> {
    return Promise.resolve(null);
  }

  /** The graph is in Snowflake already; publishing is how a fleet's graph gets here. */
  get snowflakePublishing(): SnowflakePublishing | null {
    return null;
  }

  // ---------------------------------------------------------------------------
  // Graph access

  requireGraph(graphId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError> {
    return this.attempt(() => this.roleOrFail(graphId.trim(), needed), "Unable to check access");
  }

  requireRun(runId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError> {
    return this.attempt(async () => {
      const graphId = await this.requireRunRole(runId, needed);
      return (await this.standing(graphId)).role!;
    }, "Unable to check access");
  }

  visibleGraphs(): Effect.Effect<ReadonlyMap<string, GraphRole>, ControlPlaneError> {
    return this.attempt(async () => {
      const viewer = await this.resolvedViewer();
      const graphs = await this.visibleGraphRows();
      const grants = await this.allGrants();
      const visible = new Map<string, GraphRole>();
      for (const row of graphs) {
        const graphId = text(row.graph_id);
        const record = { owner: text(row.owner) || null, grants: grants.get(graphId) ?? [] };
        const role = roleFor(record, viewer, SNOWFLAKE_ACCESS_POLICY);
        if (role) {
          visible.set(graphId, role);
        }
      }
      return visible;
    }, "Unable to list graphs");
  }

  graphAccess(graphId: string): Effect.Effect<GraphAccessView, ControlPlaneError> {
    return this.attempt(() => this.view(graphId.trim()), "Unable to read who can use this graph");
  }

  /** Give one person or role access to a graph, or change the level they have. Only its owner may. */
  shareGraph(graphId: string, principal: string, level: AccessLevel): Effect.Effect<GraphAccessView, ControlPlaneError> {
    return this.attempt(async () => {
      // Validated before any statement, so a refusal never reaches SQL.
      const grantee = shareTarget(principal);
      const target = graphId.trim();
      await this.roleOrFail(target, "owner");
      const { record } = await this.standing(target);
      if (grantee.type === "USER" && principalKey(record.owner) === grantee.name) {
        throw invalid("The owner already has full access.");
      }
      const viewer = await this.resolvedViewer();
      await this.sql.query(
        "MERGE INTO KG_GRAPH_ACL target USING (SELECT ? AS ID, ? AS GRAPH_ID, " +
          "? AS GRANTEE_TYPE, ? AS GRANTEE, ? AS ACCESS_LEVEL, ? AS GRANTED_BY) source " +
          "ON target.ID = source.ID " +
          "WHEN MATCHED THEN UPDATE SET ACCESS_LEVEL = source.ACCESS_LEVEL " +
          "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, GRANTEE_TYPE, GRANTEE, ACCESS_LEVEL, GRANTED_BY) " +
          "VALUES (source.ID, source.GRAPH_ID, source.GRANTEE_TYPE, source.GRANTEE, " +
          "source.ACCESS_LEVEL, source.GRANTED_BY)",
        [shareId(target, grantee.type, grantee.name), target, grantee.type, grantee.name, level, viewer.userName || SYSTEM_PRINCIPAL],
      );
      return this.view(target);
    }, "Unable to share graph");
  }

  /** The owner removes anyone; anyone else may remove only themselves. */
  unshareGraph(graphId: string, principal: string): Effect.Effect<GraphAccessView | null, ControlPlaneError> {
    return this.attempt(async () => {
      const target = graphId.trim();
      const viewer = await this.resolvedViewer();
      const named = principalKey(principal);
      const leaving = Boolean(viewer.userName) && named === principalKey(viewer.userName);
      const role = await this.roleOrFail(target, leaving ? "read" : "owner");
      if (leaving && role === "owner") {
        throw invalid("The owner cannot remove their own access.");
      }
      const roleName = principalRole(named);
      await this.sql.query("DELETE FROM KG_GRAPH_ACL WHERE ID = ?", [
        shareId(target, roleName ? "ROLE" : "USER", roleName ?? named),
      ]);
      return leaving ? null : this.view(target);
    }, "Unable to remove access");
  }

  /** What deleting a graph would remove, for the confirmation to name it. */
  deletionPreview(graphId: string): Effect.Effect<GraphDeletionPreview, ControlPlaneError> {
    return this.attempt(async () => {
      const target = graphId.trim();
      await this.roleOrFail(target, "owner");
      const runs = await this.graphRuns(target);
      const { record } = await this.standing(target);
      const named = await this.sql.query("SELECT DISPLAY_NAME FROM KG_GRAPH WHERE GRAPH_ID = ?", [target]);
      return {
        graphId: target,
        graphName: text(named[0]?.display_name) || target,
        runs: runs.length,
        active: runs.filter(isActiveRun).length,
        uploads: runs.filter(holdsUpload).length,
        sharedWith: record.grants.map((item) => item.principal).sort(),
      };
    }, "Unable to read what deleting this graph removes");
  }

  /**
   * People to offer when sharing: those the console has seen, and those on the
   * graphs the viewer can already see. Who else owns or holds what stays private.
   */
  people(): Effect.Effect<readonly string[], ControlPlaneError> {
    return this.attempt(async () => {
      const viewer = await this.resolvedViewer();
      const known = new Set<string>(Object.keys((await loadAccess(this.stateRoot)).people));
      const grants = await this.allGrants();
      for (const row of await this.visibleGraphRows()) {
        known.add(principalKey(text(row.owner)));
        for (const item of grants.get(text(row.graph_id)) ?? []) {
          known.add(item.principal);
        }
      }
      known.delete(principalKey(viewer.userName));
      known.delete(SYSTEM_PRINCIPAL);
      return [...known].filter(Boolean).sort();
    }, "Unable to list people");
  }

  /**
   * Remove a graph and everything recorded about it, owner only: its rows in
   * every graph table, the documents the app uploaded for its runs, the job
   * services Snowflake keeps after a run stops, and what the console holds for
   * it on disk. A run still under way has to be cancelled first.
   */
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return this.attempt(async () => {
      const target = graphId.trim();
      if (!target) {
        throw invalid("graph ID must not be empty");
      }
      await this.roleOrFail(target, "owner");
      const runs = await this.graphRuns(target);
      const active = runs.filter(isActiveRun);
      if (active.length > 0) {
        throw invalid(
          `This graph still has ${active.length === 1 ? "a run" : `${active.length} runs`} under way; cancel ${active.length === 1 ? "it" : "them"} before deleting the graph. Active runs: ${active.map((run) => text(run.id)).join(", ")}.`,
        );
      }
      const removed = await this.graphRowCounts(target);
      // Before the rows go: the uploads and job service of each run are addressed
      // from what those rows record, and nothing else remembers them.
      await this.releaseGraphStorage(runs);
      for (const table of GRAPH_DATA_TABLES) {
        await this.sql.query(`DELETE FROM ${table} WHERE GRAPH_ID = ?`, [target]);
      }
      const runIds = runs.map((run) => text(run.id));
      await removeGraphRecords(this.stateRoot, await graphFootprint(this.stateRoot, target, runIds));
      return { ...removed, runs: runs.length, uploads: runs.filter(holdsUpload).length };
    }, "Unable to delete graph");
  }

  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError> {
    return this.attempt(async () => {
      await this.checkRequest(request);
      const prepared = await this.prepare(request);
      const config = await this.effectiveConfig(prepared.request);
      return stringifyYaml(redactedConfig(config), { sortMapEntries: false });
    }, "Unable to preview configuration");
  }

  /**
   * Return the width the graph tables store, when they agree on one, so the
   * form offers an embedding model whose run can actually be written.
   */
  graphEmbeddingWidth(): Effect.Effect<number | null, ControlPlaneError> {
    return this.attempt(async () => {
      try {
        const widths = new Set(Object.values(await this.embeddingWidths()));
        return widths.size === 1 ? [...widths][0]! : null;
      } catch {
        return null;
      }
    }, "Unable to read the graph tables' embedding width");
  }

  /** Stages in the app's schema that documents can be read from, upload stage first. */
  documentStages(): Effect.Effect<readonly StageChoice[], ControlPlaneError> {
    return this.attempt(() => this.documentStageList(), "Unable to list the app's stages");
  }

  /**
   * Folders of one document stage the viewer may run on. The stage is listed
   * with the app's rights, so folders are filtered by the rule submission
   * applies before they are offered.
   */
  stageFolders(stage: string): Effect.Effect<readonly StageFolder[], ControlPlaneError> {
    return this.attempt(async () => {
      const known = await this.documentStageList();
      const wanted = stage.trim().replace(/^@+/, "").toUpperCase();
      const choice = known.find((item) => item.stage.replace(/^@+/, "").toUpperCase() === wanted);
      if (!choice) {
        throw invalid("Choose one of the app's document stages");
      }
      const viewer = await this.resolvedViewer();
      const names: string[] = [];
      for await (const row of this.sql.stream(`LIST ${stageLocation(choice.stage)}`)) {
        names.push(String(row.name ?? ""));
        // A request ends at the ingress after about 90 seconds; past this many
        // objects the folders are offered from what was read so far.
        if (names.length >= MAX_FOLDER_LISTING) {
          break;
        }
      }
      return readableStageFolders(names, choice.stage, viewer);
    }, "Unable to list the stage's folders");
  }

  /** The account's Cortex models: those that can extract a graph, and embeddings of known width. */
  modelChoices(): Effect.Effect<ModelChoices, ControlPlaneError> {
    return this.attempt(
      async () => cortexModelChoices(await this.showNames("SHOW MODELS IN SNOWFLAKE.MODELS")),
      "Unable to list the Cortex models",
    );
  }

  private async documentStageList(): Promise<StageChoice[]> {
    const { database, schema } = await this.namespace();
    const rows = await this.discovered("SHOW STAGES rows", async () => {
      try {
        return await this.sql.query("SHOW STAGES");
      } catch {
        return [] as SqlRow[];
      }
    });
    return documentStageChoices(rows, database, schema);
  }

  // ---------------------------------------------------------------------------
  // Identity and authorisation

  private resolvedViewer(): Promise<Viewer> {
    this.viewerPromise ??= (async () => {
      const base = this.options.viewer ?? unidentifiedViewer();
      if (!base.userName) {
        return base;
      }
      const roles = await this.viewerRoles(base.userName);
      return { ...base, roles: normalizeRoles([...base.roles, ...roles]) };
    })();
    return this.viewerPromise;
  }

  /**
   * Return the roles Snowflake grants the viewer.
   *
   * Asked of a caller's-rights session when the request carries the viewer's
   * token, since CURRENT_ROLE in the app's own session names the app. Without
   * one, the grants to the named user are read instead. Either answer is cached
   * briefly so a page of requests costs one lookup.
   */
  private async viewerRoles(userName: string): Promise<string[]> {
    const cache = this.cache();
    const key = `roles:${userName}`;
    const hit = cache.get(key);
    if (hit && Date.now() - hit.at < ROLE_CACHE_MS) {
      return hit.value as string[];
    }
    let roles: string[] = [];
    const token = this.options.callerToken?.trim();
    if (token) {
      try {
        const lookup = this.options.callerRoles ?? ((value: string) => callerAvailableRoles(this.options.connection, value));
        roles = normalizeRoles(await lookup(token));
      } catch (error) {
        console.warn(`Caller's-rights role lookup failed: ${messageOf(error)}`);
      }
    }
    if (roles.length === 0) {
      roles = await this.grantedRoles(userName);
    }
    cache.set(key, { at: Date.now(), value: roles });
    return roles;
  }

  /**
   * Read the roles granted to one named user.
   *
   * SHOW GRANTS takes an identifier, not a bindable value, and usernames come
   * from the identity provider, so the name is escaped for double quotes.
   */
  private async grantedRoles(userName: string): Promise<string[]> {
    try {
      const rows = await this.sql.query(`SHOW GRANTS TO USER "${quotedIdentifier(userName)}"`);
      return normalizeRoles(rows.map((row) => row.role ?? row.name));
    } catch {
      return [];
    }
  }

  /** The graph's owner and grants, and whether the account has heard of it at all. */
  private async standing(graphId: string): Promise<{ record: GraphAccessRecord; known: boolean; role: GraphRole | null }> {
    const viewer = await this.resolvedViewer();
    const owners = await this.sql.query("SELECT OWNER FROM KG_GRAPH WHERE GRAPH_ID = ?", [graphId]);
    const grants = await this.sql.query(
      "SELECT GRAPH_ID, GRANTEE_TYPE, GRANTEE, ACCESS_LEVEL, GRANTED_BY, CREATED_AT FROM KG_GRAPH_ACL WHERE GRAPH_ID = ?",
      [graphId],
    );
    const record = { owner: text(owners[0]?.owner) || null, grants: grants.map(grantFromRow) };
    const known =
      owners.length > 0 ||
      (await this.sql.query("SELECT 1 AS KNOWN FROM KG_JOB WHERE GRAPH_ID = ? LIMIT 1", [graphId])).length > 0;
    // An unclaimed graph is open to everyone, which must not make every id a
    // graph: one the account has never heard of is nobody's to open.
    return { record, known, role: known ? roleFor(record, viewer, SNOWFLAKE_ACCESS_POLICY) : null };
  }

  private async roleOrFail(graphId: string, needed: GraphRole): Promise<GraphRole> {
    const { role } = await this.standing(graphId);
    const refusal = accessRefusal(role, needed);
    if (refusal) {
      throw forbidden(refusal);
    }
    return role!;
  }

  /** Check the viewer's role on the graph a run belongs to, and return that graph. */
  private async requireRunRole(runId: string, needed: GraphRole): Promise<string> {
    const graphId = await this.runGraphId(runId);
    if (!graphId) {
      throw notFound(`Unknown Snowflake job: ${runId}`);
    }
    await this.roleOrFail(graphId, needed);
    return graphId;
  }

  private async view(graphId: string): Promise<GraphAccessView> {
    const { record, role } = await this.standing(graphId);
    const refusal = accessRefusal(role, "read");
    if (refusal) {
      throw forbidden(refusal);
    }
    return {
      graphId,
      owner: record.owner,
      role: role!,
      shares: [...record.grants].sort((left, right) => left.principal.localeCompare(right.principal)),
    };
  }

  /** Every grant in the account. The table holds a row per share, so reading it whole stays cheap. */
  private async allGrants(): Promise<Map<string, GraphGrant[]>> {
    const rows = await this.sql.query(
      "SELECT GRAPH_ID, GRANTEE_TYPE, GRANTEE, ACCESS_LEVEL, GRANTED_BY, CREATED_AT FROM KG_GRAPH_ACL",
    );
    const grants = new Map<string, GraphGrant[]>();
    for (const row of rows) {
      const graphId = text(row.graph_id);
      grants.set(graphId, [...(grants.get(graphId) ?? []), grantFromRow(row)]);
    }
    return grants;
  }

  /** The graphs with a run that the viewer can see, each once with its owner. */
  private async visibleGraphRows(): Promise<SqlRow[]> {
    const visible = visibleGraphPredicate(await this.resolvedViewer(), "G");
    return this.sql.query(
      "SELECT DISTINCT J.GRAPH_ID, G.OWNER FROM KG_JOB J LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID " +
        `WHERE ${visible.sql}`,
      visible.parameters,
    );
  }

  /**
   * Resolve a folder on this container's disk and require it to be the viewer's
   * own upload. The bytes are read with the console's rights and end up in the
   * viewer's stage folder, so a folder anyone could name would hand one viewer
   * another's documents.
   */
  private async ownUploadFolder(named: string): Promise<string> {
    const folder = await resolveUploadFolder(this.stateRoot, named);
    const jobId = path.relative(await realpath(uploadsRoot(this.stateRoot)), folder).split(path.sep)[0] ?? "";
    const viewer = await this.resolvedViewer();
    const me = principalKey(viewer.userName);
    if (!me || (await loadAccess(this.stateRoot)).uploads[jobId]?.owner !== me) {
      throw forbidden(FOREIGN_UPLOAD_FOLDER);
    }
    return folder;
  }

  /**
   * Refuse to read another viewer's uploads out of a shared stage.
   *
   * The app lists a stage with its own privileges, so the resolved path is
   * authorised rather than either field: a stage identifier can carry a path,
   * and the stage root contains every viewer's folder.
   */
  private async requireReadableStage(stage: string, prefix: string): Promise<void> {
    const viewer = await this.resolvedViewer();
    const relative = stageRelativePath(stage, prefix);
    if (!relative) {
      // The stage root holds every viewer's upload folder, so a run always names a folder.
      throw invalid("Choose a folder in the stage for the documents.");
    }
    if (!mayReadStagePath(viewer, relative)) {
      throw forbidden(FOREIGN_UPLOADS);
    }
  }

  /**
   * What a run, a preflight or a preview may name: a run id nobody has used,
   * since every statement that follows is keyed on it and a taken one would land
   * in someone's run, and a graph the viewer may write, or one nobody has yet.
   * A preflight or a preview would otherwise answer questions about a graph that
   * is not theirs. Tells the caller whether the graph exists already.
   */
  private async checkRequest(request: IngestionRequest): Promise<{ graphId: string; jobId: string; existing: boolean }> {
    const graphId = request.graphId.trim();
    const jobId = request.jobId.trim();
    if (!graphId || !jobId) {
      throw invalid("A run needs a graph ID and a run ID");
    }
    if (request.revision) {
      throw notSupported("A new version is built by running the graph again inside Snowflake");
    }
    if (await this.runGraphId(jobId)) {
      throw invalid("That run id is already taken; start the run again for a new one.");
    }
    const { known, role } = await this.standing(graphId);
    const refusal = known ? accessRefusal(role, "write") : null;
    if (refusal) {
      throw forbidden(refusal);
    }
    return { graphId, jobId, existing: known };
  }

  private async runGraphId(runId: string): Promise<string> {
    const rows = await this.sql.query("SELECT GRAPH_ID FROM KG_JOB WHERE ID = ?", [runId.trim()]);
    return String(rows[0]?.graph_id ?? "");
  }

  // ---------------------------------------------------------------------------
  // Preparing a request

  /**
   * Turn what the form sent into the request a Snowflake run actually makes.
   *
   * Inside Snowflake there is nothing for a viewer to choose about where the
   * graph is written or how the worker authenticates: the app's own namespace
   * is the destination and the worker holds a Snowflake-managed OAuth token. So
   * account coordinates and credentials in the request are ignored and filled
   * from the session, and SPCS objects left blank are discovered.
   */
  private async prepare(request: IngestionRequest): Promise<PreparedRequest> {
    const viewer = await this.resolvedViewer();
    const namespace = await this.namespace();
    const baseConfigPath = this.baseConfigPath(request.baseConfigPath);
    const base = await loadBaseConfig(baseConfigPath);
    const defaults = asRecord(base.snowflake);

    let sourceKind = request.sourceKind;
    let source: Record<string, unknown>;
    let uploadDirectory: string | null = null;
    if (sourceKind === "upload") {
      uploadDirectory = await this.ownUploadFolder(String(request.source.path ?? ""));
      // Derived from the viewer rather than chosen: the app lists stages with
      // its own privileges, so a folder anyone may name would let one viewer
      // ingest another's uploads.
      source = {
        kind: "snowflake_stage",
        stage: qualifiedStage(UPLOAD_STAGE, namespace.database, namespace.schema),
        prefix: viewerStagePrefix(viewer, request.jobId),
      };
      sourceKind = "snowflake_stage";
    } else if (sourceKind === "snowflake_stage") {
      source = {
        kind: "snowflake_stage",
        stage: String(request.source.stage ?? "").trim(),
        prefix: String(request.source.prefix ?? "").trim(),
      };
      if (!source.stage) {
        throw invalid("Choose the Snowflake stage that holds the documents");
      }
    } else {
      throw invalid("Inside Snowflake, documents come from an upload or a Snowflake stage");
    }

    const warehouse = namespace.warehouse || text(defaults.warehouse);
    if (!warehouse) {
      throw invalid("No warehouse is configured for bulk loading. Set FLAKEGRAPH_SNOWFLAKE_WAREHOUSE on the app.");
    }
    const bulkStage = text(defaults.bulk_stage) || (await this.preferredStage(["KG_LOAD_STAGE", "KG_BULK_STAGE"]));
    if (!bulkStage) {
      throw invalid("No internal stage is available for bulk loading. Create KG_LOAD_STAGE in the app's schema.");
    }
    const runtimeOptions = await this.runtimeOptions(request.runtimeOptions, defaults);
    return {
      namespace,
      uploadDirectory,
      request: {
        ...request,
        runtime: "snowflake",
        sourceKind,
        source,
        baseConfigPath,
        output: {
          kind: "snowflake",
          workspacePath: path.join(this.stateRoot, "artifacts", request.graphId, request.jobId),
          snowflake: {
            account: ACTIVE_SNOWFLAKE_SESSION,
            user: ACTIVE_SNOWFLAKE_SESSION,
            database: namespace.database,
            schema: namespace.schema,
            warehouse,
            bulkStage: qualifiedStage(bulkStage, namespace.database, namespace.schema),
            role: null,
            host: null,
            authenticator: "oauth",
            credentialEnvironmentVariable: null,
            credentialField: null,
          },
        },
        cacheProvider: "snowflake",
        runtimeOptions,
      },
    };
  }

  /**
   * Accept the SPCS choices a viewer may make and discover the rest.
   *
   * Each of these names an object that must already exist, and a mistyped pool
   * or tag fails only once the service has been created and immediately dies,
   * so blanks are filled from what Snowflake reports rather than guessed.
   */
  private async runtimeOptions(
    requested: Record<string, unknown>,
    defaults: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const options: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(requested)) {
      if (!VIEWER_RUNTIME_OPTIONS.has(key)) {
        continue;
      }
      if (typeof value === "string") {
        if (value.trim()) {
          options[key] = value.trim();
        }
      } else if (value !== null && value !== undefined) {
        options[key] = value;
      }
    }
    const chosen = (key: string) => text(options[key]) || text(defaults[key]);
    const computePool = chosen("compute_pool") || (await this.computePools())[0] || "";
    const imageRepository = chosen("image_repository") || (await this.imageRepositories())[0] || "";
    const imageName = chosen("image_name") || (imageRepository ? (await this.images(imageRepository))[0] : "") || "";
    const imageDigest = chosen("image_digest");
    for (const [key, value] of [
      ["compute_pool", computePool],
      ["image_repository", imageRepository],
      ["image_name", imageName],
      ["image_digest", imageDigest],
    ] as const) {
      if (value) {
        options[key] = value;
      }
    }
    // A job service has its node to itself and SPCS bills the node either way,
    // so the container is sized from the pool rather than from a fixed default
    // that fits the smallest machine anyone might use.
    if (computePool && !RESOURCE_OPTIONS.some((key) => key in options)) {
      Object.assign(options, containerResources(await this.computePoolCapacity(computePool)));
    }
    return options;
  }

  /**
   * Accept the base configuration the form names only if it is one of the app's
   * own profiles. Its contents are parsed and rendered back to the browser, so
   * an arbitrary path would make any readable file a disclosure.
   */
  private baseConfigPath(requested: string | null): string {
    const configs = path.join(this.repositoryRoot, "configs");
    const fallback = path.join(configs, "app-defaults.yaml");
    const candidate = requested?.trim() ? path.resolve(requested.trim()) : fallback;
    const relative = path.relative(configs, candidate);
    const chosen = !relative || relative.startsWith("..") || path.isAbsolute(relative) ? fallback : candidate;
    if (!existsSync(chosen)) {
      throw invalid(
        `Base configuration not found at ${chosen}. The deployed app must ship configs/app-defaults.yaml at its root.`,
      );
    }
    return chosen;
  }

  /**
   * Build the run configuration with its ontology carried inline.
   *
   * The worker container receives only this configuration and its image, so an
   * ontology referenced by a path in the app's bundle would dangle there and the
   * worker would abort before claiming a file.
   */
  private async effectiveConfig(request: IngestionRequest): Promise<Record<string, unknown>> {
    const config = await buildRunConfig(request);
    const ontology = asRecord(config.ontology);
    const file = text(ontology.profile_path);
    if (!file) {
      return config;
    }
    if (!existsSync(file)) {
      throw invalid(`Ontology profile ${path.basename(file)} named by the base configuration is not in the deployed app`);
    }
    const loaded: unknown = parseYaml(await readFile(file, "utf8")) ?? {};
    if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
      throw invalid(`Ontology file must contain a mapping: ${path.basename(file)}`);
    }
    config.ontology = { ...ontology, profile: loaded, profile_path: null };
    return config;
  }

  // ---------------------------------------------------------------------------
  // Discovery

  private cache(): Map<string, Cached> {
    let cache = discoveryCaches.get(this.sql);
    if (!cache) {
      cache = new Map();
      discoveryCaches.set(this.sql, cache);
    }
    return cache;
  }

  private async discovered<T>(key: string, load: () => Promise<T>): Promise<T> {
    const cache = this.cache();
    const hit = cache.get(key);
    if (hit && Date.now() - hit.at < DISCOVERY_CACHE_MS) {
      return hit.value as T;
    }
    const value = await load();
    cache.set(key, { at: Date.now(), value });
    return value;
  }

  /**
   * The app's own database, schema and warehouse. App Runtime names them in the
   * environment; the session is asked only for what it leaves out.
   */
  private namespace(): Promise<SnowflakeNamespace> {
    this.namespacePromise ??= (async () => {
      const env = this.options.connection;
      let { database, schema, warehouse } = env;
      if (!database || !schema || !warehouse) {
        const context = await this.discovered("context", async () => {
          try {
            const rows = await this.sql.query(
              "SELECT CURRENT_DATABASE() AS DATABASE_NAME, CURRENT_SCHEMA() AS SCHEMA_NAME, " +
                "CURRENT_WAREHOUSE() AS WAREHOUSE_NAME",
            );
            return rows[0] ?? {};
          } catch {
            return {} as SqlRow;
          }
        });
        database ||= text(context.database_name) || null;
        schema ||= text(context.schema_name) || null;
        warehouse ||= text(context.warehouse_name) || null;
      }
      if (!database || !schema || !IDENTIFIER.test(database) || !IDENTIFIER.test(schema)) {
        throw invalid("The app's Snowflake database and schema are unknown; set SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA.");
      }
      return { database, schema, warehouse };
    })();
    return this.namespacePromise;
  }

  /**
   * Return the name column of a SHOW statement, or nothing on failure.
   *
   * Discovery is a convenience: a role that cannot list an object type still
   * gets a precise error naming the field that was left blank.
   */
  private showNames(statement: string): Promise<string[]> {
    return this.discovered(statement, async () => {
      try {
        const rows = await this.sql.query(statement);
        return [...new Set(rows.map((row) => text(row.name)).filter(Boolean))].sort();
      } catch {
        return [];
      }
    });
  }

  private computePools(): Promise<string[]> {
    return this.showNames("SHOW COMPUTE POOLS");
  }

  private listStages(): Promise<string[]> {
    return this.showNames("SHOW STAGES");
  }

  /** Pick a stage by preference, falling back to the first that exists. */
  private async preferredStage(prefer: readonly string[]): Promise<string> {
    const stages = await this.listStages();
    return prefer.find((name) => stages.includes(name)) ?? stages[0] ?? "";
  }

  /** Image repositories as the fully qualified names SPCS expects. */
  private imageRepositories(): Promise<string[]> {
    return this.discovered("SHOW IMAGE REPOSITORIES", async () => {
      try {
        const rows = await this.sql.query("SHOW IMAGE REPOSITORIES");
        const names = rows
          .map((row) => {
            const name = text(row.name);
            const database = text(row.database_name);
            const schema = text(row.schema_name);
            return name && database && schema ? `${database}.${schema}.${name}` : name;
          })
          .filter(Boolean);
        return [...new Set(names)].sort();
      } catch {
        return [];
      }
    });
  }

  /**
   * Return image:tag pairs held in one repository, most recent first.
   *
   * Choosing the image from what has been pushed avoids a service that starts
   * and immediately dies pulling something that was never published. Order is
   * the other half of that: sorted by name, the default would be the oldest tag
   * ever pushed, and a run on an image that old fails on startup with nothing to
   * explain it. Newest first makes the default the latest push.
   */
  private images(repository: string): Promise<string[]> {
    const target = repository.trim();
    if (!/^[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$/.test(target)) {
      return Promise.resolve([]);
    }
    return this.discovered(`SHOW IMAGES IN IMAGE REPOSITORY ${target}`, async () => {
      let rows: SqlRow[];
      try {
        rows = await this.sql.query(`SHOW IMAGES IN IMAGE REPOSITORY ${target}`);
      } catch {
        return [];
      }
      const images: Array<[string, string]> = [];
      for (const row of rows) {
        const name = text(row.image_name);
        const created = text(row.created_on);
        for (const tag of text(row.tags).split(",").map((part) => part.trim()).filter(Boolean)) {
          images.push([created, `${name}:${tag}`]);
        }
      }
      // A secondary sort on the reference keeps the order total when a
      // repository reports no push time.
      images.sort((left, right) => compareText(right[0], left[0]) || compareText(right[1], left[1]));
      return [...new Set(images.map(([, image]) => image))];
    });
  }

  /**
   * Return the vCPU and memory one node of a pool provides, or nothing when
   * either cannot be read, so the caller keeps its defaults rather than guess.
   */
  private computePoolCapacity(pool: string): Promise<{ vcpu?: number; memoryGib?: number }> {
    const name = pool.trim();
    if (!IDENTIFIER.test(name)) {
      return Promise.resolve({});
    }
    return this.discovered(`capacity:${name}`, async () => {
      try {
        const pools = await this.sql.query(`SHOW COMPUTE POOLS LIKE '${name}'`);
        const family = text(pools[0]?.instance_family).toUpperCase();
        if (!family) {
          return {};
        }
        const families = await this.sql.query("SHOW COMPUTE POOL INSTANCE FAMILIES");
        const match = families.find((row) => text(row.name).toUpperCase() === family);
        if (!match) {
          return {};
        }
        const vcpu = Number(match.vcpu);
        const memoryGib = Number(match.memory_gib);
        return Number.isFinite(vcpu) && Number.isFinite(memoryGib) ? { vcpu, memoryGib } : {};
      } catch {
        return {};
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Preflight

  private async runPreflight(request: IngestionRequest): Promise<PreflightResult> {
    await this.checkRequest(request);
    const errors: string[] = [];
    const warnings: string[] = [];
    const checks: Array<Record<string, unknown>> = [];
    const check = (name: string, ok: boolean, detail?: string) =>
      checks.push({ name, ok, ...(detail ? { detail } : {}) });

    let tables: Set<string> | null = null;
    try {
      const rows = await this.sql.query("SHOW TABLES");
      tables = new Set(rows.map((row) => text(row.name).toUpperCase()));
      check("Snowflake session", true);
    } catch (error) {
      errors.push(`The app's Snowflake session cannot query its schema: ${messageOf(error)}`);
      check("Snowflake session", false, messageOf(error));
    }
    if (tables) {
      const missing = REQUIRED_TABLES.filter((table) => !tables.has(table));
      errors.push(...missing.map((table) => `Required Snowflake table is not visible: ${table}`));
      check(
        "Queue and graph tables",
        missing.length === 0,
        missing.length ? `Missing ${missing.join(", ")}` : undefined,
      );
    }

    let prepared: PreparedRequest | null = null;
    try {
      prepared = await this.prepare(request);
      check("Run configuration", true);
    } catch (error) {
      errors.push(messageOf(error));
      check("Run configuration", false, messageOf(error));
    }
    if (!prepared) {
      return { ok: false, errors, warnings, checks };
    }

    const source = prepared.request.source;
    if (prepared.uploadDirectory) {
      const files = await listLocalObjects(prepared.uploadDirectory, 1);
      const stages = await this.listStages();
      const stageMissing = stages.length > 0 && !stages.includes(UPLOAD_STAGE);
      if (files.length === 0) {
        errors.push("The upload holds no supported documents");
      }
      if (stageMissing) {
        errors.push(missingUploadStage(stages));
      }
      check("Uploaded documents", files.length > 0 && !stageMissing);
    } else {
      try {
        await this.requireReadableStage(String(source.stage), String(source.prefix));
        const objects = await this.listStage(String(source.stage), String(source.prefix), 1);
        if (objects.length === 0) {
          errors.push("The selected Snowflake stage prefix contains no supported files");
        }
        check("Stage documents", objects.length > 0);
      } catch (error) {
        errors.push(`Stage cannot be listed: ${messageOf(error)}`);
        check("Stage documents", false, messageOf(error));
      }
    }

    let config: Record<string, unknown> | null = null;
    try {
      config = await this.effectiveConfig(prepared.request);
      buildSpcsLaunch(config);
      check("Worker launch", true);
    } catch (error) {
      errors.push(`SPCS launch configuration is not ready: ${messageOf(error)}`);
      check("Worker launch", false, messageOf(error));
    }

    const dimension = embeddingDimension(
      prepared.request.embedding.provider,
      prepared.request.embedding.model,
      prepared.request.embedding.dimension,
    );
    try {
      const mismatched = Object.entries(await this.embeddingWidths()).filter(([, width]) => width !== dimension);
      if (mismatched.length > 0) {
        const stored = [...new Set(mismatched.map(([, width]) => width))].sort((a, b) => a - b);
        errors.push(
          `Embedding dimension ${dimension} does not fit the graph tables, which store [${stored.join(", ")}]: ` +
            `${mismatched.map(([table]) => table).sort().join(", ")}. Choose a model of the stored width, ` +
            "or recreate the tables at this width.",
        );
      }
      check("Embedding width", mismatched.length === 0, `${dimension}`);
    } catch (error) {
      errors.push(`Embedding dimension cannot be checked: ${messageOf(error)}`);
      check("Embedding width", false, messageOf(error));
    }
    return { ok: errors.length === 0, errors, warnings, checks };
  }

  /**
   * Return the vector width each graph table already stores.
   *
   * A VECTOR column fixes its width when the table is created and the writer is
   * the last stage of a run, so a mismatch is otherwise discovered only after
   * the whole Cortex spend of the run. A width that cannot be read is an error,
   * not agreement: a guard that exists to stop an expensive run must not pass a
   * run it could not check.
   */
  private async embeddingWidths(): Promise<Record<string, number>> {
    const widths: Record<string, number> = {};
    for (const row of await this.sql.query("SHOW COLUMNS LIKE 'EMBEDDING'")) {
      const table = text(row.table_name).toUpperCase();
      // A schema is shared; another table's embedding column says nothing about
      // where this graph is written.
      if (!EMBEDDING_TABLES.has(table)) {
        continue;
      }
      const declared = vectorWidth(row.data_type);
      if (declared === null) {
        throw new Error(`${table}.EMBEDDING does not report a vector width: ${JSON.stringify(row.data_type)}`);
      }
      widths[table] = declared;
    }
    return widths;
  }

  // ---------------------------------------------------------------------------
  // Submission

  private async runSubmit(request: IngestionRequest): Promise<RunSnapshot> {
    // Authorisation precedes every other step, so a refusal costs nothing.
    const { graphId, jobId, existing } = await this.checkRequest(request);
    const prepared = await this.prepare({ ...request, graphId, jobId });
    const stage = String(prepared.request.source.stage);
    const prefix = String(prepared.request.source.prefix);
    await this.requireReadableStage(stage, prefix);
    const config = await this.effectiveConfig(prepared.request);
    const launch = buildSpcsLaunch(config);
    const graphName = request.graphName?.trim() ? validateGraphName(request.graphName) : null;

    if (prepared.uploadDirectory) {
      // Checked before the first byte moves: Snowflake reports a PUT to a
      // missing stage from its storage client, naming neither stage nor problem.
      const stages = await this.listStages();
      if (stages.length > 0 && !stages.includes(UPLOAD_STAGE)) {
        throw invalid(missingUploadStage(stages));
      }
      await (this.options.putFolder ?? putFolderToStage)(this.sql, prepared.uploadDirectory, stage, prefix);
    }

    await this.claimGraph(graphId);
    // Two people can name one new graph at once. The claim keeps the first, so
    // the other learns here, before anything is queued under a graph not theirs.
    if (!existing) {
      await this.roleOrFail(graphId, "write");
    }
    if (graphName) {
      await this.writeDisplayName(graphId, graphName);
    }
    // What is stored is what the worker receives, minus credentials: a retry
    // rebuilds the worker from it, and it is shown to anyone who can see the run.
    const configJson = JSON.stringify(sortKeys(redactedConfig(launch.runtimeConfig)));
    await this.sql.query(
      "MERGE INTO KG_JOB target USING (" +
        "SELECT ? AS ID, ? AS GRAPH_ID, PARSE_JSON(?) AS CONFIG" +
        ") source ON target.ID = source.ID " +
        "WHEN NOT MATCHED THEN INSERT (ID, GRAPH_ID, STATUS, CONFIG) " +
        "VALUES (source.ID, source.GRAPH_ID, 'PENDING', source.CONFIG)",
      [jobId, graphId, configJson],
    );
    try {
      let submitted = 0;
      let batch: SubmissionRow[] = [];
      for await (const row of this.sql.stream(`LIST ${stageLocation(stage, prefix)}`)) {
        const item = sourceObjectFromListRow(row);
        if (!item) {
          continue;
        }
        batch.push(submissionRow(jobId, graphId, stage, item));
        if (batch.length >= SUBMISSION_BATCH_ROWS) {
          await this.mergeSubmissionBatch(batch);
          submitted += batch.length;
          batch = [];
        }
      }
      if (batch.length > 0) {
        await this.mergeSubmissionBatch(batch);
        submitted += batch.length;
      }
      if (submitted === 0) {
        throw new Error("The selected Snowflake stage prefix contains no supported files");
      }
      await this.releaseFinishedJobService(launch.serviceIdentifier);
      await this.sql.query(launch.executeSql);
    } catch (error) {
      // The cause is recorded, not replaced: a fixed message forces whoever
      // reads the failed run to reproduce it before they can guess at it.
      await this.sql.query(
        "UPDATE KG_JOB SET STATUS = 'FAILED', " +
          "ERROR = OBJECT_CONSTRUCT('message', ?, 'type', ?), " +
          "UPDATED_AT = CURRENT_TIMESTAMP() WHERE ID = ?",
        [`SPCS job submission failed: ${messageOf(error)}`.slice(0, 2000), errorType(error), jobId],
      );
      throw error;
    }
    return this.status(jobId);
  }

  /** Merge one bounded batch of queue rows, idempotently, into KG_JOB_FILE. */
  private async mergeSubmissionBatch(rows: readonly SubmissionRow[]): Promise<void> {
    const values = rows.map(() => "(?, ?, ?, ?, ?, ?, ?)").join(", ");
    await this.sql.query(
      "MERGE INTO KG_JOB_FILE target USING (" +
        "SELECT COLUMN1 AS ID, COLUMN2 AS JOB_ID, COLUMN3 AS GRAPH_ID, COLUMN4 AS FILE_ID, " +
        "COLUMN5 AS SOURCE_URI, COLUMN6 AS BLOB_PATH, COLUMN7 AS CHECKSUM " +
        `FROM VALUES ${values}) source ON target.ID = source.ID ` +
        "WHEN MATCHED THEN UPDATE SET SOURCE_URI = source.SOURCE_URI, " +
        "BLOB_PATH = source.BLOB_PATH, CHECKSUM = source.CHECKSUM, " +
        "UPDATED_AT = CURRENT_TIMESTAMP() " +
        "WHEN NOT MATCHED THEN INSERT " +
        "(ID, JOB_ID, GRAPH_ID, FILE_ID, SOURCE_URI, BLOB_PATH, CHECKSUM, STATUS) " +
        "VALUES (source.ID, source.JOB_ID, source.GRAPH_ID, source.FILE_ID, " +
        "source.SOURCE_URI, source.BLOB_PATH, source.CHECKSUM, 'QUEUED')",
      rows.flat(),
    );
  }

  /**
   * Record the submitting viewer as the graph's owner, once.
   *
   * Claimed at submission rather than completion so a run that fails still
   * belongs to whoever started it. An existing owner is never overwritten.
   */
  private async claimGraph(graphId: string): Promise<void> {
    const viewer = await this.resolvedViewer();
    if (!viewer.userName) {
      return;
    }
    await this.sql.query(
      "MERGE INTO KG_GRAPH target USING (SELECT ? AS GRAPH_ID, ? AS OWNER) source " +
        "ON target.GRAPH_ID = source.GRAPH_ID " +
        "WHEN MATCHED AND target.OWNER IS NULL THEN UPDATE SET OWNER = source.OWNER, " +
        "UPDATED_AT = CURRENT_TIMESTAMP() " +
        "WHEN NOT MATCHED THEN INSERT (GRAPH_ID, DISPLAY_NAME, OWNER) " +
        "VALUES (source.GRAPH_ID, source.GRAPH_ID, source.OWNER)",
      [graphId, viewer.userName],
    );
  }

  private async writeDisplayName(graphId: string, name: string): Promise<void> {
    await this.sql.query(
      "MERGE INTO KG_GRAPH target USING (SELECT ? AS GRAPH_ID, ? AS DISPLAY_NAME) source " +
        "ON target.GRAPH_ID = source.GRAPH_ID " +
        "WHEN MATCHED THEN UPDATE SET DISPLAY_NAME = source.DISPLAY_NAME, " +
        "UPDATED_AT = CURRENT_TIMESTAMP() " +
        "WHEN NOT MATCHED THEN INSERT (GRAPH_ID, DISPLAY_NAME) " +
        "VALUES (source.GRAPH_ID, source.DISPLAY_NAME)",
      [graphId, name],
    );
  }

  /**
   * Free a finished job service's name so the same run can start again.
   *
   * Snowflake keeps a completed job service until it is dropped, so a second
   * start under the same name fails with a bare "already exists". A service
   * that is still working is never dropped; that would destroy the run the
   * caller is trying to repeat.
   */
  private async releaseFinishedJobService(identifier: string): Promise<void> {
    const [database, schema, name] = identifier.split(".");
    if (!database || !schema || !name || ![database, schema, name].every((part) => IDENTIFIER.test(part))) {
      throw new Error(`Refusing to inspect an unsafe service name: ${identifier}`);
    }
    const rows = await this.sql.query(`SHOW SERVICES LIKE '${name}' IN SCHEMA ${database}.${schema}`);
    if (rows.length === 0) {
      return;
    }
    const state = text(rows[0]?.status).toUpperCase();
    if (state && !TERMINAL_SERVICE_STATUSES.has(state)) {
      throw new Error(
        `A job for this graph is already ${state.toLowerCase()} as ${identifier}. ` +
          "Wait for it to finish or cancel it before submitting again.",
      );
    }
    await this.sql.query(`DROP SERVICE IF EXISTS ${identifier}`);
  }

  /** Rebuild the worker a run was started with, from the configuration it stored. */
  private async storedLaunch(runId: string): Promise<SpcsLaunch> {
    const rows = await this.sql.query("SELECT CONFIG FROM KG_JOB WHERE ID = ?", [runId.trim()]);
    if (rows.length === 0) {
      throw notFound("This run is no longer in the queue");
    }
    const config = variantMapping(rows[0]?.config);
    if (text(asRecord(config.job).job_id) !== runId.trim()) {
      throw invalid("This run does not record the configuration a retry needs");
    }
    try {
      return buildSpcsLaunch(structuredClone(config));
    } catch (error) {
      throw invalid(`This run does not record the launch a retry needs: ${messageOf(error)}`);
    }
  }

  /** Return failed documents to the queue and make the job claimable again. */
  private async requeueFailedDocuments(runId: string, graphId: string, skippedFileIds: readonly string[]): Promise<number> {
    const skipped = skippedFileIds.length ? ` AND FILE_ID NOT IN (${skippedFileIds.map(() => "?").join(", ")})` : "";
    await this.sql.query(
      "UPDATE KG_JOB_FILE SET STATUS = 'QUEUED', WORKER_ID = NULL, LEASE_UNTIL = NULL, " +
        "ERROR = NULL, UPDATED_AT = CURRENT_TIMESTAMP() " +
        `WHERE JOB_ID = ? AND GRAPH_ID = ? AND STATUS = 'FAILED'${skipped}`,
      [runId, graphId, ...skippedFileIds],
    );
    const rows = await this.sql.query(
      "SELECT COUNT(*) AS QUEUED FROM KG_JOB_FILE WHERE JOB_ID = ? AND STATUS = 'QUEUED'",
      [runId],
    );
    const queued = Number(rows[0]?.queued ?? 0) || 0;
    if (queued > 0) {
      await this.sql.query(
        "UPDATE KG_JOB SET STATUS = 'PENDING', ERROR = NULL, LEASE_OWNER = NULL, " +
          "LEASE_EXPIRES_AT = NULL, UPDATED_AT = CURRENT_TIMESTAMP() WHERE ID = ?",
        [runId],
      );
    }
    return queued;
  }

  // ---------------------------------------------------------------------------
  // Run state

  /** Recent jobs with file-level counts, filtered to what this viewer may see, in one query. */
  private async queryRuns(limit: number): Promise<RunSnapshot[]> {
    const bounded = Math.max(1, Math.min(Math.trunc(limit) || 100, 500));
    const viewer = await this.resolvedViewer();
    // Owner's rights would otherwise return every graph in the account to every
    // viewer, so the predicate is the only thing between one corpus and another.
    const visible = visibleGraphPredicate(viewer, "G");
    const rows = await this.sql.query(
      "SELECT J.ID, J.GRAPH_ID, COALESCE(G.DISPLAY_NAME, J.GRAPH_ID) AS GRAPH_NAME, " +
        "J.STATUS, J.ERROR, J.CREATED_AT, J.UPDATED_AT, G.OWNER, " +
        "COUNT(F.ID) AS DOCUMENTS_TOTAL, " +
        "COUNT_IF(UPPER(F.STATUS) IN ('DONE', 'SUCCEEDED')) AS DOCUMENTS_COMPLETED, " +
        "COUNT_IF(UPPER(F.STATUS) = 'FAILED') AS DOCUMENTS_FAILED, " +
        "COUNT_IF(UPPER(F.STATUS) IN ('QUEUED', 'CLAIMED')) AS DOCUMENTS_PENDING, " +
        "TIMESTAMPDIFF('second', COALESCE(J.UPDATED_AT, J.CREATED_AT), " +
        "CURRENT_TIMESTAMP()) AS SECONDS_SINCE_UPDATE " +
        "FROM KG_JOB J LEFT JOIN KG_JOB_FILE F ON F.JOB_ID = J.ID " +
        "LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID " +
        `WHERE ${visible.sql} ` +
        "GROUP BY J.ID, J.GRAPH_ID, G.DISPLAY_NAME, G.OWNER, J.STATUS, J.ERROR, " +
        "J.CREATED_AT, J.UPDATED_AT " +
        `ORDER BY COALESCE(J.UPDATED_AT, J.CREATED_AT) DESC LIMIT ${bounded}`,
      visible.parameters,
    );
    const grants = await this.allGrants();
    const snapshots: RunSnapshot[] = [];
    for (const row of rows) {
      const runId = text(row.id);
      let status = text(row.status).toLowerCase();
      let error = jobErrorText(row.error);
      // A run only advances while its job service is alive. Checking that only
      // when a run is opened left a dead worker listed as running for as long as
      // nobody looked. Only a run gone quiet with work outstanding is worth the
      // extra round trip, so a healthy history still costs one query.
      if (
        UNSETTLED_JOB_STATUSES.has(status) &&
        Number(row.documents_pending ?? 0) > 0 &&
        Number(row.seconds_since_update ?? 0) >= QUIET_RUN_SECONDS
      ) {
        const lost = await this.lostWorkerError(runId);
        if (lost) {
          await this.failLostRun(runId, lost);
          status = "failed";
          error = lost;
        }
      }
      const graphId = text(row.graph_id);
      const record = { owner: text(row.owner) || null, grants: grants.get(graphId) ?? [] };
      snapshots.push(
        annotate(this.snapshot({
          runId,
          graphId,
          graphName: text(row.graph_name) || graphId,
          status,
          startedAt: text(row.created_at) || null,
          updatedAt: text(row.updated_at) || null,
          documentsTotal: Number(row.documents_total ?? 0) || 0,
          documentsCompleted: Number(row.documents_completed ?? 0) || 0,
          documentsFailed: Number(row.documents_failed ?? 0) || 0,
          error,
          raw: {},
        }), record, roleFor(record, viewer, SNOWFLAKE_ACCESS_POLICY)),
      );
    }
    return snapshots;
  }

  /** Read one job's state and grouped progress in one bounded query. */
  private async status(runId: string): Promise<RunSnapshot> {
    const viewer = await this.resolvedViewer();
    const visible = visibleGraphPredicate(viewer, "G");
    const rows = await this.sql.query(
      "SELECT J.ID, J.GRAPH_ID, COALESCE(G.DISPLAY_NAME, J.GRAPH_ID) AS GRAPH_NAME, G.OWNER, " +
        "J.STATUS, J.ERROR, J.CREATED_AT, J.UPDATED_AT, J.PROGRESS, " +
        "TIMESTAMPDIFF('second', COALESCE(J.UPDATED_AT, J.CREATED_AT), CURRENT_TIMESTAMP()) AS SECONDS_SINCE_UPDATE, " +
        "COALESCE(F.STAGE, 'queued') AS STAGE, F.STATUS AS FILE_STATUS, " +
        "COUNT(F.ID) AS FILE_COUNT " +
        "FROM KG_JOB J LEFT JOIN KG_JOB_FILE F ON F.JOB_ID = J.ID " +
        "LEFT JOIN KG_GRAPH G ON G.GRAPH_ID = J.GRAPH_ID " +
        `WHERE J.ID = ? AND ${visible.sql} ` +
        "GROUP BY J.ID, J.GRAPH_ID, G.DISPLAY_NAME, G.OWNER, J.STATUS, J.ERROR, " +
        "J.CREATED_AT, J.UPDATED_AT, J.PROGRESS, " +
        "COALESCE(F.STAGE, 'queued'), F.STATUS " +
        "ORDER BY STAGE, FILE_STATUS",
      [runId, ...visible.parameters],
    );
    if (rows.length === 0) {
      // Told apart so a viewer holding someone else's run id learns only that
      // it is not theirs, and an unknown id reads as unknown.
      if (await this.runGraphId(runId)) {
        throw forbidden(NO_ACCESS);
      }
      throw notFound(`Unknown Snowflake job: ${runId}`);
    }
    const job = rows[0]!;
    const grouped = new Map<string, Map<string, number>>();
    for (const row of rows) {
      const fileStatus = text(row.file_status).toLowerCase();
      const count = Number(row.file_count ?? 0) || 0;
      if (fileStatus && count) {
        const stage = text(row.stage) || "queued";
        const statuses = grouped.get(stage) ?? new Map<string, number>();
        statuses.set(fileStatus, count);
        grouped.set(stage, statuses);
      }
    }
    const fileStages: StageProgress[] = [...grouped.entries()].map(([stage, statuses]) => {
      const total = [...statuses.values()].reduce((sum, value) => sum + value, 0);
      const done = statuses.get("done") ?? 0;
      const state = statuses.get("failed")
        ? "failed"
        : statuses.get("claimed")
          ? "running"
          : done === total
            ? "completed"
            : "queued";
      return { stage, status: state, completed: done, total, elapsedMs: 0, message: null };
    });
    const stages = withLiveStageProgress(fileStages, job.progress);
    const totals = (status: string) =>
      [...grouped.values()].reduce((sum, statuses) => sum + (statuses.get(status) ?? 0), 0);
    // Every file registered for the run, in whatever state it reached.
    // Cancellation rewrites queued and claimed rows to CANCELLED, so a total
    // assembled only from working states would report no documents beside a
    // stage breakdown that still lists them.
    const documentsTotal = [...grouped.values()].reduce(
      (sum, statuses) => sum + [...statuses.values()].reduce((inner, value) => inner + value, 0),
      0,
    );
    let status = text(job.status).toLowerCase();
    let error = jobErrorText(job.error);
    if (
      UNSETTLED_JOB_STATUSES.has(status) &&
      totals("queued") + totals("claimed") > 0 &&
      Number(job.seconds_since_update ?? 0) >= QUIET_RUN_SECONDS
    ) {
      const lost = await this.lostWorkerError(runId);
      if (lost) {
        await this.failLostRun(runId, lost);
        status = "failed";
        error = lost;
      }
    }
    const { record, role } = await this.standing(text(job.graph_id));
    const snapshot = this.snapshot({
      runId,
      graphId: text(job.graph_id),
      graphName: text(job.graph_name) || text(job.graph_id),
      status,
      startedAt: text(job.created_at) || null,
      updatedAt: text(job.updated_at) || null,
      stages,
      documentsTotal,
      documentsCompleted: totals("done"),
      documentsFailed: totals("failed"),
      error,
      raw: { consumption: liveConsumption(job.progress) },
    });
    return annotate(snapshot, record, role);
  }

  private snapshot(fields: {
    runId: string;
    graphId: string;
    graphName: string;
    status: string;
    startedAt: string | null;
    updatedAt: string | null;
    stages?: StageProgress[];
    documentsTotal: number;
    documentsCompleted: number;
    documentsFailed: number;
    error: string | null;
    raw: Record<string, unknown>;
  }): RunSnapshot {
    const { database, schema } = this.options.connection;
    return {
      runId: fields.runId,
      graphId: fields.graphId,
      graphName: fields.graphName,
      status: fields.status,
      startedAt: fields.startedAt,
      updatedAt: fields.updatedAt,
      stages: fields.stages ?? [],
      events: [],
      documentsTotal: fields.documentsTotal,
      documentsCompleted: fields.documentsCompleted,
      documentsFailed: fields.documentsFailed,
      outputPath: null,
      storageKind: "snowflake",
      storageLocation: database && schema ? `${database}.${schema}` : "Snowflake",
      warnings: [],
      error: fields.error,
      listingWarning: null,
      raw: fields.raw,
    };
  }

  /**
   * Describe the run's worker if it has stopped without draining the queue.
   *
   * When a job service dies, nothing writes to the queue and nothing writes an
   * error, so the app would keep reporting what it last saw. The service's own
   * terminal state is the missing signal, addressable from the run id alone.
   * Returns null whenever the worker might still be working.
   */
  private async lostWorkerError(runId: string): Promise<string | null> {
    const name = serviceNameForJob(runId);
    let rows: SqlRow[];
    try {
      rows = await this.sql.query(`SHOW SERVICES LIKE '${name}'`);
    } catch {
      // Not being able to look is not evidence that the worker died.
      return null;
    }
    const state = text(rows[0]?.status).toUpperCase();
    if (!rows.length || !DEAD_SERVICE_STATUSES.has(state)) {
      return null;
    }
    return (
      `The worker for this run stopped without processing its documents ` +
      `(job service ${name} reported ${state}). Its container logs are held ` +
      `by that service. Retrying starts a new worker; if it stops again, ` +
      `check the image the run selected.`
    );
  }

  /**
   * Record a lost worker on the job so every view agrees the run is over.
   *
   * Written rather than only displayed, so the history and the run page agree.
   * Queued files stay queued so a retry can still claim them. Reading someone
   * else's run never writes a verdict onto it.
   */
  private async failLostRun(runId: string, message: string): Promise<void> {
    const { role } = await this.standing(await this.runGraphId(runId));
    if (accessRefusal(role, "write")) {
      return;
    }
    try {
      await this.sql.query(
        "UPDATE KG_JOB SET STATUS = 'FAILED', " +
          "ERROR = OBJECT_CONSTRUCT('message', ?, 'type', 'worker_stopped'), " +
          "UPDATED_AT = CURRENT_TIMESTAMP() " +
          "WHERE ID = ? AND STATUS NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')",
        [message, runId],
      );
    } catch {
      // The verdict is still shown; failing to store it must not fail the read.
    }
  }

  // ---------------------------------------------------------------------------
  // Graph data

  private async listStage(stage: string, prefix: string, limit: number): Promise<SourceObject[]> {
    const objects: SourceObject[] = [];
    if (limit <= 0) {
      return objects;
    }
    for await (const row of this.sql.stream(`LIST ${stageLocation(stage, prefix)}`)) {
      const item = sourceObjectFromListRow(row);
      if (item) {
        objects.push(item);
        if (objects.length >= limit) {
          break;
        }
      }
    }
    return objects;
  }

  private async readGraph(graphId: string): Promise<GraphDataset> {
    const countRows = await this.sql.query(
      graphCountQuery(COUNTED_TABLES),
      COUNTED_TABLES.map(() => graphId),
    );
    const fullCounts = Object.fromEntries(
      countRows.map((row) => [text(row.table_name).replace(/^KG_/, "").toLowerCase(), Number(row.row_count ?? 0) || 0]),
    );
    const review = async (table: keyof typeof REVIEW_TABLE_LIMITS, order: string) =>
      (
        await this.sql.query(
          `SELECT ${reviewProjection(table)} FROM ${table} WHERE GRAPH_ID = ? ORDER BY ${order} LIMIT ${REVIEW_TABLE_LIMITS[table]}`,
          [graphId],
        )
      ).map(withoutEmbedding);
    const nodes = await review("KG_NODE", "DEGREE DESC NULLS LAST, ID");
    const communities = await review("KG_COMMUNITY", "ID");
    const evidence = await review("KG_EVIDENCE", "ID");
    const edges = (
      await this.sql.query(reviewEdgeQuery(REVIEW_TABLE_LIMITS.KG_NODE, REVIEW_TABLE_LIMITS.KG_EDGE), [graphId, graphId])
    ).map(withoutEmbedding);
    await this.withEdgeDetail(graphId, edges);
    await this.withFindings(graphId, communities);
    const records = async (table: string) => (await this.sql.query(reviewRecordQuery(table), [graphId])).map(withoutEmbedding);
    const documents = await records(REVIEW_RECORD_TABLES.documents);
    const discardedWindows = await records(REVIEW_RECORD_TABLES.discardedWindows);
    const rejectedRecords = await records(REVIEW_RECORD_TABLES.rejectedRecords);
    const failedDocuments = await records(REVIEW_RECORD_TABLES.failedDocuments);
    const reportRows = await this.sql.query(RUN_REPORT_QUERY, [graphId]);
    const runReport: Record<string, unknown> = { ...variantMapping(reportRows[0]?.report), app_full_counts: fullCounts };
    const metricsRows = await this.sql.query(GRAPH_METRICS_QUERY, [graphId]);
    return {
      graphId,
      nodes,
      edges,
      communities,
      evidence,
      documents,
      chunks: [],
      discardedWindows,
      rejectedRecords,
      failedDocuments,
      runReport,
      graphMetrics: { ...variantMapping(metricsRows[0]?.metrics) },
    };
  }

  /** Put what each relation's observations state on the edge a review reads it from, as a stored graph's are. */
  private async withEdgeDetail(graphId: string, edges: readonly Record<string, unknown>[]): Promise<void> {
    const observations = (await this.sql.query(EDGE_DETAIL_QUERY, [graphId])).map((row) => ({
      ...row,
      quantities: variantList(row.quantities),
    }));
    const quantities = edgeQuantities(observations);
    const stated = edgeStatedTypes(observations);
    for (const edge of edges) {
      const id = text(edge.id);
      if (quantities.has(id)) {
        edge.quantities = quantities.get(id);
      }
      if (stated.has(id)) {
        edge.stated_relation_types = stated.get(id);
      }
    }
  }

  private async withFindings(graphId: string, communities: readonly Record<string, unknown>[]): Promise<void> {
    const findings = await this.sql.query(COMMUNITY_FINDING_QUERY, [graphId]);
    for (const community of communities) {
      const rows = findings.filter((row) => text(row.community_id) === text(community.id));
      if (rows.length > 0) {
        community.findings = rows;
      }
    }
  }

  /** How many rows each table holds for one graph, so a deletion can say what went. */
  private async graphRowCounts(graphId: string): Promise<Record<string, number>> {
    const rows = await this.sql.query(
      graphCountQuery(GRAPH_DATA_TABLES),
      GRAPH_DATA_TABLES.map(() => graphId),
    );
    const counts: Record<string, number> = {};
    for (const row of rows) {
      const count = Number(row.row_count ?? 0) || 0;
      if (count) {
        counts[text(row.table_name)] = count;
      }
    }
    return counts;
  }

  /** A graph's runs, with what a deletion needs to know about each. */
  private graphRuns(graphId: string): Promise<SqlRow[]> {
    return this.sql.query(
      "SELECT ID, STATUS, CONFIG['files']['stage_prefix']::string AS UPLOAD_PREFIX, " +
        "CONFIG['snowflake']['stage']::string AS DOCUMENT_STAGE, " +
        "CONFIG['snowflake']['database']::string AS DB, " +
        "CONFIG['snowflake']['schema']::string AS SCH " +
        "FROM KG_JOB WHERE GRAPH_ID = ?",
      [graphId],
    );
  }

  /**
   * Remove the staged uploads and job services a graph's runs left behind.
   *
   * Best effort by design: a stage that no longer exists, or a service already
   * dropped, must not stop the deletion the owner asked for.
   */
  private async releaseGraphStorage(runs: readonly SqlRow[]): Promise<void> {
    for (const row of runs) {
      const runId = text(row.id);
      await this.removeStagedUploads(text(row.document_stage), text(row.upload_prefix));
      await this.dropJobService(text(row.db), text(row.sch), runId);
    }
  }

  /**
   * Delete one run's uploaded documents. Only the app's own upload namespace is
   * ever removed: a run over a folder someone curates was built from those
   * documents, it does not own them.
   */
  private async removeStagedUploads(stage: string, prefix: string): Promise<void> {
    if (!stage || !prefix || !prefix.startsWith(`${UPLOAD_NAMESPACE}/`)) {
      return;
    }
    await this.bestEffort(() => `REMOVE ${stageLocation(stage, prefix)}`);
  }

  private async dropJobService(database: string, schema: string, runId: string): Promise<void> {
    if (!database || !schema || !runId || !IDENTIFIER.test(database) || !IDENTIFIER.test(schema)) {
      return;
    }
    await this.bestEffort(() => `DROP SERVICE IF EXISTS ${database}.${schema}.${serviceNameForJob(runId)}`);
  }

  private async bestEffort(statement: () => string): Promise<void> {
    try {
      await this.sql.query(statement());
    } catch {
      // Already gone, or never there; the deletion continues.
    }
  }

  private attempt<A>(run: () => Promise<A>, fallback: string): Effect.Effect<A, ControlPlaneError> {
    return Effect.tryPromise({ try: run, catch: (cause) => fromCause(cause, fallback) });
  }
}

/** The plane a hosted console serves a viewer through, against the service's own account. */
export function createSnowflakePlane(viewer?: Viewer, callerToken?: string | null): GraphAccessPlane {
  const env = appEnv();
  return new SnowflakeRuntime({
    repositoryRoot: env.repositoryRoot,
    stateRoot: env.stateRoot,
    sql: sharedSnowflakeSql(env.snowflake),
    connection: env.snowflake,
    viewer,
    callerToken,
  });
}

/** Read a share's target: a Snowflake user, or `role:NAME` for a role. */
function shareTarget(principal: string): { type: "USER" | "ROLE"; name: string } {
  const role = principalRole(principal);
  const name = validPrincipal(role ?? principal, SNOWFLAKE_ACCESS_POLICY);
  if (!name) {
    throw invalid(SHARE_TARGET);
  }
  return { type: role ? "ROLE" : "USER", name };
}

/** One stable identity per (graph, grantee), so sharing twice changes one row. */
export function shareId(graphId: string, granteeType: string, grantee: string): string {
  return createHash("sha256").update(`${graphId.trim()}|${granteeType}|${grantee}`).digest("hex");
}

/** A row of KG_GRAPH_ACL as `roleFor` reads a grant. */
function grantFromRow(row: SqlRow): GraphGrant {
  const grantee = text(row.grantee);
  return {
    principal: text(row.grantee_type).toUpperCase() === "ROLE" ? rolePrincipal(grantee) : principalKey(grantee),
    level: text(row.access_level) as AccessLevel,
    grantedBy: principalKey(text(row.granted_by)) || SYSTEM_PRINCIPAL,
    grantedAt: text(row.created_at),
  };
}

function annotate(run: RunSnapshot, record: GraphAccessRecord, role: GraphRole | null): RunSnapshot {
  return {
    ...run,
    owner: record.owner,
    access: role,
    sharedWith: record.grants.length,
    raw: { ...run.raw, owner: record.owner },
  };
}

function isActiveRun(row: SqlRow): boolean {
  return isActiveStatus(text(row.status));
}

/** Whether a run read documents the console itself uploaded, which go with its graph. */
function holdsUpload(row: SqlRow): boolean {
  return text(row.upload_prefix).startsWith(`${UPLOAD_NAMESPACE}/`);
}

/** Each queue state as the phase a document reads as, with the failure it recorded. */
const DOCUMENT_PHASES: Record<string, DocumentPhase> = {
  QUEUED: "queued",
  CLAIMED: "running",
  DONE: "indexed",
  SUCCEEDED: "indexed",
  CANCELLED: "cancelled",
  FAILED: "poison",
};

function documentStatusFromRow(row: SqlRow): DocumentStatus {
  const location = text(row.blob_path);
  const stage = text(row.stage);
  return {
    fileId: text(row.file_id),
    name: path.posix.basename(location) || undefined,
    location: location || undefined,
    phase: DOCUMENT_PHASES[text(row.status).toUpperCase()] ?? "queued",
    detail: jobErrorText(row.error) ?? stage,
  };
}

function missingUploadStage(stages: readonly string[]): string {
  return (
    `Internal stage ${UPLOAD_STAGE} does not exist in this schema, so uploads have nowhere to go. ` +
    `Available stages: ${stages.join(", ")}.`
  );
}

function withoutEmbedding(row: SqlRow): Record<string, unknown> {
  const { embedding: _embedding, ...rest } = row;
  return rest;
}

function quotedIdentifier(value: string): string {
  return value.trim().replaceAll('"', '""');
}

function text(value: unknown): string {
  return value === null || value === undefined ? "" : String(value).trim();
}

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? { ...(value as Record<string, unknown>) } : {};
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorType(error: unknown): string {
  if (error instanceof ControlPlaneError) {
    return error.code;
  }
  return error instanceof Error ? error.name || "Error" : "Error";
}

