import postgres from "postgres";
import { drizzle } from "drizzle-orm/postgres-js";
import { and, desc, eq, inArray, sql } from "drizzle-orm";
import { appEnv } from "../env";
import { DOCUMENT_STAGES } from "../documents";
import * as schema from "./schema";

type Database = ReturnType<typeof drizzle<typeof schema>>;

// One pool for the process: a pool per query would hold its connections
// open until the process exits.
let pool: { url: string; db: Database } | null = null;

export function createDrizzle(): Database | null {
  const url = appEnv().databaseUrl;
  if (!url) {
    return null;
  }
  if (pool?.url !== url) {
    const client = postgres(url, { max: 4, prepare: false });
    pool = { url, db: drizzle(client, { schema }) };
  }
  return pool.db;
}

export async function listPostgresRuns(limit = 100) {
  const db = createDrizzle();
  if (!db) {
    return [];
  }
  return db.select().from(schema.flakegraphRun).orderBy(desc(schema.flakegraphRun.updatedAt)).limit(limit);
}

/** Whether the coordination store holds a run of this graph; null without a store. */
export async function postgresGraphExists(graphId: string): Promise<boolean | null> {
  const db = createDrizzle();
  if (!db) {
    return null;
  }
  const rows = await db
    .select({ id: schema.flakegraphRun.id })
    .from(schema.flakegraphRun)
    .where(eq(schema.flakegraphRun.graphId, graphId))
    .limit(1);
  return rows.length > 0;
}

/** Whether the coordination store holds a run with this id; null without a store. */
export async function postgresRunExists(runId: string): Promise<boolean | null> {
  const db = createDrizzle();
  if (!db) {
    return null;
  }
  const rows = await db
    .select({ id: schema.flakegraphRun.id })
    .from(schema.flakegraphRun)
    .where(eq(schema.flakegraphRun.id, runId))
    .limit(1);
  return rows.length > 0;
}

/**
 * Per-run document counters from the durable stages: every discovered
 * document owns one prepare task, and a document is complete once its
 * compaction succeeds. One grouped query for every listed run.
 */
export async function documentCountsByRun(
  runIds: readonly string[],
): Promise<Map<string, { total: number | null; completed: number; failed: number }>> {
  const db = createDrizzle();
  const counts = new Map<string, { total: number | null; completed: number; failed: number }>();
  if (!db || runIds.length === 0) {
    return counts;
  }
  const rows = await db
    .select({
      runId: schema.flakegraphTask.runId,
      stage: schema.flakegraphTask.stage,
      status: schema.flakegraphTask.status,
      count: sql<number>`count(*)::int`,
    })
    .from(schema.flakegraphTask)
    .where(inArray(schema.flakegraphTask.runId, [...runIds]))
    .groupBy(schema.flakegraphTask.runId, schema.flakegraphTask.stage, schema.flakegraphTask.status);
  for (const row of rows) {
    const entry = counts.get(row.runId) ?? { total: null, completed: 0, failed: 0 };
    if (row.stage === "prepare_document") {
      entry.total = (entry.total ?? 0) + row.count;
      if (row.status === "failed") {
        entry.failed += row.count;
      }
    } else if (row.stage === "compact_document") {
      if (row.status === "succeeded") {
        entry.completed += row.count;
      } else if (row.status === "failed") {
        entry.failed += row.count;
      }
    }
    counts.set(row.runId, entry);
  }
  // A document no parser could read has a prepare task that succeeded - it
  // was recorded, not retried forever - so the tag on that task's output is
  // what says it failed.
  const unreadable = await db
    .select({
      runId: schema.flakegraphTask.runId,
      count: sql<number>`sum(jsonb_array_length(${schema.flakegraphArtifact.metadataJson} -> 'failed_documents'))::int`,
    })
    .from(schema.flakegraphTask)
    .innerJoin(schema.flakegraphTaskOutput, eq(schema.flakegraphTaskOutput.taskId, schema.flakegraphTask.id))
    .innerJoin(schema.flakegraphArtifact, eq(schema.flakegraphArtifact.id, schema.flakegraphTaskOutput.artifactId))
    .where(
      and(
        inArray(schema.flakegraphTask.runId, [...runIds]),
        eq(schema.flakegraphTask.stage, "prepare_document"),
        eq(schema.flakegraphTask.status, "succeeded"),
        eq(schema.flakegraphArtifact.kind, "extracted_document"),
        sql`jsonb_typeof(${schema.flakegraphArtifact.metadataJson} -> 'failed_documents') = 'array'`,
      ),
    )
    .groupBy(schema.flakegraphTask.runId);
  for (const row of unreadable) {
    const entry = counts.get(row.runId) ?? { total: null, completed: 0, failed: 0 };
    entry.failed += row.count;
    counts.set(row.runId, entry);
  }
  // A revision's kept documents are in the graph without a task of their
  // own; the finalizer's payload counts them.
  const kept = await db
    .select({
      runId: schema.flakegraphTask.runId,
      count: sql<number>`sum(jsonb_array_length(entry -> 'file_ids'))::int`,
    })
    .from(schema.flakegraphTask)
    .innerJoin(sql`jsonb_array_elements(${schema.flakegraphTask.payloadJson} -> 'inherit') AS entry`, sql`true`)
    .where(and(inArray(schema.flakegraphTask.runId, [...runIds]), eq(schema.flakegraphTask.stage, "finalize_graph")))
    .groupBy(schema.flakegraphTask.runId);
  for (const row of kept) {
    const entry = counts.get(row.runId) ?? { total: null, completed: 0, failed: 0 };
    entry.total = (entry.total ?? 0) + row.count;
    entry.completed += row.count;
    counts.set(row.runId, entry);
  }
  return counts;
}

export interface DocumentTaskRow {
  stage: string;
  scopeId: string;
  status: string;
  attempts: number;
  lastError: unknown;
  startedAt: Date | null;
  completedAt: Date | null;
  /** The source artifact's metadata, on the prepare task that carries it. */
  sourceMetadata: unknown;
  /** Why the document could not be read, when its prepare task recorded that. */
  unreadable: unknown;
}

/**
 * The per-document tasks of one run, each joined to the source artifact its
 * prepare task names so the document can be shown by filename.
 */
export async function documentTasksByRun(runId: string): Promise<DocumentTaskRow[]> {
  const db = createDrizzle();
  if (!db) {
    return [];
  }
  const rows = await db
    .select({
      stage: schema.flakegraphTask.stage,
      scopeId: schema.flakegraphTask.scopeId,
      status: schema.flakegraphTask.status,
      attempts: schema.flakegraphTask.attempts,
      lastError: schema.flakegraphTask.lastErrorJson,
      startedAt: schema.flakegraphTask.startedAt,
      completedAt: schema.flakegraphTask.completedAt,
      sourceMetadata: schema.flakegraphArtifact.metadataJson,
      unreadable: sql<unknown>`(
        SELECT extracted.metadata_json -> 'failed_documents' -> 0
        FROM flakegraph_task_output AS output
        JOIN flakegraph_artifact AS extracted ON extracted.id = output.artifact_id
        WHERE output.task_id = ${schema.flakegraphTask.id}
          AND extracted.kind = 'extracted_document'
          AND ${schema.flakegraphTask.stage} = 'prepare_document'
          AND ${schema.flakegraphTask.status} = 'succeeded'
        LIMIT 1
      )`,
    })
    .from(schema.flakegraphTask)
    .leftJoin(
      schema.flakegraphArtifact,
      eq(schema.flakegraphArtifact.id, sql`${schema.flakegraphTask.payloadJson}->>'source_artifact_id'`),
    )
    .where(and(eq(schema.flakegraphTask.runId, runId), inArray(schema.flakegraphTask.stage, [...DOCUMENT_STAGES])))
    .orderBy(schema.flakegraphTask.scopeId, schema.flakegraphTask.stage);
  return rows;
}

export interface LeasedTaskRow {
  taskId: string;
  runId: string;
  graphId: string;
  stage: string;
  scopeId: string;
  leaseOwner: string;
  updatedAt: Date | null;
}

/** Every task a worker currently holds a lease on, with the run's graph. */
export async function leasedTasks(): Promise<LeasedTaskRow[]> {
  const db = createDrizzle();
  if (!db) {
    return [];
  }
  const rows = await db
    .select({
      taskId: schema.flakegraphTask.id,
      runId: schema.flakegraphTask.runId,
      graphId: schema.flakegraphRun.graphId,
      stage: schema.flakegraphTask.stage,
      scopeId: schema.flakegraphTask.scopeId,
      leaseOwner: schema.flakegraphTask.leaseOwner,
      updatedAt: schema.flakegraphTask.updatedAt,
    })
    .from(schema.flakegraphTask)
    .innerJoin(schema.flakegraphRun, eq(schema.flakegraphRun.id, schema.flakegraphTask.runId))
    .where(eq(schema.flakegraphTask.status, "running"))
    .orderBy(schema.flakegraphTask.updatedAt);
  return rows.filter((row): row is LeasedTaskRow => Boolean(row.leaseOwner));
}

export interface GraphVersionRow {
  graphId: string;
  runId: string;
  createdAt: Date;
  /** Whether this version is the one the graph's head points at. */
  head: boolean;
}

/**
 * Every published version of the given graphs, oldest first, with the head
 * marked. A run that has not published a graph has no version.
 */
export async function graphVersionsByGraph(graphIds: readonly string[]): Promise<Map<string, GraphVersionRow[]>> {
  const versions = new Map<string, GraphVersionRow[]>();
  const db = createDrizzle();
  if (!db || graphIds.length === 0) {
    return versions;
  }
  const rows = await db
    .select({
      graphId: schema.flakegraphGraphVersion.graphId,
      runId: schema.flakegraphGraphVersion.runId,
      createdAt: schema.flakegraphGraphVersion.createdAt,
      headVersionId: schema.flakegraphGraphHead.versionId,
      versionId: schema.flakegraphGraphVersion.id,
    })
    .from(schema.flakegraphGraphVersion)
    .leftJoin(schema.flakegraphGraphHead, eq(schema.flakegraphGraphHead.graphId, schema.flakegraphGraphVersion.graphId))
    .where(inArray(schema.flakegraphGraphVersion.graphId, [...graphIds]))
    .orderBy(schema.flakegraphGraphVersion.createdAt, schema.flakegraphGraphVersion.id);
  for (const row of rows) {
    const list = versions.get(row.graphId) ?? [];
    list.push({
      graphId: row.graphId,
      runId: row.runId,
      createdAt: row.createdAt,
      head: row.headVersionId === row.versionId,
    });
    versions.set(row.graphId, list);
  }
  return versions;
}

/** The finalizer's payload of a run, which names the documents a revision kept. */
export async function finalizerPayload(runId: string): Promise<Record<string, unknown> | null> {
  const db = createDrizzle();
  if (!db) {
    return null;
  }
  const rows = await db
    .select({ payload: schema.flakegraphTask.payloadJson })
    .from(schema.flakegraphTask)
    .where(and(eq(schema.flakegraphTask.runId, runId), eq(schema.flakegraphTask.stage, "finalize_graph")))
    .limit(1);
  const payload = rows[0]?.payload;
  return payload && typeof payload === "object" && !Array.isArray(payload) ? (payload as Record<string, unknown>) : null;
}
