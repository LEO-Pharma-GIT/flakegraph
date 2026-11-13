import { bigint, integer, jsonb, pgTable, text, timestamp } from "drizzle-orm/pg-core";

export const flakegraphRun = pgTable("flakegraph_run", {
  id: text("id").primaryKey(),
  graphId: text("graph_id").notNull(),
  configJson: jsonb("config_json").notNull(),
  configDigest: text("config_digest").notNull(),
  status: text("status").notNull(),
  priorityBand: text("priority_band").notNull(),
  errorJson: jsonb("error_json"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull(),
});

export const flakegraphTask = pgTable("flakegraph_task", {
  id: text("id").primaryKey(),
  runId: text("run_id").notNull(),
  stage: text("stage").notNull(),
  scopeId: text("scope_id").notNull(),
  payloadJson: jsonb("payload_json").notNull(),
  status: text("status").notNull(),
  attempts: integer("attempts").notNull(),
  leaseOwner: text("lease_owner"),
  lastErrorJson: jsonb("last_error_json"),
  startedAt: timestamp("started_at", { withTimezone: true }),
  completedAt: timestamp("completed_at", { withTimezone: true }),
  updatedAt: timestamp("updated_at", { withTimezone: true }),
});

export const flakegraphArtifact = pgTable("flakegraph_artifact", {
  id: text("id").primaryKey(),
  runId: text("run_id").notNull(),
  kind: text("kind").notNull(),
  mediaType: text("media_type").notNull(),
  sizeBytes: bigint("size_bytes", { mode: "number" }).notNull(),
  storageUri: text("storage_uri"),
  metadataJson: jsonb("metadata_json").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const flakegraphTaskOutput = pgTable("flakegraph_task_output", {
  taskId: text("task_id").notNull(),
  position: integer("position").notNull(),
  artifactId: text("artifact_id").notNull(),
});

export const flakegraphGraphVersion = pgTable("flakegraph_graph_version", {
  id: text("id").primaryKey(),
  graphId: text("graph_id").notNull(),
  runId: text("run_id").notNull(),
  artifactId: text("artifact_id").notNull(),
  mediaType: text("media_type").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull(),
});

export const flakegraphGraphHead = pgTable("flakegraph_graph_head", {
  graphId: text("graph_id").primaryKey(),
  versionId: text("version_id").notNull(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull(),
});
