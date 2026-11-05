import { Effect } from "effect";
import {
  type ClusterSnapshot,
  type GraphDataset,
  type GraphAccessView,
  type GraphDeletionPreview,
  type GraphRole,
  type GraphVersion,
  type IngestionRequest,
  type NodeWorkAssignment,
  type PreflightResult,
  type RunSnapshot,
  type RuntimeMode,
  type SourceObject,
  type SourceSummary,
  type Viewer,
  type Capability,
} from "./schema";
import { ControlPlaneError } from "./errors";
import type { DocumentStatus } from "../documents";
import type { AccessLevel } from "../access/graph-access";
import type { FleetProfile } from "../fleet";

/** Where inside the fleet's Snowflake account a published graph lands. */
export interface SnowflakePublishTarget {
  database: string;
  schema: string;
  warehouse: string;
  role?: string | null;
  bulkStage: string;
}

/** Copy a run's graph into Snowflake; what was written. */
export type SnowflakePublishing = (
  runId: string,
  target: SnowflakePublishTarget,
) => Promise<{ nodes: number; edges: number }>;

/** Browser review is bounded; server retrieval searches the complete published graph. */
export interface GraphLoadOptions {
  purpose?: "review" | "retrieval";
}

/** A stage the form offers as a document source. */
export interface StageChoice {
  /** The stage as the run reads it, `@DATABASE.SCHEMA.STAGE`. */
  stage: string;
  name: string;
}

/** A folder inside a stage, with the documents under it that a run would read. */
export interface StageFolder {
  prefix: string;
  documents: number;
}

/** The models a runtime's model service offers, split by the role they play in a run. */
export interface ModelChoices {
  llm: string[];
  embedding: Array<{ model: string; dimension: number }>;
}

export interface ControlPlane {
  readonly runtime: RuntimeMode;
  readonly capabilities: ReadonlySet<Capability>;
  viewer(): Effect.Effect<Viewer, ControlPlaneError>;
  listSourceObjects(
    source: Record<string, unknown>,
    limit?: number,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError>;
  /** Count what a source holds without listing it, however large it is. */
  summarizeSource(source: Record<string, unknown>): Effect.Effect<SourceSummary, ControlPlaneError>;
  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError>;
  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError>;
  listRuns(limit?: number): Effect.Effect<readonly RunSnapshot[], ControlPlaneError>;
  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError>;
  /** Where each of a run's documents stands, by whatever the runtime records. */
  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError>;
  /** The published versions of a graph, oldest first; empty where a runtime keeps none. */
  versions(graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError>;
  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError>;
  retry(runId: string, skippedFileIds?: readonly string[]): Effect.Effect<RunSnapshot, ControlPlaneError>;
  recover(runId: string): Effect.Effect<string, ControlPlaneError>;
  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError>;
  loadGraph(location: string, graphId?: string | null, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError>;
  loadRunGraph(snapshot: RunSnapshot, options?: GraphLoadOptions): Effect.Effect<GraphDataset, ControlPlaneError>;
  /** Where a loaded run's graph files are on this host, for tools that read them; null where none are. */
  graphDirectory(snapshot: RunSnapshot): string | null;
  cluster(namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError>;
  nodeAssignments(
    namespace: string,
    nodeName: string,
  ): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError>;
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError>;
  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError>;
  /**
   * The embedding width the destination's graph tables store, when they agree on
   * one. Only a runtime that writes to fixed-width tables implements it.
   */
  graphEmbeddingWidth?(): Effect.Effect<number | null, ControlPlaneError>;
  /** Stages documents can be read from, for a runtime that reads them from stages. */
  documentStages?(): Effect.Effect<readonly StageChoice[], ControlPlaneError>;
  /** Folders of one stage the viewer may run on, with their document counts. */
  stageFolders?(stage: string): Effect.Effect<readonly StageFolder[], ControlPlaneError>;
  /** The models a runtime's own model service offers, by the role they play in a run. */
  modelChoices?(): Effect.Effect<ModelChoices, ControlPlaneError>;
}

/**
 * A runtime seen through one viewer's access: what the console's procedures
 * hold. Every read is limited to graphs the viewer owns or has been given,
 * every change to the level it needs, and sharing is the owner's.
 */
export interface GraphAccessPlane extends ControlPlane {
  /** The viewer's role on a graph; fails when the graph is not theirs to see or lacks the level. */
  requireGraph(graphId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError>;
  /** The same, for the graph a run belongs to. */
  requireRun(runId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError>;
  /** Every graph the viewer can see, with their role on it. */
  visibleGraphs(): Effect.Effect<ReadonlyMap<string, GraphRole>, ControlPlaneError>;
  graphAccess(graphId: string): Effect.Effect<GraphAccessView, ControlPlaneError>;
  shareGraph(graphId: string, principal: string, level: AccessLevel): Effect.Effect<GraphAccessView, ControlPlaneError>;
  unshareGraph(graphId: string, principal: string): Effect.Effect<GraphAccessView | null, ControlPlaneError>;
  /** What deleting a graph would remove; owner only. */
  deletionPreview(graphId: string): Effect.Effect<GraphDeletionPreview, ControlPlaneError>;
  /** People the console has seen signed in, to offer when sharing. */
  people(): Effect.Effect<readonly string[], ControlPlaneError>;
  /** The profile a fleet's workers mount, where the runtime is a fleet. */
  fleetProfile(): Promise<FleetProfile | null>;
  /** Publishing a run's graph to Snowflake, where the runtime can; checked for write access. */
  readonly snowflakePublishing: SnowflakePublishing | null;
}

export function hasCapability(plane: ControlPlane, capability: Capability): boolean {
  return plane.capabilities.has(capability);
}
