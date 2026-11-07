import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";
import { evidenceDocumentId, evidenceEntityId, evidenceRelationId } from "@/lib/evidence";
import { GraphCache } from "./graph-cache";
import { capGraph } from "./graph-filter";
import { GRAPH_REVIEW_ROW_LIMITS, type GraphDataset } from "./protocol/schema";
import type { GraphLoadOptions } from "./protocol/runtime";

/** About a gigabyte of parsed rows: several fleet graphs, within Node's default heap. */
const CACHED_GRAPH_ROWS = 1_000_000;
const CACHED_GRAPH_IDLE_MS = 10 * 60_000;

const shared = globalThis as typeof globalThis & { flakegraphGraphCache?: GraphCache<GraphDataset> };

/** One cache for the process: Next may bundle the routes that read graphs apart. */
function graphCache(): GraphCache<GraphDataset> {
  shared.flakegraphGraphCache ??= new GraphCache(datasetRows, {
    maxRows: CACHED_GRAPH_ROWS,
    idleMs: CACHED_GRAPH_IDLE_MS,
  });
  return shared.flakegraphGraphCache;
}

function datasetRows(dataset: GraphDataset): number {
  return Object.values(dataset).reduce((total, value) => total + (Array.isArray(value) ? value.length : 0), 0);
}

/** Changes whenever a graph's files are replaced, whichever format they are in. */
async function graphVersion(directory: string): Promise<string> {
  const files = ["nodes.parquet", "nodes.json", "edges.parquet", "edges.json", "run_report.json"];
  const stamps = await Promise.all(
    files.map(async (file) => {
      const found = await stat(path.join(directory, file)).catch(() => null);
      return found ? `${found.mtimeMs}:${found.size}` : "-";
    }),
  );
  return stamps.join("/");
}

/**
 * A graph as the browser uses it. Chunks are most of a graph's bytes and the
 * browser only follows them from an entity to its document, so they travel as
 * references; their text stays on the server for retrieval.
 */
export function graphForBrowser(dataset: GraphDataset): GraphDataset {
  return {
    ...dataset,
    chunks: dataset.chunks.map((chunk) => ({ id: chunk.id, document_id: evidenceDocumentId(chunk) })),
  };
}

export async function loadLocalGraph(directory: string, options: GraphLoadOptions = {}): Promise<GraphDataset> {
  if (!existsSync(directory)) {
    throw new Error(`No graph artifact directory exists at this location: ${directory}`);
  }
  const key = `${options.purpose ?? "review"}:${directory}`;
  return graphCache().get(key, await graphVersion(directory), () => readLocalGraph(directory, options));
}

async function readLocalGraph(directory: string, options: GraphLoadOptions): Promise<GraphDataset> {
  if (!existsSync(directory)) {
    throw new Error(`No graph artifact directory exists at this location: ${directory}`);
  }
  const parquetNodes = path.join(directory, "nodes.parquet");
  const jsonNodes = path.join(directory, "nodes.json");
  if (!existsSync(parquetNodes) && !existsSync(jsonNodes)) {
    throw new Error(`Graph directory must contain nodes.parquet or nodes.json: ${directory}`);
  }
  // Every row is read so the review sample is the best-connected part of the
  // graph rather than whichever rows the writer happened to put first; vector
  // columns are dropped at the reader, which is what makes that affordable.
  const [allNodes, nodeCount] = await readTable(directory, "nodes", Number.MAX_SAFE_INTEGER);
  const [allEdges, edgeCount] = await readTable(directory, "edges", Number.MAX_SAFE_INTEGER);
  const nodeIds = new Set(allNodes.map((node) => String(node.id ?? "")));
  const joinedEdges = allEdges.filter(
    (edge) => nodeIds.has(String(edge.source_node_id ?? edge.source ?? "")) &&
      nodeIds.has(String(edge.target_node_id ?? edge.target ?? "")),
  );
  const limits = options.purpose === "retrieval"
    ? { nodes: Number.MAX_SAFE_INTEGER, edges: Number.MAX_SAFE_INTEGER, communities: Number.MAX_SAFE_INTEGER, evidence: Number.MAX_SAFE_INTEGER }
    : GRAPH_REVIEW_ROW_LIMITS;
  const sample = options.purpose === "retrieval"
    ? { nodes: allNodes, edges: joinedEdges.length > 0 ? joinedEdges : allEdges }
    : capGraph(allNodes, joinedEdges.length > 0 ? joinedEdges : allEdges, GRAPH_REVIEW_ROW_LIMITS.nodes);
  const nodes = sample.nodes;
  const edges = sample.edges.slice(0, limits.edges);
  const [communities, communityCount] = await readTable(
    directory,
    "communities",
    limits.communities,
  );
  const findings = await readOptionalTable(directory, "community_findings");
  const findingsByCommunity = new Map<string, Record<string, unknown>[]>();
  for (const finding of findings) {
    const id = String(finding.community_id ?? "");
    const rows = findingsByCommunity.get(id) ?? [];
    rows.push(finding);
    findingsByCommunity.set(id, rows);
  }
  for (const community of communities) {
    const rows = findingsByCommunity.get(String(community.id ?? ""));
    if (rows) community.findings = rows;
  }
  const [allEvidence, evidenceCount] = await readTable(
    directory,
    "evidence",
    Number.MAX_SAFE_INTEGER,
  );
  const evidence = options.purpose === "retrieval" ? allEvidence : reviewEvidence(allEvidence, nodes, edges, limits.evidence);
  const documents = await readOptionalTable(directory, "documents");
  // The values each relation states live on its observations; the review reads
  // them on the edge they describe.
  const observations = await readOptionalTable(directory, "edge_observations", [
    "edge_id",
    "quantities",
    "proposed_relation_type",
  ]);
  const quantitiesByEdge = edgeQuantities(observations);
  const statedByEdge = edgeStatedTypes(observations);
  for (const edge of edges) {
    const quantities = quantitiesByEdge.get(String(edge.id ?? ""));
    if (quantities) {
      edge.quantities = quantities;
    }
    // A relation kept under the ontology's fallback carries the type the model
    // stated, so the edge says what it was before it was generalised.
    const stated = statedByEdge.get(String(edge.id ?? ""));
    if (stated) {
      edge.stated_relation_types = stated;
    }
  }
  const chunks = await readOptionalTable(directory, "chunks");
  const discardedWindows = await readOptionalTable(directory, "discarded_windows");
  const rejectedRecords = await readOptionalTable(directory, "rejected_records");
  const failedDocuments = await readOptionalTable(directory, "failed_documents");
  const runReport = await readJsonObject(path.join(directory, "run_report.json"));
  runReport.app_full_counts = {
    node: nodeCount,
    edge: edgeCount,
    community: communityCount,
    evidence: evidenceCount,
    document: documents.length,
    chunk: chunks.length,
  };
  const graphId = String(
    runReport.graph_id ?? (nodes[0]?.graph_id as string | undefined) ?? path.basename(directory),
  );
  return {
    graphId,
    nodes,
    edges,
    communities,
    evidence,
    documents,
    chunks,
    discardedWindows,
    rejectedRecords,
    failedDocuments,
    runReport,
    graphMetrics: await readJsonObject(path.join(directory, "graph_metrics.json")),
  };
}

/** Give each retained relation grounding before filling the bounded review with node quotes. */
export function reviewEvidence(rows: Record<string, unknown>[], nodes: Record<string, unknown>[], edges: Record<string, unknown>[], limit: number): Record<string, unknown>[] {
  if (limit <= 0) return [];
  const edgeId = (row: Record<string, unknown>) => evidenceRelationId(row) ?? "";
  const nodeId = (row: Record<string, unknown>) => evidenceEntityId(row) ?? "";
  const keptEdges = new Set(edges.map((row) => String(row.id)));
  const keptNodes = new Set(nodes.map((row) => String(row.id)));
  const matching = rows.filter((row) => keptEdges.has(edgeId(row)) || keptNodes.has(nodeId(row)));
  const selected = new Set<Record<string, unknown>>();
  for (const [identify, kept] of [[edgeId, keptEdges], [nodeId, keptNodes]] as const) {
    const seen = new Set<string>();
    for (const row of matching) {
      const id = identify(row);
      if (!kept.has(id) || seen.has(id) || !row.quote) continue;
      seen.add(id);
      selected.add(row);
      if (selected.size === limit) return [...selected];
    }
  }
  for (const row of matching) {
    selected.add(row);
    if (selected.size === limit) break;
  }
  return [...selected];
}

/** Edge id → the distinct values its observations state, in source order. */
export function edgeQuantities(observations: readonly Record<string, unknown>[]): Map<string, unknown[]> {
  const byEdge = new Map<string, unknown[]>();
  for (const observation of observations) {
    const quantities = Array.isArray(observation.quantities) ? observation.quantities : [];
    if (quantities.length === 0) {
      continue;
    }
    const edgeId = String(observation.edge_id ?? "");
    const known = byEdge.get(edgeId) ?? [];
    const seen = new Set(known.map((item) => JSON.stringify(item)));
    for (const quantity of quantities) {
      if (!seen.has(JSON.stringify(quantity))) {
        known.push(quantity);
        seen.add(JSON.stringify(quantity));
      }
    }
    byEdge.set(edgeId, known);
  }
  return byEdge;
}

/** Edge id → the relation types the model stated for it, when a fallback relation kept them. */
export function edgeStatedTypes(observations: readonly Record<string, unknown>[]): Map<string, string[]> {
  const byEdge = new Map<string, string[]>();
  for (const observation of observations) {
    const stated = observation.proposed_relation_type;
    if (typeof stated !== "string" || !stated) {
      continue;
    }
    const edgeId = String(observation.edge_id ?? "");
    const known = byEdge.get(edgeId) ?? [];
    if (!known.includes(stated)) {
      known.push(stated);
    }
    byEdge.set(edgeId, known);
  }
  return byEdge;
}

export function graphArtifactsExist(directory: string): boolean {
  return (
    existsSync(path.join(directory, "nodes.parquet")) && existsSync(path.join(directory, "edges.parquet"))
  ) || (
    existsSync(path.join(directory, "nodes.json")) && existsSync(path.join(directory, "edges.json"))
  );
}

async function readTable(
  directory: string,
  name: string,
  limit: number,
  only?: readonly string[],
): Promise<[Record<string, unknown>[], number]> {
  const parquet = path.join(directory, `${name}.parquet`);
  const json = path.join(directory, `${name}.json`);
  if (existsSync(parquet)) {
    const rows = await readParquetRows(parquet, limit, only);
    const count = await countParquet(parquet);
    return [rows, count];
  }
  if (existsSync(json)) {
    const rows = await readJsonArray(json);
    return [rows.slice(0, limit), rows.length];
  }
  return [[], 0];
}

async function readOptionalTable(
  directory: string,
  name: string,
  only?: readonly string[],
): Promise<Record<string, unknown>[]> {
  const [rows] = await readTable(directory, name, Number.MAX_SAFE_INTEGER, only);
  return rows;
}

/** Column names the console never renders and that dominate a table's size. */
const VECTOR_COLUMN = /(^|_)embedding(s)?$/;

async function readParquetRows(
  file: string,
  limit: number,
  only?: readonly string[],
): Promise<Record<string, unknown>[]> {
  const { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects, parquetSchema } = await import("hyparquet");
  const buffer = await asyncBufferFromFile(file);
  const metadata = await parquetMetadataAsync(buffer);
  const columns = parquetSchema(metadata)
    .children.map((child) => child.element.name)
    .filter((name) => (only ? only.includes(name) : !VECTOR_COLUMN.test(name)));
  const rows = await parquetReadObjects({
    file: buffer,
    metadata,
    columns,
    rowEnd: Math.min(limit, Number(metadata.num_rows)),
  });
  return rows.map(normalizeRow);
}

async function countParquet(file: string): Promise<number> {
  const { asyncBufferFromFile, parquetMetadataAsync } = await import("hyparquet");
  const metadata = await parquetMetadataAsync(await asyncBufferFromFile(file));
  return Number(metadata.num_rows);
}

async function readJsonArray(file: string): Promise<Record<string, unknown>[]> {
  const value = JSON.parse(await readFile(file, "utf8")) as unknown;
  return Array.isArray(value) ? value.map((row) => normalizeRow(row as Record<string, unknown>)) : [];
}

async function readJsonObject(file: string): Promise<Record<string, unknown>> {
  if (!existsSync(file)) {
    return {};
  }
  const value = JSON.parse(await readFile(file, "utf8")) as unknown;
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

export function normalizeRow(row: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(row)) {
    if (VECTOR_COLUMN.test(key)) {
      continue;
    }
    result[key] = normalizeValue(value);
  }
  return result;
}

function normalizeValue(value: unknown): unknown {
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(value)) {
    return value.toString("utf8");
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.map(normalizeValue);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nested]) => [key, normalizeValue(nested)]),
    );
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  return value;
}
