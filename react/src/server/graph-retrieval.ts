import type { GraphDataset } from "./protocol/schema";

/** The layout blocks that hold structure a passage's text only renders. */
export const STRUCTURED_BLOCK_KINDS = ["table", "chart", "figure"] as const;
export type StructuredBlockKind = (typeof STRUCTURED_BLOCK_KINDS)[number];

/** The columns of the blocks table retrieval reads: its text is already in the chunks. */
export const STRUCTURED_BLOCK_COLUMNS = ["id", "file_id", "page_number", "kind", "metadata"] as const;

/**
 * What retrieval knows about a graph beyond the rows of its dataset.
 *
 * It is kept beside the dataset rather than on it, because a `GraphDataset`
 * is what the console sends to the browser and caches by row count: neither
 * should carry a run's table blocks or learn where its chunk vectors are.
 * Only a graph loaded for retrieval has one.
 */
export interface RetrievalSource {
  /** Where the graph's artifact files are, so its chunk vectors can be read when a search needs them. */
  directory: string | null;
  /** The run's table, chart and figure blocks, each with its `metadata` JSON as written. */
  blocks: readonly Record<string, unknown>[];
}

// Keyed by the dataset object, so a source lives exactly as long as the
// cached graph it describes and leaves with it.
const sources = new WeakMap<GraphDataset, RetrievalSource>();

export function attachRetrievalSource(dataset: GraphDataset, source: RetrievalSource): void {
  sources.set(dataset, source);
}

export function retrievalSource(dataset: GraphDataset): RetrievalSource | undefined {
  return sources.get(dataset);
}

export function isStructuredBlockKind(kind: unknown): kind is StructuredBlockKind {
  return typeof kind === "string" && (STRUCTURED_BLOCK_KINDS as readonly string[]).includes(kind);
}

/**
 * A block's structure as its parser kept it: `{table: {...}}` for a table,
 * `{chart: {...}}` for a chart (see the Python package's `domain/tables.py`).
 * Null when the metadata is absent or not an object.
 */
export function blockStructure(block: Record<string, unknown>): Record<string, unknown> | null {
  const raw = block.metadata;
  let value: unknown = raw;
  if (typeof raw === "string") {
    try {
      value = JSON.parse(raw) as unknown;
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}
