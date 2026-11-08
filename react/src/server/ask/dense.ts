import { embed } from "ai";
import { readChunkVectors, type ChunkVectors } from "../chunk-vectors";
import { retrievalSource } from "../graph-retrieval";
import type { GraphDataset } from "../protocol/schema";
import { ASK_EMBEDDING_TIMEOUT_MS } from "./constants";
import type { AskEmbedding } from "./embedding-model";
import { errorMessage } from "./errors";
import { TopK, type Ranked } from "./ranking";
import { asNumber, asString } from "./score";

/** The embedding model and width a graph's chunks were embedded with, as its run recorded them. */
export interface RecordedEmbedding {
  model: string | null;
  dimension: number | null;
}

/** Why a search ran on BM25 alone. */
export type DenseFallback =
  | "no_model"
  | "unrecorded_model"
  | "model_mismatch"
  | "no_vectors"
  | "width_mismatch"
  | "embed_failed";

/**
 * The embedding a graph's run recorded: `embedding.model` and
 * `embedding.dimension` in graph_metrics.json, else in run_report.json.
 */
export function recordedEmbedding(dataset: GraphDataset): RecordedEmbedding {
  for (const report of [dataset.graphMetrics, dataset.runReport]) {
    const embedding = report?.embedding;
    if (!embedding || typeof embedding !== "object" || Array.isArray(embedding)) {
      continue;
    }
    const row = embedding as Record<string, unknown>;
    const model = asString(row.model).trim() || null;
    const width = asNumber(row.dimension, Number.NaN);
    const dimension = Number.isInteger(width) && width > 0 ? width : null;
    if (model || dimension) {
      return { model, dimension };
    }
  }
  return { model: null, dimension: null };
}

/**
 * Whether the configured model is the one the chunks were embedded with.
 * Case is ignored, and so is a hub organisation prefix: a server often serves
 * "Org/Model-Name" as "model-name". Anything else - another size, another
 * version - is a different vector space and fails.
 */
export function sameEmbeddingModel(configured: string, recorded: string): boolean {
  const normal = (id: string) => id.trim().toLowerCase();
  const bare = (id: string) => normal(id).split("/").pop() ?? "";
  return normal(configured) === normal(recorded) || bare(configured) === bare(recorded);
}

const vectorsByDataset = new WeakMap<GraphDataset, Promise<ChunkVectors | null>>();

/** Use these vectors for a dataset's chunks, in place of reading its artifacts. */
export function attachChunkVectors(dataset: GraphDataset, vectors: ChunkVectors | null): void {
  vectorsByDataset.set(dataset, Promise.resolve(vectors));
}

/**
 * A dataset's chunk vectors, read from its artifacts the first time a search
 * needs them and shared by every later search of the same loaded graph.
 * Null for a graph with no vectors or none on disk.
 */
export function chunkVectors(dataset: GraphDataset): Promise<ChunkVectors | null> {
  let pending = vectorsByDataset.get(dataset);
  if (!pending) {
    const directory = retrievalSource(dataset)?.directory;
    const ids = dataset.chunks.map((chunk) => asString(chunk.id ?? chunk.chunk_id));
    pending = directory
      ? readChunkVectors(directory, ids).catch((error: unknown) => {
          console.warn(`Ask: chunk vectors could not be read from ${directory}: ${errorMessage(error)}`);
          return null;
        })
      : Promise.resolve(null);
    vectorsByDataset.set(dataset, pending);
  }
  return pending;
}

/** Why dense search cannot run for this graph and model, before any vector is read; null when it can. */
export function denseBlocker(dataset: GraphDataset, embedding: AskEmbedding | null): DenseFallback | null {
  if (!embedding) {
    return "no_model";
  }
  const recorded = recordedEmbedding(dataset);
  if (!recorded.model) {
    return "unrecorded_model";
  }
  return sameEmbeddingModel(embedding.modelId, recorded.model) ? null : "model_mismatch";
}

/** Start reading a graph's chunk vectors when dense search will want them, so the first search does not wait as long. */
export function prepareDense(dataset: GraphDataset, embedding: AskEmbedding | null): void {
  if (denseBlocker(dataset, embedding) === null) {
    void chunkVectors(dataset);
  }
}

export interface DenseRankingOptions {
  embedding: AskEmbedding | null;
  limit: number;
  allow?: (chunk: number) => boolean;
  abortSignal?: AbortSignal;
}

/**
 * The chunks nearest a query by cosine similarity, best first; null when
 * dense search is not available for this graph, which callers answer with
 * BM25 alone.
 *
 * It runs only when an embedding model is configured, the graph recorded the
 * model its chunks were embedded with, the two are the same, and the stored
 * vectors, the recorded width and the query vector all have one width:
 * vectors from different models are not comparable, and comparing them would
 * rank noise. Each reason to fall back is logged once per process.
 */
export async function denseRanking(
  dataset: GraphDataset,
  query: string,
  options: DenseRankingOptions,
): Promise<Ranked[] | null> {
  const blocker = denseBlocker(dataset, options.embedding);
  if (blocker || !options.embedding) {
    return fallBack(blocker ?? "no_model", dataset);
  }
  const vectors = await chunkVectors(dataset);
  if (!vectors) {
    return fallBack("no_vectors", dataset);
  }
  const recorded = recordedEmbedding(dataset);
  if (recorded.dimension !== null && recorded.dimension !== vectors.dimension) {
    return fallBack("width_mismatch", dataset, `recorded ${recorded.dimension}, stored ${vectors.dimension}`);
  }
  let queryVector: number[];
  try {
    const signal = options.abortSignal
      ? AbortSignal.any([options.abortSignal, AbortSignal.timeout(ASK_EMBEDDING_TIMEOUT_MS)])
      : AbortSignal.timeout(ASK_EMBEDDING_TIMEOUT_MS);
    // One attempt: a retry would spend the time the timeout is there to save.
    queryVector = (await embed({ model: options.embedding.model, value: query, abortSignal: signal, maxRetries: 0 })).embedding;
  } catch (error) {
    return fallBack("embed_failed", dataset, errorMessage(error));
  }
  if (queryVector.length !== vectors.dimension) {
    return fallBack("width_mismatch", dataset, `query ${queryVector.length}, stored ${vectors.dimension}`);
  }
  return nearestChunks(vectors, queryVector, options.limit, options.allow);
}

/**
 * Brute-force cosine ranking: one pass over the stored unit vectors, the best
 * kept in a bounded heap. At a hundred thousand chunks of a thousand
 * components that is a hundred million multiply-adds - under a tenth of a
 * second - with no per-chunk allocation.
 */
export function nearestChunks(
  vectors: ChunkVectors,
  queryVector: ArrayLike<number>,
  limit: number,
  allow?: (chunk: number) => boolean,
): Ranked[] {
  const { dimension, data, present } = vectors;
  const unit = new Float64Array(dimension);
  let squared = 0;
  for (let component = 0; component < dimension; component += 1) {
    const value = Number(queryVector[component]);
    unit[component] = value;
    squared += value * value;
  }
  if (!(squared > 0) || !Number.isFinite(squared)) {
    return [];
  }
  const scale = 1 / Math.sqrt(squared);
  for (let component = 0; component < dimension; component += 1) {
    unit[component]! *= scale;
  }
  const top = new TopK(limit);
  // Four running sums rather than one: independent additions the engine can
  // overlap, about twice as fast as a single accumulator at this width.
  const unrolled = dimension - (dimension % 4);
  for (let chunk = 0; chunk < present.length; chunk += 1) {
    if (present[chunk] === 0 || (allow && !allow(chunk))) {
      continue;
    }
    const offset = chunk * dimension;
    let first = 0;
    let second = 0;
    let third = 0;
    let fourth = 0;
    let component = 0;
    for (; component < unrolled; component += 4) {
      const at = offset + component;
      first += data[at]! * unit[component]!;
      second += data[at + 1]! * unit[component + 1]!;
      third += data[at + 2]! * unit[component + 2]!;
      fourth += data[at + 3]! * unit[component + 3]!;
    }
    for (; component < dimension; component += 1) {
      first += data[offset + component]! * unit[component]!;
    }
    top.offer(chunk, first + second + third + fourth);
  }
  return top.ranked();
}

const FALLBACK_REASONS: Record<DenseFallback, string> = {
  no_model: "no FLAKEGRAPH_ASK_EMBEDDING_MODEL is configured",
  unrecorded_model: "the graph does not record the embedding model its chunks were embedded with",
  model_mismatch: "the configured embedding model is not the one the graph's chunks were embedded with",
  no_vectors: "the graph has no chunk vectors",
  width_mismatch: "the vector widths do not match",
  embed_failed: "the query could not be embedded",
};

const reported = new Set<DenseFallback>();

/** Forget which fallbacks were logged, for tests. */
export function resetDenseReports(): void {
  reported.clear();
}

function fallBack(reason: DenseFallback, dataset: GraphDataset, detail?: string): null {
  if (!reported.has(reason)) {
    reported.add(reason);
    const suffix = detail ? ` (${detail})` : "";
    console.info(`Ask: passage search uses BM25 alone, since ${FALLBACK_REASONS[reason]}${suffix}; graph ${dataset.graphId}. Said once per process.`);
  }
  return null;
}
