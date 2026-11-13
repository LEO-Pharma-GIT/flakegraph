import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { benchmarkGraph, packBaselines, type BenchmarkBaseline, type BenchmarkReport } from "./benchmark";
import { failedDocuments, type FailedDocuments } from "./failed-documents";
import { extractionGaps, type ExtractionGaps } from "./gaps";
import { rejectedRecords, type RejectedRecords } from "./rejected";
import { compareToGold, type GoldComparison, type GoldGraph } from "./gold";
import { goldBesideSource, uploadedGoldPath } from "./gold-store";
import { graphCounts, type GraphDataset, type RunSnapshot } from "./protocol/schema";

/** Where the gold came from, so the tab can say it and offer the right actions. */
export type GoldSource = "uploaded" | "beside-source";

export interface QualityReport {
  nodeCount: number;
  edgeCount: number;
  documentCount: number;
  documentsFailed: number;
  zeroEntityFailure: boolean;
  gold: GoldComparison | null;
  goldSource: GoldSource | null;
  /** The pipeline's own scores against the gold - what the local benchmarks report. */
  benchmark: BenchmarkReport | null;
  /** Why the gold could not be scored, when there is one and scoring failed. */
  benchmarkError: string | null;
  /** Scores the sample pack records for models already measured on the same gold. */
  baselines: BenchmarkBaseline[];
  /** Text the model read and nothing was kept from - what the graph is known to be missing. */
  gaps: ExtractionGaps;
  /** Every record the model stated that validation did not keep - what the graph may be missing. */
  rejected: RejectedRecords;
  /** Source documents no parser could read: nothing from them is in the graph. */
  failed: FailedDocuments;
}

export async function qualityForGraph(options: {
  snapshot: RunSnapshot;
  dataset: GraphDataset;
  repositoryRoot: string;
  stateRoot: string;
  /** The graph's files on this host; without them the gold is compared by name only. */
  graphDirectory?: string | null;
}): Promise<QualityReport> {
  const counts = graphCounts(options.dataset);
  const located = await loadGoldForGraph(options.snapshot, options.repositoryRoot, options.stateRoot);
  const gold = located?.gold ?? null;
  let benchmark: BenchmarkReport | null = null;
  let benchmarkError: string | null = null;
  if (located && options.graphDirectory) {
    try {
      benchmark = await benchmarkGraph({
        graphDirectory: options.graphDirectory,
        goldPath: located.path,
        cacheFile: path.join(options.stateRoot, "benchmarks", `${options.snapshot.runId}.json`),
        repositoryRoot: options.repositoryRoot,
      });
    } catch (error) {
      benchmarkError = error instanceof Error ? error.message : String(error);
    }
  }
  const comparison = gold ? withEvaluatorMisses(compareToGold(options.dataset, gold), gold, benchmark) : null;
  const zeroEntityFailure = counts.nodes === 0;
  const failed = failedDocuments(options.dataset);
  return {
    nodeCount: counts.nodes,
    edgeCount: counts.edges,
    documentCount: counts.documents,
    // The graph's own record once it has one; the run's count while it has not.
    documentsFailed: Math.max(failed.documents, options.snapshot.documentsFailed ?? 0),
    zeroEntityFailure,
    gold: comparison,
    goldSource: located?.source ?? null,
    benchmark,
    benchmarkError,
    baselines: located?.source === "beside-source" ? await packBaselines(located.path) : [],
    gaps: extractionGaps(options.dataset),
    rejected: rejectedRecords(options.dataset),
    failed,
  };
}

/**
 * The gold a graph is held to: one a person uploaded for this graph first,
 * else the one shipped beside the run's source folder (a sample pack).
 * Never guessed from the graph's name - a graph called "deep dive" is not a
 * deep-learning benchmark.
 */
async function loadGoldForGraph(
  snapshot: RunSnapshot,
  repositoryRoot: string,
  stateRoot: string,
): Promise<{ gold: GoldGraph; path: string; source: GoldSource } | null> {
  const uploaded = uploadedGoldPath(stateRoot, snapshot.graphId);
  if (existsSync(uploaded)) {
    return { gold: JSON.parse(await readFile(uploaded, "utf8")) as GoldGraph, path: uploaded, source: "uploaded" };
  }
  const sourcePath = snapshot.raw.sourcePath;
  const beside = goldBesideSource(typeof sourcePath === "string" ? sourcePath : null, repositoryRoot);
  if (beside) {
    return { gold: JSON.parse(await readFile(beside, "utf8")) as GoldGraph, path: beside, source: "beside-source" };
  }
  return null;
}

/**
 * The required relations the evaluator found missing, in place of the
 * by-name guess, once it has scored the graph: the evaluator matches the
 * relation itself, not only its two ends, so the list and the
 * scores all agree.
 */
function withEvaluatorMisses(
  comparison: GoldComparison,
  gold: GoldGraph,
  benchmark: BenchmarkReport | null,
): GoldComparison {
  if (!benchmark) {
    return comparison;
  }
  const missing = new Set(benchmark.triples.missingRelationIds);
  const names = new Map((gold.entities ?? []).map((entity) => [entity.id, entity.name]));
  const missingRequired = (gold.relations ?? [])
    .filter((relation) => relation.required !== false && missing.has(relation.id))
    .map((relation) => ({
      id: relation.id,
      source: names.get(relation.source) ?? relation.source,
      target: names.get(relation.target) ?? relation.target,
      relationType: relation.relation_type,
    }));
  return { ...comparison, missingRequired, matchedRequired: comparison.requiredTotal - missingRequired.length };
}
