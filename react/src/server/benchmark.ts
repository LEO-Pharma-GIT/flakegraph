import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { atomicWriteJson, readJsonFile } from "./catalog";
import { cliFailure, lastJsonObject, runFlakegraph } from "./cli";

/** One acceptance gate of the gold file: what it asks for, and whether the graph meets it. */
export interface BenchmarkGate {
  gate: string;
  passed: boolean;
}

/**
 * How a graph scores against its gold, as the pipeline's own evaluator
 * (`flakegraph inspect evaluate`) measures it - the numbers the benchmarks
 * report. Precision and F1 are null when the gold is a reference rather than
 * an exhaustive list: extra findings cannot then be counted as wrong.
 */
export interface BenchmarkReport {
  ok: boolean;
  goldName: string;
  coverage: { entities: string; relations: string };
  entities: {
    precision: number | null;
    recall: number;
    f1: number | null;
    expected: number;
    found: number;
    missing: string[];
  };
  triples: {
    precision: number | null;
    recall: number;
    f1: number | null;
    expected: number;
    found: number;
    missing: string[];
    missingRelationIds: string[];
    missingByReason: Record<string, number>;
  };
  evidenceSupport: number | null;
  twoHopRecoverability: number | null;
  informationRetention: number | null;
  gates: BenchmarkGate[];
  thresholds: Record<string, number>;
  evaluatedAt: string;
}

/** Missing items shown per list; the counts say how many there are in all. */
const MISSING_LIMIT = 200;

/**
 * Score a graph against its gold with the pipeline's evaluator.
 *
 * The result is cached beside the console's other state and reused while the
 * gold file and the graph's files are unchanged, so opening the Quality tab
 * does not re-score a graph that has not moved.
 */
export async function benchmarkGraph(options: {
  graphDirectory: string;
  goldPath: string;
  cacheFile: string;
  repositoryRoot: string;
}): Promise<BenchmarkReport> {
  const key = await cacheKey(options.graphDirectory, options.goldPath);
  const cached = await readJsonFile<{ key: string; report: BenchmarkReport }>(options.cacheFile);
  if (cached?.key === key) {
    return cached.report;
  }
  const result = await runFlakegraph(
    ["inspect", "evaluate", "--gold", options.goldPath, "--output", options.graphDirectory, "--no-fail"],
    { cwd: options.repositoryRoot },
  );
  const raw = result.exitCode === 0 ? lastJsonObject(result.stdout) : null;
  if (!raw) {
    throw new Error(cliFailure(result.stderr, "The evaluator returned no result"));
  }
  const report = benchmarkReport(raw);
  await atomicWriteJson(options.cacheFile, { key, report });
  return report;
}

/** The evaluator's JSON, kept to what the Quality tab shows. */
export function benchmarkReport(raw: Record<string, unknown>): BenchmarkReport {
  const record = (value: unknown) => (value && typeof value === "object" ? (value as Record<string, unknown>) : {});
  const number = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const strings = (value: unknown) => (Array.isArray(value) ? value.map(String) : []);
  const entities = record(raw.entities);
  const triples = record(raw.triples);
  const scope = record(raw.evaluation_scope);
  const acceptance = record(raw.acceptance);
  const thresholds = Object.fromEntries(
    Object.entries(record(raw.thresholds)).flatMap(([name, value]) => (typeof value === "number" ? [[name, value]] : [])),
  );
  return {
    ok: raw.ok === true,
    goldName: String(raw.gold_name ?? "gold"),
    coverage: {
      entities: String(scope.entity_coverage ?? "exhaustive"),
      relations: String(scope.relation_coverage ?? "exhaustive"),
    },
    entities: {
      precision: number(entities.precision),
      recall: number(entities.recall) ?? 0,
      f1: number(entities.f1),
      expected: number(entities.expected) ?? 0,
      found: (number(entities.expected) ?? 0) - strings(entities.missing).length,
      missing: strings(entities.missing).slice(0, MISSING_LIMIT),
    },
    triples: {
      precision: number(triples.precision),
      recall: number(triples.recall) ?? 0,
      f1: number(triples.f1),
      expected: number(triples.expected) ?? 0,
      found: (number(triples.expected) ?? 0) - strings(triples.missing).length,
      missing: strings(triples.missing).slice(0, MISSING_LIMIT),
      missingRelationIds: strings(triples.missing_relation_ids),
      missingByReason: Object.fromEntries(
        Object.entries(record(triples.missing_by_reason)).flatMap(([reason, count]) =>
          typeof count === "number" && count > 0 ? [[reason, count]] : [],
        ),
      ),
    },
    evidenceSupport: number(record(raw.evidence).support_rate),
    twoHopRecoverability: number(raw.two_hop_recoverability),
    informationRetention: number(raw.information_retention),
    gates: Object.entries(acceptance).map(([gate, passed]) => ({ gate, passed: passed === true })),
    thresholds,
    evaluatedAt: new Date().toISOString(),
  };
}

/** The gold file's bytes and the graph files' sizes and times: what a score depends on. */
async function cacheKey(graphDirectory: string, goldPath: string): Promise<string> {
  const hash = createHash("sha256");
  hash.update(await readFile(goldPath));
  const entries = await readdir(graphDirectory, { withFileTypes: true });
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const info = await stat(path.join(graphDirectory, entry.name));
    hash.update(`${entry.name}:${info.size}:${info.mtimeMs}\n`);
  }
  return hash.digest("hex");
}

/** One recorded benchmark result a sample pack ships: a model's scores on the same gold. */
export interface BenchmarkBaseline {
  resultId: string;
  measuredAt: string | null;
  model: string;
  entities: { precision: number | null; recall: number | null; f1: number | null };
  triples: { precision: number | null; recall: number | null; f1: number | null };
  evidenceSupport: number | null;
  twoHopRecoverability: number | null;
  informationRetention: number | null;
  thresholdsMet: boolean | null;
}

/**
 * The results a sample pack records beside its gold (`results/*.json`), newest
 * first, so a graph built from the pack can be read against the models already
 * measured on it. A result on another gold, or on an earlier version of this
 * one, is not a baseline and is left out.
 */
export async function packBaselines(goldPath: string): Promise<BenchmarkBaseline[]> {
  const directory = path.join(path.dirname(goldPath), "results");
  let names: string[];
  try {
    names = (await readdir(directory)).filter((name) => name.endsWith(".json"));
  } catch {
    return [];
  }
  const gold = JSON.parse(await readFile(goldPath, "utf8")) as { name?: unknown; version?: unknown };
  const baselines = await Promise.all(
    names.map(async (name) => {
      const raw = await readJsonFile<Record<string, unknown>>(path.join(directory, name));
      return raw ? baselineFromResult(raw, gold) : null;
    }),
  );
  return baselines
    .filter((item): item is BenchmarkBaseline => item !== null)
    .sort((left, right) => (right.measuredAt ?? "").localeCompare(left.measuredAt ?? ""));
}

function baselineFromResult(
  raw: Record<string, unknown>,
  gold: { name?: unknown; version?: unknown },
): BenchmarkBaseline | null {
  const record = (value: unknown) => (value && typeof value === "object" ? (value as Record<string, unknown>) : {});
  const number = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const dataset = record(raw.dataset);
  const differs = (field: unknown, value: unknown) => typeof value === "string" && field !== undefined && field !== value;
  if (differs(dataset.id, gold.name) || differs(dataset.version, gold.version)) {
    return null;
  }
  const quality = record(record(raw.measurements).quality);
  if (Object.keys(quality).length === 0) {
    return null;
  }
  const llm = record(record(raw.models).llm);
  const met = quality.acceptance_thresholds_met;
  return {
    resultId: String(raw.result_id ?? "result"),
    measuredAt: typeof raw.measured_at_utc === "string" ? raw.measured_at_utc : null,
    model: String(llm.name ?? "unknown model"),
    entities: {
      precision: number(quality.entity_precision),
      recall: number(quality.entity_recall),
      f1: number(quality.entity_f1),
    },
    triples: {
      precision: number(quality.triple_precision),
      recall: number(quality.triple_recall),
      f1: number(quality.triple_f1),
    },
    evidenceSupport: number(quality.evidence_support_rate),
    twoHopRecoverability: number(quality.two_hop_recoverability),
    informationRetention: number(quality.information_retention),
    thresholdsMet: typeof met === "boolean" ? met : null,
  };
}
