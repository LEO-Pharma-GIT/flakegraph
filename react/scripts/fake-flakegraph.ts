#!/usr/bin/env bun
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { goldToDataset, writeDataset } from "../src/server/gold";
import type { GoldGraph } from "../src/server/gold";

const [, , command, ...rest] = process.argv;

async function main() {
  if (command === "preflight") {
    await preflight(argValue("--config"));
    return;
  }
  if (command === "worker") {
    await worker(argValue("--config"));
    return;
  }
  if (command === "distributed") {
    await distributed(rest[0] ?? "", argValue("--config"), argValue("--run-id"));
    return;
  }
  if (command === "inspect" && rest[0] === "evaluate") {
    await evaluate(argValue("--gold"), argValue("--output"));
    return;
  }
  if (command === "sources" && rest[0] === "list") {
    await sourcesList(argValue("--config"), Number(argValue("--limit") || 1000));
    return;
  }
  console.error(`Unknown flakegraph command: ${command}`);
  process.exit(1);
}

// The evaluator's report, scored by name: a gold entity is found when a node
// carries its name, a gold relation when an edge joins the two by its type.
async function evaluate(goldPath: string, output: string) {
  const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph & {
    evaluation_scope?: { entity_coverage?: string; relation_coverage?: string };
  };
  const read = async (name: string) =>
    JSON.parse(await readFile(path.join(output, name), "utf8").catch(() => "[]")) as Array<Record<string, unknown>>;
  const nodes = await read("nodes.json");
  const edges = await read("edges.json");
  const norm = (value: unknown) => String(value ?? "").trim().toLowerCase();
  const nameById = new Map(nodes.map((node) => [String(node.id), norm(node.name)]));
  const names = new Set(nameById.values());
  const entities = gold.entities ?? [];
  const missingEntities = entities.filter((entity) => !names.has(norm(entity.name)));
  const goldNames = new Map(entities.map((entity) => [entity.id, norm(entity.name)]));
  const triples = new Set(
    edges.map(
      (edge) =>
        `${nameById.get(String(edge.source_node_id))}|${norm(edge.relation_type)}|${nameById.get(String(edge.target_node_id))}`,
    ),
  );
  const required = (gold.relations ?? []).filter((relation) => relation.required !== false);
  const missingRelations = required.filter(
    (relation) =>
      !triples.has(`${goldNames.get(relation.source)}|${norm(relation.relation_type)}|${goldNames.get(relation.target)}`),
  );
  const scope = { entity_coverage: "exhaustive", relation_coverage: "exhaustive", ...gold.evaluation_scope };
  const score = (expected: number, missing: number, actual: number, exhaustive: boolean) => {
    const recall = expected ? (expected - missing) / expected : 1;
    const precision = exhaustive ? (actual ? (expected - missing) / actual : 0) : null;
    const f1 = precision === null ? null : precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { recall, precision, f1 };
  };
  const entityScore = score(entities.length, missingEntities.length, nodes.length, scope.entity_coverage === "exhaustive");
  const tripleScore = score(required.length, missingRelations.length, edges.length, scope.relation_coverage === "exhaustive");
  const acceptance = { entity_recall: entityScore.recall >= 0.8, triple_recall: tripleScore.recall >= 0.8, evidence_support: true };
  process.stdout.write(
    `${JSON.stringify({
      gold_name: gold.name ?? "gold",
      evaluation_scope: scope,
      ok: Object.values(acceptance).every(Boolean),
      acceptance,
      thresholds: { entity_recall: 0.8, triple_recall: 0.8, min_evidence_support: 0.9 },
      entities: { ...entityScore, expected: entities.length, missing: missingEntities.map((entity) => entity.name) },
      triples: {
        ...tripleScore,
        expected: required.length,
        missing: missingRelations.map((relation) => `${relation.source} ${relation.relation_type} ${relation.target}`),
        missing_relation_ids: missingRelations.map((relation) => relation.id).sort(),
        missing_by_reason: missingRelations.length ? { no_edge: missingRelations.length } : {},
      },
      evidence: { support_rate: 1 },
      two_hop_recoverability: tripleScore.recall,
      information_retention: (entityScore.recall + tripleScore.recall) / 2,
    })}\n`,
  );
}

function argValue(flag: string): string {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] ?? "" : "";
}

async function preflight(configPath: string) {
  const config = await readConfig(configPath);
  const inputPath = String((config.files as Record<string, unknown> | undefined)?.input_path ?? "");
  if (inputPath.includes("missing-corpus")) {
    process.stdout.write(`${JSON.stringify({ ok: false, errors: ["Input path does not exist"] })}\n`);
    process.exit(1);
  }
  process.stdout.write(
    `${JSON.stringify({ ok: true, errors: [], warnings: [], checks: [{ name: "source", ok: true }] })}\n`,
  );
}

// A bucket listing shaped like the real command's: the fake bucket holds
// three objects under the requested prefix, one of them unsupported.
async function sourcesList(configPath: string, limit: number) {
  const config = await readConfig(configPath);
  const kind = String((config.files as Record<string, unknown> | undefined)?.source ?? "");
  if (kind !== "s3" && kind !== "azure_blob") {
    process.stderr.write(`${kind} sources cannot be listed without fetching\n`);
    process.exit(2);
  }
  const backend = (config[kind] ?? {}) as Record<string, unknown>;
  const container = String(backend.bucket ?? backend.container ?? "");
  if (!container) {
    process.stderr.write(`${kind} file source requires a bucket\n`);
    process.exit(1);
  }
  if (container === "slow") {
    // A credential chain that never answers.
    await new Promise((resolve) => setTimeout(resolve, 60_000));
  }
  if (container === "missing") {
    process.stderr.write("NoSuchBucket: The specified bucket does not exist\n");
    process.exit(1);
  }
  const prefix = String(backend.prefix ?? "");
  const scheme = kind === "s3" ? "s3" : "az";
  const rows = [
    { name: "judo.md", size_bytes: 512 },
    { name: "karate.pdf", size_bytes: 40_960 },
  ].map((row) => ({
    uri: `${scheme}://${container}/${prefix}${row.name}`,
    name: row.name,
    size_bytes: row.size_bytes,
    modified_at: "2026-09-18T10:00:00+00:00",
    checksum: "etag-" + row.name,
  }));
  process.stdout.write(`${JSON.stringify(rows.slice(0, limit), null, 2)}\n`);
}

async function worker(configPath: string) {
  const config = await readConfig(configPath);
  const outputPath = String((config.writer as Record<string, unknown> | undefined)?.output_path ?? "");
  const graphId = String((config.job as Record<string, unknown> | undefined)?.graph_id ?? "graph");
  const inputPath = String((config.files as Record<string, unknown> | undefined)?.input_path ?? "");
  const fail = inputPath.includes("fail-worker") || process.env.FAKE_FLAKEGRAPH_FAIL === "1";
  const delay = Number(process.env.FAKE_FLAKEGRAPH_DELAY_MS ?? 150);
  const files = inputPath.includes("deep_learning") ? 8 : 10;
  emit("file_source", "started", { files_seen: files });
  await sleep(delay);
  emit("file_source", "completed", { files_seen: files, files_total: files });
  for (let index = 0; index < Math.min(files, 3); index += 1) {
    emit("ocr", "running", { files_total: files }, `doc_${index}`);
    await sleep(delay);
    emit("ocr", "completed", { files_total: files }, `doc_${index}`);
  }
  if (fail) {
    emit("graph_extraction", "failed", {}, null, "Simulated worker failure");
    process.exit(1);
  }
  emit("write", "completed", { files_total: files });
  await mkdir(outputPath, { recursive: true });
  const repo = process.env.FLAKEGRAPH_REPOSITORY_ROOT || process.cwd();
  const goldPath = inputPath.includes("deep_learning")
    ? path.resolve(repo, "data/deep_learning_papers/gold.json")
    : path.resolve(repo, "data/martial_arts/gold.json");
  try {
    const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph;
    const dataset = goldToDataset(gold, graphId);
    await writeDataset(outputPath, { ...dataset, runReport: { ...dataset.runReport, job_id: (config.job as Record<string, unknown> | undefined)?.job_id } });
  } catch {
    const dataset = goldToDataset({ name: graphId, entities: [{ id: "n1", name: "Node", type: "THING" }], relations: [] }, graphId);
    await writeDataset(outputPath, { ...dataset, runReport: { ...dataset.runReport, job_id: (config.job as Record<string, unknown> | undefined)?.job_id } });
  }
}

async function distributed(sub: string, configPath: string, runId: string) {
  const id = runId || "distributed-run";
  if (sub === "submit") {
    process.stdout.write(`${JSON.stringify({ id, status: "queued", config: configPath })}\n`);
    return;
  }
  if (sub === "status") {
    process.stdout.write(`${JSON.stringify({ id, status: "running" })}\n`);
    return;
  }
  if (sub === "cancel") {
    process.stdout.write(`${JSON.stringify({ id, status: "cancelled" })}\n`);
    return;
  }
  if (sub === "retry") {
    process.stdout.write(`${JSON.stringify({ id, status: "queued" })}\n`);
    return;
  }
  process.stdout.write(`${JSON.stringify({ ok: true, sub })}\n`);
}

async function readConfig(configPath: string): Promise<Record<string, unknown>> {
  const loaded = parseYaml(await readFile(configPath, "utf8")) ?? {};
  return loaded as Record<string, unknown>;
}

function emit(
  stage: string,
  status: string,
  counts: Record<string, unknown> = {},
  fileId: string | null = null,
  message: string | null = null,
) {
  const record = {
    event: "kg_processor.progress",
    timestamp: new Date().toISOString(),
    stage,
    status,
    file_id: fileId,
    message,
    elapsed_ms: 25,
    counts,
    metadata: {},
  };
  process.stderr.write(`${JSON.stringify(record)}\n`);
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

await main();
