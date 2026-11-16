import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import type { GraphDataset } from "./protocol/schema";

interface GoldDocument {
  id: string;
  path: string;
  title?: string;
}

interface GoldEntity {
  id: string;
  name: string;
  type: string;
  aliases?: string[];
  description?: string;
}

interface GoldRelation {
  id: string;
  source: string;
  target: string;
  relation_type: string;
  evidence_contains?: string[];
  observations?: Array<{ document?: string; sentence?: string; evidence_contains?: string }>;
  required?: boolean;
}

export interface GoldGraph {
  name?: string;
  description?: string;
  evaluation_scope?: { entity_coverage?: "exhaustive" | "reference"; relation_coverage?: "exhaustive" | "induced" | "reference" };
  documents?: GoldDocument[];
  entities?: GoldEntity[];
  relations?: GoldRelation[];
}

export function goldToDataset(gold: GoldGraph, graphId: string): GraphDataset {
  const nodes = (gold.entities ?? []).map((entity) => ({
    id: entity.id,
    graph_id: graphId,
    name: entity.name,
    primary_type: entity.type,
    aliases: entity.aliases ?? [],
    description: entity.description ?? (entity.aliases ?? []).join(", "),
  }));
  const edges = (gold.relations ?? []).map((relation) => ({
    id: relation.id,
    graph_id: graphId,
    source_node_id: relation.source,
    target_node_id: relation.target,
    relation_type: relation.relation_type,
    confidence: relation.required === false ? 0.6 : 0.95,
    description: relation.evidence_contains?.[0] ?? "",
  }));
  const evidence = (gold.relations ?? []).flatMap((relation) =>
    (relation.observations ?? []).map((observation, index) => ({
      id: `${relation.id}_ev_${index}`,
      relation_id: relation.id,
      document_id: observation.document ?? "",
      quote: observation.sentence ?? observation.evidence_contains ?? "",
    })),
  );
  const byType = new Map<string, string[]>();
  for (const node of nodes) {
    const type = String(node.primary_type);
    const members = byType.get(type) ?? [];
    members.push(String(node.id));
    byType.set(type, members);
  }
  const communities = [...byType.entries()].map(([type, members]) => ({
    id: `community_${type.toLowerCase()}`,
    title: type,
    summary: `${members.length} ${type} entities`,
    members,
    // The pipeline's community reports carry the questions they can answer;
    // a gold-derived community offers the one its members plainly can.
    suggested_questions: [`Which ${type.toLowerCase().replaceAll("_", " ")} entities connect to the most others?`],
  }));
  const documents = (gold.documents ?? []).map((document) => ({
    id: document.id,
    path: document.path ?? "",
    title: document.title ?? document.id,
  }));
  return {
    graphId,
    nodes,
    edges,
    communities,
    evidence,
    documents,
    chunks: [],
    discardedWindows: [],
    rejectedRecords: [],
    failedDocuments: [],
    runReport: {
      graph_id: graphId,
      name: gold.name ?? graphId,
      description: gold.description ?? "",
    },
    graphMetrics: {
      consumption: {
        totals: { billed_usd: 0, avoided_usd: graphId.includes("deep") ? 12.4 : 3.1, unpriced_calls: 0 },
        events: [
          {
            stage: "graph_extraction",
            operation: "complete",
            provider: "vllm_local",
            model: "Qwen/Qwen3-8B",
            locality: "local",
            calls: edges.length,
            prompt_tokens: edges.length * 400,
            completion_tokens: edges.length * 80,
            usage: { prompt_tokens: edges.length * 400, completion_tokens: edges.length * 80, total_tokens: edges.length * 480 },
            pages: documents.length,
          },
        ],
      },
    },
  };
}

export async function writeDataset(directory: string, dataset: GraphDataset): Promise<void> {
  await mkdir(directory, { recursive: true });
  await writeJson(path.join(directory, "nodes.json"), dataset.nodes);
  await writeJson(path.join(directory, "edges.json"), dataset.edges);
  await writeJson(path.join(directory, "communities.json"), dataset.communities);
  await writeJson(path.join(directory, "evidence.json"), dataset.evidence);
  await writeJson(path.join(directory, "documents.json"), dataset.documents);
  if (dataset.discardedWindows.length > 0) {
    await writeJson(path.join(directory, "discarded_windows.json"), dataset.discardedWindows);
  }
  if (dataset.rejectedRecords.length > 0) {
    await writeJson(path.join(directory, "rejected_records.json"), dataset.rejectedRecords);
  }
  if (dataset.failedDocuments.length > 0) {
    await writeJson(path.join(directory, "failed_documents.json"), dataset.failedDocuments);
  }
  await writeJson(path.join(directory, "run_report.json"), dataset.runReport);
  await writeJson(path.join(directory, "graph_metrics.json"), dataset.graphMetrics);
}

export interface GoldComparison {
  goldName: string;
  expectedEntities: number;
  foundEntities: number;
  expectedRelations: number;
  foundRelations: number;
  requiredTotal: number;
  matchedRequired: number;
  missingRequired: Array<{
    id: string;
    source: string;
    target: string;
    relationType: string;
  }>;
}

export function compareToGold(dataset: GraphDataset, gold: GoldGraph): GoldComparison {
  const names = new Set(
    dataset.nodes.flatMap((node) => {
      const aliases = Array.isArray(node.aliases) ? node.aliases.map(String) : [];
      return [String(node.name ?? ""), String(node.id ?? ""), ...aliases].map(normalizeName).filter(Boolean);
    }),
  );
  const goldById = new Map((gold.entities ?? []).map((entity) => [entity.id, entity]));
  const required = (gold.relations ?? []).filter((relation) => relation.required !== false);
  const missingRequired = required.flatMap((relation) => {
    const source = goldById.get(relation.source);
    const target = goldById.get(relation.target);
    const sourceName = source ? normalizeName(source.name) : "";
    const targetName = target ? normalizeName(target.name) : "";
    const matches = (name: string) => new Set(dataset.nodes.filter((node) =>
      [node.name, node.id, ...(Array.isArray(node.aliases) ? node.aliases : [])]
        .some((value) => normalizeName(String(value ?? "")) === name),
    ).map((node) => String(node.id)));
    const sourceIds = matches(sourceName);
    const targetIds = matches(targetName);
    const present = Boolean(sourceName && targetName && names.has(sourceName) && names.has(targetName)) &&
      dataset.edges.some((edge) => sourceIds.has(String(edge.source_node_id ?? edge.source)) &&
        targetIds.has(String(edge.target_node_id ?? edge.target)) &&
        normalizePredicate(String(edge.relation_type)) === normalizePredicate(relation.relation_type));
    if (present) {
      return [];
    }
    return [
      {
        id: relation.id,
        source: source?.name ?? relation.source,
        target: target?.name ?? relation.target,
        relationType: relation.relation_type,
      },
    ];
  });
  return {
    goldName: gold.name ?? "gold",
    expectedEntities: gold.entities?.length ?? 0,
    foundEntities: dataset.nodes.length,
    expectedRelations: gold.relations?.length ?? 0,
    foundRelations: dataset.edges.length,
    requiredTotal: required.length,
    matchedRequired: required.length - missingRequired.length,
    missingRequired,
  };
}

export function ontologyCoverage(proposedTypes: readonly string[], gold: GoldGraph): {
  proposed: string[];
  goldTypes: string[];
  covered: string[];
  missing: string[];
} {
  const goldTypes = [...new Set((gold.entities ?? []).map((entity) => entity.type.toUpperCase()))];
  const proposed = [...new Set(proposedTypes.map((type) => type.toUpperCase()).filter(Boolean))];
  const proposedSet = new Set(proposed);
  return {
    proposed,
    goldTypes,
    covered: goldTypes.filter((type) => proposedSet.has(type)),
    missing: goldTypes.filter((type) => !proposedSet.has(type)),
  };
}

function normalizePredicate(value: string): string {
  return value.toUpperCase().replace(/-/g, " ").trim().split(/\s+/).join("_");
}

function normalizeName(value: string): string {
  return value.trim().toLowerCase();
}

async function writeJson(file: string, value: unknown): Promise<void> {
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
