import { existsSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import type { GoldGraph } from "./gold";
import type { GraphDataset } from "./protocol/schema";
import { graphEdgeEnds, graphEdgeId, graphNodeId, graphNodeLabel } from "@/lib/graph-geometry";

/** Bytes a gold file may take; a hand-written QA contract is kilobytes, not a graph dump. */
export const GOLD_MAX_BYTES = 2 * 1024 * 1024;

/** Where a graph's uploaded gold lives: the console's state, keyed by graph. */
export function uploadedGoldPath(stateRoot: string, graphId: string): string {
  const safe = graphId.replace(/[^A-Za-z0-9_.-]/g, "_");
  return path.join(stateRoot, "gold", `${safe}.json`);
}

export async function readUploadedGold(stateRoot: string, graphId: string): Promise<GoldGraph | null> {
  const file = uploadedGoldPath(stateRoot, graphId);
  if (!existsSync(file)) {
    return null;
  }
  return JSON.parse(await readFile(file, "utf8")) as GoldGraph;
}

export async function saveUploadedGold(stateRoot: string, graphId: string, gold: GoldGraph): Promise<void> {
  const file = uploadedGoldPath(stateRoot, graphId);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(gold, null, 2), "utf8");
}

export async function removeUploadedGold(stateRoot: string, graphId: string): Promise<void> {
  await rm(uploadedGoldPath(stateRoot, graphId), { force: true });
}

/**
 * A gold file beside the run's source folder - the shape the sample packs
 * ship in (`data/<pack>/files` with `gold.json` one level up) - so a pack's
 * QA contract applies to every run built from it without anyone uploading it.
 */
export function goldBesideSource(sourcePath: string | null | undefined, repositoryRoot: string): string | null {
  if (!sourcePath) {
    return null;
  }
  const absolute = path.isAbsolute(sourcePath) ? sourcePath : path.join(repositoryRoot, sourcePath);
  const candidate = path.join(path.dirname(absolute), "gold.json");
  return existsSync(candidate) ? candidate : null;
}

export class GoldValidationError extends Error {}

/**
 * Check a gold file before it is kept, and say what is wrong in the words
 * of the format: an outsider writing one by hand should be told the field,
 * not shown a stack.
 */
export function validateGold(value: unknown): GoldGraph {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new GoldValidationError("A gold file is a JSON object with entities and relations.");
  }
  const gold = value as Record<string, unknown>;
  // The evaluator that scores the graph refuses a gold file without these.
  for (const field of ["name", "description"] as const) {
    if (typeof gold[field] !== "string" || !(gold[field] as string).trim()) {
      throw new GoldValidationError(`A gold file needs a string "${field}".`);
    }
  }
  const entities = gold.entities;
  const relations = gold.relations;
  if (!Array.isArray(entities) || entities.length === 0) {
    throw new GoldValidationError('"entities" must be a non-empty list of { id, name, type }.');
  }
  const ids = new Set<string>();
  entities.forEach((entity, index) => {
    const record = entity as Record<string, unknown>;
    for (const field of ["id", "name", "type"] as const) {
      if (typeof record?.[field] !== "string" || !record[field]) {
        throw new GoldValidationError(`entities[${index}] needs a string "${field}".`);
      }
    }
    if (ids.has(record.id as string)) {
      throw new GoldValidationError(`entities[${index}] repeats the id "${String(record.id)}".`);
    }
    ids.add(record.id as string);
  });
  if (!Array.isArray(relations)) {
    throw new GoldValidationError('"relations" must be a list of { id, source, target, relation_type, required? }, even an empty one.');
  }
  relations.forEach((relation, index) => {
    const record = relation as Record<string, unknown>;
    for (const field of ["id", "source", "target", "relation_type"] as const) {
      if (typeof record?.[field] !== "string" || !record[field]) {
        throw new GoldValidationError(`relations[${index}] needs a string "${field}".`);
      }
    }
    for (const end of ["source", "target"] as const) {
      if (!ids.has(record[end] as string)) {
        throw new GoldValidationError(
          `relations[${index}].${end} is "${String(record[end])}", which is not an entity id in this file.`,
        );
      }
    }
    if (record.required !== undefined && typeof record.required !== "boolean") {
      throw new GoldValidationError(`relations[${index}].required must be true or false.`);
    }
  });
  const scope = gold.evaluation_scope as Record<string, unknown> | undefined;
  if (scope !== undefined) {
    if (!scope || typeof scope !== "object" || Array.isArray(scope)) {
      throw new GoldValidationError('"evaluation_scope" must be { entity_coverage, relation_coverage }.');
    }
    if (scope.entity_coverage !== undefined && !["exhaustive", "reference"].includes(String(scope.entity_coverage))) {
      throw new GoldValidationError('evaluation_scope.entity_coverage is "exhaustive" or "reference".');
    }
    if (
      scope.relation_coverage !== undefined &&
      !["exhaustive", "induced", "reference"].includes(String(scope.relation_coverage))
    ) {
      throw new GoldValidationError('evaluation_scope.relation_coverage is "exhaustive", "induced" or "reference".');
    }
  }
  if (gold.documents !== undefined && !Array.isArray(gold.documents)) {
    throw new GoldValidationError('"documents" must be a list of { id, path, title? }.');
  }
  if (Array.isArray(gold.documents)) {
    gold.documents.forEach((document, index) => {
      const record = document as Record<string, unknown>;
      for (const field of ["id", "path"] as const) {
        if (typeof record?.[field] !== "string" || !String(record[field]).trim()) {
          throw new GoldValidationError(`documents[${index}] needs a string "${field}".`);
        }
      }
    });
  }
  return gold as GoldGraph;
}

/** How many entities and relations a template takes from the graph. */
const TEMPLATE_ENTITIES = 20;
const TEMPLATE_RELATIONS = 10;

/**
 * A starting gold file drawn from the graph itself: its documents, its
 * best-connected entities and its surest relations, every relation marked
 * required so the person editing it decides what to keep. Writing the
 * format by hand from a blank page is the thing nobody should have to do.
 */
export function goldTemplate(dataset: GraphDataset, graphName: string): GoldGraph {
  const degree = new Map<string, number>();
  for (const edge of dataset.edges) {
    const ends = graphEdgeEnds(edge);
    for (const end of [ends.source, ends.target]) {
      const id = String(end ?? "");
      degree.set(id, (degree.get(id) ?? 0) + 1);
    }
  }
  const nodes = [...dataset.nodes]
    .sort((left, right) => (degree.get(graphNodeId(right)) ?? 0) - (degree.get(graphNodeId(left)) ?? 0))
    .slice(0, TEMPLATE_ENTITIES);
  const kept = new Set(nodes.map((node) => graphNodeId(node)));
  const edges = dataset.edges
    .map((edge, index) => ({ edge, index, ends: graphEdgeEnds(edge) }))
    .filter(({ ends }) => kept.has(String(ends.source)) && kept.has(String(ends.target)))
    .sort((left, right) => Number(right.edge.confidence ?? 0) - Number(left.edge.confidence ?? 0))
    .slice(0, TEMPLATE_RELATIONS);
  return {
    name: graphName,
    description: "Edit this file: keep the entities and relations the graph must contain, drop the rest, mark the relations that matter as required.",
    // A handful drawn from the graph, not everything in the corpus: what the
    // graph holds beyond them is not counted as wrong. Annotate the corpus in
    // full and make both "exhaustive" to be scored on precision and F1 too.
    evaluation_scope: { entity_coverage: "reference", relation_coverage: "reference" },
    documents: (dataset.documents ?? []).filter((document) => document.path || document.source_uri).map((document) => ({
      id: String(document.id ?? ""),
      path: String(document.path || document.source_uri),
      ...(document.title ? { title: String(document.title) } : {}),
    })),
    entities: nodes.map((node) => ({
      id: graphNodeId(node),
      name: graphNodeLabel(node),
      type: String(node.primary_type ?? node.type ?? "ENTITY"),
      ...(Array.isArray(node.aliases) && node.aliases.length ? { aliases: node.aliases.map(String) } : {}),
    })),
    relations: edges.map(({ edge, index, ends }) => ({
      id: graphEdgeId(edge, index),
      source: String(ends.source),
      target: String(ends.target),
      relation_type: String(edge.relation_type ?? "RELATED_TO"),
      required: true,
    })),
  };
}
