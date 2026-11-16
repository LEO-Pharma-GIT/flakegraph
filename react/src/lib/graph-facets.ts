import type { GraphDataset } from "@/server/protocol/schema";
import { communityMemberIds } from "@/server/graph-filter";

/** One pickable value of a facet, with how much of the graph it covers. */
export interface FacetOption {
  id: string;
  label: string;
  count: number;
}

/** A word in both numbers, for sentences that count things. */
export interface Noun {
  one: string;
  many: string;
}

export const ENTITY_TYPE_NOUN: Noun = { one: "entity type", many: "entity types" };
export const RELATION_TYPE_NOUN: Noun = { one: "relation type", many: "relation types" };
export const NEIGHBORHOOD_NOUN: Noun = { one: "neighborhood", many: "neighborhoods" };
export const ENTITY_NOUN: Noun = { one: "entity", many: "entities" };
export const RELATION_NOUN: Noun = { one: "relation", many: "relations" };

export function pluralize(count: number, noun: Noun): string {
  return `${count.toLocaleString()} ${count === 1 ? noun.one : noun.many}`;
}

/** Largest first, so the values that shape the graph most are seen first; ties read alphabetically. */
export function rankFacetOptions(options: readonly FacetOption[]): FacetOption[] {
  return [...options].sort((left, right) => right.count - left.count || left.label.localeCompare(right.label));
}

export interface GraphFacetOptions {
  /** Counted by the entities that carry the type. */
  nodeTypes: FacetOption[];
  /** Counted by the relations that carry the type. */
  relationTypes: FacetOption[];
  /** The communities of the graph, counted by their members. */
  neighborhoods: FacetOption[];
}

/** Every filterable value of a dataset, ranked and labelled for the Filters card. */
export function graphFacetOptions(
  dataset: Pick<GraphDataset, "nodes" | "edges" | "communities">,
): GraphFacetOptions {
  const nodeTypes = tally(dataset.nodes.map((node) => String(node.primary_type ?? node.type ?? "Unknown")));
  const relationTypes = tally(dataset.edges.map((edge) => String(edge.relation_type ?? "RELATED")));
  const neighborhoods = (dataset.communities ?? [])
    .map((community) => ({
      id: String(community.id ?? ""),
      label: String(community.title ?? community.id ?? ""),
      count: communityMemberIds(community).length,
    }))
    .filter((item) => item.id);
  return {
    nodeTypes: rankFacetOptions(nodeTypes.map(([id, count]) => ({ id, label: prettyLabel(id), count }))),
    relationTypes: rankFacetOptions(relationTypes.map(([id, count]) => ({ id, label: prettyLabel(id), count }))),
    neighborhoods: rankFacetOptions(neighborhoods),
  };
}

export interface FilterState {
  nodeTypes: readonly string[];
  relationTypes: readonly string[];
  communityIds: readonly string[];
  minimumConfidence: number;
  includeIsolates: boolean;
  includeClaimed: boolean;
}

/**
 * What the collapsed Filters card says about its facets, or null when
 * nothing narrows the graph.
 *
 * Counts rather than names: three chosen relation types would not fit on a
 * summary line, and the count is what the reader needs to know whether to
 * open the card.
 */
export function filterSummary(state: FilterState): string | null {
  const parts: string[] = [];
  if (state.nodeTypes.length > 0) {
    parts.push(pluralize(state.nodeTypes.length, ENTITY_TYPE_NOUN));
  }
  if (state.relationTypes.length > 0) {
    parts.push(pluralize(state.relationTypes.length, RELATION_TYPE_NOUN));
  }
  if (state.communityIds.length > 0) {
    parts.push(pluralize(state.communityIds.length, NEIGHBORHOOD_NOUN));
  }
  if (state.minimumConfidence > 0) {
    parts.push(`confidence ≥ ${state.minimumConfidence.toFixed(2)}`);
  }
  if (!state.includeIsolates) {
    parts.push("connected only");
  }
  if (!state.includeClaimed) {
    parts.push("reported only");
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

/** `PRIMARY_TYPE` as `Primary Type`, for ontology names and column headers. */
export function prettyLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function tally(values: readonly string[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const value of values) {
    if (!value) {
      continue;
    }
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()];
}

/**
 * The questions a graph's community reports say they can answer, largest
 * neighborhood first, so the Ask tab can start from something the graph
 * actually holds rather than a placeholder.
 */
export function graphStarterQuestions(communities: readonly Record<string, unknown>[], limit = 6): string[] {
  const ranked = [...communities].sort((left, right) => communityMemberIds(right).length - communityMemberIds(left).length);
  const seen = new Set<string>();
  const questions: string[] = [];
  // One question per community first, so four starters come from four
  // neighborhoods rather than the largest one's whole list.
  for (let index = 0; questions.length < limit; index += 1) {
    let any = false;
    for (const community of ranked) {
      const raw = community.suggested_questions;
      const list = Array.isArray(raw) ? raw.map(String) : [];
      const candidate = list[index]?.trim();
      if (!candidate) {
        continue;
      }
      any = true;
      const key = candidate.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        questions.push(candidate);
        if (questions.length >= limit) {
          break;
        }
      }
    }
    if (!any) {
      break;
    }
  }
  return questions;
}
