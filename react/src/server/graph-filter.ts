import { isClaimed } from "@/lib/evidence";
import type { GraphDataset } from "./protocol/schema";

const MAX_CANVAS_NODES = 1_200;

export interface GraphFilters {
  search?: string;
  nodeTypes?: string[];
  relationTypes?: string[];
  communityIds?: string[];
  minimumConfidence?: number;
  includeIsolates?: boolean;
  /** Keep relations every source only claims; off leaves the reported ones. */
  includeClaimed?: boolean;
  /** How many entities the result may carry; the best-connected ones win. */
  limit?: number;
}

export interface FilteredGraph {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
  /** Entities that matched before the limit was applied. */
  totalNodes: number;
}

export interface GraphFacets {
  nodeTypes: string[];
  relationTypes: string[];
  communityIds: string[];
}

export function graphFacets(dataset: GraphDataset): GraphFacets {
  const nodeTypes = unique(dataset.nodes.map((node) => String(node.primary_type ?? node.type ?? "Unknown")));
  const relationTypes = unique(
    dataset.edges.map((edge) => String(edge.relation_type ?? "RELATED")),
  );
  const communityIds = unique(dataset.communities.map((community) => String(community.id ?? "")));
  return { nodeTypes, relationTypes, communityIds };
}

export function filterGraph(dataset: GraphDataset, filters: GraphFilters = {}): FilteredGraph {
  const search = (filters.search ?? "").trim().toLowerCase();
  const allowedTypes = new Set(filters.nodeTypes ?? []);
  const allowedRelations = new Set(filters.relationTypes ?? []);
  const allowedCommunities = new Set(filters.communityIds ?? []);
  const membership = communityMembership(dataset.communities);
  const minimumConfidence = filters.minimumConfidence ?? 0;
  const includeIsolates = filters.includeIsolates ?? true;
  const includeClaimed = filters.includeClaimed ?? true;
  const nodes = dataset.nodes.filter((raw) => {
    const nodeId = String(raw.id ?? "");
    const haystack = `${raw.name ?? ""} ${raw.description ?? ""} ${Array.isArray(raw.aliases) ? raw.aliases.join(" ") : ""}`.toLowerCase();
    if (search && !haystack.includes(search)) {
      return false;
    }
    if (allowedTypes.size > 0 && !allowedTypes.has(String(raw.primary_type ?? raw.type ?? "Unknown"))) {
      return false;
    }
    if (allowedCommunities.size > 0) {
      const owned = membership.get(nodeId) ?? new Set<string>();
      if (![...allowedCommunities].some((id) => owned.has(id))) {
        return false;
      }
    }
    return true;
  });
  const visible = new Set(nodes.map((node) => String(node.id ?? "")));
  const edges = dataset.edges.filter((edge) => {
    const source = String(edge.source_node_id ?? edge.source ?? "");
    const target = String(edge.target_node_id ?? edge.target ?? "");
    const relation = String(edge.relation_type ?? "");
    const confidence = asFloat(edge.confidence);
    return (
      visible.has(source) &&
      visible.has(target) &&
      (allowedRelations.size === 0 || allowedRelations.has(relation)) &&
      confidence >= minimumConfidence &&
      (includeClaimed || !isClaimed(edge))
    );
  });
  if (!includeIsolates) {
    const connected = new Set<string>();
    for (const edge of edges) {
      connected.add(String(edge.source_node_id ?? edge.source ?? ""));
      connected.add(String(edge.target_node_id ?? edge.target ?? ""));
    }
    return capGraph(
      nodes.filter((node) => connected.has(String(node.id ?? ""))),
      edges,
      filters.limit ?? MAX_CANVAS_NODES,
    );
  }
  return capGraph(nodes, edges, filters.limit ?? MAX_CANVAS_NODES);
}

/**
 * Keep the `limit` best-connected entities and the edges between them.
 *
 * Degree is counted over the edges passed in, so a filtered view ranks by the
 * connections that survived the filter. Ties keep the incoming order, which
 * makes the choice stable across renders.
 */
export function capGraph(
  nodes: Record<string, unknown>[],
  edges: Record<string, unknown>[],
  limit: number,
): FilteredGraph {
  const totalNodes = nodes.length;
  if (nodes.length <= limit) {
    return { nodes, edges, totalNodes };
  }
  const degree = new Map<string, number>();
  for (const edge of edges) {
    for (const end of [edge.source_node_id ?? edge.source, edge.target_node_id ?? edge.target]) {
      const id = String(end ?? "");
      degree.set(id, (degree.get(id) ?? 0) + 1);
    }
  }
  const ranked = nodes
    .map((node, index) => ({ node, index, degree: degree.get(String(node.id ?? "")) ?? 0 }))
    .sort((left, right) => right.degree - left.degree || left.index - right.index)
    .slice(0, limit)
    .sort((left, right) => left.index - right.index)
    .map((entry) => entry.node);
  const ids = new Set(ranked.map((node) => String(node.id ?? "")));
  return {
    nodes: ranked,
    edges: edges.filter(
      (edge) =>
        ids.has(String(edge.source_node_id ?? edge.source ?? "")) &&
        ids.has(String(edge.target_node_id ?? edge.target ?? "")),
    ),
    totalNodes,
  };
}

export function communityMembership(communities: readonly Record<string, unknown>[]): Map<string, Set<string>> {
  const membership = new Map<string, Set<string>>();
  for (const community of communities) {
    const id = String(community.id ?? "");
    const members = communityMemberIds(community);
    for (const member of members) {
      const set = membership.get(member) ?? new Set<string>();
      set.add(id);
      membership.set(member, set);
    }
  }
  return membership;
}

/**
 * The entities a community row names, whichever of the pipeline's parquet
 * column (`member_node_ids`), the gold format's `members`, or an export's
 * `member_ids`/`nodes` carries them.
 */
export function communityMemberIds(community: Record<string, unknown>): string[] {
  return variantSequence(
    community.member_node_ids ?? community.members ?? community.member_ids ?? community.nodes,
  );
}

export function variantSequence(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.map((item) => String(item));
      }
    } catch {
      return value.split(",").map((item) => item.trim()).filter(Boolean);
    }
  }
  return [];
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function asFloat(value: unknown): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : 0;
}

export const CANVAS_NODE_LIMIT = MAX_CANVAS_NODES;
