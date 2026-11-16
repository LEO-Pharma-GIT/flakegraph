import { isClaimed } from "@/lib/evidence";

export type ColorMode = "type" | "community" | "degree";
export type LayoutMode = "force" | "clustered";
export type FocusDepth = 1 | 2 | 3 | null;

export interface GraphCamera {
  x: number;
  y: number;
  scale: number;
}

export interface PathHighlight {
  nodeIds: Set<string>;
  edgeIds: Set<string>;
}

export interface GraphHover {
  kind: "node" | "edge";
  id: string;
  x: number;
  y: number;
}

export interface LayoutNode {
  id: string;
  label: string;
  type: string;
  description: string;
  communityId: string | null;
  degree: number;
  x: number;
  y: number;
  size: number;
}

export interface LayoutEdge {
  id: string;
  source: string;
  target: string;
  relationType: string;
  weight: number;
  description: string;
  confidence: number;
  /** Only claimed by its sources; drawn dashed. */
  claimed: boolean;
}

export const MIN_SCALE = 0.12;
export const MAX_SCALE = 6;
export const TYPE_COLORS = ["#16806A", "#D5654F", "#D39A25", "#3468A4", "#8056A6", "#6A7B2C", "#B34D78", "#397E8E"];
const DEGREE_STOPS = ["#94a3b8", "#64748b", "#0ea5e9", "#16806a", "#d39a25", "#d5654f"];
const COMMUNITY_COLORS = ["#16806A", "#3468A4", "#8056A6", "#D39A25", "#B34D78", "#397E8E", "#D5654F", "#6A7B2C", "#0f766e", "#7c3aed"];

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}

export function hash01(seed: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0) / 4294967296;
}

export function colorForType(type: string): string {
  let hash = 0;
  for (const character of type) {
    hash = (hash + character.charCodeAt(0) * 17) % TYPE_COLORS.length;
  }
  return TYPE_COLORS[hash] ?? TYPE_COLORS[0]!;
}

export function colorForCommunity(communityId: string | null): string {
  if (!communityId) {
    return "#94a3b8";
  }
  return COMMUNITY_COLORS[Math.floor(hash01(communityId) * COMMUNITY_COLORS.length)] ?? COMMUNITY_COLORS[0]!;
}

export function colorForDegree(rank01: number): string {
  const clamped = Math.min(0.999, Math.max(0, rank01));
  const index = Math.floor(clamped * DEGREE_STOPS.length);
  return DEGREE_STOPS[index] ?? DEGREE_STOPS[0]!;
}

export function nodeColor(node: LayoutNode, colorMode: ColorMode, maxDegree: number): string {
  if (colorMode === "community") {
    return colorForCommunity(node.communityId);
  }
  if (colorMode === "degree") {
    return colorForDegree(node.degree / Math.max(1, maxDegree));
  }
  return colorForType(node.type);
}

export function worldToScreen(camera: GraphCamera, x: number, y: number, width: number, height: number) {
  return {
    x: (x - camera.x) * camera.scale + width / 2,
    y: (y - camera.y) * camera.scale + height / 2,
  };
}

export function screenToWorld(camera: GraphCamera, x: number, y: number, width: number, height: number) {
  return {
    x: camera.x + (x - width / 2) / camera.scale,
    y: camera.y + (y - height / 2) / camera.scale,
  };
}

export function zoomAt(
  camera: GraphCamera,
  screenX: number,
  screenY: number,
  width: number,
  height: number,
  factor: number,
): GraphCamera {
  const world = screenToWorld(camera, screenX, screenY, width, height);
  const scale = clampScale(camera.scale * factor);
  return {
    scale,
    x: world.x - (screenX - width / 2) / scale,
    y: world.y - (screenY - height / 2) / scale,
  };
}

export function fitCamera(nodes: Array<{ x: number; y: number }>, width: number, height: number, padding = 64): GraphCamera {
  if (!nodes.length || width < 8 || height < 8) {
    return { x: 0, y: 0, scale: 1 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const node of nodes) {
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x);
    maxY = Math.max(maxY, node.y);
  }
  const boundsWidth = Math.max(48, maxX - minX);
  const boundsHeight = Math.max(48, maxY - minY);
  const scale = clampScale(Math.min((width - padding * 2) / boundsWidth, (height - padding * 2) / boundsHeight, 2.8));
  return {
    x: (minX + maxX) / 2,
    y: (minY + maxY) / 2,
    scale,
  };
}

export function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - (2 - 2 * t) ** 2 / 2;
}

export function graphNodeId(node: Record<string, unknown>, index = 0): string {
  return String(node.id ?? index);
}

export function graphNodeLabel(node: Record<string, unknown>): string {
  return String(node.name ?? node.id ?? "");
}

export function graphNodeType(node: Record<string, unknown>): string {
  return String(node.primary_type ?? node.type ?? "Unknown");
}

export function graphNodeDescription(node: Record<string, unknown>): string {
  return String(node.description ?? node.canonical_description ?? "");
}

export function graphEdgeEnds(edge: Record<string, unknown>): { source: string; target: string } {
  return {
    source: String(edge.source_node_id ?? edge.source ?? ""),
    target: String(edge.target_node_id ?? edge.target ?? ""),
  };
}

export function graphEdgeId(edge: Record<string, unknown>, index = 0): string {
  const ends = graphEdgeEnds(edge);
  return String(edge.id ?? `${ends.source}->${ends.target}:${index}`);
}

export function graphEdgeType(edge: Record<string, unknown>): string {
  return String(edge.relation_type ?? "RELATED");
}

export function asFinite(value: unknown, fallback = 0): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function communityIdForNode(
  node: Record<string, unknown>,
  membership: Map<string, Set<string>>,
): string | null {
  const explicit = node.community_id;
  if (explicit != null && String(explicit)) {
    return String(explicit);
  }
  const owned = membership.get(graphNodeId(node));
  if (!owned || owned.size === 0) {
    return null;
  }
  return [...owned][0] ?? null;
}

export function normalizeGraph(
  nodes: Record<string, unknown>[],
  edges: Record<string, unknown>[],
  membership: Map<string, Set<string>> = new Map(),
): { nodes: LayoutNode[]; edges: LayoutEdge[] } {
  const degree = new Map<string, number>();
  const layoutEdges: LayoutEdge[] = [];
  const seen = new Set<string>();
  for (const [index, edge] of edges.entries()) {
    const ends = graphEdgeEnds(edge);
    if (!ends.source || !ends.target) {
      continue;
    }
    const id = graphEdgeId(edge, index);
    if (seen.has(id)) {
      continue;
    }
    seen.add(id);
    degree.set(ends.source, (degree.get(ends.source) ?? 0) + 1);
    degree.set(ends.target, (degree.get(ends.target) ?? 0) + 1);
    layoutEdges.push({
      id,
      source: ends.source,
      target: ends.target,
      relationType: graphEdgeType(edge),
      weight: asFinite(edge.weight ?? edge.confidence, 1),
      description: String(edge.description ?? ""),
      confidence: asFinite(edge.confidence),
      claimed: isClaimed(edge),
    });
  }
  const layoutNodes = nodes.map((node, index) => {
    const id = graphNodeId(node, index);
    const nodeDegree = degree.get(id) ?? 0;
    return {
      id,
      label: graphNodeLabel(node),
      type: graphNodeType(node),
      description: graphNodeDescription(node),
      communityId: communityIdForNode(node, membership),
      degree: nodeDegree,
      x: 0,
      y: 0,
      size: Math.min(16, 5 + Math.log(1 + nodeDegree) * 2.4),
    };
  });
  return { nodes: layoutNodes, edges: layoutEdges };
}

export function layoutGraph(nodes: LayoutNode[], edges: LayoutEdge[], mode: LayoutMode): LayoutNode[] {
  const copies = nodes.map((node) => ({ ...node }));
  const count = copies.length;
  if (count === 0) {
    return copies;
  }
  const communities = [...new Set(copies.map((node) => node.communityId).filter((id): id is string => Boolean(id)))].sort();
  const ringRadius = Math.max(90, communities.length * 32);
  copies.forEach((node, index) => {
    const jitterA = hash01(node.id);
    const jitterB = hash01(`${node.id}:y`);
    if (mode === "clustered" && node.communityId && communities.length > 0) {
      const communityIndex = Math.max(0, communities.indexOf(node.communityId));
      const theta = (2 * Math.PI * communityIndex) / communities.length;
      const radius = 16 + jitterA * 28;
      const phi = jitterB * Math.PI * 2;
      node.x = Math.cos(theta) * ringRadius + Math.cos(phi) * radius;
      node.y = Math.sin(theta) * ringRadius + Math.sin(phi) * radius;
      return;
    }
    const theta = (index / count) * Math.PI * 2 + jitterA * 0.35;
    const radius = 56 + count * 1.7 + (index % 7) * 9;
    node.x = Math.cos(theta) * radius;
    node.y = Math.sin(theta) * radius;
  });

  const byId = new Map(copies.map((node) => [node.id, node]));
  const iterations = count > 280 ? 24 : count > 90 ? 40 : 64;
  const area = Math.max(1_200, count * 110);
  const k = Math.sqrt(area / count);
  // Fruchterman-Reingold with a cooling step cap: forces are accumulated per
  // node and each node moves at most `temperature` per iteration, so a long
  // edge on a big ring cannot overshoot and blow the layout up to NaN.
  const spread = Math.max(...copies.map((node) => Math.hypot(node.x, node.y)));
  const startTemperature = Math.max(k, spread / 4);
  const endTemperature = k / 4;
  const displacement = new Map(copies.map((node) => [node.id, { x: 0, y: 0 }]));
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (const vector of displacement.values()) {
      vector.x = 0;
      vector.y = 0;
    }
    for (let i = 0; i < count; i += 1) {
      const a = copies[i]!;
      const aDisp = displacement.get(a.id)!;
      for (let j = i + 1; j < count; j += 1) {
        const b = copies[j]!;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.hypot(dx, dy) || 0.05;
        const force = ((k * k) / dist) * 0.06;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        aDisp.x -= fx;
        aDisp.y -= fy;
        const bDisp = displacement.get(b.id)!;
        bDisp.x += fx;
        bDisp.y += fy;
      }
    }
    for (const edge of edges) {
      const source = byId.get(edge.source);
      const target = byId.get(edge.target);
      if (!source || !target || source === target) {
        continue;
      }
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.hypot(dx, dy) || 0.05;
      const force = ((dist * dist) / k) * 0.012;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      const sourceDisp = displacement.get(source.id)!;
      const targetDisp = displacement.get(target.id)!;
      sourceDisp.x += fx;
      sourceDisp.y += fy;
      targetDisp.x -= fx;
      targetDisp.y -= fy;
    }
    const temperature =
      startTemperature + ((endTemperature - startTemperature) * iteration) / Math.max(1, iterations - 1);
    for (const node of copies) {
      const vector = displacement.get(node.id)!;
      const length = Math.hypot(vector.x, vector.y);
      if (length > 0) {
        const step = Math.min(length, temperature) / length;
        node.x += vector.x * step;
        node.y += vector.y * step;
      }
      node.x -= node.x * 0.012;
      node.y -= node.y * 0.012;
    }
  }
  return copies;
}

export function nHopNodeIds(
  edges: Array<{ source: string; target: string }>,
  seeds: Iterable<string>,
  depth: number,
): Set<string> {
  const adjacency = new Map<string, string[]>();
  const add = (from: string, to: string) => {
    const list = adjacency.get(from) ?? [];
    list.push(to);
    adjacency.set(from, list);
  };
  for (const edge of edges) {
    add(edge.source, edge.target);
    add(edge.target, edge.source);
  }
  const visited = new Set([...seeds].filter(Boolean));
  let frontier = [...visited];
  for (let hop = 0; hop < depth; hop += 1) {
    const next: string[] = [];
    for (const id of frontier) {
      for (const neighbor of adjacency.get(id) ?? []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  return visited;
}

export function selectionSeeds(
  selectedId: string | null,
  edges: Array<{ id: string; source: string; target: string }>,
): string[] {
  if (!selectedId) {
    return [];
  }
  const selectedEdge = edges.find((edge) => edge.id === selectedId);
  if (selectedEdge) {
    return [selectedEdge.source, selectedEdge.target].filter(Boolean);
  }
  return [selectedId];
}

export function hopNeighborhood(
  nodes: Record<string, unknown>[],
  edges: Record<string, unknown>[],
  selectedId: string,
): { nodes: Record<string, unknown>[]; edges: Record<string, unknown>[] } {
  const layoutEdges = edges.map((edge, index) => ({
    id: graphEdgeId(edge, index),
    ...graphEdgeEnds(edge),
  }));
  const seeds = selectionSeeds(selectedId, layoutEdges);
  const ids = nHopNodeIds(layoutEdges, seeds, 1);
  return {
    nodes: nodes.filter((node) => ids.has(graphNodeId(node))),
    edges: edges.filter((edge) => {
      const ends = graphEdgeEnds(edge);
      return ids.has(ends.source) && ids.has(ends.target);
    }),
  };
}

export function findShortestPath(
  edges: Array<{ id: string; source: string; target: string }>,
  src: string,
  dst: string,
  options: { directed?: boolean } = {},
): PathHighlight | null {
  const directed = options.directed ?? false;
  if (!src || !dst) {
    return null;
  }
  if (src === dst) {
    return { nodeIds: new Set([src]), edgeIds: new Set() };
  }
  const adjacency = new Map<string, Array<{ neighbor: string; edgeId: string }>>();
  const add = (from: string, to: string, id: string) => {
    const list = adjacency.get(from) ?? [];
    list.push({ neighbor: to, edgeId: id });
    adjacency.set(from, list);
  };
  for (const edge of edges) {
    add(edge.source, edge.target, edge.id);
    if (!directed) {
      add(edge.target, edge.source, edge.id);
    }
  }
  const parent = new Map<string, { prev: string; edgeId: string }>();
  const queue = [src];
  const visited = new Set<string>([src]);
  while (queue.length > 0) {
    const current = queue.shift() as string;
    if (current === dst) {
      break;
    }
    for (const step of adjacency.get(current) ?? []) {
      if (visited.has(step.neighbor)) {
        continue;
      }
      visited.add(step.neighbor);
      parent.set(step.neighbor, { prev: current, edgeId: step.edgeId });
      queue.push(step.neighbor);
    }
  }
  if (!parent.has(dst)) {
    return null;
  }
  const nodeIds = new Set<string>([dst]);
  const edgeIds = new Set<string>();
  let cursor: string | undefined = dst;
  while (cursor && cursor !== src) {
    const step = parent.get(cursor);
    if (!step) {
      break;
    }
    edgeIds.add(step.edgeId);
    nodeIds.add(step.prev);
    cursor = step.prev;
  }
  return { nodeIds, edgeIds };
}

export function distanceToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax;
  const dy = by - ay;
  const length = dx * dx + dy * dy;
  if (length === 0) {
    return Math.hypot(px - ax, py - ay);
  }
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / length));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

export function xmlEscape(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export function buildGraphML(nodes: Record<string, unknown>[], edges: Record<string, unknown>[]): string {
  const nodeXml = nodes
    .map((node) => {
      const id = xmlEscape(graphNodeId(node));
      return `<node id="${id}">
  <data key="label">${xmlEscape(graphNodeLabel(node))}</data>
  <data key="type">${xmlEscape(graphNodeType(node))}</data>
  <data key="description">${xmlEscape(graphNodeDescription(node))}</data>
</node>`;
    })
    .join("\n");
  const edgeXml = edges
    .map((edge, index) => {
      const ends = graphEdgeEnds(edge);
      return `<edge id="${xmlEscape(graphEdgeId(edge, index))}" source="${xmlEscape(ends.source)}" target="${xmlEscape(ends.target)}">
  <data key="relationType">${xmlEscape(graphEdgeType(edge))}</data>
  <data key="weight">${asFinite(edge.weight ?? edge.confidence, 1)}</data>
  <data key="description">${xmlEscape(String(edge.description ?? ""))}</data>
</edge>`;
    })
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="label" for="node" attr.name="label" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <key id="description" for="node" attr.name="description" attr.type="string"/>
  <key id="relationType" for="edge" attr.name="relationType" attr.type="string"/>
  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
  <key id="description_edge" for="edge" attr.name="description" attr.type="string"/>
  <graph edgedefault="directed">
${nodeXml}
${edgeXml}
  </graph>
</graphml>`;
}
