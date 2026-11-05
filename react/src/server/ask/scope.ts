import { evidenceEntityId, evidenceRelationId } from "@/lib/evidence";
import { communityMemberIds, filterGraph } from "../graph-filter";
import type { GraphDataset } from "../protocol/schema";
import { documentIdOf, edgeEnds, nodeName } from "./retrieve";
import type { AskScope } from "./types";

export function scopeDataset(dataset: GraphDataset, scope?: AskScope): GraphDataset {
  if (!isScoped(scope)) {
    return dataset;
  }
  let next = dataset;
  if (scope?.documentIds !== undefined) {
    next = restrictToDocuments(next, scope.documentIds);
  }
  if (!scope?.search && !(scope?.communityIds && scope.communityIds.length > 0)) {
    return next;
  }
  const filtered = filterGraph(next, {
    search: scope.search,
    communityIds: scope.communityIds,
    includeIsolates: true,
    limit: Number.POSITIVE_INFINITY,
  });
  const nodeIds = new Set(filtered.nodes.map((node) => String(node.id ?? "")));
  const edgeIds = new Set(filtered.edges.map((edge) => String(edge.id ?? "")));
  const nodes = next.nodes.filter((node) => nodeIds.has(String(node.id ?? "")));
  const edges = next.edges.filter((edge) => edgeIds.has(String(edge.id ?? "")));
  const chunkIds = new Set<string>();
  for (const row of [...nodes, ...edges]) {
    const ids = row.source_chunk_ids;
    if (Array.isArray(ids)) ids.forEach((id) => chunkIds.add(String(id)));
  }
  const evidence = next.evidence.filter((row) => {
    const relationId = evidenceRelationId(row);
    const entityId = evidenceEntityId(row);
    return relationId ? edgeIds.has(relationId) : entityId ? nodeIds.has(entityId) : chunkIds.has(String(row.chunk_id ?? ""));
  });
  for (const row of evidence) if (row.chunk_id) chunkIds.add(String(row.chunk_id));
  const chunks = next.chunks.filter((row) => {
    const entityId = evidenceEntityId(row);
    return chunkIds.has(String(row.id ?? row.chunk_id ?? "")) || Boolean(entityId && nodeIds.has(entityId));
  });
  const documentIds = new Set([...evidence, ...chunks].map(documentIdOf));
  return {
    ...next,
    nodes,
    edges,
    evidence,
    chunks,
    documents: next.documents.filter((row) => [row.id, row.document_id, row.file_id].some((id) => id != null && documentIds.has(String(id)))),
    communities: scope.communityIds?.length
      ? next.communities.filter((community) => scope.communityIds!.includes(String(community.id ?? "")))
      : next.communities.filter((community) => {
          const members = communityMemberIds(community);
          return members.some((id) => nodeIds.has(id));
        }),
  };
}

export function isScoped(scope?: AskScope): boolean {
  return Boolean(
    scope?.search ||
      (scope?.communityIds && scope.communityIds.length > 0) ||
      scope?.documentIds !== undefined,
  );
}

export function mergeAskScope(base?: AskScope, extra?: AskScope): AskScope | undefined {
  if (!base && !extra) {
    return undefined;
  }
  const documentIds = intersectIds(extra?.documentIds, base?.documentIds);
  const communityIds = extra?.communityIds ?? base?.communityIds;
  const search = extra?.search ?? base?.search;
  if (!search && !(communityIds && communityIds.length) && documentIds === undefined) {
    return undefined;
  }
  return { search, communityIds, documentIds };
}

export function intersectIds(requested?: string[], existing?: string[]): string[] | undefined {
  if (requested !== undefined && existing !== undefined) {
    const allowed = new Set(existing);
    return requested.filter((id) => allowed.has(id));
  }
  if (requested !== undefined) return requested;
  if (existing !== undefined) return existing;
  return undefined;
}

/** Rebuild document context from source rows, never from merged corpus prose. */
function restrictToDocuments(dataset: GraphDataset, documentIds: string[]): GraphDataset {
  const requested = new Set(documentIds);
  const documents = dataset.documents.filter((document) =>
    [document.id, document.document_id, document.file_id].some((id) => requested.has(String(id ?? ""))),
  );
  const allowed = new Set(documentIds);
  for (const document of documents) {
    for (const id of [document.id, document.document_id, document.file_id]) {
      if (id != null) allowed.add(String(id));
    }
  }
  const evidence = dataset.evidence.filter((row) => allowed.has(documentIdOf(row))).map(sourceRow);
  const chunks = dataset.chunks.filter((row) => allowed.has(documentIdOf(row))).map(sourceRow);
  const chunkIds = new Set(chunks.map((row) => String(row.id ?? row.chunk_id ?? "")));
  for (const row of evidence) {
    if (row.chunk_id) chunkIds.add(String(row.chunk_id));
  }
  const relationIds = new Set<string>();
  const entityIds = new Set<string>();
  for (const row of [...evidence, ...chunks]) {
    const relationId = evidenceRelationId(row);
    const entityId = evidenceEntityId(row);
    if (relationId) relationIds.add(relationId);
    if (entityId) entityIds.add(entityId);
  }
  const hasDocumentSupport = (row: Record<string, unknown>) =>
    sourceChunkIds(row).some((id) => chunkIds.has(id)) ||
    sourceFileIds(row).some((id) => allowed.has(id));
  const linkedEdges = dataset.edges.filter((edge) =>
    relationIds.has(String(edge.id ?? "")) || hasDocumentSupport(edge),
  );
  for (const edge of linkedEdges) {
    const ends = edgeEnds(edge);
    entityIds.add(ends.source);
    entityIds.add(ends.target);
  }
  for (const node of dataset.nodes) {
    if (hasDocumentSupport(node)) entityIds.add(String(node.id ?? ""));
  }
  if (entityIds.size === 0 && (evidence.length > 0 || chunks.length > 0)) {
    const haystack = [...evidence, ...chunks].map(sourceText).join(" ").toLowerCase();
    for (const node of dataset.nodes) {
      const name = nodeName(node).toLowerCase();
      if (name && haystack.includes(name)) entityIds.add(String(node.id ?? ""));
    }
  }
  const evidenceByEntity = groupSources(evidence, evidenceEntityId);
  const evidenceByRelation = groupSources(evidence, evidenceRelationId);
  const chunksById = new Map(chunks.map((row) => [String(row.id ?? row.chunk_id ?? ""), row]));
  const subjectEvidence = (row: Record<string, unknown>, kind: "node" | "edge") =>
    (kind === "node" ? evidenceByEntity : evidenceByRelation).get(String(row.id ?? "")) ?? [];
  const scopedDescription = (row: Record<string, unknown>, kind: "node" | "edge") => {
    const direct = subjectEvidence(row, kind);
    const supporting = new Set(sourceChunkIds(row).filter((chunkId) => chunkIds.has(chunkId)));
    const sources = direct.length > 0 ? direct : [...supporting].flatMap((id) => {
      const row = chunksById.get(id);
      return row ? [row] : [];
    });
    return [...new Set(sources.map(sourceText).filter(Boolean))].join("\n");
  };
  const scopedChunks = (row: Record<string, unknown>, kind: "node" | "edge") => {
    const ids = new Set(sourceChunkIds(row).filter((id) => chunkIds.has(id)));
    for (const item of subjectEvidence(row, kind)) {
      if (item.chunk_id) ids.add(String(item.chunk_id));
    }
    return [...ids];
  };
  // Canonical identity is useful across documents, but aliases, descriptions,
  // quantities and arbitrary metadata can contain assertions from other files.
  const nodes = dataset.nodes.filter((node) => entityIds.has(String(node.id ?? ""))).map((node) => ({
    id: node.id,
    name: nodeName(node),
    primary_type: node.primary_type ?? node.type ?? "Unknown",
    description: scopedDescription(node, "node"),
    source_chunk_ids: scopedChunks(node, "node"),
  }));
  const nodeIds = new Set(nodes.map((node) => String(node.id ?? "")));
  const edges = linkedEdges.filter((edge) => {
    const ends = edgeEnds(edge);
    return nodeIds.has(ends.source) && nodeIds.has(ends.target);
  }).map((edge) => ({
    id: edge.id,
    source_node_id: edgeEnds(edge).source,
    target_node_id: edgeEnds(edge).target,
    relation_type: edge.relation_type ?? edge.type ?? "RELATED_TO",
    description: scopedDescription(edge, "edge"),
    source_chunk_ids: scopedChunks(edge, "edge"),
    // Numeric ranking signals carry no text from other documents.
    ...pick(edge, ["weight", "confidence"]),
    // Whether the sources only claim the relation, which an answer must say.
    ...pick(edge, ["assertion"]),
  }));
  const names = new Map(nodes.map((node) => [String(node.id), node.name]));
  const communities = dataset.communities.flatMap((community) => {
    const members = communityMemberIds(community).filter((id) => nodeIds.has(id));
    if (members.length === 0) return [];
    const memberIds = new Set(members);
    const descriptions = [
      ...nodes.filter((node) => memberIds.has(String(node.id))),
      ...edges.filter((edge) => memberIds.has(edge.source_node_id) && memberIds.has(edge.target_node_id)),
    ].map((row) => row.description).filter(Boolean);
    if (descriptions.length === 0) return [];
    return [{
      id: community.id,
      level: community.level ?? 0,
      member_node_ids: members,
      title: members.map((id) => names.get(id) ?? id).join(", "),
      summary: [...new Set(descriptions)].join("\n"),
      findings: [],
    }];
  });
  return {
    graphId: dataset.graphId,
    documents: documents.map((document) => pick(document, [
      "id", "document_id", "file_id", "filename", "name", "title", "source_uri", "path", "uri", "summary",
    ])),
    evidence, chunks, nodes, edges, communities,
    runReport: {}, graphMetrics: {}, discardedWindows: [], rejectedRecords: [], failedDocuments: [],
  };
}

function pick(row: Record<string, unknown>, fields: string[]): Record<string, unknown> {
  return Object.fromEntries(fields.filter((field) => row[field] != null).map((field) => [field, row[field]]));
}

/** Keep the selected document's raw text and provenance, without derived metadata. */
function sourceRow(row: Record<string, unknown>): Record<string, unknown> {
  return pick(row, [
    "id", "chunk_id", "document_id", "file_id", "documentId", "entity_id", "node_id",
    "relation_id", "edge_id", "subject_id", "subject_kind", "quote", "sentence", "content", "text",
    "page_number", "start_offset", "end_offset", "chunk_index", "method",
  ]);
}

function sourceText(row: Record<string, unknown>): string {
  return String(row.quote ?? row.sentence ?? row.content ?? row.text ?? "");
}

function sourceChunkIds(row: Record<string, unknown>): string[] {
  const ids = row.source_chunk_ids;
  return Array.isArray(ids) ? ids.map(String) : [];
}

function sourceFileIds(row: Record<string, unknown>): string[] {
  const ids = row.source_file_ids;
  const single = row.source_file_id ?? row.file_id ?? row.document_id;
  return [...(Array.isArray(ids) ? ids.map(String) : []), ...(single != null ? [String(single)] : [])];
}

function groupSources(
  rows: Record<string, unknown>[],
  subjectId: (row: Record<string, unknown>) => string | null,
): Map<string, Record<string, unknown>[]> {
  const groups = new Map<string, Record<string, unknown>[]>();
  for (const row of rows) {
    const id = subjectId(row);
    if (!id) continue;
    const sources = groups.get(id) ?? [];
    sources.push(row);
    groups.set(id, sources);
  }
  return groups;
}
