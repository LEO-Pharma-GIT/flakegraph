import { nHopNodeIds } from "@/lib/graph-geometry";
import { documentDisplayName, evidenceDocumentId, evidenceEntityId, evidenceRelationId, isClaimed } from "@/lib/evidence";
import { communityMemberIds } from "../graph-filter";
import type { GraphDataset } from "../protocol/schema";
import {
  ACCESSIBLE_RELATIONS_LIMIT,
  COMMUNITY_BLEND_DISTANCE_WEIGHT,
  COMMUNITY_BLEND_RATING_WEIGHT,
  COMMUNITY_RATING_MAX,
  EDGE_EVIDENCE_CHAR_LIMIT,
  EDGE_EVIDENCE_CHUNK_MAX,
  ENTITY_SOURCE_CHUNK_CHAR_LIMIT,
  ENTITY_SOURCE_CHUNK_MAX,
  GLOBAL_CONTRIBUTING_ENTITY_OVERSAMPLE_FACTOR,
  GLOBAL_ENTITY_BLEND_COMMUNITY_WEIGHT,
  GLOBAL_ENTITY_BLEND_ENTITY_WEIGHT,
  GRAPH_LOCAL_ENTITY_EVIDENCE_MULTIPLIER,
  GRAPH_LOCAL_ENTITY_TOP_K_MIN,
  GRAPH_LOCAL_ENTITY_TOP_K_RATIO,
  GRAPH_LOCAL_RELATION_TOP_K_FLOOR,
  LOCAL_BACKDROP_COMMUNITY_CAP,
  RELATION_BLEND_DISTANCE_WEIGHT,
  RELATION_BLEND_TOPOLOGY_WEIGHT,
  RRF_K,
} from "./constants";
import { asNumber, asString, asStringArray, distanceFromScore, scoreHaystack, tokenize } from "./score";
import type { CommunityHit, EntityHit, EvidenceHit, GraphDocument, RelationHit } from "./types";

export function fuseRrf<T extends { id: string }>(left: T[], right: T[], topK: number): T[] {
  const scores = new Map<string, number>();
  const items = new Map<string, T>();
  const add = (list: T[]) => {
    list.forEach((item, rank) => {
      scores.set(item.id, (scores.get(item.id) ?? 0) + 1 / (RRF_K + rank + 1));
      if (!items.has(item.id)) {
        items.set(item.id, item);
      }
    });
  };
  add(left);
  add(right);
  return [...items.values()]
    .sort((a, b) => (scores.get(b.id) ?? 0) - (scores.get(a.id) ?? 0))
    .slice(0, topK);
}

export function nodeName(node: Record<string, unknown>): string {
  return asString(node.name ?? node.title ?? node.id);
}

export function nodeType(node: Record<string, unknown>): string {
  return asString(node.primary_type ?? node.type ?? "Unknown");
}

export function edgeEnds(edge: Record<string, unknown>): { source: string; target: string } {
  return {
    source: asString(edge.source_node_id ?? edge.source),
    target: asString(edge.target_node_id ?? edge.target),
  };
}

export function searchEntities(
  dataset: GraphDataset,
  query: string,
  topK = 12,
): EntityHit[] {
  const terms = tokenize(query);
  if (terms.length === 0) {
    return [];
  }
  return dataset.nodes
    .map((node) => {
      const name = nodeName(node);
      const description = asString(node.description);
      const aliases = asStringArray(node.aliases).join(" ");
      const score =
        scoreHaystack(`${name} ${aliases}`, terms) * 3 + scoreHaystack(description, terms);
      return {
        id: asString(node.id),
        name,
        type: nodeType(node),
        description,
        score,
        distance: distanceFromScore(score),
      };
    })
    .filter((hit) => hit.id && hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}

export function searchCommunities(
  dataset: GraphDataset,
  query: string,
  topK = 8,
  options?: { level?: number; scanAll?: boolean },
): CommunityHit[] {
  const terms = tokenize(query);
  const scanAll = options?.scanAll ?? false;
  const level = options?.level ?? 0;
  if (!scanAll && terms.length === 0) {
    return [];
  }
  const nameById = entityNameIndex(dataset);
  const hits = dataset.communities
    .map((community) => toCommunityHit(community, nameById, terms))
    .filter((hit) => hit.id && hit.level === level)
    .filter((hit) => scanAll || hit.score > 0)
    .sort((left, right) => {
      const blend = blendedCommunityScore(left) - blendedCommunityScore(right);
      return blend !== 0 ? blend : right.score - left.score;
    });
  return scanAll ? hits : hits.slice(0, topK);
}

export function blendedCommunityScore(hit: { distance: number; rating: number | null }): number {
  const ratingPenalty =
    typeof hit.rating === "number"
      ? 1 - Math.max(0, Math.min(COMMUNITY_RATING_MAX, hit.rating)) / COMMUNITY_RATING_MAX
      : 0.5;
  return COMMUNITY_BLEND_DISTANCE_WEIGHT * hit.distance + COMMUNITY_BLEND_RATING_WEIGHT * ratingPenalty;
}

/** Marks a hit whose edge every source only claims, leaving a reported one as it was. */
function claimedMark(edge: Record<string, unknown>): Pick<RelationHit, "claimed"> {
  return isClaimed(edge) ? { claimed: true } : {};
}

export function searchRelations(
  dataset: GraphDataset,
  query: string,
  topK = 20,
): RelationHit[] {
  const terms = tokenize(query);
  if (terms.length === 0) {
    return [];
  }
  const nameById = entityNameIndex(dataset);
  return dataset.edges
    .map((edge) => {
      const ends = edgeEnds(edge);
      const sourceName = nameById.get(ends.source) ?? ends.source;
      const targetName = nameById.get(ends.target) ?? ends.target;
      const relationType = asString(edge.relation_type ?? "RELATED");
      const description = asString(edge.description);
      const score =
        scoreHaystack(`${sourceName} ${targetName} ${relationType} ${description}`, terms);
      const weight = asNumber(edge.weight ?? edge.confidence, 1);
      return {
        id: asString(edge.id),
        sourceId: ends.source,
        targetId: ends.target,
        sourceName,
        targetName,
        relationType,
        description,
        weight,
        score,
        distance: distanceFromScore(score),
        ...claimedMark(edge),
      };
    })
    .filter((hit) => hit.id && hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}

export function searchEvidence(
  dataset: GraphDataset,
  query: string,
  topK = 12,
): EvidenceHit[] {
  const terms = tokenize(query);
  if (terms.length === 0) {
    return [];
  }
  const nameById = entityNameIndex(dataset);
  const fromEvidence = dataset.evidence.map((row, index) => {
    const quote = asString(row.quote ?? row.sentence ?? row.content);
    const entityId = evidenceEntityId(row);
    return {
      id: asString(row.id) || `evidence_${index}`,
      quote,
      documentId: evidenceDocumentId(row),
      entityId,
      entityName: entityId ? nameById.get(entityId) ?? null : null,
      relationId: evidenceRelationId(row),
      score: scoreHaystack(quote, terms),
      distance: 0,
    };
  });
  const fromChunks = dataset.chunks.map((row, index) => {
    const quote = asString(row.content ?? row.text ?? row.quote);
    const entityId = evidenceEntityId(row);
    return {
      id: asString(row.id) || `chunk_${index}`,
      quote,
      documentId: evidenceDocumentId(row),
      entityId,
      entityName: entityId ? nameById.get(entityId) ?? null : null,
      relationId: evidenceRelationId(row),
      score: scoreHaystack(quote, terms),
      distance: 0,
    };
  });
  return [...fromEvidence, ...fromChunks]
    .map((hit) => ({ ...hit, distance: distanceFromScore(hit.score) }))
    .filter((hit) => hit.quote && hit.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, topK);
}

export function incidentRelations(dataset: GraphDataset, entityIds: string[]): RelationHit[] {
  const allowed = new Set(entityIds);
  const nameById = entityNameIndex(dataset);
  return dataset.edges
    .map((edge) => {
      const ends = edgeEnds(edge);
      if (!allowed.has(ends.source) && !allowed.has(ends.target)) {
        return null;
      }
      const weight = asNumber(edge.weight ?? edge.confidence, 1);
      return {
        id: asString(edge.id),
        sourceId: ends.source,
        targetId: ends.target,
        sourceName: nameById.get(ends.source) ?? ends.source,
        targetName: nameById.get(ends.target) ?? ends.target,
        relationType: asString(edge.relation_type ?? "RELATED"),
        description: asString(edge.description),
        weight,
        score: weight,
        distance: distanceFromScore(weight),
        ...claimedMark(edge),
      } satisfies RelationHit;
    })
    .filter((hit): hit is RelationHit => Boolean(hit?.id))
    .sort((left, right) => right.weight - left.weight)
    .slice(0, ACCESSIBLE_RELATIONS_LIMIT);
}

export function blendRelations(semantic: RelationHit[], topology: RelationHit[]): RelationHit[] {
  const semanticDistance = new Map(semantic.map((hit) => [hit.id, hit.distance]));
  const byId = new Map<string, RelationHit>();
  for (const hit of topology) {
    byId.set(hit.id, hit);
  }
  for (const hit of semantic) {
    if (!byId.has(hit.id)) {
      byId.set(hit.id, hit);
    }
  }
  return [...byId.values()]
    .map((hit) => {
      const distance = semanticDistance.get(hit.id) ?? 1;
      const blended =
        RELATION_BLEND_DISTANCE_WEIGHT * (1 - Math.min(1, distance)) +
        RELATION_BLEND_TOPOLOGY_WEIGHT * Math.min(1, hit.weight);
      return { ...hit, score: blended, distance: 1 - blended };
    })
    .sort((left, right) => right.score - left.score);
}

export function communitiesForEntities(
  dataset: GraphDataset,
  entityIds: string[],
  maxCommunities = LOCAL_BACKDROP_COMMUNITY_CAP,
  level = 0,
): CommunityHit[] {
  const wanted = new Set(entityIds);
  const nameById = entityNameIndex(dataset);
  const hits: CommunityHit[] = [];
  for (const community of dataset.communities) {
    const hit = toCommunityHit(community, nameById, []);
    if (!hit.id || hit.level !== level) {
      continue;
    }
    const overlap = hit.memberIds.filter((id) => wanted.has(id)).length;
    if (overlap === 0) {
      continue;
    }
    hits.push({ ...hit, score: overlap, distance: 0 });
  }
  return hits.sort((left, right) => right.score - left.score).slice(0, maxCommunities);
}

export function evidenceForEntities(
  dataset: GraphDataset,
  entityIds: string[],
  query: string,
  topK: number,
): EvidenceHit[] {
  const allowed = new Set(entityIds);
  const nameById = entityNameIndex(dataset);
  const terms = tokenize(query);
  const names = [...allowed].map((id) => (nameById.get(id) ?? "").toLowerCase()).filter(Boolean);
  return dataset.evidence
    .map((row, index) => {
      const quote = asString(row.quote ?? row.sentence ?? row.content);
      const entityId = evidenceEntityId(row);
      const relationId = evidenceRelationId(row);
      const haystack = quote.toLowerCase();
      const mentions = names.some((name) => name && haystack.includes(name));
      const relatedEntity = entityId ? allowed.has(entityId) : false;
      if (!quote || (!mentions && !relatedEntity && scoreHaystack(quote, terms) === 0)) {
        return null;
      }
      const score =
        (relatedEntity ? 2 : 0) + (mentions ? 2 : 0) + scoreHaystack(quote, terms);
      return {
        id: asString(row.id) || `evidence_${index}`,
        quote,
        documentId: asString(row.document_id ?? row.file_id),
        entityId,
        entityName: entityId ? nameById.get(entityId) ?? null : null,
        relationId,
        score,
        distance: distanceFromScore(score),
      } satisfies EvidenceHit;
    })
    .filter((hit): hit is EvidenceHit => Boolean(hit))
    .sort((left, right) => right.score - left.score)
    .slice(0, topK * GRAPH_LOCAL_ENTITY_EVIDENCE_MULTIPLIER);
}

export function contributingEntities(
  dataset: GraphDataset,
  communities: CommunityHit[],
  query: string,
  topK: number,
): EntityHit[] {
  const memberIds = new Set(communities.flatMap((community) => community.memberIds));
  const communityDistance = new Map<string, number>();
  for (const community of communities) {
    for (const memberId of community.memberIds) {
      const prior = communityDistance.get(memberId);
      if (prior === undefined || community.distance < prior) {
        communityDistance.set(memberId, community.distance);
      }
    }
  }
  const lexical = new Map(
    searchEntities(dataset, query, topK * GLOBAL_CONTRIBUTING_ENTITY_OVERSAMPLE_FACTOR).map((hit) => [hit.id, hit]),
  );
  return dataset.nodes
    .filter((node) => memberIds.has(asString(node.id)))
    .map((node) => {
      const id = asString(node.id);
      const lexicalHit = lexical.get(id);
      const entityDistance = lexicalHit?.distance ?? 1;
      const inherited = communityDistance.get(id) ?? 1;
      const distance =
        GLOBAL_ENTITY_BLEND_ENTITY_WEIGHT * entityDistance + GLOBAL_ENTITY_BLEND_COMMUNITY_WEIGHT * inherited;
      return {
        id,
        name: nodeName(node),
        type: nodeType(node),
        description: asString(node.description),
        score: lexicalHit?.score ?? 1 / Math.max(0.05, distance),
        distance,
      } satisfies EntityHit;
    })
    .sort((left, right) => left.distance - right.distance)
    .slice(0, topK);
}

export function maybePromoteToSuperCommunity(dataset: GraphDataset, hits: CommunityHit[]): CommunityHit[] {
  if (hits.length < 2) {
    return hits;
  }
  if (hits.some((hit) => hit.level !== 0)) {
    return hits;
  }
  const parentIds = new Set(hits.map((hit) => hit.parentCommunityId));
  if (parentIds.size !== 1) {
    return hits;
  }
  const [parentId] = [...parentIds];
  if (!parentId) {
    return hits;
  }
  const parent = dataset.communities.find((community) => asString(community.id) === parentId);
  if (!parent) {
    return hits;
  }
  const bestDistance = hits.reduce((min, hit) => Math.min(min, hit.distance), Number.POSITIVE_INFINITY);
  return [
    {
      ...toCommunityHit(parent, entityNameIndex(dataset), []),
      score: hits.reduce((sum, hit) => sum + hit.score, 0),
      distance: bestDistance,
    },
  ];
}

export function neighborhood(
  dataset: GraphDataset,
  seedId: string,
  hops = 1,
): { entities: EntityHit[]; relations: RelationHit[] } {
  const layoutEdges = dataset.edges.map((edge) => {
    const ends = edgeEnds(edge);
    return { source: ends.source, target: ends.target };
  });
  const ids = nHopNodeIds(layoutEdges, [seedId], Math.max(1, Math.min(3, hops)));
  const nameById = entityNameIndex(dataset);
  const entities = dataset.nodes
    .filter((node) => ids.has(asString(node.id)))
    .map((node) => ({
      id: asString(node.id),
      name: nodeName(node),
      type: nodeType(node),
      description: asString(node.description),
      score: asString(node.id) === seedId ? 10 : 1,
      distance: asString(node.id) === seedId ? 0 : 0.5,
    }));
  const relations = dataset.edges
    .map((edge) => {
      const ends = edgeEnds(edge);
      if (!ids.has(ends.source) || !ids.has(ends.target)) {
        return null;
      }
      return {
        id: asString(edge.id),
        sourceId: ends.source,
        targetId: ends.target,
        sourceName: nameById.get(ends.source) ?? ends.source,
        targetName: nameById.get(ends.target) ?? ends.target,
        relationType: asString(edge.relation_type ?? "RELATED"),
        description: asString(edge.description),
        weight: asNumber(edge.weight ?? edge.confidence, 1),
        score: 1,
        distance: 0.5,
        ...claimedMark(edge),
      } satisfies RelationHit;
    })
    .filter((hit): hit is RelationHit => Boolean(hit?.id));
  return { entities, relations };
}

export function localTopK(topK: number): { entityTopK: number; relationTopK: number } {
  return {
    entityTopK: Math.max(GRAPH_LOCAL_ENTITY_TOP_K_MIN, Math.ceil(topK * GRAPH_LOCAL_ENTITY_TOP_K_RATIO)),
    relationTopK: Math.max(GRAPH_LOCAL_RELATION_TOP_K_FLOOR, topK),
  };
}

export function entityNameIndex(dataset: GraphDataset): Map<string, string> {
  return new Map(dataset.nodes.map((node) => [asString(node.id), nodeName(node)]));
}

export function listGraphDocuments(dataset: GraphDataset): GraphDocument[] {
  const quoteCount = new Map<string, number>();
  const bump = (id: string) => {
    if (!id) {
      return;
    }
    quoteCount.set(id, (quoteCount.get(id) ?? 0) + 1);
  };
  for (const row of dataset.evidence) {
    bump(documentIdOf(row));
  }
  for (const row of dataset.chunks) {
    bump(documentIdOf(row));
  }
  const byId = new Map<string, GraphDocument>();
  for (const document of dataset.documents) {
    const id = asString(document.id);
    if (!id) {
      continue;
    }
    byId.set(id, {
      id,
      title: documentDisplayName(document),
      path: asString(document.path ?? document.source_uri),
      quoteCount: quoteCount.get(id) ?? 0,
    });
  }
  for (const [id, count] of quoteCount) {
    if (!byId.has(id)) {
      byId.set(id, { id, title: id, path: "", quoteCount: count });
    }
  }
  return [...byId.values()].sort((left, right) => left.title.localeCompare(right.title));
}

export function evidenceForDocument(
  dataset: GraphDataset,
  documentId: string,
  startIndex = 0,
  count = 8,
): EvidenceHit[] {
  return allQuotes(dataset)
    .filter((hit) => hit.documentId === documentId && hit.quote)
    .slice(Math.max(0, startIndex), Math.max(0, startIndex) + Math.max(1, count));
}

export function evidenceForEntity(dataset: GraphDataset, entityId: string, limit = ENTITY_SOURCE_CHUNK_MAX): EvidenceHit[] {
  const node = dataset.nodes.find((item) => asString(item.id) === entityId);
  if (!node) {
    return [];
  }
  const name = nodeName(node).toLowerCase();
  const relatedRelations = new Set(
    dataset.edges
      .filter((edge) => {
        const ends = edgeEnds(edge);
        return ends.source === entityId || ends.target === entityId;
      })
      .map((edge) => asString(edge.id)),
  );
  return allQuotes(dataset)
    .filter((hit) => {
      if (hit.entityId === entityId) {
        return true;
      }
      if (hit.relationId && relatedRelations.has(hit.relationId)) {
        return true;
      }
      return Boolean(name) && hit.quote.toLowerCase().includes(name);
    })
    .map((hit) => ({ ...hit, quote: hit.quote.slice(0, ENTITY_SOURCE_CHUNK_CHAR_LIMIT) }))
    .slice(0, limit);
}

export function evidenceForRelation(
  dataset: GraphDataset,
  relationId: string,
  limit = EDGE_EVIDENCE_CHUNK_MAX,
): EvidenceHit[] {
  const edge = dataset.edges.find((item) => asString(item.id) === relationId);
  if (!edge) {
    return [];
  }
  const ends = edgeEnds(edge);
  return allQuotes(dataset)
    .filter((hit) => hit.relationId === relationId || hit.entityId === ends.source || hit.entityId === ends.target)
    .map((hit) => ({ ...hit, quote: hit.quote.slice(0, EDGE_EVIDENCE_CHAR_LIMIT) }))
    .slice(0, limit);
}

export function documentIdOf(row: Record<string, unknown>): string {
  return evidenceDocumentId(row);
}

function toCommunityHit(
  community: Record<string, unknown>,
  nameById: Map<string, string>,
  terms: string[],
): CommunityHit {
  const memberIds = communityMemberIds(community);
  const memberNames = memberIds.map((id) => nameById.get(id) ?? "").join(" ");
  const title = asString(community.title ?? community.name ?? community.id);
  const summary = asString(community.summary ?? community.report);
  const findings = asFindings(community);
  const ratingValue = asNumber(community.rating, Number.NaN);
  const score =
    terms.length === 0
      ? 0
      : scoreHaystack(title, terms) * 3 +
        scoreHaystack(summary, terms) * 2 +
        scoreHaystack(memberNames, terms) +
        findings.reduce((sum, finding) => sum + scoreHaystack(`${finding.summary} ${finding.explanation}`, terms), 0);
  return {
    id: asString(community.id),
    title,
    summary,
    memberIds,
    rating: Number.isFinite(ratingValue) ? ratingValue : null,
    score,
    distance: distanceFromScore(score),
    findings,
    level: Math.max(0, Math.floor(asNumber(community.level, 0))),
    parentCommunityId: asString(community.parent_community_id) || null,
  };
}

function allQuotes(dataset: GraphDataset): EvidenceHit[] {
  const nameById = entityNameIndex(dataset);
  const fromEvidence = dataset.evidence.map((row, index) => quoteHit(row, index, "evidence", nameById));
  const fromChunks = dataset.chunks.map((row, index) => quoteHit(row, index, "chunk", nameById));
  return [...fromEvidence, ...fromChunks].filter((hit) => hit.quote);
}

function quoteHit(
  row: Record<string, unknown>,
  index: number,
  prefix: string,
  nameById: Map<string, string>,
): EvidenceHit {
  const quote = asString(row.quote ?? row.sentence ?? row.content ?? row.text);
  const entityId = evidenceEntityId(row);
  return {
    id: asString(row.id) || `${prefix}_${index}`,
    quote,
    documentId: documentIdOf(row),
    entityId,
    entityName: entityId ? nameById.get(entityId) ?? null : null,
    relationId: evidenceRelationId(row),
    score: 1,
    distance: 0.5,
  };
}

function asFindings(community: Record<string, unknown>): { summary: string; explanation: string }[] {
  const raw = community.findings;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .map((item) => {
      if (!item || typeof item !== "object") {
        return null;
      }
      const row = item as Record<string, unknown>;
      const summary = asString(row.summary ?? row.title);
      if (!summary) {
        return null;
      }
      return { summary, explanation: asString(row.explanation ?? row.detail) };
    })
    .filter((item): item is { summary: string; explanation: string } => Boolean(item));
}
