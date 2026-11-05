import { tool } from "ai";
import { documentNameIndex, entityDocumentIndex } from "@/lib/evidence";
import { z } from "zod";
import type { GraphDataset } from "../protocol/schema";
import {
  EDGE_EVIDENCE_CHUNK_MAX,
  ENTITY_SOURCE_CHUNK_MAX,
  TOOL_DEFAULT_TOP_K,
  TOOL_DOCUMENT_QUOTE_DEFAULT,
  TOOL_DOCUMENT_QUOTE_MAX,
  TOOL_MAX_TOP_K,
} from "./constants";
import { formatGraphSearch, hasGraphHit } from "./format";
import { graphSearch } from "./graph-search";
import { structuredObservation } from "./observations";
import {
  evidenceForDocument,
  evidenceForEntity,
  evidenceForRelation,
  listGraphDocuments,
  neighborhood,
  searchCommunities,
  searchEntities,
  searchEvidence,
  searchRelations,
} from "./retrieve";
import { intersectIds, scopeDataset } from "./scope";
import type { AskScope, QueryMode, SearchProgressUpdate } from "./types";

export interface AskToolSession {
  dataset: GraphDataset;
  scope?: AskScope;
  preferredMode?: QueryMode;
  defaultTopK?: number;
  onProgress?: (update: SearchProgressUpdate) => void;
}

const topKSchema = z.number().int().min(1).max(TOOL_MAX_TOP_K).optional();
const modeSchema = z.enum(["local", "global", "hybrid", "drift"]).optional();

export const ASK_TOOL_NAMES = [
  "searchGraph",
  "searchEntities",
  "searchCommunities",
  "searchRelations",
  "searchEvidence",
  "getNeighborhood",
  "listDocuments",
  "readDocumentEvidence",
  "getEntityEvidence",
  "getRelationEvidence",
] as const;

export function createAskTools(input: AskToolSession) {
  const session = { ...input, dataset: scopeDataset(input.dataset, input.scope) };
  const documentNames = documentNameIndex(session.dataset.documents);
  const entityDocuments = entityDocumentIndex(session.dataset);
  // A citation names its document by file; one quoting an entity's own
  // description, which no single quote grounds, names the document the
  // entity was extracted from.
  const named = <T extends { documentId: string; entityId?: string | null }>(
    citation: T,
  ): T & { documentName?: string } => {
    const documentId =
      citation.documentId || (citation.entityId ? (entityDocuments.get(citation.entityId) ?? "") : "");
    const documentName = documentNames.get(documentId);
    return { ...citation, documentId, ...(documentName ? { documentName } : {}) };
  };
  return {
    searchGraph: tool({
      description: `Search this knowledge graph at the right retrieval level.

Choose mode deliberately:
- local: named entities, relationships, definitions, or "in document X, what is the relationship between A and B?"
- global: themes, summaries, cross-document patterns (community reports)
- hybrid: comparisons that need both entities and themes
- drift: multi-hop exploratory questions (how ideas evolve or influence each other)

If the user names a document, pass its exact document id from listDocuments. Omit mode when unsure; the planner will pick one. Never invent entities that tools do not return.`,
      inputSchema: z.object({
        query: z.string().min(1),
        mode: modeSchema.describe(
          "Graph retrieval route. Use local for specific named entities/relationships, global for broad themes/summaries, hybrid for comparisons needing entity and theme context, and drift for exploratory evolution/influence questions. Omit when uncertain.",
        ),
        documentIds: z.array(z.string().min(1)).optional(),
        rerank: z.boolean().optional(),
        communityLevel: z.number().int().min(0).max(4).optional(),
        topK: topKSchema,
      }),
      execute: async ({ query, mode, documentIds, rerank, communityLevel, topK }) => {
        if (session.dataset.nodes.length === 0) {
          return structuredObservation("GRAPH_HAS_NO_ENTITIES", {
            message: "This graph has no entities to search.",
          });
        }
        const available = listGraphDocuments(session.dataset);
        const availableIds = new Set(available.map((document) => document.id));
        if (documentIds?.length) {
          const unknownDocumentIds = documentIds.filter((id) => !availableIds.has(id));
          if (unknownDocumentIds.length > 0) {
            return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", {
              requestedDocumentIds: documentIds,
              unknownDocumentIds,
              availableDocuments: available,
            });
          }
        }
        // An empty list from the model means no restriction, as an omitted one does.
        const mergedDocumentIds = intersectIds(documentIds?.length ? documentIds : undefined, session.scope?.documentIds);
        if (documentIds?.length && session.scope?.documentIds?.length && (!mergedDocumentIds || mergedDocumentIds.length === 0)) {
          return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", {
            requestedDocumentIds: documentIds,
            scopedDocumentIds: session.scope.documentIds,
            availableDocuments: available,
          });
        }
        const result = await graphSearch(session.dataset, {
          query,
          mode: (mode ?? session.preferredMode) as QueryMode | undefined,
          topK: topK ?? session.defaultTopK ?? TOOL_DEFAULT_TOP_K,
          scope: { ...session.scope, documentIds: mergedDocumentIds },
          documentIds: mergedDocumentIds,
          rerank: rerank ?? false,
          communityLevel,
          onProgress: session.onProgress,
        });
        if (!hasGraphHit(result)) {
          return structuredObservation("GRAPH_EMPTY_RESULT", {
            query,
            mode: result.mode,
            searchedDocumentIds: mergedDocumentIds ?? available.map((document) => document.id),
          });
        }
        const formatted = formatGraphSearch(result);
        const entityIds = [...new Set([...result.entities, ...result.contributingEntities].map((entity) => entity.id))];
        return {
          mode: result.mode,
          plan: result.plan,
          text: formatted.text,
          entityIds: formatted.entityIds,
          relationIds: formatted.relationIds,
          communityIds: formatted.communityIds,
          citations: formatted.citations.map(named),
          counts: {
            entities: entityIds.length,
            contributingEntities: result.contributingEntities.length,
            relations: result.relations.length,
            communities: result.communities.length,
            evidence: result.evidence.length,
          },
        };
      },
    }),
    searchEntities: tool({
      description: "Look up specific named entities (people, organizations, techniques, places) and their descriptions.",
      inputSchema: z.object({
        query: z.string().min(1),
        topK: topKSchema,
      }),
      execute: async ({ query, topK }) => {
        session.onProgress?.({ phase: "searching_entities", message: "Searching entities", detail: query });
        const hits = searchEntities(session.dataset, query, topK ?? session.defaultTopK ?? TOOL_DEFAULT_TOP_K);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "searchEntities" });
        }
        return {
          entities: hits.map((hit) => ({
            id: hit.id,
            name: hit.name,
            type: hit.type,
            description: hit.description,
          })),
        };
      },
    }),
    searchCommunities: tool({
      description: "Search community reports for themes, summaries, and clusters. Use for global / thematic questions.",
      inputSchema: z.object({
        query: z.string().min(1),
        topK: topKSchema,
        communityLevel: z.number().int().min(0).max(4).optional(),
      }),
      execute: async ({ query, topK, communityLevel }) => {
        session.onProgress?.({
          phase: "searching_communities",
          message: "Searching communities",
          detail: query,
        });
        const hits = searchCommunities(session.dataset, query, topK ?? 8, { level: communityLevel ?? 0 });
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "searchCommunities" });
        }
        return {
          communities: hits.map((hit) => ({
            id: hit.id,
            title: hit.title,
            summary: hit.summary,
            members: hit.memberIds.length,
            rating: hit.rating,
            level: hit.level,
            findings: hit.findings.slice(0, 5),
          })),
        };
      },
    }),
    searchRelations: tool({
      description: "Search typed relationships between entities (developed_by, taught_at, part_of, and similar).",
      inputSchema: z.object({
        query: z.string().min(1),
        topK: topKSchema,
      }),
      execute: async ({ query, topK }) => {
        session.onProgress?.({
          phase: "searching_relations",
          message: "Searching relations",
          detail: query,
        });
        const hits = searchRelations(session.dataset, query, topK ?? 20);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "searchRelations" });
        }
        return {
          relations: hits.map((hit) => ({
            id: hit.id,
            source: hit.sourceName,
            target: hit.targetName,
            type: hit.relationType,
            description: hit.description,
            ...(hit.claimed ? { claimed: true } : {}),
          })),
        };
      },
    }),
    searchEvidence: tool({
      description: "Search source quotes and document passages that ground an answer. Use this before citing a fact.",
      inputSchema: z.object({
        query: z.string().min(1),
        topK: topKSchema,
      }),
      execute: async ({ query, topK }) => {
        session.onProgress?.({
          phase: "searching_evidence",
          message: "Searching evidence",
          detail: query,
        });
        const hits = searchEvidence(session.dataset, query, topK ?? session.defaultTopK ?? TOOL_DEFAULT_TOP_K);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "searchEvidence" });
        }
        return {
          citations: hits.map((hit) =>
            named({
              quote: hit.quote,
              documentId: hit.documentId,
              entityId: hit.entityId,
              entityName: hit.entityName,
            }),
          ),
        };
      },
    }),
    getNeighborhood: tool({
      description: "Expand 1-3 hops around a known entity id to inspect its local subgraph.",
      inputSchema: z.object({
        entityId: z.string().min(1),
        hops: z.number().int().min(1).max(3).optional(),
      }),
      execute: async ({ entityId, hops }) => {
        const exists = session.dataset.nodes.some((node) => String(node.id ?? "") === entityId);
        if (!exists) {
          return structuredObservation("UNKNOWN_ENTITY", { entityId });
        }
        const result = neighborhood(session.dataset, entityId, hops ?? 1);
        return {
          entities: result.entities.map((entity) => ({
            id: entity.id,
            name: entity.name,
            type: entity.type,
            description: entity.description,
          })),
          relations: result.relations.map((relation) => ({
            id: relation.id,
            source: relation.sourceName,
            target: relation.targetName,
            type: relation.relationType,
            description: relation.description,
            ...(relation.claimed ? { claimed: true } : {}),
          })),
        };
      },
    }),
    listDocuments: tool({
      description: `List every document in this graph with quote counts.

Use this as the first action when the user refers to "the file", a document name, or an attachment without giving an exact document id. Then pass that id to searchGraph via documentIds, or read ordered quotes with readDocumentEvidence.`,
      inputSchema: z.object({}),
      execute: async () => ({
        documents: listGraphDocuments(session.dataset).map((document) => ({
          id: document.id,
          title: document.title,
          path: document.path,
          quoteCount: document.quoteCount,
          graphAvailable: true,
        })),
      }),
    }),
    readDocumentEvidence: tool({
      description: `Read an ordered window of source quotes from one document id returned by listDocuments.

Use this for summaries or walkthroughs of a single document when searchGraph is not enough. Prefer searchGraph with documentIds for questions about entities in that document.`,
      inputSchema: z.object({
        documentId: z.string().min(1),
        startIndex: z.number().int().min(0).optional().default(0),
        count: z.number().int().min(1).max(TOOL_DOCUMENT_QUOTE_MAX).optional().default(TOOL_DOCUMENT_QUOTE_DEFAULT),
      }),
      execute: async ({ documentId, startIndex, count }) => {
        const documents = listGraphDocuments(session.dataset);
        if (!documents.some((document) => document.id === documentId)) {
          return structuredObservation("UNKNOWN_DOCUMENT", {
            documentId,
            availableDocuments: documents,
          });
        }
        const hits = evidenceForDocument(session.dataset, documentId, startIndex ?? 0, count ?? TOOL_DOCUMENT_QUOTE_DEFAULT);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { documentId, tool: "readDocumentEvidence" });
        }
        return {
          documentId,
          citations: hits.map((hit) =>
            named({
              quote: hit.quote,
              documentId: hit.documentId,
              entityId: hit.entityId,
              entityName: hit.entityName,
            }),
          ),
        };
      },
    }),
    getEntityEvidence: tool({
      description: "Fetch source quotes that mention a known entity id. Use after searchEntities or searchGraph when you need verbatim grounding for that entity.",
      inputSchema: z.object({
        entityId: z.string().min(1),
      }),
      execute: async ({ entityId }) => {
        const exists = session.dataset.nodes.some((node) => String(node.id ?? "") === entityId);
        if (!exists) {
          return structuredObservation("UNKNOWN_ENTITY", { entityId });
        }
        const hits = evidenceForEntity(session.dataset, entityId, ENTITY_SOURCE_CHUNK_MAX);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { entityId, tool: "getEntityEvidence" });
        }
        return {
          entityId,
          citations: hits.map((hit) =>
            named({
              quote: hit.quote,
              documentId: hit.documentId,
              entityId: hit.entityId,
              entityName: hit.entityName,
            }),
          ),
        };
      },
    }),
    getRelationEvidence: tool({
      description: "Fetch source quotes that support a known relation id. Use after searchRelations when you need the passages behind an edge.",
      inputSchema: z.object({
        relationId: z.string().min(1),
      }),
      execute: async ({ relationId }) => {
        const exists = session.dataset.edges.some((edge) => String(edge.id ?? "") === relationId);
        if (!exists) {
          return structuredObservation("UNKNOWN_RELATION", { relationId });
        }
        const hits = evidenceForRelation(session.dataset, relationId, EDGE_EVIDENCE_CHUNK_MAX);
        if (hits.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { relationId, tool: "getRelationEvidence" });
        }
        return {
          relationId,
          citations: hits.map((hit) =>
            named({
              quote: hit.quote,
              documentId: hit.documentId,
              entityId: hit.entityId,
              entityName: hit.entityName,
            }),
          ),
        };
      },
    }),
  };
}

export type AskTools = ReturnType<typeof createAskTools>;
