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
  TOOL_ENTITY_DESCRIPTION_CHAR_LIMIT,
  TOOL_MAX_TOP_K,
  PASSAGE_DEFAULT_TOP_K,
  TOOL_NEIGHBORHOOD_ENTITY_MAX,
  TOOL_NEIGHBORHOOD_RELATION_MAX,
  TOOL_RELATION_SEGMENT_CHAR_LIMIT,
} from "./constants";
import { prepareDense } from "./dense";
import { resolveAskEmbedding } from "./embedding-model";
import { formatGraphSearch, hasGraphHit } from "./format";
import { graphSearch } from "./graph-search";
import { structuredObservation } from "./observations";
import { searchPassages } from "./passages";
import { findDocuments, renderCards } from "./catalogue";
import type { QuestionScope } from "./question-scope";
import {
  PER_DOCUMENT_MAX_DOCUMENTS,
  PER_DOCUMENT_PASSAGES,
  readEachDocument,
  renderFindings,
  type PerDocumentSource,
} from "./per-document";
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

/**
 * What the agent learned about the question before searching (question-scope.ts),
 * filled in on its first step and read by the tools that need it.
 */
export interface AskQuestionContext {
  scope?: QuestionScope;
}

export interface AskToolSession {
  dataset: GraphDataset;
  question?: AskQuestionContext;
  scope?: AskScope;
  preferredMode?: QueryMode;
  defaultTopK?: number;
  onProgress?: (update: SearchProgressUpdate) => void;
}

const topKSchema = z.number().int().min(1).max(TOOL_MAX_TOP_K).optional();
const modeSchema = z.enum(["local", "global", "hybrid", "drift"]).optional();

export const ASK_TOOL_NAMES = [
  "searchPassages",
  "findDocuments",
  "readEachDocument",
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
  // Passages are searched in the graph as loaded, whose index is built once
  // and kept, and narrowed to the scope by id: a scoped copy would rebuild
  // the index for every question and lose the chunks' section headings.
  const embedding = resolveAskEmbedding();
  prepareDense(input.dataset, embedding);
  const scopedChunkIds = input.scope?.search || input.scope?.communityIds?.length
    ? new Set(session.dataset.chunks.map((chunk) => String(chunk.id ?? chunk.chunk_id ?? "")))
    : undefined;
  const documentScope = (documentIds: string[] | undefined) => {
    const available = listGraphDocuments(session.dataset);
    const availableIds = new Set(available.map((document) => document.id));
    const unknownDocumentIds = (documentIds ?? []).filter((id) => !availableIds.has(id));
    if (unknownDocumentIds.length > 0) {
      return { mismatch: { requestedDocumentIds: documentIds, unknownDocumentIds, availableDocuments: available } };
    }
    // An empty list from the model means no restriction, as an omitted one does.
    const merged = intersectIds(documentIds?.length ? documentIds : undefined, session.scope?.documentIds);
    if (documentIds?.length && session.scope?.documentIds?.length && (!merged || merged.length === 0)) {
      return { mismatch: { requestedDocumentIds: documentIds, scopedDocumentIds: session.scope.documentIds, availableDocuments: available } };
    }
    return { documentIds: merged };
  };
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
  // The documents a question spanning several of them reads, when it names
  // none: all of them when the graph is small enough, otherwise those whose
  // cards match its words, then those holding its best passages.
  const candidateDocuments = async (words: string) => {
    const all = findDocuments(session.dataset, { limit: PER_DOCUMENT_MAX_DOCUMENTS });
    if (all.total <= PER_DOCUMENT_MAX_DOCUMENTS) {
      return all.cards;
    }
    const byCard = findDocuments(session.dataset, { query: words, limit: PER_DOCUMENT_MAX_DOCUMENTS }).cards;
    const chosen = new Map(byCard.map((card) => [card.documentId, card]));
    if (chosen.size < PER_DOCUMENT_MAX_DOCUMENTS) {
      const found = await searchPassages(input.dataset, {
        query: words,
        topK: PER_DOCUMENT_MAX_DOCUMENTS * 2,
        chunkIds: scopedChunkIds,
        embedding,
      });
      const held = new Set(found.passages.map((passage) => passage.documentId));
      const cards = findDocuments(session.dataset, { documentIds: [...held], limit: PER_DOCUMENT_MAX_DOCUMENTS }).cards;
      for (const card of cards) {
        if (chosen.size >= PER_DOCUMENT_MAX_DOCUMENTS) break;
        chosen.set(card.documentId, card);
      }
    }
    return [...chosen.values()];
  };
  return {
    searchPassages: tool({
      description: `Search the documents' own text - every passage, tables and charts included - by its words and its meaning.

Use this first for any factual question: exact facts, numbers, percentages, dates, names, definitions, quotes, or anything a document states. It reads the source passages themselves, which the other tools only summarise. Pass documentIds from listDocuments to search within named documents, and expand: true to read the text around each passage. Search again with other words before concluding a fact is not in the graph. Cite what it returns by document.`,
      inputSchema: z.object({
        query: z.string().min(1).describe("What to find, in the words a document would use: names, numbers, terms."),
        documentIds: z.array(z.string().min(1)).optional(),
        topK: topKSchema,
        expand: z.boolean().optional().describe("Add the end of the previous passage and the start of the next one."),
      }),
      execute: async ({ query, documentIds, topK, expand }, options) => {
        session.onProgress?.({ phase: "searching_evidence", message: "Searching passages", detail: query });
        const scoped = documentScope(documentIds);
        if (scoped.mismatch) {
          return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", scoped.mismatch);
        }
        const result = await searchPassages(input.dataset, {
          query,
          topK: topK ?? PASSAGE_DEFAULT_TOP_K,
          expand: expand ?? false,
          documentIds: scoped.documentIds,
          chunkIds: scopedChunkIds,
          embedding,
          abortSignal: options?.abortSignal,
        });
        if (result.passages.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "searchPassages" });
        }
        return {
          retrieval: result.retrieval,
          passages: result.passages.map((passage) => ({
            chunkId: passage.chunkId,
            documentId: passage.documentId,
            documentName: passage.documentName,
            pageNumber: passage.pageNumber,
            sectionPath: passage.sectionPath,
            text: passage.text,
            contains: passage.contains,
            method: passage.method,
          })),
        };
      },
    }),
    findDocuments: tool({
      description: `List the graph's documents with their catalogue cards: each one's title, type, subject, period, places, organizations, topics, method and key findings.

Use it for questions about the documents themselves - which of them cover something, how many did something, what each one is - and to see which documents a question spanning several of them should read. Omit query to list every document; pass query to rank cards by its words.`,
      inputSchema: z.object({
        query: z.string().optional().describe("Words to rank the cards by; omit to list every document."),
        documentIds: z.array(z.string().min(1)).optional(),
        limit: z.number().int().min(1).max(80).optional(),
      }),
      execute: async ({ query, documentIds, limit }) => {
        session.onProgress?.({ phase: "searching_evidence", message: "Reading the document catalogue", detail: query });
        const scoped = documentScope(documentIds);
        if (scoped.mismatch) {
          return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", scoped.mismatch);
        }
        const result = findDocuments(session.dataset, { query, documentIds: scoped.documentIds, limit });
        if (result.cards.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { query, tool: "findDocuments" });
        }
        return { text: renderCards(result), documentIds: result.cards.map((card) => card.documentId) };
      },
    }),
    readEachDocument: tool({
      description: `Read several documents one by one for the same question, in parallel, and get what each says on it - or that it says nothing.

Use it whenever the answer spans documents: comparing them, listing or counting the ones that found or did something, or "which/all/every" questions. One search across all documents favours a few of them; this reads each separately. Pass documentIds from findDocuments or searchPassages, or omit them to read the documents whose cards and passages best match the question (up to ${PER_DOCUMENT_MAX_DOCUMENTS}). Read the whole set in one call - never split it into batches - and follow up only on the documents it could not read. Your answer must then account for every document that bears on the question.`,
      inputSchema: z.object({
        question: z.string().min(1).describe("The question each document is read for, as the user asked it."),
        query: z.string().optional().describe("Words to find each document's relevant passages; defaults to the question."),
        documentIds: z.array(z.string().min(1)).max(PER_DOCUMENT_MAX_DOCUMENTS).optional(),
      }),
      execute: async ({ question, query, documentIds }, options) => {
        const scoped = documentScope(documentIds);
        if (scoped.mismatch) {
          return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", scoped.mismatch);
        }
        const words = query?.trim() || question;
        const chosen = scoped.documentIds?.length
          ? findDocuments(session.dataset, { documentIds: scoped.documentIds, limit: PER_DOCUMENT_MAX_DOCUMENTS }).cards
          : await candidateDocuments(words);
        if (chosen.length === 0) {
          return structuredObservation("GRAPH_EMPTY_RESULT", { question, tool: "readEachDocument" });
        }
        session.onProgress?.({
          phase: "searching_evidence",
          message: `Reading ${chosen.length} documents one by one`,
          detail: question,
        });
        const sources: PerDocumentSource[] = await Promise.all(
          chosen.map(async (card) => {
            const found = await searchPassages(input.dataset, {
              query: words,
              topK: PER_DOCUMENT_PASSAGES,
              documentIds: [card.documentId],
              chunkIds: scopedChunkIds,
              embedding,
              abortSignal: options?.abortSignal,
            });
            return {
              documentId: card.documentId,
              documentName: card.name,
              card: card.card,
              group: card.folder,
              passages: found.passages.map((passage) => ({
                page: passage.pageNumber ?? 0,
                sectionPath: passage.sectionPath,
                text: passage.text,
              })),
            };
          }),
        );
        const findings = await readEachDocument(question, sources, {
          signal: options?.abortSignal,
          assumptions: session.question?.scope?.assumptions ?? [],
          onProgress: (done, total) =>
            session.onProgress?.({ phase: "searching_evidence", message: `Read ${done} of ${total} documents`, detail: question }),
        });
        return {
          text: renderFindings(findings),
          documentIds: findings.filter((finding) => finding.relevant).map((finding) => finding.documentId),
        };
      },
    }),
    searchGraph: tool({
      description: `Search this knowledge graph at the right retrieval level.

Choose mode deliberately:
- local: named entities, relationships, definitions, or "in document X, what is the relationship between A and B?"
- global: themes, summaries, cross-document patterns (community reports)
- hybrid: comparisons that need both entities and themes
- drift: multi-hop exploratory questions (how ideas evolve or influence each other)

If the user names a document, pass its exact document id from listDocuments. Omit mode when unsure; the planner will pick one. Never invent entities that tools do not return. For an exact fact, number or quote, searchPassages reads the documents' text itself.`,
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
        const scoped = documentScope(documentIds);
        if (scoped.mismatch) {
          return structuredObservation("GRAPH_DOCUMENT_SCOPE_MISMATCH", scoped.mismatch);
        }
        const mergedDocumentIds = scoped.documentIds;
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
            searchedDocumentIds: mergedDocumentIds ?? listGraphDocuments(session.dataset).map((document) => document.id),
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
      description: "Search the quotes that ground the graph's entities and relations. For what a document says - facts, numbers, tables, with page and section - prefer searchPassages.",
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
        // Three hops from a hub reach most of a graph: keep the strongest
        // relations and the entities they join, and say how much is left out.
        const relations = result.relations
          .toSorted((left, right) => right.weight - left.weight)
          .slice(0, TOOL_NEIGHBORHOOD_RELATION_MAX);
        const joined = new Set(relations.flatMap((relation) => [relation.sourceId, relation.targetId]));
        const entities = result.entities
          .filter((entity) => entity.id === entityId || joined.has(entity.id))
          .toSorted((left, right) => right.score - left.score)
          .slice(0, TOOL_NEIGHBORHOOD_ENTITY_MAX);
        return {
          entities: entities.map((entity) => ({
            id: entity.id,
            name: entity.name,
            type: entity.type,
            description: entity.description.slice(0, TOOL_ENTITY_DESCRIPTION_CHAR_LIMIT),
          })),
          relations: relations.map((relation) => ({
            id: relation.id,
            source: relation.sourceName,
            target: relation.targetName,
            type: relation.relationType,
            description: relation.description.slice(0, TOOL_RELATION_SEGMENT_CHAR_LIMIT),
            ...(relation.claimed ? { claimed: true } : {}),
          })),
          leftOut: {
            entities: result.entities.length - entities.length,
            relations: result.relations.length - relations.length,
          },
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
