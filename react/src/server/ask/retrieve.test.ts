import { describe, expect, it } from "vitest";
import { goldToDataset } from "../gold";
import {
  blendedCommunityScore,
  evidenceForEntity,
  evidenceForRelation,
  fuseRrf,
  listGraphDocuments,
  maybePromoteToSuperCommunity,
  neighborhood,
  searchCommunities,
  searchEntities,
  searchEvidence,
  searchRelations,
} from "./retrieve";
import { formatGraphSearch } from "./format";
import { lexicalAsk } from "./lexical";
import { scopeDataset } from "./scope";
import { graphSearch } from "./graph-search";
import { resetAskModel } from "./model";
import { rerankEntities } from "./rerank";
import { createAskTools } from "./tools";
import { structuredObservation } from "./observations";
import type { GraphSearchResult } from "./types";

const dataset = goldToDataset(
  {
    name: "ask-fixture",
    documents: [
      { id: "doc_overview", path: "overview.pdf", title: "Overview" },
      { id: "doc_karate", path: "karate.pdf", title: "Karate notes" },
    ],
    entities: [
      { id: "jigoro_kano", name: "Jigoro Kano", type: "PERSON", description: "Educator who founded Kodokan judo" },
      { id: "judo", name: "Judo", type: "MARTIAL_ART", description: "Modern Japanese martial art" },
      { id: "karate", name: "Karate", type: "MARTIAL_ART", description: "Okinawan striking art" },
    ],
    relations: [
      {
        id: "rel_judo_developed_by",
        source: "judo",
        target: "jigoro_kano",
        relation_type: "DEVELOPED_BY",
        evidence_contains: ["Judo was developed by Jigoro Kano."],
        observations: [
          { document: "doc_overview", sentence: "Judo was developed by Jigoro Kano at the Kodokan." },
        ],
      },
      {
        id: "rel_karate_origin",
        source: "karate",
        target: "judo",
        relation_type: "DISTINCT_FROM",
        observations: [{ document: "doc_karate", sentence: "Karate developed in Okinawa." }],
      },
    ],
  },
  "graph_martial_arts",
);

const emptySearch: GraphSearchResult = {
  mode: "local",
  entities: [],
  contributingEntities: [],
  relations: [],
  communities: [],
  evidence: [],
  plan: { mode: "local", highLevelKeywords: [], lowLevelKeywords: [] },
};

describe("ask retrieval", () => {
  it("ranks the named entity for a local question", () => {
    const hits = searchEntities(dataset, "Who is Jigoro Kano?", 5);
    expect(hits[0]?.id).toBe("jigoro_kano");
  });

  it("returns quotes that mention judo", () => {
    const hits = searchEvidence(dataset, "Who developed judo?", 5);
    expect(hits.some((hit) => /kano/i.test(hit.quote))).toBe(true);
  });

  it("finds the developed_by relation", () => {
    const hits = searchRelations(dataset, "judo developed kano", 5);
    expect(hits.some((hit) => hit.id === "rel_judo_developed_by")).toBe(true);
  });

  it("marks a claimed relation in the answer context and leaves a reported one unchanged", () => {
    const claimed = {
      ...dataset,
      edges: dataset.edges.map((edge) => (edge.id === "rel_karate_origin" ? { ...edge, assertion: "claimed" } : edge)),
    };
    const relations = [
      ...searchRelations(claimed, "judo developed kano", 5),
      ...searchRelations(claimed, "karate distinct", 5),
    ].filter((hit, index, hits) => hits.findIndex((other) => other.id === hit.id) === index);
    const text = formatGraphSearch({ ...emptySearch, relations }).text;
    expect(relations.find((hit) => hit.id === "rel_karate_origin")?.claimed).toBe(true);
    expect(relations.find((hit) => hit.id === "rel_judo_developed_by")).not.toHaveProperty("claimed");
    expect(text).toContain("- Karate -> Judo (DISTINCT_FROM, claimed): ");
    expect(text).toContain("- Judo -> Jigoro Kano (DEVELOPED_BY): ");
    expect(neighborhood(claimed, "karate", 1).relations.map((hit) => hit.claimed)).toEqual([true]);
    const scoped = scopeDataset(claimed, { documentIds: ["doc_karate"] });
    expect(scoped.edges.find((edge) => edge.id === "rel_karate_origin")?.assertion).toBe("claimed");
  });

  it("tells the agent's relation tools which relations are claimed", async () => {
    const claimed = {
      ...dataset,
      edges: dataset.edges.map((edge) => (edge.id === "rel_karate_origin" ? { ...edge, assertion: "claimed" } : edge)),
    };
    const tools = createAskTools({ dataset: claimed });
    const result = (await tools.getNeighborhood.execute!({ entityId: "judo" }, {
      toolCallId: "t3",
      messages: [],
      abortSignal: new AbortController().signal,
    } as never)) as { relations: Array<Record<string, unknown>> };
    const byId = new Map(result.relations.map((relation) => [relation.id, relation]));
    expect(byId.get("rel_karate_origin")?.claimed).toBe(true);
    expect(byId.get("rel_judo_developed_by")).not.toHaveProperty("claimed");
  });

  it("groups martial art communities", () => {
    const hits = searchCommunities(dataset, "martial art entities", 5);
    expect(hits.some((hit) => hit.id === "community_martial_art")).toBe(true);
  });

  it("expands a neighborhood around judo", () => {
    const result = neighborhood(dataset, "judo", 1);
    expect(result.entities.some((entity) => entity.id === "jigoro_kano")).toBe(true);
    expect(result.relations.some((relation) => relation.id === "rel_judo_developed_by")).toBe(true);
  });

  it("returns empty hits for stopword-only questions", () => {
    expect(searchEntities(dataset, "who is the", 5)).toEqual([]);
    expect(searchEvidence(dataset, "what", 5)).toEqual([]);
  });

  it("fuses ranked lists with reciprocal rank fusion", () => {
    const fused = fuseRrf(
      [{ id: "a" }, { id: "b" }],
      [{ id: "b" }, { id: "c" }],
      3,
    );
    expect(fused.map((item) => item.id)).toEqual(["b", "a", "c"]);
  });
});

describe("lexicalAsk", () => {
  it("answers a local name question without community citations", () => {
    const answer = lexicalAsk(dataset, "Who is Jigoro Kano?", "local");
    expect(answer.entityIds).toContain("jigoro_kano");
    expect(answer.usedCommunityReport).toBe(false);
    expect(answer.queryConsumption.billed).toBe(false);
    expect(answer.citations.length).toBeGreaterThan(0);
  });

  it("uses community summaries for global questions", () => {
    const answer = lexicalAsk(dataset, "martial art entities", "global");
    expect(answer.communityIds.length).toBeGreaterThan(0);
    expect(answer.usedCommunityReport).toBe(true);
    expect(answer.citations).toEqual([]);
  });

  it("scopes answers to a perspective search", () => {
    const answer = lexicalAsk(dataset, "Who developed judo?", "local", { search: "judo" });
    expect(answer.scoped).toBe(true);
    expect(answer.entityIds).not.toContain("karate");
  });
});

describe("graphSearch without an LLM", () => {
  it("runs local, global, hybrid, and drift over the in-memory graph", async () => {
    resetAskModel();
    const local = await graphSearch(dataset, { query: "Who developed judo?", mode: "local" });
    expect(local.entities.some((entity) => entity.id === "judo" || entity.id === "jigoro_kano")).toBe(true);
    expect(local.evidence.length).toBeGreaterThan(0);

    const global = await graphSearch(dataset, { query: "martial art", mode: "global" });
    expect(global.communities.length).toBeGreaterThan(0);

    const hybrid = await graphSearch(dataset, { query: "Who developed judo?", mode: "hybrid" });
    expect(hybrid.entities.length + hybrid.communities.length).toBeGreaterThan(0);

    const drift = await graphSearch(dataset, { query: "How did judo spread?", mode: "drift" });
    expect(drift.mode).toBe("drift");
    expect(drift.entities.length + drift.communities.length).toBeGreaterThan(0);
  });

  it("returns empty local results for an empty graph", async () => {
    const empty = goldToDataset({ name: "empty", entities: [], relations: [] }, "graph_empty");
    const result = await graphSearch(empty, { query: "Who developed judo?", mode: "local" });
    expect(result.entities).toEqual([]);
    expect(result.evidence).toEqual([]);
  });

  it("applies community id scope", () => {
    const scoped = scopeDataset(dataset, { communityIds: ["community_person"] });
    expect(scoped.nodes.every((node) => String(node.primary_type) === "PERSON")).toBe(true);
  });

  it("restricts retrieval to selected document ids", async () => {
    resetAskModel();
    const scoped = scopeDataset(dataset, { documentIds: ["doc_overview"] });
    expect(scoped.nodes.map((node) => String(node.id))).toEqual(expect.arrayContaining(["judo", "jigoro_kano"]));
    expect(scoped.nodes.map((node) => String(node.id))).not.toContain("karate");
    const result = await graphSearch(dataset, {
      query: "Who developed judo?",
      mode: "local",
      documentIds: ["doc_overview"],
    });
    expect(result.evidence.every((hit) => hit.documentId === "doc_overview")).toBe(true);
    expect(result.entities.every((entity) => entity.id !== "karate")).toBe(true);
  });
});

describe("hub-parity ranking and documents", () => {
  it("ranks equally matched communities by rating blend", () => {
    const rated = {
      ...dataset,
      communities: [
        {
          id: "low",
          title: "theme cluster",
          summary: "theme cluster",
          members: ["judo"],
          rating: 1,
          level: 0,
        },
        {
          id: "high",
          title: "theme cluster",
          summary: "theme cluster",
          members: ["karate"],
          rating: 9,
          level: 0,
        },
      ],
    };
    const hits = searchCommunities(rated, "theme cluster", 2);
    expect(hits[0]?.id).toBe("high");
    expect(blendedCommunityScore(hits[0]!)).toBeLessThan(blendedCommunityScore(hits[1]!));
  });

  it("promotes homogeneous leaf communities to their parent", () => {
    const hierarchical = {
      ...dataset,
      communities: [
        ...dataset.communities.map((community) => ({
          ...community,
          level: 0,
          parent_community_id: "super_arts",
        })),
        {
          id: "super_arts",
          title: "All arts",
          summary: "Parent of people and martial arts",
          members: ["judo", "karate", "jigoro_kano"],
          level: 1,
        },
      ],
    };
    const leaves = searchCommunities(hierarchical, "entities", 8);
    expect(leaves.length).toBeGreaterThan(1);
    const promoted = maybePromoteToSuperCommunity(hierarchical, leaves);
    expect(promoted).toHaveLength(1);
    expect(promoted[0]?.id).toBe("super_arts");
  });

  it("lists documents and entity or relation evidence", () => {
    const documents = listGraphDocuments(dataset);
    expect(documents.map((document) => document.id)).toEqual(expect.arrayContaining(["doc_overview", "doc_karate"]));
    expect(evidenceForEntity(dataset, "jigoro_kano").some((hit) => /kano/i.test(hit.quote))).toBe(true);
    expect(evidenceForRelation(dataset, "rel_judo_developed_by").some((hit) => /kodokan/i.test(hit.quote))).toBe(true);
  });

  it("cites entity descriptions as well as source quotes", async () => {
    resetAskModel();
    const result = await graphSearch(dataset, { query: "Who developed judo?", mode: "local" });
    const formatted = formatGraphSearch(result);
    expect(formatted.citations.some((citation) => citation.entityId === "judo" || citation.entityId === "jigoro_kano")).toBe(
      true,
    );
    expect(formatted.text).toMatch(/Judo|Jigoro Kano/);
  });

  it("keeps original order when rerank has no model", async () => {
    resetAskModel();
    const hits = searchEntities(dataset, "art", 5);
    expect(hits.length).toBeGreaterThan(1);
    const reranked = await rerankEntities("martial arts", hits, 1);
    expect(reranked).toHaveLength(1);
    expect(reranked[0]?.id).toBe(hits[0]?.id);
  });

  it("returns GRAPH_HAS_NO_ENTITIES on an empty graph", async () => {
    const tools = createAskTools({
      dataset: goldToDataset({ name: "empty", entities: [], relations: [] }, "graph_empty"),
    });
    const result = await tools.searchGraph.execute!({ query: "Who developed judo?" }, {
      toolCallId: "t1",
      messages: [],
      abortSignal: new AbortController().signal,
    } as never);
    expect(result).toMatchObject(structuredObservation("GRAPH_HAS_NO_ENTITIES", { message: "This graph has no entities to search." }));
  });

  it("rejects unknown document ids", async () => {
    const tools = createAskTools({ dataset });
    const result = await tools.searchGraph.execute!({ query: "Who developed judo?", documentIds: ["doc_missing"] }, {
      toolCallId: "t2",
      messages: [],
      abortSignal: new AbortController().signal,
    } as never);
    expect(result).toMatchObject({ observation: "ERROR_OR_EMPTY_RESULT", code: "GRAPH_DOCUMENT_SCOPE_MISMATCH" });
  });
});
