import type { GraphDataset } from "../protocol/schema";
import { formatGraphSearch } from "./format";
import { searchCommunities, searchEntities, searchEvidence, searchRelations } from "./retrieve";
import { isScoped, scopeDataset } from "./scope";
import type { AskAnswer, AskScope } from "./types";

export type { AskScope };

/** The modes a graphs.ask caller may send. */
export type LexicalAskMode = "local" | "global";

export function lexicalAsk(
  dataset: GraphDataset,
  question: string,
  mode: LexicalAskMode = "local",
  scope?: AskScope,
): AskAnswer {
  const scopedDataset = scopeDataset(dataset, scope);
  const scoped = isScoped(scope);
  if (mode === "global") {
    const communities = searchCommunities(scopedDataset, question, 3);
    return {
      mode,
      question,
      summary: communities.length
        ? communities.map((community) => `${community.title}: ${community.summary}`).join(" ")
        : "No community summary matched. Try a local entity question.",
      entityIds: [],
      relationIds: [],
      communityIds: communities.map((community) => community.id),
      citations: [],
      usedCommunityReport: communities.length > 0,
      scoped,
      queryConsumption: {
        retrievedEntities: 0,
        retrievedQuotes: 0,
        retrievedCommunities: communities.length,
        retrievedRelations: 0,
        locality: "graph",
        billed: false,
        note: "Lexical community retrieval. Not a billed LLM call.",
      },
    };
  }
  const entities = searchEntities(scopedDataset, question, 8);
  const entityIds = new Set(entities.map((entity) => entity.id));
  const relations = searchRelations(scopedDataset, question, 20).filter(
    (relation) => entityIds.has(relation.sourceId) || entityIds.has(relation.targetId),
  );
  const evidence = searchEvidence(scopedDataset, question, 8);
  const formatted = formatGraphSearch({
    mode: "local",
    plan: { mode: "local", highLevelKeywords: [], lowLevelKeywords: [] },
    entities,
    contributingEntities: [],
    relations,
    communities: [],
    evidence,
  });
  const names = entities.map((entity) => entity.name);
  return {
    mode: "local",
    question,
    summary: names.length
      ? `Matched ${names.join(", ")}. Citations below are document quotes, not community reports.`
      : "No entity matched. Try a name you already believe is in the corpus.",
    entityIds: [...entityIds],
    relationIds: relations.map((relation) => relation.id),
    communityIds: [],
    citations: formatted.citations,
    usedCommunityReport: false,
    scoped,
    queryConsumption: {
      retrievedEntities: entityIds.size,
      retrievedQuotes: evidence.length,
      retrievedCommunities: 0,
      retrievedRelations: relations.length,
      locality: "graph",
      billed: false,
      note: "Query retrieval over this graph. Not a billed LLM call.",
    },
  };
}

export const askGraph = lexicalAsk;
