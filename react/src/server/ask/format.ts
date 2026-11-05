import {
  GRAPH_SEARCH_CONTEXT_CHAR_BUDGET,
  TOOL_COMMUNITY_FINDINGS_SLICE,
  TOOL_ENTITY_DESCRIPTION_CHAR_LIMIT,
  TOOL_ENTITY_RESULTS_SLICE,
  TOOL_RELATION_RESULTS_SLICE,
  TOOL_RELATION_SEGMENT_CHAR_LIMIT,
} from "./constants";
import type { AskAnswer, AskCitation, GraphSearchResult, QueryMode } from "./types";

export interface FormattedGraphContext {
  text: string;
  citations: AskCitation[];
  entityIds: string[];
  relationIds: string[];
  communityIds: string[];
  usedCommunityReport: boolean;
}

export function formatGraphSearch(result: GraphSearchResult): FormattedGraphContext {
  const sections: string[] = [];
  const citations: AskCitation[] = [];
  let remaining = GRAPH_SEARCH_CONTEXT_CHAR_BUDGET;
  const consume = (line: string): boolean => {
    const cost = line.length + 1;
    if (cost > remaining) {
      return false;
    }
    remaining -= cost;
    return true;
  };

  if (result.communities.length > 0) {
    const lines = ["## Community context"];
    if (consume(lines[0]!)) {
      for (const community of result.communities) {
        const ratingSuffix =
          community.rating != null ? `, impact ${community.rating.toFixed(1)}/10` : "";
        const line = `- **${community.title}** (${community.memberIds.length} members${ratingSuffix}): ${community.summary}`;
        if (!consume(line)) {
          break;
        }
        lines.push(line);
        for (const finding of community.findings.slice(0, TOOL_COMMUNITY_FINDINGS_SLICE)) {
          const findingLine = `  - *${finding.summary}*: ${finding.explanation}`;
          if (!consume(findingLine)) {
            break;
          }
          lines.push(findingLine);
        }
      }
      if (lines.length > 1) {
        sections.push(lines.join("\n"));
      }
    }
  }

  const seen = new Set<string>();
  const entities = [...result.entities, ...result.contributingEntities].filter((entity) => {
    if (seen.has(entity.id)) {
      return false;
    }
    seen.add(entity.id);
    return true;
  });

  if (entities.length > 0 || result.relations.length > 0) {
    const lines = ["## Entities and relations"];
    if (consume(lines[0]!)) {
      for (const entity of entities.slice(0, TOOL_ENTITY_RESULTS_SLICE)) {
        const rendered = `- **${entity.name}** (${entity.type}): ${entity.description.slice(0, TOOL_ENTITY_DESCRIPTION_CHAR_LIMIT)}`;
        if (!consume(rendered)) {
          break;
        }
        lines.push(rendered);
        if (entity.description.trim()) {
          citations.push({
            quote: entity.description,
            documentId: documentIdForEntity(result.evidence, entity.id),
            entityId: entity.id,
            entityName: entity.name,
          });
        }
      }
      for (const relation of result.relations.slice(0, TOOL_RELATION_RESULTS_SLICE)) {
        const rendered = `- ${relation.sourceName} -> ${relation.targetName} (${relation.relationType}${relation.claimed ? ", claimed" : ""}): ${relation.description.slice(0, TOOL_RELATION_SEGMENT_CHAR_LIMIT)}`;
        if (!consume(rendered)) {
          break;
        }
        lines.push(rendered);
      }
      if (lines.length > 1) {
        sections.push(lines.join("\n"));
      }
    }
  }

  if (result.evidence.length > 0) {
    const lines = ["## Source quotes"];
    if (consume(lines[0]!)) {
      for (const hit of result.evidence) {
        const rendered = `- [${hit.documentId || "document"}] ${hit.quote}`;
        if (!consume(rendered)) {
          break;
        }
        lines.push(rendered);
        citations.push({
          quote: hit.quote,
          documentId: hit.documentId,
          entityId: hit.entityId,
          entityName: hit.entityName,
        });
      }
      if (lines.length > 1) {
        sections.push(lines.join("\n"));
      }
    }
  }

  return {
    text: sections.join("\n\n---\n\n") || "No matching graph context.",
    citations,
    entityIds: entities.map((entity) => entity.id),
    relationIds: result.relations.map((relation) => relation.id),
    communityIds: result.communities.map((community) => community.id),
    usedCommunityReport: result.communities.length > 0,
  };
}

function documentIdForEntity(evidence: GraphSearchResult["evidence"], entityId: string): string {
  return evidence.find((hit) => hit.entityId === entityId && hit.documentId)?.documentId ?? "";
}

export function hasGraphHit(result: GraphSearchResult): boolean {
  return (
    result.evidence.length > 0 ||
    result.entities.length > 0 ||
    result.communities.length > 0 ||
    result.contributingEntities.length > 0
  );
}

export function answerFromSearch(
  question: string,
  mode: QueryMode,
  result: GraphSearchResult,
  summary: string,
  scoped: boolean,
  billed: boolean,
): AskAnswer {
  const formatted = formatGraphSearch(result);
  return {
    mode,
    question,
    summary,
    entityIds: formatted.entityIds,
    relationIds: formatted.relationIds,
    communityIds: formatted.communityIds,
    citations: formatted.citations,
    usedCommunityReport: formatted.usedCommunityReport,
    scoped,
    plan: result.plan,
    queryConsumption: {
      retrievedEntities: new Set([...result.entities, ...result.contributingEntities].map((entity) => entity.id)).size,
      retrievedQuotes: result.evidence.length,
      retrievedCommunities: result.communities.length,
      retrievedRelations: result.relations.length,
      locality: "graph",
      billed,
      note: billed
        ? "Graph retrieval plus a billed LLM answer."
        : "Lexical graph retrieval only. No LLM call.",
    },
  };
}
