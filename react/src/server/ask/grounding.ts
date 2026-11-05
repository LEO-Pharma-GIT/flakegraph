import type { AskAnswer, AskCitation, QueryMode, QueryRetrievalPlan } from "./types";

export interface AskGrounding {
  entityIds: string[];
  relationIds: string[];
  communityIds: string[];
  citations: AskCitation[];
  retrievedEntities: number;
  retrievedQuotes: number;
  retrievedCommunities: number;
  retrievedRelations: number;
  usedCommunityReport: boolean;
  plan?: QueryRetrievalPlan;
  mode?: QueryMode;
}

export function emptyGrounding(): AskGrounding {
  return {
    entityIds: [],
    relationIds: [],
    communityIds: [],
    citations: [],
    retrievedEntities: 0,
    retrievedQuotes: 0,
    retrievedCommunities: 0,
    retrievedRelations: 0,
    usedCommunityReport: false,
  };
}

export function mergeGrounding(left: AskGrounding, right: Partial<AskGrounding>): AskGrounding {
  return {
    entityIds: unique([...left.entityIds, ...(right.entityIds ?? [])]),
    relationIds: unique([...left.relationIds, ...(right.relationIds ?? [])]),
    communityIds: unique([...left.communityIds, ...(right.communityIds ?? [])]),
    citations: mergeCitations(left.citations, right.citations ?? []),
    retrievedEntities: Math.max(left.retrievedEntities, right.retrievedEntities ?? 0),
    retrievedQuotes: Math.max(left.retrievedQuotes, right.retrievedQuotes ?? 0),
    retrievedCommunities: Math.max(left.retrievedCommunities, right.retrievedCommunities ?? 0),
    retrievedRelations: Math.max(left.retrievedRelations, right.retrievedRelations ?? 0),
    usedCommunityReport: left.usedCommunityReport || Boolean(right.usedCommunityReport),
    plan: right.plan ?? left.plan,
    mode: right.mode ?? left.mode,
  };
}

export function groundingFromToolOutput(output: unknown): AskGrounding {
  const grounding = emptyGrounding();
  if (!output || typeof output !== "object") {
    return grounding;
  }
  const row = output as Record<string, unknown>;
  const entityIds = [
    ...asStringArray(row.entityIds),
    ...asObjectIds(row.entities),
  ];
  const relationIds = [
    ...asStringArray(row.relationIds),
    ...asObjectIds(row.relations),
  ];
  const communityIds = [
    ...asStringArray(row.communityIds),
    ...asObjectIds(row.communities),
  ];
  const citations = asCitations(row.citations);
  const counts = row.counts && typeof row.counts === "object" ? (row.counts as Record<string, unknown>) : {};
  return mergeGrounding(grounding, {
    entityIds,
    relationIds,
    communityIds,
    citations,
    retrievedEntities: asNumber(counts.entities) || entityIds.length,
    retrievedQuotes: asNumber(counts.evidence) || citations.length,
    retrievedCommunities: asNumber(counts.communities) || communityIds.length,
    retrievedRelations: asNumber(counts.relations) || relationIds.length,
    usedCommunityReport: communityIds.length > 0,
    plan: isPlan(row.plan) ? row.plan : undefined,
    mode: isMode(row.mode) ? row.mode : undefined,
  });
}

export function groundingFromGenerateSteps(steps: Array<{ toolResults?: Array<{ output?: unknown }> }>): AskGrounding {
  return steps.reduce(
    (grounding, step) =>
      (step.toolResults ?? []).reduce(
        (current, result) => mergeGrounding(current, groundingFromToolOutput(result.output)),
        grounding,
      ),
    emptyGrounding(),
  );
}

export function answerFromGrounding(args: {
  question: string;
  mode: QueryMode;
  summary: string;
  scoped: boolean;
  billed: boolean;
  grounding: AskGrounding;
}): AskAnswer {
  const { grounding } = args;
  return {
    mode: grounding.mode ?? args.mode,
    question: args.question,
    summary: args.summary,
    entityIds: grounding.entityIds,
    relationIds: grounding.relationIds,
    communityIds: grounding.communityIds,
    citations: grounding.citations,
    usedCommunityReport: grounding.usedCommunityReport,
    scoped: args.scoped,
    plan: grounding.plan,
    queryConsumption: {
      retrievedEntities: grounding.retrievedEntities || grounding.entityIds.length,
      retrievedQuotes: grounding.retrievedQuotes || grounding.citations.length,
      retrievedCommunities: grounding.retrievedCommunities || grounding.communityIds.length,
      retrievedRelations: grounding.retrievedRelations || grounding.relationIds.length,
      locality: "graph",
      billed: args.billed,
      note: args.billed
        ? "Graph retrieval plus a billed LLM answer."
        : "Graph retrieval plus a local model answer.",
    },
  };
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item)).filter(Boolean);
}

function asObjectIds(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }
      return String((item as { id?: unknown }).id ?? "");
    })
    .filter(Boolean);
}

function asCitations(value: unknown): AskCitation[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const row = item as Record<string, unknown>;
    const quote = String(row.quote ?? "").trim();
    if (!quote) {
      return [];
    }
    return [
      {
        quote,
        documentId: String(row.documentId ?? ""),
        ...(typeof row.documentName === "string" && row.documentName ? { documentName: row.documentName } : {}),
        entityId: row.entityId == null ? null : String(row.entityId),
        entityName: row.entityName == null ? null : String(row.entityName),
      },
    ];
  });
}

function mergeCitations(left: AskCitation[], right: AskCitation[]): AskCitation[] {
  const seen = new Set<string>();
  const out: AskCitation[] = [];
  for (const citation of [...left, ...right]) {
    const key = `${citation.documentId}:${citation.quote}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(citation);
  }
  return out;
}

function asNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function isMode(value: unknown): value is QueryMode {
  return value === "local" || value === "global" || value === "hybrid" || value === "drift";
}

function isPlan(value: unknown): value is QueryRetrievalPlan {
  if (!value || typeof value !== "object") {
    return false;
  }
  const row = value as Record<string, unknown>;
  return isMode(row.mode) && Array.isArray(row.highLevelKeywords) && Array.isArray(row.lowLevelKeywords);
}
