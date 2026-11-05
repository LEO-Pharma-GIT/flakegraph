import { generateText, Output } from "ai";
import { z } from "zod";
import {
  LLM_RERANK_CONCURRENCY,
  RERANK_CHUNK_PROMPT_CHAR_LIMIT,
  RERANK_COMMUNITY_PROMPT_CHAR_LIMIT,
  RERANK_ENTITY_PROMPT_CHAR_LIMIT,
  RERANK_LLM_CALL_TIMEOUT_MS,
  RERANK_SCORE_MAX,
  RERANK_SCORE_MIN,
} from "./constants";
import { resolveAskModel } from "./model";
import type { CommunityHit, EntityHit, EvidenceHit, SearchProgressUpdate } from "./types";
import { errorMessage } from "./errors";

const RelevanceSchema = z.object({
  score: z.number().int().min(RERANK_SCORE_MIN).max(RERANK_SCORE_MAX),
  rationale: z.string().max(240),
});

export interface RerankProgressUpdate {
  completed: number;
  failed: number;
  total: number;
}

type Reranked<T> = T & { relevanceScore?: number };

function relevanceSystemPrompt(itemNoun: string): string {
  return `You rate how relevant a ${itemNoun} is to a user question.

Rules:
- Score is an integer from ${RERANK_SCORE_MIN} to ${RERANK_SCORE_MAX}.
- ${RERANK_SCORE_MAX} = the ${itemNoun} directly answers the question.
- ${RERANK_SCORE_MIN} = the ${itemNoun} is unrelated.
- Middle scores reflect partial or tangential relevance.
- Score only on the content provided. Do NOT speculate about what may surround the ${itemNoun} in the source.
- Return exactly one JSON object with keys "score" and "rationale"; do not return an array, markdown, or multiple objects.`;
}

export async function rerankEntities(
  query: string,
  hits: EntityHit[],
  topK: number,
  onProgress?: (update: RerankProgressUpdate) => void,
): Promise<EntityHit[]> {
  if (hits.length === 0 || hits.length <= topK) {
    return hits;
  }
  const scored = await rerankWithLlm({
    items: hits,
    buildPrompt: (hit) => ({
      system: relevanceSystemPrompt("knowledge-graph entity"),
      user: `Question:\n${query}\n\nEntity:\nName: ${hit.name}\nType: ${hit.type}\nDescription:\n${hit.description.slice(0, RERANK_ENTITY_PROMPT_CHAR_LIMIT) || "(no description)"}`,
    }),
    tiebreak: (hit) => hit.distance,
    onProgress,
  });
  return scored.slice(0, topK).map(stripScore);
}

export async function rerankEvidence(
  query: string,
  hits: EvidenceHit[],
  topK: number,
  onProgress?: (update: RerankProgressUpdate) => void,
): Promise<EvidenceHit[]> {
  if (hits.length === 0 || hits.length <= topK) {
    return hits;
  }
  const scored = await rerankWithLlm({
    items: hits,
    buildPrompt: (hit) => ({
      system: relevanceSystemPrompt("source passage"),
      user: `Question:\n${query}\n\nPassage:\n${hit.quote.slice(0, RERANK_CHUNK_PROMPT_CHAR_LIMIT)}`,
    }),
    tiebreak: (hit) => hit.distance,
    onProgress,
  });
  return scored.slice(0, topK).map(stripScore);
}

export async function rerankCommunities(
  query: string,
  hits: CommunityHit[],
  onProgress?: (update: RerankProgressUpdate) => void,
): Promise<CommunityHit[]> {
  if (hits.length === 0) {
    return hits;
  }
  const scored = await rerankWithLlm({
    items: hits,
    buildPrompt: (hit) => {
      const findings = hit.findings
        .map((finding, index) => `${index + 1}. ${finding.summary} — ${finding.explanation}`)
        .join("\n")
        .slice(0, RERANK_COMMUNITY_PROMPT_CHAR_LIMIT);
      return {
        system: relevanceSystemPrompt("knowledge-graph community"),
        user: `Question:\n${query}\n\nCommunity: ${hit.title}\n\nSummary:\n${hit.summary.slice(0, RERANK_COMMUNITY_PROMPT_CHAR_LIMIT)}\n\nKey findings:\n${findings || "(none)"}`,
      };
    },
    tiebreak: (hit) => hit.distance,
    onProgress,
  });
  return scored.map(stripScore);
}

export function rerankProgressDetail(noun: string, progress: RerankProgressUpdate): string {
  return `Scored ${progress.completed}/${progress.total} ${noun}`;
}

export function emitRerankProgress(
  onProgress: ((update: SearchProgressUpdate) => void) | undefined,
  phase: "reranking_entities" | "reranking_evidence" | "reranking_communities",
  noun: string,
): ((update: RerankProgressUpdate) => void) | undefined {
  if (!onProgress) {
    return undefined;
  }
  const labels = {
    reranking_entities: "Reranking entities",
    reranking_evidence: "Reranking evidence",
    reranking_communities: "Reranking communities",
  } as const;
  return (update) => {
    onProgress({ phase, message: labels[phase], detail: rerankProgressDetail(noun, update) });
  };
}

async function rerankWithLlm<T>(args: {
  items: T[];
  buildPrompt: (item: T) => { system: string; user: string };
  tiebreak: (item: T) => number;
  onProgress?: (update: RerankProgressUpdate) => void;
}): Promise<Reranked<T>[]> {
  const model = resolveAskModel();
  if (!model || args.items.length === 0) {
    return args.items as Reranked<T>[];
  }
  let completed = 0;
  let failed = 0;
  const report = () => args.onProgress?.({ completed, failed, total: args.items.length });
  report();
  const scored = await mapLimit(args.items, LLM_RERANK_CONCURRENCY, async (item) => {
    const prompt = args.buildPrompt(item);
    try {
      const result = await generateText({
        model: model.languageModel,
        output: Output.object({ schema: RelevanceSchema }),
        ...model.structuredOptions,
        temperature: 0,
        abortSignal: AbortSignal.timeout(RERANK_LLM_CALL_TIMEOUT_MS),
        instructions: prompt.system,
        prompt: prompt.user,
      });
      completed += 1;
      report();
      return { ...item, relevanceScore: result.output?.score };
    } catch (error) {
      // The item keeps its vector order. Say why once per batch rather than
      // never: a call the SDK rejects outright (not a timeout) means every
      // item falls back, and that has to be visible in the server log.
      if (failed === 0) {
        console.error(`Rerank: the model did not score an item: ${errorMessage(error)}`);
      }
      completed += 1;
      failed += 1;
      report();
      return { ...item, relevanceScore: undefined };
    }
  });
  return scored.sort((left, right) => {
    const leftScore = left.relevanceScore ?? -1;
    const rightScore = right.relevanceScore ?? -1;
    if (leftScore !== rightScore) {
      return rightScore - leftScore;
    }
    return args.tiebreak(left) - args.tiebreak(right);
  });
}

function stripScore<T extends { relevanceScore?: number }>(item: T): Omit<T, "relevanceScore"> {
  const { relevanceScore: _ignored, ...rest } = item;
  return rest;
}

async function mapLimit<T, R>(items: T[], concurrency: number, mapper: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      out[index] = await mapper(items[index]!);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
  return out;
}
