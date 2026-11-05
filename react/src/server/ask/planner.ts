import { generateText, Output } from "ai";
import { z } from "zod";
import { dedupeKeywords, tokenize } from "./score";
import type { QueryMode, QueryRetrievalPlan } from "./types";
import { resolveAskModel } from "./model";
import { errorMessage } from "./errors";

const QueryPlanSchema = z.object({
  mode: z.enum(["local", "global", "hybrid", "drift"]),
  highLevelKeywords: z.array(z.string().min(1).max(80)).max(8),
  lowLevelKeywords: z.array(z.string().min(1).max(80)).max(8),
});

export function entityQueryText(plan: QueryRetrievalPlan, query: string): string {
  return plan.lowLevelKeywords.length > 0 ? plan.lowLevelKeywords.join(", ") : query;
}

export function communityQueryText(plan: QueryRetrievalPlan, query: string): string {
  return plan.highLevelKeywords.length > 0 ? plan.highLevelKeywords.join(", ") : query;
}

export function lexicalPlan(query: string, mode: QueryMode = "hybrid"): QueryRetrievalPlan {
  return {
    mode,
    highLevelKeywords: [],
    lowLevelKeywords: tokenize(query),
  };
}

export async function planQueryRetrieval(query: string): Promise<QueryRetrievalPlan> {
  const trimmed = query.trim();
  if (!trimmed) {
    return { mode: "hybrid", highLevelKeywords: [], lowLevelKeywords: [] };
  }
  const model = resolveAskModel();
  if (!model) {
    return lexicalPlan(trimmed);
  }
  try {
    const result = await generateText({
      model: model.languageModel,
      output: Output.object({ schema: QueryPlanSchema }),
      ...model.structuredOptions,
      instructions: `You plan retrieval for a graph-based RAG system. Given a user question, return JSON with:

- mode: "local" for questions about specific named entities / relationships; "global" for overarching themes, summaries, and cross-document patterns; "hybrid" when the question mixes both.
- "drift" for multi-hop exploratory questions that need a high-level primer and targeted follow-up retrieval (e.g. "how did X evolve and what influenced it?", "compare and contrast strategy A vs B across teams"). Only pick drift when both breadth and depth matter.
- highLevelKeywords: 1-6 abstract topic/theme keywords (e.g. "pricing strategy", "security posture"). These retrieve community-level context.
- lowLevelKeywords: 1-6 specific entity-like keywords (e.g. "GPT-4o", "Acme Q3 contract", "billing service"). These retrieve entity- and relation-level context.

Return empty arrays for a side when no keywords of that kind exist. Never invent entities. Keep each keyword short (ideally 1-4 words) and grounded in the question.`,
      prompt: `Question: ${trimmed}`,
    });
    const object = result.output;
    if (!object) {
      return lexicalPlan(trimmed);
    }
    return {
      mode: object.mode,
      highLevelKeywords: dedupeKeywords(object.highLevelKeywords),
      lowLevelKeywords: dedupeKeywords(object.lowLevelKeywords),
    };
  } catch (error) {
    // The lexical plan still retrieves; a model that is configured but
    // refuses every call must not do so invisibly.
    console.error(`Query planner: the model did not plan the question: ${errorMessage(error)}`);
    return lexicalPlan(trimmed);
  }
}
