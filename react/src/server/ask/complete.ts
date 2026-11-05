import "server-only";

import { ASK_STREAM_TIMEOUT_MS } from "./constants";
import { createAskAgent } from "./agent";
import { answerFromGrounding, groundingFromGenerateSteps } from "./grounding";
import { lexicalAsk } from "./lexical";
import { probeAskModel } from "./model";
import { mergeAskScope, isScoped } from "./scope";
import type { AskAnswer, AskScope, QueryMode } from "./types";
import type { GraphDataset } from "../protocol/schema";

export async function completeAsk(args: {
  dataset: GraphDataset;
  question: string;
  mode?: QueryMode;
  scope?: AskScope;
  documentIds?: string[];
  topK?: number;
  abortSignal?: AbortSignal;
}): Promise<AskAnswer> {
  const scope = mergeAskScope(
    args.scope,
    args.documentIds?.length ? { documentIds: args.documentIds } : undefined,
  );
  const model = await probeAskModel();
  if (!model) {
    return lexicalAsk(args.dataset, args.question, args.mode === "global" ? "global" : "local", scope);
  }
  const agent = createAskAgent({
    dataset: args.dataset,
    scope,
    preferredMode: args.mode,
    defaultTopK: args.topK,
  });
  const result = await agent.generate({
    prompt: args.question,
    abortSignal: args.abortSignal,
    timeout: ASK_STREAM_TIMEOUT_MS,
  });
  const grounding = groundingFromGenerateSteps(result.steps);
  const summary = result.text.trim() || "No answer was generated from this graph.";
  return answerFromGrounding({
    question: args.question,
    mode: args.mode ?? grounding.mode ?? "hybrid",
    summary,
    scoped: isScoped(scope),
    billed: model.billed,
    grounding,
  });
}
