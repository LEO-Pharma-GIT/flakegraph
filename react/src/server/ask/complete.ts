import "server-only";

import { ASK_STREAM_TIMEOUT_MS } from "./constants";
import { createAskAgent } from "./agent";
import { answerFromGrounding, groundingFromGenerateSteps } from "./grounding";
import { lexicalAsk } from "./lexical";
import { probeAskModel } from "./model";
import { renderStudies, studyFindingsFromOutput, type StudyFinding } from "./per-document";
import { mergeAskScope, isScoped } from "./scope";
import type { AskAnswer, AskScope, QueryMode } from "./types";
import type { GraphDataset } from "../protocol/schema";

/** One tool call starting, with its input, or ending, with its output. */
export interface AskToolEvent {
  name: string;
  state: "start" | "done";
  input?: unknown;
  output?: unknown;
}

function toolEventCallbacks(onToolEvent: (event: AskToolEvent) => void) {
  return {
    onToolExecutionStart: (event: { toolCall: { toolName: string; input: unknown } }) =>
      onToolEvent({ name: event.toolCall.toolName, state: "start", input: event.toolCall.input }),
    onToolExecutionEnd: (event: { toolCall: { toolName: string }; toolOutput: { type: string; output?: unknown; error?: unknown } }) =>
      onToolEvent({
        name: event.toolCall.toolName,
        state: "done",
        output: event.toolOutput.type === "tool-result" ? event.toolOutput.output : { error: String(event.toolOutput.error) },
      }),
  };
}

export async function completeAsk(args: {
  dataset: GraphDataset;
  question: string;
  mode?: QueryMode;
  scope?: AskScope;
  documentIds?: string[];
  topK?: number;
  abortSignal?: AbortSignal;
  /** Each tool call as it starts and ends, for tracing where an answer's time and evidence went. */
  onToolEvent?: (event: AskToolEvent) => void;
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
    ...(args.onToolEvent ? toolEventCallbacks(args.onToolEvent) : {}),
  });
  const grounding = groundingFromGenerateSteps(result.steps);
  // As the streamed answer does: every document read one by one that bears on
  // the question follows the answer with what it says.
  const studies = new Map<string, StudyFinding>();
  for (const step of result.steps) {
    for (const toolResult of step.toolResults) {
      for (const study of studyFindingsFromOutput(toolResult.output)) {
        studies.set(study.documentId, study);
      }
    }
  }
  const written = result.text.trim();
  const appendix = written ? renderStudies([...studies.values()]) : "";
  const summary = (appendix ? `${written}\n\n${appendix}` : written) || "No answer was generated from this graph.";
  return answerFromGrounding({
    question: args.question,
    mode: args.mode ?? grounding.mode ?? "hybrid",
    summary,
    scoped: isScoped(scope),
    billed: model.billed,
    grounding,
  });
}
