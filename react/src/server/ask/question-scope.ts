import { generateText, Output } from "ai";
import { z } from "zod";
import { errorMessage } from "./errors";
import { resolveAskModel } from "./model";

/**
 * What a question asks about, decided once before any search.
 *
 * Whether an answer must account for many documents is not something to leave
 * to the answering model: left to choose, it reads them one by one in about a
 * third of the questions that need it and summarises a few top search hits in
 * the rest. A short structured call settles it up front, and also lists what
 * the question takes for granted, so each document can be checked against
 * those assumptions and a false one corrected rather than answered.
 */

export const QUESTION_SCOPES = ["one_document", "several_documents", "collection", "other"] as const;
export type QuestionScopeKind = (typeof QUESTION_SCOPES)[number];

export interface QuestionScope {
  scope: QuestionScopeKind;
  /** What the question presupposes to be true, each as a short statement. */
  assumptions: string[];
  /** Whether the question counts things (how many, which and how many). */
  counts: boolean;
}

export const QUESTION_SCOPE_TIMEOUT_MS = 30_000;
const QUESTION_SCOPE_MAX_OUTPUT_TOKENS = 400;
const MAX_ASSUMPTIONS = 5;
const MAX_ASSUMPTION_CHARACTERS = 240;

const ScopeSchema = z.object({
  scope: z.enum(QUESTION_SCOPES),
  assumptions: z.array(z.string().max(MAX_ASSUMPTION_CHARACTERS)).max(MAX_ASSUMPTIONS),
  counts: z.boolean(),
});

const SYSTEM = `You classify one question asked of a collection of documents, before it is answered.

scope:
- one_document: it is about one document, study, report or case, named or described.
- several_documents: the answer must combine or compare what several documents say - across studies, markets, waves, products or years - or list the ones that say something.
- collection: it is about the collection as a whole - which documents cover something, how many do, what they have in common.
- other: anything else, such as a definition.

assumptions: what the question takes for granted about the documents' content - that a study exists, included some group, found some result, covered some place or time - each as one short plain statement. Leave out what it merely asks. Return none when it assumes nothing specific.

counts: true when the question asks how many, or for a number of things to be listed and counted.`;

/** Classify a question; a failure answers "other", which changes nothing downstream. */
export async function classifyQuestion(question: string, signal?: AbortSignal): Promise<QuestionScope> {
  const fallback: QuestionScope = { scope: "other", assumptions: [], counts: false };
  const model = resolveAskModel();
  if (!model || !question.trim()) {
    return fallback;
  }
  const timeout = AbortSignal.timeout(QUESTION_SCOPE_TIMEOUT_MS);
  try {
    const result = await generateText({
      model: model.languageModel,
      output: Output.object({ schema: ScopeSchema }),
      ...model.structuredOptions,
      temperature: 0,
      maxOutputTokens: QUESTION_SCOPE_MAX_OUTPUT_TOKENS,
      abortSignal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      instructions: SYSTEM,
      prompt: question,
    });
    const output = result.output;
    if (!output) {
      return fallback;
    }
    return {
      scope: output.scope,
      assumptions: output.assumptions.map((item) => item.trim()).filter(Boolean),
      counts: output.counts,
    };
  } catch (error) {
    console.error(`Ask: the question could not be classified: ${errorMessage(error)}`);
    return fallback;
  }
}

/** Whether an answer must account for document after document. */
export function spansDocuments(scope: QuestionScope): boolean {
  return scope.scope === "several_documents" || scope.scope === "collection";
}

/** What the agent is told about the question before it starts. */
export function scopeNote(scope: QuestionScope): string | null {
  const lines: string[] = [];
  if (scope.assumptions.length) {
    lines.push(
      "This question takes for granted:",
      ...scope.assumptions.map((item) => `- ${item}`),
      "Check each against what the documents say. If one is contradicted or not supported, say so in your first sentence.",
    );
  }
  if (spansDocuments(scope)) {
    lines.push("The answer must account for every document that bears on the question: read them one by one first.");
  }
  if (scope.counts) {
    lines.push("The question counts things: list each one you count, and say whether you counted documents or folders.");
  }
  return lines.length ? lines.join("\n") : null;
}
