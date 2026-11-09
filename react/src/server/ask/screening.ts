import { generateText, Output } from "ai";
import { z } from "zod";
import type { CatalogueCard } from "./catalogue";
import { errorMessage } from "./errors";
import { resolveAskModel } from "./model";

/**
 * Decide from their catalogue cards which documents a question spanning many
 * of them should read.
 *
 * Reading every document of a collection one by one finds what each says, but
 * a document read from a handful of passages can still look silent when the
 * part that matters is not among them; reading all of them deeply costs a long
 * prefill per document. A card says what a document is, covers and finds, so a
 * few short calls over batches of cards settle which documents could bear on
 * the question, and only those are read - each from many more passages.
 *
 * The screen keeps what it is unsure of: a document is dropped only when its
 * card shows it is about something else. A batch that fails keeps all of its
 * documents, and a document without a card is always read.
 */

/** Fewer candidates than this are read without a screen: it would save little. */
export const SCREEN_MIN_DOCUMENTS = 12;
/** Cards one call judges. */
export const SCREEN_BATCH_SIZE = 10;
/** How much of one card the screen reads. */
export const SCREEN_CARD_CHAR_LIMIT = 3_200;
export const SCREEN_CONCURRENCY = 8;
export const SCREEN_CALL_TIMEOUT_MS = 60_000;
const SCREEN_MAX_OUTPUT_TOKENS = 600;

export const SCREEN_VERDICTS = ["bears", "may_bear", "unrelated"] as const;
export type ScreenVerdict = (typeof SCREEN_VERDICTS)[number];

export interface ScreenResult {
  /** The documents to read, in the order given. */
  kept: CatalogueCard[];
  /** The documents whose cards show they are about something else. */
  dropped: CatalogueCard[];
  /** Batches whose call failed; their documents are kept. */
  failedBatches: number;
}

const VerdictSchema = z.object({
  documents: z
    .array(z.object({ number: z.number().int(), verdict: z.enum(SCREEN_VERDICTS) }))
    .max(SCREEN_BATCH_SIZE * 2),
});

const SYSTEM = `You decide, from catalogue cards, which documents could say something on one question. Each card gives a document's title, type, subject, period, places, organizations, topics, method and key findings; it is a summary, so a document can cover more than its card names.

For each numbered document give one verdict:
- bears: the card shows the document addresses the question.
- may_bear: the card does not settle it, but the document's subject, products, places, people asked or methods are close enough that it could.
- unrelated: the card shows the document is about something else entirely.

When unsure, answer may_bear. Judge every number given.`;

/** Render a batch of cards, numbered from 1, for one call. */
export function screenPrompt(question: string, cards: readonly CatalogueCard[], assumptions: readonly string[] = []): string {
  const parts = [`Question: ${question}`];
  if (assumptions.length) {
    parts.push(`The question takes for granted:\n${assumptions.map((item) => `- ${item}`).join("\n")}`);
  }
  cards.forEach((card, position) => {
    const lines = [`[${position + 1}] ${card.name}`];
    if (card.folder) {
      lines.push(`Folder: ${card.folder}`);
    }
    lines.push((card.card ?? "").slice(0, SCREEN_CARD_CHAR_LIMIT));
    parts.push(lines.join("\n"));
  });
  return parts.join("\n\n");
}

/**
 * Screen the candidates of a question by their cards. Documents in
 * `alwaysRead` - those holding the question's best passages, say - are kept
 * whatever their card shows, and so is every document without one.
 */
export async function screenDocuments(
  question: string,
  candidates: readonly CatalogueCard[],
  options: {
    assumptions?: readonly string[];
    alwaysRead?: ReadonlySet<string>;
    signal?: AbortSignal;
  } = {},
): Promise<ScreenResult> {
  const model = resolveAskModel();
  const judged = candidates.filter((card) => card.card && !options.alwaysRead?.has(card.documentId));
  if (!model || candidates.length < SCREEN_MIN_DOCUMENTS || judged.length === 0) {
    return { kept: [...candidates], dropped: [], failedBatches: 0 };
  }
  const batches: CatalogueCard[][] = [];
  for (let start = 0; start < judged.length; start += SCREEN_BATCH_SIZE) {
    batches.push(judged.slice(start, start + SCREEN_BATCH_SIZE));
  }
  const unrelated = new Set<string>();
  let failedBatches = 0;
  let next = 0;
  const worker = async () => {
    while (next < batches.length) {
      const batch = batches[next++]!;
      const verdicts = await judgeBatch(question, batch, model, options);
      if (!verdicts) {
        failedBatches += 1;
        continue;
      }
      for (const [position, verdict] of verdicts) {
        const card = batch[position];
        if (card && verdict === "unrelated") {
          unrelated.add(card.documentId);
        }
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(SCREEN_CONCURRENCY, batches.length) }, worker));
  return {
    kept: candidates.filter((card) => !unrelated.has(card.documentId)),
    dropped: candidates.filter((card) => unrelated.has(card.documentId)),
    failedBatches,
  };
}

/** One batch's verdicts by position; a document the model leaves out stays unjudged and is read. */
async function judgeBatch(
  question: string,
  batch: readonly CatalogueCard[],
  model: NonNullable<ReturnType<typeof resolveAskModel>>,
  options: { assumptions?: readonly string[]; signal?: AbortSignal },
): Promise<Map<number, ScreenVerdict> | null> {
  const timeout = AbortSignal.timeout(SCREEN_CALL_TIMEOUT_MS);
  try {
    const result = await generateText({
      model: model.languageModel,
      output: Output.object({ schema: VerdictSchema }),
      ...model.structuredOptions,
      temperature: 0,
      maxOutputTokens: SCREEN_MAX_OUTPUT_TOKENS,
      abortSignal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
      instructions: SYSTEM,
      prompt: screenPrompt(question, batch, options.assumptions),
    });
    if (!result.output) {
      return null;
    }
    const verdicts = new Map<number, ScreenVerdict>();
    for (const item of result.output.documents) {
      const position = item.number - 1;
      if (position >= 0 && position < batch.length) {
        verdicts.set(position, item.verdict);
      }
    }
    return verdicts;
  } catch (error) {
    if (options.signal?.aborted) {
      throw error;
    }
    console.error(`Ask: a batch of document cards could not be screened: ${errorMessage(error)}`);
    return null;
  }
}
