import { generateText, Output } from "ai";
import { z } from "zod";
import { errorMessage } from "./errors";
import { resolveAskModel } from "./model";

/**
 * Read each document on its own, then answer from what every one of them says.
 *
 * A question across many documents - which of them found X, how they compare,
 * how many did Y - fails when one search gathers passages for all of them at
 * once: the documents that rank highest crowd out the rest, and the answer
 * summarises a few of them as if they were all. Here every candidate document
 * is read separately, with its own passages and catalogue card, by parallel
 * calls, and reports what it says on the question or that it says nothing.
 * The agent then writes one answer that has to account for each of them.
 */

/** The most documents one question reads separately. */
export const PER_DOCUMENT_MAX_DOCUMENTS = 64;
/** Calls in flight at once: enough to keep a fleet of engines busy, few enough to share it. */
export const PER_DOCUMENT_CONCURRENCY = 8;
/** Passages each document is read from. */
export const PER_DOCUMENT_PASSAGES = 10;
/** How much of one document's passages and card a call reads. */
export const PER_DOCUMENT_CONTEXT_CHAR_LIMIT = 14_000;
export const PER_DOCUMENT_MAX_OUTPUT_TOKENS = 1_200;
export const PER_DOCUMENT_CALL_TIMEOUT_MS = 90_000;
const MAX_FACTS = 8;
const MAX_FACT_CHARACTERS = 400;
const MAX_ANSWER_CHARACTERS = 900;

export interface DocumentPassage {
  page: number;
  sectionPath: string[];
  text: string;
}

export interface PerDocumentSource {
  documentId: string;
  documentName: string;
  /** The document's catalogue card, when the graph has one. */
  card: string | null;
  passages: DocumentPassage[];
}

export interface DocumentFinding {
  documentId: string;
  documentName: string;
  relevant: boolean;
  answer: string;
  facts: { statement: string; page: number | null }[];
  /** Why the document could not be read, when it could not. */
  failure?: string;
}

const FindingSchema = z.object({
  relevant: z.boolean(),
  answer: z.string().max(MAX_ANSWER_CHARACTERS),
  facts: z
    .array(
      z.object({
        statement: z.string().max(MAX_FACT_CHARACTERS),
        page: z.number().int().nullable(),
      }),
    )
    .max(MAX_FACTS),
});

const SYSTEM = `You read one document for one question and report only what that document states that bears on it.

- relevant is true when the document says anything that answers or bears on the question, false when it says nothing about it.
- answer is what this document says on the question, in one to three sentences, keeping its numbers, units, dates, places and qualifiers exactly as written. Leave it empty when relevant is false.
- facts are the specific statements behind the answer, each with the page it is on (null when the page is not given).
- Use only the card and passages given. Never use outside knowledge, and never infer what the document does not say.
- If the question assumes something this document contradicts, say so in answer.`;

/** Render one document's card and passages within the context bound. */
export function documentContext(source: PerDocumentSource, limit = PER_DOCUMENT_CONTEXT_CHAR_LIMIT): string {
  const parts: string[] = [`Document: ${source.documentName}`];
  if (source.card) {
    parts.push(`Card:\n${source.card}`);
  }
  let used = parts.join("\n\n").length;
  for (const passage of source.passages) {
    const heading = passage.sectionPath.length ? ` ${passage.sectionPath.join(" > ")}` : "";
    const block = `[p${passage.page}]${heading}\n${passage.text}`;
    if (used + block.length + 2 > limit) {
      const room = limit - used - 2;
      if (room > 200) {
        parts.push(`${block.slice(0, room)} …`);
      }
      break;
    }
    parts.push(block);
    used += block.length + 2;
  }
  return parts.join("\n\n");
}

/**
 * Read every source for the question, in parallel, and return their findings
 * in the order given. A document whose call fails is reported as unread
 * rather than left out, so the answer can say it could not check it.
 */
export async function readEachDocument(
  question: string,
  sources: PerDocumentSource[],
  options: { concurrency?: number; signal?: AbortSignal; onProgress?: (done: number, total: number) => void } = {},
): Promise<DocumentFinding[]> {
  const model = resolveAskModel();
  if (!model) {
    throw new Error("Reading documents one by one needs the console's model.");
  }
  const results: DocumentFinding[] = new Array(sources.length);
  let next = 0;
  let done = 0;
  const worker = async () => {
    while (next < sources.length) {
      const index = next++;
      const source = sources[index]!;
      results[index] = await readOne(question, source, model, options.signal);
      done += 1;
      options.onProgress?.(done, sources.length);
    }
  };
  const width = Math.max(1, Math.min(options.concurrency ?? PER_DOCUMENT_CONCURRENCY, sources.length));
  await Promise.all(Array.from({ length: width }, worker));
  return results;
}

async function readOne(
  question: string,
  source: PerDocumentSource,
  model: NonNullable<ReturnType<typeof resolveAskModel>>,
  signal: AbortSignal | undefined,
): Promise<DocumentFinding> {
  const base = { documentId: source.documentId, documentName: source.documentName };
  if (!source.card && source.passages.length === 0) {
    return { ...base, relevant: false, answer: "", facts: [] };
  }
  const timeout = AbortSignal.timeout(PER_DOCUMENT_CALL_TIMEOUT_MS);
  try {
    const result = await generateText({
      model: model.languageModel,
      output: Output.object({ schema: FindingSchema }),
      ...model.structuredOptions,
      temperature: 0,
      maxOutputTokens: PER_DOCUMENT_MAX_OUTPUT_TOKENS,
      abortSignal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      instructions: SYSTEM,
      prompt: `Question: ${question}\n\n${documentContext(source)}`,
    });
    const output = result.output;
    if (!output) {
      return { ...base, relevant: false, answer: "", facts: [], failure: "the model returned nothing" };
    }
    return {
      ...base,
      relevant: output.relevant,
      answer: output.relevant ? output.answer.trim() : "",
      facts: output.relevant ? output.facts.filter((fact) => fact.statement.trim()) : [],
    };
  } catch (error) {
    return { ...base, relevant: false, answer: "", facts: [], failure: errorMessage(error) };
  }
}

/** Findings as the agent reads them: relevant documents first, then the rest by name. */
export function renderFindings(findings: DocumentFinding[]): string {
  const relevant = findings.filter((finding) => finding.relevant);
  const unread = findings.filter((finding) => finding.failure);
  const silent = findings.filter((finding) => !finding.relevant && !finding.failure);
  const lines: string[] = [
    `Read ${findings.length} documents one by one: ${relevant.length} bear on the question, ${silent.length} do not${unread.length ? `, ${unread.length} could not be read` : ""}.`,
  ];
  for (const finding of relevant) {
    lines.push("", `## ${finding.documentName} (${finding.documentId})`, finding.answer);
    for (const fact of finding.facts) {
      lines.push(`- ${fact.page != null ? `[p${fact.page}] ` : ""}${fact.statement}`);
    }
  }
  if (silent.length) {
    lines.push("", `Say nothing on it: ${silent.map((finding) => finding.documentName).join("; ")}`);
  }
  if (unread.length) {
    lines.push("", `Could not be read: ${unread.map((finding) => finding.documentName).join("; ")}`);
  }
  return lines.join("\n");
}
