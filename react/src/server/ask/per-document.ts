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
/**
 * Calls in flight at once. Each is short - a few thousand tokens in, a few
 * hundred out - so this is what decides how long a question across many
 * documents takes: enough to keep several engines busy, and still a share of
 * what a serving fleet admits at once.
 */
export const PER_DOCUMENT_CONCURRENCY = 16;
/** Passages each document is read from. */
export const PER_DOCUMENT_PASSAGES = 10;
/** How much of one document's passages and card a call reads. */
export const PER_DOCUMENT_CONTEXT_CHAR_LIMIT = 14_000;
export const PER_DOCUMENT_MAX_OUTPUT_TOKENS = 1_600;
export const PER_DOCUMENT_CALL_TIMEOUT_MS = 90_000;
const MAX_FACTS = 10;
const MAX_FACT_CHARACTERS = 400;
const MAX_QUOTE_CHARACTERS = 300;
const MAX_ANSWER_CHARACTERS = 900;
const MAX_ISSUES = 5;

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
  /** The folder the document sits in: documents of one folder are usually one study, project or case. */
  group: string | null;
  passages: DocumentPassage[];
}

/** An assumption of the question that a document contradicts or leaves unsupported where it should state it. */
export interface PremiseIssue {
  assumption: string;
  verdict: "contradicted" | "unsupported";
  why: string;
}

export interface DocumentFinding {
  documentId: string;
  documentName: string;
  group?: string | null;
  relevant: boolean;
  answer: string;
  /** Each fact with the exact words of the document it rests on. */
  facts: { statement: string; quote: string; page: number | null }[];
  premiseIssues?: PremiseIssue[];
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
        quote: z.string().max(MAX_QUOTE_CHARACTERS),
        page: z.number().int().nullable(),
      }),
    )
    .max(MAX_FACTS),
  premise_issues: z
    .array(
      z.object({
        assumption: z.string().max(MAX_FACT_CHARACTERS),
        verdict: z.enum(["contradicted", "unsupported"]),
        why: z.string().max(MAX_FACT_CHARACTERS),
      }),
    )
    .max(MAX_ISSUES),
});

const SYSTEM = `You read one document for one question and report only what that document states that bears on it.

- relevant is true when the document says anything that answers or bears on the question, false when it says nothing about it.
- answer is what this document says on the question, in one to three sentences, keeping its numbers, units, dates, places, sample sizes and qualifiers exactly as written. Leave it empty when relevant is false.
- facts are the specific statements behind the answer - each with quote, the exact words of the passage or card it rests on (copied, not paraphrased), and the page it is on (null when not given). Prefer facts with numbers, and give every number the document states on the question.
- premise_issues: the question may take things for granted (listed as assumptions). For each assumption this document is about, say if the document contradicts it, or does not support it where it is the document that should state it, and why in one sentence. Leave it empty when the document agrees or does not concern the assumption.
- Use only the card and passages given. Never use outside knowledge, and never infer what the document does not say.`;

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
  options: {
    concurrency?: number;
    signal?: AbortSignal;
    onProgress?: (done: number, total: number) => void;
    /** What the question takes for granted, checked against each document. */
    assumptions?: readonly string[];
  } = {},
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
      results[index] = await readOne(question, source, model, options.signal, options.assumptions ?? []);
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
  assumptions: readonly string[],
): Promise<DocumentFinding> {
  const base = { documentId: source.documentId, documentName: source.documentName, group: source.group };
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
      prompt: [
        `Question: ${question}`,
        assumptions.length ? `Assumptions of the question:\n${assumptions.map((item) => `- ${item}`).join("\n")}` : "",
        documentContext(source),
      ]
        .filter(Boolean)
        .join("\n\n"),
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
      premiseIssues: output.premise_issues.filter((issue) => issue.assumption.trim()),
    };
  } catch (error) {
    return { ...base, relevant: false, answer: "", facts: [], failure: errorMessage(error) };
  }
}

/**
 * Findings as the agent reads them.
 *
 * Problems with the question's assumptions come first, because an answer must
 * open with them. Relevant documents follow, grouped by the folder they sit in
 * when several share one - such documents are usually one study, project or
 * case, and a count of those is a count of folders, not files - and each fact
 * carries the document's own words, so numbers are not carried from memory.
 */
export function renderFindings(findings: DocumentFinding[]): string {
  const relevant = findings.filter((finding) => finding.relevant);
  const unread = findings.filter((finding) => finding.failure);
  const silent = findings.filter((finding) => !finding.relevant && !finding.failure);
  const groups = new Map<string, DocumentFinding[]>();
  for (const finding of relevant) {
    const key = finding.group?.trim() || finding.documentName;
    groups.set(key, [...(groups.get(key) ?? []), finding]);
  }
  const grouped = [...groups.values()].some((members) => members.length > 1);
  const lines: string[] = [
    `Read ${findings.length} documents one by one: ${relevant.length} bear on the question` +
      (grouped ? `, in ${groups.size} folders` : "") +
      `, ${silent.length} do not${unread.length ? `, ${unread.length} could not be read` : ""}. ` +
      "Every document that bears on it is listed below; answer from these findings and account for each.",
  ];
  const issues = findings.flatMap((finding) =>
    (finding.premiseIssues ?? []).map((issue) => ({ ...issue, documentName: finding.documentName })),
  );
  if (issues.length) {
    lines.push("", "## Problems with the question's assumptions - open the answer with these");
    for (const issue of issues) {
      lines.push(`- ${issue.verdict}: "${issue.assumption}" - ${issue.why} (${issue.documentName})`);
    }
  }
  for (const [group, members] of groups) {
    if (grouped) {
      lines.push("", `## Folder: ${group} (${members.length} document${members.length === 1 ? "" : "s"})`);
    }
    for (const finding of members) {
      lines.push("", `${grouped ? "###" : "##"} ${finding.documentName} (${finding.documentId})`, finding.answer);
      for (const fact of finding.facts) {
        const page = fact.page != null ? `[p${fact.page}] ` : "";
        const quote = fact.quote.trim() ? ` - "${fact.quote.trim()}"` : "";
        lines.push(`- ${page}${fact.statement}${quote}`);
      }
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
