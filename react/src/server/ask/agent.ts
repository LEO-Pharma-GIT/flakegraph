import "server-only";

import { isStepCount, ToolLoopAgent } from "ai";
import { AGENT_MAX_STEPS, ASK_ANSWER_BY_MS, ASK_ANSWER_MAX_OUTPUT_TOKENS } from "./constants";
import { AskModelMissingError } from "./errors";
import { budgetedTools, EvidenceBudget } from "./evidence-budget";
import { resolveAskModel } from "./model";
import { classifyQuestion, scopeNote, spansDocuments } from "./question-scope";
import { createAskTools, type AskQuestionContext, type AskToolSession } from "./tools";

export { AskModelMissingError } from "./errors";

const ANSWER_NOW =
  "Stop searching and write your final answer now from what the tools returned. Say plainly what you could not find or check.";

export function createAskAgent(session: AskToolSession) {
  const model = resolveAskModel();
  if (!model) {
    throw new AskModelMissingError();
  }
  const budget = new EvidenceBudget();
  const question: AskQuestionContext = {};
  const tools = budgetedTools(createAskTools({ ...session, question }), budget);
  const startedAt = Date.now();
  const preferred = session.preferredMode
    ? `The operator selected retrieval mode "${session.preferredMode}". Pass that mode to searchGraph unless a different level is clearly required.`
    : "When you use searchGraph, omit mode so the planner can choose.";
  return new ToolLoopAgent({
    model: model.languageModel,
    instructions: `You answer questions about one FlakeGraph knowledge graph built from documents. Use tools; do not invent entities, relations, quotes or numbers.

Tools:
- searchPassages: the documents' own text, tables and charts included. Use it first for any fact, number, date, name or quote.
- findDocuments: the catalogue of documents with a card for each (what it is, covers and finds). Use it for questions about the documents themselves and to see which documents a question involves.
- readEachDocument: reads several documents one by one for the same question. Use it whenever the answer spans documents: comparisons, "which", "how many", "all" or "every" questions, or a summary across them.
- searchGraph, searchEntities, searchRelations, getNeighborhood, searchCommunities: the extracted entities, relations and community themes.
- listDocuments, readDocumentEvidence, getEntityEvidence, getRelationEvidence: document ids and verbatim quotes for a known id.

Rules:
- Call at least one retrieval tool before answering. If a search finds nothing, search again with other words before concluding the graph does not say.
- If the user names a document, find its exact id with findDocuments or listDocuments and pass it in documentIds. Do not guess ids.
- For an answer that spans documents, account for every document that bears on it: name each one and what it says. Count by listing, never by estimate.
- Give numbers exactly as the documents state them, with their units, base and period. A chart value marked as estimated from an image is approximate; say so.
- If the question assumes something the documents contradict or do not support - a study, a result, a number, a comparison - say so plainly in your first sentence, then answer only what the documents do support. Never answer as if a false assumption were true. An assumption about what the collection holds stands when any document supports it: one document that does not mention it does not contradict it.
- Documents that sit in one folder are usually one study, project or case. When a question counts studies, projects or cases, count folders, not files.
- When a question counts, ranks or asks which studies, reports or organisations qualify: first state the rule you count by, in the question's own terms and no narrower - never add a condition the question does not set. Then go through every candidate by name, a few words each on whether it qualifies and why, and give the total from that list. Count only things of the kind asked for: a report the collection holds, not one a document cites; an organisation in the role asked about, not the client or others it mentions.
- If the documents do not contain the answer, say it is not in this graph. Never answer from general knowledge.
- Cite only what tools returned. Name entities exactly as returned. Keep answers concise.
- Write only the final answer as text. Do not narrate your searches or plans.
- A relation marked claimed is one its source claims or proposes, not an established finding; say so when your answer uses it.
- ${preferred}`,
    tools,
    maxOutputTokens: ASK_ANSWER_MAX_OUTPUT_TOKENS,
    stopWhen: isStepCount(AGENT_MAX_STEPS),
    // A question that runs out of steps, evidence or time still gets an answer
    // from what was found. The instruction goes last so the prompt keeps the
    // prefix the engine may already hold.
    prepareStep: async ({ stepNumber, messages }) => {
      if (stepNumber === 0) {
        // Decided once, before any search: whether the answer must account
        // for document after document, and what the question takes for
        // granted. Such a question starts by reading the documents one by one.
        question.scope = await classifyQuestion(lastUserText(messages));
        const note = scopeNote(question.scope);
        if (spansDocuments(question.scope)) {
          return {
            toolChoice: { type: "tool", toolName: "readEachDocument" },
            ...(note ? { messages: [...messages, { role: "user", content: note }] } : {}),
          };
        }
        return note ? { messages: [...messages, { role: "user", content: note }] } : undefined;
      }
      const lastStep = stepNumber >= AGENT_MAX_STEPS - 1;
      if (!lastStep && !budget.spent && Date.now() - startedAt < ASK_ANSWER_BY_MS) {
        return undefined;
      }
      return { toolChoice: "none", messages: [...messages, { role: "user", content: ANSWER_NOW }] };
    },
  });
}


/** The text of the last thing the user said, which is the question being answered. */
function lastUserText(messages: readonly { role: string; content: unknown }[]): string {
  const last = [...messages].reverse().find((message) => message.role === "user");
  if (!last) {
    return "";
  }
  if (typeof last.content === "string") {
    return last.content;
  }
  return Array.isArray(last.content)
    ? last.content
        .map((part) => (part && typeof part === "object" && "text" in part ? String(part.text) : ""))
        .join(" ")
    : "";
}
