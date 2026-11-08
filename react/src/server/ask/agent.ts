import "server-only";

import { isStepCount, ToolLoopAgent } from "ai";
import { AGENT_MAX_STEPS, ASK_ANSWER_BY_MS, ASK_ANSWER_MAX_OUTPUT_TOKENS } from "./constants";
import { AskModelMissingError } from "./errors";
import { budgetedTools, EvidenceBudget } from "./evidence-budget";
import { resolveAskModel } from "./model";
import { createAskTools, type AskToolSession } from "./tools";

export { AskModelMissingError } from "./errors";

const ANSWER_NOW =
  "Stop searching and write your final answer now from what the tools returned. Say plainly what you could not find or check.";

export function createAskAgent(session: AskToolSession) {
  const model = resolveAskModel();
  if (!model) {
    throw new AskModelMissingError();
  }
  const budget = new EvidenceBudget();
  const tools = budgetedTools(createAskTools(session), budget);
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
- If the question assumes something the documents contradict or do not support - a study, a result, a number, a comparison - say so plainly before anything else.
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
    prepareStep: ({ stepNumber, messages }) => {
      const lastStep = stepNumber >= AGENT_MAX_STEPS - 1;
      if (!lastStep && !budget.spent && Date.now() - startedAt < ASK_ANSWER_BY_MS) {
        return undefined;
      }
      return { toolChoice: "none", messages: [...messages, { role: "user", content: ANSWER_NOW }] };
    },
  });
}
