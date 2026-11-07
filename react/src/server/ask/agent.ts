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
    : "If the operator did not pick a mode, call searchGraph without mode so the planner can choose.";
  return new ToolLoopAgent({
    model: model.languageModel,
    instructions: `You answer questions about one FlakeGraph knowledge graph. Use tools; do not invent entities, relations, or quotes.

Retrieval levels:
- searchEntities / getNeighborhood / searchRelations / searchEvidence for local facts
- searchCommunities for themes and summaries
- searchGraph when you want the planner to pick local, global, hybrid, or drift
- listDocuments, then pass exact document ids to searchGraph, when the user names a file
- getEntityEvidence / getRelationEvidence / readDocumentEvidence for verbatim quotes on a known id

Rules:
- Call at least one retrieval tool before answering.
- Prefer searchGraph for the first hop unless the user already named a specific entity id.
- If the user names a document, call listDocuments and pass that document's exact id in documentIds. Do not guess ids.
- Cite only quotes returned by tools. If tools return GRAPH_EMPTY_RESULT, say you could not find it in this graph.
- Name entities exactly as returned. Keep answers concise.
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
