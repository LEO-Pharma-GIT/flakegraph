import "server-only";

import { isStepCount, ToolLoopAgent } from "ai";
import { AGENT_MAX_STEPS } from "./constants";
import { AskModelMissingError } from "./errors";
import { resolveAskModel } from "./model";
import { createAskTools, type AskToolSession } from "./tools";

export { AskModelMissingError } from "./errors";

export function createAskAgent(session: AskToolSession) {
  const model = resolveAskModel();
  if (!model) {
    throw new AskModelMissingError();
  }
  const tools = createAskTools(session);
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
- A relation marked claimed is one its source claims or proposes, not an established finding; say so when your answer uses it.
- ${preferred}`,
    tools,
    stopWhen: isStepCount(AGENT_MAX_STEPS),
  });
}
