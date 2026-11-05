import "server-only";

import { generateText, Output } from "ai";
import { z } from "zod";
import { probeAskModel } from "./ask/model";
import { proposeOntology as heuristicProposal, type OntologyProposal } from "./workspace";
import { errorMessage } from "./ask/errors";

const ProposalSchema = z.object({
  entityTypes: z
    .array(
      z.object({
        name: z.string().regex(/^[A-Z][A-Z0-9_]{1,31}$/),
        description: z.string().max(160),
      }),
    )
    .min(3)
    .max(8),
  relationTypes: z
    .array(
      z.object({
        name: z.string().regex(/^[A-Z][A-Z0-9_]{1,31}$/),
        description: z.string().max(160),
      }),
    )
    .min(2)
    .max(6),
});

const SYSTEM = `You design the ontology for an evidence-grounded knowledge graph extracted from documents.
Given a description of the graph someone wants, propose the entity types and relation types an extractor should use.
- Names are UPPER_SNAKE_CASE nouns for entity types (PERSON, ORGANIZATION, TECHNIQUE) and UPPER_SNAKE_CASE predicates for relation types (FOUNDED_BY, PART_OF, LOCATED_IN).
- Prefer types a reader of the documents would recognise; keep them distinct and non-overlapping.
- Include DATE only when time matters to the description, and always include a general RELATED_TO relation.
- Descriptions are one short sentence each, stating what qualifies.`;

// A reasoning model behind the gateway takes tens of seconds to write a
// proposal; the button says it is working.
const PROPOSAL_TIMEOUT_MS = 90_000;

export interface OntologyProposalResult extends OntologyProposal {
  /** Whether a model wrote the proposal or the word heuristic did. */
  source: "model" | "heuristic";
  descriptions: Record<string, string>;
  /** Why the model did not answer, when one was configured and asked. */
  modelFailure: string | null;
}

/**
 * Propose an ontology for a described graph.
 *
 * The console's language model writes it when one is configured; otherwise
 * the proposal falls back to the words of the description, and says so.
 */
export async function proposeOntologyForIntent(
  intent: string,
  goldTypes: readonly string[] = [],
): Promise<OntologyProposalResult> {
  const model = await probeAskModel();
  let modelFailure: string | null = null;
  if (model) {
    try {
      const result = await generateText({
        model: model.languageModel,
        output: Output.object({ schema: ProposalSchema }),
        ...model.structuredOptions,
        temperature: 0,
        abortSignal: AbortSignal.timeout(PROPOSAL_TIMEOUT_MS),
        instructions: SYSTEM,
        prompt: `The graph I want: ${intent.trim()}`,
      });
      const output = result.output;
      if (output) {
        const types = unique(output.entityTypes.map((item) => item.name));
        const relations = unique(output.relationTypes.map((item) => item.name));
        const missing = goldTypes.filter((type) => !types.includes(type.toUpperCase()));
        return {
          intent,
          types,
          relations,
          descriptions: Object.fromEntries(
            [...output.entityTypes, ...output.relationTypes].map((item) => [item.name, item.description]),
          ),
          warning: missing.length
            ? `Proposed types are a coverage overlay. Gold still needs ${missing.join(", ")} before a full corpus run.`
            : "Proposed by the model from your description. Confirm before a full corpus run.",
          source: "model",
          modelFailure: null,
        };
      }
      modelFailure = "The model returned no proposal.";
    } catch (error) {
      // The heuristic below still answers; the caller sees which one did,
      // and why the model did not.
      modelFailure = error instanceof Error ? error.message : String(error);
      console.error(`Ontology proposal: the model did not answer: ${errorMessage(error)}`);
    }
  }
  return { ...heuristicProposal(intent, goldTypes), descriptions: {}, source: "heuristic", modelFailure };
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}
