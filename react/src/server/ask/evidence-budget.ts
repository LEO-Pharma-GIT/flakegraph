import type { Tool, ToolSet } from "ai";
import { ASK_EVIDENCE_CHAR_BUDGET, GRAPH_SEARCH_CONTEXT_CHAR_BUDGET } from "./constants";

const SPENT_NOTE = "\n\n[Left out: this question has used its evidence budget. Answer from what you have.]";

/**
 * How much tool text one question may put in front of the model.
 *
 * Every step resends everything before it, and an engine reads about 1,300
 * tokens a second, so each search makes every later step slower; past the
 * model's context window the question fails outright. Charging at execution
 * covers tools called in parallel, which no usage figure sees until after the
 * next prompt has been sent.
 */
export class EvidenceBudget {
  private left: number;
  private readonly views = new Map<string, string>();

  constructor(chars: number = ASK_EVIDENCE_CHAR_BUDGET) {
    this.left = chars;
  }

  charge(toolCallId: string, text: string): void {
    const allowed = Math.min(this.left, GRAPH_SEARCH_CONTEXT_CHAR_BUDGET);
    const view = text.length > allowed ? `${text.slice(0, allowed)}${SPENT_NOTE}` : text;
    this.left -= Math.min(text.length, allowed);
    this.views.set(toolCallId, view);
  }

  view(toolCallId: string): string | undefined {
    return this.views.get(toolCallId);
  }

  get spent(): boolean {
    return this.left <= 0;
  }
}

/**
 * A tool result as the model reads it. The output itself, with its citations
 * and full id lists, still reaches the caller; the model gets the same
 * evidence once, without the copies and ids it has no use for.
 */
export function renderForModel(output: unknown): string {
  const row = (output && typeof output === "object" ? output : {}) as Record<string, unknown>;
  const citations = Array.isArray(row.citations) ? (row.citations as ModelCitation[]) : [];
  if (typeof row.text === "string") {
    const names = documentNames(citations);
    return names.length > 0 ? `${row.text}\n\n## Documents\n${names.join("\n")}` : row.text;
  }
  if (Array.isArray(row.documents)) {
    return (row.documents as Array<{ id: string; title: string; quoteCount: number }>)
      .map((document) => `- ${document.id}: ${document.title} (${document.quoteCount} quotes)`)
      .join("\n");
  }
  if (citations.length > 0) {
    return renderCitations(citations);
  }
  return JSON.stringify(output);
}

interface ModelCitation {
  quote: string;
  documentId: string;
  documentName?: string;
  entityId?: string | null;
  entityName?: string | null;
}

function documentNames(citations: ModelCitation[]): string[] {
  const names = new Map(citations.filter((row) => row.documentName).map((row) => [row.documentId, row.documentName]));
  return [...names].map(([id, name]) => `- ${id}: ${name}`);
}

function renderCitations(citations: ModelCitation[]): string {
  const byDocument = Map.groupBy(citations, (citation) => citation.documentId);
  return [...byDocument]
    .map(([documentId, rows]) => {
      const heading = `## ${rows[0]?.documentName ?? "Document"} (${documentId})`;
      const quotes = rows.map((row) => {
        const entity = row.entityName ? `[${row.entityName}, ${row.entityId}] ` : "";
        return `- ${entity}${row.quote}`;
      });
      return [heading, ...quotes].join("\n");
    })
    .join("\n\n");
}

/** Charge every tool's result to the budget and show the model its own view of it. */
export function budgetedTools<TOOLS extends ToolSet>(tools: TOOLS, budget: EvidenceBudget): TOOLS {
  const wrapped = Object.entries(tools).map(([name, tool]) => {
    const execute = tool.execute!;
    return [
      name,
      {
        ...tool,
        execute: async (input: unknown, options: Parameters<typeof execute>[1]) => {
          const output = await execute(input, options);
          budget.charge(options.toolCallId, renderForModel(output));
          return output;
        },
        toModelOutput: ({ toolCallId, output }: { toolCallId: string; output: unknown }) => ({
          type: "text" as const,
          value: budget.view(toolCallId) ?? renderForModel(output),
        }),
      } as Tool,
    ];
  });
  return Object.fromEntries(wrapped) as TOOLS;
}
