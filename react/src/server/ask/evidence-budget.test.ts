import { tool } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { GRAPH_SEARCH_CONTEXT_CHAR_BUDGET } from "./constants";
import { budgetedTools, EvidenceBudget, renderForModel } from "./evidence-budget";
import { createAskTools } from "./tools";
import type { GraphDataset } from "../protocol/schema";

const calls: LanguageModelV4CallOptions[] = [];
const toolCallEveryStep = new MockLanguageModelV4({
  doGenerate: async (options) => {
    calls.push(options);
    const usage = {
      inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
      outputTokens: { total: 1, text: 1, reasoning: undefined },
    };
    if (options.toolChoice?.type === "none") {
      return { content: [{ type: "text", text: "From what was found." }], finishReason: { unified: "stop", raw: undefined }, usage, warnings: [] };
    }
    return {
      content: [{ type: "tool-call", toolCallId: `t${calls.length}`, toolName: "listDocuments", input: "{}" }],
      finishReason: { unified: "tool-calls", raw: undefined },
      usage,
      warnings: [],
    };
  },
});

vi.mock("./model", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./model")>()),
  resolveAskModel: () => ({ languageModel: toolCallEveryStep, provider: "openai", modelId: "test", billed: false, structuredOptions: {} }),
}));

const { createAskAgent } = await import("./agent");

const options = { toolCallId: "fixture", messages: [], context: {} };

function hubDataset(spokes: number): GraphDataset {
  const ids = Array.from({ length: spokes }, (_, index) => `spoke_${index}`);
  return {
    graphId: "hub",
    nodes: [{ id: "hub", name: "Hub", description: "H".repeat(2_000) }, ...ids.map((id) => ({ id, name: id, description: "S".repeat(2_000) }))],
    edges: ids.map((id, index) => ({ id: `edge_${id}`, source_node_id: "hub", target_node_id: id, relation_type: "LINKS", description: "L".repeat(2_000), weight: index })),
    communities: [],
    documents: [],
    chunks: [],
    evidence: [],
    runReport: {},
    graphMetrics: {},
    discardedWindows: [],
    rejectedRecords: [],
    failedDocuments: [],
  };
}

describe("what the model reads of a tool result", () => {
  it("gets a search once, with the documents named, and no id lists", () => {
    const view = renderForModel({
      text: "## Source quotes\n- [doc_1] A finding.",
      relationIds: ["edge_hidden"],
      citations: [{ quote: "A finding.", documentId: "doc_1", documentName: "Study.pptx" }],
    });
    expect(view).toContain("doc_1: Study.pptx");
    expect(view.match(/A finding\./g)).toHaveLength(1);
    expect(view).not.toContain("edge_hidden");
  });

  it("reads quotes under their document and keeps entity ids to fetch more", () => {
    const view = renderForModel({
      citations: [
        { quote: "First.", documentId: "doc_1", documentName: "Study.pptx", entityId: "node_1", entityName: "Agency" },
        { quote: "Second.", documentId: "doc_1", documentName: "Study.pptx", entityId: null, entityName: null },
      ],
    });
    expect(view).toBe("## Study.pptx (doc_1)\n- [Agency, node_1] First.\n- Second.");
  });

  it("keeps the caller's full output while the model reads less", async () => {
    const budget = new EvidenceBudget(100);
    const tools = budgetedTools(
      { search: tool({ inputSchema: z.object({}), execute: async () => ({ text: "x".repeat(500), relationIds: ["edge_1"] }) }) },
      budget,
    );
    const output = await tools.search.execute!({}, options);
    expect(output).toMatchObject({ relationIds: ["edge_1"] });
    expect(budget.view("fixture")?.startsWith("x".repeat(100))).toBe(true);
    expect(budget.view("fixture")).toContain("evidence budget");
    expect(budget.spent).toBe(true);
  });

  it("holds any one result to a search's size", () => {
    const budget = new EvidenceBudget();
    budget.charge("t1", "x".repeat(GRAPH_SEARCH_CONTEXT_CHAR_BUDGET * 2));
    expect(budget.view("t1")!.length).toBeLessThan(GRAPH_SEARCH_CONTEXT_CHAR_BUDGET + 200);
    expect(budget.spent).toBe(false);
  });

  it("charges tools called together, before the next prompt is sent", () => {
    const budget = new EvidenceBudget(100);
    budget.charge("t1", "x".repeat(60));
    budget.charge("t2", "y".repeat(60));
    expect(budget.view("t2")).toContain("y".repeat(40));
    expect(budget.view("t2")).not.toContain("y".repeat(41));
    expect(budget.spent).toBe(true);
  });
});

describe("the neighborhood of a hub", () => {
  it("stays bounded, keeps the strongest relations and says what it left out", async () => {
    const tools = createAskTools({ dataset: hubDataset(500) });
    const result = (await tools.getNeighborhood.execute!({ entityId: "hub", hops: 3 }, options)) as {
      entities: Array<{ id: string; description: string }>;
      relations: Array<{ id: string }>;
      leftOut: { entities: number; relations: number };
    };
    expect(result.relations).toHaveLength(60);
    expect(result.relations[0]?.id).toBe("edge_spoke_499");
    expect(result.entities[0]?.id).toBe("hub");
    expect(result.entities).toHaveLength(40);
    expect(result.entities[0]?.description.length).toBeLessThanOrEqual(400);
    expect(result.leftOut).toEqual({ entities: 461, relations: 440 });
  });
});

describe("a question that never stops searching", () => {
  it("is made to answer on its last step, from what it found", async () => {
    calls.length = 0;
    const agent = createAskAgent({ dataset: hubDataset(1) });
    const result = await agent.generate({ prompt: "Which agency ran it?" });

    expect(result.text).toBe("From what was found.");
    // One call classifies the question, then one per step.
    expect(calls).toHaveLength(9);
    expect(calls.at(-1)?.toolChoice).toEqual({ type: "none" });
    expect(calls.at(-1)?.prompt.at(-1)).toMatchObject({ role: "user" });
    expect(calls.at(-2)?.toolChoice).not.toEqual({ type: "none" });
    expect(calls[1]?.maxOutputTokens).toBeGreaterThan(0);
  });
});
