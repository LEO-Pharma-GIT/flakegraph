import { MockLanguageModelV4 } from "ai/test";
import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";
import { describe, expect, it, vi } from "vitest";
import type { GraphDataset } from "../protocol/schema";

const calls: LanguageModelV4CallOptions[] = [];
let classification = "";
const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
  outputTokens: { total: 1, text: 1, reasoning: undefined },
};
const model = new MockLanguageModelV4({
  doGenerate: async (options) => {
    calls.push(options);
    // The classification is the one call made without tools.
    if (!options.tools?.length) {
      return { content: [{ type: "text", text: classification }], finishReason: { unified: "stop", raw: undefined }, usage, warnings: [] };
    }
    if (options.toolChoice?.type === "tool") {
      return {
        content: [{ type: "tool-call", toolCallId: "t1", toolName: options.toolChoice.toolName, input: JSON.stringify({ question: "q" }) }],
        finishReason: { unified: "tool-calls", raw: undefined },
        usage,
        warnings: [],
      };
    }
    return { content: [{ type: "text", text: "The answer." }], finishReason: { unified: "stop", raw: undefined }, usage, warnings: [] };
  },
});

vi.mock("./model", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./model")>()),
  resolveAskModel: () => ({ languageModel: model, provider: "openai", modelId: "test", billed: false, structuredOptions: {} }),
}));

const { createAskAgent } = await import("./agent");

const dataset = {
  graphId: "g",
  nodes: [{ id: "n", name: "Node" }],
  edges: [],
  communities: [],
  documents: [{ id: "doc_a", source_uri: "s3://b/a.pdf", folder: "study-1", summary: "Title: A" }],
  chunks: [],
  evidence: [],
  runReport: {},
  graphMetrics: {},
  discardedWindows: [],
  rejectedRecords: [],
  failedDocuments: [],
} as unknown as GraphDataset;

describe("a question that spans documents", () => {
  it("starts by reading the documents one by one, told what it takes for granted", async () => {
    calls.length = 0;
    classification = JSON.stringify({ scope: "collection", assumptions: ["Every study surveyed growers"], counts: true });
    const result = await createAskAgent({ dataset }).generate({ prompt: "How many studies surveyed growers?" });

    expect(result.text).toBe("The answer.");
    const first = calls[1];
    expect(first?.toolChoice).toEqual({ type: "tool", toolName: "readEachDocument" });
    expect(JSON.stringify(first?.prompt)).toContain("Every study surveyed growers");
  });

  it("leaves a question about one document to the model's own choice of tools", async () => {
    calls.length = 0;
    classification = JSON.stringify({ scope: "one_document", assumptions: [], counts: false });
    await createAskAgent({ dataset }).generate({ prompt: "What did study A find?" });

    expect(calls[1]?.toolChoice).not.toEqual({ type: "tool", toolName: "readEachDocument" });
  });
});
