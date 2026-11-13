import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import type { AskModel } from "./model";

let reply: string | Error = "";

function scriptedModel(): AskModel {
  const languageModel = new MockLanguageModelV4({
    doGenerate: async () => {
      if (reply instanceof Error) {
        throw reply;
      }
      return {
        content: [{ type: "text", text: reply }],
        finishReason: { unified: "stop", raw: undefined },
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
          outputTokens: { total: 1, text: 1, reasoning: undefined },
        },
        warnings: [],
      };
    },
  });
  return { languageModel, provider: "openai", modelId: "llm-interactive", billed: false, structuredOptions: {} };
}

vi.mock("./model", async (importOriginal) => {
  const original = await importOriginal<typeof import("./model")>();
  return { ...original, resolveAskModel: () => scriptedModel() };
});

const { classifyQuestion, scopeNote, spansDocuments } = await import("./question-scope");

describe("classifyQuestion", () => {
  it("returns the scope, the assumptions and whether the question counts", async () => {
    reply = JSON.stringify({
      scope: "several_documents",
      assumptions: ["Both trials measured yield"],
      counts: false,
    });
    const scope = await classifyQuestion("How did yield differ between the two trials?");

    expect(scope).toEqual({ scope: "several_documents", assumptions: ["Both trials measured yield"], counts: false });
    expect(spansDocuments(scope)).toBe(true);
  });

  it("answers other when the model fails, which changes nothing downstream", async () => {
    reply = new Error("engine unavailable");
    const scope = await classifyQuestion("What is a trial?");

    expect(scope).toEqual({ scope: "other", assumptions: [], counts: false });
    expect(spansDocuments(scope)).toBe(false);
    expect(scopeNote(scope)).toBeNull();
  });
});

describe("scopeNote", () => {
  it("names the assumptions to check, the documents to cover and the unit to count", () => {
    const note = scopeNote({ scope: "collection", assumptions: ["A pilot study exists"], counts: true }) ?? "";

    expect(note).toContain("- A pilot study exists");
    expect(note).toContain("say so in your first sentence");
    expect(note).toContain("account for every document");
    expect(note).toContain("whether you counted documents or folders");
  });
});
