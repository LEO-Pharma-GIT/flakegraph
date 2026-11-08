import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import type { AskModel } from "./model";

const prompts: string[] = [];

function scriptedModel(): AskModel {
  const languageModel = new MockLanguageModelV4({
    doGenerate: async (options) => {
      const prompt = JSON.stringify(options.prompt);
      prompts.push(prompt);
      if (prompt.includes("Broken report")) {
        throw new Error("engine unavailable");
      }
      const relevant = prompt.includes("With irrigation");
      const text = JSON.stringify({
        relevant,
        answer: relevant ? "Irrigated plots yielded 12% more." : "",
        facts: relevant ? [{ statement: "Yield rose 12% with irrigation.", page: 3 }] : [],
      });
      return {
        content: [{ type: "text", text }],
        finishReason: { unified: "stop", raw: undefined },
        usage: {
          inputTokens: { total: 1, noCache: 1, cacheRead: undefined, cacheWrite: undefined },
          outputTokens: { total: 1, text: 1, reasoning: undefined },
        },
        warnings: [],
      };
    },
  });
  return {
    languageModel,
    provider: "openai",
    modelId: "llm-interactive",
    billed: false,
    structuredOptions: { providerOptions: { openai: { reasoningEffort: "none" } } },
  };
}

vi.mock("./model", async (importOriginal) => {
  const original = await importOriginal<typeof import("./model")>();
  return { ...original, resolveAskModel: () => scriptedModel() };
});

const { documentContext, readEachDocument, renderFindings } = await import("./per-document");

const source = (id: string, name: string, text: string, card: string | null = null) => ({
  documentId: id,
  documentName: name,
  card,
  passages: [{ page: 3, sectionPath: ["Results"], text }],
});

describe("readEachDocument", () => {
  it("reads every document separately and keeps the order it was given", async () => {
    prompts.length = 0;
    const findings = await readEachDocument("What did irrigation change?", [
      source("doc_a", "Orchard trial", "With irrigation the yield rose 12%."),
      source("doc_b", "Soil survey", "Soil pH ranged from 5.5 to 7."),
      source("doc_c", "Broken report", "Unreadable."),
      { documentId: "doc_d", documentName: "Empty file", card: null, passages: [] },
    ]);

    expect(findings.map((finding) => [finding.documentId, finding.relevant])).toEqual([
      ["doc_a", true],
      ["doc_b", false],
      ["doc_c", false],
      ["doc_d", false],
    ]);
    expect(findings[0]?.facts).toEqual([{ statement: "Yield rose 12% with irrigation.", page: 3 }]);
    expect(findings[2]?.failure).toContain("engine unavailable");
    // A document with nothing to read costs no call.
    expect(prompts).toHaveLength(3);
    expect(prompts.every((prompt) => prompt.includes("What did irrigation change?"))).toBe(true);
  });
});

describe("renderFindings", () => {
  it("lists what each relevant document says and names the rest", () => {
    const text = renderFindings([
      {
        documentId: "doc_a",
        documentName: "Orchard trial",
        relevant: true,
        answer: "Irrigated plots yielded 12% more.",
        facts: [{ statement: "Yield rose 12% with irrigation.", page: 3 }],
      },
      { documentId: "doc_b", documentName: "Soil survey", relevant: false, answer: "", facts: [] },
      {
        documentId: "doc_c",
        documentName: "Broken report",
        relevant: false,
        answer: "",
        facts: [],
        failure: "timeout",
      },
    ]);

    expect(text).toBe(
      [
        "Read 3 documents one by one: 1 bear on the question, 1 do not, 1 could not be read.",
        "",
        "## Orchard trial (doc_a)",
        "Irrigated plots yielded 12% more.",
        "- [p3] Yield rose 12% with irrigation.",
        "",
        "Say nothing on it: Soil survey",
        "",
        "Could not be read: Broken report",
      ].join("\n"),
    );
  });
});

describe("documentContext", () => {
  it("puts the card first and stops within the bound", () => {
    const long = "word ".repeat(2_000);
    const context = documentContext(
      {
        documentId: "doc",
        documentName: "Trial",
        card: "Title: Trial",
        passages: [
          { page: 1, sectionPath: [], text: "First passage." },
          { page: 2, sectionPath: ["Results"], text: long },
        ],
      },
      1_000,
    );

    expect(context.startsWith("Document: Trial\n\nCard:\nTitle: Trial\n\n[p1]\nFirst passage.")).toBe(true);
    expect(context.length).toBeLessThanOrEqual(1_002);
    expect(context).toContain("[p2] Results");
  });
});
