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
        facts: relevant ? [{ statement: "Yield rose 12% with irrigation.", quote: "With irrigation the yield rose 12%.", page: 3 }] : [],
        premise_issues: prompt.includes("Soil survey")
          ? [{ assumption: "The soil survey covered irrigation", verdict: "contradicted", why: "It measured pH only." }]
          : [],
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

const source = (id: string, name: string, text: string, group: string | null = null) => ({
  documentId: id,
  documentName: name,
  card: null,
  group,
  passages: [{ page: 3, sectionPath: ["Results"], text }],
});

describe("readEachDocument", () => {
  it("reads every document separately, with the question's assumptions, in the order given", async () => {
    prompts.length = 0;
    const findings = await readEachDocument(
      "What did irrigation change?",
      [
        source("doc_a", "Orchard trial", "With irrigation the yield rose 12%.", "farms/trial"),
        source("doc_b", "Soil survey", "Soil pH ranged from 5.5 to 7."),
        source("doc_c", "Broken report", "Unreadable."),
        { documentId: "doc_d", documentName: "Empty file", card: null, group: null, passages: [] },
      ],
      { assumptions: ["The soil survey covered irrigation"] },
    );

    expect(findings.map((finding) => [finding.documentId, finding.relevant])).toEqual([
      ["doc_a", true],
      ["doc_b", false],
      ["doc_c", false],
      ["doc_d", false],
    ]);
    expect(findings[0]?.facts[0]?.quote).toBe("With irrigation the yield rose 12%.");
    expect(findings[0]?.group).toBe("farms/trial");
    expect(findings[1]?.premiseIssues?.[0]?.verdict).toBe("contradicted");
    expect(findings[2]?.failure).toContain("engine unavailable");
    // A document with nothing to read costs no call; every call sees the assumptions.
    expect(prompts).toHaveLength(3);
    expect(prompts.every((prompt) => prompt.includes("The soil survey covered irrigation"))).toBe(true);
  });
});

describe("renderFindings", () => {
  const finding = (id: string, name: string, group: string | null) => ({
    documentId: id,
    documentName: name,
    group,
    relevant: true,
    answer: `${name} says yes.`,
    facts: [{ statement: "A fact.", quote: "the exact words", page: 2 }],
  });

  it("opens with problems in the question's assumptions and lists each document with its quotes", () => {
    const text = renderFindings([
      finding("doc_a", "Orchard trial", null),
      {
        documentId: "doc_b",
        documentName: "Soil survey",
        relevant: false,
        answer: "",
        facts: [],
        premiseIssues: [{ assumption: "It covered irrigation", verdict: "contradicted" as const, why: "pH only." }],
      },
      { documentId: "doc_c", documentName: "Broken report", relevant: false, answer: "", facts: [], failure: "timeout" },
    ]);

    expect(text.split("\n")[0]).toContain("Read 3 documents one by one: 1 bear on the question, 1 do not, 1 could not be read.");
    expect(text).toContain(
      "## Problems with the question's assumptions - open the answer with these\n- contradicted: \"It covered irrigation\" - pH only. (Soil survey)",
    );
    expect(text).toContain("## Orchard trial (doc_a)\nOrchard trial says yes.\n- [p2] A fact. - \"the exact words\"");
    expect(text.indexOf("Problems with")).toBeLessThan(text.indexOf("## Orchard trial"));
    expect(text).toContain("Say nothing on it: Soil survey");
    expect(text).toContain("Could not be read: Broken report");
  });

  it("groups documents that share a folder and counts the folders", () => {
    const text = renderFindings([
      finding("doc_a", "Wave 1 report", "study-17"),
      finding("doc_b", "Wave 1 appendix", "study-17"),
      finding("doc_c", "Pilot", "study-20"),
    ]);

    expect(text).toContain("3 bear on the question, in 2 folders");
    expect(text).toContain("## Folder: study-17 (2 documents)\n\n### Wave 1 report (doc_a)");
    expect(text).toContain("## Folder: study-20 (1 document)");
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
        group: null,
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
