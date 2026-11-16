import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import type { AskModel } from "./model";

const prompts: string[] = [];
let inFlight = 0;
let mostInFlight = 0;
const flaky = new Set<string>();

function scriptedModel(): AskModel {
  const languageModel = new MockLanguageModelV4({
    doGenerate: async (options) => {
      const prompt = JSON.stringify(options.prompt);
      prompts.push(prompt);
      if (prompt.includes("Broken report")) {
        throw new Error("engine unavailable");
      }
      const name = /Document: (Flaky [^\\]+?)\\n/.exec(prompt)?.[1];
      if (name && !flaky.has(name)) {
        flaky.add(name);
        throw new Error("timed out");
      }
      if (prompt.includes("Slow")) {
        inFlight += 1;
        mostInFlight = Math.max(mostInFlight, inFlight);
        await new Promise((resolve) => setTimeout(resolve, 20));
        inFlight -= 1;
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

process.env.FLAKEGRAPH_ASK_READ_SLOTS = "3";
const { PER_DOCUMENT_ATTEMPTS, documentContext, readEachDocument, renderFindings, renderStudies, studyFindings, studyFindingsFromOutput, withoutGaps } =
  await import("./per-document");

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
    // A document with nothing to read costs no call, one that fails is read once more; every call sees the assumptions.
    expect(prompts).toHaveLength(2 + PER_DOCUMENT_ATTEMPTS);
    expect(prompts.filter((prompt) => prompt.includes("Broken report"))).toHaveLength(PER_DOCUMENT_ATTEMPTS);
    expect(prompts.every((prompt) => prompt.includes("The soil survey covered irrigation"))).toBe(true);
  });
});

describe("readEachDocument under load", () => {
  it("reads a document again when its call fails once", async () => {
    const [finding] = await readEachDocument("What did irrigation change?", [
      source("doc_f", "Flaky report", "With irrigation the yield rose 12%."),
    ]);
    expect(finding?.failure).toBeUndefined();
    expect(finding?.relevant).toBe(true);
  });

  it("keeps the calls of every question together within the shared slots", async () => {
    mostInFlight = 0;
    const slow = (prefix: string) =>
      Array.from({ length: 5 }, (_, index) => source(`${prefix}_${index}`, `Slow ${prefix} ${index}`, "Soil pH ranged from 5.5 to 7."));
    const [first, second] = await Promise.all([
      readEachDocument("What did irrigation change?", slow("a")),
      readEachDocument("What did irrigation change?", slow("b")),
    ]);
    expect(first).toHaveLength(5);
    expect(second).toHaveLength(5);
    expect(mostInFlight).toBe(3);
  });
});

describe("withoutGaps", () => {
  it("drops sentences that report what the document leaves out and keeps what it states", () => {
    expect(
      withoutGaps("The document does not state the 2020 strengths. It reports that Concept W was the most memorable. It does not mention pricing."),
    ).toBe("It reports that Concept W was the most memorable.");
    expect(withoutGaps("There is no information on Japan. The report does not specify a sample size.")).toBe("");
  });

  it("keeps the document's statements about its own scope and gaps that carry a finding", () => {
    const scope = "No, the study did not include pharmacists; it surveyed 81 dermatologists.";
    expect(withoutGaps(scope)).toBe(scope);
    const withFinding = "The document does not state a total, but lists 3 waves.";
    expect(withoutGaps(withFinding)).toBe(withFinding);
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
    expect(text.split("\n")[0]).toContain("Each one's findings are added after your answer, study by study");
    expect(text).toContain("Open with what they show.");
    expect(text).toContain("## Orchard trial (doc_a)\nOrchard trial says yes.\n- [p2] A fact. - \"the exact words\"");
    // A document that says nothing on the question does not settle its assumptions while others bear on it.
    expect(text).not.toContain("Problems with");
    expect(text).toContain("Say nothing on it: Soil survey");
    expect(text).toContain("Could not be read: Broken report");
  });

  it("opens with the problems that documents bearing on the question raise, or those of the rest when none does", () => {
    const issue = { assumption: "It covered irrigation", verdict: "contradicted" as const, why: "pH only." };
    const bearing = renderFindings([{ ...finding("doc_a", "Orchard trial", null), premiseIssues: [issue] }]);
    expect(bearing).toContain(
      "## Problems with the question's assumptions - open the answer with these. Each is what one document states about itself; " +
        "an assumption about the collection as a whole holds when any document listed below supports it\n" +
        "- contradicted: \"It covered irrigation\" - pH only. (Orchard trial)",
    );
    expect(bearing.indexOf("Problems with")).toBeLessThan(bearing.indexOf("## Orchard trial"));

    const silent = renderFindings([
      { documentId: "doc_b", documentName: "Soil survey", relevant: false, answer: "", facts: [], premiseIssues: [issue] },
    ]);
    expect(silent).toContain("- contradicted: \"It covered irrigation\" - pH only. (Soil survey)");
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

describe("studyFindings and renderStudies", () => {
  it("lists each document that bears on the question under its folder, with the pages its facts rest on", () => {
    const studies = studyFindings([
      {
        documentId: "doc_a",
        documentName: "Wave 1 report",
        group: "study-17",
        relevant: true,
        answer: "Yield rose 12%.",
        facts: [
          { statement: "A.", quote: "a", page: 7 },
          { statement: "B.", quote: "b", page: 3 },
          { statement: "C.", quote: "c", page: 7 },
        ],
      },
      { documentId: "doc_b", documentName: "Soil survey", relevant: false, answer: "", facts: [] },
      { documentId: "doc_c", documentName: "Pilot", group: null, relevant: true, answer: "No change.", facts: [] },
    ]);
    expect(studies.map((study) => study.documentId)).toEqual(["doc_a", "doc_c"]);
    expect(renderStudies(studies)).toBe(
      "## Study by study\n\n### study-17\n- **Wave 1 report** (p. 3, 7): Yield rose 12%.\n\n### Pilot\nNo change.",
    );
    expect(renderStudies([])).toBe("");
  });

  it("reads study findings only from a tool output that carries them", () => {
    expect(studyFindingsFromOutput({ text: "x" })).toEqual([]);
    expect(studyFindingsFromOutput({ studies: [{ documentId: "d", documentName: "D", answer: "A", pages: [], group: null }, { bad: 1 }] })).toHaveLength(1);
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
