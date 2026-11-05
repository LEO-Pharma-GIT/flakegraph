import { describe, expect, it } from "vitest";
import { collectAskText, encodeAskEvent, readAskNdjson } from "./protocol";
import { extractQuestion, parseDocumentIds, parseMode } from "./request";
import { AskHttpError } from "./errors";
import { answerFromGrounding, groundingFromToolOutput, mergeGrounding, emptyGrounding } from "./grounding";

describe("ask protocol", () => {
  it("round-trips NDJSON events", async () => {
    const events = [
      { type: "status" as const, phase: "planning" as const, message: "Planning retrieval" },
      { type: "text" as const, delta: "Judo " },
      { type: "text" as const, delta: "was developed by Kano." },
    ];
    const body = events.map((event) => encodeAskEvent(event)).join("");
    const response = new Response(body, { headers: { "content-type": "application/x-ndjson" } });
    const parsed = [];
    for await (const event of readAskNdjson(response)) {
      parsed.push(event);
    }
    expect(parsed).toEqual(events);
    expect(collectAskText(parsed)).toBe("Judo was developed by Kano.");
  });
});

describe("ask request parsing", () => {
  it("reads the last user message from AI SDK parts", () => {
    expect(
      extractQuestion({
        messages: [
          { role: "user", parts: [{ type: "text", text: "Who developed judo?" }] },
        ],
      }),
    ).toBe("Who developed judo?");
  });

  it("prefers an explicit question field", () => {
    expect(
      extractQuestion({
        question: "Named question",
        messages: [{ role: "user", content: "Ignored" }],
      }),
    ).toBe("Named question");
  });

  it("treats auto as an unspecified mode", () => {
    expect(parseMode("auto")).toBeUndefined();
    expect(parseMode("drift")).toBe("drift");
    expect(() => parseMode("sideways")).toThrow(AskHttpError);
  });

  it("parses document id subsets", () => {
    expect(parseDocumentIds(["doc_overview", "doc_overview"])).toEqual(["doc_overview"]);
    expect(parseDocumentIds(undefined)).toBeUndefined();
    expect(() => parseDocumentIds("doc_overview")).toThrow(AskHttpError);
  });
});

describe("ask grounding", () => {
  it("merges tool outputs and citations", () => {
    const first = groundingFromToolOutput({
      entityIds: ["judo"],
      citations: [{ quote: "Judo was developed by Jigoro Kano.", documentId: "doc_overview", entityId: "judo", entityName: "Judo" }],
      counts: { entities: 1, evidence: 1, communities: 0, relations: 1 },
    });
    const merged = mergeGrounding(first, groundingFromToolOutput({ entityIds: ["jigoro_kano"], citations: [] }));
    expect(merged.entityIds).toEqual(["judo", "jigoro_kano"]);
    const answer = answerFromGrounding({
      question: "Who developed judo?",
      mode: "local",
      summary: "Jigoro Kano developed judo.",
      scoped: false,
      billed: true,
      grounding: merged,
    });
    expect(answer.citations).toHaveLength(1);
    expect(answer.queryConsumption.billed).toBe(true);
    expect(emptyGrounding().entityIds).toEqual([]);
  });

  it("keeps the document name a tool gave its citation", () => {
    const grounding = groundingFromToolOutput({
      citations: [
        { quote: "Judo was developed by Jigoro Kano.", documentId: "doc_overview", documentName: "overview.md", entityId: "judo", entityName: "Judo" },
      ],
    });
    expect(grounding.citations[0]?.documentName).toBe("overview.md");
  });
});
