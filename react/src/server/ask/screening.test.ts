import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import type { CatalogueCard } from "./catalogue";
import type { AskModel } from "./model";

const prompts: string[] = [];

/** Calls the cards about soil unrelated, fails any batch holding a "Broken" card, and leaves "Unjudged" out. */
function scriptedModel(): AskModel {
  const languageModel = new MockLanguageModelV4({
    doGenerate: async (options) => {
      const prompt = JSON.stringify(options.prompt);
      prompts.push(prompt);
      if (prompt.includes("Broken")) {
        throw new Error("engine unavailable");
      }
      const numbered = [...prompt.matchAll(/\[(\d+)\] ([^\\]+)/g)].map((match) => ({
        number: Number(match[1]),
        name: match[2]!,
      }));
      const documents = numbered
        .filter((item) => !item.name.startsWith("Unjudged"))
        .map((item) => ({ number: item.number, verdict: item.name.startsWith("Soil") ? "unrelated" : "may_bear" }));
      return {
        content: [{ type: "text", text: JSON.stringify({ documents }) }],
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

const { SCREEN_BATCH_SIZE, SCREEN_MIN_DOCUMENTS, screenDocuments, screenPrompt } = await import("./screening");

const card = (id: string, name: string, text: string | null = `${name} card.`): CatalogueCard => ({
  documentId: id,
  name,
  location: name,
  folder: null,
  card: text,
});

const irrigation = (count: number) =>
  Array.from({ length: count }, (_, index) => card(`doc_${index}`, `Irrigation trial ${index}`));

describe("screenDocuments", () => {
  it("drops only documents judged about something else, keeping the order given", async () => {
    prompts.length = 0;
    const candidates = [
      ...irrigation(SCREEN_MIN_DOCUMENTS),
      card("soil_1", "Soil survey north"),
      card("soil_2", "Soil survey south"),
      card("unjudged", "Unjudged report"),
    ];
    const result = await screenDocuments("What did irrigation change?", candidates, {
      assumptions: ["Irrigation was trialled"],
    });

    expect(result.dropped.map((item) => item.documentId)).toEqual(["soil_1", "soil_2"]);
    expect(result.kept.map((item) => item.documentId)).toEqual(
      candidates.map((item) => item.documentId).filter((id) => !id.startsWith("soil")),
    );
    expect(prompts).toHaveLength(Math.ceil(candidates.length / SCREEN_BATCH_SIZE));
    expect(prompts.every((prompt) => prompt.includes("Irrigation was trialled"))).toBe(true);
  });

  it("reads documents without a card, those it must always read, and a failed batch's documents", async () => {
    prompts.length = 0;
    const candidates = [
      ...irrigation(SCREEN_MIN_DOCUMENTS),
      card("soil_bare", "Soil survey bare", null),
      card("soil_best", "Soil survey with the best passage"),
    ];
    const result = await screenDocuments("What did irrigation change?", candidates, {
      alwaysRead: new Set(["soil_best"]),
    });
    expect(result.dropped).toEqual([]);
    expect(prompts.join("")).not.toContain("Soil survey bare");

    const failing = [...irrigation(SCREEN_MIN_DOCUMENTS - 1), card("broken", "Broken report"), card("soil_3", "Soil survey west")];
    const failed = await screenDocuments("What did irrigation change?", failing);
    expect(failed.failedBatches).toBeGreaterThan(0);
    expect(failed.kept.map((item) => item.documentId)).toContain("broken");
  });

  it("does not screen a short list", async () => {
    prompts.length = 0;
    const few = [card("soil_1", "Soil survey north"), ...irrigation(SCREEN_MIN_DOCUMENTS - 2)];
    const result = await screenDocuments("What did irrigation change?", few);

    expect(result.kept).toHaveLength(few.length);
    expect(prompts).toHaveLength(0);
  });
});

describe("screenPrompt", () => {
  it("numbers the cards from one and names their folders", () => {
    const prompt = screenPrompt("Which trials irrigated?", [
      { ...card("a", "Orchard trial"), folder: "farms/2024" },
      card("b", "Soil survey"),
    ]);
    expect(prompt).toContain("[1] Orchard trial\nFolder: farms/2024\nOrchard trial card.");
    expect(prompt).toContain("[2] Soil survey\nSoil survey card.");
  });

  it("shows each document's best passages beside its card", () => {
    const prompt = screenPrompt("Which trials irrigated?", [card("b", "Soil survey")], [], new Map([["b", ["Plots were\n  irrigated weekly.", "pH 6.", "third", "fourth"]]]));
    expect(prompt).toContain("Soil survey card.\nPassages that best match the question:\n- Plots were irrigated weekly.\n- pH 6.\n- third");
    expect(prompt).not.toContain("fourth");
  });
});
