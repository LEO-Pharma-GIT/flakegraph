import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, it, vi } from "vitest";
import type { LanguageModelV4CallOptions } from "@ai-sdk/provider";
import type { AskModel } from "./model";
import type { CommunityHit, EntityHit } from "./types";

/**
 * Every call the console makes to its language model, through the SDK's own
 * prompt validation with a model that answers.
 *
 * The SDK rejects a system message inside `messages` (instructions go in
 * `instructions`), and each caller here has a fallback that swallows the
 * rejection: the planner answers lexically, the reranker keeps vector
 * order, DRIFT asks no follow-ups, the ontology proposal lists the nouns
 * of the description. A model that is configured and fails every call
 * therefore looks like one that was never configured. These tests hold
 * each call to a model that answers, so a rejected prompt fails here.
 */

const calls: LanguageModelV4CallOptions[] = [];
const answers: string[] = [];

function jsonModel(): AskModel {
  const languageModel = new MockLanguageModelV4({
    doGenerate: async (options) => {
      calls.push(options);
      const text = answers.shift();
      if (text === undefined) {
        throw new Error("no scripted answer left");
      }
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
  return {
    ...original,
    resolveAskModel: () => jsonModel(),
    probeAskModel: async () => jsonModel(),
  };
});

const { planQueryRetrieval } = await import("./planner");
const { rerankEntities } = await import("./rerank");
const { generateDriftFollowUps } = await import("./drift");
const { proposeOntologyForIntent } = await import("../ontology");

function entity(id: string, distance: number): EntityHit {
  return { id, name: id, type: "PERSON", description: `About ${id}`, score: 1, distance };
}

describe("console calls to the language model", () => {
  it("plans a question with the model, not the lexical fallback", async () => {
    calls.length = 0;
    answers.push(JSON.stringify({ mode: "local", highLevelKeywords: ["judo history"], lowLevelKeywords: ["Jigoro Kano"] }));

    const plan = await planQueryRetrieval("Who founded judo?");

    expect(plan).toEqual({ mode: "local", highLevelKeywords: ["judo history"], lowLevelKeywords: ["Jigoro Kano"] });
    expect(calls).toHaveLength(1);
    expect(calls[0]?.prompt.map((message) => message.role)).toEqual(["system", "user"]);
    // A helper call asks for no reasoning: the plan is the same, tens of seconds sooner.
    expect(calls[0]?.providerOptions?.openai?.reasoningEffort).toBe("none");
  });

  it("reranks entities by the model's scores", async () => {
    calls.length = 0;
    answers.push(
      JSON.stringify({ score: 1, rationale: "off topic" }),
      JSON.stringify({ score: 9, rationale: "the founder" }),
    );

    const reranked = await rerankEntities("Who founded judo?", [entity("karate", 0.1), entity("kano", 0.2)], 1);

    expect(reranked.map((hit) => hit.id)).toEqual(["kano"]);
    expect(calls).toHaveLength(2);
  });

  it("asks the model for DRIFT follow-up questions", async () => {
    calls.length = 0;
    answers.push(
      JSON.stringify({ followUps: [{ question: "Which schools did Kano's students found?", rationale: "lineage" }] }),
    );
    const primer: CommunityHit = {
      id: "c1",
      title: "Kodokan lineage",
      summary: "Kano and his students.",
      memberIds: [],
      rating: null,
      score: 1,
      distance: 0.1,
      findings: [],
      level: 0,
      parentCommunityId: null,
    };

    const followUps = await generateDriftFollowUps("How did judo spread?", [primer]);

    expect(followUps.map((item) => item.question)).toEqual(["Which schools did Kano's students found?"]);
  });

  it("proposes an ontology from the model rather than the nouns of the description", async () => {
    calls.length = 0;
    answers.push(
      JSON.stringify({
        entityTypes: [
          { name: "SPONSOR", description: "An organisation funding a trial." },
          { name: "DRUG", description: "An investigational product." },
          { name: "TRIAL", description: "A registered clinical study." },
        ],
        relationTypes: [
          { name: "SPONSORS", description: "Sponsor funds trial." },
          { name: "RELATED_TO", description: "Any other link." },
        ],
      }),
    );

    const proposal = await proposeOntologyForIntent("clinical trials, their sponsors and drugs");

    expect(proposal.source).toBe("model");
    expect(proposal.modelFailure).toBeNull();
    expect(proposal.types).toEqual(["SPONSOR", "DRUG", "TRIAL"]);
    expect(proposal.relations).toEqual(["SPONSORS", "RELATED_TO"]);
    expect(proposal.descriptions.DRUG).toBe("An investigational product.");
  });
});
