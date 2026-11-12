import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AskStreamEvent } from "./types";

// What the agent streams for the next question; a `pause` entry is silence.
let agentChunks: Array<Record<string, unknown>> = [];

vi.mock("ai", async (importOriginal) => ({
  ...(await importOriginal<typeof import("ai")>()),
  createAgentUIStream: async () =>
    (async function* () {
      for (const chunk of agentChunks) {
        if (typeof chunk.pause === "number") {
          await new Promise((resolve) => setTimeout(resolve, chunk.pause as number));
        } else {
          yield chunk;
        }
      }
    })(),
}));
vi.mock("./constants", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./constants")>()),
  ASK_HEARTBEAT_MS: 20,
}));
vi.mock("./agent", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./agent")>()),
  createAskAgent: () => ({}),
}));
vi.mock("./load", () => ({
  loadAskRunGraph: async () => ({ dataset: { graphId: "g1", nodes: [], edges: [] }, snapshot: { graphId: "g1" } }),
  perspectiveScope: async () => undefined,
}));
vi.mock("./model", () => ({
  probeAskModel: async () => ({ provider: "test", billed: false }),
  resolveAskModel: () => ({ provider: "test", billed: false }),
}));

const { handleAskRequest } = await import("./stream");
const { readAskNdjson } = await import("./protocol");

async function ask(): Promise<AskStreamEvent[]> {
  const response = await handleAskRequest(
    new Request("https://console.example.test/api/ask", {
      method: "POST",
      body: JSON.stringify({ runId: "run_1", question: "Who conducted the study?", format: "ndjson" }),
    }),
  );
  const events: AskStreamEvent[] = [];
  for await (const event of readAskNdjson(response)) events.push(event);
  return events;
}

describe("the NDJSON ask stream", () => {
  beforeEach(() => {
    agentChunks = [];
  });

  it("ends an answered question with done", async () => {
    agentChunks = [{ type: "text-delta", id: "a", delta: "An agency did." }];
    const last = (await ask()).at(-1);
    expect(last).toMatchObject({ type: "done", answer: { summary: "An agency did." } });
  });

  it("answers with what the model wrote last, not what it said on the way", async () => {
    agentChunks = [
      { type: "start-step" },
      { type: "text-delta", id: "a", delta: "Let me search for that." },
      { type: "tool-input-start", toolCallId: "t1", toolName: "searchGraph" },
      { type: "start-step" },
      { type: "text-delta", id: "b", delta: "An agency did." },
    ];
    expect((await ask()).at(-1)).toMatchObject({ type: "done", answer: { summary: "An agency did." } });
  });

  it("follows the answer with every document read one by one that bears on the question", async () => {
    const study = (documentId: string, documentName: string, answer: string) => ({
      documentId,
      documentName,
      group: "study-17",
      answer,
      pages: [3],
    });
    agentChunks = [
      { type: "tool-input-available", toolCallId: "t1", toolName: "readEachDocument", input: {} },
      {
        type: "tool-output-available",
        toolCallId: "t1",
        output: { text: "findings", studies: [study("doc_a", "Wave 1", "Yield rose 12%."), study("doc_b", "Wave 2", "Yield rose 9%.")] },
      },
      { type: "tool-input-available", toolCallId: "t2", toolName: "readEachDocument", input: {} },
      { type: "tool-output-available", toolCallId: "t2", output: { text: "findings", studies: [study("doc_a", "Wave 1", "Yield rose 12%.")] } },
      { type: "start-step" },
      { type: "text-delta", id: "a", delta: "Both waves found higher yields." },
    ];
    const events = await ask();
    const summary = (events.at(-1) as Extract<AskStreamEvent, { type: "done" }>).answer.summary;
    expect(summary).toBe(
      "Both waves found higher yields.\n\n## Study by study\n\n### study-17\n- **Wave 1** (p. 3): Yield rose 12%.\n- **Wave 2** (p. 3): Yield rose 9%.",
    );
    const streamed = events.flatMap((event) => (event.type === "text" ? [event.delta] : [])).join("");
    expect(streamed).toBe(summary);
  });

  it("adds the documents read one by one to the console's own message stream, before it finishes", async () => {
    agentChunks = [
      { type: "start" },
      { type: "tool-input-available", toolCallId: "t1", toolName: "readEachDocument", input: {} },
      {
        type: "tool-output-available",
        toolCallId: "t1",
        output: { text: "findings", studies: [{ documentId: "doc_a", documentName: "Wave 1", group: null, answer: "Yield rose 12%.", pages: [] }] },
      },
      { type: "text-start", id: "a" },
      { type: "text-delta", id: "a", delta: "Yields rose." },
      { type: "text-end", id: "a" },
      { type: "finish" },
    ];
    const response = await handleAskRequest(
      new Request("https://console.example.test/api/ask", {
        method: "POST",
        body: JSON.stringify({ runId: "run_1", question: "What changed?", format: "ui" }),
      }),
    );
    const chunks = (await response.text())
      .split("\n")
      .filter((line) => line.startsWith("data: {"))
      .map((line) => JSON.parse(line.slice(6)) as { type: string; delta?: string });
    const types = chunks.map((chunk) => chunk.type);
    const added = chunks.find((chunk) => chunk.type === "text-delta" && chunk.delta?.includes("## Study by study"));
    expect(added?.delta).toBe("\n\n## Study by study\n\n### Wave 1\nYield rose 12%.");
    expect(types.indexOf("finish")).toBeGreaterThan(chunks.indexOf(added!));
    expect(types.at(-1)).toBe("finish");
  });

  it("ends a question the time limit cut short with a timeout error, not an answer", async () => {
    agentChunks = [{ type: "abort", reason: "TimeoutError" }];
    const events = await ask();
    expect(events.at(-1)).toMatchObject({ type: "error", code: "timeout" });
    expect(events.some((event) => event.type === "done")).toBe(false);
  });

  it("ends a question the model left unanswered with a no_answer error", async () => {
    agentChunks = [{ type: "tool-input-start", toolCallId: "t1", toolName: "searchGraph" }];
    const events = await ask();
    expect(events.at(-1)).toMatchObject({ type: "error", code: "no_answer" });
    expect(events.some((event) => event.type === "done")).toBe(false);
  });

  it("repeats a status while the answer is silent, so the connection stays alive", async () => {
    agentChunks = [{ pause: 120 }, { type: "text-delta", id: "a", delta: "An agency did." }];
    const events = await ask();
    const answering = events.filter((event) => event.type === "status" && event.phase === "answering");
    // One when answering starts, then at least one repeat during the silence.
    expect(answering.length).toBeGreaterThan(1);
    expect(events.at(-1)).toMatchObject({ type: "done" });
  });
});
