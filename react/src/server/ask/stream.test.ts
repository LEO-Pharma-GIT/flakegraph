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
