import { Effect } from "effect";
import { describe, expect, it, vi } from "vitest";
import type { RunSnapshot } from "@/server/protocol/schema";

// What the next request resolves to: the viewer's plane, or a refusal.
let resolveAskContext: () => Promise<unknown>;

vi.mock("@/server/ask/load", () => ({
  resolveAskContext: () => resolveAskContext(),
}));

const { GET } = await import("./route");
const { AskHttpError } = await import("@/server/ask/errors");

const run = (fields: Partial<RunSnapshot>) =>
  ({ graphName: null, documentsCompleted: 0, updatedAt: null, ...fields }) as RunSnapshot;

describe("GET /api/graphs", () => {
  it("lists the graphs the viewer's plane returns, each at its newest finished version", async () => {
    const listRuns = vi.fn(() =>
      Effect.succeed([
        run({ runId: "r3", graphId: "g1", status: "running" }),
        run({ runId: "r2", graphId: "g1", status: "succeeded", graphName: "Trials", documentsCompleted: 12, updatedAt: "2026-01-02T00:00:00Z" }),
        run({ runId: "r1", graphId: "g2", status: "failed" }),
      ]),
    );
    resolveAskContext = async () => ({ controlPlane: { listRuns } });

    const response = await GET(new Request("https://console.example.test/api/graphs"));

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      graphs: [{ runId: "r2", graphId: "g1", name: "Trials", documentCount: 12, updatedAt: "2026-01-02T00:00:00Z" }],
    });
  });

  it("answers a caller nobody vouched for with the refusal", async () => {
    resolveAskContext = async () => {
      throw new AskHttpError(401, "Sign in through the gate.", "unauthorized");
    };

    const response = await GET(new Request("https://console.example.test/api/graphs"));

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: "Sign in through the gate." });
  });
});
