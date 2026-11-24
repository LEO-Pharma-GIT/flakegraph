import { afterEach, describe, expect, it, vi } from "vitest";
import { polledAskFetch } from "./polled-ask-fetch";

afterEach(() => vi.unstubAllGlobals());

describe("polledAskFetch", () => {
  it("starts a job and returns, as one response, what its polls stream", async () => {
    const polls = [
      { status: null, contentType: null, text: "", next: 0, done: false },
      { status: 200, contentType: "text/event-stream", text: "data: a\n\n", next: 9, done: false },
      { status: 200, contentType: "text/event-stream", text: "data: b\n\n", next: 18, done: true },
    ];
    const calls: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push(`${init?.method ?? "GET"} ${url}`);
        if (url.endsWith("/api/ask/jobs")) return Response.json({ jobId: "job-1" }, { status: 202 });
        return Response.json(polls.shift());
      }),
    );
    vi.useFakeTimers({ shouldAdvanceTime: true, advanceTimeDelta: 200 });
    try {
      const response = await polledAskFetch("http://console/api/ask", { method: "POST", body: "{}" });
      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toBe("text/event-stream");
      expect(await response.text()).toBe("data: a\n\ndata: b\n\n");
    } finally {
      vi.useRealTimers();
    }
    expect(calls[0]).toBe("POST http://console/api/ask/jobs");
    expect(calls.slice(1).every((call) => call.startsWith("GET http://console/api/ask/jobs/job-1?after="))).toBe(true);
  });

  it("returns the refusal when the job cannot start", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ error: "busy" }, { status: 429 })));
    const response = await polledAskFetch("http://console/api/ask", { method: "POST", body: "{}" });
    expect(response.status).toBe(429);
  });
});
