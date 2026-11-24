import { describe, expect, it, vi } from "vitest";

let release: () => void = () => undefined;
vi.mock("./stream", () => ({
  handleAskRequest: async (request: Request) => {
    const body = (await request.json()) as { question: string };
    if (body.question === "refused") {
      return Response.json({ error: "no model" }, { status: 409 });
    }
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        controller.enqueue(encoder.encode("data: first\n\n"));
        await new Promise<void>((resolve) => (release = resolve));
        controller.enqueue(encoder.encode("data: second\n\n"));
        controller.close();
      },
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } });
  },
}));

const { cancelAskJob, readAskJob, startAskJob } = await import("./jobs");
const ask = (question: string) =>
  new Request("http://console/api/ask/jobs", { method: "POST", body: JSON.stringify({ runId: "run", question }) });
const settle = () => new Promise((resolve) => setTimeout(resolve, 10));

describe("question jobs", () => {
  it("keeps what the answer streams for its owner to read in pieces until it is done", async () => {
    const id = await startAskJob(ask("What changed?"), "ana@example.com");
    await settle();
    const first = readAskJob(id, "ana@example.com", 0);
    expect(first).toMatchObject({ status: 200, contentType: "text/event-stream", text: "data: first\n\n", done: false });
    expect(readAskJob(id, "ben@example.com", 0)).toBeNull();

    release();
    await settle();
    expect(readAskJob(id, "ana@example.com", first!.next)).toMatchObject({ text: "data: second\n\n", done: true });
  });

  it("passes on a refusal's status, and stops a job its owner cancels", async () => {
    const refused = await startAskJob(ask("refused"), "ana@example.com");
    await settle();
    expect(readAskJob(refused, "ana@example.com", 0)).toMatchObject({ status: 409, done: true });

    const running = await startAskJob(ask("What else?"), "ana@example.com");
    expect(cancelAskJob(running, "ben@example.com")).toBe(false);
    expect(cancelAskJob(running, "ana@example.com")).toBe(true);
    expect(readAskJob(running, "ana@example.com", 0)).toBeNull();
    release();
  });
});
