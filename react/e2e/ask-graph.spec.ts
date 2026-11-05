import { expect, test, type APIRequestContext } from "@playwright/test";
import { collectAskText, type AskStreamEvent } from "../src/server/ask/protocol";

test.describe.configure({ timeout: 180_000 });

const RUN = "run_martial_arts";
const HEADERS = {
  "content-type": "application/json",
  "x-flakegraph-runtime": "local",
};

async function requireAskModel(request: APIRequestContext) {
  const response = await request.get("/api/ask");
  expect(response.ok(), await response.text()).toBeTruthy();
  const body = (await response.json()) as { model?: { configured?: boolean; provider?: string } };
  // A live model is an environment, not a fixture: without one these
  // journeys are skipped, and the lexical fallback's are exercised instead.
  test.skip(
    !body.model?.configured,
    "Ask model is not configured. Set FLAKEGRAPH_ASK_API_KEY and FLAKEGRAPH_ASK_BASE_URL (or FLAKEGRAPH_ASK_SECRETS_FILE), or start Ollama.",
  );
}

async function readNdjson(response: { text: () => Promise<string> }): Promise<AskStreamEvent[]> {
  const text = await response.text();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as AskStreamEvent);
}

async function askNdjson(
  request: APIRequestContext,
  body: Record<string, unknown>,
  headers: Record<string, string> = {},
) {
  return request.post("/api/ask", {
    headers: { ...HEADERS, accept: "application/x-ndjson", ...headers },
    data: { format: "ndjson", ...body },
    timeout: 120_000,
  });
}

interface Principal {
  userName: string;
  email?: string;
  roles: string[];
  role: "operator" | "analyst" | "staff";
}

/** The principal the identity dialog's "Sign in as ALICE" assumes. */
const ALICE: Principal = { userName: "ALICE", roles: ["APP_OPERATOR"], role: "operator" };

/**
 * Assume a demo principal the way the identity dialog does. The e2e server
 * runs with FLAKEGRAPH_DEMO_IDENTITIES=1, and the assumed identity is
 * workspace state that outlives this test.
 */
async function assumePrincipal(request: APIRequestContext, principal: Principal) {
  const response = await request.post("/api/trpc/auth.assume", {
    headers: HEADERS,
    data: { json: principal },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
}

/** Whoever the workspace currently treats as signed in, so a test can hand it back. */
async function currentPrincipal(request: APIRequestContext): Promise<Principal> {
  const response = await request.get(
    `/api/trpc/auth.session?input=${encodeURIComponent(JSON.stringify({ json: null }))}`,
    { headers: HEADERS },
  );
  expect(response.ok(), await response.text()).toBeTruthy();
  const payload = (await response.json()) as {
    result: { data: { json: { viewer: { userName: string; email?: string; roles?: string[] }; role: Principal["role"] } } };
  };
  const { viewer, role } = payload.result.data.json;
  return { userName: viewer.userName, email: viewer.email ?? "", roles: [...(viewer.roles ?? [])], role };
}

/** Run `work` signed in as `principal`, then restore whoever was signed in before. */
async function signedInAs<T>(request: APIRequestContext, principal: Principal, work: () => Promise<T>): Promise<T> {
  const previous = await currentPrincipal(request);
  await assumePrincipal(request, principal);
  try {
    return await work();
  } finally {
    await assumePrincipal(request, previous);
  }
}

async function createApiKey(request: APIRequestContext): Promise<{ secret: string; id: string }> {
  const response = await request.post("/api/trpc/workspace.createKey", {
    headers: HEADERS,
    data: { json: { name: `ask-e2e-${Date.now()}` } },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
  const payload = (await response.json()) as {
    result?: { data?: { json?: { secret?: string; state?: { apiKeys?: Array<{ id: string }> } } } };
  };
  const secret = payload.result?.data?.json?.secret;
  const id = payload.result?.data?.json?.state?.apiKeys?.at(-1)?.id;
  expect(secret).toMatch(/^fg_/);
  expect(id).toBeTruthy();
  return { secret: secret!, id: id! };
}

async function revokeApiKey(request: APIRequestContext, id: string) {
  const response = await request.post("/api/trpc/workspace.revokeKey", {
    headers: HEADERS,
    data: { json: { id } },
  });
  expect(response.ok(), await response.text()).toBeTruthy();
}

test.describe("ask HTTP contract", () => {
  test("GET /api/ask describes the stream protocol", async ({ request }) => {
    const response = await request.get("/api/ask");
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body.path).toBe("/api/ask");
    expect(body.formats.ndjson).toMatch(/status/);
    expect(body.body.runId).toBe("required");
    expect(body.tools).toEqual(
      expect.arrayContaining(["searchGraph", "listDocuments", "getEntityEvidence", "getRelationEvidence", "readDocumentEvidence"]),
    );
    expect(body.body.documentIds).toMatch(/document/);
  });

  test("rejects missing runId, short questions, and invalid modes", async ({ request }) => {
    const missingRun = await request.post("/api/ask", { headers: HEADERS, data: { question: "Who developed judo?" } });
    expect(missingRun.status()).toBe(400);
    expect((await missingRun.json()).code).toBe("missing_run");

    const missingQuestion = await request.post("/api/ask", { headers: HEADERS, data: { runId: RUN } });
    expect(missingQuestion.status()).toBe(400);
    expect((await missingQuestion.json()).code).toBe("missing_question");

    const short = await request.post("/api/ask", { headers: HEADERS, data: { runId: RUN, question: "a" } });
    expect(short.status()).toBe(400);

    const invalidMode = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: RUN, question: "Who developed judo?", mode: "sideways" },
    });
    expect(invalidMode.status()).toBe(400);
    expect((await invalidMode.json()).code).toBe("invalid_mode");

    const invalidDocuments = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: RUN, question: "Who developed judo?", documentIds: "doc_overview" },
    });
    expect(invalidDocuments.status()).toBe(400);
    expect((await invalidDocuments.json()).code).toBe("invalid_document_ids");
  });

  test("rejects an unknown perspective", async ({ request }) => {
    const response = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: RUN, question: "Who developed judo?", perspectiveId: "perspective_missing" },
    });
    expect(response.status()).toBe(400);
    expect((await response.json()).code).toBe("unknown_perspective");
  });

  test("returns 404 for an unknown run", async ({ request }) => {
    const response = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: "run_does_not_exist", question: "Who developed judo?" },
    });
    expect(response.status()).toBe(404);
  });

  test("returns 409 when the graph is not finished", async ({ request }) => {
    const failed = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: "run_failed_llm", question: "Who developed judo?" },
    });
    expect(failed.status()).toBe(409);
    expect((await failed.json()).code).toBe("not_ready");

    const queued = await request.post("/api/ask", {
      headers: { ...HEADERS, "x-flakegraph-runtime": "snowflake" },
      data: { runId: "run_snow_queued", question: "Who developed judo?" },
    });
    expect(queued.status()).toBe(409);
  });

  test("returns 409 when artifacts are missing", async ({ request }) => {
    const response = await request.post("/api/ask", {
      headers: HEADERS,
      data: { runId: "run_missing_artifacts", question: "Who developed judo?" },
    });
    expect(response.status()).toBe(409);
  });

  test("returns 401 for a presented but invalid API key", async ({ request }) => {
    const response = await request.post("/api/ask", {
      headers: { ...HEADERS, authorization: "Bearer fg_revoked_or_wrong" },
      data: { runId: RUN, question: "Who developed judo?" },
    });
    expect(response.status()).toBe(401);
  });
});

test.describe("ask live model stream", () => {
  test("streams NDJSON status, tools, tokens, and a done event", async ({ request }) => {
    await requireAskModel(request);
    const response = await askNdjson(request, {
      runId: RUN,
      question: "Who developed judo?",
      mode: "local",
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    expect(response.headers()["content-type"]).toContain("application/x-ndjson");
    expect(response.headers()["x-flakegraph-ask-protocol"]).toBe("ndjson-v1");
    const events = await readNdjson(response);
    expect(events.some((event) => event.type === "error"), JSON.stringify(events)).toBeFalsy();
    expect(events.some((event) => event.type === "status" && event.phase === "loading_graph")).toBeTruthy();
    expect(events.some((event) => event.type === "status")).toBeTruthy();
    expect(events.some((event) => event.type === "tool")).toBeTruthy();
    const text = collectAskText(events).toLowerCase();
    expect(text.length).toBeGreaterThan(8);
    expect(text).toMatch(/kano|judo|kodokan/);
    const done = events.find((event) => event.type === "done");
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.answer.summary.toLowerCase()).toMatch(/kano|judo|kodokan/);
      expect(done.answer.entityIds.length + done.answer.citations.length).toBeGreaterThan(0);
    }
    expect(events.some((event) => event.type === "error")).toBeFalsy();
  });

  test("streams a UI message response for useChat consumers", async ({ request }) => {
    await requireAskModel(request);
    const response = await request.post("/api/ask", {
      headers: HEADERS,
      data: {
        runId: RUN,
        format: "ui",
        messages: [{ id: "u1", role: "user", parts: [{ type: "text", text: "Who developed judo?" }] }],
        mode: "local",
      },
      timeout: 120_000,
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    expect(response.headers()["x-vercel-ai-ui-message-stream"]).toBeTruthy();
    const body = await response.text();
    expect(body.length).toBeGreaterThan(20);
  });

  test("answers global, hybrid, and drift questions", async ({ request }) => {
    await requireAskModel(request);
    for (const mode of ["global", "hybrid", "drift"] as const) {
      const response = await askNdjson(request, {
        runId: RUN,
        question: mode === "global" ? "What themes appear across martial arts?" : "How did judo spread from Japan?",
        mode,
      });
      expect(response.ok(), `${mode}: ${await response.text()}`).toBeTruthy();
      const events = await readNdjson(response);
      expect(events.some((event) => event.type === "error"), `${mode}: ${JSON.stringify(events)}`).toBeFalsy();
      const done = events.find((event) => event.type === "done");
      expect(done?.type, mode).toBe("done");
      expect(collectAskText(events).length, mode).toBeGreaterThan(8);
    }
  });

  test("scopes retrieval to the Judo perspective", async ({ request }) => {
    await requireAskModel(request);
    const response = await askNdjson(request, {
      runId: RUN,
      question: "Who developed this art?",
      mode: "local",
      perspectiveId: "perspective_judo",
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const events = await readNdjson(response);
    const done = events.find((event) => event.type === "done");
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.answer.scoped).toBe(true);
    }
  });

  test("restricts retrieval to documentIds like hub fileIds", async ({ request }) => {
    await requireAskModel(request);
    const response = await askNdjson(request, {
      runId: RUN,
      question: "Who developed judo?",
      mode: "local",
      documentIds: ["doc_overview"],
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const events = await readNdjson(response);
    expect(events.some((event) => event.type === "error"), JSON.stringify(events)).toBeFalsy();
    const text = collectAskText(events).toLowerCase();
    expect(text).toMatch(/kano|judo|kodokan/);
    const done = events.find((event) => event.type === "done");
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.answer.scoped).toBe(true);
    }
    expect(events.some((event) => event.type === "tool")).toBeTruthy();
  });

  test("says it cannot find an entity that is not in the graph", async ({ request }) => {
    await requireAskModel(request);
    const response = await askNdjson(request, {
      runId: RUN,
      question: "Who is Admiral Horatio Nelson of Trafalgar?",
      mode: "local",
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const events = await readNdjson(response);
    expect(events.some((event) => event.type === "error"), JSON.stringify(events)).toBeFalsy();
    const text = collectAskText(events).toLowerCase();
    expect(text).not.toMatch(/developed judo/);
    expect(text.length).toBeGreaterThan(5);
  });

  test("authenticates a machine key for the stream", async ({ request }) => {
    await requireAskModel(request);
    // A key belongs to whoever minted it, so minting one needs someone signed in.
    await signedInAs(request, ALICE, async () => {
      const key = await createApiKey(request);
      try {
        const response = await askNdjson(
          request,
          { runId: RUN, question: "Who developed judo?", mode: "local" },
          { authorization: `Bearer ${key.secret}` },
        );
        expect(response.ok(), await response.text()).toBeTruthy();
        const events = await readNdjson(response);
        expect(events.some((event) => event.type === "done" || event.type === "text")).toBeTruthy();
      } finally {
        await revokeApiKey(request, key.id);
      }
    });
  });

  test("graphs.ask JSON convenience still returns an answer", async ({ request }) => {
    await requireAskModel(request);
    const response = await request.post("/api/trpc/graphs.ask", {
      headers: HEADERS,
      data: { json: { runId: RUN, question: "Who developed judo?", mode: "local" } },
      timeout: 120_000,
    });
    expect(response.ok(), await response.text()).toBeTruthy();
    const payload = (await response.json()) as {
      result?: { data?: { json?: { summary?: string; entityIds?: string[] } } };
    };
    const answer = payload.result?.data?.json;
    expect(answer?.summary?.length).toBeGreaterThan(8);
  });

  test("cancels an in-flight NDJSON stream", async ({ request, baseURL }) => {
    await requireAskModel(request);
    const controller = new AbortController();
    const response = await fetch(`${baseURL}/api/ask`, {
      method: "POST",
      headers: { ...HEADERS, accept: "application/x-ndjson" },
      body: JSON.stringify({ runId: RUN, question: "Who developed judo?", mode: "drift", format: "ndjson" }),
      signal: controller.signal,
    });
    expect(response.ok).toBeTruthy();
    const reader = response.body?.getReader();
    expect(reader).toBeTruthy();
    const first = await reader!.read();
    expect(first.done).toBeFalsy();
    await reader!.cancel();
    controller.abort();
  });
});

test.describe("ask console", () => {
  test("streams an answer in the Ask tab", async ({ page }) => {
    await page.goto("/?runtime=local&page=run&run=run_martial_arts");
    await page.getByRole("tab", { name: "Ask" }).click();
    await expect(page.getByRole("button", { name: "Auto", exact: true })).toHaveAttribute("aria-pressed", "true");
    await page.getByLabel("Ask the graph").fill("Who developed judo?");
    await page.getByRole("button", { name: "Ask", exact: true }).click();
    await expect(page.getByTestId("ask-status")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("ask-answer")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByTestId("ask-answer")).toContainText(/kano|judo|kodokan/i);
    await expect(page.getByTestId("query-consumption")).toBeVisible();
  });

  test("documents the streaming Ask endpoint on SDK keys", async ({ page }) => {
    await page.goto("/?runtime=local&page=keys");
    await page.getByRole("tab", { name: "Stream ask" }).click();
    await expect(page.getByTestId("sdk-ask-stream-example")).toContainText("/api/ask");
    await expect(page.getByTestId("sdk-ask-stream-example")).toContainText("application/x-ndjson");
    await page.getByRole("button", { name: "TypeScript", exact: true }).click();
    await expect(page.getByTestId("sdk-ask-stream-example")).toContainText("DefaultChatTransport");
    await expect(page.getByTestId("sdk-ask-stream-example")).toContainText("application/x-ndjson");
    const docs = await page.request.get("/api/docs");
    const catalog = await docs.json();
    expect(catalog.ask).toContain("/api/ask");
    expect(catalog.askStream.formats).toContain("ndjson");
  });
});

