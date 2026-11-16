import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { NO_ACCESS } from "@/server/access/controlled-plane";
import { writeRunRecord } from "@/server/catalog";
import { appEnv, resetAppEnv } from "@/server/env";
import { GET } from "./route";

const original = { ...process.env };
const ORIGIN = "https://console.example.test";

beforeEach(async () => {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-inspect-"));
  for (const name of ["FLAKEGRAPH_DEMO_IDENTITIES", "FLAKEGRAPH_APP_ANONYMOUS_READ", "FLAKEGRAPH_SINGLE_OPERATOR", "FLAKEGRAPH_SNOWFLAKE_HOSTED"]) {
    delete process.env[name];
  }
  process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS = "1";
  process.env.FLAKEGRAPH_APP_REQUIRE_SIGN_IN = "1";
  process.env.FLAKEGRAPH_APP_DEFAULT_RUNTIME = "local";
  process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
  resetAppEnv();
  // A fleet run of Alice's graph.
  await writeRunRecord(path.join(appEnv().stateRoot, "runs", "run_fleet"), {
    runId: "run_fleet",
    graphId: "graph_fleet",
    status: "succeeded",
    runtime: "kubernetes",
    owner: "ALICE@EXAMPLE.TEST",
  });
});

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

const inspect = (runId: string, headers: Record<string, string> = {}) =>
  GET(new Request(`${ORIGIN}/api/inspect?run=${runId}`, { headers }));

describe("GET /api/inspect", () => {
  it("answers a caller nobody vouched for with 401", async () => {
    const response = await inspect("run_fleet");
    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: expect.any(String) });
  });

  it("answers 403 for a graph the caller cannot read, as tRPC does", async () => {
    const response = await inspect("run_fleet", { "x-auth-request-email": "bob@example.test" });
    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ error: NO_ACCESS });
  });

  it("serves the runtime x-flakegraph-runtime names, not the one the run's record does", async () => {
    const response = await inspect("run_fleet", { "x-auth-request-email": "alice@example.test", "x-flakegraph-runtime": "local" });
    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: expect.stringMatching(/belongs to the kubernetes runtime/) });
  });

  it("answers 404 for a run no runtime holds, and 400 for a run id that names a path", async () => {
    const missing = await inspect("run_nowhere", { "x-auth-request-email": "alice@example.test" });
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: expect.any(String) });
    const escaped = await inspect("..%2F..%2Fetc", { "x-auth-request-email": "alice@example.test" });
    expect(escaped.status).toBe(400);
  });
});
