import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { TRPCError } from "@trpc/server";
import { afterEach, describe, expect, it } from "vitest";
import { resetAppEnv } from "../env";
import { createContext } from "./init";
import { appRouter } from "./router";
import { emptyWorkspace, saveWorkspace, loadWorkspace, recordRuntimeSuccess, revokeApiKey, setSuggestionMode } from "../workspace";
import { appEnv } from "../env";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

async function context(headers: Record<string, string>, flags: Record<string, string>) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-context-"));
  delete process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS;
  delete process.env.FLAKEGRAPH_DEMO_IDENTITIES;
  delete process.env.FLAKEGRAPH_APP_ANONYMOUS_READ;
  process.env.FLAKEGRAPH_APP_DEFAULT_RUNTIME = "kubernetes";
  process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
  Object.assign(process.env, flags);
  resetAppEnv();
  return createContext({ headers: new Headers(headers) });
}

describe("request context", () => {
  it("filters mutation responses and session metadata through caller access", async () => {
    const ctx = await context({ "x-auth-request-email": "bob@example.test" }, {
      FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1", FLAKEGRAPH_APP_REQUIRE_SIGN_IN: "1", FLAKEGRAPH_APP_OPERATORS: "bob@example.test",
    });
    await saveWorkspace({ ...emptyWorkspace(),
      reviews: [{ id: "private", graphId: "alice_graph", edgeId: "e", quote: "PRIVATE_QUOTE", confidence: 1, decision: null }],
      apiKeys: [{ id: "alice_key", name: "private key", owner: "ALICE@EXAMPLE.TEST", secretHash: "synthetic-hash", preview: "fixture", createdAt: "", note: "" }],
    });
    await recordRuntimeSuccess("kubernetes", "local_path:/synthetic/alice/documents");
    const caller = appRouter.createCaller(ctx);
    const response = await caller.auth.setSuggestions({ mode: "off" });
    expect(response.reviews).toEqual([]);
    expect(response.apiKeys).toEqual([]);
    expect(response).not.toHaveProperty("lastConfigDigest");
    expect(await caller.auth.session()).not.toHaveProperty("lastConfigDigest");
    const key = await caller.workspace.createKey({ name: "bob" });
    expect(key.secret).toMatch(/^fg_/);
    expect(key.state.apiKeys).toHaveLength(1);
    expect(key.state.apiKeys[0]).not.toHaveProperty("secretHash");
    expect(key.state.reviews).toEqual([]);
  });

  it("keeps revocations committed during overlapping preference writes", async () => {
    await context({}, { FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    for (let i = 0; i < 30; i++) {
      await saveWorkspace({ ...emptyWorkspace(), apiKeys: [{ id: "key", name: "fixture", owner: "ALICE", secretHash: "synthetic", preview: "fixture", createdAt: "", note: "" }] });
      await Promise.all([revokeApiKey("key", "ALICE"), setSuggestionMode("off")]);
      const state = await loadWorkspace(appEnv().stateRoot);
      expect(state.apiKeys).toEqual([]);
      expect(state.suggestionMode).toBe("off");
    }
  });
  it("takes the gate's identity when the gate is trusted", async () => {
    const ctx = await context(
      { "x-auth-request-email": "alice@example.test", "x-auth-request-user": "c7s1v9-opaque-subject" },
      { FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" },
    );
    // The address, not the provider's opaque subject id, is who the viewer is.
    expect(ctx.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(ctx.viewer.email).toBe("alice@example.test");
    expect(ctx.authorization).toBe("gate");
  });

  it("refuses identity headers nobody vouched for", async () => {
    await expect(context({ "x-auth-request-user": "mallory" }, {})).rejects.toBeInstanceOf(TRPCError);
    // Even where anonymous reads are open, the header does not make them mallory.
    const ctx = await context({ "x-auth-request-user": "mallory" }, { FLAKEGRAPH_APP_ANONYMOUS_READ: "1" });
    expect(ctx.viewer.userName).toBe("");
    expect(ctx.authorization).toBe("anonymous");
  });

  it("opens on the configured runtime when the address names none", async () => {
    const ctx = await context({}, { FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    expect(ctx.runtimeName).toBe("kubernetes");
    expect(ctx.authorization).toBe("demo");
    const named = await context({ "x-flakegraph-runtime": "local" }, { FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    expect(named.runtimeName).toBe("local");
  });
});
