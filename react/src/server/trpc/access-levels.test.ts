import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { writeRunRecord } from "../catalog";
import { appEnv, resetAppEnv } from "../env";
import { createApiKey, defaultGrants, loadWorkspace } from "../workspace";
import { createContext } from "./init";
import { appRouter } from "./router";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

/** A console with its own empty state, configured by `flags`. */
async function consoleWith(flags: Record<string, string>) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-access-levels-"));
  for (const name of [
    "FLAKEGRAPH_TRUST_IDENTITY_HEADERS",
    "FLAKEGRAPH_DEMO_IDENTITIES",
    "FLAKEGRAPH_APP_ANONYMOUS_READ",
    "FLAKEGRAPH_APP_REQUIRE_SIGN_IN",
    "FLAKEGRAPH_APP_OPERATORS",
    "FLAKEGRAPH_SINGLE_OPERATOR",
    "FLAKEGRAPH_SNOWFLAKE_HOSTED",
  ]) {
    delete process.env[name];
  }
  process.env.FLAKEGRAPH_APP_DEFAULT_RUNTIME = "local";
  process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
  Object.assign(process.env, flags);
  resetAppEnv();
}

async function caller(headers: Record<string, string>) {
  return appRouter.createCaller(await createContext({ headers: new Headers(headers) }));
}

const GATED = { FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1", FLAKEGRAPH_APP_REQUIRE_SIGN_IN: "1" };
const alice = { "x-auth-request-email": "alice@example.test" };
const bob = { "x-auth-request-email": "bob@example.test" };
/** What reaches the console when an application presents a person's access token to the gate. */
const aliceByToken = { ...alice, authorization: "Bearer eyJhbGciOiJSUzI1NiJ9.e30.synthetic-signature" };

describe("key management", () => {
  it("is refused to a key, so a leaked key mints nothing and revokes none of its owner's keys", async () => {
    await consoleWith(GATED);
    const { secret } = await createApiKey("ci", "ALICE@EXAMPLE.TEST");
    const other = (await createApiKey("deploy", "ALICE@EXAMPLE.TEST")).state.apiKeys.find((key) => key.name === "deploy")!;
    const leaked = await caller({ authorization: `Bearer ${secret}` });

    // The key is served as Alice, and still may not act on her keys.
    expect((await leaked.auth.session()).viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    await expect(leaked.workspace.createKey({ name: "successor" })).rejects.toMatchObject({ code: "FORBIDDEN" });
    await expect(leaked.workspace.revokeKey({ id: other.id })).rejects.toMatchObject({ code: "FORBIDDEN" });
    const viaHeader = await caller({ "x-flakegraph-api-key": secret });
    await expect(viaHeader.workspace.createKey({ name: "successor" })).rejects.toMatchObject({ code: "FORBIDDEN" });

    expect((await loadWorkspace(appEnv().stateRoot)).apiKeys.map((key) => key.name).sort()).toEqual(["ci", "deploy"]);
  });

  it("is refused to a delegated access token, and served to the same person's browser session", async () => {
    await consoleWith(GATED);
    const delegated = await caller(aliceByToken);
    expect((await delegated.auth.session()).viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    await expect(delegated.workspace.createKey({ name: "from-an-app" })).rejects.toMatchObject({ code: "FORBIDDEN" });

    const browser = await caller(alice);
    const minted = await browser.workspace.createKey({ name: "mine" });
    expect(minted.secret).toMatch(/^fg_/);
    const id = minted.state.apiKeys[0]!.id;
    await expect(delegated.workspace.revokeKey({ id })).rejects.toMatchObject({ code: "FORBIDDEN" });
    expect((await browser.workspace.revokeKey({ id })).apiKeys).toEqual([]);
  });
});

describe("console-wide settings", () => {
  const changes = (client: Awaited<ReturnType<typeof caller>>) => [
    () => client.auth.setSuggestions({ mode: "off" }),
    () => client.workspace.markGrant({ object: defaultGrants()[2]!.object, ok: true }),
    () =>
      client.workspace.identityIncident({ expectedPrincipal: "ALICE", actualPrincipal: "subject-1", graphOwner: "BOB", grants: "none" }),
  ];

  it("are closed to everyone behind the gate until operators are named", async () => {
    await consoleWith(GATED);
    const signedIn = await caller(bob);
    expect((await signedIn.auth.session()).operator).toBe(false);
    for (const change of changes(signedIn)) {
      await expect(change()).rejects.toMatchObject({ code: "FORBIDDEN" });
    }
    const state = await loadWorkspace(appEnv().stateRoot);
    expect([state.suggestionMode, state.grants[2]!.ok, state.identityIncident]).toEqual(["on-request", false, null]);
  });

  it("are open to the people and roles FLAKEGRAPH_APP_OPERATORS names, in their own session only", async () => {
    await consoleWith({ ...GATED, FLAKEGRAPH_APP_OPERATORS: "alice@example.test, FLAKEGRAPH_STAFF" });
    const operator = await caller(alice);
    expect((await operator.auth.session()).operator).toBe(true);
    for (const change of changes(operator)) {
      await change();
    }
    const staff = await caller({ ...bob, "x-auth-request-groups": "analysts,flakegraph_staff" });
    await staff.auth.setSuggestions({ mode: "auto-fill" });
    expect((await loadWorkspace(appEnv().stateRoot)).suggestionMode).toBe("auto-fill");

    // Not Bob without the role, and not Alice through a key or an access token.
    for (const change of changes(await caller(bob))) {
      await expect(change()).rejects.toMatchObject({ code: "FORBIDDEN" });
    }
    const { secret } = await createApiKey("ci", "ALICE@EXAMPLE.TEST");
    for (const client of [await caller({ authorization: `Bearer ${secret}` }), await caller(aliceByToken)]) {
      for (const change of changes(client)) {
        await expect(change()).rejects.toMatchObject({ code: "FORBIDDEN" });
      }
    }
  });

  it("are the operator's on a console one person runs, and never a key's", async () => {
    await consoleWith({ FLAKEGRAPH_SINGLE_OPERATOR: "solo@example.test" });
    await (await caller({})).auth.setSuggestions({ mode: "off" });

    await consoleWith({ FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    const laptop = await caller({});
    await laptop.auth.assume({ userName: "alice", roles: [], role: "operator" });
    const { secret } = await createApiKey("ci", "ALICE");
    const key = await caller({ authorization: `Bearer ${secret}` });
    await expect(key.auth.assume({ userName: "mallory", roles: [], role: "staff" })).rejects.toMatchObject({ code: "FORBIDDEN" });
    await expect(key.auth.setSuggestions({ mode: "off" })).rejects.toMatchObject({ code: "FORBIDDEN" });
    expect((await loadWorkspace(appEnv().stateRoot)).identity?.userName).toBe("ALICE");
  });
});

describe("ingestion.estimate", () => {
  const input = { objectCount: 10, runtime: "local", ocrProvider: "built_in_text", parallelism: 4 };

  it("only estimates; keeping an estimate on a run is a mutation that needs write access to it", async () => {
    await consoleWith({ ...GATED });
    await writeRunRecord(path.join(appEnv().stateRoot, "runs", "run_alice"), {
      runId: "run_alice",
      graphId: "graph_alice",
      status: "succeeded",
      runtime: "local",
      owner: "ALICE@EXAMPLE.TEST",
    });
    const owner = await caller(alice);

    expect((await owner.ingestion.estimate(input)).usdHigh).toBeGreaterThan(0);
    await expect(owner.ingestion.estimate({ ...input, runId: "run_alice" } as typeof input)).rejects.toMatchObject({
      code: "BAD_REQUEST",
    });
    expect((await loadWorkspace(appEnv().stateRoot)).estimates).toEqual({});

    await expect((await caller(bob)).ingestion.recordEstimate({ ...input, runId: "run_alice" })).rejects.toMatchObject({
      code: "FORBIDDEN",
    });
    const recorded = await owner.ingestion.recordEstimate({ ...input, runId: "run_alice" });
    expect((await loadWorkspace(appEnv().stateRoot)).estimates).toEqual({ run_alice: recorded });
  });
});
