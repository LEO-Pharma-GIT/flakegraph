import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GET as inspect } from "@/app/api/inspect/route";
import { resetAppEnv } from "../env";
import { RecordingSql } from "../snowflake/recording-sql";
import { SnowflakeRuntime } from "../runtime/snowflake";
import { resolveRuntime } from "../runtime/resolve";
import { createApiKey } from "../workspace";
import { createContext } from "./init";
import { appRouter } from "./router";

const account = vi.hoisted(() => ({ sql: null as RecordingSql | null, callerRoles: [] as string[] }));

vi.mock("../snowflake/client", async (load) => ({
  ...(await load<typeof import("../snowflake/client")>()),
  sharedSnowflakeSql: () => account.sql,
  callerAvailableRoles: async () => account.callerRoles,
}));

const original = { ...process.env };

async function hosted(flags: Record<string, string> = {}, sql = new RecordingSql()) {
  account.sql = sql;
  account.callerRoles = [];
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-hosted-"));
  process.env.FLAKEGRAPH_SNOWFLAKE_HOSTED = "1";
  delete process.env.FLAKEGRAPH_STUB_RUNTIMES;
  delete process.env.FLAKEGRAPH_APP_OPERATORS;
  Object.assign(process.env, flags);
  resetAppEnv();
}

const signedIn = (user: string, extra: Record<string, string> = {}) => ({ "sf-context-current-user": user, ...extra });
const caller = async (headers: Record<string, string>) =>
  appRouter.createCaller(await createContext({ headers: new Headers(headers) }));

/** A graph ALICE owns, with one run. */
const alicesGraph = () =>
  new RecordingSql([
    ["SELECT OWNER FROM KG_GRAPH", [{ owner: "ALICE" }]],
    ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "alices-graph" }]],
  ]);

beforeEach(() => {
  process.env = { ...original };
});

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

describe("a console hosted in Snowflake", () => {
  it("serves the runtime itself as the access plane, with no console store around it", async () => {
    await hosted();
    expect(resolveRuntime("local", { userName: "ALICE", email: "", roles: [] })).toBeInstanceOf(SnowflakeRuntime);
  });

  it("names whoever the ingress names, and offers no Ask without a model", async () => {
    await hosted();
    const session = await (await caller(signedIn("alice"))).auth.session();
    expect(session).toMatchObject({ runtime: "snowflake", authorization: "gate", snowflakeHosted: true, demoIdentities: false, operator: false });
    expect(session.viewer.userName).toBe("ALICE");
    expect(session.capabilities).toContain("share");
    expect(session.capabilities).not.toContain("ask");
    expect(session.availableRuntimes).toEqual(["snowflake"]);
  });

  it("refuses a caller the ingress did not name, whatever the deployment opened", async () => {
    await hosted({ FLAKEGRAPH_DEMO_IDENTITIES: "1", FLAKEGRAPH_APP_ANONYMOUS_READ: "1" });
    await expect(createContext({ headers: new Headers() })).rejects.toMatchObject({ code: "UNAUTHORIZED" });
  });

  it("assumes no identity, operator or not", async () => {
    await hosted({ FLAKEGRAPH_DEMO_IDENTITIES: "1", FLAKEGRAPH_APP_OPERATORS: "ALICE" });
    const alice = await caller(signedIn("alice"));
    await expect(alice.auth.assume({ userName: "BOB", roles: [], role: "operator" })).rejects.toMatchObject({ code: "FORBIDDEN" });
  });

  it("takes an operator's role from Snowflake, never from a header beside the name", async () => {
    await hosted({ FLAKEGRAPH_APP_OPERATORS: "console_admins" });
    const forged = await caller(signedIn("mallory", { "sf-context-current-user-roles": "CONSOLE_ADMINS" }));
    await expect(forged.auth.setSuggestions({ mode: "off" })).rejects.toMatchObject({ code: "FORBIDDEN" });
    await expect(forged.workspace.markGrant({ object: "x", ok: true })).rejects.toMatchObject({ code: "FORBIDDEN" });

    await hosted({ FLAKEGRAPH_APP_OPERATORS: "console_admins" }, new RecordingSql([["SHOW GRANTS TO USER", [{ role: "CONSOLE_ADMINS" }]]]));
    const granted = await caller(signedIn("alice"));
    expect((await granted.auth.session()).operator).toBe(true);
    expect((await granted.auth.setSuggestions({ mode: "off" })).suggestionMode).toBe("off");
  });

  it("asks a caller's-rights session for the viewer's roles when the request carries their token", async () => {
    await hosted({ FLAKEGRAPH_APP_OPERATORS: "console_admins" });
    account.callerRoles = ["CONSOLE_ADMINS"];
    const session = await (await caller(signedIn("alice", { "sf-context-current-user-token": "viewer-token" }))).auth.session();
    expect(session.viewer.roles).toEqual(["CONSOLE_ADMINS"]);
    expect(session.operator).toBe(true);
  });

  it("names operators by user too, and never a program", async () => {
    await hosted({ FLAKEGRAPH_APP_OPERATORS: "alice" });
    expect((await (await caller(signedIn("alice"))).auth.session()).operator).toBe(true);
    const program = await caller(signedIn("alice", { authorization: 'Snowflake Token="synthetic"' }));
    await expect(program.auth.setSuggestions({ mode: "off" })).rejects.toMatchObject({ code: "FORBIDDEN" });
    await expect(program.workspace.createKey({ name: "successor" })).rejects.toMatchObject({ code: "FORBIDDEN" });
  });

  it("serves a key as the person who minted it, not the token that carried it", async () => {
    await hosted({}, new RecordingSql([["SHOW GRANTS TO USER", [{ role: "ALICES_ROLE" }]]]));
    account.callerRoles = ["BOBS_ROLE"];
    const { secret } = await createApiKey("ci", "ALICE");
    const session = await (await caller(signedIn("bob", { authorization: `Bearer ${secret}`, "sf-context-current-user-token": "bobs-token" }))).auth.session();
    expect(session.viewer).toMatchObject({ userName: "ALICE", roles: ["ALICES_ROLE"] });
    expect(session.authorization).toBe("machine");
  });

  it("keeps an estimate off a run the viewer cannot write", async () => {
    await hosted({}, alicesGraph());
    const bob = await caller(signedIn("bob"));
    await expect(
      bob.ingestion.recordEstimate({ runId: "alices-run", objectCount: 1, runtime: "snowflake", ocrProvider: "snowflake_cortex", parallelism: 1 }),
    ).rejects.toMatchObject({ code: "FORBIDDEN" });
  });

  it("answers the inspect report with the viewer's own access", async () => {
    await hosted({}, alicesGraph());
    const response = await inspect(new Request("http://console/api/inspect?run=alices-run", { headers: signedIn("bob") }));
    expect(response.status).toBe(403);
  });

  it("refuses a question rather than answer without a model", async () => {
    await hosted({}, alicesGraph());
    const alice = await caller(signedIn("alice"));
    await expect(alice.graphs.ask({ runId: "alices-run", question: "Who founded judo?", mode: "local" })).rejects.toMatchObject({ code: "CONFLICT" });
  });
});
