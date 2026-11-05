import { mkdir, mkdtemp, realpath, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect, Either } from "effect";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parse as parseYaml } from "yaml";
import { resetAppEnv, type SnowflakeConnectionEnv } from "../env";
import { ControlPlaneError } from "../protocol/errors";
import type { IngestionRequest, Viewer } from "../protocol/schema";
import { claimUploadFolder, NO_ACCESS } from "../access/controlled-plane";
import { GRAPH_DATA_TABLES, REQUIRED_TABLES } from "../snowflake/graph-tables";
import { RecordingSql } from "../snowflake/recording-sql";
import { buildSpcsLaunch } from "../snowflake/spcs";
import { SnowflakeRuntime, shareId, type SnowflakeRuntimeOptions } from "./snowflake";

const REPOSITORY_ROOT = path.resolve(import.meta.dirname, "../../../..");
const CONNECTION: SnowflakeConnectionEnv = {
  account: "ACCT",
  host: "acct.snowflakecomputing.com",
  database: "DB",
  schema: "SCH",
  warehouse: "WH",
  tokenPath: "/nonexistent",
};

// A run's configuration is judged as a hosted console's: its session is the
// operator-approved destination.
beforeAll(() => {
  process.env.FLAKEGRAPH_SNOWFLAKE_HOSTED = "1";
  resetAppEnv();
});

afterAll(() => {
  delete process.env.FLAKEGRAPH_SNOWFLAKE_HOSTED;
  resetAppEnv();
});

function viewer(userName: string, roles: string[] = []): Viewer {
  return { userName, email: "", roles };
}

function runtime(
  sql: RecordingSql,
  who: Viewer = viewer("ALICE"),
  extra: Partial<SnowflakeRuntimeOptions> = {},
): SnowflakeRuntime {
  return new SnowflakeRuntime({
    repositoryRoot: REPOSITORY_ROOT,
    stateRoot: extra.stateRoot ?? path.join(tmpdir(), "fg-snowflake-unused"),
    sql,
    connection: CONNECTION,
    viewer: who,
    // Roles are asked of Snowflake; these tests answer with none unless they say so.
    callerRoles: async () => [],
    ...extra,
  });
}

async function failure(effect: Effect.Effect<unknown, ControlPlaneError>): Promise<ControlPlaneError> {
  const result = await Effect.runPromise(Effect.either(effect));
  if (Either.isRight(result)) {
    throw new Error("expected the operation to be refused");
  }
  return result.left;
}

function request(overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  const provider = (name: string, model: string | null, dimension: number | null = null) => ({
    provider: name,
    model,
    endpoint: null,
    apiKeyEnvironmentVariable: null,
    dimension,
    options: {},
  });
  return {
    runtime: "snowflake",
    jobId: "run-1",
    graphId: "graph-1",
    graphName: null,
    sourceKind: "snowflake_stage",
    source: { kind: "snowflake_stage", stage: "KG_DOCS", prefix: "corpus" },
    ocr: provider("snowflake_cortex", null),
    llm: provider("snowflake_cortex", "claude-4-sonnet"),
    embedding: provider("snowflake_cortex", "snowflake-arctic-embed-l-v2.0", 1024),
    output: {
      kind: "snowflake",
      workspacePath: "/client/chosen/path",
      snowflake: {
        account: "attacker-account",
        user: "MALLORY",
        database: "OTHER_DB",
        schema: "OTHER_SCHEMA",
        warehouse: "",
        bulkStage: "",
        role: "ACCOUNTADMIN",
        host: "evil.example.com",
        authenticator: null,
        credentialEnvironmentVariable: "SNOWFLAKE_PASSWORD",
        credentialField: "password",
      },
    },
    baseConfigPath: path.join(REPOSITORY_ROOT, "configs", "app-defaults.yaml"),
    includeGlobs: ["**/*"],
    cacheProvider: "local",
    providerParallelism: 4,
    runtimeOptions: {},
    ontology: null,
    revision: null,
    priority: "bulk",
    ...overrides,
  };
}

/** Answers a graph ALICE owns, which is how most of these tests start. */
const OWNED_BY_ALICE: [string, Array<Record<string, unknown>>] = ["SELECT OWNER FROM KG_GRAPH", [{ owner: "ALICE" }]];

/** Discovery answers for an account with one pool, one repository and two image tags. */
const DISCOVERY: Array<[string, Array<Record<string, unknown>>]> = [
  ["SHOW STAGES", [{ name: "KG_DOCS" }, { name: "KG_LOAD_STAGE" }, { name: "APP_STAGE" }]],
  ["SHOW COMPUTE POOLS LIKE", [{ name: "POOL_A", instance_family: "CPU_X64_M" }]],
  ["SHOW COMPUTE POOL INSTANCE FAMILIES", [{ name: "CPU_X64_M", vcpu: 4, memory_gib: 16 }]],
  ["SHOW COMPUTE POOLS", [{ name: "POOL_A" }]],
  ["SHOW IMAGE REPOSITORIES", [{ name: "REPO", database_name: "DB", schema_name: "SCH" }]],
  [
    "SHOW IMAGES IN IMAGE REPOSITORY",
    [
      { image_name: "flakegraph", tags: "0.1.0", created_on: "2026-01-01T00:00:00Z" },
      { image_name: "flakegraph", tags: "0.2.0", created_on: "2026-03-01T00:00:00Z" },
    ],
  ],
];

describe("Snowflake visibility", () => {
  it("filters the run history in SQL with bound identities, never inlined ones", async () => {
    const sql = new RecordingSql();
    await Effect.runPromise(
      runtime(sql, viewer("O'BRIEN"), { callerToken: "token", callerRoles: async () => ["RESEARCHERS"] }).listRuns(10),
    );
    const listing = sql.statements.find((statement) => statement.sql.includes("FROM KG_JOB J"))!;
    expect(listing.sql).toContain("WHERE (G.OWNER IS NULL OR UPPER(G.OWNER) = ?");
    expect(listing.sql).toContain("KG_GRAPH_ACL");
    expect(listing.sql).not.toContain("O'BRIEN");
    expect(listing.binds).toEqual(["O'BRIEN", "O'BRIEN", "RESEARCHERS"]);
  });

  it("shows an unidentified viewer only graphs nobody owns", async () => {
    const sql = new RecordingSql();
    await Effect.runPromise(runtime(sql, viewer("")).listRuns());
    expect(sql.texts.find((text) => text.includes("FROM KG_JOB J"))).toContain("WHERE G.OWNER IS NULL GROUP BY");
  });

  it("refuses to open someone else's run, and says an unknown one is unknown", async () => {
    const owned = new RecordingSql([["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "alices-graph" }]]]);
    expect((await failure(runtime(owned, viewer("BOB")).getRun("alices-run"))).code).toBe("forbidden");
    expect((await failure(runtime(new RecordingSql(), viewer("BOB")).getRun("nobodys-run"))).code).toBe("not_found");
  });

  it("gates opening a graph independently of the listing", async () => {
    const sql = new RecordingSql([["SELECT OWNER FROM KG_GRAPH", [{ owner: "CAROL" }]]]);
    expect((await failure(runtime(sql).loadGraph("current", "someone-elses-graph"))).code).toBe("forbidden");
    expect(sql.texts.some((text) => text.includes("FROM KG_NODE"))).toBe(false);
  });

  it("opens no graph the account has never heard of, unclaimed or not", async () => {
    const sql = new RecordingSql();
    expect((await failure(runtime(sql).loadGraph("current", "no-such-graph"))).code).toBe("forbidden");
    expect(sql.texts.some((text) => text.includes("FROM KG_NODE"))).toBe(false);
  });

  it("opens a stored graph only by its id", async () => {
    expect((await failure(runtime(new RecordingSql()).loadGraph("current"))).message).toMatch(/graph id/);
  });

  it("refuses to list another viewer's upload folder however the path is split", async () => {
    for (const [stage, prefix] of [
      ["KG_DOCS", "app/ALICE/job-1"],
      ["KG_DOCS/app/ALICE/job-1", ""],
      ["KG_DOCS/app", "ALICE/job-1"],
    ]) {
      const sql = new RecordingSql();
      const refused = await failure(runtime(sql, viewer("BOB")).listSourceObjects({ kind: "snowflake_stage", stage, prefix }));
      expect(refused.message).toMatch(/other users' uploads/);
      expect(sql.texts.some((text) => text.startsWith("LIST"))).toBe(false);
    }
  });

  it("asks for a folder rather than listing a stage's root, which holds every viewer's uploads", async () => {
    const sql = new RecordingSql();
    const refused = await failure(
      runtime(sql, viewer("BOB")).listSourceObjects({ kind: "snowflake_stage", stage: "KG_DOCS", prefix: "" }),
    );
    expect(refused.message).toMatch(/Choose a folder/);
    expect(sql.texts.some((text) => text.startsWith("LIST"))).toBe(false);
  });

  it("lists only the app's own upload folders from local disk", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-uploads-"));
    await mkdir(path.join(stateRoot, "uploads", "job-1"), { recursive: true });
    await writeFile(path.join(stateRoot, "uploads", "job-1", "note.md"), "Judo.\n");
    await claimUploadFolder(stateRoot, "job-1", viewer("ALICE"));
    const plane = runtime(new RecordingSql(), viewer("ALICE"), { stateRoot });
    const listed = await Effect.runPromise(
      plane.listSourceObjects({ kind: "local", path: path.join(stateRoot, "uploads", "job-1") }),
    );
    expect(listed.map((item) => item.name)).toEqual(["note.md"]);
    await expect(Effect.runPromise(plane.listSourceObjects({ kind: "local", path: "/etc" }))).rejects.toThrow(
      /upload folders/,
    );
    await expect(
      Effect.runPromise(plane.listSourceObjects({ kind: "local", path: path.join(stateRoot, "uploads", "..") })),
    ).rejects.toThrow(/upload folders/);
    const summary = await Effect.runPromise(plane.summarizeSource({ kind: "local", path: path.join(stateRoot, "uploads", "job-1") }));
    expect(summary).toMatchObject({ count: 1, complete: true });
  });

  it("keeps one viewer's upload folder from another, however it is reached", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-uploads-"));
    await mkdir(path.join(stateRoot, "uploads", "job-1"), { recursive: true });
    await writeFile(path.join(stateRoot, "uploads", "job-1", "note.md"), "Judo.\n");
    await claimUploadFolder(stateRoot, "job-1", viewer("ALICE"));
    const folder = path.join(stateRoot, "uploads", "job-1");
    const bob = runtime(new RecordingSql(), viewer("BOB"), { stateRoot });
    for (const refused of [
      bob.listSourceObjects({ kind: "local", path: folder }),
      bob.summarizeSource({ kind: "local", path: folder }),
      bob.previewConfig(request({ sourceKind: "upload", source: { kind: "local", path: folder } })),
    ]) {
      expect((await failure(refused as Effect.Effect<unknown, ControlPlaneError>)).message).toMatch(/belongs to someone else/);
    }
    // A preflight reports it as its run configuration check.
    const result = await Effect.runPromise(bob.preflight(request({ sourceKind: "upload", source: { kind: "local", path: folder } })));
    expect(result.ok).toBe(false);
    expect(result.errors.join(" ")).toMatch(/belongs to someone else/);
  });

  it("asks Snowflake for the viewer's roles, falling back to their grants with the name escaped", async () => {
    const withToken = new RecordingSql();
    const tokens: string[] = [];
    const plane = runtime(withToken, viewer("ALICE"), {
      callerToken: "user-token",
      callerRoles: async (token) => {
        tokens.push(token);
        return ['"Analysts"', "PUBLIC"];
      },
    });
    expect((await Effect.runPromise(plane.viewer())).roles).toEqual(["ANALYSTS", "PUBLIC"]);
    expect(tokens).toEqual(["user-token"]);

    const fallback = new RecordingSql([["SHOW GRANTS TO USER", [{ role: "RESEARCHERS" }]]]);
    const named = await Effect.runPromise(runtime(fallback, viewer('ALICE" OR "1"="1')).viewer());
    expect(fallback.texts.at(-1)).toBe('SHOW GRANTS TO USER "ALICE"" OR ""1""=""1"');
    expect(named.roles).toEqual(["RESEARCHERS"]);
  });
});

describe("Snowflake ownership", () => {
  const acl = (level: "read" | "write", grantee = "BOB", type = "USER") => ({
    grantee_type: type,
    grantee,
    access_level: level,
    granted_by: "ALICE",
    created_at: "2026-01-01T00:00:00.000Z",
  });
  /** BOB holds a share on ALICE's graph, at the level the row says. */
  const sharedWithBob = (level: "read" | "write" = "read") =>
    new RecordingSql([
      OWNED_BY_ALICE,
      ["FROM KG_GRAPH_ACL", [{ graph_id: "alices-graph", ...acl(level) }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "alices-graph" }]],
    ]);

  it.each([
    ["cancel", (plane: SnowflakeRuntime) => plane.cancel("alices-run")],
    ["retry", (plane: SnowflakeRuntime) => plane.retry("alices-run")],
    ["rename", (plane: SnowflakeRuntime) => plane.renameGraph("alices-graph", "Renamed")],
  ])("does not let a read share %s the owner's graph", async (_name, act) => {
    const sql = sharedWithBob();
    const refused = await failure(act(runtime(sql, viewer("BOB"))) as Effect.Effect<unknown, ControlPlaneError>);
    expect(refused.code).toBe("forbidden");
    expect(refused.message).toMatch(/needs write access/);
    expect(sql.texts.some((text) => /^(UPDATE|DELETE|MERGE|EXECUTE|DROP|REMOVE)/.test(text))).toBe(false);
  });

  it.each([
    ["share", (plane: SnowflakeRuntime) => plane.shareGraph("alices-graph", "CAROL", "read")],
    ["unshare", (plane: SnowflakeRuntime) => plane.unshareGraph("alices-graph", "CAROL")],
    ["delete", (plane: SnowflakeRuntime) => plane.deleteGraph("alices-graph")],
    ["preview a delete", (plane: SnowflakeRuntime) => plane.deletionPreview("alices-graph")],
  ])("keeps %s to the owner, whatever the share", async (_name, act) => {
    const sql = sharedWithBob("write");
    const refused = await failure(act(runtime(sql, viewer("BOB"))) as Effect.Effect<unknown, ControlPlaneError>);
    expect(refused.code).toBe("forbidden");
    expect(refused.message).toMatch(/Only the graph's owner/);
    expect(sql.texts.some((text) => /^(UPDATE|DELETE|MERGE|EXECUTE|DROP|REMOVE)/.test(text))).toBe(false);
  });

  it("lets a write share change the graph", async () => {
    const sql = sharedWithBob("write");
    expect(await Effect.runPromise(runtime(sql, viewer("BOB")).renameGraph("alices-graph", "Renamed"))).toBe("Renamed");
    expect(sql.texts.some((text) => text.startsWith("MERGE INTO KG_GRAPH target"))).toBe(true);
  });

  it("hears a role's grant, at its level", async () => {
    const sql = new RecordingSql([OWNED_BY_ALICE, ["FROM KG_GRAPH_ACL", [acl("write", "RESEARCHERS", "ROLE")]]]);
    const member = runtime(sql, viewer("BOB"), { callerToken: "token", callerRoles: async () => ["RESEARCHERS"] });
    expect(await Effect.runPromise(member.requireGraph("alices-graph", "write"))).toBe("write");
    expect((await failure(runtime(sql, viewer("CAROL")).requireGraph("alices-graph", "read"))).code).toBe("forbidden");
  });

  it("leaves an unclaimed graph open to everyone signed in", async () => {
    const sql = new RecordingSql([["SELECT 1 AS KNOWN FROM KG_JOB", [{ known: 1 }]]]);
    expect(await Effect.runPromise(runtime(sql, viewer("BOB")).requireGraph("cli-graph", "owner"))).toBe("owner");
  });

  it("refuses a run it cannot find as unknown, and someone else's as no access", async () => {
    expect((await failure(runtime(new RecordingSql(), viewer("BOB")).requireRun("nobodys-run", "read"))).code).toBe("not_found");
    const owned = new RecordingSql([OWNED_BY_ALICE, ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "alices-graph" }]]]);
    expect((await failure(runtime(owned, viewer("BOB")).requireRun("alices-run", "read"))).message).toBe(NO_ACCESS);
  });

  it("refuses a share target that names nobody before any SQL", async () => {
    const sql = new RecordingSql();
    expect((await failure(runtime(sql).shareGraph("graph-1", "two words", "read"))).message).toMatch(/user name/);
    expect((await failure(runtime(sql).shareGraph("graph-1", "role:", "read"))).message).toMatch(/user name/);
    expect(sql.statements).toEqual([]);
  });

  it("shares idempotently, with the identity the rows already carry", async () => {
    const sql = new RecordingSql([OWNED_BY_ALICE]);
    await Effect.runPromise(runtime(sql).shareGraph("graph-1", " bob ", "write"));
    const merge = sql.statements.find((statement) => statement.sql.startsWith("MERGE INTO KG_GRAPH_ACL"))!;
    expect(merge.binds).toEqual([shareId("graph-1", "USER", "BOB"), "graph-1", "USER", "BOB", "write", "ALICE"]);
    expect(merge.sql).toContain("WHEN MATCHED THEN UPDATE SET ACCESS_LEVEL");
    expect(shareId("graph-1", "USER", "BOB")).toBe("cf2bac1dc6c26d75a785e937232a539904f88c4da994683b8d75ac7622d88f3c");
  });

  it("shares with a role, and withdraws that share by the same name", async () => {
    const sql = new RecordingSql([OWNED_BY_ALICE]);
    await Effect.runPromise(runtime(sql).shareGraph("graph-1", "role:analysts", "read"));
    const merge = sql.statements.find((statement) => statement.sql.startsWith("MERGE INTO KG_GRAPH_ACL"))!;
    expect(merge.binds.slice(0, 5)).toEqual([shareId("graph-1", "ROLE", "ANALYSTS"), "graph-1", "ROLE", "ANALYSTS", "read"]);
    await Effect.runPromise(runtime(sql).unshareGraph("graph-1", "ROLE:ANALYSTS"));
    const removal = sql.statements.find((statement) => statement.sql.startsWith("DELETE FROM KG_GRAPH_ACL"))!;
    expect(removal.binds).toEqual([shareId("graph-1", "ROLE", "ANALYSTS")]);
  });

  it("gives the owner nothing to share with themselves", async () => {
    const sql = new RecordingSql([OWNED_BY_ALICE]);
    expect((await failure(runtime(sql).shareGraph("graph-1", "alice", "read"))).message).toMatch(/already has full access/);
  });

  it("lets anyone leave a graph shared with them, except its owner", async () => {
    const sql = sharedWithBob();
    expect(await Effect.runPromise(runtime(sql, viewer("BOB")).unshareGraph("alices-graph", "bob"))).toBeNull();
    const removal = sql.statements.find((statement) => statement.sql.startsWith("DELETE FROM KG_GRAPH_ACL"))!;
    expect(removal.binds).toEqual([shareId("alices-graph", "USER", "BOB")]);
    expect((await failure(runtime(sql, viewer("ALICE")).unshareGraph("alices-graph", "alice"))).message).toMatch(/cannot remove/);
  });

  it("says who holds a graph, as the sharing panel reads it", async () => {
    const sql = new RecordingSql([
      OWNED_BY_ALICE,
      ["FROM KG_GRAPH_ACL", [acl("write", "ZOE"), acl("read", "ANALYSTS", "ROLE")]],
    ]);
    expect(await Effect.runPromise(runtime(sql).graphAccess("graph-1"))).toEqual({
      graphId: "graph-1",
      owner: "ALICE",
      role: "owner",
      shares: [
        { principal: "ROLE:ANALYSTS", level: "read", grantedBy: "ALICE", grantedAt: "2026-01-01T00:00:00.000Z" },
        { principal: "ZOE", level: "write", grantedBy: "ALICE", grantedAt: "2026-01-01T00:00:00.000Z" },
      ],
    });
  });

  it("lists the graphs a viewer can see, with their role on each", async () => {
    const sql = new RecordingSql([
      ["SELECT DISTINCT J.GRAPH_ID", [{ graph_id: "mine", owner: "BOB" }, { graph_id: "shared", owner: "ALICE" }, { graph_id: "open", owner: null }]],
      ["FROM KG_GRAPH_ACL", [{ graph_id: "shared", ...acl("read") }]],
    ]);
    const graphs = await Effect.runPromise(runtime(sql, viewer("BOB")).visibleGraphs());
    expect([...graphs]).toEqual([["mine", "owner"], ["shared", "read"], ["open", "owner"]]);
    expect(sql.texts.find((text) => text.startsWith("SELECT DISTINCT"))).toContain("KG_GRAPH_ACL");
  });

  it("offers the people on graphs the viewer can see, and nobody else's", async () => {
    const sql = new RecordingSql([
      ["SELECT DISTINCT J.GRAPH_ID", [{ graph_id: "shared", owner: "ALICE" }]],
      ["FROM KG_GRAPH_ACL", [{ graph_id: "shared", ...acl("read") }, { graph_id: "hidden", ...acl("read", "MALLORY") }]],
    ]);
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-people-"));
    expect(await Effect.runPromise(runtime(sql, viewer("CAROL"), { stateRoot }).people())).toEqual(["ALICE", "BOB"]);
  });

  it("refuses a run id already in use, on any graph", async () => {
    const sql = new RecordingSql([OWNED_BY_ALICE, ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "alices-graph" }]]]);
    const refused = await failure(runtime(sql, viewer("BOB")).submit(request({ jobId: "alices-run", graphId: "bobs-graph" })));
    expect(refused.message).toMatch(/already taken/);
    expect(sql.texts.some((text) => text.startsWith("MERGE") || text.startsWith("EXECUTE"))).toBe(false);
  });

  it.each([
    ["no access", null, NO_ACCESS],
    ["read access", "read" as const, "needs write access"],
  ])("answers nothing about a graph a viewer holds %s to", async (_name, level, message) => {
    const sql = new RecordingSql([
      OWNED_BY_ALICE,
      ["FROM KG_GRAPH_ACL", level ? [acl(level)] : []],
      ["SHOW TABLES", [{ name: "KG_JOB" }]],
    ]);
    for (const refused of [
      runtime(sql, viewer("BOB")).preflight(request()),
      runtime(sql, viewer("BOB")).previewConfig(request()),
      runtime(sql, viewer("BOB")).submit(request()),
    ]) {
      expect((await failure(refused as Effect.Effect<unknown, ControlPlaneError>)).message).toContain(message);
    }
    expect(sql.texts.some((text) => /^(SHOW TABLES|LIST|MERGE|PUT|EXECUTE)/.test(text))).toBe(false);
  });

  it("leaves new versions of a graph to the runtimes that keep versions", async () => {
    const revision = { baseRunId: "run-0", dropFileIds: [], addDocuments: true };
    expect((await failure(runtime(new RecordingSql()).submit(request({ revision })))).code).toBe("not_supported");
  });
});

describe("Snowflake runs", () => {
  const runningRow = (quietFor: number) => ({
    id: "run-1",
    graph_id: "graph-1",
    graph_name: "graph-1",
    status: "RUNNING",
    error: null,
    created_at: "2026-01-01T00:00:00.000Z",
    updated_at: "2026-01-01T00:00:00.000Z",
    owner: "ALICE",
    documents_total: 5,
    documents_completed: 0,
    documents_failed: 0,
    documents_pending: 5,
    seconds_since_update: quietFor,
  });

  it("notices a dead worker in the history, not only when the run is opened", async () => {
    const sql = new RecordingSql([
      ["FROM KG_JOB J", [runningRow(7_200)]],
      ["SHOW SERVICES LIKE", [{ status: "FAILED" }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
    ]);
    const [run] = await Effect.runPromise(runtime(sql).listRuns(10));
    expect(run?.status).toBe("failed");
    expect(run?.error).toMatch(/stopped without processing/);
    expect(sql.texts.some((text) => text.startsWith("UPDATE KG_JOB SET STATUS = 'FAILED'"))).toBe(true);
  });

  it("does not interrogate a run that just reported progress", async () => {
    const sql = new RecordingSql([["FROM KG_JOB J", [runningRow(3)]]]);
    const [run] = await Effect.runPromise(runtime(sql).listRuns(10));
    expect(run?.status).toBe("running");
    expect(sql.texts.some((text) => text.startsWith("SHOW SERVICES"))).toBe(false);
  });

  it("never writes a verdict onto someone else's quiet run", async () => {
    const sql = new RecordingSql([
      ["FROM KG_JOB J", [{ ...runningRow(7_200), file_status: "QUEUED", file_count: 5, stage: "queued" }]],
      ["SHOW SERVICES LIKE", [{ status: "DONE" }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
    ]);
    const snapshot = await Effect.runPromise(runtime(sql, viewer("BOB")).getRun("run-1"));
    expect(snapshot.status).toBe("failed");
    expect(sql.texts.some((text) => text.startsWith("UPDATE"))).toBe(false);
  });

  it("reports per-stage progress, the worker's live counts and its spend so far", async () => {
    const progress = { stage: "graph_extraction", counts: { batches_completed: 2, batches_total: 5 }, consumption: { calls: 7 } };
    const base = { id: "run-1", graph_id: "graph-1", graph_name: "Judo", owner: "ALICE", status: "RUNNING", progress, seconds_since_update: 1 };
    const sql = new RecordingSql([
      OWNED_BY_ALICE,
      [
        "FROM KG_JOB J",
        [
          { ...base, stage: "graph_extraction", file_status: "CLAIMED", file_count: 2 },
          { ...base, stage: "queued", file_status: "QUEUED", file_count: 3 },
        ],
      ],
    ]);
    const snapshot = await Effect.runPromise(runtime(sql).getRun("run-1"));
    expect(snapshot).toMatchObject({ graphName: "Judo", status: "running", documentsTotal: 5, storageLocation: "DB.SCH" });
    expect(snapshot.stages.find((stage) => stage.stage === "graph_extraction")).toMatchObject({ completed: 2, total: 5, status: "running" });
    expect(snapshot.raw).toEqual({ owner: "ALICE", consumption: { calls: 7 } });
    expect(snapshot).toMatchObject({ owner: "ALICE", access: "owner", sharedWith: 0 });
  });

  it("cancels the parent before its queued documents", async () => {
    const sql = new RecordingSql([
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
      ["FROM KG_JOB J", [{ id: "run-1", graph_id: "graph-1", status: "CANCELLED" }]],
    ]);
    await Effect.runPromise(runtime(sql).cancel("run-1"));
    const updates = sql.texts.filter((text) => text.startsWith("UPDATE"));
    expect(updates[0]).toMatch(/^UPDATE KG_JOB SET STATUS = 'CANCELLED'/);
    expect(updates[1]).toMatch(/^UPDATE KG_JOB_FILE SET STATUS = 'CANCELLED'/);
  });

  it("retries by requeueing only failed documents and starting the worker the run stored", async () => {
    const stored = buildSpcsLaunch({
      job: { job_id: "run-1", graph_id: "graph-1" },
      snowflake: { database: "DB", schema: "SCH", compute_pool: "POOL_A", image_repository: "DB.SCH.REPO", image_name: "flakegraph:0.2.0" },
    });
    const sql = new RecordingSql([
      ["SELECT CONFIG FROM KG_JOB", [{ config: JSON.stringify(stored.runtimeConfig) }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
      ["COUNT(*) AS QUEUED", [{ queued: 4 }]],
      ["SHOW SERVICES LIKE", [{ status: "DONE" }]],
      ["FROM KG_JOB J", [{ id: "run-1", graph_id: "graph-1", status: "PENDING", stage: "queued", file_status: "QUEUED", file_count: 4 }]],
    ]);
    const snapshot = await Effect.runPromise(runtime(sql).retry("run-1"));
    expect(snapshot.status).toBe("pending");
    const requeue = sql.texts.find((text) => text.startsWith("UPDATE KG_JOB_FILE SET STATUS = 'QUEUED'"))!;
    expect(requeue).toContain("STATUS = 'FAILED'");
    expect(sql.texts).toContain("DROP SERVICE IF EXISTS DB.SCH.FLAKEGRAPH_APP_4E65D3FBE8AD6535");
    expect(sql.texts.some((text) => text.startsWith("UPDATE KG_JOB SET STATUS = 'PENDING'"))).toBe(true);
    expect(sql.texts).toContain(stored.executeSql);
  });

  it("leaves documents the viewer chose to skip out of a retry", async () => {
    const stored = buildSpcsLaunch({
      job: { job_id: "run-1", graph_id: "graph-1" },
      snowflake: { database: "DB", schema: "SCH", compute_pool: "POOL_A", image_repository: "DB.SCH.REPO", image_name: "flakegraph:0.2.0" },
    });
    const sql = new RecordingSql([
      ["SELECT CONFIG FROM KG_JOB", [{ config: stored.runtimeConfig }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
      ["FROM KG_JOB J", [{ id: "run-1", graph_id: "graph-1", status: "FAILED", stage: "queued", file_status: "FAILED", file_count: 1 }]],
    ]);
    await Effect.runPromise(runtime(sql).retry("run-1", ["file-9"]));
    const requeue = sql.statements.find((statement) => statement.sql.startsWith("UPDATE KG_JOB_FILE SET STATUS = 'QUEUED'"))!;
    expect(requeue.sql).toContain("FILE_ID NOT IN (?)");
    expect(requeue.binds).toEqual(["run-1", "graph-1", "file-9"]);
    // Nothing else was queued, so no worker is started for nothing.
    expect(sql.texts.some((text) => text.startsWith("EXECUTE"))).toBe(false);
  });

  it("reads each document's standing off the file queue", async () => {
    const sql = new RecordingSql([
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
      [
        "FROM KG_JOB_FILE WHERE JOB_ID",
        [
          { file_id: "f1", blob_path: "corpus/a.pdf", status: "DONE", stage: "graph_write", error: null },
          { file_id: "f2", blob_path: "corpus/b.pdf", status: "FAILED", stage: "ocr", error: { message: "Cortex said no" } },
        ],
      ],
    ]);
    expect(await Effect.runPromise(runtime(sql).documents("run-1"))).toEqual([
      { fileId: "f1", name: "a.pdf", location: "corpus/a.pdf", phase: "indexed", detail: "graph_write" },
      { fileId: "f2", name: "b.pdf", location: "corpus/b.pdf", phase: "poison", detail: "Cortex said no" },
    ]);
  });

  it("shows no document to a viewer who cannot see the run", async () => {
    const sql = new RecordingSql([["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]], OWNED_BY_ALICE]);
    expect((await failure(runtime(sql, viewer("BOB")).documents("run-1"))).code).toBe("forbidden");
  });

  it("will not restart a worker that is still running", async () => {
    const stored = buildSpcsLaunch({
      job: { job_id: "run-1" },
      snowflake: { database: "DB", schema: "SCH", compute_pool: "POOL_A", image_repository: "DB.SCH.REPO", image_name: "flakegraph:0.2.0" },
    });
    const sql = new RecordingSql([
      ["SELECT CONFIG FROM KG_JOB", [{ config: stored.runtimeConfig }]],
      ["SELECT GRAPH_ID FROM KG_JOB", [{ graph_id: "graph-1" }]],
      OWNED_BY_ALICE,
      ["COUNT(*) AS QUEUED", [{ queued: 1 }]],
      ["SHOW SERVICES LIKE", [{ status: "RUNNING" }]],
    ]);
    expect((await failure(runtime(sql).retry("run-1"))).message).toMatch(/already running/);
    expect(sql.texts.some((text) => text.startsWith("EXECUTE"))).toBe(false);
  });
});

describe("Snowflake submission", () => {
  async function uploadFolder(): Promise<{ stateRoot: string; folder: string }> {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-submit-"));
    const folder = path.join(stateRoot, "uploads", "upload-1");
    await mkdir(folder, { recursive: true });
    await writeFile(path.join(folder, "note.md"), "Judo was developed by Jigoro Kano.\n");
    await claimUploadFolder(stateRoot, "upload-1", viewer("ALICE"));
    return { stateRoot, folder };
  }

  it("puts the upload in the viewer's own stage folder, queues it, and starts a discovered worker", async () => {
    const { stateRoot, folder } = await uploadFolder();
    const puts: Array<[string, string, string]> = [];
    const sql = new RecordingSql([
      ...DISCOVERY,
      OWNED_BY_ALICE,
      [
        "LIST @DB.SCH.KG_DOCS/app/ALICE/run-1",
        [
          { name: "kg_docs/app/ALICE/run-1/note.md", size: 36, md5: "abc" },
          { name: "kg_docs/app/ALICE/run-1/ignored.exe", size: 1, md5: "def" },
        ],
      ],
      ["FROM KG_JOB J", [{ id: "run-1", graph_id: "graph-1", status: "PENDING", stage: "queued", file_status: "QUEUED", file_count: 1 }]],
    ]);
    const plane = runtime(sql, viewer("ALICE"), {
      stateRoot,
      putFolder: async (_sql, directory, stage, prefix) => {
        puts.push([directory, stage, prefix]);
        return 1;
      },
    });
    const snapshot = await Effect.runPromise(
      plane.submit(
        request({
          graphName: "Judo history",
          sourceKind: "upload",
          source: { kind: "local", path: folder },
          // A viewer may pick SPCS objects, never the namespace or credentials.
          runtimeOptions: { database: "OTHER_DB", password: "hunter2" },
          ontology: { entityTypes: [{ name: "Person", description: "A person" }], relationTypes: [], relations: "open" },
        }),
      ),
    );
    expect(snapshot.runId).toBe("run-1");
    expect(puts).toEqual([[await realpath(folder), "@DB.SCH.KG_DOCS", "app/ALICE/run-1"]]);

    const claim = sql.texts.find((text) => text.startsWith("MERGE INTO KG_GRAPH target USING (SELECT ? AS GRAPH_ID, ? AS OWNER)"));
    expect(claim).toContain("WHEN MATCHED AND target.OWNER IS NULL");
    const job = sql.statements.find((statement) => statement.sql.startsWith("MERGE INTO KG_JOB target"))!;
    const stored = JSON.parse(String(job.binds[2])) as Record<string, Record<string, unknown>>;
    expect(stored.snowflake).toMatchObject({
      database: "DB",
      schema: "SCH",
      warehouse: "WH",
      bulk_stage: "@DB.SCH.KG_LOAD_STAGE",
      stage: "@DB.SCH.KG_DOCS",
      compute_pool: "POOL_A",
      image_repository: "DB.SCH.REPO",
      // Newest first: the oldest tag ever pushed must not become the default.
      image_name: "flakegraph:0.2.0",
      service_cpu_request: "3",
      service_memory_limit: "14Gi",
      authenticator: "oauth",
    });
    for (const key of ["account", "user", "host", "password"]) {
      expect(stored.snowflake).not.toHaveProperty(key);
    }
    expect(stored.snowflake?.role ?? null).toBeNull();
    expect(JSON.stringify(stored)).not.toMatch(/hunter2|OTHER_DB|attacker-account|evil\.example|SNOWFLAKE_PASSWORD/);
    expect(stored.files).toMatchObject({ source: "snowflake_stage", stage_prefix: "app/ALICE/run-1" });
    // The worker cannot read the app's bundle, so the ontology travels inline.
    expect(stored.ontology?.profile_path ?? null).toBeNull();
    expect(stored.ontology?.profile).toMatchObject({ entity_types: [{ name: "Person", description: "A person" }] });
    // The stored copy is enough to start the same worker again.
    const relaunch = buildSpcsLaunch(structuredClone(stored));

    const files = sql.statements.find((statement) => statement.sql.startsWith("MERGE INTO KG_JOB_FILE"))!;
    expect(files.binds).toHaveLength(7);
    expect(files.binds.slice(1, 6)).toEqual([
      "run-1",
      "graph-1",
      files.binds[3],
      "@kg_docs/app/ALICE/run-1/note.md",
      "app/ALICE/run-1/note.md",
    ]);
    const execute = sql.texts.find((text) => text.startsWith("EXECUTE JOB SERVICE"))!;
    expect(execute).toBe(relaunch.executeSql);
    expect(execute).toContain("IN COMPUTE POOL POOL_A");
    expect(execute).toContain("image: /DB/SCH/REPO/flakegraph:0.2.0");
    expect(sql.texts.some((text) => text.startsWith("PUT"))).toBe(false);
    expect(sql.statements.find((statement) => statement.sql.includes("AS DISPLAY_NAME) source"))?.binds).toEqual([
      "graph-1",
      "Judo history",
    ]);
  });

  it("carries the base profile's ontology inline when the form names none", async () => {
    const sql = new RecordingSql([...DISCOVERY]);
    const preview = parseYaml(await Effect.runPromise(runtime(sql).previewConfig(request()))) as Record<string, Record<string, unknown>>;
    expect(preview.ontology?.profile_path).toBeNull();
    expect(preview.ontology?.profile).toBeTruthy();
  });

  it("stops before queueing anything when someone else claimed the new graph first", async () => {
    const owners = [[], [{ owner: "CAROL" }]];
    const sql = new RecordingSql([
      ...DISCOVERY,
      ["SELECT OWNER FROM KG_GRAPH", () => owners.shift() ?? []],
      ["LIST @KG_DOCS/corpus", [{ name: "kg_docs/corpus/a.pdf", size: 3, md5: "x" }]],
    ]);
    expect((await failure(runtime(sql).submit(request()))).message).toBe(NO_ACCESS);
    expect(sql.texts.some((text) => text.startsWith("MERGE INTO KG_JOB") || text.startsWith("EXECUTE"))).toBe(false);
  });

  it("records why the worker could not be started, and rethrows", async () => {
    const sql = new RecordingSql([
      ...DISCOVERY,
      OWNED_BY_ALICE,
      ["LIST @KG_DOCS/corpus", [{ name: "kg_docs/corpus/a.pdf", size: 3, md5: "x" }]],
      ["EXECUTE JOB SERVICE", new Error("Compute pool POOL_A is suspended")],
    ]);
    const refused = await failure(runtime(sql).submit(request()));
    expect(refused.message).toMatch(/suspended/);
    const failed = sql.statements.find((statement) => statement.sql.startsWith("UPDATE KG_JOB SET STATUS = 'FAILED'"))!;
    expect(failed.binds[0]).toBe("SPCS job submission failed: Compute pool POOL_A is suspended");
  });

  it("refuses a stage prefix with nothing the pipeline can read", async () => {
    const sql = new RecordingSql([...DISCOVERY, OWNED_BY_ALICE]);
    expect((await failure(runtime(sql).submit(request()))).message).toMatch(/no supported files/);
    expect(sql.texts.some((text) => text.startsWith("EXECUTE"))).toBe(false);
  });
});

describe("Snowflake preflight", () => {
  const tables = [["SHOW TABLES", REQUIRED_TABLES.map((name) => ({ name }))]] as Array<[string, Array<Record<string, unknown>>]>;
  const embeddingColumns = (width: number) =>
    ["KG_NODE", "KG_EDGE", "KG_CHUNK", "KG_COMMUNITY", "OTHER_TABLE"].map((table) => ({
      table_name: table,
      data_type: JSON.stringify({ type: "VECTOR", dimension: table === "OTHER_TABLE" ? 3 : width }),
    }));

  it("passes a run the account can carry out", async () => {
    const sql = new RecordingSql([
      ...tables,
      ...DISCOVERY,
      ["SHOW COLUMNS LIKE 'EMBEDDING'", embeddingColumns(1024)],
      ["LIST @KG_DOCS/corpus", [{ name: "kg_docs/corpus/a.pdf" }]],
    ]);
    const result = await Effect.runPromise(runtime(sql).preflight(request()));
    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
    expect(result.checks.every((check) => check.ok)).toBe(true);
  });

  it("reports an embedding width the graph tables cannot store, before any spend", async () => {
    const sql = new RecordingSql([
      ...tables,
      ...DISCOVERY,
      ["SHOW COLUMNS LIKE 'EMBEDDING'", embeddingColumns(1024)],
      ["LIST @KG_DOCS/corpus", [{ name: "kg_docs/corpus/a.pdf" }]],
    ]);
    const result = await Effect.runPromise(
      runtime(sql).preflight(
        request({
          embedding: {
            provider: "snowflake_cortex",
            model: "snowflake-arctic-embed-m-v1.5",
            endpoint: null,
            apiKeyEnvironmentVariable: null,
            dimension: 768,
            options: {},
          },
        }),
      ),
    );
    expect(result.ok).toBe(false);
    expect(result.errors.join(" ")).toMatch(/Embedding dimension 768 does not fit the graph tables, which store \[1024\]/);
    expect(result.errors.join(" ")).toMatch(/recreate the tables/);
  });

  it("treats a width it cannot read as a failure, not agreement", async () => {
    const sql = new RecordingSql([
      ...tables,
      ...DISCOVERY,
      ["SHOW COLUMNS LIKE 'EMBEDDING'", [{ table_name: "KG_NODE", data_type: "VECTOR" }]],
      ["LIST @KG_DOCS/corpus", [{ name: "kg_docs/corpus/a.pdf" }]],
    ]);
    const result = await Effect.runPromise(runtime(sql).preflight(request()));
    expect(result.errors.join(" ")).toMatch(/Embedding dimension cannot be checked/);
  });

  it("names missing tables, an unlaunchable worker and a missing base configuration", async () => {
    const missing = new RecordingSql([
      ["SHOW TABLES", [{ name: "KG_GRAPH" }]],
      ["SHOW STAGES", [{ name: "KG_LOAD_STAGE" }]],
    ]);
    const result = await Effect.runPromise(runtime(missing).preflight(request()));
    expect(result.errors).toContain("Required Snowflake table is not visible: KG_JOB");
    // No compute pool exists to discover, so the worker cannot be launched.
    expect(result.errors.join(" ")).toMatch(/SPCS launch configuration is not ready: .*compute_pool/);

    const noBase = await Effect.runPromise(
      runtime(new RecordingSql(tables), viewer("ALICE"), { repositoryRoot: await mkdtemp(path.join(tmpdir(), "fg-nobase-")) }).preflight(
        request({ baseConfigPath: null }),
      ),
    );
    expect(noBase.ok).toBe(false);
    expect(noBase.errors.join(" ")).toMatch(/Base configuration not found .*app-defaults\.yaml/);
  });
});

describe("Snowflake graph deletion", () => {
  const stateRoot = () => mkdtemp(path.join(tmpdir(), "fg-snow-delete-"));
  const deleting = (uploadPrefix: string, status = "SUCCEEDED") =>
    new RecordingSql([
      OWNED_BY_ALICE,
      ["UNION ALL", [{ table_name: "KG_NODE", row_count: 97 }, { table_name: "KG_EDGE", row_count: 0 }]],
      [
        "stage_prefix",
        [
          {
            id: "run-1",
            status,
            upload_prefix: uploadPrefix,
            document_stage: "@DB.SCH.KG_DOCS",
            db: "DB",
            sch: "SCH",
          },
        ],
      ],
    ]);

  it("clears every graph table and reports what went", async () => {
    const sql = deleting("app/ALICE/run-1");
    const removed = await Effect.runPromise(runtime(sql, viewer("ALICE"), { stateRoot: await stateRoot() }).deleteGraph("graph-1"));
    expect(removed).toEqual({ KG_NODE: 97, runs: 1, uploads: 1 });
    const cleared = sql.texts.filter((text) => text.startsWith("DELETE FROM ")).map((text) => text.split(" ")[2]);
    expect(new Set(cleared)).toEqual(new Set(GRAPH_DATA_TABLES));
  });

  it("also removes the run's uploaded documents and job service", async () => {
    const sql = deleting("app/ALICE/run-1");
    await Effect.runPromise(runtime(sql, viewer("ALICE"), { stateRoot: await stateRoot() }).deleteGraph("graph-1"));
    expect(sql.texts).toContain("REMOVE @DB.SCH.KG_DOCS/app/ALICE/run-1");
    expect(sql.texts).toContain("DROP SERVICE IF EXISTS DB.SCH.FLAKEGRAPH_APP_4E65D3FBE8AD6535");
  });

  it("never empties a curated source stage", async () => {
    const sql = deleting("corporate/reference-corpus");
    await Effect.runPromise(runtime(sql, viewer("ALICE"), { stateRoot: await stateRoot() }).deleteGraph("graph-1"));
    expect(sql.texts.some((text) => text.includes("corporate/reference-corpus"))).toBe(false);
  });

  it("waits for a run under way to be cancelled", async () => {
    const sql = deleting("app/ALICE/run-1", "RUNNING");
    const refused = await failure(runtime(sql, viewer("ALICE"), { stateRoot: await stateRoot() }).deleteGraph("graph-1"));
    expect(refused.message).toMatch(/still has a run under way/);
    expect(sql.texts.some((text) => /^(DELETE|DROP|REMOVE)/.test(text))).toBe(false);
  });

  it("names what a deletion would take", async () => {
    const sql = deleting("app/ALICE/run-1", "RUNNING");
    expect(await Effect.runPromise(runtime(sql).deletionPreview("graph-1"))).toEqual({
      graphId: "graph-1",
      graphName: "graph-1",
      runs: 1,
      active: 1,
      uploads: 1,
      sharedWith: [],
    });
  });
});

describe("Snowflake graph review", () => {
  it("loads a bounded projection with full counts and only edges between shown nodes", async () => {
    const sql = new RecordingSql([
      OWNED_BY_ALICE,
      ["UNION ALL", [{ table_name: "KG_NODE", row_count: 3 }, { table_name: "KG_DOCUMENT", row_count: 2 }]],
      [
        "WITH REVIEW_NODES",
        [
          { id: "e1", source_node_id: "n1", target_node_id: "n1", assertion: "claimed" },
          { id: "e2", source_node_id: "n1", target_node_id: "n1", assertion: "reported" },
        ],
      ],
      ["FROM KG_NODE WHERE GRAPH_ID", [{ id: "n1", name: "Judo", aliases: ["Jūdō"], embedding: [0.1] }]],
      [
        "FROM KG_EDGE_OBSERVATION",
        [
          { edge_id: "e1", quantities: '[{"value":"3","unit":"mg"}]', proposed_relation_type: "TREATS" },
          { edge_id: "e2", quantities: null, proposed_relation_type: null },
        ],
      ],
      ["FROM KG_COMMUNITY WHERE", [{ id: "c1", title: "Throws" }]],
      ["FROM KG_COMMUNITY_FINDING", [{ id: "f1", community_id: "c1", summary: "Kano", explanation: "Founder" }]],
      ["FROM KG_DOCUMENT WHERE", [{ id: "d1", folder: "Papers", kind: "paper" }]],
      ["FROM KG_DISCARDED_WINDOW", [{ id: "w1", reason: "no_records" }]],
      ["FROM KG_REJECTED_RECORD", [{ id: "r1", reason: "ungrounded" }]],
      ["FROM KG_FAILED_DOCUMENT", [{ id: "x1", error_type: "unreadable" }]],
      ["FROM KG_RUN_REPORT", [{ report: '{"graph_id":"graph-1"}' }]],
      ["FROM KG_GRAPH_METRICS", [{ metrics: { modularity: 0.4 } }]],
    ]);
    const dataset = await Effect.runPromise(runtime(sql).loadGraph("current", "graph-1"));
    expect(dataset.nodes).toEqual([{ id: "n1", name: "Judo", aliases: ["Jūdō"] }]);
    expect(dataset.edges).toEqual([
      {
        id: "e1",
        source_node_id: "n1",
        target_node_id: "n1",
        assertion: "claimed",
        quantities: [{ value: "3", unit: "mg" }],
        stated_relation_types: ["TREATS"],
      },
      { id: "e2", source_node_id: "n1", target_node_id: "n1", assertion: "reported" },
    ]);
    expect(dataset.communities[0]?.findings).toEqual([{ id: "f1", community_id: "c1", summary: "Kano", explanation: "Founder" }]);
    expect(dataset.documents).toEqual([{ id: "d1", folder: "Papers", kind: "paper" }]);
    expect(dataset.discardedWindows).toEqual([{ id: "w1", reason: "no_records" }]);
    expect(dataset.rejectedRecords).toEqual([{ id: "r1", reason: "ungrounded" }]);
    expect(dataset.failedDocuments).toEqual([{ id: "x1", error_type: "unreadable" }]);
    expect(dataset.runReport).toMatchObject({ graph_id: "graph-1", app_full_counts: { node: 3, document: 2 } });
    expect(dataset.graphMetrics).toEqual({ modularity: 0.4 });
    expect(sql.texts).toContain(
      "SELECT * EXCLUDE EMBEDDING FROM KG_NODE WHERE GRAPH_ID = ? ORDER BY DEGREE DESC NULLS LAST, ID LIMIT 10000",
    );
    expect(sql.texts.find((text) => text.startsWith("WITH REVIEW_NODES"))).toContain("JOIN REVIEW_NODES TARGET");
  });
});
