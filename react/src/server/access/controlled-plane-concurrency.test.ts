import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { setImmediate } from "node:timers/promises";
import { Effect } from "effect";
import { afterEach, describe, expect, it, vi } from "vitest";
import { listRunRecords, runDirectory, snapshotFromRecord, writeRunRecord, withdrawRun } from "../catalog";
import { fromCause } from "../protocol/errors";
import type { ControlPlane } from "../protocol/runtime";
import type { IngestionRequest, RuntimeMode } from "../protocol/schema";
import { controlledPlane, NO_ACCESS } from "./controlled-plane";
import { loadAccess } from "./store";

vi.mock("../db/client", () => ({ postgresGraphExists: async () => null, postgresRunExists: async () => null }));

const roots: string[] = [];
const alice = "ALICE@EXAMPLE.COM";
const bob = "BOB@EXAMPLE.COM";
const provider = { provider: "fake", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} };
function request(jobId: string, graphId: string): IngestionRequest {
  return {
    runtime: "local", jobId, graphId, graphName: null,
    sourceKind: "s3", source: { bucket: "synthetic" },
    ocr: provider, llm: provider, embedding: provider,
    output: { kind: "local_files", workspacePath: "/synthetic/output", snowflake: null },
    baseConfigPath: null, includeGlobs: ["**/*"], cacheProvider: "none",
    providerParallelism: 1, runtimeOptions: {}, ontology: null, revision: null, priority: "bulk",
  };
}
const run$ = <A, E>(effect: Effect.Effect<A, E>) => Effect.runPromise(effect);
const as = (inner: ControlPlane, root: string, userName = alice) => controlledPlane(inner, { userName, email: "", roles: [] }, root, { strict: true });

function signal() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}
const gate = () => ({ entered: signal(), released: signal() });

async function stateRoot() {
  const root = await mkdtemp(path.join(tmpdir(), "fg-access-concurrency-"));
  roots.push(root);
  return root;
}

afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

/** Only temporary catalog/config files are written; no worker, CLI, or service is started. */
function runtime(root: string, owner = alice, mode: RuntimeMode = "local", gates: {
  submit?: ReturnType<typeof gate>; preflight?: ReturnType<typeof gate>; delete?: ReturnType<typeof gate>;
  failSubmit?: boolean; liveStatus?: string;
} = {}) {
  const calls: string[] = [];
  const exclusions: string[][] = [];
  const effect = <A>(action: () => Promise<A>) => Effect.tryPromise({ try: action, catch: (cause) => fromCause(cause, "synthetic failure") });
  const enter = async (name: "submit" | "preflight" | "delete") => {
    calls.push(name);
    const held = gates[name];
    if (held) { held.entered.resolve(); await held.released.promise; }
  };
  const config = async (input: IngestionRequest, value: string) => {
    const directory = runDirectory(root, input.jobId);
    await mkdir(directory, { recursive: true });
    await writeFile(path.join(directory, "config.yaml"), `${value}:${input.graphId}`);
    return directory;
  };
  const inner = {
    runtime: mode, capabilities: new Set(),
    listRuns: () => effect(async () => (await listRunRecords(root, Number.MAX_SAFE_INTEGER))
      .filter(({ record }) => record.runtime === mode).map(({ record }) => snapshotFromRecord(record))
      .map((run) => gates.liveStatus ? { ...run, status: gates.liveStatus } : run)),
    getRun: (runId: string) => effect(async () => {
      const found = (await listRunRecords(root)).find(({ record }) => record.runId === runId);
      if (!found) throw new Error("unknown run");
      return snapshotFromRecord(found.record);
    }),
    preflight: (input: IngestionRequest) => effect(async () => {
      await enter("preflight"); await config(input, "preflight");
      return { ok: true, errors: [], warnings: [], checks: [] };
    }),
    previewConfig: () => Effect.succeed("synthetic preview"),
    submit: (input: IngestionRequest) => effect(async () => {
      await enter("submit");
      if (gates.failSubmit) throw new Error("synthetic submit failure");
      return snapshotFromRecord(await writeRunRecord(await config(input, "submitted"), {
        runId: input.jobId, graphId: input.graphId, runtime: mode, owner, status: "queued",
      }));
    }),
    deleteGraph: () => effect(async () => { await enter("delete"); return { synthetic: 1 }; }),
    retry: (runId: string, skipped: readonly string[] = []) => effect(async () => {
      exclusions.push([...skipped]);
      return snapshotFromRecord((await listRunRecords(root)).find(({ record }) => record.runId === runId)!.record);
    }),
  } as unknown as ControlPlane;
  return { inner, calls, exclusions };
}

async function completed(root: string) {
  await writeRunRecord(runDirectory(root, "completed"), { runId: "completed", graphId: "graph", runtime: "local", owner: alice, status: "succeeded" });
}

/** Start the second action while the first is paused, then release without timer-based scheduling. */
async function overlap(first: Promise<unknown>, held: ReturnType<typeof gate>, second: () => Promise<unknown>, waiting: () => void) {
  await held.entered.promise;
  const results = Promise.allSettled([first, second()]);
  try { await setImmediate(); waiting(); }
  finally { held.released.resolve(); }
  return results;
}

describe("submission reservations across request contexts", () => {
  it("admits a single first owner before runtime side effects, across runtimes", async () => {
    const root = await stateRoot(); const held = gate();
    const left = runtime(root, alice, "local", { submit: held }); const right = runtime(root, bob, "kubernetes");
    const results = await overlap(run$(as(left.inner, root).submit(request("left", "shared"))), held,
      () => run$(as(right.inner, root, bob).submit(request("right", "shared"))), () => expect(right.calls).toEqual([]));
    expect(results.map((item) => item.status)).toEqual(["fulfilled", "rejected"]);
    expect(right.calls).toEqual([]);
    expect((await loadAccess(root)).graphs.shared.owner).toBe(alice);
    expect((await listRunRecords(root)).map(({ record }) => record.runId)).toEqual(["left"]);
  });

  it("reserves the run id when concurrent requests name different graphs", async () => {
    const root = await stateRoot(); const held = gate();
    const left = runtime(root, alice, "local", { submit: held }); const right = runtime(root, alice, "kubernetes");
    const results = await overlap(run$(as(left.inner, root).submit(request("shared-run", "left"))), held,
      () => run$(as(right.inner, root).submit(request("shared-run", "right"))), () => expect(right.calls).toEqual([]));
    expect(results.map((item) => item.status)).toEqual(["fulfilled", "rejected"]);
    expect(String((results[1] as PromiseRejectedResult).reason)).toMatch(/already taken/);
    expect((await listRunRecords(root)).map(({ record }) => record.graphId)).toEqual(["left"]);
    expect(await readFile(path.join(runDirectory(root, "shared-run"), "config.yaml"), "utf8")).toBe("submitted:left");
  });

  it("releases graph and run reservations to a queued request when submission fails", async () => {
    const root = await stateRoot(); const held = gate();
    const failing = runtime(root, alice, "local", { failSubmit: true, submit: held });
    const succeeding = runtime(root, bob);
    const results = await overlap(run$(as(failing.inner, root).submit(request("reusable", "graph"))), held,
      () => run$(as(succeeding.inner, root, bob).submit(request("reusable", "graph"))), () => expect(succeeding.calls).toEqual([]));
    expect(results.map((item) => item.status)).toEqual(["rejected", "fulfilled"]);
    expect((await loadAccess(root)).graphs.graph.owner).toBe(bob);
    expect(succeeding.calls).toEqual(["submit"]);
  });

  it("allows an existing owner to submit another run after the first reservation completes", async () => {
    const root = await stateRoot(); const held = gate();
    const first = runtime(root, alice, "local", { submit: held }); const second = runtime(root, alice, "kubernetes");
    const results = await overlap(run$(as(first.inner, root).submit(request("left", "graph"))), held,
      () => run$(as(second.inner, root).submit(request("right", "graph"))), () => expect(second.calls).toEqual([]));
    expect(results.every((item) => item.status === "fulfilled")).toBe(true);
    expect(await listRunRecords(root)).toHaveLength(2);
    expect(second.calls).toEqual(["submit"]);
  });

  it("keeps unrelated graph/run resources independent", async () => {
    const root = await stateRoot(); const held = gate();
    const first = run$(as(runtime(root, alice, "local", { submit: held }).inner, root).submit(request("first", "first")));
    await held.entered.promise;
    try { expect((await run$(as(runtime(root, bob).inner, root, bob).submit(request("second", "second")))).access).toBe("owner"); }
    finally { held.released.resolve(); await first; }
  });

  it("shares reservations across canonical state-root aliases", async () => {
    const root = await stateRoot(); const alias = path.join(await stateRoot(), "alias"); await symlink(root, alias);
    const held = gate(); const left = runtime(root, alice, "local", { submit: held }); const right = runtime(root, bob);
    const results = await overlap(run$(as(left.inner, root).submit(request("left", "graph"))), held,
      () => run$(as(right.inner, alias, bob).submit(request("right", "graph"))), () => expect(right.calls).toEqual([]));
    expect(results.map((item) => item.status)).toEqual(["fulfilled", "rejected"]);
  });
});

describe("preflight and deletion coordination", () => {
  it("rejects a late preflight before it can overwrite a submitted run's config", async () => {
    const root = await stateRoot(); const held = gate(); const inner = runtime(root, alice, "local", { submit: held });
    const results = await overlap(run$(as(inner.inner, root).submit(request("run", "graph"))), held,
      () => run$(as(inner.inner, root).preflight(request("run", "other"))), () => expect(inner.calls).toEqual(["submit"]));
    expect(results.map((item) => item.status)).toEqual(["fulfilled", "rejected"]);
    expect(inner.calls).toEqual(["submit"]);
    expect(await readFile(path.join(runDirectory(root, "run"), "config.yaml"), "utf8")).toBe("submitted:graph");
  });

  it("lets preflight finish before submitting the same run", async () => {
    const root = await stateRoot(); const held = gate();
    const checking = runtime(root, alice, "local", { preflight: held }); const submitting = runtime(root);
    const results = await overlap(run$(as(checking.inner, root).preflight(request("run", "graph"))), held,
      () => run$(as(submitting.inner, root).submit(request("run", "graph"))), () => expect(submitting.calls).toEqual([]));
    expect(results.every((item) => item.status === "fulfilled")).toBe(true);
    expect(await readFile(path.join(runDirectory(root, "run"), "config.yaml"), "utf8")).toBe("submitted:graph");
  });

  it("rechecks active work after waiting for a graph submission", async () => {
    const root = await stateRoot(); await completed(root); const held = gate();
    const submitting = runtime(root, alice, "local", { submit: held }); const deleting = runtime(root);
    const results = await overlap(run$(as(submitting.inner, root).submit(request("new-run", "graph"))), held,
      () => run$(as(deleting.inner, root).deleteGraph("graph")), () => expect(deleting.calls).toEqual([]));
    expect(results.map((item) => item.status)).toEqual(["fulfilled", "rejected"]);
    expect(String((results[1] as PromiseRejectedResult).reason)).toMatch(/under way/);
    expect(deleting.calls).toEqual([]); expect(await listRunRecords(root)).toHaveLength(2);
  });

  it("finishes deletion before admitting a new submission to that graph", async () => {
    const root = await stateRoot(); await completed(root); const held = gate();
    const deleting = runtime(root, alice, "local", { delete: held }); const submitting = runtime(root);
    const results = await overlap(run$(as(deleting.inner, root).deleteGraph("graph")), held,
      () => run$(as(submitting.inner, root).submit(request("new-run", "graph"))), () => expect(submitting.calls).toEqual([]));
    expect(results.every((item) => item.status === "fulfilled")).toBe(true);
    expect((await listRunRecords(root)).map(({ record }) => record.runId)).toEqual(["new-run"]);
    expect((await loadAccess(root)).graphs.graph.owner).toBe(alice);
  });
});

describe("complete graph deletion guard", () => {
  it.each(["kubernetes", "snowflake"] as const)("includes an active %s run when local is selected", async (otherMode) => {
    const root = await stateRoot(); await completed(root);
    await writeRunRecord(runDirectory(root, "other"), { runId: "other", graphId: "graph", runtime: otherMode, owner: alice, status: "queued" });
    const config = path.join(runDirectory(root, "other"), "config.yaml"); await writeFile(config, "synthetic configuration");
    const inner = runtime(root); const control = as(inner.inner, root);
    const preview = await run$(control.deletionPreview("graph")); expect([preview.runs, preview.active]).toEqual([2, 1]);
    await expect(run$(control.deleteGraph("graph"))).rejects.toThrow(`other (${otherMode})`);
    expect(inner.calls).toEqual([]); expect(await listRunRecords(root)).toHaveLength(2);
    expect(await readFile(config, "utf8")).toBe("synthetic configuration");
    await expect(run$(as(inner.inner, root, bob).deletionPreview("graph"))).rejects.toThrow(NO_ACCESS);
    await expect(run$(as(inner.inner, root, bob).deleteGraph("graph"))).rejects.toThrow(NO_ACCESS);
  });

  it("includes withdrawn catalog entries in its affected-run guard", async () => {
    const root = await stateRoot(); await completed(root);
    await writeRunRecord(runDirectory(root, "hidden-active"), { runId: "hidden-active", graphId: "graph", runtime: "kubernetes", owner: alice, status: "queued" });
    await withdrawRun(root, "hidden-active"); const control = as(runtime(root).inner, root);
    expect((await run$(control.deletionPreview("graph"))).active).toBe(1);
    await expect(run$(control.deleteGraph("graph"))).rejects.toThrow(/under way/);
  });

  it("prefers a fresh selected-runtime status to an older terminal catalog record", async () => {
    const root = await stateRoot(); await completed(root);
    const control = as(runtime(root, alice, "local", { liveStatus: "running" }).inner, root);
    expect((await run$(control.deletionPreview("graph"))).active).toBe(1);
    await expect(run$(control.deleteGraph("graph"))).rejects.toThrow(/under way/);
  });

  it("forwards retry exclusions after the viewer passes the write-access guard", async () => {
    const root = await stateRoot(); await completed(root); const inner = runtime(root); const control = as(inner.inner, root);
    await run$(control.retry("completed", ["skip-file"])); await run$(control.retry("completed"));
    expect(inner.exclusions).toEqual([["skip-file"], []]);
    await expect(run$(as(inner.inner, root, bob).retry("completed", ["skip-file"]))).rejects.toThrow(NO_ACCESS);
    expect(inner.exclusions).toHaveLength(2);
  });
});
