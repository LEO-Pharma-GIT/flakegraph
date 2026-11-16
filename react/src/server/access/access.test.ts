import { mkdir, mkdtemp, writeFile, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect } from "effect";
import { describe, expect, it } from "vitest";
import type { ControlPlane } from "../protocol/runtime";
import { ClusterSnapshot, NodeStatus, WorkloadStatus, type IngestionRequest, type RunSnapshot, type Viewer } from "../protocol/schema";
import { controlledPlane, NO_ACCESS, notePerson } from "./controlled-plane";
import { allows, roleFor, validPrincipal } from "./graph-access";
import { writeRunRecord } from "../catalog";
import { claimUploadFolder, controlledPlane as plane } from "./controlled-plane";
import { grant, loadAccess, updateAccess } from "./store";

const person = (userName: string): Viewer => ({ userName, email: "", roles: [] });

function run(runId: string, graphId: string, owner: string | null = null): RunSnapshot {
  return {
    runId,
    graphId,
    status: "succeeded",
    startedAt: null,
    updatedAt: null,
    graphName: graphId,
    stages: [],
    events: [],
    documentsTotal: null,
    documentsCompleted: 0,
    documentsFailed: 0,
    outputPath: `/graphs/${graphId}`,
    storageKind: "local_files",
    storageLocation: null,
    warnings: [],
    error: null,
    listingWarning: null,
    raw: owner ? { owner } : {},
  };
}

/** A runtime that serves a fixed list of runs, newest first, and records what it was asked to do. */
function fakeRuntime(runs: RunSnapshot[]): ControlPlane & { calls: string[] } {
  const calls: string[] = [];
  const find = (runId: string) => {
    const found = runs.find((item) => item.runId === runId);
    return found ? Effect.succeed(found) : Effect.fail(new Error("unknown") as never);
  };
  return {
    runtime: "local",
    capabilities: new Set(),
    calls,
    viewer: () => Effect.succeed(person("")),
    listSourceObjects: () => Effect.succeed([]),
    summarizeSource: () => Effect.succeed({ count: 0, sizeBytes: 0, complete: true }),
    preflight: (request: IngestionRequest) => {
      calls.push(`preflight:${request.graphId}`);
      return Effect.succeed({ ok: true, errors: [], warnings: [], checks: [] });
    },
    submit: (request: IngestionRequest) => {
      calls.push(`submit:${request.graphId}`);
      const created = run(request.jobId, request.graphId);
      runs.unshift(created);
      return Effect.succeed(created);
    },
    listRuns: () => Effect.succeed([...runs]),
    getRun: find,
    documents: () => Effect.succeed([]),
    versions: () => Effect.succeed([]),
    cancel: (runId: string) => {
      calls.push(`cancel:${runId}`);
      return find(runId);
    },
    retry: (runId: string) => {
      calls.push(`retry:${runId}`);
      return find(runId);
    },
    recover: () => Effect.succeed("ok"),
    renameGraph: (_graphId: string, name: string) => Effect.succeed(name),
    loadGraph: (location: string) => Effect.succeed({ location } as never),
    loadRunGraph: () => Effect.die("unused"),
    graphDirectory: () => null,
    cluster: () => Effect.succeed(null),
    nodeAssignments: () =>
      Effect.succeed(runs.map((item) => ({ runId: item.runId, graphId: item.graphId, taskId: "t", scopeId: "s" })) as never),
    deleteGraph: (graphId: string) => {
      calls.push(`delete:${graphId}`);
      return Effect.succeed({ runs: 1 });
    },
    previewConfig: (request: IngestionRequest) => {
      calls.push(`preview:${request.graphId}`);
      return Effect.succeed("");
    },
  };
}

async function stateRoot(): Promise<string> {
  return mkdtemp(path.join(tmpdir(), "fg-access-"));
}

const run$ = <A, E>(effect: Effect.Effect<A, E>) => Effect.runPromise(effect);

describe("roleFor", () => {
  const record = { owner: "ALICE@EXAMPLE.COM", grants: [{ principal: "CAROL@EXAMPLE.COM", level: "read" as const, grantedBy: "ALICE@EXAMPLE.COM", grantedAt: "" }] };

  it("gives the owner and each grantee their standing, in any case, and nobody else anything", () => {
    const strict = { strict: true };
    expect(roleFor(record, person("alice@example.com"), strict)).toBe("owner");
    expect(roleFor(record, person("Carol@Example.com"), strict)).toBe("read");
    expect(roleFor(record, person("bob@example.com"), strict)).toBeNull();
    expect(roleFor(record, person(""), strict)).toBeNull();
  });

  it("treats a graph nobody owns as nobody's where sign-in is required, and the laptop's otherwise", () => {
    const unowned = { owner: null, grants: [] };
    expect(roleFor(unowned, person("alice@example.com"), { strict: true })).toBeNull();
    expect(roleFor(unowned, person(""), { strict: false, laptop: true })).toBe("owner");
    expect(roleFor(null, person("alice@example.com"), { strict: false })).toBe("owner");
  });

  it("gives a laptop session that assumed nobody every graph, and an assumed identity only its own", () => {
    const laptop = { strict: false, laptop: true };
    expect(roleFor(record, person(""), laptop)).toBe("owner");
    expect(roleFor(record, person("bob@example.com"), laptop)).toBeNull();
    expect(roleFor(record, person("carol@example.com"), laptop)).toBe("read");
  });

  it("gives an anonymous reader off a laptop nothing that belongs to someone", () => {
    const shared = { strict: false };
    expect(roleFor(record, person(""), shared)).toBeNull();
    expect(roleFor({ owner: null, grants: [] }, person(""), shared)).toBe("read");
    expect(roleFor(record, person(""), { strict: true, laptop: true })).toBeNull();
  });

  it("orders roles owner over write over read", () => {
    expect(allows("write", "read")).toBe(true);
    expect(allows("read", "write")).toBe(false);
    expect(allows("write", "owner")).toBe(false);
    expect(allows(null, "read")).toBe(false);
  });
});

describe("validPrincipal", () => {
  it("wants a sign-in address behind a gate and accepts a bare name on a laptop", () => {
    expect(validPrincipal(" dana@example.com ", { strict: true })).toBe("DANA@EXAMPLE.COM");
    expect(validPrincipal("CAROL", { strict: true })).toBeNull();
    expect(validPrincipal("CAROL", { strict: false })).toBe("CAROL");
    expect(validPrincipal("two words", { strict: false })).toBeNull();
    expect(validPrincipal("", { strict: false })).toBeNull();
  });
});

describe("access store", () => {
  it("keeps every grant when several are made at once", async () => {
    const root = await stateRoot();
    await Promise.all(
      ["A", "B", "C", "D", "E"].map((name) =>
        updateAccess(root, (state) => {
          grant(state, "graph_1", `${name}@EXAMPLE.COM`, "read", "OWNER@EXAMPLE.COM");
        }),
      ),
    );
    const state = await loadAccess(root);
    expect(state.graphs.graph_1?.grants.map((item) => item.principal).sort()).toEqual([
      "A@EXAMPLE.COM",
      "B@EXAMPLE.COM",
      "C@EXAMPLE.COM",
      "D@EXAMPLE.COM",
      "E@EXAMPLE.COM",
    ]);
  });

  it("notes each person once a day", async () => {
    const root = await stateRoot();
    await notePerson(root, person("alice@example.com"));
    const first = (await loadAccess(root)).people["ALICE@EXAMPLE.COM"]?.lastSeenAt;
    await notePerson(root, person("ALICE@example.com"));
    expect((await loadAccess(root)).people["ALICE@EXAMPLE.COM"]?.lastSeenAt).toBe(first);
    await notePerson(root, person(""));
    expect(Object.keys((await loadAccess(root)).people)).toEqual(["ALICE@EXAMPLE.COM"]);
  });
});

describe("controlledPlane", () => {
  const strict = { strict: true };

  it("lists only the viewer's graphs, taking the owner from the earliest run a graph has", async () => {
    const root = await stateRoot();
    const runs = [
      run("run_3", "graph_a", "BOB@EXAMPLE.COM"),
      run("run_2", "graph_b", "BOB@EXAMPLE.COM"),
      run("run_1", "graph_a", "ALICE@EXAMPLE.COM"),
      run("run_0", "graph_orphan"),
    ];
    const alice = controlledPlane(fakeRuntime(runs), person("alice@example.com"), root, strict);
    const listed = await run$(alice.listRuns());
    expect(listed.map((item) => [item.runId, item.access, item.owner])).toEqual([
      ["run_3", "owner", "ALICE@EXAMPLE.COM"],
      ["run_1", "owner", "ALICE@EXAMPLE.COM"],
    ]);
    await expect(run$(alice.getRun("run_2"))).rejects.toThrow(NO_ACCESS);
    await expect(run$(alice.getRun("run_0"))).rejects.toThrow(NO_ACCESS);
  });

  it("lets a grantee in at their level, and the owner change the level or take it back", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_1", "graph_a", "ALICE@EXAMPLE.COM")]);
    const alice = controlledPlane(runtime, person("alice@example.com"), root, strict);
    const carol = () => controlledPlane(runtime, person("carol@example.com"), root, strict);

    await expect(run$(carol().getRun("run_1"))).rejects.toThrow(NO_ACCESS);
    const shared = await run$(alice.shareGraph("graph_a", "Carol@Example.com", "read"));
    expect(shared.owner).toBe("ALICE@EXAMPLE.COM");
    expect(shared.shares.map((item) => [item.principal, item.level, item.grantedBy])).toEqual([
      ["CAROL@EXAMPLE.COM", "read", "ALICE@EXAMPLE.COM"],
    ]);
    // Sharing wrote the owner down, so it no longer depends on the run records.
    expect((await loadAccess(root)).graphs.graph_a?.owner).toBe("ALICE@EXAMPLE.COM");

    expect((await run$(carol().getRun("run_1"))).access).toBe("read");
    await expect(run$(carol().retry("run_1"))).rejects.toThrow(/write access/);
    await expect(run$(carol().shareGraph("graph_a", "dave@example.com", "read"))).rejects.toThrow(/owner/);

    await run$(alice.shareGraph("graph_a", "carol@example.com", "write"));
    await run$(carol().retry("run_1"));
    expect(runtime.calls).toEqual(["retry:run_1"]);
    await expect(run$(carol().deleteGraph("graph_a"))).rejects.toThrow(/owner/);

    const after = await run$(alice.unshareGraph("graph_a", "carol@example.com"));
    expect(after?.shares).toEqual([]);
    await expect(run$(carol().getRun("run_1"))).rejects.toThrow(NO_ACCESS);
  });

  it("lets a grantee leave, but not the owner", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_1", "graph_a", "ALICE@EXAMPLE.COM")]);
    const alice = controlledPlane(runtime, person("alice@example.com"), root, strict);
    await run$(alice.shareGraph("graph_a", "carol@example.com", "read"));
    const carol = controlledPlane(runtime, person("carol@example.com"), root, strict);
    expect(await run$(carol.unshareGraph("graph_a", "carol@example.com"))).toBeNull();
    await expect(run$(alice.unshareGraph("graph_a", "alice@example.com"))).rejects.toThrow(/owner cannot remove/);
    await expect(run$(alice.shareGraph("graph_a", "alice@example.com", "read"))).rejects.toThrow(/already has full access/);
    await expect(run$(alice.shareGraph("graph_a", "not an address", "read"))).rejects.toThrow(/sign-in address/);
  });

  it("makes the submitter the owner of a new graph, and needs write access to add to an existing one", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_1", "graph_a", "ALICE@EXAMPLE.COM")]);
    const bob = controlledPlane(runtime, person("bob@example.com"), root, strict);
    const created = await run$(bob.submit({ jobId: "run_2", graphId: "graph_new" } as IngestionRequest));
    expect([created.access, created.owner]).toEqual(["owner", "BOB@EXAMPLE.COM"]);
    await expect(run$(bob.submit({ jobId: "run_3", graphId: "graph_a" } as IngestionRequest))).rejects.toThrow(NO_ACCESS);
    expect(runtime.calls).toEqual(["submit:graph_new"]);
  });

  it("deletes a graph's access record with the graph, owner only", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_1", "graph_a", "ALICE@EXAMPLE.COM")]);
    const alice = controlledPlane(runtime, person("alice@example.com"), root, strict);
    await run$(alice.shareGraph("graph_a", "carol@example.com", "write"));
    await run$(alice.deleteGraph("graph_a"));
    expect((await loadAccess(root)).graphs.graph_a).toBeUndefined();
  });

  it("offers the people it has seen, owners and grantees, but not the viewer", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_1", "graph_a", "ALICE@EXAMPLE.COM")]);
    await notePerson(root, person("dave@example.com"));
    const alice = controlledPlane(runtime, person("alice@example.com"), root, strict);
    await run$(alice.shareGraph("graph_a", "carol@example.com", "read"));
    expect(await run$(alice.people())).toEqual(["CAROL@EXAMPLE.COM", "DAVE@EXAMPLE.COM"]);
  });

  it("opens a stored graph only through its run where sign-in is required", async () => {
    const root = await stateRoot();
    const alice = controlledPlane(fakeRuntime([]), person("alice@example.com"), root, strict);
    await expect(run$(alice.loadGraph("/graphs/anything"))).rejects.toThrow(/through its run/);
  });

});

describe("controlledPlane against the ways around it", () => {
  const strict = { strict: true };
  const request = (overrides: Partial<IngestionRequest>) =>
    ({ jobId: "run_new", graphId: "graph_new", source: {}, ...overrides }) as IngestionRequest;

  it("refuses a run id that is taken, so nobody's run record is overwritten", async () => {
    const root = await stateRoot();
    await writeRunRecord(path.join(root, "runs", "run_alice"), { runId: "run_alice", graphId: "graph_a", owner: "ALICE@EXAMPLE.COM" });
    const mallory = plane(fakeRuntime([]), person("mallory@example.com"), root, strict);
    await expect(run$(mallory.submit(request({ jobId: "run_alice" })))).rejects.toThrow(/already taken/);
    await expect(run$(mallory.preflight(request({ jobId: "run_alice" })))).rejects.toThrow(/already taken/);
  });

  it("does not let a graph another runtime knows be claimed through this one", async () => {
    const root = await stateRoot();
    // Recorded by the fleet runtime; this plane serves another runtime.
    await writeRunRecord(path.join(root, "runs", "run_fleet"), { runId: "run_fleet", graphId: "graph_fleet", runtime: "kubernetes" });
    const mallory = plane(fakeRuntime([]), person("mallory@example.com"), root, strict);
    await expect(run$(mallory.submit(request({ graphId: "graph_fleet" })))).rejects.toThrow(NO_ACCESS);
    expect((await loadAccess(root)).graphs.graph_fleet).toBeUndefined();
  });

  it("reads an upload folder only for whoever uploaded it, or can read a graph built from it", async () => {
    const root = await stateRoot();
    const uploads = path.join(root, "uploads");
    await mkdir(path.join(uploads, "job_alice"), { recursive: true });
    expect(await claimUploadFolder(root, "job_alice", person("alice@example.com"))).toBe(true);
    expect(await claimUploadFolder(root, "job_alice", person("mallory@example.com"))).toBe(false);
    const built = { ...run("run_1", "graph_a", "ALICE@EXAMPLE.COM"), raw: { owner: "ALICE@EXAMPLE.COM", sourcePath: path.join(uploads, "job_alice") } };
    const runtime = fakeRuntime([built]);
    const source = { kind: "local", path: path.join(uploads, "job_alice") };
    const as = (name: string) => plane(runtime, person(name), root, strict);

    await run$(as("alice@example.com").summarizeSource(source));
    await expect(run$(as("mallory@example.com").summarizeSource(source))).rejects.toThrow(/belongs to someone else/);
    await expect(run$(as("mallory@example.com").submit(request({ source })))).rejects.toThrow(/belongs to someone else/);
    await run$(as("alice@example.com").shareGraph("graph_a", "carol@example.com", "read"));
    await run$(as("carol@example.com").summarizeSource(source));
  });

  it("checks private local inputs using sourceKind for every ingestion operation", async () => {
    const root = await stateRoot();
    const folder = path.join(root, "uploads", "alice");
    await mkdir(folder, { recursive: true });
    await claimUploadFolder(root, "alice", person("alice@example.com"));
    const mallory = plane(fakeRuntime([]), person("mallory@example.com"), root, strict);
    for (const sourceKind of ["local_path", "upload"] as const) {
      const input = request({ sourceKind, source: { kind: "s3", path: folder } });
      await expect(run$(mallory.submit(input))).rejects.toThrow(/belongs to someone else/);
      await expect(run$(mallory.preflight(input))).rejects.toThrow(/belongs to someone else/);
      await expect(run$(mallory.previewConfig(input))).rejects.toThrow(/belongs to someone else/);
    }
  });

  it("refuses ancestor sources and symlinks to protected console state", async () => {
    const parent = await stateRoot();
    const root = path.join(parent, ".flakegraph", "app");
    await mkdir(root, { recursive: true });
    const mallory = plane(fakeRuntime([]), person("mallory@example.com"), root, strict);
    await expect(run$(mallory.listSourceObjects({ kind: "local", path: parent }))).rejects.toThrow(/own records/);
    const privateFolder = path.join(root, "uploads", "alice");
    await mkdir(privateFolder, { recursive: true });
    await claimUploadFolder(root, "alice", person("alice@example.com"));
    const alias = path.join(parent, "alias");
    await symlink(privateFolder, alias);
    await expect(run$(mallory.listSourceObjects({ kind: "local", path: alias }))).rejects.toThrow(/belongs to someone else/);
    await expect(run$(mallory.submit(request({ sourceKind: "local_path", source: { kind: "local", path: parent } })))).rejects.toThrow(/own records/);
  });

  it("never reads the console's own records as documents", async () => {
    const root = await stateRoot();
    await mkdir(path.join(root, "corpora", "contracts"), { recursive: true });
    await writeFile(path.join(root, "workspace.json"), "{}");
    const mallory = plane(fakeRuntime([]), person("mallory@example.com"), root, strict);
    for (const target of [root, path.join(root, "runs"), path.join(root, "graphs", "graph_a"), path.join(root, "uploads"), path.join(root, "workspace.json")]) {
      await expect(run$(mallory.listSourceObjects({ kind: "local", path: target }))).rejects.toThrow(/own records|belongs/);
    }
    // A documents folder an operator placed in the state root stays a source.
    await run$(mallory.listSourceObjects({ kind: "local", path: path.join(root, "corpora", "contracts") }));
  });

  it("opens a graph by location only where that graph is stored", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_m", "graph_m", "MALLORY@EXAMPLE.COM"), run("run_a", "graph_a", "ALICE@EXAMPLE.COM")]);
    const mallory = plane(runtime, person("mallory@example.com"), root, strict);
    await run$(mallory.loadGraph("/graphs/graph_m", "graph_m"));
    await expect(run$(mallory.loadGraph("/graphs/graph_a", "graph_m"))).rejects.toThrow(/not where this graph is stored/);
    await expect(run$(mallory.loadGraph("/graphs/graph_a"))).rejects.toThrow(/through its run/);
  });

  it("preflights and previews a request on someone else's graph only with write access to it", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_a", "graph_a", "ALICE@EXAMPLE.COM")]);
    const mallory = plane(runtime, person("mallory@example.com"), root, strict);
    await expect(run$(mallory.preflight(request({ graphId: "graph_a" })))).rejects.toThrow(NO_ACCESS);
    await expect(run$(mallory.previewConfig(request({ graphId: "graph_a" })))).rejects.toThrow(NO_ACCESS);
    await run$(plane(runtime, person("alice@example.com"), root, strict).shareGraph("graph_a", "carol@example.com", "read"));
    const carol = plane(runtime, person("carol@example.com"), root, strict);
    await expect(run$(carol.preflight(request({ graphId: "graph_a" })))).rejects.toThrow(/write access/);
    // A graph nobody has built yet is the caller's to plan, and the owner's own is theirs.
    await run$(mallory.preflight(request({ graphId: "graph_m" })));
    await run$(mallory.previewConfig(request({ graphId: "graph_m" })));
    const alice = plane(runtime, person("alice@example.com"), root, strict);
    await run$(alice.preflight(request({ graphId: "graph_a" })));
    expect(runtime.calls).toEqual(["preflight:graph_m", "preview:graph_m", "preflight:graph_a"]);
  });

  it("builds a revision only on a run of the same graph", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_m", "graph_m", "MALLORY@EXAMPLE.COM"), run("run_a", "graph_a", "ALICE@EXAMPLE.COM")]);
    const mallory = plane(runtime, person("mallory@example.com"), root, strict);
    const revision = { baseRunId: "run_a", addDocuments: false } as IngestionRequest["revision"];
    await expect(run$(mallory.submit(request({ graphId: "graph_m", revision })))).rejects.toThrow(/same graph/);
  });

  it("shows a node's work only for the viewer's graphs", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_m", "graph_m", "MALLORY@EXAMPLE.COM"), run("run_a", "graph_a", "ALICE@EXAMPLE.COM")]);
    const mallory = plane(runtime, person("mallory@example.com"), root, strict);
    expect((await run$(mallory.nodeAssignments("ns", "node"))).map((item) => item.graphId)).toEqual(["graph_m"]);
  });

  it("serves a cluster snapshot that holds nothing of anyone's runs, graphs or documents", () => {
    // fleet.cluster is not limited by graph access because nothing in it
    // belongs to a graph; a field that did would have to be limited first,
    // as fleet.nodeAssignments is.
    const fields = [ClusterSnapshot, NodeStatus, WorkloadStatus].flatMap((schema) => Object.keys(schema.fields));
    expect(fields.filter((field) => /run|graph|document|owner|source|job/i.test(field))).toEqual([]);
  });

  it("offers people who signed in and people on the viewer's graphs, not everyone's grants", async () => {
    const root = await stateRoot();
    const runtime = fakeRuntime([run("run_m", "graph_m", "MALLORY@EXAMPLE.COM"), run("run_a", "graph_a", "ALICE@EXAMPLE.COM")]);
    await run$(plane(runtime, person("alice@example.com"), root, strict).shareGraph("graph_a", "secret@example.com", "read"));
    const mallory = plane(runtime, person("mallory@example.com"), root, strict);
    expect(await run$(mallory.people())).not.toContain("SECRET@EXAMPLE.COM");
  });
});
