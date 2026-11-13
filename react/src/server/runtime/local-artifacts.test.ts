import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect } from "effect";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runDirectory, snapshotFromRecord, writeRunRecord } from "../catalog";
import type { CommandResult } from "../cli";
import { LocalRuntime } from "./local";

const cli = vi.hoisted(() => ({ run: vi.fn() }));
vi.mock("../cli", async (importOriginal) => ({
  ...await importOriginal<typeof import("../cli")>(),
  runFlakegraph: cli.run,
  spawnFlakegraph: vi.fn(() => { throw new Error("These artifact tests must never spawn a worker"); }),
}));

const roots: string[] = [];

beforeEach(() => {
  cli.run.mockReset();
  cli.run.mockRejectedValue(new Error("Unexpected CLI call"));
});

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture(storageKind: "local_files" | "snowflake" = "local_files") {
  const root = await mkdtemp(path.join(tmpdir(), "fg-local-artifacts-"));
  roots.push(root);
  const repositoryRoot = path.join(root, "repository");
  const stateRoot = path.join(root, "state");
  const outputPath = path.join(repositoryRoot, "shared-output");
  const directory = runDirectory(stateRoot, "current-run");
  const configPath = path.join(directory, "config.yaml");
  const record = await writeRunRecord(directory, {
    runId: "current-run",
    graphId: "current-graph",
    runtime: "local",
    status: storageKind === "snowflake" ? "succeeded" : "running",
    pid: null,
    outputPath,
    configPath,
    storageKind,
    storageLocation: storageKind === "snowflake" ? "DATABASE.GRAPHS" : outputPath,
  });
  await mkdir(repositoryRoot, { recursive: true });
  await writeFile(configPath, "writer:\n  provider: snowflake_direct\n");
  return {
    root, stateRoot, repositoryRoot, outputPath, directory, configPath, record,
    runtime: new LocalRuntime(repositoryRoot, stateRoot),
    snapshot: snapshotFromRecord(record),
  };
}

async function graphFiles(directory: string, graphId: string, nodeCount = 2) {
  await mkdir(directory, { recursive: true });
  await Promise.all([
    writeFile(path.join(directory, "nodes.json"), JSON.stringify(Array.from({ length: nodeCount }, (_, index) => ({
      id: `n${index}`, graph_id: graphId, name: `Node ${index}`,
    })))),
    writeFile(path.join(directory, "edges.json"), JSON.stringify(nodeCount > 1 ? [{
      id: "e1", graph_id: graphId, source_node_id: "n0", target_node_id: "n1",
    }] : [])),
  ]);
}

function exportedDirectory(args: string[]): string {
  const index = args.indexOf("--output");
  if (index < 0 || !args[index + 1]) throw new Error("Export command did not name its destination");
  return args[index + 1];
}

async function exportGraphFiles(args: string[], nodeCount = 2) {
  const destination = exportedDirectory(args);
  await graphFiles(destination, "current-graph", nodeCount);
  await writeFile(path.join(destination, "run_report.json"), JSON.stringify({ job_id: "current-run", graph_id: "current-graph" }));
  await writeFile(path.join(destination, "graph_metrics.json"), "{}");
}

async function successfulExport(args: string[]): Promise<CommandResult> {
  // Exported diagnostics retain the selected durable job identity.
  await exportGraphFiles(args);
  return { exitCode: 0, stdout: "{}", stderr: "" };
}

describe("local artifact recovery after a console restart", () => {
  it.each([
    ["a matching committed report", { job_id: "current-run", graph_id: "current-graph" }, "succeeded"],
    ["another run's report", { job_id: "previous-run", graph_id: "current-graph" }, "interrupted"],
    ["another graph's report", { job_id: "current-run", graph_id: "previous-graph" }, "interrupted"],
    ["a report missing job identity", { graph_id: "current-graph" }, "interrupted"],
    ["a report missing graph identity", { job_id: "current-run" }, "interrupted"],
    ["a missing report", null, "interrupted"],
    ["an invalid report", "not valid JSON", "interrupted"],
  ])("classifies existing output with %s", async (_name, report, expected) => {
    const f = await fixture();
    await graphFiles(f.outputPath, "current-graph");
    if (report !== null) {
      await writeFile(path.join(f.outputPath, "run_report.json"), typeof report === "string" ? report : JSON.stringify(report));
    }
    const listed = await Effect.runPromise(f.runtime.listRuns());
    const opened = await Effect.runPromise(f.runtime.getRun(f.record.runId));
    expect(listed.map((run) => [run.runId, run.status])).toEqual([["current-run", expected]]);
    expect(opened.status).toBe(expected);
    expect(cli.run).not.toHaveBeenCalled();
  });

  it.each(["failed", "cancelled"])("retains an explicitly %s run despite matching artifacts", async (status) => {
    const f = await fixture();
    await writeRunRecord(f.directory, { status });
    await graphFiles(f.outputPath, "current-graph");
    await writeFile(path.join(f.outputPath, "run_report.json"), JSON.stringify({ job_id: "current-run", graph_id: "current-graph" }));
    expect((await Effect.runPromise(f.runtime.getRun("current-run"))).status).toBe(status);
  });

  it("does not recover a Snowflake run from unrelated shared local output", async () => {
    const f = await fixture("snowflake");
    await writeRunRecord(f.directory, { status: "running" });
    await graphFiles(f.outputPath, "current-graph");
    await writeFile(path.join(f.outputPath, "run_report.json"), JSON.stringify({ job_id: "current-run", graph_id: "current-graph" }));
    expect((await Effect.runPromise(f.runtime.getRun("current-run"))).status).toBe("interrupted");
    expect(cli.run).not.toHaveBeenCalled();
  });
});

describe("local runtime inspection of Snowflake output", () => {
  it("exports with the saved configuration into the same directory graphDirectory reports", async () => {
    const f = await fixture("snowflake");
    await graphFiles(f.outputPath, "unrelated-local-output");
    cli.run.mockImplementation(successfulExport);
    const destination = path.join(f.stateRoot, "artifacts", "current-run");
    expect(f.runtime.graphDirectory(f.snapshot)).toBe(destination);
    const graph = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    expect(cli.run).toHaveBeenCalledExactlyOnceWith([
      "snowflake", "export", "--graph-id", "current-graph", "--job-id", "current-run", "--output", destination, "--config", f.configPath,
    ], { cwd: f.repositoryRoot, timeoutMs: expect.any(Number) });
    expect(graph.graphId).toBe("current-graph");
    expect(graph.nodes.map((node) => node.id)).toEqual(["n0", "n1"]);
    expect(graph.edges).toHaveLength(1);
    // A restarted runtime reuses a complete materialization for this run.
    const restarted = new LocalRuntime(f.repositoryRoot, f.stateRoot);
    expect((await Effect.runPromise(restarted.loadRunGraph(f.snapshot))).graphId).toBe("current-graph");
    expect(cli.run).toHaveBeenCalledTimes(1);
    expect(restarted.graphDirectory(f.snapshot)).toBe(destination);
  });

  it("refreshes a cached graph that has no persisted diagnostics", async () => {
    const f = await fixture("snowflake");
    const destination = path.join(f.stateRoot, "artifacts", "current-run");
    await graphFiles(destination, "current-graph");
    cli.run.mockImplementation(successfulExport);
    const graph = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    expect(graph.runReport).toMatchObject({ job_id: "current-run", graph_id: "current-graph" });
    expect(cli.run).toHaveBeenCalledTimes(1);
  });

  it("retains the requested graph identity when Snowflake exports an empty graph", async () => {
    const f = await fixture("snowflake");
    cli.run.mockImplementation(async (args: string[]) => {
      await exportGraphFiles(args, 0);
      return { exitCode: 0, stdout: "{}", stderr: "" };
    });
    const graph = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    expect(graph.nodes).toEqual([]);
    expect(graph.edges).toEqual([]);
    expect(graph.graphId).toBe(f.snapshot.graphId);
    expect(f.runtime.graphDirectory(f.snapshot)).toBe(path.join(f.stateRoot, "artifacts", "current-run"));
  });

  it.each(["running", "failed"])("does not export or keep the Snowflake graph of a %s run", async (status) => {
    const f = await fixture("snowflake");
    cli.run.mockImplementation(successfulExport);
    await expect(Effect.runPromise(f.runtime.loadRunGraph({ ...f.snapshot, status }))).rejects.toThrow(/has not finished/);
    expect(cli.run).not.toHaveBeenCalled();
  });

  it("refuses an uncached export when the run did not save its configuration", async () => {
    const f = await fixture("snowflake");
    await writeRunRecord(f.directory, { configPath: null });
    await expect(Effect.runPromise(f.runtime.loadRunGraph(f.snapshot))).rejects.toThrow(/does not record its Snowflake configuration/);
    expect(cli.run).not.toHaveBeenCalled();
  });

  it("reports a missing saved configuration file through the export failure", async () => {
    const f = await fixture("snowflake");
    await rm(f.configPath);
    cli.run.mockResolvedValue({ exitCode: 1, stdout: "", stderr: "Saved configuration file does not exist" });
    await expect(Effect.runPromise(f.runtime.loadRunGraph(f.snapshot))).rejects.toThrow(/configuration file does not exist/);
    expect(cli.run.mock.calls[0][0]).toContain(f.configPath);
  });

  it.each(["nonzero exit", "rejected command"])("allows export retry after a %s", async (failure) => {
    const f = await fixture("snowflake");
    if (failure === "nonzero exit") {
      cli.run.mockResolvedValueOnce({ exitCode: 1, stdout: "", stderr: "Synthetic export failure" });
    } else {
      cli.run.mockRejectedValueOnce(new Error("Synthetic export failure"));
    }
    cli.run.mockImplementationOnce(successfulExport);
    await expect(Effect.runPromise(f.runtime.loadRunGraph(f.snapshot))).rejects.toThrow(/Synthetic export failure/);
    const graph = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    expect(graph.graphId).toBe("current-graph");
    expect(cli.run).toHaveBeenCalledTimes(2);
  });

  it("shares one in-flight export across concurrent runtime instances", async () => {
    const f = await fixture("snowflake");
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    cli.run.mockImplementation(async (args: string[]) => {
      await gate;
      return successfulExport(args);
    });
    const secondRuntime = new LocalRuntime(f.repositoryRoot, f.stateRoot);
    const first = Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    const second = Effect.runPromise(secondRuntime.loadRunGraph(f.snapshot, { purpose: "retrieval" }));
    try {
      await vi.waitFor(() => expect(cli.run).toHaveBeenCalledTimes(1));
    } finally {
      release();
      await Promise.allSettled([first, second]);
    }
    const graphs = await Promise.all([first, second]);
    expect(graphs.map((graph) => graph.graphId)).toEqual(["current-graph", "current-graph"]);
    expect(graphs.map((graph) => graph.nodes.length)).toEqual([2, 2]);
    expect(cli.run).toHaveBeenCalledTimes(1);
  });

  it("keeps the full exported graph available to retrieval beyond review limits", async () => {
    const f = await fixture("snowflake");
    cli.run.mockImplementation(async (args: string[]) => {
      await exportGraphFiles(args, 10_001);
      return { exitCode: 0, stdout: "{}", stderr: "" };
    });
    const review = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot));
    const retrieval = await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot, { purpose: "retrieval" }));
    expect(review.nodes).toHaveLength(10_000);
    expect(retrieval.nodes).toHaveLength(10_001);
    expect(retrieval.nodes.at(-1)?.id).toBe("n10000");
    expect(f.runtime.graphDirectory(f.snapshot)).toBe(path.join(f.stateRoot, "artifacts", "current-run"));
    expect(cli.run).toHaveBeenCalledTimes(1);
  });

  it("preserves the ordinary local artifact directory contract", async () => {
    const f = await fixture();
    await graphFiles(f.outputPath, "current-graph");
    await writeFile(path.join(f.outputPath, "run_report.json"), JSON.stringify({ job_id: "current-run", graph_id: "current-graph" }));
    expect(f.runtime.graphDirectory(f.snapshot)).toBe(f.outputPath);
    expect((await Effect.runPromise(f.runtime.loadRunGraph(f.snapshot))).graphId).toBe("current-graph");
    expect(cli.run).not.toHaveBeenCalled();
  });
});


describe("local artifact identity", () => {
  it("marks a succeeded run unavailable when a newer job replaces its directory", async () => {
    const f = await fixture();
    await writeRunRecord(f.directory, { status: "succeeded" });
    await graphFiles(f.outputPath, "current-graph");
    await writeFile(path.join(f.outputPath, "run_report.json"), JSON.stringify({ job_id: "newer-run", graph_id: "current-graph" }));
    expect((await Effect.runPromise(f.runtime.getRun("current-run"))).status).toBe("unavailable");
    expect((await Effect.runPromise(f.runtime.listRuns()))[0].status).toBe("unavailable");
    await expect(Effect.runPromise(f.runtime.loadRunGraph(f.snapshot))).rejects.toThrow(/superseded/);
  });
});
