import { existsSync } from "node:fs";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect } from "effect";
import { describe, expect, it } from "vitest";
import { controlledPlane } from "./access/controlled-plane";
import { loadAccess, updateAccess, claimUpload } from "./access/store";
import { readGraphNames, renameGraph, writeRunRecord } from "./catalog";
import { readUploadedGold, saveUploadedGold } from "./gold-store";
import type { ControlPlane } from "./protocol/runtime";
import type { RunSnapshot } from "./protocol/schema";
import { loadWorkspace, patchWorkspace } from "./workspace";

const person = (userName: string) => ({ userName, email: "", roles: [] });

function snapshot(runId: string, graphId: string, status = "succeeded"): RunSnapshot {
  return {
    runId,
    graphId,
    status,
    graphName: graphId,
    startedAt: null,
    updatedAt: null,
    stages: [],
    events: [],
    documentsTotal: null,
    documentsCompleted: 0,
    documentsFailed: 0,
    outputPath: null,
    storageKind: "local_files",
    storageLocation: null,
    warnings: [],
    error: null,
    listingWarning: null,
    raw: { owner: "ALICE@EXAMPLE.COM" },
  };
}

function runtime(runs: RunSnapshot[], deleted: string[]): ControlPlane {
  return {
    runtime: "local",
    capabilities: new Set(["delete_graph"]),
    listRuns: () => Effect.succeed(runs),
    getRun: (runId: string) => Effect.succeed(runs.find((run) => run.runId === runId)!),
    deleteGraph: (graphId: string) => {
      deleted.push(graphId);
      return Effect.succeed({ tasks: 7 });
    },
  } as unknown as ControlPlane;
}

describe("deleting a graph", () => {
  it("removes the console's records of it and the uploads only it used, and keeps what another graph needs", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "fg-delete-"));
    const uploads = path.join(root, "uploads");
    for (const job of ["job_mine", "job_shared", "job_other"]) {
      await mkdir(path.join(uploads, job), { recursive: true });
      await writeFile(path.join(uploads, job, "doc.md"), "text");
      await updateAccess(root, (state) => claimUpload(state, job, "ALICE@EXAMPLE.COM"));
    }
    const record = (runId: string, graphId: string, job: string) =>
      writeRunRecord(path.join(root, "runs", runId), {
        runId,
        graphId,
        status: "succeeded",
        runtime: "local",
        owner: "ALICE@EXAMPLE.COM",
        outputPath: path.join(root, "graphs", graphId),
        sourcePath: path.join(uploads, job),
      });
    await record("run_a1", "graph_a", "job_mine");
    await record("run_a2", "graph_a", "job_shared");
    await record("run_b1", "graph_b", "job_shared");
    await record("run_b2", "graph_b", "job_other");
    // A record naming another graph's output directory as its own.
    await writeRunRecord(path.join(root, "runs", "run_a0"), {
      runId: "run_a0",
      graphId: "graph_a",
      status: "failed",
      runtime: "local",
      outputPath: path.join(root, "graphs", "graph_b"),
    });
    for (const graphId of ["graph_a", "graph_b"]) {
      await mkdir(path.join(root, "graphs", graphId), { recursive: true });
      await renameGraph(root, graphId, `${graphId} name`);
      await saveUploadedGold(root, graphId, { name: "gold", entities: [], relations: [] } as never);
      await patchWorkspace(
        (current) => ({
          ...current,
          perspectives: [
            ...current.perspectives,
            { id: `view_${graphId}`, graphId, name: "view", lifecycle: "draft", search: "", communityIds: [], suggestedQuestions: [] },
          ],
        }),
        root,
      );
    }
    await mkdir(path.join(root, "artifacts", "run_a2"), { recursive: true });
    const runs = ["run_a2", "run_a1", "run_b2", "run_b1"].map((id) => snapshot(id, id.startsWith("run_a") ? "graph_a" : "graph_b"));
    const deleted: string[] = [];
    const alice = controlledPlane(runtime(runs, deleted), person("alice@example.com"), root, { strict: true });

    const preview = await Effect.runPromise(alice.deletionPreview("graph_a"));
    expect([preview.runs, preview.uploads, preview.active]).toEqual([3, 1, 0]);

    await Effect.runPromise(alice.shareGraph("graph_a", "carol@example.com", "read"));
    const carol = controlledPlane(runtime(runs, deleted), person("carol@example.com"), root, { strict: true });
    await expect(Effect.runPromise(carol.deleteGraph("graph_a"))).rejects.toThrow(/owner/);

    const result = await Effect.runPromise(alice.deleteGraph("graph_a"));
    expect(result).toMatchObject({ runs: 3, uploads: 1, tasks: 7 });
    expect(deleted).toEqual(["graph_a"]);

    // Everything the console kept for graph_a is gone...
    expect(existsSync(path.join(root, "runs", "run_a1"))).toBe(false);
    expect(existsSync(path.join(root, "runs", "run_a2"))).toBe(false);
    expect(existsSync(path.join(root, "graphs", "graph_a"))).toBe(false);
    expect(existsSync(path.join(root, "artifacts", "run_a2"))).toBe(false);
    expect(existsSync(path.join(uploads, "job_mine"))).toBe(false);
    expect((await readGraphNames(root)).graph_a).toBeUndefined();
    expect(await readUploadedGold(root, "graph_a")).toBeNull();
    const access = await loadAccess(root);
    expect(access.graphs.graph_a).toBeUndefined();
    expect(access.uploads.job_mine).toBeUndefined();
    expect((await loadWorkspace(root)).perspectives.map((item) => item.graphId)).toEqual(["graph_b"]);

    // ...and graph_b keeps its runs, output, name, gold and both its uploads,
    // including the one graph_a was also built from.
    expect(existsSync(path.join(root, "runs", "run_b1"))).toBe(true);
    expect(existsSync(path.join(root, "graphs", "graph_b"))).toBe(true);
    expect(existsSync(path.join(uploads, "job_shared"))).toBe(true);
    expect(existsSync(path.join(uploads, "job_other"))).toBe(true);
    expect((await readGraphNames(root)).graph_b).toBe("graph_b name");
    expect(await readUploadedGold(root, "graph_b")).not.toBeNull();
    expect(access.uploads.job_shared).toBeDefined();
  });

  it("waits for work under way to be cancelled", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "fg-delete-"));
    const runs = [snapshot("run_1", "graph_a", "running")];
    const alice = controlledPlane(runtime(runs, []), person("alice@example.com"), root, { strict: true });
    expect((await Effect.runPromise(alice.deletionPreview("graph_a"))).active).toBe(1);
    await expect(Effect.runPromise(alice.deleteGraph("graph_a"))).rejects.toThrow(/under way/);
  });
});
