import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect } from "effect";
import { afterEach, describe, expect, it, vi } from "vitest";
import { writeRunRecord, runDirectory, snapshotFromRecord } from "../catalog";
import { runFlakegraph } from "../cli";
import { KubernetesRuntime } from "./kubernetes";
import { appRouter } from "../trpc/router";
import { appEnv, resetAppEnv } from "../env";
import { emptyWorkspace, saveWorkspace } from "../workspace";
import { createContext } from "../trpc/init";

vi.mock("../cli", async (original) => ({ ...await original<typeof import("../cli")>(), runFlakegraph: vi.fn() }));
const originalEnvironment = { ...process.env };
const roots: string[] = [];
afterEach(async () => {
  process.env = { ...originalEnvironment }; resetAppEnv();
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});
async function root() { const value = await mkdtemp(path.join(tmpdir(), "fg-retry-")); roots.push(value); return value; }

describe("retry quarantine", () => {
  it("passes repeatable file exclusions to the durable retry command", async () => {
    const state = await root();
    await writeRunRecord(runDirectory(state, "run"), { runId: "run", graphId: "graph", runtime: "kubernetes", status: "failed", configPath: path.join(state, "run.yaml") });
    const runtime = new KubernetesRuntime(state, state, false);
    vi.spyOn(runtime, "getRun").mockReturnValue(Effect.succeed(snapshotFromRecord({ runId: "run", graphId: "graph", runtime: "kubernetes", status: "queued" })));
    vi.mocked(runFlakegraph).mockResolvedValue({ exitCode: 0, stdout: "{}", stderr: "" });
    await Effect.runPromise(runtime.retry("run", ["poison-a", "poison-b"]));
    expect(runFlakegraph).toHaveBeenCalledWith(["distributed", "retry", "--run-id", "run", "--skip-file-id", "poison-a", "--skip-file-id", "poison-b", "--config", path.join(state, "run.yaml")], { cwd: state });
  });
  it("uses the workspace's saved quarantine when the Retry API is called", async () => {
    process.env.FLAKEGRAPH_APP_STATE_ROOT = await root();
    process.env.FLAKEGRAPH_DEMO_IDENTITIES = "1";
    process.env.FLAKEGRAPH_STUB_RUNTIMES = "1";
    resetAppEnv();
    await saveWorkspace({ ...emptyWorkspace(), skippedFiles: { run: ["poison"] } }, appEnv().stateRoot);
    const ctx = await createContext({ headers: new Headers({ "x-flakegraph-runtime": "kubernetes" }) });
    const retry = vi.spyOn(ctx.controlPlane, "retry").mockReturnValue(Effect.succeed(snapshotFromRecord({ runId: "run", graphId: "graph", runtime: "kubernetes", status: "queued" })));
    await appRouter.createCaller(ctx).runs.retry({ runId: "run" });
    expect(retry).toHaveBeenCalledWith("run", ["poison"]);
  });
});
