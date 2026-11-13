import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { Effect } from "effect";
import { describe, expect, it } from "vitest";
import { LocalRuntime } from "./local";
import { processBirth, processRegistry, pidIsRunning } from "./process-registry";
import { writeRunRecord, runDirectory } from "../catalog";

async function fixture(script = "setInterval(() => {}, 1000)") {
  const root = await mkdtemp(path.join(tmpdir(), "fg-lifecycle-"));
  const child = spawn(process.execPath, ["-e", script], { detached: true, stdio: ["ignore", "pipe", "ignore"] });
  await once(child, "spawn");
  const directory = runDirectory(root, "run");
  const outputPath = path.join(root, "graph");
  await writeRunRecord(directory, { runId: "run", graphId: "g", runtime: "local", status: "running", pid: child.pid, processBirth: await processBirth(child.pid), outputPath });
  return { root, child, outputPath, directory, runtime: new LocalRuntime(root, root) };
}

async function cleanup(child: ChildProcess, root: string) {
  if (child.exitCode === null && child.signalCode === null) {
    const closed = once(child, "close");
    child.kill("SIGKILL");
    await closed;
  }
  processRegistry.forget("run");
  await rm(root, { recursive: true, force: true });
}

describe("local worker lifecycle", () => {
  it("records a managed signal exit as failed", async () => {
    const f = await fixture();
    try {
      processRegistry.start({ runId: "run", graphId: "g", process: f.child, eventsPath: path.join(f.directory, "events.jsonl"), outputPath: f.outputPath, stdoutPath: "", configPath: "", startedAt: new Date().toISOString(), storageKind: "local_files", storageLocation: f.outputPath });
      const closed = once(f.child, "close");
      f.child.kill("SIGTERM"); await closed;
      const result = await Effect.runPromise(f.runtime.getRun("run"));
      expect(result.status).toBe("failed");
      expect(result.error).toContain("SIGTERM");
    } finally { await cleanup(f.child, f.root); }
  });

  it("stops a persisted worker without a process-registry entry", async () => {
    const f = await fixture();
    try {
      expect(processRegistry.get("run")).toBeUndefined();
      expect((await Effect.runPromise(f.runtime.getRun("run"))).status).toBe("running");
      const result = await Effect.runPromise(f.runtime.cancel("run"));
      expect(result.status).toBe("cancelled");
      expect(pidIsRunning(f.child.pid)).toBe(false);
    } finally { await cleanup(f.child, f.root); }
  });

  it("waits for surviving group members before completing cancellation", async () => {
    const descendant = 'process.on("SIGTERM", () => {}); process.stdout.write(String(process.pid)+"\\n"); setInterval(() => {}, 1000)';
    const script = `require("node:child_process").spawn(process.execPath, ["-e", ${JSON.stringify(descendant)}], { stdio: ["ignore", "inherit", "ignore"] }); setInterval(() => {}, 1000)`;
    const f = await fixture(script);
    const pid = await new Promise<number>((resolve) => f.child.stdout!.once("data", (data) => resolve(Number(String(data).trim()))));
    try {
      const result = await Effect.runPromise(f.runtime.cancel("run"));
      expect(result.status).toBe("cancelled");
      expect(pidIsRunning(pid)).toBe(false);
    } finally {
      try { process.kill(pid, "SIGKILL"); } catch { /* The fixture has already exited. */ }
      await cleanup(f.child, f.root);
    }
  }, 10_000);

  it("never signals a reused PID and cancels the run whose worker has exited", async () => {
    const f = await fixture();
    try {
      await writeRunRecord(f.directory, { processBirth: "not this worker" });
      expect((await Effect.runPromise(f.runtime.cancel("run"))).status).toBe("cancelled");
      expect(pidIsRunning(f.child.pid)).toBe(true);
    } finally { await cleanup(f.child, f.root); }
  });

  it("refuses to signal a worker whose identity was never recorded and leaves its run active", async () => {
    const f = await fixture();
    try {
      await writeRunRecord(f.directory, { processBirth: null });
      await expect(Effect.runPromise(f.runtime.cancel("run"))).rejects.toThrow(/verify/);
      expect(pidIsRunning(f.child.pid)).toBe(true);
      expect((await Effect.runPromise(f.runtime.getRun("run"))).status).toBe("running");
    } finally { await cleanup(f.child, f.root); }
  });
});
