import type { ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import type { CatalogRecord } from "../catalog";

export interface ManagedProcess {
  runId: string;
  graphId: string;
  process: ChildProcess;
  eventsPath: string;
  outputPath: string;
  stdoutPath: string;
  configPath: string;
  startedAt: string;
  storageKind: "local_files" | "snowflake";
  storageLocation: string;
}

class ProcessRegistry {
  private readonly processes = new Map<string, ManagedProcess>();

  start(managed: ManagedProcess): ManagedProcess {
    const existing = this.processes.get(managed.runId);
    if (existing && existing.process.exitCode === null && existing.process.signalCode === null && !existing.process.killed) {
      throw new Error(`Run is already active: ${managed.runId}`);
    }
    this.processes.set(managed.runId, managed);
    return managed;
  }

  get(runId: string): ManagedProcess | undefined {
    return this.processes.get(runId);
  }

  cancel(runId: string): ManagedProcess {
    const managed = this.processes.get(runId);
    if (!managed) {
      throw new Error(`Unknown local run: ${runId}`);
    }
    terminateProcess(managed.process);
    return managed;
  }

  forget(runId: string): void {
    const managed = this.processes.get(runId);
    if (managed && managed.process.exitCode === null) {
      terminateProcess(managed.process);
    }
    this.processes.delete(runId);
  }
}

export const processRegistry = new ProcessRegistry();

export function terminateProcess(process: ChildProcess): void {
  if (process.exitCode !== null || process.signalCode !== null || process.killed) {
    return;
  }
  try {
    if (process.pid) {
      process.kill("SIGTERM");
    }
  } catch {
    // The child may have already exited between the poll and the signal.
  }
}

const runFile = promisify(execFile);

/** OS process birth time identifies a recorded worker even after a console restart. */
export async function processBirth(pid: number | undefined): Promise<string | null> {
  if (!pid) return null;
  try {
    const { stdout } = await runFile("ps", ["-p", String(pid), "-o", "lstart="], {
      timeout: 2_000,
      env: { ...process.env, LC_ALL: "C" },
    });
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

/** Stop a verified worker group and wait for exit before allowing graph deletion. */
export async function stopLocalWorker(record: CatalogRecord, child?: ChildProcess): Promise<void> {
  const pid = child?.pid ?? record.pid;
  let grouped = false;
  const alive = () => grouped && pid
    ? processGroupIsRunning(pid)
    : child ? child.exitCode === null && child.signalCode === null : pidIsRunning(pid);
  if (!pid || !alive()) return;
  if (!Number.isInteger(pid) || pid <= 0) throw new Error("Invalid recorded worker PID");
  const birth = await processBirth(pid);
  if (!child) {
    if (!birth || !record.processBirth) throw new Error("Cannot verify the recorded worker identity; cancellation was not completed.");
    // Another process holds the recorded PID: the worker itself has exited.
    if (birth !== record.processBirth) return;
  }
  grouped = processGroupIsRunning(pid);
  const signal = (value: NodeJS.Signals) => {
    try {
      process.kill(grouped ? -pid : pid, value);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
  };
  for (const value of ["SIGTERM", "SIGKILL"] as const) {
    if (!alive()) return;
    if (!child) {
      const currentBirth = await processBirth(pid);
      if (currentBirth !== birth && (currentBirth || !grouped)) return;
    }
    signal(value);
    const deadline = Date.now() + 2_000;
    while (alive() && Date.now() < deadline) await delay(25);
  }
  if (alive()) throw new Error("The worker is still active; cancellation was not completed.");
}

function processGroupIsRunning(pid: number): boolean {
  try {
    process.kill(-pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

export function pidIsRunning(pid: unknown): boolean {
  const value = typeof pid === "number" ? pid : Number(pid);
  if (!Number.isInteger(value) || value <= 0) {
    return false;
  }
  try {
    process.kill(value, 0);
    return true;
  } catch {
    return existsSync(`/proc/${value}`);
  }
}
