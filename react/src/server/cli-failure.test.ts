import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { cliFailure, runFlakegraph } from "./cli";
import { resetAppEnv } from "./env";

describe("cliFailure", () => {
  it("keeps a traceback's own message and drops the stack", () => {
    const stderr = [
      "Traceback (most recent call last):",
      '  File "/opt/venv/bin/flakegraph", line 10, in <module>',
      "    sys.exit(app())",
      '  File "/opt/venv/lib/python3.13/site-packages/kg_processor/adapters/distributed/s3_blob.py", line 138, in delete_prefix',
      "botocore.exceptions.ClientError: An error occurred (MissingContentMD5) when calling the DeleteObjects operation: Missing required header for this request: Content-Md5.",
    ].join("\n");
    expect(cliFailure(stderr, "failed")).toBe(
      "An error occurred (MissingContentMD5) when calling the DeleteObjects operation: Missing required header for this request: Content-Md5.",
    );
  });

  it("passes a plain message through and falls back when there is none", () => {
    expect(cliFailure("graph g still has work under way (run_1); cancel it first\n", "failed")).toBe(
      "graph g still has work under way (run_1); cancel it first",
    );
    expect(cliFailure("  ", "Deleting g failed")).toBe("Deleting g failed");
  });

  it("says a CLI that was killed was killed, where it could not say so itself", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "cli-kill-"));
    const stub = path.join(root, "killed.mjs");
    await writeFile(stub, "process.kill(process.pid, 'SIGKILL');\n");
    process.env.FLAKEGRAPH_CLI = `node ${stub}`;
    resetAppEnv();
    const result = await runFlakegraph(["distributed", "export"], { cwd: root });
    expect(result.exitCode).toBeNull();
    expect(cliFailure(result.stderr, "failed")).toBe(
      "flakegraph was stopped by SIGKILL; the console most likely ran out of memory",
    );
  });
});

afterEach(() => {
  delete process.env.FLAKEGRAPH_CLI;
  resetAppEnv();
});
