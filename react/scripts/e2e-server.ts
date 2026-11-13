#!/usr/bin/env bun
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";

const port = process.env.PORT ?? "3100";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(root, "..");
const stateRoot = process.env.FLAKEGRAPH_APP_STATE_ROOT ?? path.join(tmpdir(), "flakegraph-e2e");

// FLAKEGRAPH_ASK_NO_MODEL runs the journeys the way CI does: no hosted
// credentials and no local engine. Otherwise the Ask model is whatever the
// environment names: FLAKEGRAPH_ASK_* variables, the one secrets file
// FLAKEGRAPH_ASK_SECRETS_FILE points at, or an Ollama on the loopback.
if (process.env.FLAKEGRAPH_ASK_NO_MODEL === "1") {
  process.env.FLAKEGRAPH_ASK_DISABLE_OLLAMA = "1";
  delete process.env.FLAKEGRAPH_ASK_SECRETS_FILE;
}

if (!process.env.FLAKEGRAPH_KEEP_STATE) {
  await rm(stateRoot, { recursive: true, force: true });
}

const seed = spawn("bun", ["scripts/seed.ts"], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    FLAKEGRAPH_APP_STATE_ROOT: stateRoot,
    FLAKEGRAPH_REPOSITORY_ROOT: repo,
  },
});
const seedCode = await new Promise<number>((resolve) => seed.on("close", (code) => resolve(code ?? 1)));
if (seedCode !== 0) {
  process.exit(seedCode);
}

const child = spawn("bun", ["run", "dev", "--port", port, "--hostname", "127.0.0.1"], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    PORT: port,
    FLAKEGRAPH_APP_STATE_ROOT: stateRoot,
    FLAKEGRAPH_REPOSITORY_ROOT: repo,
    FLAKEGRAPH_STUB_RUNTIMES: "1",
    // A laptop demo has no sign-in gate: the identities are the demo ones
    // the sidebar offers, and a session that assumed nobody is still served.
    FLAKEGRAPH_DEMO_IDENTITIES: "1",
    FLAKEGRAPH_APP_GRAFANA_URL: "https://grafana.example.test",
    FLAKEGRAPH_CLI: `bun ${path.join(root, "scripts/fake-flakegraph.ts")}`,
  },
});

child.on("exit", (code) => process.exit(code ?? 0));
