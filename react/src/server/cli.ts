import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { Command } from "@effect/platform";
import { appEnv } from "./env";

/**
 * What the pipeline may see of this process's environment, by exact name:
 * what a shell needs, what Python and its package caches read, and the
 * coordination store the fleet CLI submits to.
 */
const CLI_ENVIRONMENT_NAMES = new Set([
  "PATH",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "LANG",
  "LANGUAGE",
  "TZ",
  "TMPDIR",
  "TMP",
  "TEMP",
  "TERM",
  "HOSTNAME",
  "POD_NAME",
  "POD_IP",
  "KUBECONFIG",
  "DATABASE_URL",
  "VIRTUAL_ENV",
  "PYTHONPATH",
  "PYTHONUNBUFFERED",
  "PYTHONDONTWRITEBYTECODE",
  "PYTHONIOENCODING",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "REQUESTS_CA_BUNDLE",
  "CURL_CA_BUNDLE",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "NO_PROXY",
  "http_proxy",
  "https_proxy",
  "no_proxy",
  "OLLAMA_HOST",
]);

/**
 * Families the pipeline reads: its own `KG_*` settings and credentials,
 * model and package caches, accelerator selection, and the cloud SDKs it
 * lists sources with. The console's own `FLAKEGRAPH_*` variables are not
 * among them: the Ask credential and the identity flags are the console's.
 */
const CLI_ENVIRONMENT_PREFIXES = [
  "KG_",
  "LC_",
  "XDG_",
  "UV_",
  "PIP_",
  "HF_",
  "HUGGINGFACE_",
  "TRANSFORMERS_",
  "SENTENCE_TRANSFORMERS_",
  "TOKENIZERS_",
  "TORCH_",
  "CUDA_",
  "NVIDIA_",
  "OMP_",
  "MKL_",
  "MINERU_",
  "SNOWFLAKE_",
  "AWS_",
  "AZURE_",
  "KUBERNETES_",
  "FLAKEGRAPH_SPARK_",
];

/**
 * The environment a spawned pipeline gets: the allow-listed part of this
 * process's, never the whole of it. What the console holds for itself - the
 * Ask credential, the identity flags - stays here.
 */
export function cliEnvironment(base: Record<string, string | undefined> = process.env): Record<string, string> {
  const environment: Record<string, string> = {};
  for (const [name, value] of Object.entries(base)) {
    if (typeof value !== "string") {
      continue;
    }
    if (CLI_ENVIRONMENT_NAMES.has(name) || CLI_ENVIRONMENT_PREFIXES.some((prefix) => name.startsWith(prefix))) {
      environment[name] = value;
    }
  }
  return environment;
}

export interface CommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export function flakegraphCommand(args: string[]): Command.Command {
  const [binary, ...prefix] = appEnv().flakegraphCommand;
  return Command.make(binary, ...prefix, ...args);
}

export async function runFlakegraph(
  args: string[],
  options: { cwd: string; env?: Record<string, string>; timeoutMs?: number },
): Promise<CommandResult> {
  const command = flakegraphCommand(args).pipe(Command.workingDirectory(options.cwd));
  if (command._tag !== "StandardCommand") {
    throw new Error("Expected a standard flakegraph command");
  }
  return await new Promise((resolve) => {
    const child = spawn(command.command, [...command.args], {
      cwd: options.cwd,
      env: (options.env ?? cliEnvironment()) as NodeJS.ProcessEnv,
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    // A bounded call is killed at its deadline and says so; the caller
    // decides what the silence meant.
    const timer =
      options.timeoutMs && options.timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            child.kill("SIGKILL");
          }, options.timeoutMs)
        : null;
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      if (timer) clearTimeout(timer);
      resolve({ exitCode: 1, stdout, stderr: error.message });
    });
    child.on("close", (code, signal) => {
      if (timer) clearTimeout(timer);
      if (timedOut) {
        resolve({ exitCode: null, stdout, stderr: `${stderr.trim()}\nflakegraph did not finish within ${options.timeoutMs} ms`.trim() });
        return;
      }
      if (signal) {
        // Killed from outside, which leaves no error of its own: say so. A
        // SIGKILL nobody sent is the kernel ending the process for memory.
        const cause = signal === "SIGKILL" ? "; the console most likely ran out of memory" : "";
        resolve({ exitCode: null, stdout, stderr: `${stderr.trim()}\nflakegraph was stopped by ${signal}${cause}`.trim() });
        return;
      }
      resolve({ exitCode: code, stdout, stderr });
    });
  });
}

export function spawnFlakegraph(
  args: string[],
  options: {
    cwd: string;
    env: Record<string, string>;
    stdoutPath: string;
    stderrPath: string;
  },
) {
  const command = flakegraphCommand(args);
  if (command._tag !== "StandardCommand") {
    throw new Error("Expected a standard flakegraph command");
  }
  const stdout = createWriteStream(options.stdoutPath, { flags: "w" });
  const stderr = createWriteStream(options.stderrPath, { flags: "w" });
  const child = spawn(command.command, [...command.args], {
    cwd: options.cwd,
    env: options.env as NodeJS.ProcessEnv,
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
  });
  child.stdout?.pipe(stdout);
  child.stderr?.pipe(stderr);
  child.unref();
  return child;
}

/**
 * What a failed CLI call says to a person: the error itself, not the Python
 * traceback in front of it. A traceback ends with the exception's own line,
 * so that line is kept; plain messages pass through, cut to a readable size.
 */
export function cliFailure(stderr: string, fallback: string): string {
  const text = stderr.trim();
  if (!text) {
    return fallback;
  }
  if (text.includes("Traceback (most recent call last)")) {
    const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
    const last = lines.at(-1) ?? "";
    // "botocore.exceptions.ClientError: An error occurred ..." -> the message.
    const message = last.replace(/^[\w.]+(?:Error|Exception|Exit)\b:\s*/, "");
    return message || fallback;
  }
  return text.length > 600 ? `${text.slice(0, 600)}…` : text;
}

export function lastJsonObject(text: string): Record<string, unknown> | null {
  const candidates = [
    ...text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .reverse(),
    text.trim(),
  ];
  for (const candidate of candidates) {
    try {
      const value = JSON.parse(candidate) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }
  return null;
}
