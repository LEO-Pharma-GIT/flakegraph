import "server-only";

import { readFileSync } from "node:fs";

/**
 * Where the Ask models' provider settings come from, shared by the language
 * model and the embedding model so both read one environment the same way:
 * the variables, the optional secrets file, and how base URLs are spelled.
 */

/** The Ask model's key, then the conventional names read as fallbacks. */
export const ASK_KEY_VARIABLES = ["FLAKEGRAPH_ASK_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"];
/** The Ask model's OpenAI-compatible or Azure base URL, then the conventional fallbacks. */
export const ASK_BASE_VARIABLES = ["FLAKEGRAPH_ASK_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE_URL"];
export const ASK_API_VERSION_VARIABLES = ["FLAKEGRAPH_ASK_API_VERSION", "AZURE_OPENAI_API_VERSION"];
export const OLLAMA_BASE_VARIABLES = ["FLAKEGRAPH_ASK_OLLAMA_BASE_URL", "OLLAMA_HOST"];

// The secrets file is read once per process and said so once: an operator
// reading the log should see which file, if any, the credentials came from.
let secretFile: { path: string | null; values: Record<string, string> } | undefined;

/** The secrets file's variables overlaid by the process environment. */
export function askEnv(): Record<string, string | undefined> {
  return { ...loadOptionalSecretFile(), ...process.env };
}

export function resetProviderEnv(): void {
  secretFile = undefined;
}

/** An Ollama address as its OpenAI-compatible `/v1` base. */
export function ollamaOpenAiBase(base: string): string {
  return base.replace(/\/$/, "").replace(/\/v1$/, "") + "/v1";
}

export function isAzureHost(url: string): boolean {
  return /cognitiveservices\.azure\.com|openai\.azure\.com|services\.ai\.azure\.com/i.test(url);
}

export function azureOpenAiBase(url: string): string {
  const trimmed = url.replace(/\/$/, "");
  if (trimmed.endsWith("/openai/v1")) {
    return trimmed.replace(/\/v1$/, "");
  }
  if (trimmed.endsWith("/openai")) {
    return trimmed;
  }
  return `${trimmed}/openai`;
}

export function ollamaReachable(base: string): boolean {
  if (process.env.FLAKEGRAPH_ASK_DISABLE_OLLAMA === "1") {
    return false;
  }
  try {
    const url = new URL(base.includes("://") ? base : `http://${base}`);
    return url.hostname === "127.0.0.1" || url.hostname === "localhost";
  } catch {
    return false;
  }
}

export function first(env: Record<string, string | undefined>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = env[key]?.trim();
    if (value) {
      return value;
    }
  }
  return undefined;
}

/**
 * Credentials for the Ask model may come from one dotenv-style file, and
 * only the one `FLAKEGRAPH_ASK_SECRETS_FILE` names. Nothing is looked for
 * by convention: a file the operator did not name must not be able to turn
 * on a billed provider. Which file was read is logged once.
 */
function loadOptionalSecretFile(): Record<string, string> {
  if (secretFile) {
    return secretFile.values;
  }
  const configured = process.env.FLAKEGRAPH_ASK_SECRETS_FILE?.trim();
  if (!configured) {
    secretFile = { path: null, values: {} };
    return secretFile.values;
  }
  try {
    const values = parseEnvFile(readFileSync(configured, "utf8"));
    console.info(`Ask: read ${Object.keys(values).length} variables from FLAKEGRAPH_ASK_SECRETS_FILE (${configured})`);
    secretFile = { path: configured, values };
  } catch (error) {
    console.warn(
      `Ask: FLAKEGRAPH_ASK_SECRETS_FILE (${configured}) could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
    secretFile = { path: configured, values: {} };
  }
  return secretFile.values;
}

function parseEnvFile(contents: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of contents.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      continue;
    }
    const index = trimmed.indexOf("=");
    const key = trimmed.slice(0, index).replace(/^export\s+/, "").trim();
    const value = trimmed.slice(index + 1).trim().replace(/^['"]|['"]$/g, "");
    if (key) {
      env[key] = value;
    }
  }
  return env;
}
