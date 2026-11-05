import "server-only";

import { readFileSync } from "node:fs";
import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import type { SharedV4ProviderOptions as ProviderOptions } from "@ai-sdk/provider";
import type { LanguageModel } from "ai";
import {
  DEFAULT_AZURE_API_VERSION,
  DEFAULT_AZURE_DEPLOYMENT,
  DEFAULT_OLLAMA_BASE_URL,
  DEFAULT_OLLAMA_MODEL,
} from "./constants";

export interface AskModel {
  languageModel: LanguageModel;
  provider: "azure" | "openai" | "ollama";
  modelId: string;
  billed: boolean;
  /**
   * Call options for the structured helper calls - planning, scoring, follow-up
   * questions, type suggestions - as opposed to the answer itself.
   *
   * They are classification tasks whose output is a small JSON object, and a
   * reasoning model's free-form thinking multiplies their latency without
   * changing the verdict: the query planner measured 15 s thinking against
   * 2.7 s without, and a search runs dozens of scoring calls. So they ask for
   * no reasoning effort unless `FLAKEGRAPH_ASK_STRUCTURED_REASONING` says
   * otherwise (`inherit` sends nothing and lets the model decide).
   */
  structuredOptions: { providerOptions?: ProviderOptions };
}

const STRUCTURED_REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high", "inherit"] as const;
type StructuredReasoning = (typeof STRUCTURED_REASONING_EFFORTS)[number];

function structuredOptions(env: Record<string, string | undefined>): AskModel["structuredOptions"] {
  const raw = env.FLAKEGRAPH_ASK_STRUCTURED_REASONING?.trim() || "none";
  const effort = (STRUCTURED_REASONING_EFFORTS as readonly string[]).includes(raw) ? (raw as StructuredReasoning) : null;
  if (effort === null) {
    throw new Error(
      `FLAKEGRAPH_ASK_STRUCTURED_REASONING is "${raw}"; expected one of ${STRUCTURED_REASONING_EFFORTS.join(", ")}.`,
    );
  }
  // Azure serves the same chat model class, so both read the openai options.
  return effort === "inherit" ? {} : { providerOptions: { openai: { reasoningEffort: effort } } };
}

let cached: AskModel | null | undefined;
// The secrets file is read once per process and said so once: an operator
// reading the log should see which file, if any, the credentials came from.
let secretFile: { path: string | null; values: Record<string, string> } | undefined;
// Whether the resolved model answered a probe. A local engine is assumed
// from its address alone, and one that is not running would otherwise be
// "configured" and fail every call; the probe settles it once per process.
let probed: Promise<void> | null = null;

export function resetAskModel(): void {
  cached = undefined;
  probed = null;
  secretFile = undefined;
}

export function resolveAskModel(): AskModel | null {
  if (cached !== undefined) {
    return cached;
  }
  cached = loadAskModel();
  return cached;
}

/**
 * Resolve the model and confirm a local engine is actually there.
 *
 * Every request entry point awaits this before the synchronous readers run;
 * a hosted provider is taken at its word, an Ollama address is asked for its
 * model list with a short timeout and dropped when it does not answer. The
 * probe is the one request the console makes on its own: a GET to the
 * loopback address `FLAKEGRAPH_ASK_OLLAMA_BASE_URL` or `OLLAMA_HOST` names
 * (127.0.0.1:11434 by default), once per process, and only when no hosted
 * provider is configured.
 */
export async function probeAskModel(): Promise<AskModel | null> {
  const model = resolveAskModel();
  if (!model || model.provider !== "ollama") {
    return model;
  }
  probed ??= (async () => {
    const base = ollamaBaseUrl();
    try {
      const response = await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(OLLAMA_PROBE_TIMEOUT_MS) });
      if (!response.ok) {
        cached = null;
      }
    } catch {
      cached = null;
    }
  })();
  await probed;
  return cached ?? null;
}

const OLLAMA_PROBE_TIMEOUT_MS = 1_500;

function ollamaBaseUrl(): string {
  const env = { ...loadOptionalSecretFile(), ...process.env };
  const base = first(env, ["FLAKEGRAPH_ASK_OLLAMA_BASE_URL", "OLLAMA_HOST"]) ?? DEFAULT_OLLAMA_BASE_URL;
  return base.replace(/\/$/, "").replace(/\/v1$/, "");
}

export function askModelConfigured(): boolean {
  return resolveAskModel() != null;
}

function loadAskModel(): AskModel | null {
  const env = { ...loadOptionalSecretFile(), ...process.env };
  const structured = structuredOptions(env);
  const askKey = first(env, ["FLAKEGRAPH_ASK_API_KEY", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"]);
  const askBase = first(env, ["FLAKEGRAPH_ASK_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE_URL"]);
  const askModel = first(env, ["FLAKEGRAPH_ASK_MODEL", "AZURE_OPENAI_DEPLOYMENT"]) ?? DEFAULT_AZURE_DEPLOYMENT;
  const apiVersion = first(env, ["FLAKEGRAPH_ASK_API_VERSION", "AZURE_OPENAI_API_VERSION"]) ?? DEFAULT_AZURE_API_VERSION;

  if (askKey && askBase && isAzureHost(askBase)) {
    const baseURL = azureOpenAiBase(askBase);
    const azure = createAzure({
      baseURL,
      apiKey: askKey,
      apiVersion,
      useDeploymentBasedUrls: true,
    });
    return {
      languageModel: azure.chat(askModel),
      provider: "azure",
      modelId: askModel,
      billed: true,
      structuredOptions: structured,
    };
  }

  if (askKey && askBase && !isAzureHost(askBase)) {
    const openai = createOpenAI({ baseURL: askBase.replace(/\/$/, ""), apiKey: askKey });
    return {
      languageModel: openai.chat(askModel),
      provider: "openai",
      modelId: askModel,
      billed: true,
      structuredOptions: structured,
    };
  }

  const ollamaBase = first(env, ["FLAKEGRAPH_ASK_OLLAMA_BASE_URL", "OLLAMA_HOST"]) ?? DEFAULT_OLLAMA_BASE_URL;
  const ollamaModel = first(env, ["FLAKEGRAPH_ASK_OLLAMA_MODEL"]) ?? DEFAULT_OLLAMA_MODEL;
  if (ollamaReachable(ollamaBase)) {
    const openai = createOpenAI({
      baseURL: ollamaBase.replace(/\/$/, "").replace(/\/v1$/, "") + "/v1",
      apiKey: "ollama",
    });
    return {
      languageModel: openai.chat(ollamaModel),
      provider: "ollama",
      modelId: ollamaModel,
      billed: false,
      structuredOptions: structured,
    };
  }

  return null;
}

function isAzureHost(url: string): boolean {
  return /cognitiveservices\.azure\.com|openai\.azure\.com|services\.ai\.azure\.com/i.test(url);
}

function azureOpenAiBase(url: string): string {
  const trimmed = url.replace(/\/$/, "");
  if (trimmed.endsWith("/openai/v1")) {
    return trimmed.replace(/\/v1$/, "");
  }
  if (trimmed.endsWith("/openai")) {
    return trimmed;
  }
  return `${trimmed}/openai`;
}

function ollamaReachable(base: string): boolean {
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

function first(env: Record<string, string | undefined>, keys: string[]): string | undefined {
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
