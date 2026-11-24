import "server-only";

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
import {
  ASK_API_VERSION_VARIABLES,
  ASK_BASE_VARIABLES,
  ASK_KEY_VARIABLES,
  askEnv,
  azureOpenAiBase,
  first,
  isAzureHost,
  OLLAMA_BASE_VARIABLES,
  ollamaOpenAiBase,
  ollamaReachable,
  resetProviderEnv,
} from "./provider-env";

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
// Whether the resolved model answered a probe. A local engine is assumed
// from its address alone, and one that is not running would otherwise be
// "configured" and fail every call; the probe settles it once per process.
let probed: Promise<void> | null = null;

export function resetAskModel(): void {
  cached = undefined;
  cachedReader = undefined;
  probed = null;
  resetProviderEnv();
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
  const env = askEnv();
  const base = first(env, OLLAMA_BASE_VARIABLES) ?? DEFAULT_OLLAMA_BASE_URL;
  return base.replace(/\/$/, "").replace(/\/v1$/, "");
}

/** A hosted Azure or OpenAI-compatible model, or null when its key or base URL is missing. */
function hostedModel(
  env: Record<string, string | undefined>,
  config: { key?: string; base?: string; model: string; azureApi?: string; structured: AskModel["structuredOptions"] },
): AskModel | null {
  const { key, base, model, structured } = config;
  if (!key || !base) {
    return null;
  }
  if (isAzureHost(base)) {
    const baseURL = azureOpenAiBase(base);
    const apiVersion = first(env, ASK_API_VERSION_VARIABLES) ?? DEFAULT_AZURE_API_VERSION;
    // Reasoning deployments that refuse function tools on Chat Completions take
    // them on the Responses API, which Azure serves on its v1 surface by model.
    const languageModel =
      config.azureApi === "responses"
        ? createAzure({ baseURL, apiKey: key, apiVersion: "v1" }).responses(model)
        : createAzure({ baseURL, apiKey: key, apiVersion, useDeploymentBasedUrls: true }).chat(model);
    return { languageModel, provider: "azure", modelId: model, billed: true, structuredOptions: structured };
  }
  const openai = createOpenAI({ baseURL: base.replace(/\/$/, ""), apiKey: key });
  return { languageModel: openai.chat(model), provider: "openai", modelId: model, billed: true, structuredOptions: structured };
}

let cachedReader: AskModel | null | undefined;

/**
 * The model that reads each document one by one for a question that spans
 * many. It is the Ask model unless `FLAKEGRAPH_ASK_READER_MODEL` names another,
 * with `FLAKEGRAPH_ASK_READER_BASE_URL`, `FLAKEGRAPH_ASK_READER_API_KEY` and
 * `FLAKEGRAPH_ASK_READER_AZURE_API` (each the Ask model's when unset): a
 * question reads dozens of documents and writes one answer, so the two can be
 * priced and placed apart.
 */
export function resolveReaderModel(): AskModel | null {
  if (cachedReader !== undefined) {
    return cachedReader;
  }
  const env = askEnv();
  const model = first(env, ["FLAKEGRAPH_ASK_READER_MODEL"]);
  cachedReader = model
    ? hostedModel(env, {
        key: first(env, ["FLAKEGRAPH_ASK_READER_API_KEY"]) ?? first(env, ASK_KEY_VARIABLES),
        base: first(env, ["FLAKEGRAPH_ASK_READER_BASE_URL"]) ?? first(env, ASK_BASE_VARIABLES),
        model,
        azureApi: first(env, ["FLAKEGRAPH_ASK_READER_AZURE_API"]) ?? first(env, ["FLAKEGRAPH_ASK_AZURE_API"]),
        structured: readerStructured(env),
      })
    : resolveAskModel();
  return cachedReader;
}

function readerStructured(env: Record<string, string | undefined>): AskModel["structuredOptions"] {
  const raw = env.FLAKEGRAPH_ASK_READER_STRUCTURED_REASONING?.trim();
  return raw ? structuredOptions({ FLAKEGRAPH_ASK_STRUCTURED_REASONING: raw }) : structuredOptions(env);
}

export function askModelConfigured(): boolean {
  return resolveAskModel() != null;
}

function loadAskModel(): AskModel | null {
  const env = askEnv();
  const structured = structuredOptions(env);
  const hosted = hostedModel(env, {
    key: first(env, ASK_KEY_VARIABLES),
    base: first(env, ASK_BASE_VARIABLES),
    model: first(env, ["FLAKEGRAPH_ASK_MODEL", "AZURE_OPENAI_DEPLOYMENT"]) ?? DEFAULT_AZURE_DEPLOYMENT,
    azureApi: first(env, ["FLAKEGRAPH_ASK_AZURE_API"]),
    structured,
  });
  if (hosted) {
    return hosted;
  }

  const ollamaBase = first(env, OLLAMA_BASE_VARIABLES) ?? DEFAULT_OLLAMA_BASE_URL;
  const ollamaModel = first(env, ["FLAKEGRAPH_ASK_OLLAMA_MODEL"]) ?? DEFAULT_OLLAMA_MODEL;
  if (ollamaReachable(ollamaBase)) {
    const openai = createOpenAI({
      baseURL: ollamaOpenAiBase(ollamaBase),
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
