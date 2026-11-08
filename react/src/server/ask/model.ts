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

export function askModelConfigured(): boolean {
  return resolveAskModel() != null;
}

function loadAskModel(): AskModel | null {
  const env = askEnv();
  const structured = structuredOptions(env);
  const askKey = first(env, ASK_KEY_VARIABLES);
  const askBase = first(env, ASK_BASE_VARIABLES);
  const askModel = first(env, ["FLAKEGRAPH_ASK_MODEL", "AZURE_OPENAI_DEPLOYMENT"]) ?? DEFAULT_AZURE_DEPLOYMENT;
  const apiVersion = first(env, ASK_API_VERSION_VARIABLES) ?? DEFAULT_AZURE_API_VERSION;

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
