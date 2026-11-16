import "server-only";

import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import type { EmbeddingModel } from "ai";
import { DEFAULT_AZURE_API_VERSION, DEFAULT_OLLAMA_BASE_URL } from "./constants";
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
} from "./provider-env";

/** The model that embeds a question for dense passage search. */
export interface AskEmbedding {
  model: EmbeddingModel;
  /** The model as configured, compared with the one the graph's chunks were embedded with. */
  modelId: string;
  provider: "azure" | "openai" | "ollama";
}

// An OpenAI-compatible server that checks no key (vLLM, a local TEI) still
// needs the header to be well formed.
const KEYLESS_API_KEY = "unused";

let cached: AskEmbedding | null | undefined;

export function resetAskEmbedding(): void {
  cached = undefined;
}

/**
 * The question embedding model, from the environment; null when none is named.
 *
 * `FLAKEGRAPH_ASK_EMBEDDING_MODEL` names it and turns dense passage search on.
 * Its endpoint and key are the Ask model's (`FLAKEGRAPH_ASK_BASE_URL`,
 * `FLAKEGRAPH_ASK_API_KEY` and their fallbacks, or the local Ollama) unless
 * `FLAKEGRAPH_ASK_EMBEDDING_BASE_URL` and `FLAKEGRAPH_ASK_EMBEDDING_API_KEY`
 * name others. An embedding endpoint of its own never receives the Ask
 * model's key: a key goes only to the host it was issued for.
 */
export function resolveAskEmbedding(): AskEmbedding | null {
  if (cached !== undefined) {
    return cached;
  }
  cached = loadAskEmbedding();
  return cached;
}

function loadAskEmbedding(): AskEmbedding | null {
  const env = askEnv();
  const modelId = first(env, ["FLAKEGRAPH_ASK_EMBEDDING_MODEL"]);
  if (!modelId) {
    return null;
  }
  const askBase = first(env, ASK_BASE_VARIABLES);
  const ownBase = first(env, ["FLAKEGRAPH_ASK_EMBEDDING_BASE_URL"]);
  const base = ownBase ?? askBase;
  const ownKey = first(env, ["FLAKEGRAPH_ASK_EMBEDDING_API_KEY"]);
  const sharesAskHost = !ownBase || sameUrl(ownBase, askBase);
  const key = ownKey ?? (sharesAskHost ? first(env, ASK_KEY_VARIABLES) : undefined);

  if (base && isAzureHost(base)) {
    if (!key) {
      return null;
    }
    const azure = createAzure({
      baseURL: azureOpenAiBase(base),
      apiKey: key,
      apiVersion: first(env, ASK_API_VERSION_VARIABLES) ?? DEFAULT_AZURE_API_VERSION,
      useDeploymentBasedUrls: true,
    });
    return { model: azure.embedding(modelId), modelId, provider: "azure" };
  }
  if (base) {
    const openai = createOpenAI({ baseURL: base.replace(/\/$/, ""), apiKey: key ?? KEYLESS_API_KEY });
    return { model: openai.embedding(modelId), modelId, provider: "openai" };
  }
  const ollamaBase = first(env, OLLAMA_BASE_VARIABLES) ?? DEFAULT_OLLAMA_BASE_URL;
  if (ollamaReachable(ollamaBase)) {
    const openai = createOpenAI({ baseURL: ollamaOpenAiBase(ollamaBase), apiKey: "ollama" });
    return { model: openai.embedding(modelId), modelId, provider: "ollama" };
  }
  return null;
}

function sameUrl(left: string, right: string | undefined): boolean {
  return right !== undefined && left.replace(/\/+$/, "") === right.replace(/\/+$/, "");
}
