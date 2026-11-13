import type { ProviderSelection, RuntimeMode } from "./protocol/schema";

/** The form uses each deployed transport's endpoint, including separate OCR contracts. */
export function fleetSelection(
  profile: { config: Record<string, unknown>; parsingEndpoint?: string | null },
  section: "ocr" | "llm" | "embedding",
  fallback: ProviderSelection,
): ProviderSelection {
  const record = profile.config[section];
  if (!record || typeof record !== "object" || Array.isArray(record)) return fallback;
  const values = record as Record<string, unknown>;
  if (typeof values.provider !== "string" || !values.provider) return fallback;
  const generic = profile.config.generic_http_ocr as Record<string, unknown> | undefined;
  const endpoint = section === "ocr" && values.provider === "generic_http" ? generic?.endpoint :
    section === "ocr" && values.provider === "mineru_api" ? profile.parsingEndpoint ?? values.mineru_api_url : values.endpoint;
  return {
    ...providerSelection(section, values.provider),
    model: typeof values.model === "string" ? values.model : null,
    endpoint: typeof endpoint === "string" && !endpoint.includes("${") ? endpoint : null,
    dimension: typeof values.dimension === "number" ? values.dimension : null,
  };
}

export interface ProviderOption {
  name: string;
  label: string;
  needsModel?: boolean;
  needsEndpoint?: boolean;
  needsApiKey?: boolean;
  defaultModel?: string | null;
  defaultEndpoint?: string | null;
  defaultDimension?: number | null;
}

/**
 * The first entry of each list is the laptop default: what a fresh checkout
 * can run with nothing installed beyond the pipeline itself. Built-in text
 * extraction needs no OCR engine; Ollama serves a small model on any
 * machine; the sentence-transformers model downloads on first use. A fleet
 * overrides all three through its profile, so these never reach a cluster.
 */
export const OCR_PROVIDERS: ProviderOption[] = [
  { name: "builtin_text", label: "Built-in document text only" },
  { name: "fallback", label: "Adaptive layout (native text, then MinerU)" },
  { name: "mineru_internal", label: "MinerU (inside worker)" },
  { name: "mineru_api", label: "MinerU API", needsEndpoint: true, needsApiKey: true },
  { name: "tesseract_internal", label: "Tesseract (inside worker)" },
  { name: "generic_http", label: "Generic HTTP OCR", needsEndpoint: true, needsApiKey: true },
  { name: "snowflake_cortex", label: "Snowflake Cortex Parse Document" },
];

export const LLM_PROVIDERS: ProviderOption[] = [
  {
    name: "ollama",
    label: "Ollama",
    needsModel: true,
    needsEndpoint: true,
    defaultModel: "qwen3:4b",
    defaultEndpoint: "http://localhost:11434",
  },
  {
    name: "vllm_local",
    label: "vLLM",
    needsModel: true,
    needsEndpoint: true,
    defaultModel: "Qwen/Qwen3-8B",
    defaultEndpoint: "http://localhost:8000/v1",
  },
  {
    name: "openai_compatible",
    label: "OpenAI-compatible API",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
  },
  {
    name: "azure_openai",
    label: "Azure OpenAI",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
  },
  { name: "snowflake_cortex", label: "Snowflake Cortex", needsModel: true, defaultModel: "mistral-large2" },
];

export const EMBEDDING_PROVIDERS: ProviderOption[] = [
  {
    name: "sentence_transformers",
    label: "Sentence Transformers",
    needsModel: true,
    defaultModel: "Qwen/Qwen3-Embedding-0.6B",
    defaultDimension: 1024,
  },
  {
    name: "openai_compatible",
    label: "OpenAI-compatible API",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
    defaultDimension: 1536,
  },
  {
    name: "azure_openai",
    label: "Azure OpenAI",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
    defaultDimension: 1536,
  },
  {
    name: "snowflake_cortex",
    label: "Snowflake Cortex",
    needsModel: true,
    defaultModel: "snowflake-arctic-embed-m-v1.5",
    defaultDimension: 768,
  },
];

/** Widths of the embedding models whose shape is known, so a selection need not state it. */
export const EMBEDDING_MODEL_DIMENSIONS: Record<string, number> = {
  "sentence-transformers/all-MiniLM-L6-v2": 384,
  // Cortex text embedding models, from Snowflake's vector embeddings guide.
  "e5-base-v2": 768,
  "snowflake-arctic-embed-m": 768,
  "snowflake-arctic-embed-m-v1.5": 768,
  "multilingual-e5-large": 1024,
  "nv-embed-qa-4": 1024,
  "snowflake-arctic-embed-l-v2.0": 1024,
  "snowflake-arctic-embed-l-v2.0-8k": 1024,
  "voyage-multilingual-2": 1024,
};

/**
 * A Snowflake VECTOR column fixes its width when the table is created, so the
 * destination decides which Cortex embedding model can be used at all.
 */
export const CORTEX_EMBEDDING_MODELS_BY_WIDTH: Record<number, string> = {
  768: "snowflake-arctic-embed-m-v1.5",
  1024: "snowflake-arctic-embed-l-v2.0",
};

/**
 * The graph-table width a Snowflake run assumes before the account has been
 * asked. Preflight still compares against what the tables really store.
 */
export const DEFAULT_SNOWFLAKE_EMBEDDING_WIDTH = 1024;

export const SUPPORTED_SUFFIXES = new Set([
  ".pdf",
  ".txt",
  ".md",
  ".markdown",
  ".html",
  ".htm",
  ".docx",
  ".png",
  ".jpg",
  ".jpeg",
  ".tif",
  ".tiff",
  ".bmp",
  ".webp",
  ".pptx",
  ".xlsx",
]);

export function embeddingDimension(
  provider: string,
  model: string | null,
  dimension: number | null,
): number {
  if (dimension && dimension > 0) {
    return dimension;
  }
  // A known model decides its own width; the provider's default width belongs
  // to the provider's default model and would mislabel any other.
  const known = model ? EMBEDDING_MODEL_DIMENSIONS[model] : undefined;
  if (known) {
    return known;
  }
  const option = EMBEDDING_PROVIDERS.find((item) => item.name === provider);
  if (option?.defaultDimension) {
    return option.defaultDimension;
  }
  return 1024;
}

export function selectionFromOption(option: ProviderOption): ProviderSelection {
  return {
    provider: option.name,
    model: option.needsModel ? option.defaultModel ?? null : null,
    endpoint: option.needsEndpoint ? option.defaultEndpoint ?? null : null,
    apiKeyEnvironmentVariable: null,
    dimension: option.defaultDimension ?? null,
    options: {},
  };
}

type ProviderKind = "ocr" | "llm" | "embedding";

function allOptions(kind: ProviderKind): ProviderOption[] {
  return kind === "ocr" ? OCR_PROVIDERS : kind === "llm" ? LLM_PROVIDERS : EMBEDDING_PROVIDERS;
}

/**
 * The providers a runtime can actually reach. Inside Snowflake a worker has no
 * route to an outside model without an external access integration, so Cortex
 * is the only choice offered there.
 */
export function providerOptions(kind: ProviderKind, runtime: RuntimeMode = "local"): ProviderOption[] {
  const options = allOptions(kind);
  return runtime === "snowflake" ? options.filter((option) => option.name === "snowflake_cortex") : options;
}

export function providerSelection(
  kind: ProviderKind,
  name: string,
  runtime: RuntimeMode = "local",
  embeddingWidth: number | null = null,
): ProviderSelection {
  const option = providerOptions(kind, runtime).find((item) => item.name === name);
  return option ? selectionFromOption(option) : defaultProvider(kind, runtime, embeddingWidth);
}

/**
 * The provider a new run starts with.
 *
 * A Snowflake run defaults to Cortex throughout, as configs/snowflake-cortex.yaml
 * does, and its embedding model is the one whose vectors fit the width the
 * graph tables already store.
 */
export function defaultProvider(
  kind: ProviderKind,
  runtime: RuntimeMode = "local",
  embeddingWidth: number | null = null,
): ProviderSelection {
  if (runtime === "snowflake") {
    if (kind === "embedding") {
      return cortexEmbeddingForWidth(embeddingWidth);
    }
    return selectionFromOption(allOptions(kind).find((item) => item.name === "snowflake_cortex")!);
  }
  if (kind === "ocr") {
    return selectionFromOption(OCR_PROVIDERS[0]);
  }
  if (kind === "llm") {
    return selectionFromOption(LLM_PROVIDERS[0]);
  }
  return selectionFromOption(EMBEDDING_PROVIDERS[0]);
}

/** The Cortex embedding selection whose vectors fit a stored width. */
export function cortexEmbeddingForWidth(width: number | null): ProviderSelection {
  const fitted = width && CORTEX_EMBEDDING_MODELS_BY_WIDTH[width] ? width : DEFAULT_SNOWFLAKE_EMBEDDING_WIDTH;
  return {
    provider: "snowflake_cortex",
    model: CORTEX_EMBEDDING_MODELS_BY_WIDTH[fitted] ?? null,
    endpoint: null,
    apiKeyEnvironmentVariable: null,
    dimension: fitted,
    options: {},
  };
}
