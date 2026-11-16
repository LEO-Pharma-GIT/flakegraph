import { createHash } from "node:crypto";
import { readFile, stat, writeFile, mkdir, readdir } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import {
  type IngestionRequest,
  type OntologySelection,
  type ProviderSelection,
  type SourceKind,
  writerProvider,
  storageLocation,
} from "./protocol/schema";
import { appEnv } from "./env";
import { embeddingDimension } from "./providers";
import { composeAgainstFleet, type FleetProfile } from "./fleet";
import { cliEnvironment } from "./cli";
import { ontologyProfile, ontologySelectionFromProfile } from "./ontology-profile";
import { samplePackProfile } from "./sample-packs";
import { assertNoRequestInterpolation, assertProviderCredentialDestinations } from "./provider-credentials";

const PARALLELISM_SETTINGS = [
  "extraction_parallelism",
  "finalization_provider_concurrency",
] as const;

const ENVIRONMENT_NAME = /^[A-Z_][A-Z0-9_]*$/;
/**
 * A key that holds a credential is masked in a preview by these substrings
 * of its name, spelled with either separator. Operator profiles and generated
 * configurations can contain credentials; previews mask uncertain values.
 */
const SENSITIVE_MARKERS = [
  "password",
  "passwd",
  "token",
  "secret",
  "api_key",
  "apikey",
  "private_key",
  "key_data",
  "client_key",
  "access_key",
  "connection_string",
  "database_url",
  "credential",
  "client_id",
  "tenant_id",
  "signature",
  "cert",
];
const SENSITIVE_EXACT_KEYS = new Set(["dsn", "key", "ca", "pem", "authorization", "cookie", "sas"]);
/**
 * A key that names where a credential is, not what it is: the variable a
 * run reads it from, a path, a prefix. `_name` and `_header` are not on the
 * list: "secret_name" is a reference in one product and the secret itself
 * in another, and the preview cannot tell which.
 */
const NON_SECRET_REFERENCE_SUFFIXES = ["_environment_variable", "_path", "_prefix"];
const ABSOLUTE_URL = /^[a-z][a-z0-9+.-]*:\/\//i;
const SAS_SIGNATURE = /[?&]sig=/i;
const SAS_QUERY_PARAMETER =
  /([?&](?:sig|se|sp|spr|sr|st|sv|skoid|sktid|skt|ske|sks|skv)=)[^&#\s]+/gi;
const SECRET_QUERY_PARAMETER =
  /([?&](?:access[_-]?key(?:[_-]?id)?|access[_-]?token|api[_-]?key|apikey|auth|auth[_-]?token|authorization|aws[_-]?access[_-]?key[_-]?id|client[_-]?secret|credential|key|passwd|password|private[_-]?token|secret|secret[_-]?access[_-]?key|session[_-]?token|sig|signature|subscription[_-]?key|token|x-amz-credential|x-amz-security-token|x-amz-signature)=)[^&#\s]+/gi;
const URL_PASSWORD = /(\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@]*:)[^\s/@]+(@)/gi;
const REDACTED_VALUE = "${REDACTED}";

const baseConfigCache = new Map<string, { mtime: bigint; size: number; value: Record<string, unknown> }>();

export async function loadBaseConfig(file: string | null): Promise<Record<string, unknown>> {
  if (!file) {
    return {};
  }
  const resolved = path.resolve(file);
  const info = await stat(resolved);
  const cacheKey = resolved;
  const cached = baseConfigCache.get(cacheKey);
  if (cached && cached.mtime === BigInt(Math.round(info.mtimeMs * 1e6)) && cached.size === info.size) {
    return structuredClone(cached.value);
  }
  const loaded = parseYaml(await readFile(resolved, "utf8")) ?? {};
  if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
    throw new Error("FlakeGraph configuration root must be a mapping");
  }
  const value = loaded as Record<string, unknown>;
  baseConfigCache.set(cacheKey, { mtime: BigInt(Math.round(info.mtimeMs * 1e6)), size: info.size, value });
  return structuredClone(value);
}

/**
 * The vocabulary a run gets when it chooses none: what the base config's
 * profile file declares, as terms the form can edit.
 */
export async function defaultOntologySelection(baseConfigPath: string): Promise<OntologySelection> {
  const config = await loadBaseConfig(baseConfigPath);
  const profilePath = ontologyProfilePath(config, baseConfigPath);
  const inline = asRecord(config.ontology).profile;
  const profile = inline && typeof inline === "object"
    ? (inline as Record<string, unknown>)
    : profilePath
      ? ((parseYaml(await readFile(profilePath, "utf8")) ?? {}) as Record<string, unknown>)
      : {};
  return ontologySelectionFromProfile(profile, asRecord(config.graph));
}

/**
 * What a run extracted, read back from the configuration it was submitted
 * with, or null when that run recorded no profile of its own.
 */
export async function runOntologySelection(configPath: string | null): Promise<OntologySelection | null> {
  if (!configPath) {
    return null;
  }
  let config: Record<string, unknown>;
  try {
    config = (parseYaml(await readFile(configPath, "utf8")) ?? {}) as Record<string, unknown>;
  } catch {
    return null;
  }
  const profile = asRecord(config.ontology).profile;
  if (!profile || typeof profile !== "object") {
    return null;
  }
  return ontologySelectionFromProfile(profile as Record<string, unknown>, asRecord(config.graph));
}

/**
 * The run's full configuration. `owner` is who the graph belongs to; a fleet
 * run records it, and the pipeline refuses a new fleet graph without one.
 */
export async function buildRunConfig(
  request: IngestionRequest,
  fleet: FleetProfile | null = null,
  owner: string | null = null,
): Promise<Record<string, unknown>> {
  assertNoRequestInterpolation(request);
  rejectLiteralSecrets(request.source, "source");
  const config = await loadBaseConfig(request.baseConfigPath);
  inlineOntologyProfile(config, ontologyProfilePath(config, request.baseConfigPath));
  rejectLiteralSecrets(config, "base configuration");
  const embedding = providerConfig(request.embedding, "embedding");
  embedding.dimension = embeddingDimension(
    request.embedding.provider,
    request.embedding.model,
    request.embedding.dimension,
  );
  const overrides: Record<string, unknown> = {
    runtime: { runtime: runtimeValue(request) },
    job: {
      job_id: request.jobId,
      graph_id: request.graphId,
      use_file_queue: request.runtime === "snowflake",
      ...(owner ? { owner } : {}),
    },
    files: { include_globs: request.includeGlobs },
    ocr: ocrProviderConfig(request.ocr),
    llm: providerConfig(request.llm, "llm"),
    embedding,
    writer: {
      provider: writerProvider(request.output.kind),
      output_path: request.output.workspacePath,
    },
    cache: {
      provider: request.cacheProvider,
      path: path.join(path.dirname(request.output.workspacePath), "cache"),
    },
  };
  if (request.output.kind === "snowflake") {
    deepMerge(overrides, snowflakeOutputConfig(request));
  }
  deepMerge(overrides, { graph: providerParallelismSettings(request.providerParallelism) });
  if (request.runtime === "snowflake" && Object.keys(request.runtimeOptions).length > 0) {
    const runtimeOptions = Object.fromEntries(
      Object.entries(request.runtimeOptions).filter(
        ([, value]) => typeof value !== "string" || value.trim(),
      ),
    );
    if (Object.keys(runtimeOptions).length > 0) {
      deepMerge(overrides, { snowflake: runtimeOptions });
    }
  }
  if (request.ocr.provider === "generic_http") {
    overrides.generic_http_ocr = externalOcrConfig(request.ocr);
  }
  deepMerge(overrides, sourceConfig(request));
  deepMerge(config, overrides);
  if (request.ontology) {
    // The run's vocabulary travels inline so it describes itself wherever it
    // runs, built on the profile its terms were offered from.
    config.ontology = { profile: ontologyProfile(request.ontology, await offeredProfile(request, config, fleet)) };
  }
  applyFolderExclusions(config, request.revision?.folderExclusions ?? null);
  sanitizeProviderSections(config, request);
  if (fleet) {
    composeAgainstFleet(config, fleet);
  }
  assertProviderCredentialDestinations(config, await operatorProviderProfiles(fleet));
  rejectLiteralSecrets(config);
  return config;
}

/**
 * A revision's folder exclusions replace the base version's: the documents
 * are named by file id, so no path the base version listed is kept.
 */
function applyFolderExclusions(config: Record<string, unknown>, fileIds: readonly string[] | null): void {
  const profile = asRecord(config.ontology).profile;
  const folder = isPlainObject(profile) ? profile.folder_entity : null;
  if (fileIds === null || !isPlainObject(folder)) {
    return;
  }
  folder.excluded_file_ids = [...new Set(fileIds)].sort();
  folder.excluded_paths = [];
}

/**
 * The account a run started inside Snowflake names. Its worker holds the token
 * Snowflake mounts into the job service and connects where that token points,
 * so this is not a coordinate anyone can aim anywhere.
 */
export const ACTIVE_SNOWFLAKE_SESSION = "active-session";

/**
 * Only server-managed profiles and the mounted fleet profile authorize credential
 * destinations. A console hosted in Snowflake adds its own session: the operator
 * configured that by deploying it there, and nothing a request says reaches it.
 */
export async function operatorProviderProfiles(fleet: FleetProfile | null = null): Promise<Record<string, unknown>[]> {
  const files: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true }).catch(() => [])) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await visit(file);
      else if (/\.ya?ml$/i.test(entry.name)) files.push(file);
    }
  };
  await visit(path.join(appEnv().repositoryRoot, "configs"));
  const profiles = await Promise.all(files.map((file) => loadBaseConfig(file).catch(() => null)));
  const hosted = appEnv().snowflakeHosted
    ? [{ snowflake: { account: ACTIVE_SNOWFLAKE_SESSION, host: null, authenticator: "oauth" } }]
    : [];
  return [...profiles.filter((profile): profile is Record<string, unknown> => profile !== null), ...hosted, ...(fleet ? [fleet.config] : [])];
}

export async function writeRunConfig(
  request: IngestionRequest,
  destination: string,
  fleet: FleetProfile | null = null,
  owner: string | null = null,
): Promise<string> {
  await mkdir(path.dirname(destination), { recursive: true });
  const config = await buildRunConfig(request, fleet, owner);
  await writeFile(destination, stringifyYaml(config, { sortMapEntries: false }), "utf8");
  return destination;
}

export function redactedConfig(config: Record<string, unknown>): Record<string, unknown> {
  const redact = (value: unknown, key = ""): unknown => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([itemKey, item]) => [
          itemKey,
          redact(item, itemKey),
        ]),
      );
    }
    if (Array.isArray(value)) {
      return value.map((item) => redact(item, key));
    }
    if (sensitiveConfigKey(key)) {
      return value ? REDACTED_VALUE : value;
    }
    if (typeof value === "string") {
      return redactedUrl(value);
    }
    return value;
  };
  return redact(config) as Record<string, unknown>;
}

export function redactedUrl(value: string): string {
  if (!ABSOLUTE_URL.test(value)) {
    return value;
  }
  let redacted = SAS_SIGNATURE.test(value) ? value.replace(SAS_QUERY_PARAMETER, "$1***") : value;
  redacted = redacted.replace(SECRET_QUERY_PARAMETER, "$1***");
  return redacted.replace(URL_PASSWORD, "$1***$2");
}

export function sensitiveConfigKey(key: string): boolean {
  const normalized = key.trim().toLowerCase().replaceAll("-", "_");
  if (NON_SECRET_REFERENCE_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) {
    return false;
  }
  return SENSITIVE_EXACT_KEYS.has(normalized) || SENSITIVE_MARKERS.some((marker) => normalized.includes(marker));
}

export async function effectiveRequestFingerprint(
  request: IngestionRequest,
  effectiveConfig?: Record<string, unknown>,
): Promise<string> {
  const config = effectiveConfig ?? (await buildRunConfig(request));
  const identity = {
    effective_config: config,
    request: {
      runtime: request.runtime,
      job_id: request.jobId,
      graph_id: request.graphId,
      source_kind: request.sourceKind,
      source: request.source,
      base_config_path: request.baseConfigPath,
      include_globs: request.includeGlobs,
      cache_provider: request.cacheProvider,
      runtime_options: request.runtimeOptions,
      ocr: selectionIdentity(request.ocr),
      llm: selectionIdentity(request.llm),
      embedding: selectionIdentity(request.embedding),
      output_kind: request.output.kind,
      output_location: storageLocation(request.output),
    },
  };
  return createHash("sha256").update(stringifyYaml(identity, { sortMapEntries: true })).digest("hex");
}

/**
 * The environment the pipeline runs with: an allow-listed view of this
 * process's, never the whole of it. A request names the credentials it
 * needs by `KG_…` name (see `confine.ts`); every `KG_*` variable the
 * operator set travels, so the config's `${KG_…}` placeholders resolve.
 */
export function environmentForRequest(
  request: IngestionRequest,
  base: Record<string, string | undefined> = process.env,
): Record<string, string> {
  const environment = cliEnvironment(base);
  const named = [
    request.ocr.apiKeyEnvironmentVariable,
    request.llm.apiKeyEnvironmentVariable,
    request.embedding.apiKeyEnvironmentVariable,
    request.output.snowflake?.credentialEnvironmentVariable,
  ];
  for (const name of named) {
    if (name && !(name in environment)) {
      throw new Error(`Credential environment variable is not set: ${name}`);
    }
  }
  return environment;
}

export function providerParallelismSettings(parallelism: number): Record<string, number> {
  const bounded = Math.max(1, Math.min(Math.trunc(parallelism), 64));
  return Object.fromEntries(PARALLELISM_SETTINGS.map((key) => [key, bounded]));
}

function sourceConfig(request: IngestionRequest): Record<string, unknown> {
  return sourceSettings(request.sourceKind, request.source, {
    downloadPath: path.join(path.dirname(request.output.workspacePath), "source-cache"),
  });
}

/**
 * The `files` section (plus its backend block) that makes the pipeline read
 * one source. A run writes it inside its full config; browsing a bucket
 * before a run writes it alone, so the listing and the run agree on what the
 * source contains.
 */
export function sourceSettings(
  sourceKind: SourceKind,
  source: Record<string, unknown>,
  options: { downloadPath?: string } = {},
): Record<string, unknown> {
  assertNoRequestInterpolation(source);
  const download = options.downloadPath ? { download_path: options.downloadPath } : {};
  if (sourceKind === "upload" || sourceKind === "local_path") {
    return { files: { source: "local", input_path: String(source.path ?? "") } };
  }
  if (sourceKind === "azure_blob") {
    return {
      files: { source: "azure_blob" },
      azure_blob: {
        account_url: source.accountUrl,
        container: source.container,
        prefix: source.prefix,
        ...download,
      },
    };
  }
  if (sourceKind === "s3") {
    return {
      files: { source: "s3" },
      s3: {
        bucket: source.bucket,
        prefix: source.prefix,
        endpoint_url: source.endpointUrl,
        region: source.region,
        ...download,
      },
    };
  }
  if (sourceKind === "snowflake_stage") {
    return {
      files: { source: "snowflake_stage", stage_prefix: source.prefix },
      snowflake: { stage: source.stage },
    };
  }
  throw new Error(`Unsupported source kind: ${sourceKind}`);
}

/** Caller options contain data settings only; executable configuration belongs to the operator. */
function providerConfig(selection: ProviderSelection, kind: "ocr" | "llm" | "embedding"): Record<string, unknown> {
  const common = kind === "llm"
    ? ["timeout_seconds", "context_window_tokens", "max_output_tokens"]
    : kind === "embedding" ? ["batch_size"] : ["language", "page_range", "timeout_seconds"];
  const mineru = ["mineru_method", "mineru_backend", "mineru_effort", "mineru_formula", "mineru_table", "mineru_image_analysis"];
  const byProvider: Record<string, readonly string[]> = kind === "ocr" ? {
    fallback: ["fallback_min_characters_per_page", "fallback_max_sparse_page_ratio", "fallback_max_unbroken_text_ratio", "fallback_max_fragmented_text_ratio", "fallback_max_unspaced_page_ratio", "fallback_max_interleaved_column_page_ratio", ...mineru, "tesseract_dpi"],
    mineru_internal: mineru,
    tesseract_internal: ["tesseract_dpi"],
    snowflake_cortex: ["snowflake_parse_mode", "snowflake_extract_images", "snowflake_page_split"],
  } : { azure_openai: ["api_version"], sentence_transformers: ["device", "torch_dtype"] };
  const allowed = new Set([...common, ...(byProvider[selection.provider] ?? [])]);
  for (const key of Object.keys(selection.options)) {
    if (!allowed.has(key)) throw new Error(`Unsupported ${kind} option: ${key}`);
  }
  const result: Record<string, unknown> = { ...selection.options, provider: selection.provider };
  if (selection.model) {
    result.model = selection.model;
  }
  if (selection.endpoint) {
    result.endpoint = selection.endpoint;
  }
  if (selection.apiKeyEnvironmentVariable) {
    const name = selection.apiKeyEnvironmentVariable.trim();
    if (!ENVIRONMENT_NAME.test(name)) {
      throw new Error(`Invalid credential environment variable name: ${name}`);
    }
    result.api_key = `\${${name}}`;
  }
  return result;
}

function ocrProviderConfig(selection: ProviderSelection): Record<string, unknown> {
  const result = providerConfig(selection, "ocr");
  delete result.endpoint;
  delete result.api_key;
  if (selection.provider === "mineru_api") {
    if (selection.endpoint) result.mineru_api_url = selection.endpoint;
    if (selection.apiKeyEnvironmentVariable) result.mineru_api_key = `\${${selection.apiKeyEnvironmentVariable.trim()}}`;
  }
  return result;
}

function externalOcrConfig(selection: ProviderSelection): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  if (selection.endpoint) {
    config.endpoint = selection.endpoint;
  }
  if (selection.apiKeyEnvironmentVariable) {
    config.api_key = `\${${selection.apiKeyEnvironmentVariable}}`;
  }
  return config;
}

function snowflakeOutputConfig(request: IngestionRequest): Record<string, unknown> {
  const snowflake = request.output.snowflake;
  if (!snowflake) {
    throw new Error("Snowflake destination requires account coordinates");
  }
  const connection: Record<string, unknown> = {
    account: snowflake.account,
    user: snowflake.user,
    database: snowflake.database,
    schema: snowflake.schema,
    warehouse: snowflake.warehouse,
    role: snowflake.role,
    host: snowflake.host,
    authenticator: snowflake.authenticator,
    bulk_stage: snowflake.bulkStage,
  };
  if (snowflake.credentialEnvironmentVariable) {
    const field = snowflake.credentialField || "password";
    if (!["password", "oauth_token", "private_key_path"].includes(field)) {
      throw new Error("Snowflake credential field must be password, oauth_token, or private_key_path");
    }
    connection[field] = `\${${snowflake.credentialEnvironmentVariable}}`;
  }
  return { snowflake: connection };
}

function sanitizeProviderSections(config: Record<string, unknown>, request: IngestionRequest): void {
  config.llm = providerSection(config.llm, providerConfig(request.llm, "llm"), {
    allowed: new Set(
      request.llm.provider === "azure_openai" ? ["timeout_seconds", "api_version"] : ["timeout_seconds"],
    ),
  });
  const embedding = providerConfig(request.embedding, "embedding");
  embedding.dimension = embeddingDimension(
    request.embedding.provider,
    request.embedding.model,
    request.embedding.dimension,
  );
  const embeddingAllowed = new Set(["batch_size"]);
  if (request.embedding.provider === "sentence_transformers") {
    embeddingAllowed.add("device");
  }
  if (request.embedding.provider === "azure_openai") {
    embeddingAllowed.add("api_version");
  }
  config.embedding = providerSection(config.embedding, embedding, { allowed: embeddingAllowed });
  const ocrAllowed = new Set(["language", "page_range", "timeout_seconds"]);
  const prefixes: string[] = [];
  if (request.ocr.provider === "fallback") {
    prefixes.push("fallback_", "mineru_", "tesseract_");
    ocrAllowed.add("model_cache_dir");
  } else if (request.ocr.provider === "mineru_internal") {
    prefixes.push("mineru_");
    ocrAllowed.add("model_cache_dir");
  } else if (request.ocr.provider === "tesseract_internal") {
    prefixes.push("tesseract_");
  } else if (request.ocr.provider === "snowflake_cortex") {
    prefixes.push("snowflake_");
  }
  config.ocr = providerSection(config.ocr, ocrProviderConfig(request.ocr), {
    allowed: ocrAllowed,
    prefixes,
    excluded: new Set(["mineru_api_url", "mineru_api_key"]),
  });
  if (request.ocr.provider === "generic_http") {
    const existing = asRecord(config.generic_http_ocr);
    delete existing.endpoint;
    delete existing.api_key;
    config.generic_http_ocr = { ...existing, ...externalOcrConfig(request.ocr) };
  } else {
    delete config.generic_http_ocr;
  }
}

function providerSection(
  existing: unknown,
  selected: Record<string, unknown>,
  options: { allowed: Set<string>; prefixes?: string[]; excluded?: Set<string> },
): Record<string, unknown> {
  const blocked = options.excluded ?? new Set<string>();
  const retained: Record<string, unknown> = {};
  const current = asRecord(existing);
  for (const [key, value] of Object.entries(current)) {
    if (blocked.has(key)) {
      continue;
    }
    if (options.allowed.has(key) || options.prefixes?.some((prefix) => key.startsWith(prefix))) {
      retained[key] = value;
    }
  }
  return { ...retained, ...selected };
}

function runtimeValue(request: IngestionRequest): string {
  if (request.runtime === "kubernetes") {
    return "kubernetes";
  }
  if (request.runtime === "snowflake") {
    return "snowflake";
  }
  return "local";
}

/**
 * The profile the form offered the run's terms from: a sample pack's own
 * when the source is one, else what the fleet's workers mount, else the base
 * configuration's. Its terms carry more than the form edits.
 */
async function offeredProfile(
  request: IngestionRequest,
  config: Record<string, unknown>,
  fleet: FleetProfile | null,
): Promise<Record<string, unknown> | null> {
  const sourcePath = request.sourceKind === "local_path" ? String(request.source.path ?? "") : "";
  const pack = sourcePath ? await samplePackProfile(appEnv().repositoryRoot, sourcePath) : null;
  if (pack) {
    return pack;
  }
  if (fleet?.ontology) {
    return fleet.ontology;
  }
  const ontology = asRecord(config.ontology);
  if (ontology.profile && typeof ontology.profile === "object") {
    return ontology.profile as Record<string, unknown>;
  }
  if (typeof ontology.profile_path !== "string") {
    return null;
  }
  try {
    return (parseYaml(await readFile(ontology.profile_path, "utf8")) ?? null) as Record<string, unknown> | null;
  } catch {
    return null;
  }
}

/**
 * Where the base configuration's profile file is. The pipeline resolves a
 * relative path against its working directory - the repository root, where
 * the console runs it - not against the configuration file naming it.
 */
function ontologyProfilePath(config: Record<string, unknown>, baseConfigPath: string | null): string | null {
  const ontology = asRecord(config.ontology);
  const profilePath = ontology.profile_path;
  if (typeof profilePath !== "string" || !profilePath) {
    return null;
  }
  if (path.isAbsolute(profilePath)) {
    return profilePath;
  }
  return path.resolve(baseConfigPath ? repositoryRootOf(baseConfigPath) : appEnv().repositoryRoot, profilePath);
}

/** The repository root a base configuration under `<root>/configs/` belongs to. */
function repositoryRootOf(baseConfigPath: string): string {
  const directory = path.dirname(path.resolve(baseConfigPath));
  return path.basename(directory) === "configs" ? path.dirname(directory) : appEnv().repositoryRoot;
}

function inlineOntologyProfile(config: Record<string, unknown>, profilePath: string | null): void {
  if (!profilePath) {
    return;
  }
  const ontology = asRecord(config.ontology);
  if (ontology.profile && typeof ontology.profile === "object") {
    return;
  }
  ontology.profile_path = profilePath;
  config.ontology = ontology;
}

function rejectLiteralSecrets(value: unknown, pathLabel = "configuration"): void {
  const walk = (item: unknown, current: string): void => {
    if (Array.isArray(item)) {
      item.forEach((entry, index) => walk(entry, `${current}[${index}]`));
      return;
    }
    if (!item || typeof item !== "object") {
      if (typeof item === "string" && looksLikeLiteralSecret(current, item)) {
        throw new Error(`Refusing to persist a literal secret at ${current}`);
      }
      return;
    }
    for (const [key, nested] of Object.entries(item as Record<string, unknown>)) {
      walk(nested, `${current}.${key}`);
    }
  };
  walk(value, pathLabel);
}

function looksLikeLiteralSecret(key: string, value: string): boolean {
  const last = key.split(".").pop() ?? key;
  if (!sensitiveConfigKey(last)) {
    return false;
  }
  if (value.startsWith("${") && value.endsWith("}")) {
    return false;
  }
  return value.trim().length > 0;
}

function selectionIdentity(selection: ProviderSelection): Record<string, unknown> {
  return {
    provider: selection.provider,
    model: selection.model,
    endpoint: selection.endpoint,
    api_key_environment_variable: selection.apiKeyEnvironmentVariable,
    dimension: selection.dimension,
    options: selection.options,
  };
}

function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>): Record<string, unknown> {
  for (const [key, value] of Object.entries(source)) {
    const existing = target[key];
    if (isPlainObject(existing) && isPlainObject(value)) {
      target[key] = deepMerge({ ...existing }, value);
    } else {
      target[key] = value;
    }
  }
  return target;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isPlainObject(value) ? { ...value } : {};
}
