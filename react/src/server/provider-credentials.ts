import { invalid } from "./protocol/errors";

type Config = Record<string, unknown>;
type Environment = Record<string, string | undefined>;
type Route = { section: string; endpoint: string; credentials: string[] };
type Field = { value: unknown; identities: string[] };
type Credential = { identity: string; field: string };

const ROUTES: Route[] = [
  { section: "llm", endpoint: "endpoint", credentials: ["api_key"] },
  { section: "embedding", endpoint: "endpoint", credentials: ["api_key"] },
  { section: "ocr", endpoint: "mineru_api_url", credentials: ["mineru_api_key"] },
  { section: "generic_http_ocr", endpoint: "endpoint", credentials: ["api_key"] },
  { section: "snowflake", endpoint: "host", credentials: ["password", "oauth_token", "private_key_path", "oauth_token_path"] },
];

// Settings._environment_names applies alias names first, then conventional
// KG_<SECTION>_<FIELD> names. Ambient Snowflake defaults precede the YAML.
const ALIASES: Record<string, string[]> = {
  "embedding.provider": ["KG_EMBED_PROVIDER"],
  "embedding.endpoint": ["KG_EMBED_ENDPOINT"],
  "embedding.api_key": ["KG_EMBED_API_KEY"],
  "embedding.api_version": ["KG_EMBED_API_VERSION"],
  "ocr.mineru_api_url": ["KG_MINERU_API_URL"],
  "ocr.mineru_api_key": ["KG_MINERU_API_KEY"],
  "files.source": ["KG_FILE_SOURCE"],
};
const AMBIENT: Record<string, string[]> = {
  "snowflake.account": ["SNOWFLAKE_ACCOUNT"],
  "snowflake.host": ["SNOWFLAKE_HOST"],
  "snowflake.password": ["SNOWFLAKE_PASSWORD"],
  "snowflake.oauth_token": ["SNOWFLAKE_OAUTH_TOKEN"],
  "snowflake.private_key_path": ["SNOWFLAKE_PRIVATE_KEY_PATH"],
  "snowflake.oauth_token_path": ["SNOWFLAKE_OAUTH_TOKEN_PATH"],
  "snowflake.authenticator": ["SNOWFLAKE_AUTH", "SNOWFLAKE_AUTHENTICATOR"],
};
const DEFAULTS: Record<string, unknown> = {
  "llm.provider": "openai_compatible", "embedding.provider": "sentence_transformers",
  "ocr.provider": "builtin_text", "ocr.fallback_primary_provider": "builtin_text",
  "ocr.fallback_secondary_provider": "mineru_internal", "files.source": "local",
  "writer.provider": "local_artifacts", "cache.provider": "none",
  "snowflake.oauth_token_path": "/snowflake/session/token",
};
const PLACEHOLDER = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

/** Request values are data; only operator profiles may interpolate the server environment. */
export function assertNoRequestInterpolation(value: unknown): void {
  if (typeof value === "string" && /\$\{[^}]*\}/.test(value)) {
    throw invalid("Environment placeholders belong to operator profiles; requests name credentials by their environment variable only.");
  }
  if (Array.isArray(value)) value.forEach(assertNoRequestInterpolation);
  else if (value && typeof value === "object") Object.values(value).forEach(assertNoRequestInterpolation);
}

/** Bind each effective server credential to an operator-declared destination. */
export function assertProviderCredentialDestinations(
  config: Config,
  operatorProfiles: readonly Config[],
  environment: Environment = process.env,
): void {
  assertAzureSourceDestination(config, operatorProfiles, environment);
  const resolve = settingsResolver(environment);
  const approved = new Map<string, Set<string>>();
  const approve = (credential: Credential, endpoint: string | null) => {
    if (!endpoint || credential.identity.startsWith("unbound:")) return;
    const destinations = approved.get(credential.identity) ?? new Set<string>();
    destinations.add(endpoint);
    approved.set(credential.identity, destinations);
  };
  for (const route of ROUTES) {
    for (const profile of operatorProfiles) {
      const endpoint = destination(profile, route, resolve);
      // Authorization comes from the physical references the operator wrote,
      // not a global key override that happens to replace them during loading.
      for (const credential of declaredCredentials(profile, route, environment)) approve(credential, endpoint);
    }
    // A complete environment/inline-JSON pair is operator configuration too.
    // Generated run files and a bare conventional credential are not authority.
    if (resolve.hasEndpoint(route)) {
      const endpoint = destination({}, route, resolve);
      for (const credential of credentials({}, route, resolve, environment)) approve(credential, endpoint);
      if (route.section === "snowflake") {
        for (const field of ["private_key_path", "oauth_token_path"]) {
          const declared = resolve.field({}, "snowflake", field);
          if (declared.value && declared.identities.length > 0) {
            approve({ identity: `file:${String(declared.value)}`, field }, endpoint);
          }
        }
      }
    }
  }
  for (const route of ROUTES) {
    const record = mapping(config[route.section]);
    const explicitlyReferenced = route.credentials.some((field) =>
      references(record[field]).length > 0 || record[`${field}_environment_variable`] != null,
    );
    if ((!resolve.hasSection(config, route) || !routeIsActive(config, route, resolve)) && !explicitlyReferenced) continue;
    const used = credentials(config, route, resolve, environment);
    if (used.length === 0) continue;
    const endpoint = destination(config, route, resolve);
    assertSafeAzureParameters(config, route, resolve);
    for (const credential of used) {
      if (!endpoint || !approved.get(credential.identity)?.has(endpoint)) {
        throw invalid(`Credential for ${route.section}.${credential.field} is not approved for this provider destination. Configure its endpoint and credential together in an operator profile under configs/.`);
      }
    }
  }
}

/** Implicit Azure identities and SAS credentials may reach only operator-approved accounts. */
export function assertAzureSourceDestination(
  config: Config,
  operatorProfiles: readonly Config[],
  environment: Environment = process.env,
): void {
  const resolve = settingsResolver(environment);
  if (provider(config, "files", resolve) !== "azure_blob") return;
  // A connection string is operator configuration and fixes the SDK endpoint;
  // source requests cannot set it through sourceSettings.
  if (resolve.field(config, "azure_blob", "connection_string").value) return;
  const route: Route = { section: "azure_blob", endpoint: "account_url", credentials: [] };
  const endpoint = destination(config, route, resolve);
  const approved = new Set(operatorProfiles.flatMap((profile) => {
    if (!mapping(profile.azure_blob).account_url) return [];
    const value = destination(profile, route, resolve);
    return value ? [value] : [];
  }));
  if (resolve.hasEndpoint(route)) {
    const value = destination({}, route, resolve);
    if (value) approved.add(value);
  }
  if (!endpoint || !approved.has(endpoint)) {
    throw invalid("Azure Blob account is not an approved source destination. Configure its account URL in an operator profile under configs/ or the pipeline environment.");
  }
}

function mapping(value: unknown): Config {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Config : {};
}

function references(value: unknown): string[] {
  return typeof value === "string" ? [...value.matchAll(PLACEHOLDER)].map((match) => match[1]) : [];
}

function lastSet(names: readonly string[], environment: Environment): string | undefined {
  return names.filter((name) => environment[name] !== undefined && environment[name] !== "").at(-1);
}

function conventional(section: string, field: string): string[] {
  return [...(ALIASES[`${section}.${field}`] ?? []), `KG_${section}_${field}`.toUpperCase()];
}

/** Resolve only routing fields, with the same precedence as Settings.load. */
function settingsResolver(environment: Environment) {
  let inline: Config = {};
  if (environment.KG_CONFIG_JSON) {
    try {
      const parsed: unknown = JSON.parse(environment.KG_CONFIG_JSON);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error();
      inline = parsed as Config;
    } catch {
      throw invalid("KG_CONFIG_JSON must contain a valid configuration object.");
    }
  }
  const backend = environment.KG_AI_BACKEND?.trim().toLowerCase();
  if (environment.KG_AI_BACKEND && backend !== "oss" && backend !== "cortex") {
    throw invalid("KG_AI_BACKEND must be oss or cortex.");
  }
  const field = (config: Config, section: string, name: string): Field => {
    const key = `${section}.${name}`;
    const override = lastSet(conventional(section, name), environment);
    if (override) return { value: environment[override], identities: [`env:${override}`] };
    if (backend && name === "provider" && ["llm", "embedding", "ocr"].includes(section)) {
      return { value: backend === "cortex" ? "snowflake_cortex" :
        ({ llm: "openai_compatible", embedding: "sentence_transformers", ocr: "mineru_internal" } as Config)[section], identities: [] };
    }
    const inlineRecord = mapping(inline[section]);
    if (Object.hasOwn(inlineRecord, name)) {
      // KG_CONFIG_JSON is parsed after YAML interpolation; its strings are not
      // expanded again by Python. Its secret values stay inside this resolver.
      return { value: inlineRecord[name], identities: [`inline:${key}`] };
    }
    const value = mapping(config[section])[name];
    if (value !== undefined) {
      let missing = false;
      const interpolated = typeof value === "string" ? value.replace(PLACEHOLDER, (_, variable: string) => {
        if (!environment[variable]) missing = true;
        return environment[variable] ?? "";
      }) : value;
      // Missing whole placeholders are omitted by Python, leaving ambient
      // defaults. Mixed missing placeholders cannot execute successfully either.
      if (!missing) return { value: interpolated, identities: references(value).map((name) => `env:${name}`) };
    }
    const ambient = lastSet(AMBIENT[key] ?? [], environment);
    return ambient ? { value: environment[ambient], identities: [`env:${ambient}`] } :
      { value: DEFAULTS[key], identities: [] };
  };
  const hasEndpoint = (route: Route) => [route.endpoint, ...(route.section === "snowflake" ? ["account"] : [])]
    .some((name) => Boolean(lastSet([...conventional(route.section, name), ...(AMBIENT[`${route.section}.${name}`] ?? [])], environment)) ||
      Boolean(mapping(inline[route.section])[name]));
  const hasSection = (config: Config, route: Route) =>
    Object.hasOwn(config, route.section) || Object.hasOwn(inline, route.section) || hasEndpoint(route) ||
    Boolean(lastSet(conventional(route.section, "provider"), environment));
  return { field, hasEndpoint, hasSection };
}

type Resolver = ReturnType<typeof settingsResolver>;

function declaredCredentials(profile: Config, route: Route, environment: Environment): Credential[] {
  const record = mapping(profile[route.section]);
  const used: Credential[] = [];
  const rawValue = (field: string) => {
    const value = record[field];
    if (typeof value !== "string") return value;
    let missing = false;
    const resolved = value.replace(PLACEHOLDER, (_, name: string) => {
      if (!environment[name]) missing = true;
      return environment[name] ?? "";
    });
    return missing ? undefined : resolved;
  };
  for (const field of route.credentials.filter((name) => !name.endsWith("_path"))) {
    if (rawValue(field)) {
      for (const name of references(record[field])) {
        if (environment[name]) used.push({ identity: `env:${name}`, field });
      }
    } else if (route.section === "snowflake") {
      const reference = rawValue(`${field}_environment_variable`);
      const slot = `KG_SNOWFLAKE_${field.toUpperCase()}`;
      if (typeof reference === "string" && environment[reference]) used.push({ identity: `env:${reference}`, field });
      // A profile that names no credential of its own runs with the
      // conventional slot, which the fleet's workers read for its account.
      else if (record[field] === undefined && record[`${field}_environment_variable`] === undefined && environment[slot]) {
        used.push({ identity: `env:${slot}`, field });
      }
    }
  }
  if (route.section === "snowflake") {
    const keyFile = rawValue("private_key_path");
    if (keyFile) used.push({ identity: `file:${String(keyFile)}`, field: "private_key_path" });
    const auth = String(rawValue("authenticator") ?? "").trim().toLowerCase();
    const tokenReference = rawValue("oauth_token_environment_variable");
    const hasToken = Boolean(rawValue("oauth_token")) ||
      typeof tokenReference === "string" && Boolean(environment[tokenReference]);
    if (["oauth", "oauth_file", "azure_oauth"].includes(auth) && !hasToken) {
      const tokenFile = rawValue("oauth_token_path");
      const path = tokenFile === undefined ? DEFAULTS["snowflake.oauth_token_path"] : tokenFile;
      if (path) used.push({ identity: `file:${String(path)}`, field: "oauth_token_path" });
    }
  }
  return used;
}

function credentials(config: Config, route: Route, resolve: Resolver, environment: Environment): Credential[] {
  const used: Credential[] = [];
  const auth = String(resolve.field(config, "snowflake", "authenticator").value ?? "").trim().toLowerCase();
  const oauth = ["oauth", "oauth_file", "azure_oauth"].includes(auth);
  for (const name of route.credentials) {
    if (name.endsWith("_path") || route.section === "snowflake" && name === "oauth_token" && !oauth) continue;
    let current = resolve.field(config, route.section, name);
    if (!current.value && route.section === "snowflake") {
      const reference = resolve.field(config, route.section, `${name}_environment_variable`).value;
      if (typeof reference === "string" && environment[reference]) {
        current = { value: environment[reference], identities: [`env:${reference}`] };
      }
    }
    if (current.value) {
      for (const identity of current.identities.length ? current.identities : [`unbound:${route.section}.${name}`]) {
        used.push({ identity, field: name });
      }
    }
  }
  if (route.section === "snowflake") {
    const keyFile = resolve.field(config, "snowflake", "private_key_path").value;
    if (keyFile) used.push({ identity: `file:${String(keyFile)}`, field: "private_key_path" });
    if (oauth && !used.some((item) => item.field === "oauth_token")) {
      const tokenFile = resolve.field(config, "snowflake", "oauth_token_path").value;
      if (tokenFile) used.push({ identity: `file:${String(tokenFile)}`, field: "oauth_token_path" });
    }
  }
  return used;
}

function provider(config: Config, section: string, resolve: Resolver): string {
  const value = String(resolve.field(config, section, section === "files" ? "source" : "provider").value ?? "");
  if (section === "ocr") {
    return ({ mineru: "mineru_internal", mineru_internal: "mineru_internal", mineru_api: "mineru_api",
      tesseract: "tesseract_internal", tesseract_internal: "tesseract_internal", cortex: "snowflake_cortex",
      snowflake_cortex: "snowflake_cortex" } as Record<string, string>)[value.trim().toLowerCase()] ?? value;
  }
  if (section === "files") {
    return ({ mount: "local", mounted: "local", blob_sdk: "azure_blob", azure_blob: "azure_blob", stage: "snowflake_stage" } as Record<string, string>)[value.trim().toLowerCase()] ?? value;
  }
  return value;
}

function routeIsActive(config: Config, route: Route, resolve: Resolver): boolean {
  if (route.section === "llm") return ["ollama", "vllm_local", "openai_compatible", "azure_openai"].includes(provider(config, "llm", resolve));
  if (route.section === "embedding") return ["openai_compatible", "azure_openai"].includes(provider(config, "embedding", resolve));
  const usesOcr = (name: string) => provider(config, "ocr", resolve) === name ||
    provider(config, "ocr", resolve) === "fallback" && ["fallback_primary_provider", "fallback_secondary_provider"]
      .some((field) => resolve.field(config, "ocr", field).value === name);
  if (route.section === "ocr") return usesOcr("mineru_api");
  if (route.section === "generic_http_ocr") return usesOcr("generic_http");
  const enabled = (field: string) => [true, 1, "1", "true", "yes", "on"].includes(
    typeof resolve.field(config, "job", field).value === "string"
      ? String(resolve.field(config, "job", field).value).toLowerCase() : resolve.field(config, "job", field).value as boolean,
  );
  return ["llm", "embedding", "ocr"].some((name) => provider(config, name, resolve) === "snowflake_cortex") ||
    usesOcr("snowflake_cortex") || provider(config, "writer", resolve).startsWith("snowflake") ||
    provider(config, "cache", resolve) === "snowflake" || provider(config, "files", resolve) === "snowflake_stage" ||
    enabled("use_lease") || enabled("use_file_queue");
}

/** Reject parser ambiguities before WHATWG normalization can hide them. */
function destination(config: Config, route: Route, resolve: Resolver): string | null {
  let endpoint = resolve.field(config, route.section, route.endpoint).value;
  if (route.section === "snowflake") {
    if (!endpoint) {
      const account = resolve.field(config, "snowflake", "account").value;
      if (typeof account !== "string" || !/^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$/.test(account)) return null;
      // Snowflake's connector constructs the host before normalizing account;
      // China-region accounts use .cn and account strings keep their full suffix.
      endpoint = `${account}.snowflakecomputing.${account.split(".")[1]?.toLowerCase().startsWith("cn-") ? "cn" : "com"}`;
    }
    if (typeof endpoint !== "string" || /[\\\s/?#@]/u.test(endpoint)) return null;
    endpoint = `https://${endpoint}`;
  }
  if (typeof endpoint !== "string" || /[\\\s\u0000-\u001f\u007f#]/u.test(endpoint) || !/^https?:\/\//i.test(endpoint)) return null;
  let urlText = endpoint;
  const parts = urlText.match(/^https?:\/\/([^/?#]*)([^?#]*)/i);
  const authority = parts?.[1];
  if (!authority || authority.includes("@")) return null;
  // WHATWG removes encoded dot segments which httpx preserves. Reject such
  // paths rather than authorize one textual route and send to another.
  for (const segment of (parts?.[2] ?? "").split("/")) {
    try {
      const decoded = decodeURIComponent(segment);
      if ([".", ".."].includes(decoded) || /[\\/]/.test(decoded)) return null;
    } catch {
      return null;
    }
  }
  if (route.section !== "generic_http_ocr" && route.section !== "snowflake") {
    urlText = urlText.replace(/\/+$/, "");
    if (route.section === "llm" && provider(config, "llm", resolve) === "ollama" && urlText.endsWith("/v1")) urlText = urlText.slice(0, -3);
  }
  try {
    const url = new URL(urlText);
    return ["http:", "https:"].includes(url.protocol) && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function assertSafeAzureParameters(config: Config, route: Route, resolve: Resolver): void {
  if (!["llm", "embedding"].includes(route.section) || provider(config, route.section, resolve) !== "azure_openai") return;
  for (const field of ["model", "api_version"]) {
    const value = resolve.field(config, route.section, field).value;
    // Azure adapters interpolate these values into the request URL. They must
    // remain one path/query value rather than replacing an approved route.
    if (typeof value === "string" && (/[\\/?#&=%\s\u0000-\u001f\u007f]/u.test(value) || value === "." || value === "..")) {
      throw invalid(`Azure ${route.section}.${field} must be a single URL value.`);
    }
  }
}
