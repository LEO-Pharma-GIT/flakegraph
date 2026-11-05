import { lastJsonObject, runFlakegraph } from "./cli";

/**
 * What the fleet's workers run, read from what they mount.
 *
 * A worker claims only a run whose semantic configuration hashes to its own,
 * and it declines silently, so a run is composed against this profile rather
 * than against the control plane's local defaults. The CLI reads it with the
 * identity the control plane already has; here it is cached briefly so a form
 * that asks on every keystroke does not turn into a stream of kubectl calls.
 */
export interface FleetProfile {
  namespace: string;
  configMap: string;
  config: Record<string, unknown>;
  parsingEndpoint: string | null;
  ontology: Record<string, unknown> | null;
}

// Whole sections the worker digest covers. The profile's version replaces the
// request's; the keys the runtime excludes from the digest (endpoints, keys,
// parallelism) are re-applied from the request afterwards.
export const FLEET_SECTIONS = ["ocr", "generic_http_ocr", "llm", "embedding", "graph", "extractors"] as const;
export const DEPLOYMENT_LOCAL_KEYS: Record<string, readonly string[]> = {
  ocr: ["mineru_api_url", "timeout_seconds"],
  generic_http_ocr: ["endpoint", "api_key", "max_response_bytes"],
  llm: ["timeout_seconds"],
  embedding: ["batch_size", "device"],
  graph: [
    "extraction_parallelism",
    "finalization_provider_concurrency",
  ],
};

const CACHE_MILLISECONDS = 60_000;
const cache = new Map<string, { at: number; profile: FleetProfile | null }>();

export async function readFleetProfile(
  namespace: string,
  options: { cwd: string; context?: string | null },
): Promise<FleetProfile | null> {
  const key = `${options.context ?? ""}:${namespace}`;
  const cached = cache.get(key);
  if (cached && Date.now() - cached.at < CACHE_MILLISECONDS) {
    return cached.profile;
  }
  const args = ["fleet", "profile", "--namespace", namespace];
  if (options.context) {
    args.push("--context", options.context);
  }
  const result = await runFlakegraph(args, { cwd: options.cwd });
  const payload = result.exitCode === 0 ? lastJsonObject(result.stdout) : null;
  const profile = payload ? fleetProfileFromPayload(payload) : null;
  cache.set(key, { at: Date.now(), profile });
  return profile;
}

export function forgetFleetProfiles(): void {
  cache.clear();
}

export function fleetProfileFromPayload(payload: Record<string, unknown>): FleetProfile | null {
  const config = payload.config;
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    return null;
  }
  const ontology = payload.ontology;
  return {
    namespace: String(payload.namespace ?? ""),
    configMap: String(payload.config_map ?? ""),
    config: config as Record<string, unknown>,
    parsingEndpoint: typeof payload.parsing_endpoint === "string" ? payload.parsing_endpoint : null,
    ontology:
      ontology && typeof ontology === "object" && !Array.isArray(ontology)
        ? (ontology as Record<string, unknown>)
        : null,
  };
}

/** The provider the profile runs for one section, as the form shows it. */
export function fleetProvider(profile: FleetProfile, section: "ocr" | "llm" | "embedding"): {
  provider: string;
  model: string | null;
  dimension: number | null;
} | null {
  const values = profile.config[section];
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return null;
  }
  const record = values as Record<string, unknown>;
  const provider = record.provider;
  if (typeof provider !== "string" || !provider) {
    return null;
  }
  return {
    provider,
    model: typeof record.model === "string" ? record.model : null,
    dimension: typeof record.dimension === "number" ? record.dimension : null,
  };
}

/**
 * Make a run config claimable by the fleet: its semantic sections become the
 * profile's, with the request's deployment-local settings re-applied and the
 * fleet's ontology carried inline.
 */
export function composeAgainstFleet(config: Record<string, unknown>, profile: FleetProfile): void {
  for (const section of FLEET_SECTIONS) {
    const deployed = profile.config[section];
    if (!deployed || typeof deployed !== "object" || Array.isArray(deployed)) {
      continue;
    }
    const requested = (config[section] ?? {}) as Record<string, unknown>;
    const composed: Record<string, unknown> = structuredClone(deployed as Record<string, unknown>);
    for (const key of DEPLOYMENT_LOCAL_KEYS[section] ?? []) {
      if (requested[key] !== undefined) {
        composed[key] = requested[key];
      }
    }
    config[section] = composed;
  }
  if (profile.parsingEndpoint) {
    const ocr = (config.ocr ?? {}) as Record<string, unknown>;
    ocr.mineru_api_url = profile.parsingEndpoint;
    config.ocr = ocr;
  }
  // The vocabulary is the run's: a profile the request carries stays. A run
  // that names none is built with what the workers mount, carried inline so
  // the run describes itself.
  const requested = config.ontology as Record<string, unknown> | undefined;
  if (requested?.profile && typeof requested.profile === "object") {
    return;
  }
  if (profile.ontology) {
    config.ontology = { profile: structuredClone(profile.ontology) };
  } else {
    delete config.ontology;
  }
}
