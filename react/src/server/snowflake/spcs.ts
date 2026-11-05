import { createHash } from "node:crypto";
import { stringify as stringifyYaml } from "yaml";
import { sensitiveConfigKey } from "../config";
import { IDENTIFIER } from "./stage";

const IMAGE_NAME = /^[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_.-]+)?$/;
const IMAGE_DIGEST = /^sha256:[a-f0-9]{64}$/;
const SPCS_TOKEN_PATH = "/snowflake/session/token";
/** Left for the node's own agents when a container is sized from its pool. */
const NODE_CPU_HEADROOM = 1;
const NODE_MEMORY_HEADROOM_GIB = 2;

/** Everything needed to start, and later restart, one run's job service. */
export interface SpcsLaunch {
  specYaml: string;
  computePool: string;
  serviceName: string;
  serviceIdentifier: string;
  runtimeConfig: Record<string, unknown>;
  executeSql: string;
}

/**
 * Return the SPCS job service a run is executed by.
 *
 * Derived from the run alone so anything holding a run id can find the service
 * that carries its logs and its terminal state, whoever submitted the run.
 */
export function serviceNameForJob(jobId: string): string {
  const digest = createHash("sha256").update(jobId).digest("hex").slice(0, 16);
  return `FLAKEGRAPH_APP_${digest.toUpperCase()}`;
}

/**
 * Render the statement that starts a job service asynchronously from an inline
 * specification.
 *
 * Inline rather than staged: staging the spec needed a PUT, and a PUT needs
 * egress from the app to stage storage. Submitting and retrying a run must not
 * depend on that.
 */
export function executeJobServiceSql(pool: string, service: string, specYaml: string): string {
  if (specYaml.includes("$$")) {
    throw new Error("The job specification cannot be quoted: it contains $$");
  }
  return [
    "EXECUTE JOB SERVICE",
    `  IN COMPUTE POOL ${pool}`,
    `  NAME = ${service}`,
    "  ASYNC = TRUE",
    `  FROM SPECIFICATION $$`,
    specYaml.trimEnd(),
    "$$",
  ].join("\n");
}

/**
 * Build a credential-free SPCS job from a run's effective configuration.
 *
 * The same function rebuilds the launch for a retry from the configuration
 * stored on KG_JOB, which is why it reads only what that stored copy keeps: the
 * stored copy is this function's own runtimeConfig, and applying it twice gives
 * the same specification.
 */
export function buildSpcsLaunch(config: Record<string, unknown>): SpcsLaunch {
  const job = section(config, "job");
  const snowflake = section(config, "snowflake");
  const jobId = requiredText(job, "job_id", "job");
  const serviceName = serviceNameForJob(jobId);
  const database = identifier(requiredText(snowflake, "database", "snowflake"), "database");
  const schema = identifier(requiredText(snowflake, "schema", "snowflake"), "schema");
  const computePool = identifier(requiredText(snowflake, "compute_pool", "snowflake"), "compute pool");
  const image = imageReference(snowflake);

  const runtimeConfig = withoutSecrets(config) as Record<string, unknown>;
  const runtimeJob = section(runtimeConfig, "job");
  runtimeJob.lease_owner = serviceName.toLowerCase();
  const runtimeSnowflake = section(runtimeConfig, "snowflake");
  // SPCS supplies connection coordinates paired with its short-lived OAuth
  // token. Client-side values can point at another account host or principal,
  // so they must not override the service identity inside the container.
  for (const key of [
    "account",
    "host",
    "user",
    "password",
    "authenticator",
    "private_key_path",
    "oauth_token",
    "oauth_token_path",
  ]) {
    delete runtimeSnowflake[key];
  }
  runtimeSnowflake.service_name = serviceName;
  runtimeSnowflake.authenticator = "oauth";
  runtimeSnowflake.oauth_token_path = SPCS_TOKEN_PATH;

  const env = {
    KG_CONFIG_FILE: "configs/snowflake-cortex.yaml",
    KG_CONFIG_JSON: JSON.stringify(sortKeys(runtimeConfig)),
    KG_RUNTIME: "spcs",
    KG_SNOWFLAKE_AUTHENTICATOR: "oauth",
    KG_SNOWFLAKE_OAUTH_TOKEN_PATH: SPCS_TOKEN_PATH,
  };
  const resources = {
    requests: {
      cpu: String(snowflake.service_cpu_request || "500m"),
      memory: String(snowflake.service_memory_request || "3Gi"),
    },
    limits: {
      cpu: String(snowflake.service_cpu_limit || "1"),
      memory: String(snowflake.service_memory_limit || "5Gi"),
    },
  };
  const spec = {
    spec: {
      containers: [
        {
          name: "flakegraph",
          image,
          args: ["worker", "--config", "configs/snowflake-cortex.yaml"],
          env,
          resources,
        },
      ],
    },
  };
  const specYaml = stringifyYaml(spec, { sortMapEntries: false, lineWidth: 0 });
  const serviceIdentifier = `${database}.${schema}.${serviceName}`;
  return {
    specYaml,
    computePool,
    serviceName,
    serviceIdentifier,
    runtimeConfig,
    executeSql: executeJobServiceSql(computePool, serviceIdentifier, specYaml),
  };
}

/**
 * Size the worker container from the node its compute pool provides.
 *
 * A job service occupies a node on its own and SPCS bills that node whether the
 * container asks for it or not, so requesting a fraction buys nothing. An
 * unreadable or implausible capacity returns nothing, leaving the defaults in
 * place rather than proposing a container the pool cannot schedule.
 */
export function containerResources(capacity: { vcpu?: number; memoryGib?: number }): Record<string, string> {
  const vcpu = Number(capacity.vcpu ?? 0) || 0;
  const memoryGib = Number(capacity.memoryGib ?? 0) || 0;
  if (vcpu <= 0 || memoryGib <= 0) {
    return {};
  }
  const usableCpu = Math.max(1, vcpu - NODE_CPU_HEADROOM);
  const usableMemory = Math.max(1, memoryGib - NODE_MEMORY_HEADROOM_GIB);
  return {
    service_cpu_request: formatNumber(usableCpu),
    service_cpu_limit: formatNumber(vcpu),
    service_memory_request: `${formatNumber(usableMemory)}Gi`,
    service_memory_limit: `${formatNumber(usableMemory)}Gi`,
  };
}

function formatNumber(value: number): string {
  return String(Number(value.toPrecision(6)));
}

/** Recursively remove credential-bearing keys from a runtime payload. */
export function withoutSecrets(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(withoutSecrets);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([key]) => !sensitiveConfigKey(key))
        .map(([key, item]) => [key, withoutSecrets(item)]),
    );
  }
  return value;
}

/** Order object keys recursively, so equal configurations serialise identically. */
export function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value as Record<string, unknown>)
        .sort()
        .map((key) => [key, sortKeys((value as Record<string, unknown>)[key])]),
    );
  }
  return value;
}

function section(parent: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = parent[key];
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Snowflake app configuration requires a ${key} mapping`);
  }
  return value as Record<string, unknown>;
}

function requiredText(parent: Record<string, unknown>, key: string, label: string): string {
  const value = parent[key];
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Snowflake app configuration requires ${label}.${key}`);
  }
  return value.trim();
}

function identifier(value: string, label: string): string {
  if (!IDENTIFIER.test(value)) {
    throw new Error(`Snowflake ${label} must be an unquoted identifier`);
  }
  return value;
}

function imageReference(snowflake: Record<string, unknown>): string {
  const repository = requiredText(snowflake, "image_repository", "snowflake").replace(/^\/+/, "");
  const parts = repository.replaceAll("/", ".").split(".");
  if (parts.length !== 3 || parts.some((part) => !IDENTIFIER.test(part))) {
    throw new Error("Snowflake image_repository must be DATABASE.SCHEMA.REPOSITORY");
  }
  const imageName = requiredText(snowflake, "image_name", "snowflake");
  if (!IMAGE_NAME.test(imageName)) {
    throw new Error("Snowflake image_name is not a safe OCI image and tag");
  }
  const digest = String(snowflake.image_digest ?? "").trim().toLowerCase();
  if (digest && !IMAGE_DIGEST.test(digest)) {
    throw new Error("Snowflake image_digest must be sha256 followed by 64 hex digits");
  }
  return `/${parts.join("/")}/${imageName}${digest ? `@${digest}` : ""}`;
}
