import path from "node:path";
import { existsSync } from "node:fs";
import type { RuntimeMode } from "./protocol/schema";

export interface AppEnv {
  repositoryRoot: string;
  stateRoot: string;
  defaultRuntime: RuntimeMode;
  kubernetesNamespace: string;
  databaseUrl: string | null;
  flakegraphCommand: string[];
  /**
   * Believe the identity headers a sign-in gate writes. Sound only when
   * nothing but that gate can reach this server: a caller who reaches it
   * directly can write the same headers and become anyone.
   */
  trustIdentityHeaders: boolean;
  stubRuntimes: boolean;
  /** Running inside a Snowflake container, where Sf-Context headers name the viewer. */
  snowflakeHosted: boolean;
  /** Refuse an anonymous caller everywhere, even where demo identities or anonymous reads would serve one. */
  requireSignIn: boolean;
  /**
   * A laptop or rehearsal console with no gate: the viewer is whoever the
   * workspace says was assumed, and a session that assumed nobody is still
   * served. Production never sets this.
   */
  demoIdentities: boolean;
  /** Fixed principal for an explicitly isolated, single-operator server. */
  singleOperator: string | null;
  /** Serve read-only catalog queries to a caller nobody identified. Closed by default. */
  anonymousRead: boolean;
  /**
   * Who administers a console the gate or Snowflake serves to many people:
   * principals (a sign-in address or user name) and roles (the gate's groups,
   * the Snowflake roles), upper-cased. Empty means nobody does.
   */
  operators: string[];
  /**
   * Folders a run may read documents from, besides the checkout and the
   * console's own state. Paths a request names are confined to these roots.
   */
  sourceRoots: string[];
  /** Where the hosted console reaches Snowflake; unset outside it. */
  snowflake: SnowflakeConnectionEnv;
}

/**
 * Where the hosted app reaches Snowflake. App Runtime supplies the account, host
 * and namespace of the service, and a short-lived OAuth token on disk; none of it
 * is a credential the app stores.
 */
export interface SnowflakeConnectionEnv {
  account: string | null;
  host: string | null;
  database: string | null;
  schema: string | null;
  warehouse: string | null;
  tokenPath: string;
}

/** Writable scratch space inside a Snowflake-hosted container. */
const SNOWFLAKE_HOSTED_STATE_ROOT = "/tmp/flakegraph";

let cached: AppEnv | null = null;

export function appEnv(): AppEnv {
  cached ??= loadEnv();
  return cached;
}

export function resetAppEnv(): void {
  cached = null;
}

export function loadEnv(overrides: Partial<NodeJS.ProcessEnv> = process.env): AppEnv {
  const cwd = process.cwd();
  const repositoryRoot = resolveRepositoryRoot(
    overrides.FLAKEGRAPH_REPOSITORY_ROOT ?? cwd,
  );
  const configuredState = overrides.FLAKEGRAPH_APP_STATE_ROOT?.trim();
  const singleOperator = overrides.FLAKEGRAPH_SINGLE_OPERATOR?.trim().toUpperCase() || null;
  if (singleOperator && (!/^[^\s@]+@[^\s@]+$/.test(singleOperator) ||
      [overrides.FLAKEGRAPH_TRUST_IDENTITY_HEADERS, overrides.FLAKEGRAPH_APP_REQUIRE_SIGN_IN,
        overrides.FLAKEGRAPH_SNOWFLAKE_HOSTED, overrides.FLAKEGRAPH_DEMO_IDENTITIES].some(isTruthy))) {
    throw new Error("FLAKEGRAPH_SINGLE_OPERATOR requires an isolated server, a valid operator address, and no identity gate or demo identities.");
  }
  // Only an explicit flag: ambient SNOWFLAKE_* variables belong to the Python
  // writer and must not switch the console into trusting Sf-Context headers
  // from whoever sends them.
  const snowflakeHosted = isTruthy(overrides.FLAKEGRAPH_SNOWFLAKE_HOSTED);
  // Hosted in Snowflake, the working directory is the read-only server bundle,
  // so app state defaults to scratch space the container can write.
  const defaultState = snowflakeHosted
    ? SNOWFLAKE_HOSTED_STATE_ROOT
    : path.join(repositoryRoot, ".flakegraph", "app");
  return {
    repositoryRoot,
    stateRoot: path.resolve(/* turbopackIgnore: true */ configuredState || defaultState),
    defaultRuntime: parseRuntime(overrides.FLAKEGRAPH_APP_DEFAULT_RUNTIME, "local"),
    kubernetesNamespace: overrides.FLAKEGRAPH_APP_KUBERNETES_NAMESPACE?.trim() || "flakegraph",
    databaseUrl: overrides.DATABASE_URL?.trim() || overrides.FLAKEGRAPH_DATABASE_URL?.trim() || null,
    flakegraphCommand: parseCommand(overrides.FLAKEGRAPH_CLI),
    trustIdentityHeaders: isTruthy(overrides.FLAKEGRAPH_TRUST_IDENTITY_HEADERS),
    stubRuntimes: isTruthy(overrides.FLAKEGRAPH_STUB_RUNTIMES),
    snowflakeHosted,
    // Snowflake's ingress names everyone it admits, so a caller with no name
    // never came through it, and nobody hosted there assumes an identity.
    requireSignIn: snowflakeHosted || isTruthy(overrides.FLAKEGRAPH_APP_REQUIRE_SIGN_IN),
    demoIdentities: !snowflakeHosted && isTruthy(overrides.FLAKEGRAPH_DEMO_IDENTITIES),
    singleOperator,
    anonymousRead: isTruthy(overrides.FLAKEGRAPH_APP_ANONYMOUS_READ),
    operators: (overrides.FLAKEGRAPH_APP_OPERATORS ?? "")
      .split(/[,\s]+/)
      .map((item) => item.trim().toUpperCase())
      .filter(Boolean),
    sourceRoots: parsePathList(overrides.FLAKEGRAPH_APP_SOURCE_ROOTS),
    snowflake: {
      account: overrides.SNOWFLAKE_ACCOUNT?.trim() || null,
      host: overrides.SNOWFLAKE_HOST?.trim() || null,
      database: overrides.SNOWFLAKE_DATABASE?.trim() || null,
      schema: overrides.SNOWFLAKE_SCHEMA?.trim() || null,
      warehouse:
        overrides.FLAKEGRAPH_SNOWFLAKE_WAREHOUSE?.trim() || overrides.SNOWFLAKE_QUERY_WAREHOUSE?.trim() || null,
      tokenPath: overrides.FLAKEGRAPH_SNOWFLAKE_TOKEN_PATH?.trim() || "/snowflake/session/token",
    },
  };
}

export function parseRuntime(value: string | undefined, fallback: RuntimeMode): RuntimeMode {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "kubernetes" || normalized === "k8s") {
    return "kubernetes";
  }
  if (normalized === "snowflake") {
    return "snowflake";
  }
  if (normalized === "local") {
    return "local";
  }
  return fallback;
}

function parseCommand(value: string | undefined): string[] {
  const trimmed = value?.trim();
  if (trimmed) {
    return trimmed.split(/\s+/);
  }
  return ["uv", "run", "flakegraph"];
}

function isTruthy(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

function parsePathList(value: string | undefined): string[] {
  return (value ?? "")
    .split(path.delimiter)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => path.resolve(item));
}

export function resolveRepositoryRoot(start: string): string {
  let current = path.resolve(start);
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(path.join(current, "configs", "app-defaults.yaml"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  const fromReact = path.resolve(start, "..");
  if (existsSync(path.join(fromReact, "configs", "app-defaults.yaml"))) {
    return fromReact;
  }
  return path.resolve(start);
}
