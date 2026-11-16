import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { buildRunConfig } from "./config";
import { resetAppEnv } from "./env";
import type { IngestionRequest } from "./protocol/schema";
import { assertNoRequestInterpolation, assertProviderCredentialDestinations as check } from "./provider-credentials";

const environment = { KG_LLM_API_KEY: "synthetic-key", KG_SNOWFLAKE_PASSWORD: "synthetic-password" };
const profile = { llm: { provider: "openai_compatible", endpoint: "https://models.example.test/v1", api_key: "${KG_LLM_API_KEY}" } };
const config = (endpoint: string, name = "KG_LLM_API_KEY") => ({ llm: { provider: "openai_compatible", endpoint, api_key: `\${${name}}` } });

describe("server credential destinations", () => {
  it("allows model changes at the operator's destination and rejects endpoint substitution", () => {
    expect(() => check({ ...config("https://models.example.test/v1/"), model: "another-model" }, [profile], environment)).not.toThrow();
    for (const endpoint of ["https://other.example.test/v1", "https://models.example.test/other", "https://models.example.test/v1?forward=elsewhere"]) {
      expect(() => check(config(endpoint), [profile], environment)).toThrow(/not approved/);
    }
  });
  it("does not let a caller send an unrelated server credential to an arbitrary API", () => {
    expect(() => check(config("https://caller.example.test/v1", "KG_SNOWFLAKE_PASSWORD"), [profile], environment)).toThrow(/not approved/);
    expect(() => check(config("https://caller.example.test/v1"), [], environment)).toThrow(/not approved/);
  });
  it("follows effective environment endpoint/key overrides", () => {
    expect(() => check(config("https://caller.example.test/v1"), [], { ...environment, KG_LLM_ENDPOINT: "https://operator.example.test/v1" })).not.toThrow();
    // A bare conventional key must not be sent to the caller's endpoint even without a YAML key reference.
    expect(() => check({ llm: { provider: "ollama", endpoint: "https://caller.example.test" } }, [], environment)).toThrow(/not approved/);
  });
  it("binds embedding and OCR credentials to their respective endpoints", () => {
    const profiles = [{ embedding: { endpoint: "https://embed.example.test/v1", api_key: "${KG_EMBED_API_KEY}" }, generic_http_ocr: { endpoint: "https://ocr.example.test/parse", api_key: "${KG_OCR_KEY}" } }];
    const env = { KG_EMBED_API_KEY: "synthetic", KG_OCR_KEY: "synthetic" };
    const cfg = { embedding: { provider: "openai_compatible", endpoint: "https://embed.example.test/v1", api_key: "${KG_EMBED_API_KEY}" }, ocr: { provider: "generic_http" }, generic_http_ocr: { endpoint: "https://ocr.example.test/parse", api_key: "${KG_OCR_KEY}" } };
    expect(() => check(cfg, profiles, env)).not.toThrow();
    expect(() => check({ ...cfg, generic_http_ocr: { ...cfg.generic_http_ocr, endpoint: "https://caller.example.test" } }, profiles, env)).toThrow(/not approved/);
  });
  it("binds Snowflake password variable references to the account and honors YAML over ambient defaults", () => {
    const profiles = [{ snowflake: { account: "trusted-account", password_environment_variable: "KG_SNOWFLAKE_PASSWORD" } }];
    const cfg = { writer: { provider: "snowflake_bulk" }, snowflake: { account: "trusted-account", password: "${KG_SNOWFLAKE_PASSWORD}" } };
    expect(() => check(cfg, profiles, { ...environment, SNOWFLAKE_HOST: "ambient.example.test" })).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, host: "caller.example.test" } }, profiles, environment)).toThrow(/not approved/);
    expect(() => check({ llm: { provider: "fake" }, snowflake: { account: "unrelated" } }, [], environment)).not.toThrow();
  });
  it("lets a fleet profile's Snowflake account receive the fleet's conventional password slot", () => {
    // The fleet names its account and user; its workers read the fixed password slot.
    const fleet = { snowflake: { account: "fleet-account", user: "fleet-user" } };
    const cfg = { writer: { provider: "snowflake_bulk" }, snowflake: { account: "fleet-account", user: "fleet-user", password: "${KG_SNOWFLAKE_PASSWORD}" } };
    expect(() => check(cfg, [fleet], environment)).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, account: "caller-account" } }, [fleet], environment)).toThrow(/not approved/);
    // A profile that names its own credential approves only that credential.
    const named = { snowflake: { account: "fleet-account", password: "${KG_OTHER_PASSWORD}" } };
    expect(() => check(cfg, [named], environment)).toThrow(/not approved/);
  });
  it("refuses request placeholders in endpoint, source and free-text values", () => {
    for (const value of [{ endpoint: "https://example.test/${KG_SECRET}" }, { source: { prefix: "${KG_SECRET}" } }, { ontology: [{ description: "${KG_SECRET}" }] }]) {
      expect(() => assertNoRequestInterpolation(value)).toThrow(/operator profiles/);
    }
    expect(() => assertNoRequestInterpolation({ endpoint: "https://example.test", apiKeyEnvironmentVariable: "KG_LLM_API_KEY" })).not.toThrow();
  });
});

const azure = (endpoint: string) => ({ llm: { provider: "azure_openai", model: "safe-deployment", endpoint, api_key: "${KG_MODEL_KEY}" } });
const azureProfile = azure("https://models.invalid/v1");

describe("effective transport boundaries", () => {
  it.each([
    "https://models.invalid\\@collector.invalid/../v1",
    "https://models.invalid\\anything\\..\\@collector.invalid/../v1",
    "https://models.invalid/v1\n",
    "https://models.invalid/v1\t",
    "https://models.invalid/v1#fragment",
    "https://@models.invalid/v1",
    "https://user:password@models.invalid/v1",
  ])("rejects raw URL forms whose normalization can hide authority or routing: %j", (endpoint) => {
    expect(() => check(azure(endpoint), [azureProfile], { KG_MODEL_KEY: "synthetic" })).toThrow(/not approved/);
  });

  it("preserves native Ollama /v1 equivalence and exact generic-HTTP query endpoints", () => {
    const env = { KG_MODEL_KEY: "synthetic" };
    const profile = { llm: { provider: "ollama", endpoint: "https://models.invalid", api_key: "${KG_MODEL_KEY}" } };
    expect(() => check({ llm: { ...profile.llm, endpoint: "https://models.invalid/v1/" } }, [profile], env)).not.toThrow();
    const ocrProfile = { generic_http_ocr: { endpoint: "https://ocr.invalid/parse?backend=approved", api_key: "${KG_MODEL_KEY}" } };
    const cfg = { ocr: { provider: "generic_http" }, ...ocrProfile };
    expect(() => check(cfg, [ocrProfile], env)).not.toThrow();
    expect(() => check({ ...cfg, generic_http_ocr: { ...ocrProfile.generic_http_ocr, endpoint: "https://ocr.invalid/parse/?backend=approved" } }, [ocrProfile], env)).toThrow(/not approved/);
    expect(() => check({ ...cfg, generic_http_ocr: { ...ocrProfile.generic_http_ocr, endpoint: "https://ocr.invalid/parse?backend=caller" } }, [ocrProfile], env)).toThrow(/not approved/);
  });

  it.each(["llm", "embedding"])("keeps Azure %s deployment and version values inside their URL components", (section) => {
    const cfg = { [section]: { provider: "azure_openai", endpoint: "https://models.invalid/v1", model: "deployment-2.v1", api_version: "2025-01-01-preview", api_key: "${KG_MODEL_KEY}" } };
    const env = { KG_MODEL_KEY: "synthetic" };
    expect(() => check(cfg, [cfg], env)).not.toThrow();
    for (const [field, value] of [["model", "../../collector"], ["model", "other%2fdeployment"], ["model", "deployment?forward=caller"], ["api_version", "version&redirect=caller"]]) {
      const changed = { [section]: { ...cfg[section], [field!]: value } };
      expect(() => check(changed, [cfg], env)).toThrow(/single URL value/);
    }
  });

  it("honors alias embedding activation, while a conventional provider override wins", () => {
    const cfg = { embedding: { provider: "hash", endpoint: "https://collector.invalid/v1" } };
    const env = { KG_EMBED_PROVIDER: "openai_compatible", KG_EMBED_API_KEY: "synthetic" };
    expect(() => check(cfg, [], env)).toThrow(/not approved/);
    expect(() => check(cfg, [], { ...env, KG_EMBEDDING_PROVIDER: "hash" })).not.toThrow();
    expect(() => check({ ...cfg, embedding: { ...cfg.embedding, api_key: "${KG_EMBED_API_KEY}" } }, [], { ...env, KG_EMBEDDING_PROVIDER: "hash" })).toThrow(/not approved/);
  });

  it("honors OCR/file activation aliases and normalized Cortex backend selection", () => {
    const cfg = { ocr: { provider: "builtin_text" }, llm: { provider: "fake" }, embedding: { provider: "hash" }, files: { source: "local" }, snowflake: { host: "collector.invalid" } };
    const env = { SNOWFLAKE_PASSWORD: "synthetic" };
    for (const extra of [{ KG_FILE_SOURCE: "stage" }, { KG_AI_BACKEND: " CORTEX " }]) {
      expect(() => check(cfg, [], { ...env, ...extra })).toThrow(/not approved/);
    }
    expect(() => check(cfg, [], { ...env, KG_FILE_SOURCE: "stage", KG_FILES_SOURCE: "local" })).not.toThrow();
    expect(() => check(cfg, [], { ...env, KG_AI_BACKEND: "CORTEX", KG_LLM_PROVIDER: "fake", KG_EMBEDDING_PROVIDER: "hash", KG_OCR_PROVIDER: "builtin_text" })).not.toThrow();
  });

  it("checks fallback OCR credentials even when no generic-HTTP section was supplied", () => {
    const cfg = { ocr: { provider: "fallback", fallback_secondary_provider: "generic_http" } };
    // The entire generic-HTTP transport is supplied through conventional env names.
    expect(() => check(cfg, [], { KG_GENERIC_HTTP_OCR_ENDPOINT: "https://operator.invalid/parse", KG_GENERIC_HTTP_OCR_API_KEY: "synthetic" })).not.toThrow();
    const partial = { ...cfg, generic_http_ocr: { endpoint: "https://collector.invalid/parse" } };
    expect(() => check(partial, [], { KG_GENERIC_HTTP_OCR_API_KEY: "synthetic" })).toThrow(/not approved/);
  });

  it("uses inline JSON routing before aliases and does not interpolate inline values", () => {
    const cfg = { embedding: { provider: "hash", endpoint: "https://collector.invalid/v1" } };
    expect(() => check(cfg, [], { KG_CONFIG_JSON: JSON.stringify({ embedding: { provider: "openai_compatible" } }), KG_EMBED_API_KEY: "synthetic" })).toThrow(/not approved/);
    const complete = { KG_CONFIG_JSON: JSON.stringify({ embedding: { provider: "openai_compatible", endpoint: "https://operator.invalid/v1", api_key: "synthetic-inline" } }) };
    expect(() => check(cfg, [], complete)).not.toThrow();
    expect(() => check(cfg, [], { ...complete, KG_EMBEDDING_PROVIDER: "hash" })).not.toThrow();
    expect(() => check(cfg, [], { KG_CONFIG_JSON: "[]" })).toThrow(/configuration object/);
  });

  it("recognizes case-sensitive Python placeholder names without allowing unrelated credentials", () => {
    const cfg = config("https://models.invalid/v1", "KG_private");
    expect(() => check(cfg, [cfg], { KG_private: "synthetic" })).not.toThrow();
    expect(() => check({ llm: { ...cfg.llm, endpoint: "https://collector.invalid/v1" } }, [cfg], { KG_private: "synthetic" })).toThrow(/not approved/);
  });
});

describe("Snowflake credential sources", () => {
  const active = { writer: { provider: "snowflake_direct" } };
  const host = "approved.snowflakecomputing.com";
  const changedHost = "collector.invalid";

  it.each([
    { cache: { provider: "snowflake" } },
    { files: { source: "stage" } },
    { job: { use_file_queue: true } },
    { job: { use_lease: true } },
    { ocr: { provider: "fallback", fallback_secondary_provider: "snowflake_cortex" } },
  ])("binds ambient secrets for every active Snowflake consumer %j", (consumer) => {
    expect(() => check({ ...consumer, snowflake: { host: changedHost } }, [], { SNOWFLAKE_PASSWORD: "synthetic" })).toThrow(/not approved/);
  });

  it.each(["password", "oauth_token"])("checks env-overridden %s reference names", (field) => {
    const key = `KG_SNOWFLAKE_${field.toUpperCase()}_ENVIRONMENT_VARIABLE`;
    const env = { [key]: "KG_REFERENCED_SECRET", KG_REFERENCED_SECRET: "synthetic" };
    const cfg = { ...active, snowflake: { host, [`${field}_environment_variable`]: "KG_REFERENCED_SECRET", authenticator: "oauth" } };
    expect(() => check(cfg, [cfg], env)).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, host: changedHost } }, [cfg], env)).toThrow(/not approved/);
  });

  it.each(["oauth", "oauth_file", "azure_oauth"])("binds the implicit mounted token file for %s", (authenticator) => {
    const cfg = { ...active, snowflake: { host, authenticator } };
    expect(() => check(cfg, [cfg], {})).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, host: changedHost } }, [cfg], {})).toThrow(/not approved/);
  });

  it.each(["private_key_path", "oauth_token_path"])("binds literal %s paths as credentials", (field) => {
    const cfg = { ...active, snowflake: { host, authenticator: "oauth", [field]: "/operator/synthetic-key" } };
    expect(() => check(cfg, [cfg], {})).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, host: changedHost } }, [cfg], {})).toThrow(/not approved/);
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, [field]: "/caller/other-key" } }, [cfg], {})).toThrow(/not approved/);
  });

  it("follows ambient < YAML < conventional file-path precedence and explicit null", () => {
    const cfg = { ...active, snowflake: { host, authenticator: "oauth", oauth_token_path: "/profile/token" } };
    expect(() => check(cfg, [cfg], { SNOWFLAKE_OAUTH_TOKEN_PATH: "/ambient/token" })).not.toThrow();
    expect(() => check(cfg, [cfg], { KG_SNOWFLAKE_OAUTH_TOKEN_PATH: "/override/token" })).toThrow(/not approved/);
    expect(() => check(cfg, [cfg], { KG_SNOWFLAKE_HOST: host, KG_SNOWFLAKE_OAUTH_TOKEN_PATH: "/override/token" })).not.toThrow();
    // An explicit null prevents the default/ambient token file from being read.
    expect(() => check({ ...active, snowflake: { host: changedHost, authenticator: "oauth", oauth_token_path: null } }, [], { SNOWFLAKE_OAUTH_TOKEN_PATH: "/ambient/token" })).not.toThrow();
  });

  it("does not bind an unused token file when an explicit OAuth token wins", () => {
    const cfg = { ...active, snowflake: { host, authenticator: "oauth", oauth_token: "${KG_TOKEN}" } };
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, oauth_token_path: "/unused/file" } }, [cfg], { KG_TOKEN: "synthetic" })).not.toThrow();
  });

  it("permits a complete environment host/file pair without approving caller hosts", () => {
    expect(() => check({ ...active, snowflake: { host: changedHost, authenticator: "oauth" } }, [], {
      SNOWFLAKE_HOST: host, SNOWFLAKE_AUTH: "oauth", SNOWFLAKE_OAUTH_TOKEN_PATH: "/operator/token",
    })).toThrow(/not approved/);
    expect(() => check({ ...active, snowflake: { host: changedHost, authenticator: "oauth" } }, [], {
      KG_SNOWFLAKE_HOST: host, KG_SNOWFLAKE_AUTHENTICATOR: "oauth", KG_SNOWFLAKE_OAUTH_TOKEN_PATH: "/operator/token",
    })).not.toThrow();
  });

  it("matches the connector's China-region account host construction", () => {
    const profile = { snowflake: { account: "account.cn-north-1", password: "${KG_PASSWORD}" } };
    const cfg = { ...active, snowflake: { ...profile.snowflake, host: "account.cn-north-1.snowflakecomputing.cn" } };
    expect(() => check(cfg, [profile], { KG_PASSWORD: "synthetic" })).not.toThrow();
    expect(() => check({ ...cfg, snowflake: { ...cfg.snowflake, host: "account.cn-north-1.snowflakecomputing.com" } }, [profile], { KG_PASSWORD: "synthetic" })).toThrow(/not approved/);
  });
});


afterEach(() => {
  vi.unstubAllEnvs();
  resetAppEnv();
});

function submission(): IngestionRequest {
  const selected = { model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} };
  return {
    runtime: "local", jobId: "test-job", graphId: "test-graph", graphName: null,
    sourceKind: "local_path", source: { path: "/synthetic/corpus" },
    ocr: { ...selected, provider: "builtin_text" },
    llm: { ...selected, provider: "azure_openai", endpoint: "https://models.invalid/v1",
      model: "safe-deployment", apiKeyEnvironmentVariable: "KG_MODEL_KEY" },
    embedding: { ...selected, provider: "hash", model: "hash", dimension: 16 },
    output: { kind: "local_files", workspacePath: "/synthetic/output", snowflake: null },
    baseConfigPath: null, includeGlobs: ["**/*"], cacheProvider: "none", providerParallelism: 1,
    runtimeOptions: {}, ontology: null, revision: null, priority: "bulk",
  };
}

it("binds real generated run configs to operator profiles, never to caller-generated bases", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "credential-routing-"));
  try {
    await mkdir(path.join(root, "configs"));
    await writeFile(path.join(root, "configs", "provider.yaml"), [
      "llm:", "  provider: azure_openai", "  endpoint: https://models.invalid/v1", "  api_key: ${KG_MODEL_KEY}",
    ].join("\n"));
    // Do not inherit routing or credential state from the machine running tests.
    for (const name of Object.keys(process.env)) {
      if (name.startsWith("KG_") || name.startsWith("SNOWFLAKE_")) vi.stubEnv(name, "");
    }
    vi.stubEnv("FLAKEGRAPH_REPOSITORY_ROOT", root);
    vi.stubEnv("KG_MODEL_KEY", "synthetic-model-key");
    vi.stubEnv("KG_INFRA_SECRET", "synthetic-infrastructure-key");
    resetAppEnv();
    const request = submission();
    await expect(buildRunConfig(request)).resolves.toMatchObject({
      llm: { endpoint: "https://models.invalid/v1", api_key: "${KG_MODEL_KEY}" },
    });
    for (const endpoint of ["https://collector.invalid/v1", "https://models.invalid\\@collector.invalid/../v1"]) {
      await expect(buildRunConfig({ ...request, llm: { ...request.llm, endpoint } })).rejects.toThrow(/not approved/);
    }
    await expect(buildRunConfig({ ...request, llm: { ...request.llm, apiKeyEnvironmentVariable: "KG_INFRA_SECRET" } })).rejects.toThrow(/not approved/);
    const generated = path.join(root, "generated-run.yaml");
    await writeFile(generated, "llm:\n  provider: azure_openai\n  endpoint: https://collector.invalid/v1\n  api_key: ${KG_INFRA_SECRET}\n");
    await expect(buildRunConfig({ ...request, baseConfigPath: generated,
      llm: { ...request.llm, endpoint: "https://collector.invalid/v1", apiKeyEnvironmentVariable: "KG_INFRA_SECRET" },
    })).rejects.toThrow(/not approved/);
    await expect(buildRunConfig({ ...request, source: { path: "${KG_INFRA_SECRET}" } })).rejects.toThrow(/operator profiles/);

    // Fixed operator aliases override the request before the Python adapter is
    // built, so a complete endpoint/key pair remains usable without a profile.
    vi.stubEnv("KG_LLM_ENDPOINT", "https://operator.invalid/v1");
    vi.stubEnv("KG_LLM_API_KEY", "synthetic-fixed-key");
    await expect(buildRunConfig({ ...request,
      llm: { ...request.llm, endpoint: "https://caller.invalid/v1", apiKeyEnvironmentVariable: "KG_INFRA_SECRET" },
    })).resolves.toHaveProperty("llm.provider", "azure_openai");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});


it("does not turn a bare conventional key into authority for a different declared credential", () => {
  const operator = { llm: { provider: "openai_compatible", endpoint: "https://example.invalid/v1", api_key: "${KG_EXAMPLE_KEY}" } };
  const cfg = { llm: { provider: "openai_compatible", endpoint: "https://example.invalid/v1" } };
  const env = { KG_EXAMPLE_KEY: "synthetic-example", KG_LLM_API_KEY: "synthetic-sensitive" };
  expect(() => check(cfg, [operator], env)).toThrow(/not approved/);
  expect(() => check(cfg, [operator], { ...env, KG_LLM_ENDPOINT: "https://example.invalid/v1" })).not.toThrow();
  expect(() => check(cfg, [{ llm: { ...operator.llm, api_key: "${KG_LLM_API_KEY}" } }], env)).not.toThrow();
});

it.each(["/temporary/../v1", "/%2e/v1", "/temporary/%2E%2e/v1", "/temporary%2fv1", "/temporary%5cv1"])("rejects ambiguous normalized URL paths %s", (pathname) => {
  expect(() => check(azure(`https://models.invalid${pathname}`), [azureProfile], { KG_MODEL_KEY: "synthetic" })).toThrow(/not approved/);
});


it("does not authorize an unused default OAuth file through a profile that declares an explicit token", () => {
  const operator = { snowflake: { host: "approved.invalid", authenticator: "oauth", oauth_token: "${KG_TOKEN}" } };
  const cfg = { writer: { provider: "snowflake_direct" }, snowflake: { host: "approved.invalid", authenticator: "oauth" } };
  expect(() => check(cfg, [operator], { KG_TOKEN: "synthetic" })).toThrow(/not approved/);
});


it("does not route an ambient OAuth token when the connector uses password authentication", () => {
  const operator = { snowflake: { host: "approved.invalid", authenticator: "snowflake", password: "${KG_PASSWORD}" } };
  const cfg = { writer: { provider: "snowflake_direct" }, ...operator };
  expect(() => check(cfg, [operator], { KG_PASSWORD: "synthetic-password", SNOWFLAKE_OAUTH_TOKEN: "unused-synthetic-token" })).not.toThrow();
  expect(() => check({ ...cfg, snowflake: { ...operator.snowflake, authenticator: "oauth" } }, [operator], { KG_PASSWORD: "synthetic-password", SNOWFLAKE_OAUTH_TOKEN: "used-synthetic-token" })).toThrow(/not approved/);
});

describe("Azure source destinations", () => {
  const source = (accountUrl = "https://approved.blob.core.windows.net") => ({ files: { source: "azure_blob" }, azure_blob: { account_url: accountUrl, container: "fixtures" } });
  const profiles = [source()];
  it("binds default identity and ambient SAS tokens to operator endpoints", () => {
    for (const env of [{}, { KG_AZURE_BLOB_SAS_TOKEN: "synthetic" }]) {
      expect(() => check(source(), profiles, env)).not.toThrow();
      expect(() => check(source("https://unapproved.example.test"), profiles, env)).toThrow(/not an approved source destination/);
    }
  });
  it("uses effective environment and inline configuration precedence", () => {
    expect(() => check(source("https://caller.example.test"), [], { KG_AZURE_BLOB_ACCOUNT_URL: "https://operator.example.test" })).not.toThrow();
    expect(() => check(source("https://caller.example.test"), [], { KG_CONFIG_JSON: JSON.stringify({ azure_blob: { account_url: "http://emulator.example.test:10000/account" } }) })).not.toThrow();
    expect(() => check(source("https://caller.example.test"), [], { KG_AZURE_BLOB_CONNECTION_STRING: "UseDevelopmentStorage=true" })).not.toThrow();
    expect(() => check(source("http://emulator.example.test:10000/account"), [source("http://emulator.example.test:10000/account")], {})).not.toThrow();
    expect(() => check(source(), [], { KG_AZURE_BLOB_SAS_TOKEN: "synthetic" })).toThrow(/not an approved source destination/);
  });
});
