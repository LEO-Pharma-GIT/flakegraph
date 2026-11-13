import { mkdtemp, mkdir, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assertCredentialReference, confineIngestionRequest, confineSource, confinedPath, isSafeId, isWithin, safeUploadPath } from "./confine";
import { cliEnvironment } from "./cli";
import { environmentForRequest } from "./config";
import { appEnv, resetAppEnv } from "./env";
import type { IngestionRequest } from "./protocol/schema";

const original = { ...process.env };
let stateRoot = "";

beforeEach(async () => {
  stateRoot = await mkdtemp(path.join(tmpdir(), "flakegraph-confine-"));
  process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
  delete process.env.FLAKEGRAPH_APP_SOURCE_ROOTS;
  resetAppEnv();
});

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

function request(overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  return {
    runtime: "local",
    jobId: "job_1",
    graphId: "graph_1",
    graphName: null,
    sourceKind: "local_path",
    source: { kind: "local", path: "data/martial_arts/files" },
    ocr: { provider: "builtin_text", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    llm: { provider: "ollama", model: "qwen3:4b", endpoint: "http://localhost:11434", apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    embedding: { provider: "sentence_transformers", model: "mini", endpoint: null, apiKeyEnvironmentVariable: null, dimension: 384, options: {} },
    output: { kind: "local_files", workspacePath: "", snowflake: null },
    baseConfigPath: null,
    includeGlobs: ["**/*"],
    cacheProvider: "local",
    providerParallelism: 4,
    runtimeOptions: {},
    ontology: null,
    revision: null,
    priority: "bulk",
    ...overrides,
  };
}

describe("identifiers", () => {
  it("admits directory-safe ids and nothing that names another directory", () => {
    expect(isSafeId("run_martial_arts")).toBe(true);
    expect(isSafeId("a.b-c_D9")).toBe(true);
    for (const bad of ["", "..", ".", "../x", "a/b", "a\\b", "a b", "run\0"]) {
      expect(isSafeId(bad), bad).toBe(false);
    }
  });
});

describe("paths", () => {
  it("confines what a request names to the roots this host opened", () => {
    expect(isWithin("/srv/state", "/srv/state/runs/x")).toBe(true);
    expect(isWithin("/srv/state", "/srv/state")).toBe(true);
    expect(isWithin("/srv/state", "/srv/state2/runs")).toBe(false);
    expect(isWithin("/srv/state", "/srv/state/../etc")).toBe(false);
    const options = { roots: [stateRoot], base: stateRoot, label: "location" };
    expect(confinedPath("graphs/g1", options)).toBe(path.join(stateRoot, "graphs", "g1"));
    expect(() => confinedPath("../../etc/passwd", options)).toThrow(/location must be inside/);
    expect(() => confinedPath("/etc/passwd", options)).toThrow(/location must be inside/);
    expect(() => confinedPath("  ", options)).toThrow(/empty/);
  });

  it("fills in the host's own defaults where the form left a path blank", () => {
    const confined = confineIngestionRequest(request());
    expect(confined.baseConfigPath).toBe(path.join(appEnv().repositoryRoot, "configs", "app-defaults.yaml"));
    expect(confined.output.workspacePath).toBe(path.join(stateRoot, "graphs", "graph_1"));
    expect(confined.source.path).toBe(path.join(appEnv().repositoryRoot, "data", "martial_arts", "files"));
  });

  it("refuses a request that names a path or an id outside the console's reach", () => {
    expect(() => confineIngestionRequest(request({ jobId: "../escape" }))).toThrow(/jobId/);
    expect(() => confineIngestionRequest(request({ graphId: "a/b" }))).toThrow(/graphId/);
    expect(() => confineIngestionRequest(request({ baseConfigPath: "/etc/passwd" }))).toThrow(/baseConfigPath/);
    expect(() =>
      confineIngestionRequest(request({ output: { kind: "local_files", workspacePath: "/tmp/elsewhere", snowflake: null } })),
    ).toThrow(/workspacePath/);
    // Nor into another graph's directory under the state root.
    expect(() =>
      confineIngestionRequest(
        request({ output: { kind: "local_files", workspacePath: path.join(stateRoot, "graphs", "graph_other"), snowflake: null } }),
      ),
    ).toThrow(/workspacePath/);
    expect(
      confineIngestionRequest(
        request({ output: { kind: "local_files", workspacePath: path.join(stateRoot, "graphs", "graph_1", "v2"), snowflake: null } }),
      ).output.workspacePath,
    ).toBe(path.join(stateRoot, "graphs", "graph_1", "v2"));
    expect(() => confineIngestionRequest(request({ source: { kind: "local", path: "/Users/someone/Documents" } }))).toThrow(
      /source.path/,
    );
    expect(() => confineSource({ kind: "local", path: "/" })).toThrow(/source.path/);
    // A bucket names no path on this host and passes through as it is.
    expect(confineSource({ kind: "s3", bucket: "corpus", prefix: "a/" })).toEqual({ kind: "s3", bucket: "corpus", prefix: "a/" });
  });

  it("leaves the source alone for a revision that adds no documents", () => {
    const dropsOnly = request({
      sourceKind: "upload",
      source: {},
      revision: { baseRunId: "run_base", dropFileIds: ["f1"], addDocuments: false },
    });
    expect(confineIngestionRequest(dropsOnly).source).toEqual({});
    expect(() =>
      confineIngestionRequest(request({ ...dropsOnly, revision: { ...dropsOnly.revision!, addDocuments: true } })),
    ).toThrow(/source.path is empty/);
    expect(() => confineIngestionRequest(request({ ...dropsOnly, revision: { ...dropsOnly.revision!, baseRunId: "../x" } }))).toThrow(
      /baseRunId/,
    );
  });

  it("opens the folders FLAKEGRAPH_APP_SOURCE_ROOTS names", async () => {
    const documents = await mkdtemp(path.join(tmpdir(), "flakegraph-documents-"));
    process.env.FLAKEGRAPH_APP_SOURCE_ROOTS = documents;
    resetAppEnv();
    expect(confineSource({ kind: "local", path: path.join(documents, "papers") }).path).toBe(path.join(documents, "papers"));
  });
});

describe("credentials", () => {
  it("lets a request name only KG_ variables, and a spawn only ones that are set", () => {
    expect(() => assertCredentialReference("KG_LLM_API_KEY")).not.toThrow();
    expect(() => assertCredentialReference(null)).not.toThrow();
    expect(() => assertCredentialReference("OPENAI_API_KEY")).toThrow(/must be named KG_/);
    expect(() => assertCredentialReference("FLAKEGRAPH_ASK_API_KEY")).toThrow(/must be named KG_/);
    expect(() => assertCredentialReference("KG_$(id)")).toThrow(/must be named/);
    const named = request({
      llm: { provider: "openai_compatible", model: "m", endpoint: "http://x", apiKeyEnvironmentVariable: "KG_MISSING", dimension: null, options: {} },
    });
    expect(() => environmentForRequest(named, { PATH: "/usr/bin" })).toThrow(/not set: KG_MISSING/);
    expect(environmentForRequest(named, { PATH: "/usr/bin", KG_MISSING: "set" }).KG_MISSING).toBe("set");
  });

  it("hands the pipeline an allow-listed environment, never the console's own", () => {
    const environment = cliEnvironment({
      PATH: "/usr/bin",
      HOME: "/home/fg",
      KG_LLM_API_KEY: "x",
      HF_HOME: "/cache",
      FLAKEGRAPH_ASK_API_KEY: "billed",
      FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "true",
      DATABASE_URL: "postgres://coordination",
      AWS_SECRET_ACCESS_KEY: "s3",
      RANDOM_TOKEN: "no",
    });
    expect(Object.keys(environment).sort()).toEqual(["AWS_SECRET_ACCESS_KEY", "DATABASE_URL", "HF_HOME", "HOME", "KG_LLM_API_KEY", "PATH"]);
  });
});

describe("safeUploadPath", () => {
  it("keeps a dropped folder's subfolders", () => {
    expect(safeUploadPath("contracts/2024/a.pdf")).toBe("contracts/2024/a.pdf");
    expect(safeUploadPath("contracts\\2024\\a.pdf")).toBe("contracts/2024/a.pdf");
  });

  it("can name nothing outside the upload folder", () => {
    expect(safeUploadPath("../../etc/passwd")).toBe("etc/passwd");
    expect(safeUploadPath("/abs/./x.pdf")).toBe("abs/x.pdf");
    expect(safeUploadPath("..")).toMatch(/^file-/);
    expect(safeUploadPath("a\0b.pdf")).toBe("a_b.pdf");
  });
});

describe("ingestion boundaries", () => {
  it("uses the selected backend kind and rejects escaping glob patterns", () => {
    const confined = confineIngestionRequest(request({ source: { kind: "s3", path: "data/martial_arts/files" } }));
    expect(confined.source.kind).toBe("local");
    for (const pattern of ["../*.txt", "/tmp/*.txt", "C:/data/*.txt", "..\\*.txt"]) {
      expect(() => confineIngestionRequest(request({ includeGlobs: [pattern] }))).toThrow(/includeGlobs/);
    }
  });

  it("rejects uploaded base configurations and source-root symlink escapes", async () => {
    expect(() => confineIngestionRequest(request({ baseConfigPath: path.join(stateRoot, "uploads", "job", "base.yaml") }))).toThrow(/baseConfigPath/);
    const outside = await mkdtemp(path.join(tmpdir(), "fg-outside-"));
    const corpus = path.join(stateRoot, "corpus");
    await mkdir(corpus);
    await symlink(outside, path.join(corpus, "outside"));
    expect(() => confineSource({ kind: "local", path: path.join(corpus, "outside") })).toThrow(/source.path/);
  });
});
