import { mkdtemp, mkdir, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Effect } from "effect";
import { resetAppEnv } from "./env";
import { LocalRuntime } from "./runtime/local";
import { listRemoteObjects } from "./sources";
import { SnowflakeStubRuntime } from "./runtime/snowflake-stub";
import { controlledPlane } from "./access/controlled-plane";
import { KubernetesRuntime, submitArguments } from "./runtime/kubernetes";
import type { IngestionRequest } from "./protocol/schema";

function request(stateRoot: string, overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  return {
    runtime: "local",
    jobId: "job_demo",
    graphId: "graph_demo",
    graphName: "Demo graph",
    sourceKind: "local_path",
    source: { path: path.join(stateRoot, "corpus") },
    ocr: { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    llm: { provider: "vllm_local", model: "qwen", endpoint: "http://localhost:8000/v1", apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    embedding: {
      provider: "sentence_transformers",
      model: "mini",
      endpoint: null,
      apiKeyEnvironmentVariable: null,
      dimension: 384,
      options: {},
    },
    output: { kind: "local_files", workspacePath: path.join(stateRoot, "graphs", "graph_demo"), snowflake: null },
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

describe("local runtime", () => {
  afterEach(() => {
    resetAppEnv();
    delete process.env.FLAKEGRAPH_CLI;
    delete process.env.FLAKEGRAPH_APP_STATE_ROOT;
  });

  it("lists local files, preflights, submits, and loads a completed graph", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-"));
    const corpus = path.join(stateRoot, "corpus");
    await mkdir(corpus, { recursive: true });
    await writeFile(path.join(corpus, "note.md"), "Judo was developed by Jigoro Kano.\n");
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const runtime = new LocalRuntime(path.resolve(import.meta.dirname, "../../.."), stateRoot);
    const objects = await Effect.runPromise(runtime.listSourceObjects({ kind: "local", path: corpus }));
    expect(objects.some((object) => object.name === "note.md")).toBe(true);
    const ingestion = request(stateRoot);
    const preflight = await Effect.runPromise(runtime.preflight(ingestion));
    expect(preflight.ok).toBe(true);
    const submitted = await Effect.runPromise(runtime.submit(ingestion));
    expect(submitted.status).toBe("running");
    const terminal = await waitFor(async () => {
      const snapshot = await Effect.runPromise(runtime.getRun(ingestion.jobId));
      if (snapshot.status === "running") {
        throw new Error("still running");
      }
      return snapshot;
    });
    expect(terminal.status).toBe("succeeded");
    const graph = await Effect.runPromise(runtime.loadRunGraph(terminal));
    expect(graph.nodes.length).toBeGreaterThan(0);
    const renamed = await Effect.runPromise(runtime.renameGraph(ingestion.graphId, "Renamed demo"));
    expect(renamed).toBe("Renamed demo");
  }, 20_000);

  it("lists an object-storage source through the pipeline's own listing", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-s3-"));
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const runtime = new LocalRuntime(path.resolve(import.meta.dirname, "../../.."), stateRoot);
    const objects = await Effect.runPromise(
      runtime.listSourceObjects({
        kind: "s3",
        bucket: "test-corpora",
        prefix: "martial_arts/",
        endpointUrl: "http://minio.local:9000",
        region: "us-east-1",
      }),
    );
    expect(objects.map((object) => object.uri)).toEqual([
      "s3://test-corpora/martial_arts/judo.md",
      "s3://test-corpora/martial_arts/karate.pdf",
    ]);
    expect(objects[0]).toMatchObject({ name: "judo.md", sizeBytes: 512, checksum: "etag-judo.md" });
    expect(objects[0]?.modifiedAt).toBe("2026-09-18T10:00:00+00:00");
    const limited = await Effect.runPromise(runtime.listSourceObjects({ kind: "s3", bucket: "test-corpora" }, 1));
    expect(limited).toHaveLength(1);
    const failure = await Effect.runPromise(runtime.listSourceObjects({ kind: "s3", bucket: "missing" }))
      .then(() => "unexpected success", (error: unknown) => String(error));
    expect(failure).toContain("Listing the s3 source failed");
    expect(failure).not.toContain("NoSuchBucket");
    // The listing's config is a scratch file the browser cleans up after itself.
    expect(await readdir(path.join(stateRoot, "browse"))).toEqual([]);
    await expect(
      Effect.runPromise(runtime.listSourceObjects({ kind: "azure_blob", container: "", prefix: "" })),
    ).rejects.toThrow(/not an approved source destination/);
    await expect(
      Effect.runPromise(runtime.listSourceObjects({ kind: "snowflake_stage", stage: "@docs" })),
    ).rejects.toThrow(/not available for snowflake_stage/);
  });

  it("gives up on a listing that never answers and says so", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-slow-"));
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const started = Date.now();
    await expect(
      listRemoteObjects("s3", { kind: "s3", bucket: "slow" }, { cwd: process.cwd(), stateRoot, timeoutMs: 1_500 }),
    ).rejects.toThrow(/did not answer within 2 s/);
    expect(Date.now() - started).toBeLessThan(10_000);
    expect(await readdir(path.join(stateRoot, "browse"))).toEqual([]);
  });

  it("fails preflight for a missing corpus marker", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-fail-"));
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const runtime = new LocalRuntime(path.resolve(import.meta.dirname, "../../.."), stateRoot);
    const result = await Effect.runPromise(
      runtime.preflight(request(stateRoot, { source: { path: "/tmp/missing-corpus" }, jobId: "job_fail" })),
    );
    expect(result.ok).toBe(false);
  });
});

describe("snowflake runtime behind graph access", () => {
  const as = (stateRoot: string, userName: string) =>
    controlledPlane(
      new SnowflakeStubRuntime(process.cwd(), stateRoot, true, { userName, email: "", roles: [] }),
      { userName, email: "", roles: [] },
      stateRoot,
      { strict: false, laptop: true },
    );

  it("shows a graph to its owner and to people it is shared with, at their level", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-"));
    const ingestion = request(stateRoot, {
      runtime: "snowflake",
      jobId: "job_snow",
      graphId: "graph_snow",
      output: {
        kind: "snowflake",
        workspacePath: path.join(stateRoot, "graphs", "graph_snow"),
        snowflake: {
          account: "xy123",
          user: "ALICE",
          database: "FG",
          schema: "PUBLIC",
          warehouse: "COMPUTE",
          bulkStage: "@stage",
          role: null,
          host: null,
          authenticator: null,
          credentialEnvironmentVariable: null,
          credentialField: null,
        },
      },
    });
    await mkdir(path.join(stateRoot, "graphs", "graph_snow"), { recursive: true });
    const submitted = await Effect.runPromise(as(stateRoot, "ALICE").submit(ingestion));
    expect(submitted.access).toBe("owner");
    expect(submitted.owner).toBe("ALICE");

    expect(await Effect.runPromise(as(stateRoot, "BOB").listRuns())).toHaveLength(0);
    await Effect.runPromise(as(stateRoot, "ALICE").shareGraph("graph_snow", "CAROL", "read"));
    const carol = as(stateRoot, "CAROL");
    const visible = await Effect.runPromise(carol.listRuns());
    expect(visible.map((run) => [run.graphId, run.access])).toEqual([["graph_snow", "read"]]);
    await expect(Effect.runPromise(carol.renameGraph("graph_snow", "Hijack"))).rejects.toThrow(/write access/i);

    await Effect.runPromise(as(stateRoot, "ALICE").shareGraph("graph_snow", "CAROL", "write"));
    expect(await Effect.runPromise(as(stateRoot, "CAROL").renameGraph("graph_snow", "Renamed"))).toBe("Renamed");
    await expect(Effect.runPromise(as(stateRoot, "CAROL").deleteGraph("graph_snow"))).rejects.toThrow(/owner/i);
    // Without a sign-in gate, a session that assumed nobody is the laptop's user.
    expect((await Effect.runPromise(as(stateRoot, "").listRuns())).map((run) => run.access)).toEqual(["owner"]);
  });
});

describe("kubernetes runtime", () => {
  it("reads a stub fleet snapshot and submits to it", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-k8s-"));
    const runtime = new KubernetesRuntime(process.cwd(), stateRoot, true);
    const cluster = await Effect.runPromise(runtime.cluster("flakegraph"));
    expect(cluster === null || Array.isArray(cluster.nodes)).toBe(true);
    const submitted = await Effect.runPromise(
      runtime.submit(request(stateRoot, { runtime: "kubernetes", jobId: "job_k8s", graphId: "graph_k8s", graphName: "Fleet job" })),
    );
    expect(submitted.status).toBe("queued");
    const listed = await Effect.runPromise(runtime.listRuns());
    expect(listed.some((run) => run.runId === "job_k8s")).toBe(true);
    const cancelled = await Effect.runPromise(runtime.cancel("job_k8s"));
    expect(cancelled.status).toBe("cancelled");
    const retried = await Effect.runPromise(runtime.retry("job_k8s"));
    expect(retried.status).toBe("queued");
  });

  it("records the identified viewer as the owner of what they submit", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-k8s-owner-"));
    // Behind a gate the viewer comes with the request, not from the
    // workspace's assumed identity, and Mine is built on that owner.
    const gated = new KubernetesRuntime(process.cwd(), stateRoot, true, {
      userName: "alice@example.test",
      email: "alice@example.test",
      roles: [],
    });
    const submitted = await Effect.runPromise(
      gated.submit(request(stateRoot, { runtime: "kubernetes", jobId: "job_owned", graphId: "graph_owned" })),
    );
    expect(submitted.raw.owner).toBe("ALICE@EXAMPLE.TEST");
    // With nobody identified, the laptop's assumed identity (none here) is the owner.
    const anonymous = new KubernetesRuntime(process.cwd(), stateRoot, true);
    const unowned = await Effect.runPromise(
      anonymous.submit(request(stateRoot, { runtime: "kubernetes", jobId: "job_unowned", graphId: "graph_unowned" })),
    );
    expect(unowned.raw.owner).toBeNull();
  });

  it("submits a run in the band the person chose, bulk unless they chose otherwise", () => {
    const stateRoot = "/state";
    const fleet = { runtime: "kubernetes" as const, jobId: "job_band" };
    expect(submitArguments(request(stateRoot, fleet), "/state/config.yaml")).toEqual([
      "distributed",
      "submit",
      "--config",
      "/state/config.yaml",
      "--run-id",
      "job_band",
      "--priority",
      "bulk",
    ]);
    const waiting = submitArguments(request(stateRoot, { ...fleet, priority: "interactive" }), "/c.yaml");
    expect(waiting.slice(-2)).toEqual(["--priority", "interactive"]);
    // A revision carries the band too: the work it adds is claimed in it.
    const revision = submitArguments(
      request(stateRoot, {
        ...fleet,
        priority: "interactive",
        revision: { baseRunId: "run_base", dropFileIds: ["doc-1"], addDocuments: false },
      }),
      "/c.yaml",
    );
    expect(revision).toEqual([
      "distributed",
      "revise",
      "--config",
      "/c.yaml",
      "--run-id",
      "job_band",
      "--priority",
      "interactive",
      "--base-run",
      "run_base",
      "--drop-file",
      "doc-1",
      "--no-add-documents",
    ]);
    // Untying documents from their folder changes no document: the kept
    // ones are finalized again from their stored extraction.
    const untied = submitArguments(
      request(stateRoot, {
        ...fleet,
        revision: { baseRunId: "run_base", dropFileIds: [], addDocuments: false, folderExclusions: ["doc-2"] },
      }),
      "/c.yaml",
    );
    expect(untied.slice(-4)).toEqual(["--base-run", "run_base", "--no-add-documents", "--rebuild"]);
  });
});

async function waitFor<T>(fn: () => Promise<T>, attempts = 40): Promise<T> {
  let last: unknown;
  for (let index = 0; index < attempts; index += 1) {
    try {
      return await fn();
    } catch (error) {
      last = error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw last;
}
