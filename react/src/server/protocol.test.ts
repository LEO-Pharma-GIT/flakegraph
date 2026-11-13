import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { decodeSchema, IngestionRequest } from "./protocol/schema";
import { redactedConfig, sensitiveConfigKey } from "./config";
import { accumulateSummary, makeProgressRecord, progressEvent, readLocalProgress } from "./progress";
import { communityMemberIds, filterGraph, graphFacets } from "./graph-filter";
import { goldToDataset, compareToGold } from "./gold";
import { viewerFromHeaders } from "./identity";
import { writeRunRecord, renameGraph, listRunRecords, withdrawRun } from "./catalog";
import { capabilitiesFor, validateGraphName } from "./protocol/schema";

describe("protocol schemas", () => {
  it("decodes an ingestion request", () => {
    const encoded = {
      runtime: "local",
      jobId: "job_1",
      graphId: "graph_1",
      graphName: "Demo",
      sourceKind: "local_path",
      source: { path: "data/martial_arts/files" },
      ocr: { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null },
      llm: { provider: "vllm_local", model: "qwen", endpoint: "http://localhost:8000/v1", apiKeyEnvironmentVariable: null, dimension: null },
      embedding: { provider: "sentence_transformers", model: "mini", endpoint: null, apiKeyEnvironmentVariable: null, dimension: 384 },
      output: { kind: "local_files", workspacePath: "/tmp/out", snowflake: null },
      baseConfigPath: null,
    };
    const request = decodeSchema(IngestionRequest, encoded);
    expect(request.providerParallelism).toBe(12);
    expect(request.includeGlobs).toEqual(["**/*"]);
    // A caller that names no band submits bulk work; only the two bands exist.
    expect(request.priority).toBe("bulk");
    expect(decodeSchema(IngestionRequest, { ...encoded, priority: "interactive" }).priority).toBe("interactive");
    expect(() => decodeSchema(IngestionRequest, { ...encoded, priority: "urgent" })).toThrow();
  });

  it("rejects an empty graph name", () => {
    expect(() => validateGraphName("   ")).toThrow(/empty/);
  });

  it("advertises runtime capabilities", () => {
    expect(capabilitiesFor("local")).toContain("upload");
    expect(capabilitiesFor("local")).toContain("cancel");
    expect(capabilitiesFor("kubernetes")).toContain("cluster");
    expect(capabilitiesFor("snowflake")).toContain("share");
    expect(capabilitiesFor("local")).not.toContain("share");
  });
});

describe("configuration redaction", () => {
  it("redacts secrets and credential URLs", () => {
    const redacted = redactedConfig({
      llm: { api_key: "sk-live", endpoint: "https://user:pass@example.com/v1?api_key=secret" },
      token_environment_variable: "LLM_API_KEY",
    });
    const llm = redacted.llm as Record<string, string>;
    expect(llm.api_key).toBe("${REDACTED}");
    expect(llm.endpoint).toContain("***");
    expect(redacted.token_environment_variable).toBe("LLM_API_KEY");
    expect(sensitiveConfigKey("api_key")).toBe(true);
    expect(sensitiveConfigKey("api_key_environment_variable")).toBe(false);
    // Wider than the obvious names: kubeconfig material, cloud tenancy
    // identifiers and anything that calls itself a credential or a
    // certificate, in either spelling. A name that merely refers to a secret
    // is masked too, because the preview cannot tell a reference from a value.
    for (const key of ["client-key-data", "credentials", "cert", "client_id", "tenant_id", "secret_name", "token_header", "key", "ca"]) {
      expect(sensitiveConfigKey(key), key).toBe(true);
    }
    for (const key of ["endpoint", "model", "bulk_stage", "password_environment_variable", "model_cache_dir"]) {
      expect(sensitiveConfigKey(key), key).toBe(false);
    }
  });
});

describe("progress aggregation", () => {
  it("counts distinct completed files and ignores retries", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fg-progress-"));
    const file = path.join(directory, "events.jsonl");
    const lines = [
      makeProgressRecord({ timestamp: "t1", stage: "ocr", status: "completed", fileId: "a", message: null, elapsedMs: 10, counts: { files_total: 2 } }),
      makeProgressRecord({ timestamp: "t2", stage: "ocr", status: "completed", fileId: "a", message: null, elapsedMs: 12, counts: { files_total: 2 } }),
      makeProgressRecord({ timestamp: "t3", stage: "ocr", status: "completed", fileId: "b", message: null, elapsedMs: 8, counts: { files_total: 2 } }),
    ];
    await writeFile(file, lines.map((line) => JSON.stringify(line)).join("\n") + "\n");
    const progress = await readLocalProgress(file);
    const ocr = progress.stages.find((stage) => stage.stage === "ocr");
    expect(ocr?.completed).toBe(2);
    expect(progress.documentsCompleted).toBe(2);
  });

  it("parses core progress records", () => {
    const event = progressEvent({ event: "kg_processor.progress", stage: "ocr", status: "running", file_id: "doc" });
    expect(event.fileId).toBe("doc");
    const summary: Record<string, unknown> = {};
    accumulateSummary(summary, { event: "kg_processor.progress", stage: "file_source", status: "completed", counts: { files_seen: 4 } });
    expect(summary.documents_total).toBe(4);
  });
});

describe("graph filters", () => {
  const dataset = goldToDataset(
    {
      entities: [
        { id: "judo", name: "Judo", type: "ART", aliases: ["Kodokan judo"] },
        { id: "kano", name: "Jigoro Kano", type: "PERSON" },
        { id: "tokyo", name: "Tokyo", type: "PLACE" },
      ],
      relations: [
        { id: "r1", source: "judo", target: "kano", relation_type: "DEVELOPED_BY", observations: [{ document: "d1", sentence: "Kano developed judo" }] },
        { id: "r2", source: "judo", target: "tokyo", relation_type: "PRACTICED_IN", required: false },
      ],
    },
    "g1",
  );

  it("filters by search, type, relation, and confidence", () => {
    expect(graphFacets(dataset).nodeTypes).toContain("PERSON");
    const people = filterGraph(dataset, { nodeTypes: ["PERSON"] });
    expect(people.nodes.map((node) => node.id)).toEqual(["kano"]);
    const developed = filterGraph(dataset, { relationTypes: ["DEVELOPED_BY"], includeIsolates: false });
    expect(developed.edges).toHaveLength(1);
    const strict = filterGraph(dataset, { minimumConfidence: 0.9 });
    expect(strict.edges.every((edge) => Number(edge.confidence) >= 0.9)).toBe(true);
    const search = filterGraph(dataset, { search: "kodokan" });
    expect(search.nodes.map((node) => node.id)).toEqual(["judo"]);
    const connected = filterGraph(dataset, { includeIsolates: false });
    expect(connected.nodes.every((node) => ["judo", "kano", "tokyo"].includes(String(node.id)))).toBe(true);
    expect(connected.totalNodes).toBe(connected.nodes.length);
  });

  it("hides the relations every source only claims when asked, and keeps them by default", () => {
    const claimed = {
      ...dataset,
      edges: dataset.edges.map((edge) => (edge.id === "r2" ? { ...edge, assertion: "claimed" } : { ...edge, assertion: "reported" })),
    };
    expect(filterGraph(claimed).edges.map((edge) => edge.id)).toEqual(["r1", "r2"]);
    const reported = filterGraph(claimed, { includeClaimed: false, includeIsolates: false });
    expect(reported.edges.map((edge) => edge.id)).toEqual(["r1"]);
    expect(reported.nodes.map((node) => node.id).sort()).toEqual(["judo", "kano"]);
  });

  it("reads community members from the pipeline's parquet column", () => {
    const exported = {
      ...dataset,
      communities: [{ id: "c1", title: "Founders", member_node_ids: ["kano", "tokyo"] }],
    };
    expect(communityMemberIds(exported.communities[0]!)).toEqual(["kano", "tokyo"]);
    const scoped = filterGraph(exported, { communityIds: ["c1"] });
    expect(scoped.nodes.map((node) => node.id)).toEqual(["kano", "tokyo"]);
  });

  it("keeps the best-connected entities when a limit applies", () => {
    const capped = filterGraph(dataset, { limit: 2 });
    expect(capped.totalNodes).toBe(dataset.nodes.length);
    expect(capped.nodes.map((node) => node.id)).toEqual(["judo", "kano"]);
    expect(capped.edges.map((edge) => edge.id)).toEqual(["r1"]);
    const uncapped = filterGraph(dataset, { limit: Number.POSITIVE_INFINITY });
    expect(uncapped.nodes).toHaveLength(dataset.nodes.length);
  });
});

describe("identity", () => {
  it("reads only the headers the gate writes, the address before the subject id", () => {
    const trusted = { trustIdentityHeaders: true, snowflakeHosted: false };
    const gated = viewerFromHeaders(
      new Headers({
        "x-auth-request-email": "alice@example.test",
        "x-auth-request-user": "c7s1v9-opaque-subject",
        "x-auth-request-groups": "app_operator, analyst",
      }),
      trusted,
    );
    expect(gated).toEqual({ userName: "ALICE@EXAMPLE.TEST", email: "alice@example.test", roles: ["APP_OPERATOR", "ANALYST"] });
    expect(viewerFromHeaders(new Headers({ "x-auth-request-user": "subject" }), trusted).userName).toBe("SUBJECT");
    // Headers the gate never overwrites are whatever the caller chose.
    const forged = viewerFromHeaders(
      new Headers({ "x-forwarded-preferred-username": "mallory", "x-forwarded-email": "mallory@example.test" }),
      trusted,
    );
    expect(forged.userName).toBe("");
    expect(viewerFromHeaders(new Headers({ "x-auth-request-email": "alice@example.test" }), { ...trusted, trustIdentityHeaders: false }).userName).toBe("");
  });

  it("reads Sf-Context headers only where the console is told it is Snowflake-hosted", () => {
    // Only the user header is the ingress's own; roles sent beside it are the caller's.
    const headers = new Headers({
      "sf-context-current-user": "alice",
      "sf-context-current-user-roles": "analyst",
      "sf-context-current-user-email": "mallory@example.com",
    });
    expect(viewerFromHeaders(headers, { trustIdentityHeaders: false, snowflakeHosted: true })).toEqual({
      userName: "ALICE",
      email: "",
      roles: [],
    });
    expect(viewerFromHeaders(headers, { trustIdentityHeaders: false, snowflakeHosted: false }).userName).toBe("");
  });
});

describe("catalog", () => {
  it("persists run records and graph names", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-catalog-"));
    const directory = path.join(stateRoot, "runs", "run_1");
    await mkdir(directory, { recursive: true });
    await writeRunRecord(directory, {
      runId: "run_1",
      graphId: "graph_1",
      status: "succeeded",
      runtime: "local",
    });
    const name = await renameGraph(stateRoot, "graph_1", "  Martial arts  ");
    expect(name).toBe("Martial arts");
    const listed = await listRunRecords(stateRoot, 10);
    expect(listed[0]?.record.runId).toBe("run_1");
    // A refused revision attempt is not listed, but its record is still there.
    await withdrawRun(stateRoot, "run_1");
    expect(await listRunRecords(stateRoot, 10)).toHaveLength(0);
    expect(await listRunRecords(stateRoot, 10, { includeWithdrawn: true })).toHaveLength(1);
  });
});

describe("gold comparison", () => {
  it("treats gold-derived datasets as fully matched", () => {
    const gold = {
      name: "demo",
      entities: [
        { id: "e1", name: "Judo", type: "MARTIAL_ART" },
        { id: "e2", name: "Kano", type: "PERSON" },
      ],
      relations: [{ id: "r1", source: "e1", target: "e2", relation_type: "DEVELOPED_BY", required: true }],
    };
    const dataset = goldToDataset(gold, "graph_demo");
    const comparison = compareToGold(dataset, gold);
    expect(comparison.missingRequired).toEqual([]);
    expect(comparison.matchedRequired).toBe(1);
  });
});
