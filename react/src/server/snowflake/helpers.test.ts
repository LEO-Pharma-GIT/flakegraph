import { existsSync, readdirSync, readFileSync } from "node:fs";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { normalizeRoles, normalizeRow, parseRoleList } from "./client";
import { jobErrorText, liveConsumption, variantMapping, vectorWidth, withLiveStageProgress } from "./progress";
import { GRAPH_DATA_TABLES } from "./graph-tables";
import { RecordingSql } from "./recording-sql";
import {
  buildSpcsLaunch,
  containerResources,
  executeJobServiceSql,
  serviceNameForJob,
} from "./spcs";
import {
  putFolderToStage,
  qualifiedStage,
  relativeStageName,
  sourceObjectFromListRow,
  stableId,
  stageLocation,
  stageRelativePath,
  validateUploadRows,
} from "./stage";
import type { StageProgress } from "../protocol/schema";

function launchConfig(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    job: { job_id: "run-1", graph_id: "graph-1" },
    llm: { provider: "snowflake_cortex", model: "claude-4-sonnet", api_key: "${LLM_KEY}" },
    snowflake: {
      account: "client-supplied",
      user: "MALLORY",
      host: "elsewhere.example.com",
      password: "${SNOWFLAKE_PASSWORD}",
      database: "DB",
      schema: "SCH",
      compute_pool: "POOL_A",
      image_repository: "DB.SCH.REPO",
      image_name: "flakegraph:1.2.3",
      ...overrides,
    },
  };
}

describe("SPCS job specification", () => {
  it("names services from the run alone, so any run stays findable", () => {
    expect(serviceNameForJob("run-1")).toBe("FLAKEGRAPH_APP_4E65D3FBE8AD6535");
  });

  it("starts the worker from an inline specification with the service's own identity", () => {
    const launch = buildSpcsLaunch(launchConfig());
    expect(launch.serviceIdentifier).toBe("DB.SCH.FLAKEGRAPH_APP_4E65D3FBE8AD6535");
    expect(launch.executeSql).toMatch(/^EXECUTE JOB SERVICE\n {2}IN COMPUTE POOL POOL_A\n {2}NAME = DB\.SCH\.FLAKEGRAPH_APP_/);
    expect(launch.executeSql).toContain("ASYNC = TRUE");
    expect(launch.executeSql).toContain("FROM SPECIFICATION $$\n");
    expect(launch.executeSql.trimEnd().endsWith("$$")).toBe(true);
    expect(launch.specYaml).toContain("image: /DB/SCH/REPO/flakegraph:1.2.3");
    const snowflake = launch.runtimeConfig.snowflake as Record<string, unknown>;
    // Client-side coordinates could point the worker at another account or principal.
    for (const key of ["account", "user", "host", "password"]) {
      expect(snowflake).not.toHaveProperty(key);
    }
    expect(snowflake.authenticator).toBe("oauth");
    expect(snowflake.oauth_token_path).toBe("/snowflake/session/token");
    expect(launch.runtimeConfig.llm).not.toHaveProperty("api_key");
    expect(launch.specYaml).not.toContain("LLM_KEY");
    expect(launch.specYaml).not.toContain("SNOWFLAKE_PASSWORD");
  });

  it("rebuilds the same worker from the configuration it stored, which is how a retry starts", () => {
    const first = buildSpcsLaunch(launchConfig());
    const again = buildSpcsLaunch(structuredClone(first.runtimeConfig));
    expect(again.executeSql).toBe(first.executeSql);
  });

  it("refuses unsafe object names and digests before they reach SQL", () => {
    expect(() => buildSpcsLaunch(launchConfig({ compute_pool: "POOL; DROP TABLE KG_NODE" }))).toThrow(/identifier/);
    expect(() => buildSpcsLaunch(launchConfig({ image_repository: "REPO" }))).toThrow(/DATABASE.SCHEMA.REPOSITORY/);
    expect(() => buildSpcsLaunch(launchConfig({ image_digest: "sha256:xyz" }))).toThrow(/digest/);
    expect(() => buildSpcsLaunch(launchConfig({ compute_pool: "" }))).toThrow(/snowflake.compute_pool/);
  });

  it("cannot be broken out of its dollar quoting", () => {
    expect(() => executeJobServiceSql("POOL", "DB.SCH.SVC", "a: $$; DROP TABLE X")).toThrow(/\$\$/);
  });

  it("sizes the container from its pool's node and keeps headroom for the node's agents", () => {
    expect(containerResources({ vcpu: 4, memoryGib: 16 })).toEqual({
      service_cpu_request: "3",
      service_cpu_limit: "4",
      service_memory_request: "14Gi",
      service_memory_limit: "14Gi",
    });
    expect(containerResources({ vcpu: 1, memoryGib: 2 })).toMatchObject({ service_cpu_request: "1", service_memory_request: "1Gi" });
    expect(containerResources({})).toEqual({});
  });
});

describe("stage helpers", () => {
  it("validates a stage before interpolating it", () => {
    expect(stageLocation("@DB.SCH.KG_DOCS", "/app/ALICE/run-1/")).toBe("@DB.SCH.KG_DOCS/app/ALICE/run-1");
    expect(stageLocation("KG_DOCS/app", "ALICE")).toBe("@KG_DOCS/app/ALICE");
    expect(() => stageLocation("KG_DOCS; DROP TABLE KG_NODE")).toThrow(/identifier/);
    expect(() => stageLocation("KG_DOCS", "app/../ALICE")).toThrow(/unsafe/);
    expect(() => stageLocation("KG_DOCS", "it's")).toThrow(/unsafe/);
  });

  it("authorises the path a stage and prefix resolve to together", () => {
    expect(stageRelativePath("KG_DOCS/app/ALICE", "job-1")).toBe("app/ALICE/job-1");
    expect(stageRelativePath("@KG_DOCS", "")).toBe("");
  });

  it("qualifies a bare stage with the app's namespace", () => {
    expect(qualifiedStage("KG_LOAD_STAGE", "DB", "SCH")).toBe("@DB.SCH.KG_LOAD_STAGE");
    expect(qualifiedStage("@OTHER.SCH.STAGE", "DB", "SCH")).toBe("@OTHER.SCH.STAGE");
  });

  it("derives queue identifiers exactly as the processing package does", () => {
    expect(stableId("stage_file", "@kg_docs/a.pdf", "abc")).toBe("stage_file_4e87dc6a4f9e23095206c09883df14e7");
  });

  it("reads LIST rows and skips objects the pipeline cannot read", () => {
    expect(sourceObjectFromListRow({ name: "kg_docs/app/ALICE/a.pdf", size: 10, md5: "m" })).toEqual({
      uri: "@kg_docs/app/ALICE/a.pdf",
      name: "kg_docs/app/ALICE/a.pdf",
      sizeBytes: 10,
      modifiedAt: null,
      checksum: "m",
    });
    expect(sourceObjectFromListRow({ name: "kg_docs/app/ALICE/a.exe" })).toBeNull();
    expect(relativeStageName("kg_docs/app/ALICE/a.pdf", "@DB.SCH.KG_DOCS")).toBe("app/ALICE/a.pdf");
    expect(relativeStageName("KG_DOCS/a.pdf", "KG_DOCS")).toBe("a.pdf");
  });

  it("accepts an upload only when every object was freshly written", async () => {
    expect(() => validateUploadRows([{ status: "UPLOADED" }, { status: "SKIPPED" }], "docs")).toThrow(/SKIPPED/);
    expect(() => validateUploadRows([], "docs")).toThrow(/no status/);
  });

  it("puts every folder of an upload under names the worker can fetch, and leaves nothing behind", async () => {
    const upload = await mkdtemp(path.join(tmpdir(), "fg-put-"));
    await mkdir(path.join(upload, "Market Reports 2024", "Q1"), { recursive: true });
    await writeFile(path.join(upload, "note.md"), "Judo.\n");
    await writeFile(path.join(upload, "a b.md"), "One.\n");
    await writeFile(path.join(upload, "a_b.md"), "Two.\n");
    await writeFile(path.join(upload, "Market Reports 2024", "Q1", "résumé (final).pdf"), "Three.\n");
    const seen = new Map<string, string[]>();
    const sql = new RecordingSql([
      [
        "PUT ",
        (): Array<Record<string, unknown>> => {
          const text = sql.texts.at(-1)!;
          const folder = /^PUT 'file:\/\/(.*)\/\*' /.exec(text)![1]!;
          seen.set(text.replace(folder, "<folder>"), readdirSync(folder, { withFileTypes: true }).filter((item) => item.isFile()).map((item) => item.name).sort());
          return [{ status: "UPLOADED" }];
        },
      ],
    ]);
    await expect(putFolderToStage(sql, upload, "@DB.SCH.KG_DOCS", "app/ALICE/job-1")).resolves.toBe(2);
    expect([...seen]).toEqual([
      ["PUT 'file://<folder>/*' @DB.SCH.KG_DOCS/app/ALICE/job-1 AUTO_COMPRESS = FALSE OVERWRITE = TRUE", ["a_b-2.md", "a_b.md", "note.md"]],
      ["PUT 'file://<folder>/*' @DB.SCH.KG_DOCS/app/ALICE/job-1/Market_Reports_2024/Q1 AUTO_COMPRESS = FALSE OVERWRITE = TRUE", ["r_sum___final_.pdf"]],
    ]);
    const mirror = /^PUT 'file:\/\/(.*)\/\*' /.exec(sql.texts[0]!)![1]!;
    expect(existsSync(mirror)).toBe(false);
  });
});

describe("live progress", () => {
  const fileStage = (stage: string, completed = 0, total: number | null = 1): StageProgress => ({
    stage,
    status: "running",
    completed,
    total,
    elapsedMs: 0,
    message: null,
  });

  it("replaces a file-derived zero with the worker's own counts", () => {
    const [stage] = withLiveStageProgress([fileStage("graph_extraction")], {
      stage: "graph_extraction",
      counts: { batches_completed: 3, batches_total: 8 },
      message: "Extracting",
    });
    expect(stage).toMatchObject({ completed: 3, total: 8, message: "Extracting" });
  });

  it("keeps file counts when the worker reports nothing", () => {
    expect(withLiveStageProgress([fileStage("ocr", 1, 2)], null)).toEqual([fileStage("ocr", 1, 2)]);
  });

  it("still shows a stage the file rows have not recorded", () => {
    const stages = withLiveStageProgress([], JSON.stringify({ stage: "merge", counts: { completed: 1, total: 4 } }));
    expect(stages).toEqual([{ stage: "merge", status: "running", completed: 1, total: 4, elapsedMs: 0, message: null }]);
  });

  it("reads VARIANT as an object or as JSON text, and spend from either", () => {
    expect(variantMapping('{"a":1}')).toEqual({ a: 1 });
    expect(variantMapping("not json")).toEqual({});
    expect(liveConsumption({ consumption: { calls: 7 } })).toEqual({ calls: 7 });
  });

  it("puts a stored job error in front of a person as a sentence", () => {
    expect(jobErrorText({ message: "Image not found" })).toBe("Image not found");
    expect(jobErrorText('{"failed_files": 1}')).toBe("1 document failed. Open the failures below for the reason.");
    expect(jobErrorText("")).toBeNull();
  });

  it("reads a vector width from SHOW COLUMNS", () => {
    expect(vectorWidth('{"type":"VECTOR","dimension":1024}')).toBe(1024);
    expect(vectorWidth('{"type":"TEXT"}')).toBeNull();
  });
});

describe("driver rows", () => {
  it("lower-cases keys and turns timestamps into ISO strings", () => {
    const row = normalizeRow({ ID: "a", CREATED_AT: new Date("2026-01-02T03:04:05Z"), CONFIG: { Nested: 1 } });
    expect(row).toEqual({ id: "a", created_at: "2026-01-02T03:04:05.000Z", config: { Nested: 1 } });
  });

  it("normalises role names however Snowflake quotes them", () => {
    expect(parseRoleList('["PUBLIC","\\"Analysts\\""]')).toEqual(["ANALYSTS", "PUBLIC"]);
    expect(normalizeRoles(["a", "A", " "])).toEqual(["A"]);
  });
});

describe("graph tables", () => {
  it("clears every table the schema gives a graph id", () => {
    const schema = readFileSync(
      path.resolve(import.meta.dirname, "../../../../src/kg_processor/application/snowflake_schema.py"),
      "utf8",
    );
    const owning = [...schema.matchAll(/CREATE TABLE IF NOT EXISTS (KG_\w+) \(([^;]*)\);/g)]
      .filter(([, , body]) => /^\s*GRAPH_ID /m.test(body ?? ""))
      .map(([, table]) => table)
      .sort();
    expect([...GRAPH_DATA_TABLES].sort()).toEqual(owning);
  });
});
