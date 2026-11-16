import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildRunConfig, defaultOntologySelection, runOntologySelection } from "./config";
import { ontologyProfile, ontologySelectionFromProfile } from "./ontology-profile";
import type { IngestionRequest, OntologySelection } from "./protocol/schema";
import { appEnv } from "./env";
import { availableSamplePacks } from "./sample-packs";

function request(overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  return {
    runtime: "local",
    jobId: "job_1",
    graphId: "graph_1",
    graphName: null,
    sourceKind: "local_path",
    source: { path: "/corpus" },
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
    output: { kind: "local_files", workspacePath: "/graphs/graph_1", snowflake: null },
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

const selection: OntologySelection = {
  entityTypes: [
    { name: "PERSON", description: "A named human being." },
    { name: "TECHNIQUE", description: "" },
  ],
  relationTypes: [{ name: "TRAINED_UNDER", description: "The source studied under the target." }],
  relations: "guided",
};

describe("the run's ontology", () => {
  it("travels inline as a profile the pipeline validates, built on the default profile", async () => {
    // What every confined request carries when the form names no base.
    const baseConfigPath = path.resolve(import.meta.dirname, "../../../configs/app-defaults.yaml");
    const config = await buildRunConfig(request({ ontology: selection, baseConfigPath }));
    expect(config.ontology).toEqual({
      profile: {
        name: "general",
        description: "General-purpose profile for evidence-grounded document graphs.",
        mode: "hybrid",
        entity_types: [
          { name: "PERSON", description: "A named human being." },
          { name: "TECHNIQUE", description: "A source-grounded technique." },
        ],
        relation_types: [{ name: "TRAINED_UNDER", description: "The source studied under the target." }],
      },
    });
  });

  it("keeps what a sample pack's terms carry beyond their names, minus what an edit removed", async () => {
    const pack = await availableSamplePacks(appEnv().repositoryRoot);
    const martial = pack.find((item) => item.path === "data/martial_arts/files");
    expect(martial?.ontology?.entityTypes.map((term) => term.name)).toContain("SCHOOL");
    const withoutSchools = {
      ...martial!.ontology!,
      entityTypes: martial!.ontology!.entityTypes.filter((term) => term.name !== "SCHOOL"),
    };
    const config = await buildRunConfig(
      // As a confined request carries it: resolved against the repository.
      request({ source: { path: path.join(appEnv().repositoryRoot, "data/martial_arts/files") }, ontology: withoutSchools }),
    );
    const profile = (config.ontology as { profile: Record<string, unknown> }).profile;
    expect(profile.name).toBe("martial-arts-history");
    const founded = (profile.relation_types as Array<Record<string, unknown>>).find((item) => item.name === "FOUNDED_BY");
    expect(founded?.evidence_cues).toEqual(["\\b(?:founded|established|created|formed)\\b"]);
    expect(founded?.aliases).toEqual(["established_by", "created_by"]);
    // SCHOOL is gone, so the pipeline must not be told FOUNDED_BY starts at one.
    expect(founded?.source_types).toEqual(["ORGANIZATION"]);
    const martialArt = (profile.entity_types as Array<Record<string, unknown>>).find((item) => item.name === "MARTIAL_ART");
    expect(martialArt?.aliases).toEqual(["style", "system", "combat_sport"]);
  });

  it("drops what refers to a removed term, and aliases that would collide", () => {
    const base = {
      name: "base",
      description: "A base profile.",
      entity_types: [
        { name: "PERSON", description: "" },
        { name: "PLACE", description: "" },
      ],
      relation_types: [
        { name: "PARENT_OF", description: "", inverse: "CHILD_OF", source_types: ["PLACE"], aliases: ["father_of"] },
        { name: "CHILD_OF", description: "", inverse: "PARENT_OF" },
      ],
    };
    const profile = ontologyProfile(
      {
        entityTypes: [{ name: "PERSON", description: "" }],
        relationTypes: [
          { name: "PARENT_OF", description: "" },
          { name: "Father-Of", description: "Stated as father." },
        ],
        relations: "guided",
      },
      base,
    );
    const [parent, father] = profile.relation_types as Array<Record<string, unknown>>;
    expect(parent).toEqual({ name: "PARENT_OF", description: "A source-grounded parent of.", aliases: [] });
    expect(father).toEqual({ name: "Father-Of", description: "Stated as father." });
  });

  it("maps the relation choice onto the profile's mode", () => {
    expect(ontologyProfile({ ...selection, relations: "fixed" }).mode).toBe("closed");
    const open = ontologyProfile({ ...selection, relations: "open" });
    expect(open.mode).toBe("open");
    expect(open.relation_types).toEqual([]);
  });

  it("removes fallback and complete rewrite rules whose vocabulary was removed", () => {
    const base = {
      fallback_relation: "RELATED_TO",
      entity_types: [{ name: "PERSON" }, { name: "PLACE" }],
      relation_types: [{ name: "RELATED_TO" }, { name: "LOCATED_IN" }],
      relation_rewrites: [
        { relation: "RELATED_TO", to: "LOCATED_IN", source_types: ["PERSON"], target_types: ["PLACE"] },
        { relation: "RELATED_TO", to: "RELATED_TO", source_types: ["PERSON"] },
      ],
    };
    const kept = { entityTypes: [{ name: "PERSON", description: "" }], relationTypes: [{ name: "RELATED_TO", description: "" }], relations: "guided" as const };
    expect(ontologyProfile(kept, base)).toMatchObject({ fallback_relation: "RELATED_TO", relation_rewrites: [base.relation_rewrites[1]] });
    const open = ontologyProfile({ ...kept, relations: "open" }, base);
    expect(open).not.toHaveProperty("fallback_relation");
    expect(open.relation_rewrites).toEqual([]);
  });

  it("keeps a rewrite that names a kept relation by an alias or another spelling", () => {
    const base = {
      entity_types: [{ name: "PERSON" }],
      relation_types: [{ name: "RELATED_TO", aliases: ["associated with"] }, { name: "KNOWS" }],
      relation_rewrites: [{ relation: "associated with", to: "knows", source_types: ["PERSON"] }],
    };
    const kept = {
      entityTypes: [{ name: "PERSON", description: "" }],
      relationTypes: [{ name: "RELATED_TO", description: "" }, { name: "KNOWS", description: "" }],
      relations: "guided" as const,
    };
    expect(ontologyProfile(kept, base).relation_rewrites).toEqual(base.relation_rewrites);
  });

  it("reads a profile back into the same terms, and is read back from a run's config", async () => {
    const roundTrip = ontologySelectionFromProfile(ontologyProfile(selection));
    expect(roundTrip.entityTypes.map((term) => term.name)).toEqual(["PERSON", "TECHNIQUE"]);
    expect(roundTrip.relations).toBe("guided");
    expect(ontologySelectionFromProfile(ontologyProfile({ ...selection, relations: "fixed" })).relations).toBe("fixed");
    expect(ontologySelectionFromProfile(ontologyProfile({ ...selection, relations: "open" })).relations).toBe("open");

    const directory = await mkdtemp(path.join(tmpdir(), "fg-config-"));
    const configPath = path.join(directory, "config.yaml");
    await writeFile(
      configPath,
      "ontology:\n  profile:\n    name: console\n    mode: closed\n    entity_types:\n      - name: WIDGET\n        description: A widget.\n    relation_types:\n      - name: PART_OF\n        description: Part.\n",
      "utf8",
    );
    expect(await runOntologySelection(configPath)).toEqual({
      entityTypes: [{ name: "WIDGET", description: "A widget." }],
      relationTypes: [{ name: "PART_OF", description: "Part." }],
      relations: "fixed",
    });
    expect(await runOntologySelection(path.join(directory, "missing.yaml"))).toBeNull();
    await writeFile(path.join(directory, "bare.yaml"), "ontology:\n  profile_path: general.yaml\n", "utf8");
    expect(await runOntologySelection(path.join(directory, "bare.yaml"))).toBeNull();
  });

  it("carries the folder setting in the profile, so it survives composing against a fleet", async () => {
    const folderEntity = {
      entityType: "PERSON",
      relationType: "APPEARS_IN",
      level: 2,
      excludedFileIds: ["file-a"],
      excludedPaths: ["Studies/Northwind Pricing Study/notes.txt"],
    };
    const profile = ontologyProfile({ ...selection, folderEntity });
    expect(profile.folder_entity).toEqual({
      entity_type: "PERSON",
      relation_type: "APPEARS_IN",
      level: 2,
      excluded_file_ids: ["file-a"],
      excluded_paths: ["Studies/Northwind Pricing Study/notes.txt"],
    });
    // The pipeline adds the structural relation; reading the profile back
    // gives the same terms and the same setting, not an extra relation.
    const stored = {
      ...profile,
      relation_types: [
        ...(profile.relation_types as unknown[]),
        { name: "APPEARS_IN", description: "Filed.", target_types: ["PERSON"], structural: true },
      ],
    };
    const read = ontologySelectionFromProfile(stored);
    expect(read.relationTypes.map((term) => term.name)).toEqual(["TRAINED_UNDER"]);
    expect(read.folderEntity).toEqual(folderEntity);
    expect(ontologyProfile(read)).toEqual({ ...profile, name: "console", description: "Chosen on the console for this graph." });
    // A base profile's folder setting is the base's, not the selection's.
    expect(ontologyProfile(selection, stored)).not.toHaveProperty("folder_entity");

    const fleet = { namespace: "fleet", configMap: "fleet-config", config: {}, parsingEndpoint: null, ontology: { name: "fleet" } };
    const config = await buildRunConfig(
      request({ runtime: "kubernetes", ontology: { ...selection, folderEntity } }),
      fleet,
      "owner@example.com",
    );
    expect((config.ontology as { profile: Record<string, unknown> }).profile.folder_entity).toMatchObject({ level: 2 });
  });

  it("puts a revision's folder exclusions in place of the base version's", async () => {
    const folderEntity = {
      entityType: "PERSON",
      relationType: "APPEARS_IN",
      level: 1,
      excludedFileIds: ["file-a"],
      excludedPaths: ["Studies/notes.txt"],
    };
    const config = await buildRunConfig(
      request({
        ontology: { ...selection, folderEntity },
        revision: { baseRunId: "run_base", dropFileIds: [], addDocuments: false, folderExclusions: ["file-c", "file-b", "file-c"] },
      }),
    );
    expect((config.ontology as { profile: Record<string, unknown> }).profile.folder_entity).toMatchObject({
      excluded_file_ids: ["file-b", "file-c"],
      excluded_paths: [],
    });
  });

  it("offers the base configuration's profile file as the default", async () => {
    const defaults = await defaultOntologySelection(path.resolve(import.meta.dirname, "../../../configs/app-defaults.yaml"));
    expect(defaults.entityTypes.map((term) => term.name)).toEqual([
      "PERSON",
      "ORGANIZATION",
      "LOCATION",
      "EVENT",
      "CONCEPT",
      "WORK",
      "DATE",
    ]);
    expect(defaults.relations).toBe("guided");
    expect(defaults.relationTypes.map((term) => term.name)).toContain("LOCATED_IN");
  });
});

describe("the owner a run records", () => {
  it("names the submitter as the graph's owner, and nobody when there is none", async () => {
    const owned = await buildRunConfig(request(), null, "ALICE@EXAMPLE.COM");
    expect((owned.job as Record<string, unknown>).owner).toBe("ALICE@EXAMPLE.COM");
    const unowned = await buildRunConfig(request());
    expect((unowned.job as Record<string, unknown>).owner).toBeUndefined();
  });
});

describe("caller provider options", () => {
  it("accepts only credential-bearing Snowflake fields", async () => {
    const snowflake = { account: "synthetic-account", user: "synthetic-user", database: "GRAPH", schema: "PUBLIC", warehouse: "COMPUTE", role: null, host: null, authenticator: null, bulkStage: "@GRAPH.PUBLIC.LOAD", credentialEnvironmentVariable: "KG_SNOWFLAKE_PASSWORD", credentialField: "host" };
    await expect(buildRunConfig(request({ output: { kind: "snowflake", workspacePath: "/graphs/graph_1", snowflake } }))).rejects.toThrow(/credential field/);
  });
  it("refuses environment interpolation before config composition", async () => {
    await expect(buildRunConfig(request({ source: { path: "/corpus/${KG_SECRET}" } }))).rejects.toThrow(/operator profiles/);
  });
  it("refuses executable and provider overrides while accepting data settings", async () => {
    for (const options of [{ tesseract_command: "untrusted" }, { mineru_command: "untrusted" }, { tesseract_pdf_renderer_command: "untrusted" }, { provider: "untrusted" }]) {
      await expect(buildRunConfig(request({ ocr: { ...request().ocr, options } }))).rejects.toThrow(/Unsupported ocr option/);
    }
    const config = await buildRunConfig(request({ ocr: { ...request().ocr, options: { language: "en", timeout_seconds: 45 } } }));
    expect(config.ocr).toMatchObject({ language: "en", timeout_seconds: 45 });
  });

  it("offers only the LLM settings the pipeline declares", async () => {
    await expect(buildRunConfig(request({ llm: { ...request().llm, options: { temperature: 0.2 } } }))).rejects.toThrow(/Unsupported llm option/);
    const config = await buildRunConfig(request({ llm: { ...request().llm, options: { context_window_tokens: 16384, max_output_tokens: 2048 } } }));
    expect(config.llm).toMatchObject({ context_window_tokens: 16384, max_output_tokens: 2048 });
  });
});

describe("hosted OCR configuration", () => {
  it.each([
    ["mineru_api", "ocr", "mineru_api_url", "mineru_api_key"],
    ["generic_http", "generic_http_ocr", "endpoint", "api_key"],
  ])("maps %s connection fields to its transport", async (provider, section, endpointField, keyField) => {
    const endpoint = provider === "mineru_api" ? "https://parse.example.test" : "https://parse.example.test/parse";
    const profile = { [section]: { [endpointField]: endpoint, [keyField]: "${KG_SYNTHETIC_OCR_KEY}" } };
    const file = path.join(await mkdtemp(path.join(tmpdir(), "fg-ocr-config-")), "profile.yaml");
    await writeFile(file, JSON.stringify(profile));
    process.env.KG_SYNTHETIC_OCR_KEY = "synthetic";
    try {
      const cfg = await buildRunConfig(request({ baseConfigPath: file, ocr: { provider, model: null, endpoint, apiKeyEnvironmentVariable: "KG_SYNTHETIC_OCR_KEY", dimension: null, options: {} } }), { config: profile } as never);
      expect(cfg[section]).toMatchObject({ [endpointField]: endpoint, [keyField]: "${KG_SYNTHETIC_OCR_KEY}" });
      expect(cfg.ocr).not.toHaveProperty("endpoint");
      expect(cfg.ocr).not.toHaveProperty("api_key");
    } finally { delete process.env.KG_SYNTHETIC_OCR_KEY; }
  });
});
