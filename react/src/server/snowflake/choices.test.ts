import { describe, expect, it } from "vitest";
import { cortexModelChoices, documentStageChoices, readableStageFolders } from "./choices";
import type { Viewer } from "../protocol/schema";

const alice: Viewer = { userName: "ALICE@EXAMPLE.COM", email: "", roles: [] };

describe("documentStageChoices", () => {
  it("offers document stages, upload stage first, and leaves out the worker's stages and repositories", () => {
    const rows = [
      { name: "KG_LOAD_STAGE", type: "INTERNAL" },
      { name: "ARCHIVE", type: "INTERNAL" },
      { name: "APP_REPO", type: "ARTIFACT REPOSITORY" },
      { name: "KG_SERVICE_SPECS", type: "INTERNAL" },
      { name: "KG_DOCS", type: "INTERNAL" },
    ];

    expect(documentStageChoices(rows, "GRAPH_DB", "GRAPH")).toEqual([
      { stage: "@GRAPH_DB.GRAPH.KG_DOCS", name: "KG_DOCS" },
      { stage: "@GRAPH_DB.GRAPH.ARCHIVE", name: "ARCHIVE" },
    ]);
  });
});

describe("readableStageFolders", () => {
  const names = [
    "kg_docs/app/ALICE_EXAMPLE.COM/run-1/a.pdf",
    "kg_docs/app/ALICE_EXAMPLE.COM/run-1/b.md",
    "kg_docs/app/ALICE_EXAMPLE.COM/run-1/notes.exe",
    "kg_docs/app/BOB_EXAMPLE.COM/run-2/c.pdf",
    "kg_docs/shared/papers/d.pdf",
    "kg_docs/top-level.pdf",
  ];

  it("offers only folders the viewer may run on, counting the documents under each", () => {
    expect(readableStageFolders(names, "@GRAPH_DB.GRAPH.KG_DOCS", alice)).toEqual([
      { prefix: "app/ALICE_EXAMPLE.COM", documents: 2 },
      { prefix: "app/ALICE_EXAMPLE.COM/run-1", documents: 2 },
      { prefix: "shared", documents: 1 },
      { prefix: "shared/papers", documents: 1 },
    ]);
  });

  it("never offers the upload namespace or another viewer's uploads", () => {
    const prefixes = readableStageFolders(names, "@GRAPH_DB.GRAPH.KG_DOCS", alice).map((folder) => folder.prefix);

    expect(prefixes).not.toContain("app");
    expect(prefixes.some((prefix) => prefix.includes("BOB"))).toBe(false);
  });
});

describe("cortexModelChoices", () => {
  it("splits completion models from embeddings of known width and drops task models", () => {
    const choices = cortexModelChoices([
      "MISTRAL-LARGE2",
      "claude-sonnet-4-5",
      "snowflake-arctic-embed-l-v2.0",
      "snowflake-arctic-embed-m-v1.5",
      "voyage-multilingual-2",
      "arctic-extract",
      "arctic-translate",
      "llamaguard3-8b",
      "voyage-multimodal-3",
      "snowflake-arctic-embed-xs",
    ]);

    expect(choices.llm).toEqual(["claude-sonnet-4-5", "mistral-large2"]);
    expect(choices.embedding).toEqual([
      { model: "snowflake-arctic-embed-m-v1.5", dimension: 768 },
      { model: "snowflake-arctic-embed-l-v2.0", dimension: 1024 },
      { model: "voyage-multilingual-2", dimension: 1024 },
    ]);
  });
});
