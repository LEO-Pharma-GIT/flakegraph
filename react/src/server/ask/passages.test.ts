import { MockEmbeddingModelV4 } from "ai/test";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { VectorBuilder } from "../chunk-vectors";
import { attachRetrievalSource } from "../graph-retrieval";
import type { GraphDataset } from "../protocol/schema";
import { attachChunkVectors, recordedEmbedding, resetDenseReports, sameEmbeddingModel } from "./dense";
import { resetAskEmbedding, type AskEmbedding } from "./embedding-model";
import { renderForModel } from "./evidence-budget";
import { groundingFromToolOutput } from "./grounding";
import { leadingContext, searchPassages, trailingContext } from "./passages";
import { searchEvidence } from "./retrieve";
import { ASK_TOOL_NAMES, createAskTools } from "./tools";

// One page of an invented harbour survey, cut into three chunks that overlap
// the way a chunker's windows do.
const PAGE =
  "Lantern sales in the EU rose by twelve percent. Harbour traffic doubled after the new quay opened. " +
  "Kite festivals moved inland when the dunes closed.";
const cut = (start: number, end: number) => ({ start_offset: start, end_offset: end, content: PAGE.slice(start, end) });
const A0 = cut(0, 60);
const A1 = cut(40, 110);
const A2 = cut(95, PAGE.length);

function fixture(): GraphDataset {
  return {
    graphId: "graph_harbour",
    nodes: [{ id: "n_lantern", name: "Lantern" }],
    edges: [],
    communities: [],
    evidence: [
      { id: "ev_1", subject_kind: "node", subject_id: "n_lantern", file_id: "doc_a", quote: "Lantern sales rose." },
      { id: "ev_2", subject_kind: "node", subject_id: "n_lantern", file_id: "doc_b", quote: "Otters nap in the kelp." },
    ],
    documents: [
      { id: "doc_a", filename: "harbour-survey.pdf" },
      { id: "doc_b", filename: "dune-notes.pdf" },
      { id: "doc_c", file_id: "file_c", filename: "tide-tables.xlsx" },
    ],
    chunks: [
      // Out of reading order on purpose: neighbours come from chunk_index.
      { id: "a2", document_id: "doc_a", chunk_index: 2, page_number: 1, section_path: ["Results"], ...A2 },
      { id: "a0", document_id: "doc_a", chunk_index: 0, page_number: 1, section_path: ["Results"], ...A0 },
      { id: "a1", document_id: "doc_a", chunk_index: 1, page_number: 1, section_path: ["Results"], ...A1 },
      { id: "a3", document_id: "doc_a", chunk_index: 3, page_number: 2, section_path: ["Methods"], content: "Counts were taken weekly by volunteers.", start_offset: 0, end_offset: 39 },
      { id: "b0", document_id: "doc_b", chunk_index: 0, page_number: 4, content: "Otters nap in the kelp beds near the dunes." },
      { id: "b1", document_id: "doc_b", chunk_index: 1, page_number: 5, content: "Chart (bar): Visitors by season\n| Season | Visitors |\n| --- | --- |\n| Summer | 900 |" },
      { id: "c0", file_id: "file_c", chunk_index: 0, page_number: 1, content: "| Tide | Height |\n| --- | --- |\n| High | 4.2 m |", block_ids: ["blk_table"] },
      { id: "c1", file_id: "file_c", chunk_index: 1, page_number: 2, content: "A drawing of the estuary at low water.", block_ids: ["blk_figure"] },
    ],
    runReport: {},
    graphMetrics: {},
    discardedWindows: [],
    rejectedRecords: [],
    failedDocuments: [],
  };
}

const ids = (result: { passages: { chunkId: string }[] }) => result.passages.map((passage) => passage.chunkId);

describe("passage search", () => {
  it("ranks passages by BM25 and says which method found them", async () => {
    const result = await searchPassages(fixture(), { query: "EU lantern sales", embedding: null });
    expect(result.retrieval).toBe("bm25");
    expect(result.passages[0]).toMatchObject({
      chunkId: "a0",
      documentId: "doc_a",
      documentName: "harbour-survey.pdf",
      pageNumber: 1,
      sectionPath: ["Results"],
      method: "bm25",
    });
  });

  it("finds a passage by its section heading and by its document's name", async () => {
    expect(ids(await searchPassages(fixture(), { query: "methods", embedding: null }))).toEqual(["a3"]);
    expect(ids(await searchPassages(fixture(), { query: "tide tables", embedding: null }))).toEqual(expect.arrayContaining(["c0", "c1"]));
  });

  it("keeps to the documents and chunks the caller may see", async () => {
    const dataset = fixture();
    const only = (documentIds: string[]) => searchPassages(dataset, { query: "dunes", documentIds, embedding: null });
    expect(ids(await searchPassages(dataset, { query: "dunes", embedding: null }))).toEqual(expect.arrayContaining(["a2", "b0"]));
    // b1 says nothing of dunes, but its document is named for them.
    expect(ids(await only(["doc_b"]))).toEqual(["b0", "b1"]);
    expect(ids(await only([]))).toEqual([]);
    // A document named by its file id is the same document.
    expect(ids(await searchPassages(dataset, { query: "tide", documentIds: ["file_c"], embedding: null }))).toEqual(
      expect.arrayContaining(["c0"]),
    );
    expect(ids(await searchPassages(dataset, { query: "dunes", chunkIds: new Set(["a2"]), embedding: null }))).toEqual(["a2"]);
  });

  it("marks passages that hold a table, a chart or a figure", async () => {
    const dataset = fixture();
    attachRetrievalSource(dataset, {
      directory: null,
      blocks: [
        { id: "blk_table", kind: "table" },
        { id: "blk_figure", kind: "figure" },
      ],
    });
    const contains = async (query: string) =>
      Object.fromEntries((await searchPassages(dataset, { query, embedding: null })).passages.map((passage) => [passage.chunkId, passage.contains]));
    expect((await contains("visitors season"))["b1"]).toEqual(["table", "chart"]);
    expect((await contains("tide height"))["c0"]).toEqual(["table"]);
    expect((await contains("estuary drawing"))["c1"]).toEqual(["figure"]);
    expect((await contains("otters"))["b0"]).toEqual([]);
  });
});

describe("neighbour expansion", () => {
  it("adds the neighbours of the same document without repeating what overlaps", async () => {
    const result = await searchPassages(fixture(), { query: "harbour traffic quay", expand: true, topK: 1, embedding: null });
    expect(ids(result)).toEqual(["a1"]);
    const text = result.passages[0]!.text;
    expect(text).toBe([PAGE.slice(0, 40).trim(), A1.content, PAGE.slice(110).trim()].join("\n"));
    // The words both windows hold appear once.
    expect(text.split("Harbour traffic").length - 1).toBe(1);
    expect(text.split("closed").length - 1).toBe(1);
  });

  it("trims only overlaps on one page whose offsets span their text", () => {
    const hit = { page_number: 1, start_offset: 10, end_offset: 20, content: "0123456789" };
    expect(leadingContext({ page_number: 1, start_offset: 0, end_offset: 15, content: "abcdefghijKLMNO" }, hit)).toBe("abcdefghij");
    expect(trailingContext(hit, { page_number: 1, start_offset: 15, end_offset: 25, content: "KLMNOpqrst" })).toBe("pqrst");
    // Another page: offsets are into another page's text, so nothing is shared.
    expect(leadingContext({ page_number: 2, start_offset: 0, end_offset: 15, content: "abcdefghijKLMNO" }, hit)).toBe("abcdefghijKLMNO");
    // Offsets that do not span the text cannot be trusted to cut it.
    expect(leadingContext({ page_number: 1, start_offset: 0, end_offset: 99, content: "abcdefghijKLMNO" }, hit)).toBe("abcdefghijKLMNO");
  });

  it("does not repeat a neighbour that is itself a result", async () => {
    const result = await searchPassages(fixture(), { query: "lantern harbour", expand: true, topK: 2, embedding: null });
    expect(ids(result).sort()).toEqual(["a0", "a1"]);
    const all = result.passages.map((passage) => passage.text).join("\n");
    expect(all.split("Lantern sales").length - 1).toBe(1);
  });

  it("leaves a passage alone unless asked", async () => {
    const result = await searchPassages(fixture(), { query: "harbour traffic quay", topK: 1, embedding: null });
    expect(result.passages[0]!.text).toBe(A1.content);
  });
});

// Invented three-dimensional "embeddings": each axis is a theme.
const THEMES: Record<string, number[]> = {
  a0: [1, 0, 0],
  a1: [1, 0.2, 0],
  a2: [0, 1, 0],
  a3: [0, 0, 1],
  b0: [0, 1, 0.1],
  b1: [0.2, 0, 1],
  c0: [0, 0.1, 1],
  c1: [0, 0.9, 0.3],
};

function withVectors(dataset: GraphDataset, record: Record<string, unknown> = { model: "Org/Tiny-Embed", dimension: 3 }): GraphDataset {
  const recorded = { ...dataset, graphMetrics: { embedding: record } };
  const builder = new VectorBuilder(recorded.chunks.map((chunk) => String(chunk.id)));
  for (const [id, vector] of Object.entries(THEMES)) builder.add(id, vector);
  attachChunkVectors(recorded, builder.build());
  return recorded;
}

function embedding(vector: number[] | Error, modelId = "tiny-embed"): AskEmbedding & { model: MockEmbeddingModelV4 } {
  const model = new MockEmbeddingModelV4({
    modelId,
    doEmbed: async () => {
      if (vector instanceof Error) throw vector;
      return { embeddings: [vector], warnings: [] };
    },
  });
  return { model, modelId, provider: "openai" };
}

describe("dense retrieval", () => {
  let info: MockInstance<typeof console.info>;
  beforeEach(() => {
    resetDenseReports();
    info = vi.spyOn(console, "info").mockImplementation(() => {});
  });
  afterEach(() => info.mockRestore());

  it("fuses dense and BM25 rankings when the model and widths match", async () => {
    // "estuary drawing" names c1 lexically; the query vector points at the
    // dunes theme, which only the vectors know b0 and a2 belong to.
    const result = await searchPassages(withVectors(fixture()), { query: "estuary drawing", embedding: embedding([0, 1, 0]) });
    expect(result.retrieval).toBe("hybrid");
    const methods = Object.fromEntries(result.passages.map((passage) => [passage.chunkId, passage.method]));
    expect(methods.c1).toBe("hybrid");
    expect(methods.a2).toBe("dense");
    expect(ids(result)[0]).toBe("c1");
  });

  it("falls back to BM25 quietly, once per reason, when dense retrieval cannot run", async () => {
    const plain = fixture();
    const cases: Array<[string, GraphDataset, AskEmbedding | null]> = [
      ["no model configured", withVectors(plain), null],
      ["graph records no model", withVectors(plain, { dimension: 3 }), embedding([0, 1, 0])],
      ["another model", withVectors(plain, { model: "other-embed", dimension: 3 }), embedding([0, 1, 0])],
      ["no vectors", { ...plain, graphMetrics: { embedding: { model: "tiny-embed", dimension: 3 } } }, embedding([0, 1, 0])],
      ["recorded width differs", withVectors(plain, { model: "tiny-embed", dimension: 4 }), embedding([0, 1, 0])],
      ["query width differs", withVectors(plain), embedding([0, 1])],
      ["embedding call fails", withVectors(plain), embedding(new Error("endpoint down"))],
    ];
    for (const [label, dataset, model] of cases) {
      const result = await searchPassages(dataset, { query: "estuary drawing", embedding: model });
      expect([label, result.retrieval]).toEqual([label, "bm25"]);
      expect([label, ids(result)]).toEqual([label, ["c1"]]);
    }
    // The model is not asked to embed a query its vectors could not be compared with.
    expect((cases[2]![2]!.model as MockEmbeddingModelV4).doEmbedCalls).toHaveLength(0);
    expect((cases[1]![2]!.model as MockEmbeddingModelV4).doEmbedCalls).toHaveLength(0);
    const logged = info.mock.calls.map((call) => String(call[0]));
    expect(logged).toHaveLength(6);
    await searchPassages(cases[0]![1], { query: "estuary", embedding: null });
    expect(info.mock.calls).toHaveLength(6);
    expect(logged.every((line) => line.includes("BM25 alone"))).toBe(true);
  });

  it("reads the recorded model from graph metrics, then the run report", () => {
    const plain = fixture();
    expect(recordedEmbedding(plain)).toEqual({ model: null, dimension: null });
    expect(recordedEmbedding({ ...plain, runReport: { embedding: { dimension: 1024 } } })).toEqual({ model: null, dimension: 1024 });
    expect(
      recordedEmbedding({ ...plain, graphMetrics: { embedding: { model: "m1", dimension: 8 } }, runReport: { embedding: { model: "m2" } } }),
    ).toEqual({ model: "m1", dimension: 8 });
  });

  it("matches a model by name, ignoring case and a hub organisation", () => {
    expect(sameEmbeddingModel("Org/Tiny-Embed", "tiny-embed")).toBe(true);
    expect(sameEmbeddingModel("TINY-EMBED", "tiny-embed")).toBe(true);
    expect(sameEmbeddingModel("tiny-embed-v2", "tiny-embed")).toBe(false);
  });
});

describe("the searchPassages tool", () => {
  const options = { toolCallId: "passages", messages: [], context: {} };
  beforeEach(() => resetAskEmbedding());

  it("is registered and returns passages with their place in the document", async () => {
    const tools = createAskTools({ dataset: fixture() });
    expect(ASK_TOOL_NAMES).toContain("searchPassages");
    expect(Object.keys(tools).sort()).toEqual([...ASK_TOOL_NAMES].sort());
    const output = await tools.searchPassages.execute!({ query: "EU lantern sales", topK: 1 }, options);
    expect(output).toEqual({
      retrieval: "bm25",
      passages: [
        {
          chunkId: "a0",
          documentId: "doc_a",
          documentName: "harbour-survey.pdf",
          pageNumber: 1,
          sectionPath: ["Results"],
          text: A0.content,
          contains: [],
          method: "bm25",
        },
      ],
    });
  });

  it("respects the session's document scope and the ids the model passes", async () => {
    const scoped = createAskTools({ dataset: fixture(), scope: { documentIds: ["doc_b"] } });
    const output = (await scoped.searchPassages.execute!({ query: "dunes" }, options)) as { passages: { documentId: string }[] };
    expect(output.passages.length).toBeGreaterThan(0);
    expect(output.passages.every((passage) => passage.documentId === "doc_b")).toBe(true);
    const tools = createAskTools({ dataset: fixture() });
    expect(await tools.searchPassages.execute!({ query: "dunes", documentIds: ["doc_missing"] }, options)).toMatchObject({
      code: "GRAPH_DOCUMENT_SCOPE_MISMATCH",
      unknownDocumentIds: ["doc_missing"],
    });
    expect(await tools.searchPassages.execute!({ query: "zeppelin" }, options)).toMatchObject({ code: "GRAPH_EMPTY_RESULT" });
  });

  it("keeps to a perspective's chunks", async () => {
    const dataset: GraphDataset = {
      ...fixture(),
      nodes: [{ id: "n_otter", name: "Otter", source_chunk_ids: ["b0"] }],
      evidence: [],
    };
    const tools = createAskTools({ dataset, scope: { search: "Otter" } });
    const output = (await tools.searchPassages.execute!({ query: "dunes" }, options)) as { passages: { chunkId: string }[] };
    expect(output.passages.map((passage) => passage.chunkId)).toEqual(["b0"]);
  });

  it("cites its passages in the answer, by document", async () => {
    const tools = createAskTools({ dataset: fixture() });
    const output = await tools.searchPassages.execute!({ query: "EU lantern sales", topK: 1 }, options);
    expect(groundingFromToolOutput(output).citations).toEqual([
      { quote: A0.content, documentId: "doc_a", documentName: "harbour-survey.pdf", entityId: null, entityName: null },
    ]);
  });

  it("is shown to the model grouped by document, each passage under its page and section", async () => {
    const tools = createAskTools({ dataset: fixture() });
    const output = await tools.searchPassages.execute!({ query: "dunes visitors", topK: 3 }, options);
    expect(renderForModel(output)).toBe(
      [
        "## dune-notes.pdf (doc_b)",
        "",
        "[p5] (table, chart)",
        "Chart (bar): Visitors by season\n| Season | Visitors |\n| --- | --- |\n| Summer | 900 |",
        "",
        "[p4]",
        "Otters nap in the kelp beds near the dunes.",
        "",
        "## harbour-survey.pdf (doc_a)",
        "",
        "[p1] Results",
        A2.content,
      ].join("\n"),
    );
  });
});

describe("quote search", () => {
  it("ranks quotes and chunks by BM25, whole words only", () => {
    const hits = searchEvidence(fixture(), "otters kelp", 3);
    expect(hits.map((hit) => hit.id)).toEqual(expect.arrayContaining(["ev_2", "b0"]));
    expect(hits.every((hit) => hit.score > 0 && hit.distance < 1)).toBe(true);
    // "ant" is inside "Lantern" but is not a word of it.
    expect(searchEvidence(fixture(), "ant", 3)).toEqual([]);
  });
});
