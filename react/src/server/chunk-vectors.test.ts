import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { chunkVectors } from "./ask/dense";
import { readChunkVectors } from "./chunk-vectors";
import { graphForBrowser, loadLocalGraph } from "./graph";
import { blockStructure, retrievalSource } from "./graph-retrieval";

// Five invented chunks over three row groups. c3 has no vector and c4 one of
// another width; c1 is (3, 4, 0), c2 (0, 0, 2) and c5 (0, 5, 0).
const FIXTURE = path.join(import.meta.dirname, "__fixtures__", "chunk-vectors");

const vectorOf = (vectors: { dimension: number; data: Float32Array }, index: number) =>
  Array.from(vectors.data.subarray(index * vectors.dimension, (index + 1) * vectors.dimension));

describe("chunk vectors", () => {
  it("reads every row group, aligned to the dataset's chunk order, as unit vectors", async () => {
    const vectors = await readChunkVectors(FIXTURE, ["c5", "c1", "c2", "c3", "c4", "c_unknown"]);
    expect(vectors).not.toBeNull();
    expect(vectors!.dimension).toBe(3);
    expect(vectors!.data).toBeInstanceOf(Float32Array);
    expect(vectors!.data).toHaveLength(6 * 3);
    expect(Array.from(vectors!.present)).toEqual([1, 1, 1, 0, 0, 0]);
    expect(vectors!.count).toBe(3);
    expect(vectorOf(vectors!, 0)).toEqual([0, 1, 0]);
    expect(vectorOf(vectors!, 1).map((value) => Number(value.toFixed(6)))).toEqual([0.6, 0.8, 0]);
    expect(vectorOf(vectors!, 2)).toEqual([0, 0, 1]);
    expect(vectorOf(vectors!, 3)).toEqual([0, 0, 0]);
  });

  it("reads vectors from a JSON chunks table, and none from a graph without them", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fg-chunk-vectors-"));
    try {
      expect(await readChunkVectors(directory, ["x"])).toBeNull();
      await writeFile(path.join(directory, "chunks.json"), JSON.stringify([{ id: "x", embedding: [0, 2] }, { id: "y" }]));
      const vectors = await readChunkVectors(directory, ["y", "x"]);
      expect(Array.from(vectors!.present)).toEqual([0, 1]);
      expect(vectorOf(vectors!, 1)).toEqual([0, 1]);
      await writeFile(path.join(directory, "chunks.json"), JSON.stringify([{ id: "x", content: "no vector" }]));
      expect(await readChunkVectors(directory, ["x"])).toBeNull();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

describe("what a retrieval load reads beyond the rows", () => {
  it("keeps the run's table and chart blocks beside the graph, and reads chunk vectors on demand", async () => {
    const dataset = await loadLocalGraph(FIXTURE, { purpose: "retrieval" });
    const source = retrievalSource(dataset);
    expect(source?.directory).toBe(FIXTURE);
    expect(source?.blocks.map((block) => [block.id, block.kind])).toEqual([
      ["b_table", "table"],
      ["b_chart", "chart"],
    ]);
    // Their text is in the chunks already; the structure is what they add.
    expect(source?.blocks.every((block) => !("text" in block))).toBe(true);
    expect(blockStructure(source!.blocks[0]!)).toMatchObject({ table: { header_rows: 1, rows: [["Region", "Share"], ["North", "40%"]] } });
    expect(blockStructure({ metadata: "not json" })).toBeNull();
    // The dataset itself carries no vector, nor does what the browser gets.
    expect(dataset.chunks.every((chunk) => !("embedding" in chunk))).toBe(true);
    expect(JSON.stringify(graphForBrowser(dataset))).not.toContain("embedding");
    const vectors = await chunkVectors(dataset);
    expect(vectors?.count).toBe(3);
    expect(await chunkVectors(dataset)).toBe(vectors);
  });

  it("leaves the review load as it was", async () => {
    const dataset = await loadLocalGraph(FIXTURE);
    expect(retrievalSource(dataset)).toBeUndefined();
    expect(await chunkVectors(dataset)).toBeNull();
  });
});
