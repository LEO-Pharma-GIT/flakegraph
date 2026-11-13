import { describe, expect, it } from "vitest";
import { Bm25Index } from "./bm25";
import { BM25_K1 } from "./constants";
import { fuseRankings, TopK } from "./ranking";

const positions = (ranked: { position: number }[]) => ranked.map((item) => item.position);

describe("BM25 index", () => {
  const index = new Bm25Index([
    "The lighthouse keeper logged every ship.", // 0
    "Every ship passed the lighthouse at night; ships signalled.", // 1
    "Penguins waddle across the ice.", // 2
    "A ship. ".repeat(40), // 3: one common word, many times
    "The quarterly report covers the EU and UK markets.", // 4
  ]);

  it("ranks a rare term above a common one", () => {
    // "penguins" occurs once in the corpus, "ship" in three documents.
    expect(positions(index.search("ship penguins", 5))[0]).toBe(2);
  });

  it("saturates repeated terms", () => {
    const ranked = index.search("ship", 5);
    expect(positions(ranked)).toEqual(expect.arrayContaining([0, 1, 3]));
    // However often a term repeats, it adds at most idf * (k1 + 1).
    const idf = Math.log(1 + (5 - 3 + 0.5) / (3 + 0.5));
    expect(Math.max(...ranked.map((item) => item.score))).toBeLessThan(idf * (BM25_K1 + 1));
  });

  it("normalises by length: the same match counts for more in a shorter passage", () => {
    const lengths = new Bm25Index(["harbour", `harbour ${"quay ".repeat(30)}`]);
    expect(positions(lengths.search("harbour", 2))).toEqual([0, 1]);
  });

  it("matches short codes and plurals as whole terms", () => {
    expect(positions(index.search("UK", 5))).toEqual([4]);
    expect(positions(index.search("lighthouses", 5))).toEqual(expect.arrayContaining([0, 1]));
    expect(index.search("ice cream", 5).map((item) => item.position)).toEqual([2]);
  });

  it("returns nothing for a query of stopwords or unknown terms", () => {
    expect(index.search("the and of", 5)).toEqual([]);
    expect(index.search("zeppelin", 5)).toEqual([]);
  });

  it("keeps only the documents a filter admits, and the best of them", () => {
    expect(positions(index.search("ship", 5, (document) => document !== 0 && document !== 1))).toEqual([3]);
    expect(index.search("ship", 1)).toHaveLength(1);
  });

  it("gives the same ranking when searched again", () => {
    expect(index.search("ship lighthouse", 5)).toEqual(index.search("ship lighthouse", 5));
  });

  it("copes with an empty corpus", () => {
    expect(new Bm25Index([]).search("anything", 5)).toEqual([]);
  });
});

describe("ranking helpers", () => {
  it("keeps the best k, ties in position order", () => {
    const top = new TopK(3);
    [[5, 0.1], [2, 0.9], [7, 0.5], [1, 0.5], [9, 0.05]].forEach(([position, score]) => top.offer(position!, score!));
    expect(positions(top.ranked())).toEqual([2, 1, 7]);
  });

  it("fuses rankings by reciprocal rank", () => {
    // At k = 60, 20 - second in both lists - scores 2/62 and beats 10 and 40,
    // first in one list each (1/61); their tie goes to the earlier position.
    expect(fuseRankings([[10, 20, 30], [40, 20, 50]], 3)).toEqual([20, 10, 40]);
    expect(fuseRankings([[1, 2, 3]], 2)).toEqual([1, 2]);
    expect(fuseRankings([], 2)).toEqual([]);
  });
});
