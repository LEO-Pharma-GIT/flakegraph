import { BM25_B, BM25_K1 } from "./constants";
import { TopK, type Ranked } from "./ranking";
import { terms } from "./score";

// Term frequencies are stored in 16 bits; a term repeated more often than this
// in one passage scores as if it appeared this often, which BM25's saturation
// makes indistinguishable anyway.
const MAX_STORED_TERM_FREQUENCY = 0xffff;
const INITIAL_CAPACITY = 1_024;

/**
 * An Okapi BM25 index over a fixed list of texts.
 *
 * Built once and searched many times, at the scale of a large graph's chunks
 * (about a hundred thousand passages and tens of millions of postings), so the
 * postings live in flat typed arrays - compressed sparse rows, one run per
 * term - rather than in maps or arrays of objects: about six bytes a posting,
 * and nothing for the garbage collector to walk. A search touches only the
 * postings of the query's terms.
 */
export class Bm25Index {
  readonly size: number;
  private readonly vocabulary = new Map<string, number>();
  private readonly postingStart: Uint32Array;
  private readonly postingDocument: Uint32Array;
  private readonly postingFrequency: Uint16Array;
  // k1 * (1 - b + b * length / average length), per document: the part of
  // BM25's denominator that does not depend on the query.
  private readonly lengthNorm: Float32Array;
  // One accumulator per document, reused by every search. A search runs to
  // completion without yielding, so two questions never share it at once.
  private readonly accumulator: Float64Array;

  constructor(texts: readonly string[]) {
    this.size = texts.length;
    const lengths = new Uint32Array(this.size);
    // A forward index first (each document's distinct terms and their counts,
    // documents in order), then the same pairs regrouped by term. Counting
    // goes through arrays indexed by term id, marked with the document that
    // last touched them, so no per-document map is built and cleared.
    let forwardTerms = new Uint32Array(INITIAL_CAPACITY);
    let forwardFrequencies = new Uint16Array(INITIAL_CAPACITY);
    const forwardStart = new Uint32Array(this.size + 1);
    let documentFrequency = new Uint32Array(INITIAL_CAPACITY);
    let lastDocument = new Int32Array(INITIAL_CAPACITY).fill(-1);
    let frequency = new Uint32Array(INITIAL_CAPACITY);
    const distinct: number[] = [];
    let used = 0;
    let totalLength = 0;
    for (let document = 0; document < this.size; document += 1) {
      const tokens = terms(texts[document] ?? "");
      lengths[document] = tokens.length;
      totalLength += tokens.length;
      distinct.length = 0;
      for (const token of tokens) {
        let id = this.vocabulary.get(token);
        if (id === undefined) {
          id = this.vocabulary.size;
          this.vocabulary.set(token, id);
          if (id >= lastDocument.length) {
            const capacity = lastDocument.length * 2;
            documentFrequency = grow(documentFrequency, capacity);
            frequency = grow(frequency, capacity);
            const marks = new Int32Array(capacity).fill(-1);
            marks.set(lastDocument);
            lastDocument = marks;
          }
        }
        if (lastDocument[id] !== document) {
          lastDocument[id] = document;
          frequency[id] = 0;
          distinct.push(id);
        }
        frequency[id] = frequency[id]! + 1;
      }
      if (used + distinct.length > forwardTerms.length) {
        const capacity = Math.max(forwardTerms.length * 2, used + distinct.length);
        forwardTerms = grow(forwardTerms, capacity);
        forwardFrequencies = grow(forwardFrequencies, capacity);
      }
      for (const id of distinct) {
        forwardTerms[used] = id;
        forwardFrequencies[used] = Math.min(frequency[id]!, MAX_STORED_TERM_FREQUENCY);
        documentFrequency[id] = documentFrequency[id]! + 1;
        used += 1;
      }
      forwardStart[document + 1] = used;
    }
    const vocabularySize = this.vocabulary.size;
    this.postingStart = new Uint32Array(vocabularySize + 1);
    for (let id = 0; id < vocabularySize; id += 1) {
      this.postingStart[id + 1] = this.postingStart[id]! + documentFrequency[id]!;
    }
    this.postingDocument = new Uint32Array(used);
    this.postingFrequency = new Uint16Array(used);
    const cursor = this.postingStart.slice(0, vocabularySize);
    for (let document = 0; document < this.size; document += 1) {
      for (let slot = forwardStart[document]!; slot < forwardStart[document + 1]!; slot += 1) {
        const id = forwardTerms[slot]!;
        const at = cursor[id]!;
        this.postingDocument[at] = document;
        this.postingFrequency[at] = forwardFrequencies[slot]!;
        cursor[id] = at + 1;
      }
    }
    const averageLength = this.size > 0 ? totalLength / this.size : 0;
    this.lengthNorm = new Float32Array(this.size);
    for (let document = 0; document < this.size; document += 1) {
      const relative = averageLength > 0 ? lengths[document]! / averageLength : 0;
      this.lengthNorm[document] = BM25_K1 * (1 - BM25_B + BM25_B * relative);
    }
    this.accumulator = new Float64Array(this.size);
  }

  /**
   * The `limit` best documents for a query, best first, among those `allow`
   * admits. Each distinct query term counts once: a question that repeats a
   * word is not asking for it twice as hard.
   */
  search(query: string, limit: number, allow?: (document: number) => boolean): Ranked[] {
    const ids = [...new Set(terms(query))]
      .map((token) => this.vocabulary.get(token))
      .filter((id): id is number => id !== undefined);
    if (ids.length === 0 || limit <= 0) {
      return [];
    }
    const touched: number[] = [];
    for (const id of ids) {
      const start = this.postingStart[id]!;
      const end = this.postingStart[id + 1]!;
      const frequency = end - start;
      const idf = Math.log(1 + (this.size - frequency + 0.5) / (frequency + 0.5));
      for (let slot = start; slot < end; slot += 1) {
        const document = this.postingDocument[slot]!;
        const tf = this.postingFrequency[slot]!;
        if (this.accumulator[document] === 0) {
          touched.push(document);
        }
        this.accumulator[document]! += (idf * tf * (BM25_K1 + 1)) / (tf + this.lengthNorm[document]!);
      }
    }
    const top = new TopK(limit);
    for (const document of touched) {
      const score = this.accumulator[document]!;
      this.accumulator[document] = 0;
      if (score > 0 && (!allow || allow(document))) {
        top.offer(document, score);
      }
    }
    return top.ranked();
  }
}

function grow<T extends Uint32Array | Uint16Array | Int32Array>(array: T, capacity: number): T {
  const next = new (array.constructor as new (length: number) => T)(capacity);
  next.set(array);
  return next;
}
