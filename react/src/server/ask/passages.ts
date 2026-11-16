import { documentNameIndex, evidenceDocumentId } from "@/lib/evidence";
import { isStructuredBlockKind, retrievalSource, type StructuredBlockKind } from "../graph-retrieval";
import type { GraphDataset } from "../protocol/schema";
import { Bm25Index } from "./bm25";
import {
  PASSAGE_CHAR_LIMIT,
  PASSAGE_DEFAULT_TOP_K,
  PASSAGE_FUSION_DEPTH,
  PASSAGE_NEIGHBOUR_CHAR_LIMIT,
  RRF_K,
} from "./constants";
import { denseRanking } from "./dense";
import { resolveAskEmbedding, type AskEmbedding } from "./embedding-model";
import { fuseRankings } from "./ranking";
import { asNumber, asString, asStringArray } from "./score";

/** Which ranking put a passage in the result: BM25, the embeddings, or both. */
export type PassageMethod = "bm25" | "dense" | "hybrid";

export interface PassageHit {
  chunkId: string;
  documentId: string;
  documentName: string;
  pageNumber: number | null;
  /** The headings the chunk sits under, outermost first. */
  sectionPath: string[];
  /** The chunk's text, with the edges of its neighbours when the search expanded it. */
  text: string;
  /** The table, chart or figure blocks the chunk holds, by kind. */
  contains: StructuredBlockKind[];
  method: PassageMethod;
}

export interface PassageSearchOptions {
  query: string;
  topK?: number;
  /** Add the end of the previous chunk and the start of the next one of the same document. */
  expand?: boolean;
  /** Only these documents; undefined for every document. */
  documentIds?: readonly string[];
  /** Only these chunks, as a perspective's scope leaves them; undefined for every chunk. */
  chunkIds?: ReadonlySet<string>;
  /** The question embedding model; resolved from the environment when omitted. */
  embedding?: AskEmbedding | null;
  abortSignal?: AbortSignal;
}

export interface PassageSearchResult {
  passages: PassageHit[];
  /** "hybrid" when dense retrieval took part, "bm25" when the search fell back to BM25 alone. */
  retrieval: "bm25" | "hybrid";
}

/**
 * Everything passage search needs about a graph's chunks, computed once per
 * loaded graph: the BM25 index, each chunk's document and its neighbours in
 * reading order.
 */
interface PassageIndex {
  bm25: Bm25Index;
  chunkIds: string[];
  /** Per chunk, the ordinal of its document id in `documentIds`. */
  documentOf: Int32Array;
  documentIds: string[];
  documentNames: string[];
  /** Per chunk, the chunk before and after it in its document, or -1. */
  previous: Int32Array;
  next: Int32Array;
  /** Block id to its kind, for the run's table, chart and figure blocks. */
  structuredBlocks: Map<string, StructuredBlockKind>;
}

// Keyed by the dataset a graph was loaded as, so a graph's index is built on
// its first search and dropped with the graph when the cache lets it go.
const indexes = new WeakMap<GraphDataset, PassageIndex>();

// A rendered markdown table's rule line, and a rendered chart's heading (see
// the Python package's domain/tables.py): how a chunk's text shows a table or
// a chart when the run's blocks were not loaded.
const TABLE_RULE = /^\|(?:\s*:?-{3,}:?\s*\|)+\s*$/m;
const CHART_HEADING = /^Chart \([^)\n]*\)/m;
const ELLIPSIS = "…";

export function passageIndex(dataset: GraphDataset): PassageIndex {
  const known = indexes.get(dataset);
  if (known) {
    return known;
  }
  const names = documentNameIndex(dataset.documents);
  const documentIds: string[] = [];
  const documentOrdinal = new Map<string, number>();
  const documentOf = new Int32Array(dataset.chunks.length);
  const texts: string[] = [];
  const chunkIds: string[] = [];
  dataset.chunks.forEach((chunk, position) => {
    const documentId = evidenceDocumentId(chunk);
    let ordinal = documentOrdinal.get(documentId);
    if (ordinal === undefined) {
      ordinal = documentIds.length;
      documentOrdinal.set(documentId, ordinal);
      documentIds.push(documentId);
    }
    documentOf[position] = ordinal;
    chunkIds.push(asString(chunk.id ?? chunk.chunk_id));
    // A passage is found by what it says, by the headings it sits under, and
    // by the name of its document, since questions often name the document
    // ("in the 2023 survey, ...") and its chunks rarely do.
    texts.push([chunkText(chunk), sectionPath(chunk).join(" "), names.get(documentId) ?? ""].join("\n"));
  });
  const { previous, next } = readingOrder(dataset.chunks, documentOf);
  const structuredBlocks = new Map<string, StructuredBlockKind>();
  for (const block of retrievalSource(dataset)?.blocks ?? []) {
    if (isStructuredBlockKind(block.kind)) {
      structuredBlocks.set(asString(block.id), block.kind);
    }
  }
  const index: PassageIndex = {
    bm25: new Bm25Index(texts),
    chunkIds,
    documentOf,
    documentIds,
    documentNames: documentIds.map((id) => names.get(id) ?? id),
    previous,
    next,
    structuredBlocks,
  };
  indexes.set(dataset, index);
  return index;
}

/**
 * Search a graph's source passages: BM25 over every chunk, fused by
 * reciprocal rank with dense retrieval when the graph and the configured
 * embedding model allow it, within the documents and chunks the caller may
 * see, and optionally widened by each passage's neighbours.
 */
export async function searchPassages(dataset: GraphDataset, options: PassageSearchOptions): Promise<PassageSearchResult> {
  const index = passageIndex(dataset);
  const allow = admission(dataset, index, options);
  const lexical = index.bm25.search(options.query, PASSAGE_FUSION_DEPTH, allow).map((item) => item.position);
  const embedding = options.embedding === undefined ? resolveAskEmbedding() : options.embedding;
  const dense = (
    await denseRanking(dataset, options.query, {
      embedding,
      limit: PASSAGE_FUSION_DEPTH,
      allow,
      abortSignal: options.abortSignal,
    })
  )?.map((item) => item.position);
  const topK = options.topK ?? PASSAGE_DEFAULT_TOP_K;
  const ranked = dense ? fuseRankings([lexical, dense], topK, RRF_K) : lexical.slice(0, topK);
  const lexicalSet = new Set(lexical);
  const denseSet = new Set(dense ?? []);
  const method = (position: number): PassageMethod =>
    lexicalSet.has(position) && denseSet.has(position) ? "hybrid" : denseSet.has(position) ? "dense" : "bm25";
  return {
    passages: buildPassages(dataset, index, ranked, method, options.expand ?? false),
    retrieval: dense ? "hybrid" : "bm25",
  };
}

/** Which chunks a search may return: those of the allowed documents, and of the allowed chunks. */
function admission(
  dataset: GraphDataset,
  index: PassageIndex,
  options: Pick<PassageSearchOptions, "documentIds" | "chunkIds">,
): ((chunk: number) => boolean) | undefined {
  let documentAllowed: Uint8Array | null = null;
  if (options.documentIds !== undefined) {
    // A document is named by its id, its file id or its document id; a chunk
    // may carry any of them.
    const wanted = new Set(options.documentIds);
    for (const document of dataset.documents) {
      const aliases = [document.id, document.document_id, document.file_id].filter((id) => id != null).map(String);
      if (aliases.some((id) => wanted.has(id))) {
        aliases.forEach((id) => wanted.add(id));
      }
    }
    documentAllowed = new Uint8Array(index.documentIds.length);
    index.documentIds.forEach((id, ordinal) => {
      if (wanted.has(id)) documentAllowed![ordinal] = 1;
    });
  }
  let chunkAllowed: Uint8Array | null = null;
  if (options.chunkIds !== undefined) {
    chunkAllowed = new Uint8Array(index.chunkIds.length);
    index.chunkIds.forEach((id, position) => {
      if (options.chunkIds!.has(id)) chunkAllowed![position] = 1;
    });
  }
  if (!documentAllowed && !chunkAllowed) {
    return undefined;
  }
  return (chunk) =>
    (!documentAllowed || documentAllowed[index.documentOf[chunk]!] === 1) && (!chunkAllowed || chunkAllowed[chunk] === 1);
}

function buildPassages(
  dataset: GraphDataset,
  index: PassageIndex,
  ranked: readonly number[],
  method: (position: number) => PassageMethod,
  expand: boolean,
): PassageHit[] {
  // A neighbour that is itself a result, or already shown beside another
  // result, is not repeated.
  const shown = new Set(ranked);
  return ranked.map((position) => {
    const chunk = dataset.chunks[position]!;
    let text = clip(chunkText(chunk), PASSAGE_CHAR_LIMIT);
    if (expand) {
      const before = index.previous[position]!;
      const after = index.next[position]!;
      const lead = before >= 0 && !shown.has(before) ? leadingContext(dataset.chunks[before]!, chunk) : "";
      const tail = after >= 0 && !shown.has(after) ? trailingContext(chunk, dataset.chunks[after]!) : "";
      if (lead) shown.add(before);
      if (tail) shown.add(after);
      text = [lead, text, tail].filter(Boolean).join("\n");
    }
    const ordinal = index.documentOf[position]!;
    const page = asNumber(chunk.page_number, Number.NaN);
    return {
      chunkId: index.chunkIds[position]!,
      documentId: index.documentIds[ordinal]!,
      documentName: index.documentNames[ordinal]!,
      pageNumber: Number.isFinite(page) ? page : null,
      sectionPath: sectionPath(chunk),
      text,
      contains: structures(chunk, index.structuredBlocks),
      method: method(position),
    };
  });
}

/**
 * The end of the chunk before a hit, without what the hit repeats.
 *
 * Chunks of one page may overlap; their offsets into the page say by how
 * much. When the previous chunk runs past where the hit starts, only its text
 * before that point is new. Offsets are trusted only when they span exactly
 * the chunk's text; otherwise the neighbour is taken whole.
 */
export function leadingContext(before: Record<string, unknown>, hit: Record<string, unknown>): string {
  const text = chunkText(before);
  const repeated = pageOverlap(before, hit);
  return tailOf(text.slice(0, Math.max(0, text.length - repeated)), PASSAGE_NEIGHBOUR_CHAR_LIMIT);
}

/** The start of the chunk after a hit, without what the hit already holds. */
export function trailingContext(hit: Record<string, unknown>, after: Record<string, unknown>): string {
  return headOf(chunkText(after).slice(pageOverlap(hit, after)), PASSAGE_NEIGHBOUR_CHAR_LIMIT);
}

/**
 * How many characters two consecutive chunks share: the end of `earlier`
 * that `later` starts with. Zero unless both are on one page and their
 * offsets span their texts exactly.
 */
function pageOverlap(earlier: Record<string, unknown>, later: Record<string, unknown>): number {
  const first = offsets(earlier);
  const second = offsets(later);
  if (!first || !second || asNumber(earlier.page_number, Number.NaN) !== asNumber(later.page_number, Number.NaN)) {
    return 0;
  }
  return Math.max(0, first.end - second.start);
}

/** A chunk's offsets into its page, when they span its text exactly. */
function offsets(chunk: Record<string, unknown>): { start: number; end: number } | null {
  const start = asNumber(chunk.start_offset, Number.NaN);
  const end = asNumber(chunk.end_offset, Number.NaN);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end - start !== chunkText(chunk).length) {
    return null;
  }
  return { start, end };
}

function structures(chunk: Record<string, unknown>, blocks: Map<string, StructuredBlockKind>): StructuredBlockKind[] {
  const kinds = new Set<StructuredBlockKind>();
  for (const id of asStringArray(chunk.block_ids)) {
    const kind = blocks.get(id);
    if (kind) kinds.add(kind);
  }
  const text = chunkText(chunk);
  if (TABLE_RULE.test(text)) kinds.add("table");
  if (CHART_HEADING.test(text)) kinds.add("chart");
  return [...kinds];
}

/** Each chunk's previous and next chunk in its document, by chunk index. */
function readingOrder(
  chunks: readonly Record<string, unknown>[],
  documentOf: Int32Array,
): { previous: Int32Array; next: Int32Array } {
  const previous = new Int32Array(chunks.length).fill(-1);
  const next = new Int32Array(chunks.length).fill(-1);
  const order = Array.from({ length: chunks.length }, (_, position) => position);
  const chunkIndex = (position: number) => asNumber(chunks[position]!.chunk_index, position);
  order.sort((left, right) => documentOf[left]! - documentOf[right]! || chunkIndex(left) - chunkIndex(right) || left - right);
  for (let at = 1; at < order.length; at += 1) {
    const left = order[at - 1]!;
    const right = order[at]!;
    if (documentOf[left] === documentOf[right]) {
      next[left] = right;
      previous[right] = left;
    }
  }
  return { previous, next };
}

export function chunkText(chunk: Record<string, unknown>): string {
  return asString(chunk.content ?? chunk.text ?? chunk.quote);
}

function sectionPath(chunk: Record<string, unknown>): string[] {
  return asStringArray(chunk.section_path);
}

function clip(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit)}${ELLIPSIS}` : text;
}

/** The last `limit` characters of a text, from a word boundary. */
function tailOf(text: string, limit: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= limit) {
    return trimmed;
  }
  const cut = trimmed.slice(trimmed.length - limit);
  const space = cut.search(/\s/);
  return `${ELLIPSIS}${space >= 0 ? cut.slice(space + 1) : cut}`;
}

/** The first `limit` characters of a text, to a word boundary. */
function headOf(text: string, limit: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= limit) {
    return trimmed;
  }
  const cut = trimmed.slice(0, limit);
  const space = cut.search(/\s\S*$/);
  return `${space > 0 ? cut.slice(0, space) : cut}${ELLIPSIS}`;
}
