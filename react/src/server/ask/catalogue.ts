import { documentDisplayName } from "@/lib/evidence";
import type { GraphDataset } from "../protocol/schema";
import { Bm25Index } from "./bm25";
import { asString } from "./score";

/**
 * The graph's documents as a catalogue: each one's name, where it sits, and the
 * card the pipeline wrote for it (what it is, what it covers, what it finds).
 *
 * Questions about the documents themselves - which of them cover a market, how
 * many surveyed a group, what each concluded - are answered from the cards
 * before any passage is read, and they decide which documents a question
 * spanning many of them reads one by one.
 */

/** The most cards one listing returns. */
export const CATALOGUE_MAX_CARDS = 80;
/** How much of one card a listing shows. */
export const CATALOGUE_CARD_CHAR_LIMIT = 1_600;

export interface CatalogueCard {
  documentId: string;
  name: string;
  location: string;
  /** The folder the document sits in, when the graph records one. */
  folder: string | null;
  card: string | null;
}

interface Catalogue {
  cards: CatalogueCard[];
  index: Bm25Index;
}

const catalogues = new WeakMap<GraphDataset, Catalogue>();

/** Every document of the graph with its card, in the graph's order. */
export function catalogueCards(dataset: GraphDataset): CatalogueCard[] {
  return catalogue(dataset).cards;
}

function catalogue(dataset: GraphDataset): Catalogue {
  const cached = catalogues.get(dataset);
  if (cached) {
    return cached;
  }
  const cards = dataset.documents.map((row) => {
    const source = asString(row.source_uri ?? row.path);
    const summary = asString(row.summary).trim();
    return {
      documentId: asString(row.id ?? row.file_id),
      name: documentDisplayName(row),
      location: asString(row.folder) || source,
      folder: asString(row.folder).trim() || null,
      card: summary || null,
    };
  });
  const built = {
    cards,
    index: new Bm25Index(cards.map((card) => `${card.name}\n${card.location}\n${card.card ?? ""}`)),
  };
  catalogues.set(dataset, built);
  return built;
}

/**
 * The cards matching a query, best first, or every card when there is no
 * query; restricted to the given documents when there are any.
 */
export function findDocuments(
  dataset: GraphDataset,
  options: { query?: string; documentIds?: readonly string[]; limit?: number } = {},
): { cards: CatalogueCard[]; total: number; matched: number } {
  const { cards, index } = catalogue(dataset);
  const allowed = options.documentIds?.length ? new Set(options.documentIds) : null;
  const limit = Math.min(options.limit ?? CATALOGUE_MAX_CARDS, CATALOGUE_MAX_CARDS);
  const eligible = (position: number) => !allowed || allowed.has(cards[position]!.documentId);
  const query = options.query?.trim();
  if (!query) {
    const listed = cards.filter((_card, position) => eligible(position));
    return { cards: listed.slice(0, limit), total: cards.length, matched: listed.length };
  }
  const ranked = index.search(query, cards.length, eligible);
  return {
    cards: ranked.slice(0, limit).map((hit) => cards[hit.position]!),
    total: cards.length,
    matched: ranked.length,
  };
}

/** Cards as the agent reads them: one heading per document, then its card. */
export function renderCards(result: { cards: CatalogueCard[]; total: number; matched: number }): string {
  const lines = [
    `${result.matched} of ${result.total} documents match; showing ${result.cards.length}.`,
  ];
  for (const card of result.cards) {
    lines.push("", `## ${card.name} (${card.documentId})`);
    if (card.location && card.location !== card.name) {
      lines.push(`Location: ${card.location}`);
    }
    lines.push(card.card ? card.card.slice(0, CATALOGUE_CARD_CHAR_LIMIT) : "(no card)");
  }
  return lines.join("\n");
}
