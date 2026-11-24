import { documentDisplayName } from "@/lib/evidence";
import type { GraphDataset } from "../protocol/schema";
import { Bm25Index } from "./bm25";
import { documentRecords, organizationsIn, recordYear, renderRecord, type DocumentRecord, type OrganizationRole } from "./card-record";
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

/** What a catalogue query matches documents on; every given condition must hold. */
export interface CatalogueFilter {
  /** Any of these organizations, in the given role when there is one. */
  organizations?: readonly string[];
  role?: OrganizationRole;
  /** Any of these places. */
  places?: readonly string[];
  /** Dated in these years, inclusive. */
  fromYear?: number;
  toYear?: number;
  /** A sampled group whose name holds any of these words. */
  samples?: readonly string[];
  /** Words the document's type must hold. */
  documentType?: string;
  /** Words to rank and match by in name, location and card text. */
  text?: string;
  documentIds?: readonly string[];
}

export const CATALOGUE_GROUPS = ["author", "commissioner", "subject", "source", "place", "year", "type", "folder"] as const;
export type CatalogueGroup = (typeof CATALOGUE_GROUPS)[number];

export interface CatalogueMatch {
  card: CatalogueCard;
  record: DocumentRecord | null;
}

export interface CatalogueQueryResult {
  matches: CatalogueMatch[];
  total: number;
  /** Documents without a record: they can match only on text and document ids. */
  withoutRecord: number;
  groups: { value: string; documents: CatalogueMatch[] }[] | null;
}

/** A name as compared: lower case, accents and punctuation gone, spaces single. */
export function normalName(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function holdsAny(values: readonly string[], wanted: readonly string[] | undefined): boolean {
  if (!wanted?.length) {
    return true;
  }
  const have = values.map(normalName);
  return wanted.map(normalName).some((name) => name && have.some((value) => value.includes(name) || (value && name.includes(value))));
}

/**
 * The documents whose records match a filter, optionally grouped by a field:
 * who prepared them, for whom, whom they discuss or cite, where, when, what
 * kind, which folder. A document with no record matches only a filter on text
 * or ids.
 */
export function queryCatalogue(
  dataset: GraphDataset,
  filter: CatalogueFilter,
  groupBy?: CatalogueGroup,
): CatalogueQueryResult {
  const { cards, index } = catalogue(dataset);
  const recordsById = documentRecords(dataset);
  const allowed = filter.documentIds?.length ? new Set(filter.documentIds) : null;
  const structured =
    Boolean(filter.organizations?.length) ||
    Boolean(filter.places?.length) ||
    Boolean(filter.samples?.length) ||
    filter.fromYear != null ||
    filter.toYear != null ||
    Boolean(filter.documentType?.trim()) ||
    Boolean(filter.role);
  const text = filter.text?.trim();
  const textHits = text ? new Set(index.search(text, cards.length).map((hit) => hit.position)) : null;
  const matches: CatalogueMatch[] = [];
  let withoutRecord = 0;
  cards.forEach((card, position) => {
    const record = recordsById.get(card.documentId) ?? null;
    if (!record) withoutRecord += 1;
    if (allowed && !allowed.has(card.documentId)) return;
    if (textHits && !textHits.has(position)) return;
    if (structured) {
      if (!record) return;
      const organizations = record.organizations
        .filter((item) => !filter.role || item.role === filter.role)
        .map((item) => item.name);
      if (filter.role && !filter.organizations?.length && organizations.length === 0) return;
      if (!holdsAny(organizations, filter.organizations)) return;
      if (!holdsAny(record.places, filter.places)) return;
      if (!holdsAny(record.samples.map((sample) => sample.group), filter.samples)) return;
      if (filter.documentType?.trim() && !normalName(record.documentType).includes(normalName(filter.documentType))) return;
      const year = recordYear(record);
      if ((filter.fromYear != null || filter.toYear != null) && year == null) return;
      if (filter.fromYear != null && year! < filter.fromYear) return;
      if (filter.toYear != null && year! > filter.toYear) return;
    }
    matches.push({ card, record });
  });
  return { matches, total: cards.length, withoutRecord, groups: groupBy ? groupMatches(matches, groupBy) : null };
}

function groupValues(match: CatalogueMatch, groupBy: CatalogueGroup): string[] {
  const record = match.record;
  switch (groupBy) {
    case "folder":
      return [match.card.folder ?? "(no folder)"];
    case "year":
      return [record ? String(recordYear(record) ?? "(undated)") : "(no record)"];
    case "type":
      return [record?.documentType || "(no type)"];
    case "place":
      return record?.places.length ? record.places : ["(no place)"];
    default:
      return record ? organizationsIn(record, groupBy) : [];
  }
}

function groupMatches(matches: CatalogueMatch[], groupBy: CatalogueGroup) {
  // Spellings of one name - case, punctuation, a trailing legal form - fall together under the first seen.
  const groups = new Map<string, { value: string; documents: CatalogueMatch[] }>();
  for (const match of matches) {
    for (const value of new Set(groupValues(match, groupBy))) {
      const key = normalName(value).replace(/\b(ltd|limited|inc|gmbh|llc|plc|ag|sa|a s)\b/g, "").trim() || value;
      const group = groups.get(key) ?? { value, documents: [] };
      if (!group.documents.includes(match)) group.documents.push(match);
      groups.set(key, group);
    }
  }
  if (groupBy !== "folder" && groupBy !== "year" && groupBy !== "type" && groupBy !== "place") {
    mergeShortNames(groups);
  }
  return [...groups.values()].sort((a, b) => b.documents.length - a.documents.length || a.value.localeCompare(b.value));
}

/**
 * An organization named with and without its descriptor - "Acme" and "Acme
 * Research" - is one organization: a name whose words begin exactly one longer
 * name is counted under it, both spellings shown.
 */
function mergeShortNames(groups: Map<string, { value: string; documents: CatalogueMatch[] }>): void {
  const keys = [...groups.keys()].sort((a, b) => a.length - b.length);
  for (const key of keys) {
    const longer = keys.filter((other) => other !== key && other.startsWith(`${key} `) && groups.has(other));
    if (longer.length !== 1) continue;
    const into = groups.get(longer[0]!)!;
    const from = groups.get(key)!;
    for (const document of from.documents) if (!into.documents.includes(document)) into.documents.push(document);
    into.value = `${into.value} / ${from.value}`;
    groups.delete(key);
  }
}

/** A catalogue query's result as the agent reads it: the counts first, then each document's record. */
export function renderCatalogueQuery(result: CatalogueQueryResult, groupBy?: CatalogueGroup): string {
  const folders = new Set(result.matches.map((match) => match.card.folder ?? match.card.documentId));
  const lines = [
    `${result.matches.length} of ${result.total} documents match, in ${folders.size} folder${folders.size === 1 ? "" : "s"}.` +
      (result.withoutRecord ? ` ${result.withoutRecord} documents have no record and match only by text.` : ""),
  ];
  if (result.groups && groupBy) {
    lines.push("", `By ${groupBy}:`);
    for (const group of result.groups) {
      lines.push(`- ${group.value}: ${group.documents.length} (${group.documents.map((match) => match.card.name).join("; ")})`);
    }
  }
  lines.push("", "Documents:");
  for (const match of result.matches.slice(0, CATALOGUE_MAX_CARDS)) {
    const fields = match.record ? renderRecord(match.record) : "no record";
    const folder = match.card.folder ? ` [folder ${match.card.folder}]` : "";
    lines.push(`- ${match.card.name} (${match.card.documentId})${folder}: ${fields}`);
  }
  if (result.matches.length > CATALOGUE_MAX_CARDS) {
    lines.push(`... and ${result.matches.length - CATALOGUE_MAX_CARDS} more.`);
  }
  return lines.join("\n");
}
