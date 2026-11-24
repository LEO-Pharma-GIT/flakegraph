/**
 * A document's catalogue card as typed fields: when it was written, who
 * prepared it and for whom, which organizations it discusses or cites, where
 * and whom its work covered. The pipeline keeps the card twice on each
 * document row - as summary text and, in `card`, as this record - so a
 * question about the documents themselves (which ones an agency prepared, how
 * many date from a year, which surveyed a group) is answered by filtering and
 * counting records rather than by reading every document.
 *
 * A graph built before cards carried a record has no `card` column; its
 * documents have no record and are matched by their summary text alone.
 */

import type { GraphDataset } from "../protocol/schema";
import { asString } from "./score";

export const ORGANIZATION_ROLES = ["author", "commissioner", "subject", "source", "other"] as const;
export type OrganizationRole = (typeof ORGANIZATION_ROLES)[number];

export interface RecordOrganization {
  name: string;
  role: OrganizationRole;
}

export interface RecordSample {
  group: string;
  count: number | null;
}

export interface DocumentRecord {
  title: string;
  documentType: string;
  /** When the document itself was written, issued or presented: YYYY, YYYY-MM or YYYY-MM-DD, or "". */
  date: string;
  period: string;
  places: string[];
  organizations: RecordOrganization[];
  topics: string[];
  samples: RecordSample[];
}

const records = new WeakMap<GraphDataset, Map<string, DocumentRecord>>();

/** Each document's record by document id; documents without one are absent. */
export function documentRecords(dataset: GraphDataset): Map<string, DocumentRecord> {
  const cached = records.get(dataset);
  if (cached) {
    return cached;
  }
  const byDocument = new Map<string, DocumentRecord>();
  for (const row of dataset.documents) {
    const record = parseRecord(row.card);
    if (record) {
      byDocument.set(asString(row.id ?? row.file_id), record);
    }
  }
  records.set(dataset, byDocument);
  return byDocument;
}

/** A card record from its JSON text or object; null when there is none or it is not one. */
export function parseRecord(value: unknown): DocumentRecord | null {
  let raw: unknown = value;
  if (typeof value === "string") {
    if (!value.trim()) {
      return null;
    }
    try {
      raw = JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const card = raw as Record<string, unknown>;
  const strings = (items: unknown) =>
    Array.isArray(items) ? items.map((item) => asString(item).trim()).filter(Boolean) : [];
  const organizations = Array.isArray(card.organizations)
    ? card.organizations.flatMap((item): RecordOrganization[] => {
        if (typeof item === "string") {
          return item.trim() ? [{ name: item.trim(), role: "other" }] : [];
        }
        const entry = (item ?? {}) as Record<string, unknown>;
        const name = asString(entry.name).trim();
        const role = asString(entry.role) as OrganizationRole;
        return name ? [{ name, role: ORGANIZATION_ROLES.includes(role) ? role : "other" }] : [];
      })
    : [];
  const samples = Array.isArray(card.samples)
    ? card.samples.flatMap((item): RecordSample[] => {
        const entry = (item ?? {}) as Record<string, unknown>;
        const group = asString(entry.group).trim();
        const count = typeof entry.count === "number" && Number.isFinite(entry.count) ? entry.count : null;
        return group ? [{ group, count }] : [];
      })
    : [];
  return {
    title: asString(card.title).trim(),
    documentType: asString(card.document_type).trim(),
    date: normalDate(asString(card.date)),
    period: asString(card.period).trim(),
    places: strings(card.places),
    organizations,
    topics: strings(card.topics),
    samples,
  };
}

/** A date as YYYY, YYYY-MM or YYYY-MM-DD, or "" when it is none of these. */
function normalDate(value: string): string {
  const match = /^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?/.exec(value.trim());
  if (!match) {
    return "";
  }
  return [match[1], match[2], match[3]].filter(Boolean).join("-");
}

/** The year a record's date names, or null. */
export function recordYear(record: DocumentRecord): number | null {
  const year = Number.parseInt(record.date.slice(0, 4), 10);
  return Number.isFinite(year) ? year : null;
}

/** The organizations of a record in one role. */
export function organizationsIn(record: DocumentRecord, role: OrganizationRole): string[] {
  return record.organizations.filter((item) => item.role === role).map((item) => item.name);
}

/** A record's fields on one line, for a listing or a reader's notes. */
export function renderRecord(record: DocumentRecord): string {
  const parts: string[] = [];
  if (record.date) parts.push(`dated ${record.date}`);
  if (record.documentType) parts.push(record.documentType);
  const authors = organizationsIn(record, "author");
  if (authors.length) parts.push(`prepared by ${authors.join(", ")}`);
  const commissioners = organizationsIn(record, "commissioner");
  if (commissioners.length) parts.push(`for ${commissioners.join(", ")}`);
  const sources = organizationsIn(record, "source");
  if (sources.length) parts.push(`cites ${sources.join(", ")}`);
  if (record.places.length) parts.push(`covers ${record.places.join(", ")}`);
  if (record.samples.length) {
    parts.push(
      `sample: ${record.samples.map((sample) => (sample.count != null ? `${sample.group} (${sample.count})` : sample.group)).join(", ")}`,
    );
  }
  return parts.join("; ");
}
