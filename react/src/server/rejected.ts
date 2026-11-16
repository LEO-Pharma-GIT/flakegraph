import { documentNameIndex } from "@/lib/evidence";
import type { GraphDataset } from "./protocol/schema";

/**
 * Records listed on the wire before the list is cut. Totals and the reasons
 * stay exact; the JSON download of the graph holds every record.
 */
export const REJECTED_ROWS_LIMIT = 5000;
/** Broken type rules the report names. */
export const BROKEN_RULES_LIMIT = 25;

export type RejectedKind = "entity" | "relation";

export interface RejectedRow {
  id: string;
  fileId: string;
  document: string;
  page: number | null;
  kind: RejectedKind;
  reason: string;
  /** What the model stated, readable: "Granulator (EQUIPMENT) USES_PROCESS wet granulation (PROCESS_STEP)". */
  statement: string;
  quote: string;
}

/** A relation type rule the model's statements broke, with how often and one example. */
export interface BrokenTypeRule {
  sourceType: string;
  relationType: string;
  targetType: string;
  count: number;
  example: string;
}

/**
 * Everything the model stated that validation did not keep, and why. A
 * rejection can be right - the model is sometimes wrong - so these are the
 * candidates for what the graph may be missing, not a verdict on it.
 */
export interface RejectedRecords {
  records: number;
  entities: number;
  relations: number;
  documents: number;
  documentsTotal: number;
  reasons: Record<string, number>;
  brokenRules: BrokenTypeRule[];
  rows: RejectedRow[];
  /** Records beyond the listed rows; the counts above include them. */
  omitted: number;
}

export function rejectedRecords(dataset: GraphDataset): RejectedRecords {
  const names = documentNameIndex(dataset.documents);
  const reasons: Record<string, number> = {};
  const rules = new Map<string, BrokenTypeRule>();
  const documents = new Set<string>();
  const rows: RejectedRow[] = [];
  let entities = 0;
  for (const raw of dataset.rejectedRecords) {
    const kind: RejectedKind = String(raw.kind) === "relation" ? "relation" : "entity";
    const reason = String(raw.reason ?? "rejected");
    const fileId = String(raw.file_id ?? raw.document_id ?? "");
    const quote = String(raw.quote ?? "");
    entities += kind === "entity" ? 1 : 0;
    documents.add(fileId);
    const key = `${kind}:${reason}`;
    reasons[key] = (reasons[key] ?? 0) + 1;
    if (kind === "relation" && reason === "domain_or_range_violation") {
      const rule = [raw.source_type, raw.name, raw.target_type].map((value) => String(value ?? "")).join("\u001f");
      const current = rules.get(rule);
      if (current) {
        current.count += 1;
      } else {
        rules.set(rule, {
          sourceType: String(raw.source_type ?? ""),
          relationType: String(raw.name ?? ""),
          targetType: String(raw.target_type ?? ""),
          count: 1,
          example: quote,
        });
      }
    }
    rows.push({
      id: String(raw.id ?? `${fileId}-${rows.length}`),
      fileId,
      document: names.get(fileId) ?? fileId,
      page: raw.page_number === null || raw.page_number === undefined ? null : Number(raw.page_number),
      kind,
      reason,
      statement: statement(raw, kind),
      quote,
    });
  }
  rows.sort(
    (left, right) =>
      left.document.localeCompare(right.document) || (left.page ?? 0) - (right.page ?? 0) || left.kind.localeCompare(right.kind),
  );
  return {
    records: rows.length,
    entities,
    relations: rows.length - entities,
    documents: documents.size,
    documentsTotal: dataset.documents.length,
    reasons,
    brokenRules: [...rules.values()]
      .sort(
        (left, right) =>
          right.count - left.count ||
          left.sourceType.localeCompare(right.sourceType) ||
          left.relationType.localeCompare(right.relationType),
      )
      .slice(0, BROKEN_RULES_LIMIT),
    rows: rows.slice(0, REJECTED_ROWS_LIMIT),
    omitted: Math.max(0, rows.length - REJECTED_ROWS_LIMIT),
  };
}

function statement(raw: Record<string, unknown>, kind: RejectedKind): string {
  const typed = (name: unknown, type: unknown) => (type ? `${String(name ?? "")} (${String(type)})` : String(name ?? ""));
  if (kind === "entity") {
    return typed(raw.name, raw.type);
  }
  return `${typed(raw.source, raw.source_type)} ${String(raw.name ?? "")} ${typed(raw.target, raw.target_type)}`;
}
