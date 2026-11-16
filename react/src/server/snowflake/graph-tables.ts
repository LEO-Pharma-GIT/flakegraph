import { GRAPH_REVIEW_ROW_LIMITS } from "../protocol/schema";

/**
 * Tables whose rows a review surface renders, with how many rows of each are
 * fetched. Documents and chunks are only counted: chunk rows carry the full
 * text of the corpus.
 */
export const REVIEW_TABLE_LIMITS = {
  KG_NODE: GRAPH_REVIEW_ROW_LIMITS.nodes,
  KG_EDGE: GRAPH_REVIEW_ROW_LIMITS.edges,
  KG_COMMUNITY: GRAPH_REVIEW_ROW_LIMITS.communities,
  KG_EVIDENCE: GRAPH_REVIEW_ROW_LIMITS.evidence,
} as const;

/**
 * Rows of each table that describes a graph's documents rather than its
 * structure. A corpus of thousands of documents stays well inside this; past
 * it the full counts still say how much a review does not show.
 */
const REVIEW_RECORD_LIMIT = 10_000;

/** What a review reads per document: its record, why it went unread, and what extraction dropped. */
export const REVIEW_RECORD_TABLES = {
  documents: "KG_DOCUMENT",
  discardedWindows: "KG_DISCARDED_WINDOW",
  rejectedRecords: "KG_REJECTED_RECORD",
  failedDocuments: "KG_FAILED_DOCUMENT",
} as const;

/** One bounded page of a table a review reads whole, in a stable order. */
export function reviewRecordQuery(table: string): string {
  return `SELECT * FROM ${table} WHERE GRAPH_ID = ? ORDER BY ID LIMIT ${REVIEW_RECORD_LIMIT}`;
}

/** What each relation's sources state beyond the edge itself: the values, and the type a fallback replaced. */
export const EDGE_DETAIL_QUERY =
  "SELECT EDGE_ID, QUANTITIES, PROPOSED_RELATION_TYPE FROM KG_EDGE_OBSERVATION " +
  "WHERE GRAPH_ID = ? AND (QUANTITIES IS NOT NULL OR PROPOSED_RELATION_TYPE IS NOT NULL)";

export const COMMUNITY_FINDING_QUERY = "SELECT * FROM KG_COMMUNITY_FINDING WHERE GRAPH_ID = ? ORDER BY ID";

/** Tables counted for the summary without being fetched. */
export const COUNTED_TABLES = ["KG_NODE", "KG_EDGE", "KG_COMMUNITY", "KG_EVIDENCE", "KG_DOCUMENT", "KG_CHUNK"] as const;

/**
 * Every table that records something about one graph. Removing a graph has to
 * clear all of them, or rows survive a deletion with nothing in the application
 * still pointing at them. A test pins this to the schema that creates the tables.
 */
export const GRAPH_DATA_TABLES = [
  "KG_ASSET",
  "KG_BLOCK",
  "KG_BULK_LOAD",
  "KG_CHUNK",
  "KG_COMMUNITY",
  "KG_COMMUNITY_FINDING",
  "KG_DISCARDED_WINDOW",
  "KG_DOCUMENT",
  "KG_EDGE",
  "KG_EDGE_OBSERVATION",
  "KG_ENRICHMENT_CACHE",
  "KG_ENTITY_SOURCE",
  "KG_EVIDENCE",
  "KG_EXTRACTION_CACHE",
  "KG_EXTRACTION_TRACE",
  "KG_FAILED_DOCUMENT",
  "KG_GRAPH",
  "KG_GRAPH_ACL",
  "KG_GRAPH_METRICS",
  "KG_JOB",
  "KG_JOB_FILE",
  "KG_NODE",
  "KG_PAGE",
  "KG_REJECTED_RECORD",
  "KG_RUN_REPORT",
] as const;

/** Tables with a VECTOR embedding column, whose width must match the run's model. */
export const EMBEDDING_TABLES = new Set(["KG_NODE", "KG_EDGE", "KG_CHUNK", "KG_COMMUNITY"]);

/** Tables the queue and the explorer cannot work without. */
export const REQUIRED_TABLES = [
  "KG_GRAPH",
  "KG_GRAPH_ACL",
  "KG_JOB",
  "KG_JOB_FILE",
  "KG_NODE",
  "KG_EDGE",
  "KG_DISCARDED_WINDOW",
  "KG_REJECTED_RECORD",
  "KG_FAILED_DOCUMENT",
] as const;

export const GRAPH_METRICS_QUERY =
  "SELECT METRICS FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? ORDER BY UPDATED_AT DESC LIMIT 1";

export const RUN_REPORT_QUERY =
  "SELECT REPORT FROM KG_RUN_REPORT WHERE GRAPH_ID = ? ORDER BY UPDATED_AT DESC LIMIT 1";

/** One constant-table UNION that counts a graph's rows in each table. */
export function graphCountQuery(tables: readonly string[]): string {
  return tables
    .map((table) => `SELECT '${table}' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM ${table} WHERE GRAPH_ID = ?`)
    .join(" UNION ALL ");
}

/** Select every column a review needs, never the embedding vectors. */
export function reviewProjection(table: string): string {
  return EMBEDDING_TABLES.has(table) ? "* EXCLUDE EMBEDDING" : "*";
}

/** Select only edges whose endpoints are both in the node sample a review shows. */
export function reviewEdgeQuery(nodeLimit: number, edgeLimit: number): string {
  return (
    "WITH REVIEW_NODES AS (" +
    "SELECT ID FROM KG_NODE WHERE GRAPH_ID = ? " +
    `ORDER BY DEGREE DESC NULLS LAST, ID LIMIT ${nodeLimit}` +
    ") SELECT E.* EXCLUDE EMBEDDING FROM KG_EDGE E " +
    "JOIN REVIEW_NODES SOURCE ON SOURCE.ID = E.SOURCE_NODE_ID " +
    "JOIN REVIEW_NODES TARGET ON TARGET.ID = E.TARGET_NODE_ID " +
    "WHERE E.GRAPH_ID = ? " +
    `ORDER BY E.ID LIMIT ${edgeLimit}`
  );
}
