/**
 * Passage search by words and meaning inside Snowflake, through a Cortex
 * Search service over the graph tables' chunks (`KG_CHUNK`).
 *
 * One service serves every graph of a schema: chunks are replaced in place
 * when a graph is published again, and the service's refresh follows them. A
 * query is filtered to one graph by its `GRAPH_ID` attribute, and to the
 * documents a reader asks about by `DOCUMENT_ID`. The service is created by
 * `flakegraph snowflake deploy` (cortex_search_service_sql in the Python
 * package), which also turns on the change tracking it needs.
 */

import type { ExternalSearch } from "../graph-retrieval";
import type { SnowflakeSql } from "./client";

/** How many document ids one query may filter by; more are filtered after the search. */
const MAX_FILTERED_DOCUMENTS = 200;

/** A SQL string literal: Cortex Search reads its arguments as constants. */
function literal(value: string): string {
  return `'${value.replace(/\\/g, "\\\\").replace(/'/g, "''")}'`;
}

/** The SEARCH_PREVIEW request for a query within one graph and, optionally, some of its documents. */
export function cortexSearchRequest(
  query: string,
  graphId: string,
  options: { limit: number; documentIds?: readonly string[] },
): Record<string, unknown> {
  const graph = { "@eq": { GRAPH_ID: graphId } };
  const documents =
    options.documentIds?.length && options.documentIds.length <= MAX_FILTERED_DOCUMENTS
      ? { "@or": options.documentIds.map((id) => ({ "@eq": { DOCUMENT_ID: id } })) }
      : null;
  return {
    query,
    columns: ["ID", "DOCUMENT_ID"],
    filter: documents ? { "@and": [graph, documents] } : graph,
    limit: options.limit,
  };
}

/** The SQL that runs one Cortex Search query. */
export function cortexSearchSql(service: string, request: Record<string, unknown>): string {
  return `SELECT SNOWFLAKE.CORTEX.SEARCH_PREVIEW(${literal(service)}, ${literal(JSON.stringify(request))}) AS RESULT`;
}

/** A graph's passage search through a Cortex Search service, returning chunk ids best first. */
export function cortexSearch(sql: SnowflakeSql, service: string, graphId: string): ExternalSearch {
  return async (query, options) => {
    const rows = await sql.query(cortexSearchSql(service, cortexSearchRequest(query, graphId, options)));
    const raw = rows[0]?.result;
    const parsed = typeof raw === "string" ? (JSON.parse(raw) as unknown) : raw;
    const results = (parsed as { results?: { ID?: unknown; DOCUMENT_ID?: unknown }[] } | null)?.results ?? [];
    const wanted = options.documentIds?.length ? new Set(options.documentIds) : null;
    return results
      .filter((result) => !wanted || wanted.has(String(result.DOCUMENT_ID ?? "")))
      .map((result) => String(result.ID ?? ""))
      .filter(Boolean);
  };
}
