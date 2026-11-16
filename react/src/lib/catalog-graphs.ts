import { isActiveStatus, isSuccessStatus, type RunSnapshot } from "@/server/protocol/schema";

/** One graph in the catalog: the run its row shows, and every run it has. */
export interface CatalogGraph {
  graphId: string;
  /** The run the row shows and opens. */
  run: RunSnapshot;
  runIds: string[];
}

/**
 * The catalog as graphs rather than runs: a graph's versions and retried
 * attempts are one row, which is also what a delete removes. The row shows
 * a run still under way, so progress stays visible; otherwise the newest
 * successful version; otherwise the newest attempt. Graphs keep the order of
 * their newest run, as the runs arrive newest first.
 */
export function catalogGraphs(runs: readonly RunSnapshot[]): CatalogGraph[] {
  const byGraph = new Map<string, RunSnapshot[]>();
  for (const run of runs) {
    const list = byGraph.get(run.graphId);
    if (list) {
      list.push(run);
    } else {
      byGraph.set(run.graphId, [run]);
    }
  }
  return [...byGraph.entries()].map(([graphId, list]) => ({
    graphId,
    run:
      list.find((run) => isActiveStatus(run.status)) ??
      list.find((run) => isSuccessStatus(run.status)) ??
      list[0]!,
    runIds: list.map((run) => run.runId),
  }));
}

/**
 * The graphs a question can be put to: each graph's newest successful
 * version, in the order of their newest run. A graph with no finished
 * version is left out.
 */
export function askableGraphs(runs: readonly RunSnapshot[]): RunSnapshot[] {
  return catalogGraphs(runs.filter((run) => isSuccessStatus(run.status))).map((graph) => graph.run);
}
