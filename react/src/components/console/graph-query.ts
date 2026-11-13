/**
 * A finished run's graph never changes: a revision is a new run. Going back to
 * one is instant for a while, and only a person's own browser pays to hold it.
 */
export const FINISHED_GRAPH_QUERY = { staleTime: Infinity, gcTime: 15 * 60_000 } as const;
