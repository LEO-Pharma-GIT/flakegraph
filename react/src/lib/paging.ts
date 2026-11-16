/**
 * Ordering and labelling for console lists that grow with a graph or a
 * long-lived deployment. Each list pages at a size that fits a screen and
 * says how much it is not showing; these helpers keep that wording and the
 * orderings the same across surfaces.
 */

/**
 * Versions read head first, then the one being viewed, then newest first:
 * the head is what the fleet serves, and the viewed one must not fall off
 * the first page of a long list.
 */
export function orderVersions<T extends { number: number; head: boolean; runId: string }>(
  rows: readonly T[],
  viewingRunId?: string,
): T[] {
  const rank = (row: T) => (row.head ? 0 : row.runId === viewingRunId ? 1 : 2);
  return [...rows].sort((left, right) => rank(left) - rank(right) || right.number - left.number);
}

/** Keys newest first, narrowed to the ones whose name or preview contains the search. */
export function filterKeys<T extends { name: string; preview: string; createdAt: string }>(
  keys: readonly T[],
  search: string,
): T[] {
  const needle = search.trim().toLowerCase();
  return keys
    .filter((key) => !needle || `${key.name} ${key.preview}`.toLowerCase().includes(needle))
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt));
}

/** "Last 30 of 240 events" when the list is a tail, "12 events" when it is everything. */
export function tailLabel(shown: number, total: number, noun: string): string {
  const unit = total === 1 ? noun.replace(/s$/, "") : noun;
  return shown < total
    ? `Last ${shown.toLocaleString()} of ${total.toLocaleString()} ${unit}`
    : `${total.toLocaleString()} ${unit}`;
}

/** How many more a "Show more" click reveals: a full page, or whatever is left. */
export function nextPageCount(shown: number, total: number, pageSize: number): number {
  return Math.max(0, Math.min(pageSize, total - shown));
}
