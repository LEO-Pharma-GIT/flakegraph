import { rm } from "node:fs/promises";
import path from "node:path";
import {
  changeGraphNames,
  listRunRecords,
  runDirectory,
  unwithdrawRuns,
  type CatalogRecord,
} from "./catalog";
import { isWithin } from "./confine";
import { removeUploadedGold } from "./gold-store";
import { forgetGraphItems } from "./workspace";

/**
 * What the console itself holds about a graph, found before the graph is
 * deleted: its runs' records (every runtime's, withdrawn ones included) and
 * the upload folders its documents came from.
 */
export interface GraphFootprint {
  graphId: string;
  runIds: string[];
  /** Upload folder ids this graph read from, under `uploads/`. */
  uploads: string[];
  /** Directories under the state root holding its graph output. */
  outputs: string[];
}

export async function graphFootprint(
  stateRoot: string,
  graphId: string,
  knownRunIds: readonly string[] = [],
): Promise<GraphFootprint> {
  const records = (await listRunRecords(stateRoot, Number.MAX_SAFE_INTEGER, { includeWithdrawn: true }))
    .map(({ record }) => record)
    .filter((record) => record.graphId === graphId);
  const runIds = [...new Set([...knownRunIds, ...records.map((record) => record.runId)])];
  const uploads = new Set<string>();
  for (const record of records) {
    const folder = uploadFolderOf(stateRoot, record);
    if (folder) {
      uploads.add(folder);
    }
  }
  // Only folders that are this graph's by name: its own output directory,
  // and the copies of its fleet runs' graphs. A path a run record mentions
  // is not enough - a record may name another graph's directory.
  const outputs = [
    path.join(stateRoot, "graphs", graphId),
    ...runIds.map((runId) => path.join(stateRoot, "artifacts", runId)),
  ];
  return { graphId, runIds, uploads: [...uploads], outputs };
}

/**
 * Remove the console's own traces of a deleted graph: run records, local
 * graph copies, its name, its gold file, what the workspace kept for it, and
 * the upload folders it was built from that no other graph still uses.
 * Returns the upload folders removed, for the access store to forget.
 */
export async function removeGraphRecords(stateRoot: string, footprint: GraphFootprint): Promise<string[]> {
  const removed = await exclusiveUploads(stateRoot, footprint);
  for (const runId of footprint.runIds) {
    await rm(runDirectory(stateRoot, runId), { recursive: true, force: true });
  }
  for (const output of footprint.outputs) {
    if (isWithin(stateRoot, output) && path.resolve(output) !== path.resolve(stateRoot)) {
      await rm(output, { recursive: true, force: true });
    }
  }
  await unwithdrawRuns(stateRoot, footprint.runIds);
  await changeGraphNames(stateRoot, (names) => { delete names[footprint.graphId]; });
  await removeUploadedGold(stateRoot, footprint.graphId);
  await forgetGraphItems(footprint.graphId, footprint.runIds, stateRoot);

  for (const jobId of removed) {
    await rm(path.join(stateRoot, "uploads", jobId), { recursive: true, force: true });
  }
  return removed;
}

/** The upload folders only this graph was built from: no other graph's run read them. */
export async function exclusiveUploads(stateRoot: string, footprint: GraphFootprint): Promise<string[]> {
  const usedElsewhere = new Set(
    (await listRunRecords(stateRoot, Number.MAX_SAFE_INTEGER, { includeWithdrawn: true }))
      .filter(({ record }) => record.graphId !== footprint.graphId)
      .map(({ record }) => uploadFolderOf(stateRoot, record))
      .filter((folder): folder is string => Boolean(folder)),
  );
  return footprint.uploads.filter((jobId) => !usedElsewhere.has(jobId));
}

/** The upload folder a run read its documents from, if it read one. */
function uploadFolderOf(stateRoot: string, record: CatalogRecord): string | null {
  const source = record.source as Record<string, unknown> | null | undefined;
  const named = typeof record.sourcePath === "string" && record.sourcePath ? record.sourcePath : source?.path;
  if (typeof named !== "string" || !named) {
    return null;
  }
  const uploads = path.resolve(stateRoot, "uploads");
  const target = path.resolve(named);
  if (!isWithin(uploads, target) || target === uploads) {
    return null;
  }
  return path.relative(uploads, target).split(path.sep)[0] ?? null;
}
