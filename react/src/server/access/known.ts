import { listRunRecords, runDirectory, runRecordExists } from "../catalog";
import { postgresGraphExists, postgresRunExists } from "../db/client";
import { loadAccess, recordFor } from "./store";

/**
 * Whether a graph id is already somebody's, looked up everywhere a graph can
 * be recorded - the access store, every runtime's catalog records (forgotten
 * ones too), and the fleet's coordination store - not only in the runtime a
 * request happens to name. A new graph is claimed by its first submitter, so
 * an id found anywhere must not be claimable again.
 */
export async function graphIsKnown(stateRoot: string, graphId: string): Promise<boolean> {
  if (recordFor(await loadAccess(stateRoot), graphId)) {
    return true;
  }
  const records = await listRunRecords(stateRoot, Number.MAX_SAFE_INTEGER, { includeWithdrawn: true });
  if (records.some(({ record }) => record.graphId === graphId)) {
    return true;
  }
  return (await postgresGraphExists(graphId)) === true;
}

/** Whether a run id is taken, by a catalog record of any runtime or a fleet run. */
export async function runIsKnown(stateRoot: string, runId: string): Promise<boolean> {
  if (runRecordExists(runDirectory(stateRoot, runId))) {
    return true;
  }
  return (await postgresRunExists(runId)) === true;
}
