import path from "node:path";
import { atomicWriteJson, readJsonFile } from "../catalog";
import {
  principalKey,
  type AccessLevel,
  type GraphAccessRecord,
  type GraphGrant,
} from "./graph-access";

/**
 * Who owns each graph, who else may use it, and who has used the console.
 *
 * One file in the console's state, written whole and renamed into place, and
 * changed only through `update`, which runs one change at a time so two
 * people sharing at once cannot lose each other's grant.
 */
export interface AccessState {
  version: 1;
  graphs: Record<string, GraphAccessRecord>;
  /** People the console has seen signed in, offered when sharing. */
  people: Record<string, { lastSeenAt: string }>;
  /** Who each upload folder belongs to: its documents are theirs to build from. */
  uploads: Record<string, { owner: string; createdAt: string }>;
}

const FILE = path.join("access", "graphs.json");

export function accessFile(stateRoot: string): string {
  return path.join(stateRoot, FILE);
}

export function emptyAccess(): AccessState {
  return { version: 1, graphs: {}, people: {}, uploads: {} };
}

export async function loadAccess(stateRoot: string): Promise<AccessState> {
  return (await readJsonFile<AccessState>(accessFile(stateRoot))) ?? emptyAccess();
}

const queues = new Map<string, Promise<unknown>>();

/** Apply one change to the access state, after every change already queued. */
export async function updateAccess<T>(
  stateRoot: string,
  change: (state: AccessState) => T | Promise<T>,
): Promise<T> {
  const file = accessFile(stateRoot);
  const previous = queues.get(file) ?? Promise.resolve();
  const next = previous.catch(() => undefined).then(async () => {
    const state = await loadAccess(stateRoot);
    const result = await change(state);
    await atomicWriteJson(file, state);
    return result;
  });
  queues.set(file, next);
  try {
    return await next;
  } finally {
    if (queues.get(file) === next) {
      queues.delete(file);
    }
  }
}

export function recordFor(state: AccessState, graphId: string): GraphAccessRecord | null {
  return state.graphs[graphId] ?? null;
}

/** Make someone the owner of a graph that has none; an owned graph keeps its owner. */
export function claim(state: AccessState, graphId: string, owner: string): void {
  const record = state.graphs[graphId] ?? { owner: null, grants: [] };
  if (!principalKey(record.owner)) {
    record.owner = principalKey(owner);
  }
  state.graphs[graphId] = record;
}

/** Add a grant, or change the level of an existing one. */
export function grant(
  state: AccessState,
  graphId: string,
  principal: string,
  level: AccessLevel,
  grantedBy: string,
  now = new Date(),
): GraphGrant {
  const record = state.graphs[graphId] ?? { owner: null, grants: [] };
  const key = principalKey(principal);
  const existing = record.grants.find((item) => principalKey(item.principal) === key);
  if (existing) {
    existing.level = level;
    state.graphs[graphId] = record;
    return existing;
  }
  const added: GraphGrant = { principal: key, level, grantedBy: principalKey(grantedBy), grantedAt: now.toISOString() };
  record.grants.push(added);
  state.graphs[graphId] = record;
  return added;
}

export function revoke(state: AccessState, graphId: string, principal: string): boolean {
  const record = state.graphs[graphId];
  if (!record) {
    return false;
  }
  const key = principalKey(principal);
  const before = record.grants.length;
  record.grants = record.grants.filter((item) => principalKey(item.principal) !== key);
  return record.grants.length !== before;
}

export function forgetGraph(state: AccessState, graphId: string): void {
  delete state.graphs[graphId];
}

/** Note that someone used the console, at most once a day per person. */
export function sawPerson(state: AccessState, principal: string, now = new Date()): boolean {
  const key = principalKey(principal);
  if (!key) {
    return false;
  }
  const last = Date.parse(state.people[key]?.lastSeenAt ?? "");
  if (Number.isFinite(last) && now.getTime() - last < 24 * 60 * 60 * 1000) {
    return false;
  }
  state.people[key] = { lastSeenAt: now.toISOString() };
  return true;
}

/** Record who created an upload folder; a folder keeps its first owner. */
export function claimUpload(state: AccessState, jobId: string, owner: string, now = new Date()): void {
  if (!state.uploads[jobId]) {
    state.uploads[jobId] = { owner: principalKey(owner), createdAt: now.toISOString() };
  }
}
