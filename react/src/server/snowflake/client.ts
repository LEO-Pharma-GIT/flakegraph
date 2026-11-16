import { readFile } from "node:fs/promises";
import type { Connection, ConnectionOptions } from "snowflake-sdk";
import type { SnowflakeConnectionEnv } from "../env";

export type SqlBind = string | number | boolean | null;
export type SqlRow = Record<string, unknown>;

/**
 * The statements the control plane issues, behind one seam so the runtime can be
 * exercised without an account. Rows come back with lower-case keys whatever the
 * statement: SHOW output is already lower case while SELECT aliases are upper
 * case, and reading both shapes everywhere is how a column silently goes missing.
 */
export interface SnowflakeSql {
  query(sqlText: string, binds?: readonly SqlBind[]): Promise<SqlRow[]>;
  /** Read rows one at a time, so a LIST over a large corpus is never held in memory. */
  stream(sqlText: string, binds?: readonly SqlBind[]): AsyncIterable<SqlRow>;
}

type SnowflakeModule = typeof import("snowflake-sdk");

/** How long one owner's-rights session is reused before a fresh token is read. */
const OWNER_CONNECTION_MAX_AGE_MS = 10 * 60 * 1000;
/** Snowflake error codes that mean the session itself is gone, not the statement. */
const SESSION_GONE_CODES = new Set([390111, 390112, 390114, 390318, 405501, 407002]);

/**
 * How long a console statement may wait for a warehouse slot. Job services run
 * their Cortex calls on the same warehouse, and a running extraction can fill
 * it; without a limit the console's quick reads queue behind it until the
 * ingress ends the request after about 90 seconds.
 */
const QUEUED_TIMEOUT_SECONDS = 20;
/** Snowflake's code for a statement cancelled by a statement or queue timeout. */
const STATEMENT_TIMEOUT_CODE = 630;

let sdkPromise: Promise<SnowflakeModule> | null = null;

async function loadSdk(): Promise<SnowflakeModule> {
  sdkPromise ??= import("snowflake-sdk").then((module) => {
    const sdk = ((module as unknown as { default?: SnowflakeModule }).default ?? module) as SnowflakeModule;
    // The driver otherwise writes snowflake.log into the working directory, which
    // inside App Runtime is the read-only server bundle.
    sdk.configure({
      logLevel: "WARN",
      customLogger: {
        error: (message: string) => console.error(`[snowflake-sdk] ${message}`),
        warn: (message: string) => console.warn(`[snowflake-sdk] ${message}`),
        info: () => undefined,
        debug: () => undefined,
        trace: () => undefined,
      },
    });
    return sdk;
  });
  return sdkPromise;
}

/**
 * Read the service's OAuth token. Re-read for every new connection: App Runtime
 * rotates the file, and a token captured at start-up expires under a long-lived
 * process.
 */
async function serviceToken(env: SnowflakeConnectionEnv): Promise<string> {
  try {
    return (await readFile(env.tokenPath, "utf8")).trim();
  } catch (cause) {
    throw new Error(`Snowflake session token is not readable at ${env.tokenPath}`, { cause });
  }
}

function connectionOptions(env: SnowflakeConnectionEnv, token: string): ConnectionOptions {
  if (!env.account) {
    throw new Error("SNOWFLAKE_ACCOUNT is not set; the app is not running inside Snowflake");
  }
  return {
    account: env.account,
    ...(env.host ? { accessUrl: `https://${env.host}` } : {}),
    authenticator: "OAUTH",
    token,
    ...(env.database ? { database: env.database } : {}),
    ...(env.schema ? { schema: env.schema } : {}),
    ...(env.warehouse ? { warehouse: env.warehouse } : {}),
    application: "FlakeGraph",
  };
}

async function openConnection(options: ConnectionOptions): Promise<Connection> {
  const sdk = await loadSdk();
  const connection = sdk.createConnection(options);
  await new Promise<void>((resolve, reject) => {
    connection.connect((error) => (error ? reject(error) : resolve()));
  });
  await execute(connection, `ALTER SESSION SET STATEMENT_QUEUED_TIMEOUT_IN_SECONDS = ${QUEUED_TIMEOUT_SECONDS}`);
  return connection;
}

/** Restates a queue timeout as the reason behind it, since the raw message names no warehouse. */
function explained(error: Error): Error {
  if (Number((error as { code?: unknown }).code) !== STATEMENT_TIMEOUT_CODE) {
    return error;
  }
  return new Error(
    `The warehouse stayed busy for ${QUEUED_TIMEOUT_SECONDS} seconds, usually with a running job's model calls. Try again in a moment.`,
    { cause: error },
  );
}

function closeConnection(connection: Connection): void {
  try {
    connection.destroy(() => undefined);
  } catch {
    // Already gone; nothing to release.
  }
}

function execute(connection: Connection, sqlText: string, binds?: readonly SqlBind[]): Promise<SqlRow[]> {
  return new Promise((resolve, reject) => {
    connection.execute({
      sqlText,
      ...(binds && binds.length > 0 ? { binds: [...binds] } : {}),
      rowMode: "object",
      complete: (error, _statement, rows) => {
        if (error) {
          reject(explained(error));
          return;
        }
        resolve((rows ?? []).map((row) => normalizeRow(row as Record<string, unknown>)));
      },
    });
  });
}

async function* streamRows(
  connection: Connection,
  sqlText: string,
  binds?: readonly SqlBind[],
): AsyncIterable<SqlRow> {
  const statement = connection.execute({
    sqlText,
    ...(binds && binds.length > 0 ? { binds: [...binds] } : {}),
    rowMode: "object",
    streamResult: true,
  });
  for await (const row of statement.streamRows()) {
    yield normalizeRow(row as Record<string, unknown>);
  }
}

function sessionGone(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const code = Number((error as { code?: unknown }).code);
  if (SESSION_GONE_CODES.has(code)) {
    return true;
  }
  const message = String((error as { message?: unknown }).message ?? "");
  return /session (?:token )?(?:has )?(?:expired|no longer exists)|not (?:connected|up)|connection.*terminated/i.test(
    message,
  );
}

interface PooledConnection {
  connection: Connection;
  openedAt: number;
}

/**
 * One owner's-rights session shared by every request. Opening a session is a
 * login round trip, so it is reused, and replaced on a schedule or as soon as
 * Snowflake reports it gone.
 */
export class SdkSnowflakeSql implements SnowflakeSql {
  private pooled: Promise<PooledConnection> | null = null;

  constructor(private readonly env: SnowflakeConnectionEnv) {}

  async query(sqlText: string, binds?: readonly SqlBind[]): Promise<SqlRow[]> {
    const { connection } = await this.connection();
    try {
      return await execute(connection, sqlText, binds);
    } catch (error) {
      if (!sessionGone(error)) {
        throw error;
      }
      this.discard(connection);
      const fresh = await this.connection();
      return execute(fresh.connection, sqlText, binds);
    }
  }

  async *stream(sqlText: string, binds?: readonly SqlBind[]): AsyncIterable<SqlRow> {
    const { connection } = await this.connection();
    yield* streamRows(connection, sqlText, binds);
  }

  private async connection(): Promise<PooledConnection> {
    const pending = this.pooled;
    const current = pending ? await pending.catch(() => null) : null;
    if (current && Date.now() - current.openedAt < OWNER_CONNECTION_MAX_AGE_MS && current.connection.isUp()) {
      return current;
    }
    // Another request replaced the session while this one waited; use theirs
    // rather than open a second.
    if (this.pooled && this.pooled !== pending) {
      return this.connection();
    }
    if (current) {
      this.discard(current.connection);
    }
    const opening = (async () => ({
      connection: await openConnection(connectionOptions(this.env, await serviceToken(this.env))),
      openedAt: Date.now(),
    }))();
    this.pooled = opening;
    opening.catch(() => {
      if (this.pooled === opening) {
        this.pooled = null;
      }
    });
    return opening;
  }

  private discard(connection: Connection): void {
    this.pooled = null;
    closeConnection(connection);
  }
}

let shared: SdkSnowflakeSql | null = null;

/** The process-wide owner's-rights client for the hosted app. */
export function sharedSnowflakeSql(env: SnowflakeConnectionEnv): SdkSnowflakeSql {
  shared ??= new SdkSnowflakeSql(env);
  return shared;
}

/**
 * Ask Snowflake which roles the person behind this request holds.
 *
 * The app's own session runs as the service's role, so CURRENT_ROLE there names
 * the app rather than the viewer. A caller's-rights session, authenticated with
 * the service token joined to the viewer's request token, answers as the viewer.
 */
export async function callerAvailableRoles(env: SnowflakeConnectionEnv, callerToken: string): Promise<string[]> {
  const connection = await openConnection(
    connectionOptions(env, `${await serviceToken(env)}.${callerToken.trim()}`),
  );
  try {
    const rows = await execute(connection, "SELECT CURRENT_AVAILABLE_ROLES() AS ROLES");
    return parseRoleList(rows[0]?.roles);
  } finally {
    closeConnection(connection);
  }
}

/** Read CURRENT_AVAILABLE_ROLES output, which arrives as JSON text or a parsed array. */
export function parseRoleList(value: unknown): string[] {
  let items: unknown = value;
  if (typeof value === "string") {
    try {
      items = JSON.parse(value);
    } catch {
      items = value.split(",");
    }
  }
  if (!Array.isArray(items)) {
    return [];
  }
  return normalizeRoles(items);
}

/** Strip identifier quotes and upper-case, so a role compares as Snowflake names it. */
export function normalizeRoles(items: readonly unknown[]): string[] {
  const roles = new Set<string>();
  for (const item of items) {
    const role = String(item ?? "").trim().replace(/^"+|"+$/g, "").trim().toUpperCase();
    if (role) {
      roles.add(role);
    }
  }
  return [...roles].sort();
}

/**
 * Lower-case every key and turn driver values into portable JSON data.
 *
 * Timestamps arrive as Date objects whose own toJSON prints Snowflake's session
 * format, so they are converted to ISO explicitly.
 */
export function normalizeRow(row: Record<string, unknown>): SqlRow {
  return Object.fromEntries(Object.entries(row).map(([key, value]) => [key.toLowerCase(), portableValue(value)]));
}

function portableValue(value: unknown): unknown {
  if (value === null || value === undefined) {
    return null;
  }
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value.toISOString();
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  if (Buffer.isBuffer(value)) {
    return value.toString("hex");
  }
  if (Array.isArray(value)) {
    return value.map(portableValue);
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, portableValue(item)]),
    );
  }
  return value;
}
