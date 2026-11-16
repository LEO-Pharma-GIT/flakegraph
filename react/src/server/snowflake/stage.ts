import { createHash } from "node:crypto";
import { copyFile, link, mkdir, mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { SourceObject } from "../protocol/schema";
import { SUPPORTED_SUFFIXES } from "../providers";
import type { SnowflakeSql, SqlRow } from "./client";

const SAFE_STAGE = /^[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$/;
const SAFE_STAGE_PREFIX = /^[A-Za-z0-9_./=$+:%-]*$/;
export const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_$]*$/;
/** What a stage path may not hold if the worker is to fetch it: it refuses anything else. */
const STAGE_PATH_UNSAFE = /[^A-Za-z0-9_.=$-]/g;

/** The stage folder every app upload lives under, one sub-folder per viewer. */
export const UPLOAD_NAMESPACE = "app";
/** The internal stage uploaded documents are put in. */
export const UPLOAD_STAGE = "KG_DOCS";

/**
 * Validate a stage identifier and prefix before interpolating them into LIST,
 * PUT or REMOVE, none of which accept a bound stage.
 */
export function stageLocation(stage: string, prefix?: string | null): string {
  const normalized = stage.trim().replace(/^@+/, "").replace(/\/+$/, "");
  const separator = normalized.indexOf("/");
  const stageName = separator === -1 ? normalized : normalized.slice(0, separator);
  const stagePrefix = separator === -1 ? "" : normalized.slice(separator + 1);
  if (!SAFE_STAGE.test(stageName)) {
    throw new Error("Snowflake stage must be an identifier, not arbitrary SQL");
  }
  const suffix = [stagePrefix, (prefix ?? "").replace(/^\/+|\/+$/g, "")].filter(Boolean).join("/");
  if (suffix && (suffix.split("/").includes("..") || !SAFE_STAGE_PREFIX.test(suffix))) {
    throw new Error("Snowflake stage prefix contains unsafe path characters");
  }
  return suffix ? `@${stageName}/${suffix}` : `@${stageName}`;
}

/**
 * Return the path inside the stage that a stage/prefix pair addresses.
 *
 * A stage identifier may carry a path of its own, and it is joined to the prefix
 * before the statement runs. Authorisation has to read the result, not either
 * field alone.
 */
export function stageRelativePath(stage: string, prefix?: string | null): string {
  const normalized = stage.trim().replace(/^@+/, "").replace(/\/+$/, "");
  const separator = normalized.indexOf("/");
  const embedded = separator === -1 ? "" : normalized.slice(separator + 1);
  return [embedded, (prefix ?? "").replace(/^\/+|\/+$/g, "")].filter(Boolean).join("/");
}

/** Return a stage as @DATABASE.SCHEMA.STAGE, qualifying a bare name. */
export function qualifiedStage(stage: string, database: string | null, schema: string | null): string {
  const normalized = stage.trim().replace(/^@+/, "");
  if (!normalized) {
    return "";
  }
  if (normalized.split(".").length === 3) {
    return `@${normalized}`;
  }
  if (database && schema) {
    return `@${database}.${schema}.${normalized}`;
  }
  return `@${normalized}`;
}

/** Strip the stage namespace from LIST output, leaving the path the worker reads. */
export function relativeStageName(rawName: string, stage: string): string {
  const name = rawName.replace(/^@+/, "");
  const stageName = stage.replace(/^@+/, "").replace(/\/+$/, "");
  if (name.startsWith(`${stageName}/`)) {
    return name.slice(stageName.length + 1);
  }
  const separator = name.indexOf("/");
  return separator === -1 ? name : name.slice(separator + 1);
}

/** Normalise one LIST row, discarding objects the pipeline cannot read. */
export function sourceObjectFromListRow(row: SqlRow): SourceObject | null {
  const name = String(row.name ?? "");
  if (!name || !SUPPORTED_SUFFIXES.has(path.extname(name).toLowerCase())) {
    return null;
  }
  return {
    uri: `@${name.replace(/^@+/, "")}`,
    name,
    sizeBytes: Number(row.size ?? 0) || 0,
    modifiedAt: String(row.last_modified ?? "") || null,
    checksum: String(row.md5 ?? row.etag ?? "") || null,
  };
}

/** Reproduce the processing package's deterministic identifier contract. */
export function stableId(prefix: string, ...parts: readonly unknown[]): string {
  const joined = parts.map((part) => String(part)).join("\x1f");
  return `${prefix}_${createHash("sha256").update(joined).digest("hex").slice(0, 32)}`;
}

export type SubmissionRow = [
  id: string,
  jobId: string,
  graphId: string,
  fileId: string,
  sourceUri: string,
  blobPath: string,
  checksum: string,
];

/** One durable queue row, keyed so that resubmitting the same object is idempotent. */
export function submissionRow(
  jobId: string,
  graphId: string,
  stage: string,
  item: SourceObject,
): SubmissionRow {
  const checksum = item.checksum || stableId("stage_file_checksum", item.uri);
  const fileId = stableId("stage_file", item.uri, checksum);
  return [
    stableId("snowflake_job_file", jobId, fileId),
    jobId,
    graphId,
    fileId,
    item.uri,
    relativeStageName(item.name, stage),
    checksum,
  ];
}

/**
 * Put every file in one local folder, subfolders included, into a stage folder.
 *
 * This is the only place the app moves document bytes into Snowflake. A PUT
 * writes straight to the stage's cloud storage, so it needs network egress from
 * the app's container to that storage; if that turns out to be blocked, this is
 * the function to replace.
 *
 * A PUT takes one folder's files and cannot name what they land as, while the
 * worker reads only stage paths in `STAGE_PATH_UNSAFE`'s complement. So the
 * upload is laid out again under names the worker can read, and each of its
 * folders is put in turn.
 *
 * OVERWRITE is set and every row must report UPLOADED: a SKIPPED row means the
 * stage kept an older object of the same name, and the run would read that one.
 */
export async function putFolderToStage(
  sql: SnowflakeSql,
  directory: string,
  stage: string,
  prefix: string,
): Promise<number> {
  const source = path.resolve(directory);
  const readable = await mkdtemp(path.join(tmpdir(), "flakegraph-put-"));
  try {
    let put = 0;
    for (const relative of await linkUnderStageNames(source, readable)) {
      const folder = path.join(readable, relative);
      if (/['\\]/.test(folder)) {
        throw new Error("Upload folder path contains characters a PUT statement cannot carry");
      }
      const target = stageLocation(stage, [prefix, relative].filter(Boolean).join("/"));
      const rows = await sql.query(`PUT 'file://${folder}/*' ${target} AUTO_COMPRESS = FALSE OVERWRITE = TRUE`);
      validateUploadRows(rows, path.basename(source));
      put += rows.length;
    }
    return put;
  } finally {
    await rm(readable, { recursive: true, force: true });
  }
}

/**
 * Mirror a folder's files under `target` with every path segment one the worker
 * reads. Links rather than copies, since an upload can be large and the bytes do
 * not change. Returns each folder that holds files, relative to `target`.
 */
async function linkUnderStageNames(source: string, target: string, relative = ""): Promise<string[]> {
  const entries = await readdir(source, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name));
  const taken = new Set<string>();
  const folders: string[] = [];
  let holdsFiles = false;
  for (const entry of entries) {
    const name = uniqueStageName(entry.name, taken);
    const from = path.join(source, entry.name);
    const to = path.join(target, name);
    if (entry.isDirectory()) {
      await mkdir(to);
      folders.push(...(await linkUnderStageNames(from, to, relative ? `${relative}/${name}` : name)));
    } else if (entry.isFile()) {
      await link(from, to).catch(() => copyFile(from, to));
      holdsFiles = true;
    }
  }
  return holdsFiles ? [relative, ...folders] : folders;
}

/** One name in a folder of the stage, kept apart from any other that reads the same. */
function uniqueStageName(name: string, taken: Set<string>): string {
  const readable = name.replace(STAGE_PATH_UNSAFE, "_");
  const extension = path.extname(readable);
  const stem = readable.slice(0, readable.length - extension.length);
  let candidate = readable;
  for (let attempt = 2; taken.has(candidate); attempt += 1) {
    candidate = `${stem}-${attempt}${extension}`;
  }
  taken.add(candidate);
  return candidate;
}

/** Require every PUT row to report a fresh upload, never SKIPPED. */
export function validateUploadRows(rows: readonly SqlRow[], objectName: string): void {
  if (rows.length === 0) {
    throw new Error(`Snowflake upload returned no status for ${objectName}`);
  }
  const rejected = rows
    .map((row) => String(row.status ?? "").toUpperCase())
    .filter((status) => status !== "UPLOADED")
    .map((status) => status || "UNKNOWN");
  if (rejected.length > 0) {
    throw new Error(`Snowflake did not overwrite ${objectName}: ${rejected.join(", ")}`);
  }
}
