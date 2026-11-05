import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, realpath, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { stringify as stringifyYaml } from "yaml";
import { cliEnvironment, runFlakegraph } from "./cli";
import { operatorProviderProfiles, sourceSettings } from "./config";
import { assertAzureSourceDestination, assertNoRequestInterpolation } from "./provider-credentials";
import type { SourceKind, SourceObject, SourceSummary } from "./protocol/schema";
import { SUPPORTED_SUFFIXES } from "./providers";

/**
 * List a bucket or container the way the pipeline itself would read it.
 *
 * The console has no object-storage client of its own; it writes the same
 * `files` settings a run would use and asks `flakegraph sources list` to name
 * what is there. The listing therefore matches what the run will ingest, and
 * credentials stay where the pipeline already finds them.
 */
/**
 * How long a listing may take before the console answers instead.
 *
 * A wrong endpoint or a credential chain that walks every provider before
 * giving up (Azure's does, with IMDS probes that time out) can hold a
 * listing for minutes; the form should say so rather than spin.
 */
export const LISTING_TIMEOUT_MS = 60_000;

export async function listRemoteObjects(
  sourceKind: SourceKind,
  source: Record<string, unknown>,
  options: { cwd: string; stateRoot: string; env?: Record<string, string>; operatorProfiles?: readonly Record<string, unknown>[]; limit?: number; timeoutMs?: number },
): Promise<SourceObject[]> {
  assertNoRequestInterpolation(source);
  const config = sourceSettings(sourceKind, source);
  assertAzureSourceDestination(config, [...await operatorProviderProfiles(), ...(options.operatorProfiles ?? [])], cliEnvironment(options.env));
  const directory = path.join(options.stateRoot, "browse");
  await mkdir(directory, { recursive: true });
  const configPath = path.join(directory, `${randomUUID()}.yaml`);
  await writeFile(configPath, stringifyYaml(config), "utf8");
  try {
    const timeoutMs = options.timeoutMs ?? LISTING_TIMEOUT_MS;
    const result = await runFlakegraph(
      ["sources", "list", "--config", configPath, "--limit", String(options.limit ?? 1_000)],
      { cwd: options.cwd, env: options.env, timeoutMs },
    );
    if (result.exitCode === null) {
      throw new Error(
        `The ${sourceKind} source did not answer within ${Math.round(timeoutMs / 1000)} s. ` +
          "Check the endpoint and the credentials the pipeline runs with.",
      );
    }
    if (result.exitCode !== 0) {
      // The provider's own diagnosis is for the operator's logs, not the browser.
      console.error(`Source listing: the ${sourceKind} source failed: ${result.stderr.trim() || result.stdout.trim()}`);
      throw new Error(`Listing the ${sourceKind} source failed. Check its operator-configured endpoint and credentials.`);
    }
    try {
      return parseListing(result.stdout);
    } catch {
      throw new Error("Source listing returned an invalid response");
    }
  } finally {
    await rm(configPath, { force: true });
  }
}

function parseListing(stdout: string): SourceObject[] {
  const start = stdout.indexOf("[");
  const end = stdout.lastIndexOf("]");
  if (start < 0 || end < start) {
    throw new Error("Source listing returned no JSON array");
  }
  const rows = JSON.parse(stdout.slice(start, end + 1)) as Array<Record<string, unknown>>;
  return rows.map((row) => ({
    uri: String(row.uri ?? ""),
    name: String(row.name ?? ""),
    sizeBytes: typeof row.size_bytes === "number" ? row.size_bytes : Number(row.size_bytes ?? 0),
    modifiedAt: typeof row.modified_at === "string" ? row.modified_at : null,
    checksum: typeof row.checksum === "string" ? row.checksum : null,
  }));
}

/** Objects a bucket or stage is counted up to; a listing that reaches it says "at least". */
export const REMOTE_SUMMARY_LIMIT = 10_000;

/** A count from a listing that stopped at `limit`. */
export function summarizeListing(objects: readonly SourceObject[], limit: number): SourceSummary {
  return {
    count: objects.length,
    sizeBytes: objects.reduce((sum, item) => sum + item.sizeBytes, 0),
    complete: objects.length < limit,
  };
}

/** Every supported file in a folder and all its subfolders, counted without being listed. */
export async function summarizeLocalObjects(inputPath: string): Promise<SourceSummary> {
  if (!existsSync(inputPath)) {
    throw new Error(`Input path does not exist: ${inputPath}`);
  }
  let count = 0;
  let sizeBytes = 0;
  for await (const file of localFiles(inputPath)) {
    if (SUPPORTED_SUFFIXES.has(path.extname(file).toLowerCase())) {
      count += 1;
      sizeBytes += (await stat(file)).size;
    }
  }
  return { count, sizeBytes, complete: true };
}

export async function listLocalObjects(inputPath: string, limit = 1_000): Promise<SourceObject[]> {
  if (!existsSync(inputPath)) {
    throw new Error(`Input path does not exist: ${inputPath}`);
  }
  const objects: SourceObject[] = [];
  for await (const file of localFiles(inputPath)) {
    if (!SUPPORTED_SUFFIXES.has(path.extname(file).toLowerCase())) {
      continue;
    }
    const info = await stat(file);
    objects.push({
      uri: pathToFileUrl(file),
      name: path.basename(file),
      sizeBytes: info.size,
      modifiedAt: info.mtime.toISOString(),
      checksum: null,
    });
    if (objects.length >= limit) {
      break;
    }
  }
  return objects;
}

/** The folder browser uploads are written to, one sub-folder per upload. */
export function uploadsRoot(stateRoot: string): string {
  return path.join(stateRoot, "uploads");
}

/**
 * Resolve a path a viewer named and require it to be one of the app's upload
 * folders.
 *
 * A hosted app lists files with its own privileges, so an arbitrary path would
 * let any viewer enumerate the container's filesystem. Symlinks are resolved
 * before the comparison, so a link inside the uploads folder cannot point out.
 */
export async function resolveUploadFolder(stateRoot: string, inputPath: string): Promise<string> {
  const requested = inputPath.trim();
  if (!requested) {
    throw new Error("Upload folder is not set");
  }
  const root = path.resolve(uploadsRoot(stateRoot));
  const resolvedRoot = existsSync(root) ? await realpath(root) : root;
  const candidate = path.resolve(root, requested);
  if (!existsSync(candidate)) {
    throw new Error(`Upload folder does not exist: ${path.basename(candidate)}`);
  }
  const resolved = await realpath(candidate);
  const relative = path.relative(resolvedRoot, resolved);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("Only the app's own upload folders can be listed here");
  }
  return resolved;
}

/** List supported files in one upload folder, refusing any other path. */
export async function listUploadedObjects(
  stateRoot: string,
  inputPath: string,
  limit = 1_000,
): Promise<SourceObject[]> {
  return listLocalObjects(await resolveUploadFolder(stateRoot, inputPath), limit);
}

async function* localFiles(inputPath: string): AsyncGenerator<string> {
  const info = await stat(inputPath);
  if (info.isFile()) {
    yield inputPath;
    return;
  }
  const stack = [inputPath];
  while (stack.length > 0) {
    const directory = stack.pop()!;
    const entries = (await readdir(directory, { withFileTypes: true })).sort((left, right) =>
      left.name.localeCompare(right.name),
    );
    for (const entry of entries) {
      const candidate = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        stack.push(candidate);
      } else if (entry.isFile()) {
        yield candidate;
      }
    }
  }
}

function pathToFileUrl(file: string): string {
  const resolved = path.resolve(file);
  return `file://${resolved.split(path.sep).map(encodeURIComponent).join("/")}`;
}
