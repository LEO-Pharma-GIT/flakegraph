import { realpathSync } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { appEnv } from "./env";
import { invalid } from "./protocol/errors";
import type { IngestionRequest } from "./protocol/schema";
import { uploadPathSegments } from "@/lib/upload-paths";

/**
 * Identifiers a request may use to name a run, a graph or an upload.
 *
 * They become directory names under the console's state, so a separator or
 * a parent reference in one would name a directory somewhere else.
 */
export const SAFE_ID = /^[A-Za-z0-9._-]+$/;

/** The name a credential the request references must carry, and its prefix. */
export const CREDENTIAL_ENVIRONMENT_PREFIX = "KG_";
const ENVIRONMENT_NAME = /^[A-Z_][A-Z0-9_]*$/;

export function isSafeId(value: string): boolean {
  return SAFE_ID.test(value) && value !== "." && value !== "..";
}

export function assertSafeId(value: string, label: string): string {
  if (!isSafeId(value)) {
    throw invalid(`${label} may hold only letters, digits, dots, dashes and underscores: ${JSON.stringify(value)}`);
  }
  return value;
}

/** Whether `candidate` is `root` or somewhere beneath it, by path alone. */
export function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

/** Resolve existing symlinks, retaining a nonexistent leaf for generated paths. */
export function canonicalPath(value: string): string {
  const resolved = path.resolve(value);
  try {
    return realpathSync(resolved);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    const parent = path.dirname(resolved);
    return parent === resolved ? resolved : path.join(canonicalPath(parent), path.basename(resolved));
  }
}

/** Relative globs may select descendants, never a parent or an absolute path. */
export function assertSourceGlob(pattern: string): void {
  const normalized = pattern.replaceAll("\\", "/");
  if (!normalized || path.posix.isAbsolute(normalized) || /^[A-Za-z]:/.test(normalized) || normalized.split("/").includes("..")) {
    throw invalid("includeGlobs must be relative patterns without parent-directory components");
  }
}

/**
 * Resolve a path a request named and confine it to the roots this host
 * lets a request reach. A relative path is taken against `base`, the
 * checkout, so a sample pack's `data/...` names the file there.
 */
export function confinedPath(
  value: string,
  options: { roots: readonly string[]; base: string; label: string },
): string {
  const trimmed = value.trim();
  if (!trimmed) {
    throw invalid(`${options.label} is empty`);
  }
  const resolved = path.resolve(options.base, trimmed);
  if (!options.roots.some((root) => isWithin(canonicalPath(root), canonicalPath(resolved)))) {
    throw invalid(
      `${options.label} must be inside the checkout, the console's state root or FLAKEGRAPH_APP_SOURCE_ROOTS: ${trimmed}`,
    );
  }
  return resolved;
}

/** Where documents may be read from: the checkout, the console's state, and what the operator opened. */
export function sourceRoots(): string[] {
  const env = appEnv();
  return [env.repositoryRoot, env.stateRoot, ...env.sourceRoots];
}

export function defaultBaseConfigPath(): string {
  return path.join(appEnv().repositoryRoot, "configs", "app-defaults.yaml");
}

export function defaultWorkspacePath(graphId: string): string {
  return path.join(appEnv().stateRoot, "graphs", graphId);
}

/**
 * The paths a base configuration may be read from. The checkout holds the
 * shipped profiles; a run's own directory under the state root holds the
 * configuration it was built with, which a revision reads back.
 */
export function confinedBaseConfigPath(value: string | null): string {
  if (!value?.trim()) {
    return defaultBaseConfigPath();
  }
  const env = appEnv();
  return confinedPath(value, {
    roots: [path.join(env.repositoryRoot, "configs"), path.join(env.stateRoot, "runs")],
    base: env.repositoryRoot,
    label: "baseConfigPath",
  });
}

/**
 * A credential a request names must be one the operator placed in the
 * environment for the pipeline: a `KG_…` variable, which is the family the
 * pipeline reads and the only one handed to it (see `cliEnvironment`). The
 * request carries the name, never the value, and may not name what else
 * this process holds. Whether the variable is actually set is checked where
 * a process is spawned with it; a runtime that spawns none has no need.
 */
export function assertCredentialReference(name: string | null): void {
  if (!name) {
    return;
  }
  const trimmed = name.trim();
  if (!ENVIRONMENT_NAME.test(trimmed) || !trimmed.startsWith(CREDENTIAL_ENVIRONMENT_PREFIX)) {
    throw invalid(
      `Credential environment variables must be named ${CREDENTIAL_ENVIRONMENT_PREFIX}…, as the pipeline reads them: ${trimmed}`,
    );
  }
}

/**
 * The request as the runtimes may act on it: its ids fit a directory name,
 * its paths lie where this host lets a request reach, and the credentials
 * it names are ones the operator provided. What the form left blank gets
 * the host's own default, so a browser never needs to know a server path.
 */
export function confineIngestionRequest(request: IngestionRequest): IngestionRequest {
  assertSafeId(request.jobId, "jobId");
  assertSafeId(request.graphId, "graphId");
  const env = appEnv();
  const baseConfigPath = confinedBaseConfigPath(request.baseConfigPath);
  // A run writes only into its own graph's directory: a path anywhere else
  // under the state root could be another person's graph.
  const workspacePath = request.output.workspacePath.trim()
    ? confinedPath(request.output.workspacePath, {
        roots: [defaultWorkspacePath(request.graphId)],
        base: env.stateRoot,
        label: "output.workspacePath",
      })
    : defaultWorkspacePath(request.graphId);
  for (const selection of [request.ocr, request.llm, request.embedding]) {
    assertCredentialReference(selection.apiKeyEnvironmentVariable);
  }
  assertCredentialReference(request.output.snowflake?.credentialEnvironmentVariable ?? null);
  const source: Record<string, unknown> = { ...request.source };
  // A revision that adds nothing reads no source: the runtime names the
  // run's own directory in its place, so there is no path here to confine.
  const readsSource = !(request.revision && !request.revision.addDocuments);
  if (readsSource) source.kind = request.sourceKind === "local_path" ? "local" : request.sourceKind;
  if (readsSource && localSourceKind(request.sourceKind, source)) {
    for (const pattern of request.includeGlobs) assertSourceGlob(pattern);
    source.path = confinedPath(String(source.path ?? ""), {
      roots: sourceRoots(),
      base: env.repositoryRoot,
      label: "source.path",
    });
  }
  if (request.revision) {
    assertSafeId(request.revision.baseRunId, "revision.baseRunId");
  }
  return {
    ...request,
    source,
    baseConfigPath,
    output: { ...request.output, workspacePath },
  };
}

/** A source browsed before a run, confined the same way the run's will be. */
export function confineSource(source: Record<string, unknown>): Record<string, unknown> {
  const kind = String(source.kind ?? "local");
  if (kind === "local" || kind === "upload" || kind === "local_path") {
    return {
      ...source,
      path: confinedPath(String(source.path ?? ""), {
        roots: sourceRoots(),
        base: appEnv().repositoryRoot,
        label: "source.path",
      }),
    };
  }
  return source;
}

function localSourceKind(sourceKind: IngestionRequest["sourceKind"], source: Record<string, unknown>): boolean {
  if (sourceKind === "upload" || sourceKind === "local_path") {
    return true;
  }
  const kind = String(source.kind ?? "");
  return kind === "local" || kind === "upload" || kind === "local_path";
}

/**
 * Where a dropped file is kept inside its upload folder: the path it had in
 * the folder that was dropped, as ``uploadPathSegments`` keeps it.
 */
export function safeUploadPath(name: string): string {
  const segments = uploadPathSegments(name);
  return segments.length > 0 ? segments.join("/") : `file-${randomUUID()}`;
}
