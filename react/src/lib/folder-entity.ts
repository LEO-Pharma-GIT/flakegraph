import { normalizeTermName } from "@/lib/ontology-terms";
import { uploadPathSegments } from "@/lib/upload-paths";
import { normalizeLabel } from "@/server/ontology-profile";
import type { FolderEntitySelection, OntologySelection } from "@/server/protocol/schema";

/** Folders below the upload root a folder entity may be read from. */
export const MAX_FOLDER_LEVEL = 8;

/** A path's folders and file name, without empty, `.` or `..` parts, as an upload stores it. */
export function pathParts(path: string): string[] {
  return path
    .replaceAll("\\", "/")
    .split("/")
    .map((part) => part.trim())
    .filter((part) => part && part !== "." && part !== "..");
}

/** Where a picked file lands below the upload root, exactly as the upload route stores it. */
export function uploadedPath(path: string): string {
  return uploadPathSegments(path).join("/");
}

/**
 * The folder path a file is tied to at `level` folders below the root, or
 * null when it sits above that level. The file's own name is never a folder.
 */
export function folderAtLevel(location: string, level: number): string | null {
  const folders = pathParts(location).slice(0, -1);
  return folders.length >= level ? folders.slice(0, level).join("/") : null;
}

/**
 * The entity a folder path names: the folder's own name at the level, so
 * folders of one name at that level are one entity.
 */
export function folderName(folder: string): string {
  return pathParts(folder).at(-1) ?? "";
}

/** How many leading folders every path shares; a file's own name is never a folder. */
export function sharedFolderDepth(paths: readonly string[]): number {
  const folders = paths.map((path) => pathParts(path).slice(0, -1));
  if (folders.length === 0) {
    return 0;
  }
  let depth = 0;
  while (folders.every((parts) => parts.length > depth && parts[depth] === folders[0]![depth])) {
    depth += 1;
  }
  return depth;
}

/**
 * The level a folder entity reads for these files: the first folder that
 * tells them apart, below any folder they all share - as a picked or dropped
 * folder does - but never below the deepest folder a file has.
 */
export function suggestedFolderLevel(paths: readonly string[]): number {
  const deepest = Math.max(0, ...paths.map((path) => pathParts(path).length - 1));
  return Math.max(1, Math.min(sharedFolderDepth(paths) + 1, deepest, MAX_FOLDER_LEVEL));
}

/** One folder name at each level the files reach, the first in name order, by level. */
export function levelExamples(paths: readonly string[]): Map<number, string> {
  const examples = new Map<number, string>();
  for (let level = 1; level <= MAX_FOLDER_LEVEL; level += 1) {
    const names = paths
      .map((path) => folderAtLevel(path, level))
      .filter((folder): folder is string => folder !== null)
      .map(folderName)
      .sort((left, right) => left.localeCompare(right));
    if (names.length) {
      examples.set(level, names[0]!);
    }
  }
  return examples;
}

export interface FolderPreviewRow {
  path: string;
  /** The folder path at the level, or null when the file sits above it. */
  folder: string | null;
  /** The entity that folder names. */
  name: string | null;
  excluded: boolean;
}

/** Each picked file with the folder it is tied to and whether it is left out. */
export function folderPreview(paths: readonly string[], folder: FolderEntitySelection): FolderPreviewRow[] {
  const excluded = new Set(folder.excludedPaths.map(uploadedPath));
  return [...new Set(paths.map(uploadedPath))]
    .sort((left, right) => left.localeCompare(right))
    .map((path) => {
      const at = folderAtLevel(path, folder.level);
      return { path, folder: at, name: at === null ? null : folderName(at), excluded: excluded.has(path) };
    });
}

/** The entities the included files are tied to, by folder name, with how many files each holds. */
export function folderCounts(rows: readonly FolderPreviewRow[]): Array<{ folder: string; files: number }> {
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (row.name !== null && !row.excluded) {
      counts.set(row.name, (counts.get(row.name) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([folder, files]) => ({ folder, files }))
    .sort((left, right) => left.folder.localeCompare(right.folder));
}

/** Whether a document is tied to no folder: listed by its file id or by its path. */
export function folderExcluded(
  folder: FolderEntitySelection,
  document: { fileId: string; location?: string },
): boolean {
  if (folder.excludedFileIds.includes(document.fileId)) {
    return true;
  }
  const location = document.location ? uploadedPath(document.location) : null;
  return location !== null && folder.excludedPaths.some((path) => uploadedPath(path) === location);
}

/** What keeps a folder entity from being built with this vocabulary, if anything. */
export function folderEntityProblem(selection: OntologySelection): string | null {
  const folder = selection.folderEntity;
  if (!folder) {
    return null;
  }
  if (!folder.entityType) {
    return "Choose the entity type a document's folder names.";
  }
  if (!selection.entityTypes.some((term) => term.name === folder.entityType)) {
    return `The folder entity type ${folder.entityType} is not one of the entity types.`;
  }
  if (selection.relationTypes.some((term) => normalizeLabel(term.name) === normalizeLabel(folder.relationType))) {
    return `${folder.relationType} is a relation extraction reads from text; name the folder relation differently.`;
  }
  if (normalizeTermName(folder.relationType) !== folder.relationType) {
    return "The folder relation is named like a relation type: capitals, digits and underscores, as APPEARS_IN.";
  }
  return null;
}
