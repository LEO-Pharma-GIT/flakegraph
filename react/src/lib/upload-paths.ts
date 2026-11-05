/** Folders a dropped path may keep; anything deeper is flattened into its last folders. */
export const UPLOAD_MAX_DEPTH = 16;

/**
 * The folders and file name an upload keeps of a picked path, one plain name
 * per level, so it can name nothing outside the upload folder. The upload
 * route stores a file there, and the console previews it there.
 */
export function uploadPathSegments(name: string): string[] {
  return name
    .replaceAll("\\", "/")
    .split("/")
    .map((segment) => segment.replace(/\0/g, "_").trim())
    .filter((segment) => segment && segment !== "." && segment !== "..")
    .slice(-UPLOAD_MAX_DEPTH);
}
