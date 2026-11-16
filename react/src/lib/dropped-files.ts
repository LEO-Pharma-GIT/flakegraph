/**
 * Files a person dropped or picked, with the path each had inside what they
 * chose: a dropped folder keeps its subfolders, so two `README.md` files in
 * different folders stay two documents.
 */
export interface PickedFile {
  file: File;
  /** Relative path with forward slashes, e.g. `contracts/2024/a.pdf`. */
  path: string;
}

/** Bytes one file may hold; the upload route refuses anything larger. */
export const UPLOAD_MAX_FILE_BYTES = 200 * 1024 * 1024;

/**
 * What one request carries; a larger folder goes up in several. Kept small
 * enough that each request finishes in seconds on a slow link, so progress
 * moves and a dropped connection costs one batch, not the whole folder.
 */
const BATCH_MAX_FILES = 100;
const BATCH_MAX_BYTES = 32 * 1024 * 1024;

/** Operating-system litter that is never a document. */
const LITTER = new Set(["thumbs.db", "desktop.ini"]);

/** Whether a path is a hidden file, sits in a hidden folder, or is OS litter. */
export function isIgnoredPath(path: string): boolean {
  const segments = path.split("/").filter(Boolean);
  return segments.some((segment) => segment.startsWith(".")) || LITTER.has((segments.at(-1) ?? "").toLowerCase());
}

/** Files from a file or folder picker; a folder picker reports each file's path inside it. */
export function pickedFromList(list: FileList | readonly File[]): PickedFile[] {
  return Array.from(list, (file) => ({ file, path: file.webkitRelativePath || file.name }));
}

/**
 * Everything a drop holds, walking dropped folders and all their subfolders.
 *
 * The entries are taken before the first await: a drop's items are only
 * readable while its event is being handled.
 */
export async function pickedFromDrop(data: DataTransfer): Promise<PickedFile[]> {
  const entries = Array.from(data.items ?? [])
    .filter((item) => item.kind === "file")
    .map((item) => item.webkitGetAsEntry?.() ?? null);
  if (entries.length === 0 || entries.some((entry) => entry === null)) {
    // No entry API (or a drop that is not from the file system): the flat
    // file list is all there is.
    return pickedFromList(data.files);
  }
  const picked: PickedFile[] = [];
  for (const entry of entries as FileSystemEntry[]) {
    await collect(entry, "", picked);
  }
  return picked;
}

async function collect(entry: FileSystemEntry, parent: string, into: PickedFile[]): Promise<void> {
  const path = parent ? `${parent}/${entry.name}` : entry.name;
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
    into.push({ file, path });
    return;
  }
  if (entry.isDirectory) {
    for (const child of await readAll((entry as FileSystemDirectoryEntry).createReader())) {
      await collect(child, path, into);
    }
  }
}

/** A directory reader hands entries over in pages; read until it hands over none. */
async function readAll(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  const all: FileSystemEntry[] = [];
  for (;;) {
    const page = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (page.length === 0) {
      return all;
    }
    all.push(...page);
  }
}

/** Split what was picked into what gets uploaded and what is left out, and why. */
export function sortPicked(picked: readonly PickedFile[]): {
  upload: PickedFile[];
  ignored: number;
  tooLarge: PickedFile[];
} {
  const upload: PickedFile[] = [];
  const tooLarge: PickedFile[] = [];
  let ignored = 0;
  for (const item of picked) {
    if (isIgnoredPath(item.path)) {
      ignored += 1;
    } else if (item.file.size > UPLOAD_MAX_FILE_BYTES) {
      tooLarge.push(item);
    } else {
      upload.push(item);
    }
  }
  return { upload, ignored, tooLarge };
}

/** Requests small enough for the upload route, in the order the files were found. */
export function uploadBatches(files: readonly PickedFile[]): PickedFile[][] {
  const batches: PickedFile[][] = [];
  let current: PickedFile[] = [];
  let bytes = 0;
  for (const item of files) {
    if (current.length > 0 && (current.length >= BATCH_MAX_FILES || bytes + item.file.size > BATCH_MAX_BYTES)) {
      batches.push(current);
      current = [];
      bytes = 0;
    }
    current.push(item);
    bytes += item.file.size;
  }
  if (current.length > 0) {
    batches.push(current);
  }
  return batches;
}
