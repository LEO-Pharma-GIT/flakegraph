import { describe, expect, it } from "vitest";
import { isIgnoredPath, pickedFromDrop, sortPicked, UPLOAD_MAX_FILE_BYTES, uploadBatches, type PickedFile } from "./dropped-files";

function picked(path: string, size = 10): PickedFile {
  return { path, file: { size, name: path.split("/").at(-1) } as File };
}

describe("dropped files", () => {
  it("leaves out hidden files, hidden folders and OS litter", () => {
    expect(isIgnoredPath("docs/.DS_Store")).toBe(true);
    expect(isIgnoredPath("repo/.git/config")).toBe(true);
    expect(isIgnoredPath("scans/Thumbs.db")).toBe(true);
    expect(isIgnoredPath("docs/report.v2.pdf")).toBe(false);
  });

  it("says what it left out and why", () => {
    const sorted = sortPicked([picked("a/one.pdf"), picked("a/.hidden"), picked("a/huge.pdf", UPLOAD_MAX_FILE_BYTES + 1)]);
    expect(sorted.upload.map((item) => item.path)).toEqual(["a/one.pdf"]);
    expect(sorted.ignored).toBe(1);
    expect(sorted.tooLarge.map((item) => item.path)).toEqual(["a/huge.pdf"]);
  });

  it("splits a large folder into requests the route accepts", () => {
    const files = Array.from({ length: 250 }, (_, index) => picked(`f/${index}.md`));
    expect(uploadBatches(files).map((batch) => batch.length)).toEqual([100, 100, 50]);
    const heavy = [picked("a", 150 * 1024 * 1024), picked("b", 150 * 1024 * 1024)];
    expect(uploadBatches(heavy).map((batch) => batch.length)).toEqual([1, 1]);
    // Requests stay small enough to finish in seconds on a slow link.
    const pdfs = Array.from({ length: 10 }, (_, index) => picked(`r/${index}.pdf`, 10 * 1024 * 1024));
    expect(uploadBatches(pdfs).map((batch) => batch.length)).toEqual([3, 3, 3, 1]);
  });

  it("walks dropped folders and every subfolder, reading directories page by page", async () => {
    const file = (name: string) => ({
      name,
      isFile: true,
      isDirectory: false,
      file: (resolve: (value: File) => void) => resolve({ name, size: 1 } as File),
    });
    // A directory reader hands entries over in pages, then an empty page.
    const folder = (name: string, children: unknown[], pageSize = 2) => ({
      name,
      isFile: false,
      isDirectory: true,
      createReader: () => {
        let offset = 0;
        return {
          readEntries: (resolve: (value: unknown[]) => void) => {
            resolve(children.slice(offset, offset + pageSize));
            offset += pageSize;
          },
        };
      },
    });
    const tree = folder("corpus", [
      file("a.md"),
      file("b.md"),
      file("c.md"),
      folder("judo", [file("README.md"), folder("kodokan", [file("history.md")])]),
    ]);
    const data = {
      items: [
        { kind: "file", webkitGetAsEntry: () => tree },
        { kind: "file", webkitGetAsEntry: () => file("loose.pdf") },
      ],
      files: [],
    } as unknown as DataTransfer;
    expect((await pickedFromDrop(data)).map((item) => item.path)).toEqual([
      "corpus/a.md",
      "corpus/b.md",
      "corpus/c.md",
      "corpus/judo/README.md",
      "corpus/judo/kodokan/history.md",
      "loose.pdf",
    ]);
  });
});
