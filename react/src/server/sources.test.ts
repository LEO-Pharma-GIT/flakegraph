import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { summarizeListing, summarizeLocalObjects } from "./sources";

describe("source summaries", () => {
  it("counts every supported file in a folder and its subfolders, past any listing limit", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "flakegraph-summary-"));
    await mkdir(path.join(root, "a", "b"), { recursive: true });
    const names = Array.from({ length: 1_200 }, (_, index) => path.join(root, index % 2 ? "a" : "a/b", `doc-${index}.md`));
    await Promise.all(names.map((name) => writeFile(name, "x")));
    await writeFile(path.join(root, "a", "image.xyz"), "not a document");
    expect(await summarizeLocalObjects(root)).toEqual({ count: 1_200, sizeBytes: 1_200, complete: true });
  });

  it("says a listing that reached its limit is a lower bound", () => {
    const object = { uri: "s3://b/k", name: "k", sizeBytes: 5, modifiedAt: null, checksum: null };
    expect(summarizeListing([object, object], 2)).toEqual({ count: 2, sizeBytes: 10, complete: false });
    expect(summarizeListing([object], 2)).toEqual({ count: 1, sizeBytes: 5, complete: true });
  });
});
