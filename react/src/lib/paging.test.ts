import { describe, expect, it } from "vitest";
import { filterKeys, nextPageCount, orderVersions, tailLabel } from "./paging";

describe("orderVersions", () => {
  const version = (number: number, head = false) => ({ number, head, runId: `run_${number}` });

  it("puts the head first and the rest newest first", () => {
    const rows = [version(1), version(3), version(2, true), version(4)];
    expect(orderVersions(rows).map((row) => row.number)).toEqual([2, 4, 3, 1]);
  });

  it("keeps the viewed version on the first page, right after the head", () => {
    const rows = [version(1), version(2), version(3), version(4, true)];
    expect(orderVersions(rows, "run_1").map((row) => row.number)).toEqual([4, 1, 3, 2]);
    expect(orderVersions(rows, "run_4").map((row) => row.number)).toEqual([4, 3, 2, 1]);
  });

  it("leaves the input untouched", () => {
    const rows = [version(1), version(2, true)];
    orderVersions(rows);
    expect(rows.map((row) => row.number)).toEqual([1, 2]);
  });
});

describe("filterKeys", () => {
  const keys = [
    { name: "ci-eval", preview: "fg_aaaa…1111", createdAt: "2026-01-01T00:00:00.000Z" },
    { name: "nightly", preview: "fg_bbbb…2222", createdAt: "2026-03-01T00:00:00.000Z" },
    { name: "CI deploy", preview: "fg_cccc…3333", createdAt: "2026-02-01T00:00:00.000Z" },
  ];

  it("lists newest first when nothing is searched", () => {
    expect(filterKeys(keys, "").map((key) => key.name)).toEqual(["nightly", "CI deploy", "ci-eval"]);
  });

  it("matches the name case-insensitively and the preview", () => {
    expect(filterKeys(keys, " ci ").map((key) => key.name)).toEqual(["CI deploy", "ci-eval"]);
    expect(filterKeys(keys, "2222").map((key) => key.name)).toEqual(["nightly"]);
  });
});

describe("tailLabel", () => {
  it("says how much of the log the tail leaves out", () => {
    expect(tailLabel(30, 240, "events")).toBe("Last 30 of 240 events");
  });

  it("counts plainly when everything is shown", () => {
    expect(tailLabel(12, 12, "events")).toBe("12 events");
    expect(tailLabel(1, 1, "events")).toBe("1 event");
  });
});

describe("nextPageCount", () => {
  it("is a full page while more than a page remains, then the remainder", () => {
    expect(nextPageCount(10, 35, 10)).toBe(10);
    expect(nextPageCount(30, 35, 10)).toBe(5);
    expect(nextPageCount(35, 35, 10)).toBe(0);
  });
});
