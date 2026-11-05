import { describe, expect, it } from "vitest";
import {
  CELL_PREVIEW,
  cellText,
  clipText,
  compareCells,
  countLine,
  matchesSearch,
  nextSort,
  sortRecords,
  type RecordColumn,
} from "./record-table";

const columns: RecordColumn[] = [
  { key: "name" },
  { key: "relation_type", kind: "type" },
  { key: "confidence", kind: "number" },
  { key: "source", value: (row) => `resolved:${String(row.source_node_id)}` },
];

const rows = [
  { id: "r1", name: "Judo", relation_type: "DEVELOPED_BY", confidence: 0.95, source_node_id: "judo" },
  { id: "r2", name: "aikido", relation_type: "FOUNDED_BY", confidence: 0.6, source_node_id: "aikido" },
  { id: "r3", name: "Karate", relation_type: "PART_OF", confidence: 10, source_node_id: "karate" },
  { id: "r4", name: "Kendo", relationType: "TAUGHT", confidence: 9, source_node_id: "kendo" },
];

describe("cellText", () => {
  it("reads the column field, its camelCase twin, or a resolver", () => {
    expect(cellText(rows[0]!, columns[1]!)).toBe("DEVELOPED_BY");
    expect(cellText(rows[3]!, columns[1]!)).toBe("TAUGHT");
    expect(cellText(rows[0]!, columns[3]!)).toBe("resolved:judo");
    expect(cellText({ id: "x" }, columns[0]!)).toBe("");
    expect(cellText({ name: ["a", "b"] }, columns[0]!)).toBe('["a","b"]');
  });
});

describe("matchesSearch", () => {
  it("matches any shown column without regard to case", () => {
    expect(matchesSearch(rows[0]!, columns, "developed")).toBe(true);
    expect(matchesSearch(rows[0]!, columns, "RESOLVED:JU")).toBe(true);
    expect(matchesSearch(rows[0]!, columns, "0.95")).toBe(true);
    expect(matchesSearch(rows[0]!, columns, "karate")).toBe(false);
  });

  it("matches everything on an empty or blank needle", () => {
    expect(matchesSearch(rows[1]!, columns, "")).toBe(true);
    expect(matchesSearch(rows[1]!, columns, "   ")).toBe(true);
  });

  it("ignores fields no column shows", () => {
    expect(matchesSearch(rows[0]!, columns, "r1")).toBe(false);
  });
});

describe("compareCells", () => {
  it("orders numbers by value rather than by their digits", () => {
    expect(compareCells("9", "10")).toBeLessThan(0);
    expect(compareCells("0.6", "0.95")).toBeLessThan(0);
    expect(compareCells("10", "10")).toBe(0);
  });

  it("orders text alphabetically without regard to case", () => {
    expect(compareCells("aikido", "Judo")).toBeLessThan(0);
    expect(compareCells("Karate", "aikido")).toBeGreaterThan(0);
  });
});

describe("sortRecords", () => {
  it("keeps the incoming order when nothing is sorted", () => {
    expect(sortRecords(rows, columns, null).map((row) => row.id)).toEqual(["r1", "r2", "r3", "r4"]);
    expect(sortRecords(rows, columns, { key: "missing", direction: "asc" }).map((row) => row.id)).toEqual(["r1", "r2", "r3", "r4"]);
  });

  it("sorts numeric columns by value in either direction", () => {
    expect(sortRecords(rows, columns, { key: "confidence", direction: "asc" }).map((row) => row.id)).toEqual(["r2", "r1", "r4", "r3"]);
    expect(sortRecords(rows, columns, { key: "confidence", direction: "desc" }).map((row) => row.id)).toEqual(["r3", "r4", "r1", "r2"]);
  });

  it("sorts text columns case-insensitively", () => {
    expect(sortRecords(rows, columns, { key: "name", direction: "asc" }).map((row) => row.name)).toEqual(["aikido", "Judo", "Karate", "Kendo"]);
  });

  it("sinks empty cells to the bottom in both directions and keeps ties stable", () => {
    const sparse = [
      { id: "a", name: "" },
      { id: "b", name: "Zed" },
      { id: "c", name: "" },
      { id: "d", name: "Alpha" },
    ];
    expect(sortRecords(sparse, columns, { key: "name", direction: "asc" }).map((row) => row.id)).toEqual(["d", "b", "a", "c"]);
    expect(sortRecords(sparse, columns, { key: "name", direction: "desc" }).map((row) => row.id)).toEqual(["b", "d", "a", "c"]);
  });

  it("does not mutate the rows it was given", () => {
    const copy = [...rows];
    sortRecords(rows, columns, { key: "name", direction: "desc" });
    expect(rows).toEqual(copy);
  });
});

describe("nextSort", () => {
  it("starts ascending on a new column and flips on the same one", () => {
    expect(nextSort(null, "name")).toEqual({ key: "name", direction: "asc" });
    expect(nextSort({ key: "name", direction: "asc" }, "name")).toEqual({ key: "name", direction: "desc" });
    expect(nextSort({ key: "name", direction: "desc" }, "name")).toEqual({ key: "name", direction: "asc" });
    expect(nextSort({ key: "name", direction: "desc" }, "confidence")).toEqual({ key: "confidence", direction: "asc" });
  });
});

describe("clipText", () => {
  it("leaves short text alone", () => {
    expect(clipText("short")).toEqual({ clipped: false, preview: "short" });
    expect(clipText("x".repeat(CELL_PREVIEW))).toEqual({ clipped: false, preview: "x".repeat(CELL_PREVIEW) });
  });

  it("cuts long text at the limit, trims the ragged end and marks the cut", () => {
    const text = `${"word ".repeat(30)}end`;
    const result = clipText(text, 24);
    expect(result.clipped).toBe(true);
    expect(result.preview).toBe("word word word word word…");
    expect(result.preview.length).toBeLessThanOrEqual(25);
  });
});

describe("countLine", () => {
  const entities = { one: "entity", many: "entities" };

  it("says only how many rows there are when everything is on screen", () => {
    expect(countLine(74, 74, 74, entities)).toBe("74 entities");
  });

  it("says how much of the match is on screen when paged", () => {
    expect(countLine(50, 104, 104, { one: "relation", many: "relations" })).toBe("Showing 50 of 104 relations");
  });

  it("adds the graph total when a filter or search hides rows", () => {
    expect(countLine(3, 3, 74, entities)).toBe("3 entities (of 74 total)");
    expect(countLine(50, 80, 6526, entities)).toBe("Showing 50 of 80 entities (of 6,526 total)");
  });

  it("uses the singular for one row", () => {
    expect(countLine(1, 1, 74, entities)).toBe("1 entity (of 74 total)");
    expect(countLine(1, 1, 1, entities)).toBe("1 entity");
  });
});
