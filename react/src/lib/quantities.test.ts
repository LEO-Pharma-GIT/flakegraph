import { describe, expect, it } from "vitest";
import { edgeQuantities } from "@/server/graph";
import { quantityLine } from "@/lib/evidence";

describe("relation values", () => {
  it("gathers each edge's distinct values from its observations", () => {
    const byEdge = edgeQuantities([
      { edge_id: "e1", quantities: [{ text: "≤ 0.5 %", kind: "limit" }] },
      { edge_id: "e1", quantities: [{ text: "≤ 0.5 %", kind: "limit" }, { text: "0.1 %", kind: "result" }] },
      { edge_id: "e2", quantities: [] },
    ]);
    expect(byEdge.get("e1")).toEqual([
      { text: "≤ 0.5 %", kind: "limit" },
      { text: "0.1 %", kind: "result" },
    ]);
    expect(byEdge.has("e2")).toBe(false);
  });

  it("reads values as one line, adding a unit only when the text lacks it", () => {
    expect(
      quantityLine({
        quantities: [
          { text: "≤ 0.5 %", kind: "limit", unit: "%" },
          { text: "25", kind: "result", unit: "mPa·s" },
        ],
      }),
    ).toBe("limit: ≤ 0.5 % · result: 25 mPa·s");
    expect(quantityLine({})).toBe("");
  });
});
