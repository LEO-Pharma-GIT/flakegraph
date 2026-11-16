import { describe, expect, it } from "vitest";
import {
  buildGraphML,
  findShortestPath,
  fitCamera,
  hopNeighborhood,
  layoutGraph,
  nHopNodeIds,
  normalizeGraph,
  screenToWorld,
  worldToScreen,
  zoomAt,
} from "./graph-geometry";

describe("graph-geometry", () => {
  it("round-trips world and screen coordinates", () => {
    const camera = { x: 12, y: -4, scale: 1.5 };
    const screen = worldToScreen(camera, 20, 10, 800, 400);
    const world = screenToWorld(camera, screen.x, screen.y, 800, 400);
    expect(world.x).toBeCloseTo(20);
    expect(world.y).toBeCloseTo(10);
  });

  it("keeps the cursor world point stable while zooming", () => {
    const camera = { x: 0, y: 0, scale: 1 };
    const before = screenToWorld(camera, 220, 140, 400, 300);
    const afterCamera = zoomAt(camera, 220, 140, 400, 300, 1.25);
    const after = screenToWorld(afterCamera, 220, 140, 400, 300);
    expect(after.x).toBeCloseTo(before.x);
    expect(after.y).toBeCloseTo(before.y);
    expect(afterCamera.scale).toBeCloseTo(1.25);
  });

  it("keeps a large sparse graph finite under the force layout", () => {
    // Long edges across a sparse initial ring must retain finite coordinates
    // so the layout and camera can render every node.
    const nodes = Array.from({ length: 400 }, (_, index) => ({ id: `node_${index}`, name: `n${index}` }));
    const edges = [
      { id: "e1", source_node_id: "node_0", target_node_id: "node_200", relation_type: "x" },
      { id: "e2", source_node_id: "node_3", target_node_id: "node_150", relation_type: "x" },
    ];
    const normalized = normalizeGraph(nodes, edges);
    const laid = layoutGraph(normalized.nodes, normalized.edges, "force");
    expect(laid.every((node) => Number.isFinite(node.x) && Number.isFinite(node.y))).toBe(true);
    const camera = fitCamera(laid, 895, 430);
    expect(Number.isFinite(camera.scale)).toBe(true);
    const a = laid.find((node) => node.id === "node_0")!;
    const b = laid.find((node) => node.id === "node_200")!;
    const spread = Math.max(...laid.map((node) => Math.hypot(node.x, node.y)));
    expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeLessThan(spread);
  });

  it("marks the edges its sources only claim, to draw them dashed", () => {
    const nodes = [{ id: "a" }, { id: "b" }];
    const edges = [
      { id: "e1", source_node_id: "a", target_node_id: "b", relation_type: "TREATS", assertion: "claimed" },
      { id: "e2", source_node_id: "b", target_node_id: "a", relation_type: "TREATS", assertion: "reported" },
    ];
    expect(normalizeGraph(nodes, edges).edges.map((edge) => [edge.id, edge.claimed])).toEqual([
      ["e1", true],
      ["e2", false],
    ]);
  });

  it("fits a camera around laid-out nodes", () => {
    const camera = fitCamera(
      [
        { x: -40, y: -10 },
        { x: 40, y: 10 },
      ],
      400,
      300,
    );
    expect(camera.x).toBeCloseTo(0);
    expect(camera.y).toBeCloseTo(0);
    expect(camera.scale).toBeGreaterThan(1);
  });

  it("finds an undirected shortest path", () => {
    const path = findShortestPath(
      [
        { id: "e1", source: "a", target: "b" },
        { id: "e2", source: "b", target: "c" },
      ],
      "c",
      "a",
    );
    expect(path).not.toBeNull();
    expect([...path!.nodeIds].sort()).toEqual(["a", "b", "c"]);
    expect(path!.edgeIds.size).toBe(2);
  });

  it("walks a 1-hop neighborhood from a relation", () => {
    const result = hopNeighborhood(
      [{ id: "a", name: "A" }, { id: "b", name: "B" }, { id: "c", name: "C" }],
      [
        { id: "rel_001", source_node_id: "a", target_node_id: "b" },
        { id: "rel_002", source_node_id: "b", target_node_id: "c" },
      ],
      "rel_001",
    );
    expect(result.nodes.map((node) => node.id).sort()).toEqual(["a", "b", "c"]);
    expect(result.edges).toHaveLength(2);
  });

  it("expands n-hop ids", () => {
    const ids = nHopNodeIds(
      [
        { source: "a", target: "b" },
        { source: "b", target: "c" },
      ],
      ["a"],
      2,
    );
    expect([...ids].sort()).toEqual(["a", "b", "c"]);
  });

  it("writes GraphML that Gephi can import", () => {
    const xml = buildGraphML(
      [{ id: "n1", name: "Kano", primary_type: "PERSON" }],
      [{ id: "e1", source_node_id: "n1", target_node_id: "n1", relation_type: "MENTIONS" }],
    );
    expect(xml).toContain("<node id=\"n1\">");
    expect(xml).toContain("PERSON");
    expect(xml).toContain("MENTIONS");
  });

  it("keeps clustered community seeds apart", () => {
    const { nodes, edges } = normalizeGraph(
      [
        { id: "a", name: "A", primary_type: "PERSON" },
        { id: "b", name: "B", primary_type: "PERSON" },
        { id: "c", name: "C", primary_type: "ORG" },
        { id: "d", name: "D", primary_type: "ORG" },
      ],
      [
        { id: "e1", source: "a", target: "b" },
        { id: "e2", source: "c", target: "d" },
      ],
      new Map([
        ["a", new Set(["c1"])],
        ["b", new Set(["c1"])],
        ["c", new Set(["c2"])],
        ["d", new Set(["c2"])],
      ]),
    );
    const clustered = layoutGraph(nodes, edges, "clustered");
    const a = clustered.find((node) => node.id === "a")!;
    const c = clustered.find((node) => node.id === "c")!;
    expect(Math.hypot(a.x - c.x, a.y - c.y)).toBeGreaterThan(20);
    expect(clustered.every((node) => Number.isFinite(node.x) && Number.isFinite(node.y))).toBe(true);
  });
});
