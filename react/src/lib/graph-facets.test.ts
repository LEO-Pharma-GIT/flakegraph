import { describe, expect, it } from "vitest";
import { filterSummary, graphFacetOptions, pluralize, rankFacetOptions } from "./graph-facets";

describe("rankFacetOptions", () => {
  it("puts the largest first and breaks ties by label", () => {
    const ranked = rankFacetOptions([
      { id: "b", label: "Beta", count: 2 },
      { id: "c", label: "Alpha", count: 2 },
      { id: "a", label: "Zed", count: 9 },
    ]);
    expect(ranked.map((item) => item.id)).toEqual(["a", "c", "b"]);
  });
});

describe("graphFacetOptions", () => {
  const dataset = {
    nodes: [
      { id: "kano", primary_type: "PERSON" },
      { id: "maeda", primary_type: "PERSON" },
      { id: "judo", primary_type: "MARTIAL_ART" },
      { id: "tokyo", type: "LOCATION" },
    ],
    edges: [
      { id: "r1", source_node_id: "judo", target_node_id: "kano", relation_type: "DEVELOPED_BY" },
      { id: "r2", source_node_id: "judo", target_node_id: "tokyo", relation_type: "ORIGINATED_IN" },
      { id: "r3", source_node_id: "maeda", target_node_id: "kano", relation_type: "STUDIED_UNDER" },
      { id: "r4", source_node_id: "kano", target_node_id: "tokyo" },
    ],
    communities: [
      { id: "community_person", title: "PERSON", members: ["kano", "maeda"] },
      { id: "community_martial_art", title: "MARTIAL_ART", member_node_ids: '["judo"]' },
      { title: "orphan without id", members: ["x"] },
    ],
  };

  it("counts entity types by their entities and labels them for reading", () => {
    const { nodeTypes } = graphFacetOptions(dataset);
    expect(nodeTypes).toEqual([
      { id: "PERSON", label: "Person", count: 2 },
      { id: "LOCATION", label: "Location", count: 1 },
      { id: "MARTIAL_ART", label: "Martial Art", count: 1 },
    ]);
  });

  it("counts relation types by their edges", () => {
    const { relationTypes } = graphFacetOptions(dataset);
    expect(relationTypes.map((item) => [item.id, item.count])).toEqual([
      ["DEVELOPED_BY", 1],
      ["ORIGINATED_IN", 1],
      ["RELATED", 1],
      ["STUDIED_UNDER", 1],
    ]);
  });

  it("counts neighborhoods by member and keeps the community's own title", () => {
    const { neighborhoods } = graphFacetOptions(dataset);
    expect(neighborhoods).toEqual([
      { id: "community_person", label: "PERSON", count: 2 },
      { id: "community_martial_art", label: "MARTIAL_ART", count: 1 },
    ]);
  });
});

describe("filterSummary", () => {
  const nothing = {
    nodeTypes: [],
    relationTypes: [],
    communityIds: [],
    minimumConfidence: 0,
    includeIsolates: true,
    includeClaimed: true,
  };

  it("is empty when nothing narrows the graph", () => {
    expect(filterSummary(nothing)).toBeNull();
  });

  it("counts each active facet in the card's order", () => {
    const state = {
      nodeTypes: ["PERSON", "LOCATION"],
      relationTypes: [],
      communityIds: ["community_person"],
      minimumConfidence: 0.6,
      includeIsolates: false,
      includeClaimed: false,
    };
    expect(filterSummary(state)).toBe(
      "2 entity types · 1 neighborhood · confidence ≥ 0.60 · connected only · reported only",
    );
  });

  it("names a lone relation type in the singular", () => {
    expect(filterSummary({ ...nothing, relationTypes: ["DEVELOPED_BY"] })).toBe("1 relation type");
  });
});

describe("pluralize", () => {
  it("chooses the number's form", () => {
    expect(pluralize(1, { one: "entity", many: "entities" })).toBe("1 entity");
    expect(pluralize(1200, { one: "entity", many: "entities" })).toBe("1,200 entities");
  });
});

describe("graphStarterQuestions", () => {
  it("takes one question per neighborhood, largest first, without repeats", async () => {
    const { graphStarterQuestions } = await import("./graph-facets");
    const communities = [
      { id: "small", members: ["a"], suggested_questions: ["Who is A?", "What links A?"] },
      { id: "big", members: ["a", "b", "c"], suggested_questions: ["Who is A?", "How do B and C relate?"] },
      { id: "empty", members: ["d", "e"], suggested_questions: [] },
    ];
    expect(graphStarterQuestions(communities, 3)).toEqual(["Who is A?", "How do B and C relate?", "What links A?"]);
    expect(graphStarterQuestions([], 3)).toEqual([]);
  });
});
