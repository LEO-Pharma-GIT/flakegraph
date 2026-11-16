import { describe, expect, it, vi } from "vitest";
import { completeAsk } from "./complete";
import { POST } from "@/app/api/ask/route";
import { lexicalAsk } from "./lexical";
import { mergeAskScope, scopeDataset } from "./scope";
import { createAskTools } from "./tools";
import type { GraphDataset } from "../protocol/schema";

vi.mock("./model", () => ({
  resolveAskModel: () => null,
  probeAskModel: async () => null,
}));

vi.mock("./load", () => ({
  loadAskRunGraph: async () => ({ dataset: mixedCorpus, snapshot: { graphId: "mixed-corpus" } }),
  perspectiveScope: async () => undefined,
}));
vi.mock("../auth", () => ({
  authorizeRequest: async () => ({ mode: "demo" }),
  mayMutate: () => true,
  refusal: () => null,
  SIGN_IN_HINT: "Test authentication required",
}));

const dataset: GraphDataset = {
  graphId: "g", nodes: [
    { id: "kept", name: "Albatross", source_chunk_ids: ["chunk-kept"] },
    { id: "excluded", name: "Zirconquasar", source_chunk_ids: ["chunk-excluded"] },
  ], edges: [], communities: [],
  documents: [{ id: "doc-kept" }, { id: "doc-excluded" }],
  chunks: [
    { id: "chunk-kept", file_id: "doc-kept", content: "Albatross is in this document." },
    { id: "chunk-excluded", file_id: "doc-excluded", content: "Zirconquasar is outside the perspective." },
  ],
  evidence: [
    { id: "ev-kept", subject_kind: "node", subject_id: "kept", chunk_id: "chunk-kept", file_id: "doc-kept", quote: "Albatross" },
    { id: "ev-excluded", subject_kind: "node", subject_id: "excluded", chunk_id: "chunk-excluded", file_id: "doc-excluded", quote: "Zirconquasar" },
  ], runReport: {}, graphMetrics: {}, discardedWindows: [], rejectedRecords: [], failedDocuments: [],
};

describe("Ask retrieval boundaries", () => {
  it("follows canonical chunk provenance when applying a perspective", () => {
    const scoped = scopeDataset(dataset, { search: "Albatross" });
    expect(scoped.nodes.map((n) => n.id)).toEqual(["kept"]);
    expect(scoped.chunks.map((c) => c.id)).toEqual(["chunk-kept"]);
    expect(scoped.evidence.map((e) => e.id)).toEqual(["ev-kept"]);
    expect(scoped.documents.map((d) => d.id)).toEqual(["doc-kept"]);
  });

  it.each([{ documentIds: ["doc-kept"] }, { search: "Albatross" }])("limits every direct tool to the selected scope %j", async (scope) => {
    const tools = createAskTools({ dataset, scope });
    const options = { toolCallId: "fixture", messages: [], context: {} };
    const documents = await tools.listDocuments.execute!({}, options);
    const excluded = await tools.readDocumentEvidence.execute!({ documentId: "doc-excluded", startIndex: 0, count: 10 }, options);
    const entities = await tools.searchEntities.execute!({ query: "Zirconquasar" }, options);
    const kept = await tools.readDocumentEvidence.execute!({ documentId: "doc-kept", startIndex: 0, count: 10 }, options);
    expect(JSON.stringify(documents)).not.toContain("doc-excluded");
    expect(JSON.stringify(excluded)).not.toContain("Zirconquasar");
    expect(JSON.stringify(entities)).not.toContain('"id":"excluded"');
    expect(JSON.stringify(kept)).toContain("Albatross");
  });
});


const selectedQuote = "Acme collaborates with Anchor in the selected document.";
const excludedFact = "ExcludedOnlySentinel";
const mixedCorpus: GraphDataset = {
  graphId: "mixed-corpus",
  documents: [
    { id: "doc-a", filename: "selected.txt", related_files: [excludedFact] },
    { id: "doc-b", filename: "excluded.txt" },
  ],
  nodes: [
    { id: "acme", name: "Acme", primary_type: "ORG", description: `Acme owns ${excludedFact}.`,
      aliases: [excludedFact], extra: excludedFact, source_chunk_ids: ["chunk-a", "chunk-b"] },
    { id: "anchor", name: "Anchor", primary_type: "ORG", description: `Anchor uses ${excludedFact}.`,
      source_chunk_ids: ["chunk-a", "chunk-b"] },
    { id: "beta", name: "Beta", primary_type: "ORG", description: excludedFact,
      source_chunk_ids: ["chunk-b"] },
  ],
  edges: [
    { id: "shared-edge", source_node_id: "acme", target_node_id: "anchor", relation_type: "WORKS_WITH",
      description: `Acme and Anchor launch ${excludedFact}.`, quantities: [{ text: excludedFact }],
      extra: excludedFact, source_chunk_ids: ["chunk-a", "chunk-b"], weight: 2, confidence: 0.9 },
    { id: "excluded-edge", source_node_id: "acme", target_node_id: "anchor", relation_type: "OWNS",
      description: excludedFact, source_chunk_ids: ["chunk-b"] },
  ],
  communities: [
    { id: "mixed-report", level: 0, member_node_ids: ["acme", "anchor", "beta"],
      title: `Acme ${excludedFact}`, summary: `Acme and Beta operate ${excludedFact}.`,
      findings: [{ summary: excludedFact, explanation: excludedFact }], extra: excludedFact },
    { id: "shared-only-report", level: 0, member_node_ids: ["acme", "anchor"],
      title: excludedFact, summary: excludedFact, findings: [{ summary: excludedFact }] },
    { id: "excluded-report", level: 0, member_node_ids: ["beta"], summary: excludedFact },
  ],
  chunks: [
    { id: "chunk-a", file_id: "doc-a", content: selectedQuote, metadata: { extraction: excludedFact } },
    { id: "chunk-b", file_id: "doc-b", content: `Beta establishes ${excludedFact}.` },
  ],
  evidence: [
    ...["acme", "anchor"].map((id) => ({
      id: `a-${id}`, subject_kind: "node", subject_id: id, file_id: "doc-a", chunk_id: "chunk-a",
      quote: selectedQuote, merged_description: excludedFact,
    })),
    { id: "a-edge", subject_kind: "edge", subject_id: "shared-edge", file_id: "doc-a",
      chunk_id: "chunk-a", quote: selectedQuote },
    { id: "b-beta", subject_kind: "node", subject_id: "beta", file_id: "doc-b",
      chunk_id: "chunk-b", quote: excludedFact },
    { id: "b-edge", subject_kind: "edge", subject_id: "shared-edge", file_id: "doc-b",
      chunk_id: "chunk-b", quote: excludedFact },
  ],
  runReport: { extraction_trace: [excludedFact] }, graphMetrics: { details: excludedFact },
  discardedWindows: [{ file_id: "doc-b", text: excludedFact }],
  rejectedRecords: [{ file_id: "doc-b", text: excludedFact }],
  failedDocuments: [{ file_id: "doc-b", error: excludedFact }],
};


describe("document-scoped corpus narratives", () => {
  it("rebuilds shared descriptions and reports from selected source evidence", () => {
    const scoped = scopeDataset(mixedCorpus, { documentIds: ["doc-a"] });
    expect(JSON.stringify(scoped)).not.toContain(excludedFact);
    expect(scoped.nodes.map((node) => node.id)).toEqual(["acme", "anchor"]);
    expect(scoped.nodes.every((node) => node.description === selectedQuote)).toBe(true);
    expect(scoped.nodes.every((node) => !node.aliases && !node.extra)).toBe(true);
    expect(scoped.nodes.map((node) => node.source_chunk_ids)).toEqual([["chunk-a"], ["chunk-a"]]);
    expect(scoped.edges.map((edge) => edge.id)).toEqual(["shared-edge"]);
    expect(scoped.edges[0]?.description).toBe(selectedQuote);
    expect(scoped.edges[0]?.quantities).toBeUndefined();
    // Ranking signals are numbers, not prose from other documents, so they stay.
    expect([scoped.edges[0]?.weight, scoped.edges[0]?.confidence]).toEqual([2, 0.9]);
    expect(scoped.communities.map((community) => community.id)).toEqual(["mixed-report", "shared-only-report"]);
    for (const community of scoped.communities) {
      expect(community.title).toBe("Acme, Anchor");
      expect(community.summary).toBe(selectedQuote);
      expect(community.member_node_ids).toEqual(["acme", "anchor"]);
      expect(community.findings).toEqual([]);
    }
    expect(scoped.chunks[0]?.content).toBe(selectedQuote);
    expect(scoped.evidence.every((row) => row.file_id === "doc-a")).toBe(true);
    expect(scoped.runReport).toEqual({});
    expect(scoped.graphMetrics).toEqual({});
    expect(mixedCorpus.nodes[0]?.description).toContain(excludedFact);
  });

  it("keeps excluded narratives out of direct tools and every graph retrieval mode", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("No network allowed"));
    const tools = createAskTools({ dataset: mixedCorpus, scope: { documentIds: ["doc-a"] } });
    const options = { toolCallId: "scoped-fixture", messages: [], context: {} };
    const outputs = await Promise.all([
      tools.searchEntities.execute!({ query: "Acme" }, options),
      tools.searchRelations.execute!({ query: "Acme" }, options),
      tools.searchCommunities.execute!({ query: "Acme" }, options),
      tools.getNeighborhood.execute!({ entityId: "acme", hops: 1 }, options),
      tools.searchEvidence.execute!({ query: "Acme" }, options),
      tools.getEntityEvidence.execute!({ entityId: "acme" }, options),
      tools.getRelationEvidence.execute!({ relationId: "shared-edge" }, options),
      tools.readDocumentEvidence.execute!({ documentId: "doc-a", startIndex: 0, count: 10 }, options),
    ]);
    for (const output of outputs) {
      expect(JSON.stringify(output)).not.toContain(excludedFact);
      expect(JSON.stringify(output)).toContain("Acme");
    }
    for (const mode of ["local", "global", "hybrid", "drift"] as const) {
      const output = await tools.searchGraph.execute!({ query: "Acme", mode }, options);
      expect(JSON.stringify(output)).not.toContain(excludedFact);
      expect(JSON.stringify(output)).toContain("Acme");
    }
    const globalTools = createAskTools({ dataset: mixedCorpus });
    const selectedInTool = await globalTools.searchGraph.execute!({
      query: "Acme", mode: "global", documentIds: ["doc-a"],
    }, options);
    expect(JSON.stringify(selectedInTool)).not.toContain(excludedFact);
    expect(JSON.stringify(selectedInTool)).toContain(selectedQuote);
    // An empty selection from the model restricts nothing, as an omitted one does.
    const unrestricted = await globalTools.searchGraph.execute!({ query: "Acme", mode: "local", documentIds: [] }, options);
    expect(JSON.stringify(unrestricted)).toContain("Acme");
    expect(JSON.stringify(unrestricted)).not.toContain("GRAPH_EMPTY_RESULT");
    const narrowed = await tools.searchGraph.execute!({ query: "Acme", mode: "local", documentIds: [] }, options);
    expect(JSON.stringify(narrowed)).not.toContain(excludedFact);
    expect(JSON.stringify(narrowed)).toContain("Acme");
    expect(fetch).not.toHaveBeenCalled();
  });

  it("applies document selection through lexical and public completion entry points", async () => {
    const answer = lexicalAsk(mixedCorpus, "Acme", "global", { documentIds: ["doc-a"] });
    expect(answer.scoped).toBe(true);
    expect(answer.summary).toContain(selectedQuote);
    expect(JSON.stringify(answer)).not.toContain(excludedFact);
    const completed = await completeAsk({
      dataset: mixedCorpus, question: "Acme", mode: "global", documentIds: ["doc-a"],
    });
    expect(completed).toEqual(answer);
    const empty = scopeDataset(mixedCorpus, { documentIds: ["missing-document"] });
    expect(empty.nodes).toEqual([]);
    expect(empty.edges).toEqual([]);
    expect(empty.communities).toEqual([]);
  });

  it("preserves whole-corpus reports and merged text when no document selector is present", () => {
    expect(scopeDataset(mixedCorpus)).toBe(mixedCorpus);
    expect(lexicalAsk(mixedCorpus, "Acme", "global").summary).toContain(excludedFact);
    expect(scopeDataset(mixedCorpus, { search: "Acme" }).nodes[0]?.description).toContain(excludedFact);
  });
});


it("keeps a disjoint document-scope intersection empty through completion", async () => {
  const scope = mergeAskScope({ documentIds: ["doc-a"] }, { documentIds: ["doc-b"] });
  expect(scope?.documentIds).toEqual([]);
  expect(scopeDataset(mixedCorpus, scope).nodes).toEqual([]);
  const answer = await completeAsk({
    dataset: mixedCorpus, question: "Acme", mode: "global",
    scope: { documentIds: ["doc-a"] }, documentIds: ["doc-b"],
  });
  expect(answer.scoped).toBe(true);
  expect(answer.usedCommunityReport).toBe(false);
  expect(answer.communityIds).toEqual([]);
  expect(JSON.stringify(answer)).not.toContain(excludedFact);
});


it("keeps document scope through the authorized HTTP Ask route", async () => {
  const fetch = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("No network allowed"));
  const response = await POST(new Request("https://example.invalid/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/x-ndjson" },
    body: JSON.stringify({
      runId: "test-run", question: "Acme", mode: "global", documentIds: ["doc-a"], format: "ndjson",
    }),
  }));
  expect(response.status).toBe(200);
  const events = (await response.text()).trim().split("\n").map((line) => JSON.parse(line));
  const answer = events.find((event) => event.type === "done")?.answer;
  expect(answer?.scoped).toBe(true);
  expect(answer?.summary).toContain(selectedQuote);
  expect(JSON.stringify(events)).not.toContain(excludedFact);
  expect(fetch).not.toHaveBeenCalled();
});
