import { describe, expect, it } from "vitest";
import {
  documentNameIndex,
  evidenceMethodLabel,
  entityDocumentIndex,
  evidenceDocumentId,
  evidenceEntityId,
  evidenceRelationId,
  isClaimed,
} from "./evidence";

describe("isClaimed", () => {
  it("is true only for a row its sources only claim", () => {
    expect(isClaimed({ assertion: "claimed" })).toBe(true);
    expect(isClaimed({ assertion: "reported" })).toBe(false);
    expect(isClaimed({})).toBe(false);
  });
});

describe("evidence rows", () => {
  it("reads the pipeline's subject columns and the gold format's explicit ones alike", () => {
    const node = { subject_id: "node_1", subject_kind: "node", file_id: "s3_file_1", quote: "q" };
    const edge = { subject_id: "edge_1", subject_kind: "edge", file_id: "s3_file_1" };
    const gold = { entity_id: "kano", relation_id: "r1", document_id: "d1" };
    expect([evidenceEntityId(node), evidenceRelationId(node), evidenceDocumentId(node)]).toEqual(["node_1", null, "s3_file_1"]);
    expect([evidenceEntityId(edge), evidenceRelationId(edge)]).toEqual([null, "edge_1"]);
    expect([evidenceEntityId(gold), evidenceRelationId(gold), evidenceDocumentId(gold)]).toEqual(["kano", "r1", "d1"]);
  });

  it("names documents by filename, then by the tail of their source", () => {
    const names = documentNameIndex([
      { id: "s3_file_1", file_id: "s3_file_1", source_uri: "s3://corpus/papers/Hinton-1981.pdf" },
      { id: "d2", filename: "judo.md", source_uri: "/tmp/x/judo.md" },
      { id: "d3" },
    ]);
    expect(names.get("s3_file_1")).toBe("Hinton-1981.pdf");
    expect(names.get("d2")).toBe("judo.md");
    expect(names.get("d3")).toBe("d3");
  });

  it("shows a URI's file name as written, not escaped; a plain path's % stays", () => {
    const names = documentNameIndex([
      { id: "a", source_uri: "file:///in/Lab%20A/M%C3%A5ling%20af%20brudstyrke.docx" },
      { id: "b", source_uri: "/in/100%25 cotton.pdf" },
      { id: "c", source_uri: "s3://corpus/broken%E0%A4%A.pdf" },
    ]);
    expect(names.get("a")).toBe("Måling af brudstyrke.docx");
    expect(names.get("b")).toBe("100%25 cotton.pdf");
    expect(names.get("c")).toBe("broken%E0%A4%A.pdf");
  });

  it("grounds an entity on its first quote's file, else on the chunk it came from", () => {
    const index = entityDocumentIndex({
      nodes: [
        { id: "node_a", source_chunk_ids: ["chunk_1"] },
        { id: "node_b", source_chunk_ids: ["chunk_2"] },
      ],
      evidence: [{ subject_id: "node_a", subject_kind: "node", file_id: "file_quoted" }],
      chunks: [{ id: "chunk_2", file_id: "file_chunked" }],
    });
    expect(index.get("node_a")).toBe("file_quoted");
    expect(index.get("node_b")).toBe("file_chunked");
  });

  it("says how each piece of evidence was found, reading one with no method as quoted", () => {
    expect(evidenceMethodLabel("llm_confirmed")).toBe("Model-confirmed, in other words");
    expect(evidenceMethodLabel("ocr_tolerant")).toBe("Matched despite OCR errors");
    expect(evidenceMethodLabel("llm_verified")).toBe(
      "Lines the model pointed at, verified with the document's opening",
    );
    expect(evidenceMethodLabel(undefined)).toBe("Quoted by the model");
    expect(evidenceMethodLabel("type_resolved")).toBe("Admitted on types other documents give");
    expect(evidenceMethodLabel("folder")).toBe("The folder the document is filed in");
  });
});
