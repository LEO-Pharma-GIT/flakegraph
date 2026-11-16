import { describe, expect, it } from "vitest";
import { formatProvenance } from "@/lib/utils";
import { documentProvenance, documentStatusesFromTasks, withProvenance, withSkippedFiles } from "./documents";

const source = (filename: string) => ({ input_file: { filename, source_uri: `s3://corpus/${filename}` } });

function task(
  stage: string,
  scopeId: string,
  status: string,
  extra: Partial<{ attempts: number; lastError: unknown; sourceMetadata: unknown; unreadable: unknown }> = {},
) {
  return { stage, scopeId, status, attempts: 1, lastError: null, sourceMetadata: null, ...extra };
}

describe("documentStatusesFromTasks", () => {
  it("records where each document sits below its source root", () => {
    const [status] = documentStatusesFromTasks([
      task("prepare_document", "f1", "succeeded", {
        sourceMetadata: { input_file: { filename: "report.pdf", location: "Northwind Pricing Study/report.pdf" } },
      }),
      task("compact_document", "f1", "succeeded"),
    ]);
    expect(status).toMatchObject({ fileId: "f1", name: "report.pdf", location: "Northwind Pricing Study/report.pdf" });
    expect(documentStatusesFromTasks([task("prepare_document", "f2", "queued")])[0]).not.toHaveProperty("location");
  });

  it("names each document and reads its phase from the durable stages", () => {
    const statuses = documentStatusesFromTasks([
      task("prepare_document", "f1", "succeeded", { sourceMetadata: source("b.pdf") }),
      task("extract_document_context", "f1", "succeeded"),
      task("compact_document", "f1", "succeeded"),
      task("prepare_document", "f2", "succeeded", { sourceMetadata: source("a.pdf") }),
      task("extract_document_context", "f2", "running"),
      task("prepare_document", "f3", "failed", {
        attempts: 3,
        lastError: { error_type: "RuntimeError", error_message: "mineru_api returned HTTP 400" },
      }),
      task("prepare_document", "f4", "succeeded", { sourceMetadata: source("d.pdf") }),
      task("extract_document_context", "f4", "queued"),
      task("prepare_document", "f5", "queued"),
      task("prepare_document", "f6", "succeeded", { sourceMetadata: source("c.pdf") }),
      task("compact_document", "f6", "failed", { attempts: 2, lastError: { message: "no windows" } }),
      task("prepare_document", "f7", "cancelled"),
    ]);
    expect(statuses.map((item) => [item.name ?? item.fileId, item.phase])).toEqual([
      ["a.pdf", "running"],
      ["b.pdf", "indexed"],
      ["c.pdf", "poison"],
      ["d.pdf", "scanned"],
      ["f3", "ocr-failed"],
      ["f5", "queued"],
      ["f7", "cancelled"],
    ]);
    expect(statuses.find((item) => item.fileId === "f3")?.detail).toBe("RuntimeError: mineru_api returned HTTP 400");
    expect(statuses.find((item) => item.fileId === "f6")?.detail).toBe("no windows (attempt 2)");
    expect(statuses.find((item) => item.fileId === "f2")?.detail).toBe("Extract document context");
  });

  it("reads a document its preparation recorded as unreadable as OCR-failed", () => {
    // The task succeeded - the failure was recorded so the run could go on -
    // and without its tag the document would sit at "Extraction pending".
    const statuses = documentStatusesFromTasks([
      task("prepare_document", "f1", "succeeded", {
        sourceMetadata: source("broken.pdf"),
        unreadable: { file_id: "f1", error_type: "RuntimeError", error_message: "Unsupported file type: txt" },
      }),
      task("prepare_document", "f2", "succeeded", { sourceMetadata: source("fine.pdf") }),
    ]);
    expect(statuses.map((item) => [item.name, item.phase])).toEqual([
      ["broken.pdf", "ocr-failed"],
      ["fine.pdf", "scanned"],
    ]);
    expect(statuses[0]?.detail).toBe(
      "Could not be read and was left out of the graph: RuntimeError: Unsupported file type: txt",
    );
  });

  it("marks quarantined files whatever the runtime reported", () => {
    const statuses = withSkippedFiles(
      [
        { fileId: "f1", phase: "poison", detail: "bad" },
        { fileId: "f2", phase: "indexed", detail: "Indexed" },
      ],
      ["f1"],
    );
    expect(statuses.map((item) => item.phase)).toEqual(["skipped", "indexed"]);
  });

  it("follows a document a revision reprocesses from the stage it starts at", () => {
    // No preparation: a re-extracted document starts at its inventory, a
    // revalidated one at its revalidation, and each is indexed when it ends.
    const statuses = documentStatusesFromTasks([
      task("compact_entity_inventory", "f1", "succeeded"),
      task("compact_document", "f1", "succeeded"),
      task("revalidate_document_context", "f2", "succeeded"),
      task("compact_entity_inventory", "f2", "succeeded"),
      task("revalidate_document", "f2", "succeeded"),
      task("revalidate_document", "f3", "running"),
      task("extract_document_context", "f4", "queued"),
      task("revalidate_document_context", "f5", "running"),
    ]);
    expect(statuses.map((item) => [item.fileId, item.phase, item.detail])).toEqual([
      ["f1", "indexed", "Extracted and compacted into the graph"],
      ["f2", "indexed", "Revalidated under the current rules"],
      ["f3", "running", "Revalidate document"],
      ["f4", "queued", "Waiting for a worker"],
      ["f5", "running", "Revalidate document context"],
    ]);
  });
});

describe("withProvenance", () => {
  it("labels each document as its finalizer recorded how the version came by it", () => {
    const payload = {
      inherit: [
        { run_id: "base", file_ids: ["kept"], prepared_file_ids: ["redone", "checked"] },
        { run_id: "older", file_ids: ["older"] },
      ],
      reprocessed: { "reextracted:relations": ["redone"], revalidated: ["checked"] },
    };
    const statuses = withProvenance(
      ["kept", "older", "redone", "checked", "added"].map((fileId) => ({ fileId, phase: "indexed" as const, detail: "" })),
      payload,
    );
    expect(Object.fromEntries(statuses.map((item) => [item.fileId, item.provenance]))).toEqual({
      kept: "reused",
      older: "reused",
      redone: "reextracted:relations",
      checked: "revalidated",
      added: "extracted",
    });
    expect(documentProvenance(null).size).toBe(0);
    expect(formatProvenance("reextracted:entities")).toBe("Re-extracted from entities");
    expect(formatProvenance("revalidated")).toBe("Revalidated");
  });
});
