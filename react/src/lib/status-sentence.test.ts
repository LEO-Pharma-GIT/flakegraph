import { describe, expect, it } from "vitest";
import { catalogEmptyCopy, statusSentence } from "./status-sentence";

describe("statusSentence", () => {
  it("writes a blocking sentence for running work", () => {
    expect(
      statusSentence({
        status: "running",
        error: null,
        documentsCompleted: 3,
        documentsTotal: 10,
        documentsFailed: 0,
        stages: [{ stage: "ocr", status: "running", completed: 3, total: 10, elapsedMs: 0, message: null }],
      }),
    ).toMatch(/Ocr · 3\/10 documents/i);
  });

  it("does not call a zero-entity finish succeeded", () => {
    expect(
      statusSentence(
        {
          status: "succeeded",
          error: null,
          documentsCompleted: 10,
          documentsTotal: 10,
          documentsFailed: 0,
          stages: [],
        },
        { nodeCount: 0 },
      ),
    ).toMatch(/0 entities/i);
  });

  it("prefers entity count over a zero document total", () => {
    expect(
      statusSentence(
        {
          status: "succeeded",
          error: null,
          documentsCompleted: 0,
          documentsTotal: 0,
          documentsFailed: 0,
          stages: [],
        },
        { nodeCount: 74 },
      ),
    ).toMatch(/74 entities/);
  });

  it("counts failed documents in words that agree with the number", () => {
    const finished = (documentsFailed: number) =>
      statusSentence(
        { status: "succeeded", error: null, documentsCompleted: 1, documentsTotal: 3, documentsFailed, stages: [] },
        { nodeCount: 4 },
      );
    expect(finished(1)).toBe("Finished with gaps · 1 document failed");
    expect(finished(2)).toBe("Finished with gaps · 2 documents failed");
  });
});

describe("catalogEmptyCopy", () => {
  const base = {
    loading: false,
    total: 4,
    visible: 0,
    search: "",
    storageFilter: "all",
    scope: "mine" as const,
    identified: true,
  };

  it("distinguishes first-run empty from a search miss", () => {
    expect(catalogEmptyCopy({ ...base, total: 0 })?.title).toBe("No graphs yet");
    expect(catalogEmptyCopy({ ...base, search: "foo" })?.title).toContain("foo");
  });

  it("asks an unidentified session to sign in rather than to build", () => {
    expect(catalogEmptyCopy({ ...base, total: 0, identified: false })?.detail).toMatch(/Sign in/);
  });

  it("points Mine at Shared when everything visible belongs to someone else", () => {
    const copy = catalogEmptyCopy({ ...base, scope: "mine" });
    expect(copy?.title).toBe("None of your own");
    expect(copy?.detail).toMatch(/4 graphs are shared with you/);
  });

  it("explains an empty Shared list", () => {
    expect(catalogEmptyCopy({ ...base, scope: "shared" })?.title).toBe("Nothing shared with you");
  });

  it("says nothing while the scope has rows", () => {
    expect(catalogEmptyCopy({ ...base, scope: "mine", visible: 2 })).toBeNull();
  });

  it("tells analysts when the catalog has no production perspectives", () => {
    expect(catalogEmptyCopy({ ...base, analyst: true })?.title).toBe("No production graphs");
  });
});
