import { describe, expect, it } from "vitest";
import { documentCounts, errorMessage, snapshotFromStatus, stageProgress } from "./fleet-status";
import type { RunSnapshot } from "./protocol/schema";

const base: RunSnapshot = {
  runId: "run-1",
  graphId: "graph-1",
  status: "unknown",
  startedAt: null,
  updatedAt: null,
  graphName: null,
  stages: [],
  events: [],
  documentsTotal: null,
  documentsCompleted: 0,
  documentsFailed: 0,
  outputPath: "/state/artifacts/run-1",
  storageKind: "local_files",
  storageLocation: "/state/artifacts/run-1",
  warnings: [],
  error: null,
  listingWarning: null,
  raw: {},
};

const counts = [
  { stage: "prepare_document", status: "succeeded", count: 48, started_at: "2026-09-18T10:00:00Z", completed_at: "2026-09-18T10:05:00Z" },
  { stage: "prepare_document", status: "failed", count: 1, started_at: "2026-09-18T10:00:10Z", completed_at: "2026-09-18T10:06:00Z" },
  { stage: "extract_entity_window", status: "succeeded", count: 300, started_at: "2026-09-18T10:01:00Z", completed_at: "2026-09-18T10:30:00Z" },
  { stage: "extract_entity_window", status: "running", count: 8, started_at: "2026-09-18T10:02:00Z" },
  { stage: "compact_document", status: "succeeded", count: 20, started_at: "2026-09-18T10:20:00Z", completed_at: "2026-09-18T10:31:00Z" },
  { stage: "compact_document", status: "queued", count: 28 },
  {
    stage: "finalize_graph",
    status: "running",
    count: 1,
    started_at: "2026-09-18T10:31:00Z",
    progress: { phase: "build_communities", phase_index: 6, phase_total: 8, completed: 3, total: 10, message: "Summarising communities" },
  },
];

describe("fleet status", () => {
  it("rolls task buckets into one row per stage in pipeline order", () => {
    const now = new Date("2026-09-18T10:40:00Z");
    const stages = stageProgress(counts, now);
    expect(stages.map((stage) => stage.stage)).toEqual([
      "prepare_document",
      "extract_entity_window",
      "compact_document",
      "finalize_graph",
    ]);
    const [prepare, entities, compact, finalize] = stages;
    expect(prepare).toMatchObject({ status: "failed", completed: 48, total: 49, elapsedMs: 6 * 60_000 });
    expect(entities).toMatchObject({ status: "running", completed: 300, total: 308, elapsedMs: 39 * 60_000 });
    expect(compact).toMatchObject({ status: "queued", completed: 20, total: 48 });
    // A single long task shows the phase it is in.
    expect(finalize).toMatchObject({ status: "running", completed: 5, total: 8, message: "Summarising communities · 3/10" });
  });

  it("counts documents by what was discovered and what finished", () => {
    expect(documentCounts(counts)).toEqual({ total: 49, completed: 20, failed: 1 });
    expect(documentCounts([])).toEqual({ total: null, completed: 0, failed: 0 });
    // A revision's revalidated document is discovered by the task that starts
    // its revalidation and finished by its revalidate_document task.
    expect(
      documentCounts([
        { stage: "revalidate_document_context", status: "succeeded", count: 4 },
        { stage: "revalidate_document_context", status: "failed", count: 1 },
        { stage: "revalidate_document", status: "succeeded", count: 3 },
        { stage: "revalidate_document", status: "queued", count: 1 },
        { stage: "compact_document", status: "succeeded", count: 2 },
      ]),
    ).toEqual({ total: 5, completed: 5, failed: 1 });
    // A run planned before revalidation ran window by window started and
    // finished each revalidated document with one revalidate_document task.
    expect(
      documentCounts([
        { stage: "revalidate_document", status: "succeeded", count: 3 },
        { stage: "revalidate_document", status: "failed", count: 1 },
        { stage: "compact_document", status: "succeeded", count: 2 },
      ]),
    ).toEqual({ total: 4, completed: 5, failed: 1 });
  });

  it("leads a failure with the task and the server's cause", () => {
    expect(errorMessage(null)).toBeNull();
    expect(errorMessage("boom")).toBe("boom");
    expect(errorMessage({ task_id: "task_1", error_type: "RuntimeError", error_message: "mineru_api returned HTTP 400" })).toBe(
      "Task task_1 failed with RuntimeError: mineru_api returned HTTP 400",
    );
    const java = "org.apache.spark.SparkException: Job aborted\n  Message: table not found\n  Received status: FAILED\n  at org.apache.spark.Foo(Bar.scala:1)";
    expect(errorMessage({ error_type: "SparkException", error_message: java })).toBe(
      "Distributed run failed with SparkException: table not found",
    );
  });

  it("translates a status payload onto the console's run", () => {
    const snapshot = snapshotFromStatus(
      {
        run: { id: "run-1", graph_id: "graph-1", status: "running", config_digest: "abc" },
        created_at: "2026-09-18T10:00:00Z",
        updated_at: "2026-09-18T10:39:00Z",
        task_counts: counts,
        diagnostics: {
          state: "attention_required",
          warnings: [{ code: "QUEUED_WORK_NOT_ADVANCING", message: "28 tasks queued.", remediation: "Check the workers." }],
        },
        error: null,
      },
      base,
      new Date("2026-09-18T10:40:00Z"),
    );
    expect(snapshot.status).toBe("running");
    expect(snapshot.startedAt).toBe("2026-09-18T10:00:00Z");
    expect(snapshot.documentsTotal).toBe(49);
    expect(snapshot.warnings).toEqual(["28 tasks queued. Check the workers."]);
    expect(snapshot.stages).toHaveLength(4);
    expect(snapshot.outputPath).toBe("/state/artifacts/run-1");
    expect(snapshot.raw.diagnostics).toBeDefined();
  });

  it("counts documents recorded as unreadable, whose tasks succeeded", () => {
    const payload = {
      run: { id: "run-1", graph_id: "graph-1", status: "running" },
      task_counts: counts,
      documents_failed: 2,
    };
    expect(snapshotFromStatus(payload, base).documentsFailed).toBe(documentCounts(counts).failed + 2);
  });
});
