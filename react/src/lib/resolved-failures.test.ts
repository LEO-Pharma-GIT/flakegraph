import { describe, expect, it } from "vitest";
import { resolvedFailures } from "./resolved-failures";

describe("resolvedFailures", () => {
  it("reads the failures a retry cleared from a run's status", () => {
    const { failures, total } = resolvedFailures({
      resolved_failure_count: 7,
      resolved_failures: [
        {
          task_id: "task-1",
          stage: "extract_relation_window",
          scope_id: "window-9",
          attempts: 3,
          error: { error_type: "TimeoutError", error_message: "provider timed out\ntraceback" },
          failed_at: "2026-09-28T10:00:00Z",
          retried_at: "2026-09-28T11:00:00Z",
        },
        { task_id: "task-2", stage: "publish_graph", scope_id: "pub", attempts: 1, error: null },
        "not a row",
      ],
    });

    expect(total).toBe(7);
    expect(failures).toEqual([
      {
        taskId: "task-1",
        stage: "extract_relation_window",
        scopeId: "window-9",
        attempts: 3,
        message: "TimeoutError: provider timed out",
        failedAt: "2026-09-28T10:00:00Z",
        retriedAt: "2026-09-28T11:00:00Z",
      },
      {
        taskId: "task-2",
        stage: "publish_graph",
        scopeId: "pub",
        attempts: 1,
        message: "No error was recorded",
        failedAt: null,
        retriedAt: null,
      },
    ]);
    expect(resolvedFailures({})).toEqual({ failures: [], total: 0 });
  });
});
