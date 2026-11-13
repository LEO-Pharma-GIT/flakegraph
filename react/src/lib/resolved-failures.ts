/**
 * The failures a fleet run's retries cleared, read from its status.
 *
 * Retrying a failed run requeues its failed tasks with no error, so a run
 * that later succeeded would otherwise show no sign that it stopped. The
 * coordination store keeps each cleared failure; `distributed status`
 * reports the most recent ones and how many there are in all.
 */
export interface ResolvedFailure {
  taskId: string;
  stage: string;
  scopeId: string;
  attempts: number;
  message: string;
  failedAt: string | null;
  retriedAt: string | null;
}

export function resolvedFailures(raw: Record<string, unknown>): { failures: ResolvedFailure[]; total: number } {
  const rows = Array.isArray(raw.resolved_failures) ? raw.resolved_failures : [];
  const failures = rows.flatMap((row): ResolvedFailure[] => {
    if (!row || typeof row !== "object") {
      return [];
    }
    const record = row as Record<string, unknown>;
    return [
      {
        taskId: String(record.task_id ?? ""),
        stage: String(record.stage ?? ""),
        scopeId: String(record.scope_id ?? ""),
        attempts: Number(record.attempts ?? 0),
        message: failureMessage(record.error),
        failedAt: typeof record.failed_at === "string" ? record.failed_at : null,
        retriedAt: typeof record.retried_at === "string" ? record.retried_at : null,
      },
    ];
  });
  const total = Number(raw.resolved_failure_count ?? failures.length);
  return { failures, total: Number.isFinite(total) ? Math.max(total, failures.length) : failures.length };
}

function failureMessage(error: unknown): string {
  if (!error || typeof error !== "object") {
    return typeof error === "string" ? error : "No error was recorded";
  }
  const record = error as Record<string, unknown>;
  const type = typeof record.error_type === "string" ? record.error_type : "";
  const message = String(record.error_message ?? record.message ?? "").split(/\r?\n/)[0]?.trim() ?? "";
  if (type && message) {
    return `${type}: ${message}`;
  }
  return message || type || "No error was recorded";
}
