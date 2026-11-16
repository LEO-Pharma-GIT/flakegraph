import { DISTRIBUTED_STAGE_ORDER, type RunSnapshot, type StageProgress } from "./protocol/schema";

/**
 * Translate the CLI's bounded run diagnostics into the console's run model.
 *
 * `flakegraph distributed status` reports durable task aggregates per stage
 * and status, the run's own row, and the diagnostics the coordinator derives
 * from them. Nothing here needs a worker registry or a task scan.
 */
export function snapshotFromStatus(
  payload: Record<string, unknown>,
  base: RunSnapshot,
  now: Date = new Date(),
): RunSnapshot {
  const run = asRecord(payload.run);
  const counts = Array.isArray(payload.task_counts) ? payload.task_counts.map(asRecord) : [];
  const diagnostics = asRecord(payload.diagnostics);
  const warnings = Array.isArray(diagnostics.warnings) ? diagnostics.warnings.map(diagnosticWarning) : [];
  const documents = documentCounts(counts);
  // Documents recorded as unreadable: their prepare tasks succeeded, so the
  // task counts above cannot see them.
  const unreadable = Number(payload.documents_failed ?? 0);
  return {
    ...base,
    runId: String(run.id ?? base.runId),
    graphId: String(run.graph_id ?? base.graphId),
    status: String(run.status ?? base.status),
    startedAt: optionalString(payload.created_at) ?? base.startedAt,
    updatedAt: optionalString(payload.updated_at) ?? base.updatedAt,
    stages: stageProgress(counts, now),
    documentsTotal: documents.total,
    documentsCompleted: documents.completed,
    documentsFailed: documents.failed + (Number.isFinite(unreadable) ? unreadable : 0),
    warnings,
    error: errorMessage(payload.error) ?? base.error,
    raw: { ...base.raw, ...payload },
  };
}

/** One status row per pipeline stage, in pipeline order. */
export function stageProgress(counts: Record<string, unknown>[], now: Date = new Date()): StageProgress[] {
  const grouped = new Map<string, Map<string, number>>();
  const progressByStage = new Map<string, Record<string, unknown>>();
  const started = new Map<string, number[]>();
  const completed = new Map<string, number[]>();
  for (const item of counts) {
    const stage = String(item.stage ?? "unknown");
    const status = String(item.status ?? "unknown");
    const statuses = grouped.get(stage) ?? new Map<string, number>();
    statuses.set(status, Number(item.count ?? 0));
    grouped.set(stage, statuses);
    if (item.progress && typeof item.progress === "object") {
      progressByStage.set(stage, item.progress as Record<string, unknown>);
    }
    const startedAt = timestamp(item.started_at);
    if (startedAt !== null) {
      started.set(stage, [...(started.get(stage) ?? []), startedAt]);
    }
    const completedAt = timestamp(item.completed_at);
    if (completedAt !== null) {
      completed.set(stage, [...(completed.get(stage) ?? []), completedAt]);
    }
  }
  const stages: StageProgress[] = [];
  for (const [stage, statuses] of grouped) {
    const total = [...statuses.values()].reduce((sum, count) => sum + count, 0);
    const succeeded = statuses.get("succeeded") ?? 0;
    const state = statuses.get("failed")
      ? "failed"
      : statuses.get("running")
        ? "running"
        : total === succeeded
          ? "completed"
          : "queued";
    const stageStarted = started.has(stage) ? Math.min(...(started.get(stage) ?? [])) : null;
    const stageCompleted = completed.has(stage) ? Math.max(...(completed.get(stage) ?? [])) : null;
    const end = state === "running" ? now.getTime() : stageCompleted;
    const elapsedMs = stageStarted !== null && end !== null ? Math.max(0, Math.round(end - stageStarted)) : 0;
    let displayedCompleted = succeeded;
    let displayedTotal: number | null = total;
    let message: string | null = null;
    const progress = progressByStage.get(stage);
    if (state === "running" && progress) {
      // A single long task reports the phase it is in; the row shows phases.
      const phaseIndex = Math.max(1, Number(progress.phase_index ?? 1));
      const phaseTotal = Math.max(phaseIndex, Number(progress.phase_total ?? phaseIndex));
      const innerCompleted = Number(progress.completed ?? 0);
      const innerTotal = Number(progress.total ?? 0);
      const phaseFinished = innerTotal > 0 && innerCompleted >= innerTotal;
      displayedCompleted = Math.min(phaseTotal, phaseIndex - 1 + (phaseFinished ? 1 : 0));
      displayedTotal = phaseTotal;
      message = String(progress.message ?? "") || null;
      if (message && innerTotal > 1) {
        message = `${message} · ${innerCompleted}/${innerTotal}`;
      }
    }
    stages.push({ stage, status: state, completed: displayedCompleted, total: displayedTotal, elapsedMs, message });
  }
  const order = new Map<string, number>(DISTRIBUTED_STAGE_ORDER.map((stage, index) => [stage, index]));
  return stages.sort(
    (left, right) => (order.get(left.stage) ?? order.size) - (order.get(right.stage) ?? order.size),
  );
}

/**
 * Corpus-level counters from the per-document stages: every discovered
 * document owns one prepare task, and every document a revision revalidates
 * one task that starts its revalidation, so those stages are the total; a
 * document is complete once its compaction or its revalidation succeeds. A
 * run planned before revalidation ran window by window started each
 * revalidated document with its revalidate_document task, which also
 * finishes it. A document a revision re-extracts owns neither, so it counts
 * once it completes.
 */
export function documentCounts(counts: Record<string, unknown>[]): {
  total: number | null;
  completed: number;
  failed: number;
} {
  const count = (stage: string, status?: string) =>
    counts
      .filter((item) => item.stage === stage && (status === undefined || item.status === status))
      .reduce((sum, item) => sum + Number(item.count ?? 0), 0);
  const total =
    count("prepare_document") + Math.max(count("revalidate_document_context"), count("revalidate_document"));
  return {
    total: total || null,
    completed: count("compact_document", "succeeded") + count("revalidate_document", "succeeded"),
    failed:
      count("prepare_document", "failed") +
      count("compact_document", "failed") +
      count("revalidate_document_context", "failed") +
      count("revalidate_document", "failed"),
  };
}

/** One operator-facing sentence for a redacted run failure. */
export function errorMessage(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "object") {
    const text = String(value).trim();
    return text || null;
  }
  const record = value as Record<string, unknown>;
  const errorType = String(record.error_type ?? record.type ?? "Run failure");
  const message = errorSummary(String(record.error_message ?? record.message ?? ""));
  const taskId = String(record.task_id ?? record.publication_id ?? "").trim();
  const context = taskId ? `Task ${taskId}` : "Distributed run";
  return message ? `${context} failed with ${errorType}: ${message}` : `${context} failed with ${errorType}.`;
}

// Java and remote-provider exceptions stringify as a short wrapper, a useful
// server "Message:", and hundreds of stack frames; lead with the cause.
function errorSummary(message: string, limit = 500): string {
  const lines = message
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return "";
  }
  const flattened = lines.join(" ");
  const server = /\bMessage:\s*(.+?)(?=\s+Received status:|\s+at\s+[\w.$]+\(|$)/.exec(flattened);
  const summary = (server ? server[1] : lines[0]).trim();
  return summary.length <= limit ? summary : `${summary.slice(0, limit - 1).trimEnd()}…`;
}

function diagnosticWarning(value: unknown): string {
  if (!value || typeof value !== "object") {
    return String(value);
  }
  const record = value as Record<string, unknown>;
  const message = String(record.message ?? JSON.stringify(value));
  const remediation = String(record.remediation ?? "").trim();
  return remediation ? `${message} ${remediation}` : message;
}

function timestamp(value: unknown): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(String(value));
  return Number.isNaN(parsed) ? null : parsed;
}

function optionalString(value: unknown): string | null {
  return value ? String(value) : null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
