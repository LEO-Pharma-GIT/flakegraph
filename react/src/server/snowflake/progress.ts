import type { StageProgress } from "../protocol/schema";

/** Read a Snowflake VARIANT that arrives as either an object or JSON text. */
/** A VARIANT or ARRAY column as a list, however the driver delivered it. */
export function variantList(value: unknown): unknown[] {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed: unknown = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

export function variantMapping(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed: unknown = JSON.parse(value);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : {};
    } catch {
      return {};
    }
  }
  return {};
}

/** Read the spend a running worker last reported, if it reported one. */
export function liveConsumption(progress: unknown): Record<string, unknown> {
  const totals = variantMapping(progress).consumption;
  return totals && typeof totals === "object" && !Array.isArray(totals) ? (totals as Record<string, unknown>) : {};
}

/**
 * Render a stored job error as a sentence rather than the JSON it is kept in.
 *
 * KG_JOB.ERROR is an object so the cause stays machine-readable, but the app
 * puts it straight in front of a person.
 */
export function jobErrorText(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string" && !value.trim()) {
    return null;
  }
  const payload = variantMapping(value);
  if (Object.keys(payload).length === 0) {
    return String(value).trim() || null;
  }
  if (payload.message) {
    return String(payload.message);
  }
  if (payload.failed_files !== undefined && payload.failed_files !== null) {
    const count = Number.parseInt(String(payload.failed_files), 10) || 0;
    // Deliberately does not say the retries were exhausted: a run whose worker
    // died leaves its documents on their first attempt, and telling the reader
    // they were retried sends them looking for a cause rather than pressing Retry.
    return `${count} ${count === 1 ? "document" : "documents"} failed. Open the failures below for the reason.`;
  }
  return (
    Object.entries(payload)
      .map(([key, item]) => `${key}: ${typeof item === "object" ? JSON.stringify(item) : String(item)}`)
      .join("; ") || null
  );
}

/**
 * Pick the completed/total pair a stage reports, whatever it counts in.
 *
 * Stages count in their own units (batches, reports, candidates, files), so the
 * pair is discovered rather than assumed.
 */
export function progressCounts(counts: Record<string, unknown>): { completed: number; total: number | null } {
  for (const [completedKey, totalKey] of [
    ["batches_completed", "batches_total"],
    ["reports_completed", "reports_total"],
    ["completed", "total"],
    ["files_processed", "files_total"],
  ] as const) {
    if (completedKey in counts && totalKey in counts) {
      // VARIANT counters arrive untyped; anything that is not a number is rejected.
      const completed = numeric(counts[completedKey]);
      const total = numeric(counts[totalKey]);
      if (completed === null || total === null) {
        return { completed: 0, total: null };
      }
      return { completed, total };
    }
  }
  return { completed: 0, total: null };
}

function numeric(value: unknown): number | null {
  if (typeof value !== "number" && typeof value !== "string") {
    return null;
  }
  if (typeof value === "string" && !value.trim()) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null;
}

/**
 * Replace the running stage's file counts with the worker's own counts.
 *
 * File rows only move between queued, claimed and done, so every stage of a run
 * over one claimed batch reports zero progress until the batch finishes. The
 * worker records what it is actually doing on the job row, which is the only
 * source that can tell a long stage from a stalled worker.
 */
export function withLiveStageProgress(stages: readonly StageProgress[], rawProgress: unknown): StageProgress[] {
  const progress = variantMapping(rawProgress);
  const stageName = String(progress.stage ?? "");
  if (!stageName) {
    return [...stages];
  }
  const counts = variantMapping(progress.counts);
  const { completed, total } = progressCounts(counts);
  const message = progress.message ? String(progress.message) : null;
  let seen = false;
  const updated = stages.map((stage) => {
    if (stage.stage !== stageName) {
      return stage;
    }
    seen = true;
    return {
      ...stage,
      completed: total ? completed : stage.completed,
      total: total || stage.total,
      message: message ?? stage.message,
    };
  });
  if (!seen) {
    // The worker reports a stage the file rows have not recorded yet.
    updated.push({
      stage: stageName,
      status: String(progress.status ?? "running"),
      completed,
      total,
      elapsedMs: 0,
      message,
    });
  }
  return updated;
}

/**
 * Read the fixed width Snowflake reports for a VECTOR column.
 *
 * SHOW COLUMNS describes a column's type as JSON, and only a vector carries a
 * dimension. INFORMATION_SCHEMA says only VECTOR and omits the width.
 */
export function vectorWidth(dataType: unknown): number | null {
  const parsed = variantMapping(dataType);
  const declared = parsed.dimension;
  return typeof declared === "number" && Number.isInteger(declared) ? declared : null;
}
