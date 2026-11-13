import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  const digits = value >= 10 || unit === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) {
    return "unknown";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const deltaSeconds = Math.round((date.getTime() - Date.now()) / 1000);
  const abs = Math.abs(deltaSeconds);
  const formatter = new Intl.RelativeTimeFormat("en", { numeric: "auto" });
  if (abs < 60) {
    return formatter.format(deltaSeconds, "second");
  }
  if (abs < 3600) {
    return formatter.format(Math.round(deltaSeconds / 60), "minute");
  }
  if (abs < 86400) {
    return formatter.format(Math.round(deltaSeconds / 3600), "hour");
  }
  return formatter.format(Math.round(deltaSeconds / 86400), "day");
}

/** A wall-clock span as people say it: 4s, 12m 30s, 1h 36m. */
export function formatDuration(milliseconds: number | null | undefined): string {
  if (milliseconds == null || !Number.isFinite(milliseconds) || milliseconds < 0) {
    return "—";
  }
  const seconds = Math.round(milliseconds / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ${seconds % 60}s`;
  }
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

/** An instant in the viewer's locale, or a dash when the runtime has none. */
export function formatInstant(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

export function formatCount(value: number | null | undefined): string {
  return Number(value ?? 0).toLocaleString();
}

export function titleCaseStage(stage: string): string {
  return stage
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const DOCUMENT_PHASES: Record<string, string> = {
  "ocr-failed": "OCR failed",
  "extracted-0": "No text extracted",
  skipped: "Skipped",
  poison: "Poisoned file",
  cancelled: "Cancelled",
  inherited: "Kept from an earlier version",
  scanned: "Parsed",
  indexed: "Indexed",
  queued: "Queued",
  running: "In progress",
  extracted: "Text extracted",
  embedded: "Embedded",
  done: "Done",
};

export function formatDocumentPhase(phase: string): string {
  return DOCUMENT_PHASES[phase] ?? titleCaseStage(phase);
}

/**
 * How a graph version came by a document's extraction, as a reader says it:
 * `reextracted:relations` reads "Re-extracted from relations".
 */
export function formatProvenance(provenance: string | undefined | null): string {
  if (!provenance) {
    return "";
  }
  if (provenance.startsWith("reextracted:")) {
    return `Re-extracted from ${provenance.slice("reextracted:".length)}`;
  }
  return provenance.charAt(0).toUpperCase() + provenance.slice(1);
}

export function documentPhaseNeedsSkip(phase: string): boolean {
  return phase === "poison" || phase === "ocr-failed" || phase === "extracted-0";
}
