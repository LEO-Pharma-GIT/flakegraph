import { encodeAskEvent, readAskNdjson } from "./protocol";
import type { AskRequestBody, AskStreamEvent, QueryMode } from "./types";

export interface AskClientOptions {
  origin: string;
  apiKey?: string;
  runtime?: "local" | "kubernetes" | "snowflake";
  headers?: HeadersInit;
}

export interface AskQuery {
  runId: string;
  question: string;
  mode?: QueryMode | "auto";
  perspectiveId?: string;
  topK?: number;
  signal?: AbortSignal;
}

export function askHeaders(options: AskClientOptions): Headers {
  const headers = new Headers(options.headers);
  headers.set("content-type", "application/json");
  headers.set("accept", "application/x-ndjson");
  headers.set("x-flakegraph-runtime", options.runtime ?? "local");
  if (options.apiKey) {
    headers.set("authorization", `Bearer ${options.apiKey}`);
  }
  return headers;
}

export function askUrl(origin: string): string {
  return `${origin.replace(/\/$/, "")}/api/ask`;
}

export async function* streamAsk(
  options: AskClientOptions,
  query: AskQuery,
): AsyncGenerator<AskStreamEvent> {
  const body: AskRequestBody = {
    runId: query.runId,
    question: query.question,
    mode: query.mode,
    perspectiveId: query.perspectiveId,
    topK: query.topK,
    format: "ndjson",
  };
  const response = await fetch(askUrl(options.origin), {
    method: "POST",
    headers: askHeaders(options),
    body: JSON.stringify(body),
    signal: query.signal,
  });
  if (!response.ok && !response.body) {
    const payload = (await response.json().catch(() => ({}))) as { error?: string; code?: string };
    yield {
      type: "error",
      message: payload.error ?? `Ask failed with HTTP ${response.status}`,
      code: payload.code ?? String(response.status),
    };
    return;
  }
  if (!response.ok && response.headers.get("content-type")?.includes("application/json")) {
    const payload = (await response.json().catch(() => ({}))) as { error?: string; code?: string };
    yield {
      type: "error",
      message: payload.error ?? `Ask failed with HTTP ${response.status}`,
      code: payload.code ?? String(response.status),
    };
    return;
  }
  yield* readAskNdjson(response);
}

export async function collectAskStream(
  options: AskClientOptions,
  query: AskQuery,
): Promise<AskStreamEvent[]> {
  const events: AskStreamEvent[] = [];
  for await (const event of streamAsk(options, query)) {
    events.push(event);
  }
  return events;
}

export { encodeAskEvent, readAskNdjson };
