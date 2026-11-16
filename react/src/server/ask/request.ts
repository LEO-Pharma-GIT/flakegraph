import { TOOL_DEFAULT_TOP_K, TOOL_MAX_TOP_K } from "./constants";
import { AskHttpError } from "./errors";
import type { AskRequestBody, QueryMode } from "./types";

export function acceptFormat(request: Request, body: AskRequestBody): "ui" | "ndjson" {
  if (body.format === "ui" || body.format === "ndjson") {
    return body.format;
  }
  const url = new URL(request.url);
  if (url.searchParams.get("format") === "ndjson") {
    return "ndjson";
  }
  const accept = request.headers.get("accept") ?? "";
  if (accept.includes("application/x-ndjson") || accept.includes("application/jsonl")) {
    return "ndjson";
  }
  return "ui";
}

export function extractQuestion(body: AskRequestBody): string {
  if (body.question?.trim()) {
    return body.question.trim();
  }
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as {
      role?: string;
      content?: string;
      parts?: Array<{ type?: string; text?: string }>;
    };
    if (message?.role !== "user") {
      continue;
    }
    if (typeof message.content === "string" && message.content.trim()) {
      return message.content.trim();
    }
    const text = (message.parts ?? [])
      .filter((part) => part.type === "text")
      .map((part) => part.text ?? "")
      .join("");
    if (text.trim()) {
      return text.trim();
    }
  }
  return "";
}

export function parseMode(value: unknown): QueryMode | undefined {
  if (value == null || value === "auto" || value === "") {
    return undefined;
  }
  if (value === "local" || value === "global" || value === "hybrid" || value === "drift") {
    return value;
  }
  throw new AskHttpError(400, "mode must be local, global, hybrid, drift, or auto.", "invalid_mode");
}

export function clampTopK(value: number | undefined): number {
  if (!value || !Number.isFinite(value)) {
    return TOOL_DEFAULT_TOP_K;
  }
  return Math.min(TOOL_MAX_TOP_K, Math.max(1, Math.floor(value)));
}

export function parseDocumentIds(value: unknown): string[] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new AskHttpError(400, "documentIds must be an array of strings.", "invalid_document_ids");
  }
  const ids = [...new Set(value.map((item) => String(item).trim()).filter(Boolean))];
  return ids.length > 0 ? ids : undefined;
}

export function toUiUserMessage(question: string) {
  return {
    id: "ask-user",
    role: "user" as const,
    parts: [{ type: "text" as const, text: question }],
  };
}
