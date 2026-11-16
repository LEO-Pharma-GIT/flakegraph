export class AskHttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code?: string,
  ) {
    super(message);
    this.name = "AskHttpError";
  }
}

export class AskModelMissingError extends Error {
  constructor() {
    super("No Ask model is configured. Set FLAKEGRAPH_ASK_API_KEY or run Ollama.");
    this.name = "AskModelMissingError";
  }
}

export function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const name = "name" in error ? String(error.name) : "";
  return name === "AbortError" || name === "TimeoutError" || name === "AbortSignal.timeout";
}

/**
 * What a log line says about a failure: the message, never the object. An
 * AI SDK error carries the request it was making - prompt, headers, body -
 * and a log is not where that belongs.
 */
export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
