import { Data } from "effect";

export class ControlPlaneError extends Data.TaggedError("ControlPlaneError")<{
  readonly code:
    | "not_found"
    | "not_supported"
    | "invalid"
    | "conflict"
    | "unavailable"
    | "forbidden"
    | "internal";
  readonly message: string;
  readonly cause?: unknown;
}> {}

export function notFound(message: string): ControlPlaneError {
  return new ControlPlaneError({ code: "not_found", message });
}

export function notSupported(message: string): ControlPlaneError {
  return new ControlPlaneError({ code: "not_supported", message });
}

export function invalid(message: string): ControlPlaneError {
  return new ControlPlaneError({ code: "invalid", message });
}

export function conflict(message: string): ControlPlaneError {
  return new ControlPlaneError({ code: "conflict", message });
}

export function forbidden(message: string): ControlPlaneError {
  return new ControlPlaneError({ code: "forbidden", message });
}

export function unavailable(message: string, cause?: unknown): ControlPlaneError {
  return new ControlPlaneError({ code: "unavailable", message, cause });
}

export function internal(message: string, cause?: unknown): ControlPlaneError {
  return new ControlPlaneError({ code: "internal", message, cause });
}

export function fromCause(cause: unknown, fallback: string): ControlPlaneError {
  if (cause instanceof ControlPlaneError) {
    return cause;
  }
  if (cause instanceof Error) {
    return invalid(cause.message || fallback);
  }
  return invalid(fallback);
}

export function toTrpcCode(
  error: ControlPlaneError,
):
  | "NOT_FOUND"
  | "BAD_REQUEST"
  | "FORBIDDEN"
  | "CONFLICT"
  | "PRECONDITION_FAILED"
  | "INTERNAL_SERVER_ERROR" {
  switch (error.code) {
    case "not_found":
      return "NOT_FOUND";
    case "invalid":
      return "BAD_REQUEST";
    case "forbidden":
      return "FORBIDDEN";
    case "conflict":
      return "CONFLICT";
    case "not_supported":
      return "PRECONDITION_FAILED";
    default:
      return "INTERNAL_SERVER_ERROR";
  }
}
