export type ObservationCode =
  | "GRAPH_EMPTY_RESULT"
  | "GRAPH_HAS_NO_ENTITIES"
  | "GRAPH_DOCUMENT_SCOPE_MISMATCH"
  | "UNKNOWN_ENTITY"
  | "UNKNOWN_RELATION"
  | "UNKNOWN_DOCUMENT"
  | "INVALID_MODE";

export function structuredObservation(code: ObservationCode, payload: Record<string, unknown>) {
  return {
    observation: "ERROR_OR_EMPTY_RESULT" as const,
    code,
    ...payload,
  };
}
