export type { AskStreamEvent, AskRequestBody, QueryMode, AskAnswer, AskCitation } from "./types";
export { QUERY_MODES } from "./types";

import type { AskStreamEvent } from "./types";

export function encodeAskEvent(event: AskStreamEvent): string {
  return `${JSON.stringify(event)}\n`;
}

export async function* readAskNdjson(response: Response): AsyncGenerator<AskStreamEvent> {
  if (!response.body) {
    throw new Error("Ask stream has no body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      yield JSON.parse(trimmed) as AskStreamEvent;
    }
    if (done) {
      if (buffer.trim()) {
        yield JSON.parse(buffer) as AskStreamEvent;
      }
      return;
    }
  }
}

export function collectAskText(events: AskStreamEvent[]): string {
  return events
    .filter((event): event is Extract<AskStreamEvent, { type: "text" }> => event.type === "text")
    .map((event) => event.delta)
    .join("");
}
