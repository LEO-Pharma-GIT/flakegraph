"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, isToolUIPart, type UIMessage } from "ai";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { getRuntimeHeader } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Markdown } from "@/components/ui/markdown";
import { cn } from "@/lib/utils";
import { PROGRESS_LABELS, type QueryMode, type SearchProgressUpdate } from "@/server/ask/types";

type AskUIMessage = UIMessage<never, { status: SearchProgressUpdate }>;

const MODES: Array<{ id: QueryMode | "auto"; label: string; hint: string }> = [
  { id: "local", label: "Local", hint: "Entities, relations, and quotes." },
  { id: "global", label: "Global", hint: "Community reports and themes." },
  { id: "hybrid", label: "Hybrid", hint: "Entities plus community context." },
  { id: "drift", label: "Drift", hint: "Multi-hop follow-up retrieval." },
  { id: "auto", label: "Auto", hint: "Planner picks the retrieval level." },
];

export function AskPanel({
  runId,
  perspectives = [],
  graphQuestions = [],
  onOpenEntity,
}: {
  runId: string;
  perspectives?: Array<{ id: string; name: string; lifecycle: string; suggestedQuestions: string[] }>;
  /**
   * Questions the graph itself can answer, from its community reports (the
   * pipeline writes `suggested_questions` on each), largest neighborhood
   * first. They are the starting points; a perspective's questions come
   * first when one is in production.
   */
  graphQuestions?: readonly string[];
  onOpenEntity?: (name: string) => void;
}) {
  const production = perspectives.filter((item) => item.lifecycle === "production");
  const starters = uniqueQuestions([...(production[0]?.suggestedQuestions ?? []), ...graphQuestions]).slice(0, STARTER_LIMIT);
  const [question, setQuestion] = useState(starters[0] ?? "");
  // The planner picks local, global or drift per question; a person should
  // only have to choose when they know better.
  const [mode, setMode] = useState<QueryMode | "auto">("auto");
  const [showAllCitations, setShowAllCitations] = useState(false);
  const [perspectiveId, setPerspectiveId] = useState(production[0]?.id ?? "");
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/ask",
        // The same runtime the rest of the console addresses: the one the
        // URL names, else the server's default, never a local fallback that
        // would read a fleet run's graph through the local runtime.
        headers: (): Record<string, string> => {
          const runtime = getRuntimeHeader();
          return runtime ? { "x-flakegraph-runtime": runtime } : {};
        },
        body: () => ({
          runId,
          mode,
          perspectiveId: perspectiveId || undefined,
          format: "ui",
        }),
      }),
    [mode, perspectiveId, runId],
  );
  const { messages, sendMessage, status, stop, error, setMessages } = useChat<AskUIMessage>({
    id: `${runId}:${mode}:${perspectiveId}`,
    transport,
    onError: (next) => toast.error(next.message),
  });
  useEffect(() => {
    setMessages([]);
  }, [runId, setMessages]);

  const pending = status === "submitted" || status === "streaming";
  const assistant = [...messages].reverse().find((message) => message.role === "assistant");
  const answerText = assistant ? textFromMessage(assistant) : "";
  const statuses = statusParts(messages);
  const latestStatus = statuses.at(-1);
  const tools = toolParts(messages);
  const citations = citationsFromTools(tools);
  const consumption = consumptionFromTools(tools);
  const modeHint = MODES.find((item) => item.id === mode)?.hint;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ask the graph</CardTitle>
        <CardDescription>
          The model queries entities, communities, relations, documents, and neighborhoods, then streams the answer.
          Other programs can call the same endpoint at <span className="font-mono text-xs">POST /api/ask</span>.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {MODES.map((item) => (
            <Button
              key={item.id}
              size="sm"
              variant={mode === item.id ? "default" : "outline"}
              aria-pressed={mode === item.id}
              onClick={() => setMode(item.id)}
            >
              {item.label}
            </Button>
          ))}
          <p className="self-center text-xs text-muted-foreground">{modeHint}</p>
        </div>
        {perspectives.length ? (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant={!perspectiveId ? "default" : "outline"} onClick={() => setPerspectiveId("")}>
              Whole graph
            </Button>
            {perspectives.map((item) => (
              <Button
                key={item.id}
                size="sm"
                variant={perspectiveId === item.id ? "default" : "outline"}
                onClick={() => {
                  setPerspectiveId(item.id);
                  if (item.suggestedQuestions[0]) {
                    setQuestion(item.suggestedQuestions[0]);
                  }
                }}
              >
                Scope: {item.name}
              </Button>
            ))}
          </div>
        ) : null}
        {starters.length ? (
          <div className="flex flex-wrap items-center gap-1.5" data-testid="ask-starters">
            <span className="text-xs text-muted-foreground">Try</span>
            {starters.map((item) => (
              <button
                key={item}
                type="button"
                className={cn(
                  "rounded-full border px-2.5 py-0.5 text-left text-xs transition-colors",
                  question === item ? "border-primary bg-accent text-foreground" : "border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
                onClick={() => setQuestion(item)}
              >
                {item}
              </button>
            ))}
          </div>
        ) : null}
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            aria-label="Ask the graph"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question about this graph"
            onKeyDown={(event) => {
              if (event.key === "Enter" && question.trim().length >= 2 && !pending) {
                event.preventDefault();
                void sendMessage({ text: question.trim() });
              }
            }}
          />
          {pending ? (
            <Button className="sm:shrink-0" variant="outline" onClick={() => stop()}>
              Stop
            </Button>
          ) : (
            <Button
              className="sm:shrink-0"
              onClick={() => void sendMessage({ text: question.trim() })}
              disabled={question.trim().length < 2}
            >
              Ask
            </Button>
          )}
        </div>
        {latestStatus || pending ? (
          <p className="text-xs text-muted-foreground" data-testid="ask-status">
            {latestStatus?.message ?? PROGRESS_LABELS.answering}
            {latestStatus?.detail ? ` · ${latestStatus.detail}` : ""}
          </p>
        ) : null}
        {tools.length ? (
          <ul className="space-y-1 text-xs text-muted-foreground" data-testid="ask-tools">
            {tools.map((tool) => (
              <li key={tool.id}>
                {tool.name} · {tool.state}
              </li>
            ))}
          </ul>
        ) : null}
        {error ? <p className="text-sm text-destructive">{error.message}</p> : null}
        {answerText ? (
          <div className="space-y-2 text-sm" data-testid="ask-answer">
            <Markdown>{answerText}</Markdown>
            {perspectiveId ? <p className="text-muted-foreground">Answer scoped to the selected perspective.</p> : null}
            {consumption ? (
              <p className="text-muted-foreground" data-testid="query-consumption">
                Query consumption · {consumption.entities} entities · {consumption.quotes} quotes · streamed LLM
                answer
              </p>
            ) : (
              <p className="text-muted-foreground" data-testid="query-consumption">
                Query consumption · streamed from POST /api/ask
              </p>
            )}
            {citations.length ? (
              <div className="space-y-1.5" data-testid="ask-citations">
                <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Sources
                  <span className="ml-1.5 font-normal normal-case tracking-normal">
                    {citations.length.toLocaleString()} passage{citations.length === 1 ? "" : "s"}
                  </span>
                </p>
                <ul className="space-y-1.5">
                  {(showAllCitations ? citations : citations.slice(0, CITATION_PREVIEW)).map((citation, index) => (
                    <Citation
                      key={`${citation.documentId}-${index}`}
                      documentName={citation.documentName || citation.documentId || "document"}
                      quote={citation.quote}
                      onOpen={() => onOpenEntity?.(citation.entityName || citation.quote.slice(0, 48))}
                    />
                  ))}
                </ul>
                {citations.length > CITATION_PREVIEW ? (
                  <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={() => setShowAllCitations((current) => !current)}>
                    {showAllCitations ? "Show fewer" : `Show all ${citations.length.toLocaleString()} passages`}
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function textFromMessage(message: AskUIMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function statusParts(messages: AskUIMessage[]): SearchProgressUpdate[] {
  return messages.flatMap((message) =>
    message.parts.flatMap((part) => {
      if (part.type !== "data-status") {
        return [];
      }
      return [part.data];
    }),
  );
}

function toolParts(messages: AskUIMessage[]): Array<{ id: string; name: string; state: string; output?: unknown }> {
  return messages.flatMap((message) =>
    message.parts.flatMap((part, index) => {
      if (!isToolUIPart(part)) {
        return [];
      }
      return [
        {
          id: `${message.id}-${index}`,
          name: part.type.replace(/^tool-/, ""),
          state: part.state,
          output: "output" in part ? part.output : undefined,
        },
      ];
    }),
  );
}

function citationsFromTools(tools: Array<{ output?: unknown }>): Array<{
  quote: string;
  documentId: string;
  documentName?: string;
  entityName: string | null;
}> {
  const citations: Array<{ quote: string; documentId: string; documentName?: string; entityName: string | null }> = [];
  const seen = new Set<string>();
  for (const tool of tools) {
    if (!tool.output || typeof tool.output !== "object") {
      continue;
    }
    const output = tool.output as {
      citations?: Array<{ quote?: string; documentId?: string; documentName?: string; entityName?: string | null }>;
    };
    for (const citation of output.citations ?? []) {
      const quote = citation.quote?.trim();
      if (!quote) {
        continue;
      }
      const key = `${citation.documentId ?? ""}:${quote}`;
      if (seen.has(key)) {
        continue;
      }
      seen.add(key);
      citations.push({
        quote,
        documentId: citation.documentId ?? "",
        ...(citation.documentName ? { documentName: citation.documentName } : {}),
        entityName: citation.entityName ?? null,
      });
    }
  }
  return citations;
}

function consumptionFromTools(tools: Array<{ output?: unknown }>): { entities: number; quotes: number } | null {
  for (let index = tools.length - 1; index >= 0; index -= 1) {
    const output = tools[index]?.output;
    if (!output || typeof output !== "object") {
      continue;
    }
    const row = output as {
      entityIds?: string[];
      citations?: unknown[];
      counts?: { entities?: number; evidence?: number };
    };
    const entities = row.counts?.entities ?? row.entityIds?.length;
    const quotes = row.counts?.evidence ?? row.citations?.length;
    if (entities || quotes) {
      return { entities: entities ?? 0, quotes: quotes ?? 0 };
    }
  }
  return null;
}

/** Passages listed under an answer before "Show all". */
const CITATION_PREVIEW = 8;
/** Characters of a passage shown before it has to be opened; a passage can be a whole page. */
const QUOTE_PREVIEW = 240;

function Citation({ documentName, quote, onOpen }: { documentName: string; quote: string; onOpen: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const long = quote.length > QUOTE_PREVIEW;
  const shown = expanded || !long ? quote : `${quote.slice(0, QUOTE_PREVIEW).trimEnd()}…`;
  return (
    <li className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
      <div className="flex items-baseline justify-between gap-3">
        <button type="button" className="min-w-0 truncate text-left font-medium underline-offset-2 hover:underline" onClick={onOpen}>
          {documentName}
        </button>
        {long ? (
          <button type="button" className="shrink-0 text-xs text-muted-foreground hover:text-foreground" onClick={() => setExpanded((current) => !current)}>
            {expanded ? "Less" : "More"}
          </button>
        ) : null}
      </div>
      <p className="mt-0.5 text-muted-foreground">{shown}</p>
    </li>
  );
}

/** Starter questions shown under the mode row; more than a few is a wall. */
const STARTER_LIMIT = 4;

function uniqueQuestions(values: readonly string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    const key = value.trim().toLowerCase();
    if (!key || seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}
