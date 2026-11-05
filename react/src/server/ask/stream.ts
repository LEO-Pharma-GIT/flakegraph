import "server-only";

import {
  createAgentUIStream,
  createUIMessageStream,
  createUIMessageStreamResponse,
  type UIMessage,
} from "ai";
import type { GraphDataset } from "../protocol/schema";
import { ASK_HEARTBEAT_MS, ASK_STREAM_TIMEOUT_MS } from "./constants";
import { AskModelMissingError, createAskAgent } from "./agent";
import { AskHttpError, isAbortError } from "./errors";
import { answerFromGrounding, emptyGrounding, groundingFromToolOutput, mergeGrounding } from "./grounding";
import { loadAskRunGraph, perspectiveScope } from "./load";
import { probeAskModel, resolveAskModel } from "./model";
import { encodeAskEvent } from "./protocol";
import { acceptFormat, clampTopK, extractQuestion, parseDocumentIds, parseMode, toUiUserMessage } from "./request";
import { lexicalAsk } from "./lexical";
import { mergeAskScope, isScoped } from "./scope";
import { ASK_TOOL_NAMES } from "./tools";
import type {
  AskAnswer,
  AskRequestBody,
  AskScope,
  AskStreamEvent,
  QueryMode,
  SearchProgressUpdate,
} from "./types";
import { PROGRESS_LABELS } from "./types";

export type AskUIMessage = UIMessage<never, { status: SearchProgressUpdate }>;

export async function handleAskRequest(request: Request): Promise<Response> {
  try {
    const body = (await request.json().catch(() => ({}))) as AskRequestBody;
    const format = acceptFormat(request, body);
    const runId = body.runId?.trim() || new URL(request.url).searchParams.get("runId") || "";
    if (!runId) {
      throw new AskHttpError(400, "runId is required.", "missing_run");
    }
    const question = extractQuestion(body);
    if (question.trim().length < 2) {
      throw new AskHttpError(400, "question must be at least 2 characters.", "missing_question");
    }
    const mode = parseMode(body.mode ?? new URL(request.url).searchParams.get("mode"));
    const topK = clampTopK(body.topK);
    const documentIds = parseDocumentIds(body.documentIds);
    const loaded = await loadAskRunGraph({ headers: request.headers, runId });
    const scope = mergeAskScope(
      await perspectiveScope(body.perspectiveId, loaded.snapshot.graphId),
      documentIds ? { documentIds } : undefined,
    );
    // Without a language model the graph still answers with its own text:
    // matching entities, relations and quotes. Both stream shapes carry it.
    const model = await probeAskModel();
    if (!model) {
      const answer = lexicalAsk(loaded.dataset, question, mode === "global" ? "global" : "local", scope);
      return format === "ndjson"
        ? lexicalNdjsonResponse(answer)
        : lexicalUiResponse(answer);
    }
    if (format === "ndjson") {
      return streamAskNdjson({
        request,
        question,
        mode,
        topK,
        dataset: loaded.dataset,
        scope,
      });
    }
    return streamAskUi({
      request,
      body,
      question,
      mode,
      topK,
      dataset: loaded.dataset,
      scope,
    });
  } catch (error) {
    return askErrorResponse(error);
  }
}

export function askCatalog(origin: string) {
  const model = resolveAskModel();
  return {
    service: "flakegraph-ask",
    origin,
    path: "/api/ask",
    methods: ["GET", "POST"],
    auth: {
      session: "Browser SSO / workspace identity",
      apiKey: ["Authorization: Bearer $FLAKEGRAPH_API_KEY", "x-flakegraph-api-key: $FLAKEGRAPH_API_KEY"],
    },
    formats: {
      ui: "AI SDK UI message stream for useChat / DefaultChatTransport",
      ndjson: "application/x-ndjson events: status, tool, text, citation, then one done or one error (timeout, no_answer, ...)",
    },
    body: {
      runId: "required",
      question: "string, or last user message in messages[]",
      messages: "UIMessage[] from the AI SDK",
      mode: "local | global | hybrid | drift | auto",
      perspectiveId: "optional workspace perspective",
      documentIds: "optional document id subset (same as hub fileIds)",
      format: "ui | ndjson",
      topK: "1-40",
    },
    tools: [...ASK_TOOL_NAMES],
    model: model
      ? { configured: true, provider: model.provider, billed: model.billed }
      : { configured: false, provider: null, billed: false },
  };
}

function streamAskUi(args: {
  request: Request;
  body: AskRequestBody;
  question: string;
  mode?: QueryMode;
  topK: number;
  dataset: GraphDataset;
  scope?: AskScope;
}) {
  const uiMessages =
    Array.isArray(args.body.messages) && args.body.messages.length > 0
      ? args.body.messages
      : [toUiUserMessage(args.question)];

  const stream = createUIMessageStream({
    execute: async ({ writer }) => {
      writer.write({
        type: "data-status",
        data: {
          phase: "loading_graph",
          message: PROGRESS_LABELS.loading_graph,
          detail: args.mode ? `mode=${args.mode}` : "mode=auto",
        },
        transient: true,
      });
      const agent = createAskAgent({
        dataset: args.dataset,
        scope: args.scope,
        preferredMode: args.mode,
        defaultTopK: args.topK,
        onProgress: (update) => {
          writer.write({ type: "data-status", data: update, transient: true });
        },
      });
      writer.write({
        type: "data-status",
        data: { phase: "answering", message: PROGRESS_LABELS.answering },
        transient: true,
      });
      const agentStream = await createAgentUIStream({
        agent,
        uiMessages,
        abortSignal: args.request.signal,
        timeout: ASK_STREAM_TIMEOUT_MS,
      });
      writer.merge(agentStream);
    },
  });
  return createUIMessageStreamResponse({ stream });
}

const LEXICAL_NOTE =
  "\n\n_No language model is configured for this console, so this is the graph's own text: the entities, relations and quotes that match, without a written answer._";

/** The lexical answer as the console's message stream: one tool result and one text. */
function lexicalUiResponse(answer: AskAnswer): Response {
  const toolCallId = "lexical";
  const stream = createUIMessageStream({
    execute: ({ writer }) => {
      writer.write({ type: "tool-input-available", toolCallId, toolName: "searchGraph", input: { query: answer.question, mode: answer.mode } });
      writer.write({
        type: "tool-output-available",
        toolCallId,
        output: {
          mode: answer.mode,
          citations: answer.citations,
          entityIds: answer.entityIds,
          relationIds: answer.relationIds,
          communityIds: answer.communityIds,
          counts: { entities: answer.entityIds.length, evidence: answer.citations.length },
        },
      });
      writer.write({ type: "text-start", id: "answer" });
      writer.write({ type: "text-delta", id: "answer", delta: `${answer.summary}${LEXICAL_NOTE}` });
      writer.write({ type: "text-end", id: "answer" });
    },
  });
  return createUIMessageStreamResponse({ stream });
}

/** The lexical answer as the programmatic stream: status, citations, text, done. */
function lexicalNdjsonResponse(answer: AskAnswer): Response {
  const encoder = new TextEncoder();
  const events: AskStreamEvent[] = [
    { type: "status", phase: "answering", message: PROGRESS_LABELS.answering, detail: "lexical" },
    ...answer.citations.map((citation): AskStreamEvent => ({ type: "citation", citation })),
    { type: "text", delta: answer.summary },
    { type: "done", answer },
  ];
  return new Response(encoder.encode(events.map(encodeAskEvent).join("")), {
    headers: { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store" },
  });
}

function streamAskNdjson(args: {
  request: Request;
  question: string;
  mode?: QueryMode;
  topK: number;
  dataset: GraphDataset;
  scope?: AskScope;
}): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      let lastSentAt = Date.now();
      const send = (event: AskStreamEvent) => {
        lastSentAt = Date.now();
        try {
          controller.enqueue(encoder.encode(encodeAskEvent(event)));
        } catch {
          // Consumer closed the stream.
        }
      };
      const heartbeat = setInterval(() => {
        if (Date.now() - lastSentAt >= ASK_HEARTBEAT_MS) {
          send({ type: "status", phase: "answering", message: PROGRESS_LABELS.answering });
        }
      }, ASK_HEARTBEAT_MS);
      try {
        send({ type: "status", phase: "loading_graph", message: PROGRESS_LABELS.loading_graph });
        const model = resolveAskModel();
        if (!model) {
          throw new AskModelMissingError();
        }
        const agent = createAskAgent({
          dataset: args.dataset,
          scope: args.scope,
          preferredMode: args.mode,
          defaultTopK: args.topK,
          onProgress: (update) => send({ type: "status", ...update }),
        });
        send({ type: "status", phase: "answering", message: PROGRESS_LABELS.answering });
        const uiStream = await createAgentUIStream({
          agent,
          uiMessages: [toUiUserMessage(args.question)],
          abortSignal: args.request.signal,
          timeout: ASK_STREAM_TIMEOUT_MS,
        });
        const toolNames = new Map<string, string>();
        const textParts: string[] = [];
        let timedOut = false;
        let grounding = emptyGrounding();
        const seenCitations = new Set<string>();
        for await (const chunk of uiStream) {
          if (args.request.signal.aborted) {
            throw new DOMException("Ask aborted", "AbortError");
          }
          if (chunk.type === "text-delta" && chunk.delta) {
            textParts.push(chunk.delta);
            send({ type: "text", delta: chunk.delta });
          }
          if (chunk.type === "tool-input-start") {
            toolNames.set(chunk.toolCallId, chunk.toolName);
            send({ type: "tool", name: chunk.toolName, state: "start" });
          }
          if (chunk.type === "tool-input-available") {
            toolNames.set(chunk.toolCallId, chunk.toolName);
            send({ type: "tool", name: chunk.toolName, state: "start", input: chunk.input });
          }
          if (chunk.type === "tool-output-available") {
            const name = toolNames.get(chunk.toolCallId) ?? "tool";
            send({ type: "tool", name, state: "done", output: chunk.output });
            const next = groundingFromToolOutput(chunk.output);
            grounding = mergeGrounding(grounding, next);
            for (const citation of next.citations) {
              const key = `${citation.documentId}:${citation.quote}`;
              if (seenCitations.has(key)) {
                continue;
              }
              seenCitations.add(key);
              send({ type: "citation", citation });
            }
          }
          if (chunk.type === "error") {
            send({ type: "error", message: chunk.errorText, code: "model" });
          }
          // The caller is still there, so the time limit ended the answer.
          if (chunk.type === "abort" && !args.request.signal.aborted) {
            timedOut = true;
          }
        }
        // A question without an answer ends in an error, never in a `done`
        // that a caller could pass on as the graph's answer.
        const summary = textParts.join("").trim();
        if (timedOut) {
          send({
            type: "error",
            message: `The graph did not answer within ${ASK_STREAM_TIMEOUT_MS / 60_000} minutes.`,
            code: "timeout",
          });
          return;
        }
        if (!summary) {
          send({ type: "error", message: "The graph produced no answer to this question.", code: "no_answer" });
          return;
        }
        send({
          type: "done",
          answer: answerFromGrounding({
            question: args.question,
            mode: args.mode ?? grounding.mode ?? "hybrid",
            summary,
            scoped: isScoped(args.scope),
            billed: model.billed,
            grounding,
          }),
        });
      } catch (error) {
        if (isAbortError(error) || args.request.signal.aborted) {
          send({ type: "error", message: "Ask was aborted.", code: "aborted" });
        } else if (error instanceof AskModelMissingError) {
          send({ type: "error", message: error.message, code: "model_missing" });
        } else if (error instanceof AskHttpError) {
          send({ type: "error", message: error.message, code: error.code });
        } else {
          send({
            type: "error",
            message: error instanceof Error ? error.message : "Ask failed",
            code: "internal",
          });
        }
      } finally {
        clearInterval(heartbeat);
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: {
      "content-type": "application/x-ndjson; charset=utf-8",
      "cache-control": "no-store",
      "x-flakegraph-ask-protocol": "ndjson-v1",
    },
  });
}

export function askErrorResponse(error: unknown): Response {
  if (error instanceof AskHttpError) {
    return Response.json({ error: error.message, code: error.code }, { status: error.status });
  }
  if (error instanceof AskModelMissingError) {
    return Response.json({ error: error.message, code: "model_missing" }, { status: 503 });
  }
  if (isAbortError(error)) {
    return Response.json({ error: "Ask was aborted.", code: "aborted" }, { status: 499 });
  }
  return Response.json(
    { error: error instanceof Error ? error.message : "Ask failed", code: "internal" },
    { status: 500 },
  );
}
