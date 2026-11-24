import "server-only";

import { randomUUID } from "node:crypto";
import { handleAskRequest } from "./stream";

/**
 * Questions answered as jobs the page polls, for consoles whose ingress ends a
 * request before an answer is done - Snowflake's cuts every request at about 90
 * seconds, streamed or not, and a question that reads many documents takes
 * minutes.
 *
 * A job runs exactly what `POST /api/ask` runs and keeps the response it
 * streams, as text; the page asks for what arrived since its last poll and
 * feeds it to the same stream parser a direct request would. Jobs live in the
 * console's memory, which is enough for a console of one instance (as the
 * Snowflake app is), and are dropped an hour after they were last read.
 */

const JOB_IDLE_MS = 60 * 60_000;
/** Jobs one person may have running at once. */
const RUNNING_PER_OWNER = 4;

interface AskJob {
  id: string;
  owner: string;
  /** The response's status, once the answer has started. */
  status: number | null;
  contentType: string | null;
  text: string;
  done: boolean;
  controller: AbortController;
  touched: number;
}

const jobs = new Map<string, AskJob>();

function sweep(now = Date.now()): void {
  for (const [id, job] of jobs) {
    if (now - job.touched > JOB_IDLE_MS) {
      job.controller.abort();
      jobs.delete(id);
    }
  }
}

export class AskJobRefused extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** Start answering a question as a job; returns its id at once. */
export async function startAskJob(request: Request, owner: string): Promise<string> {
  sweep();
  const running = [...jobs.values()].filter((job) => job.owner === owner && !job.done).length;
  if (running >= RUNNING_PER_OWNER) {
    throw new AskJobRefused(`At most ${RUNNING_PER_OWNER} questions run at once; wait for one to finish.`, 429);
  }
  const body = await request.text();
  const job: AskJob = {
    id: randomUUID(),
    owner,
    status: null,
    contentType: null,
    text: "",
    done: false,
    controller: new AbortController(),
    touched: Date.now(),
  };
  jobs.set(job.id, job);
  const inner = new Request(new URL("/api/ask", request.url), {
    method: "POST",
    headers: request.headers,
    body,
    signal: job.controller.signal,
  });
  void (async () => {
    try {
      const response = await handleAskRequest(inner);
      job.status = response.status;
      job.contentType = response.headers.get("content-type");
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      for (let part = await reader?.read(); part && !part.done; part = await reader!.read()) {
        job.text += decoder.decode(part.value, { stream: true });
      }
      job.text += decoder.decode();
    } catch (error) {
      job.status ??= 500;
      job.text += JSON.stringify({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      job.done = true;
    }
  })();
  return job.id;
}

export interface AskJobProgress {
  /** The response's status, once known; null while the answer has not started. */
  status: number | null;
  contentType: string | null;
  /** What the response streamed after `after` characters. */
  text: string;
  /** Where the next poll starts. */
  next: number;
  done: boolean;
}

/** What a job has streamed since `after`, for its owner only. */
export function readAskJob(id: string, owner: string, after: number): AskJobProgress | null {
  const job = jobs.get(id);
  if (!job || job.owner !== owner) {
    return null;
  }
  job.touched = Date.now();
  const start = Math.max(0, Math.min(after, job.text.length));
  return {
    status: job.status,
    contentType: job.contentType,
    text: job.text.slice(start),
    next: job.text.length,
    done: job.done,
  };
}

/** Stop a job its owner no longer waits for. */
export function cancelAskJob(id: string, owner: string): boolean {
  const job = jobs.get(id);
  if (!job || job.owner !== owner) {
    return false;
  }
  job.controller.abort();
  jobs.delete(id);
  return true;
}

/** Who a job belongs to: the signed-in person, or the machine key's name. */
export function jobOwner(authorized: { viewer: { userName: string; email?: string } }): string {
  return authorized.viewer.email || authorized.viewer.userName || "anonymous";
}
