/**
 * A `fetch` for the Ask panel's chat transport that answers through a polled
 * job instead of one long request, for consoles whose ingress cuts requests
 * short (Snowflake's ends every request after about 90 seconds).
 *
 * `POST /api/ask` becomes `POST /api/ask/jobs`; the response it returns is
 * built from polls of `GET /api/ask/jobs/{id}`, byte for byte what the direct
 * request would have streamed, so the transport parses it unchanged. Stopping
 * the answer stops the job.
 */

const POLL_MS = 1_000;

interface Progress {
  status: number | null;
  contentType: string | null;
  text: string;
  next: number;
  done: boolean;
}

const sleep = (ms: number, signal?: AbortSignal | null) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    });
  });

export const polledAskFetch: typeof fetch = async (input, init) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const base = url.replace(/\/api\/ask(?:\?.*)?$/, "");
  const headers = new Headers(init?.headers);
  const signal = init?.signal ?? null;
  const started = await fetch(`${base}/api/ask/jobs`, { method: "POST", headers, body: init?.body, signal });
  if (!started.ok) {
    return started;
  }
  const { jobId } = (await started.json()) as { jobId: string };
  const jobUrl = `${base}/api/ask/jobs/${encodeURIComponent(jobId)}`;
  const cancel = () => void fetch(jobUrl, { method: "DELETE", headers }).catch(() => undefined);
  signal?.addEventListener("abort", cancel);
  const poll = async (after: number): Promise<Progress> => {
    const response = await fetch(`${jobUrl}?after=${after}`, { headers, signal });
    if (!response.ok) {
      throw new Error(`The answer could not be followed (${response.status}).`);
    }
    return (await response.json()) as Progress;
  };

  // The status and type of the answer's response are known once it starts.
  let first = await poll(0);
  while (first.status == null && !first.done) {
    await sleep(POLL_MS, signal);
    first = await poll(0);
  }
  const status = first.status ?? 500;
  const encoder = new TextEncoder();
  let next = first.next;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      if (first.text) controller.enqueue(encoder.encode(first.text));
      if (first.done) controller.close();
    },
    async pull(controller) {
      try {
        await sleep(POLL_MS, signal);
        const progress = await poll(next);
        if (progress.text) controller.enqueue(encoder.encode(progress.text));
        next = progress.next;
        if (progress.done) controller.close();
      } catch (error) {
        controller.error(error);
      }
    },
    cancel,
  });
  return new Response(first.done && !first.text ? null : body, {
    status,
    headers: { "content-type": first.contentType ?? "text/event-stream" },
  });
};
