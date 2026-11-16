/**
 * Sending one batch of an upload, with what fetch cannot give: how many bytes
 * have left the browser so far. A folder of large PDFs takes minutes over a
 * slow link, and a count that only moves when a whole request is answered
 * reads as a stalled page.
 */

export interface UploadAnswer {
  path?: string;
  error?: string;
  jobId?: string;
  count?: number;
}

export class UploadFailed extends Error {
  constructor(
    message: string,
    /** Whether sending the same batch again may succeed: the link, not the request, failed. */
    readonly retryable: boolean,
  ) {
    super(message);
  }
}

export class UploadCancelled extends Error {
  constructor() {
    super("Upload cancelled");
  }
}

/** POST a form, reporting bytes sent; aborted through `signal`. */
export function postUpload(
  url: string,
  body: FormData,
  options: { onProgress?: (sentBytes: number) => void; signal?: AbortSignal } = {},
): Promise<UploadAnswer> {
  return new Promise((resolve, reject) => {
    if (options.signal?.aborted) {
      reject(new UploadCancelled());
      return;
    }
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.responseType = "text";
    request.upload.onprogress = (event) => options.onProgress?.(event.loaded);
    request.onload = () => {
      let answer: UploadAnswer = {};
      try {
        answer = JSON.parse(request.responseText || "{}") as UploadAnswer;
      } catch {
        // A proxy's own error page: say what status it was.
      }
      if (request.status >= 200 && request.status < 300 && answer.path) {
        resolve(answer);
        return;
      }
      reject(
        new UploadFailed(
          answer.error || `the server answered ${request.status}`,
          request.status >= 500 || request.status === 408 || request.status === 429,
        ),
      );
    };
    request.onerror = () => reject(new UploadFailed("the connection dropped", true));
    request.ontimeout = () => reject(new UploadFailed("the upload timed out", true));
    request.onabort = () => reject(new UploadCancelled());
    options.signal?.addEventListener("abort", () => request.abort(), { once: true });
    request.send(body);
  });
}

/** Attempts per batch before an upload stops: a dropped connection is tried again. */
export const UPLOAD_ATTEMPTS = 3;

/** Send a batch, trying again after a failure that was the link's rather than the request's. */
export async function postUploadWithRetry(
  url: string,
  makeBody: () => FormData,
  options: { onProgress?: (sentBytes: number) => void; signal?: AbortSignal } = {},
): Promise<UploadAnswer> {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await postUpload(url, makeBody(), options);
    } catch (error) {
      options.onProgress?.(0);
      if (!(error instanceof UploadFailed) || !error.retryable || attempt >= UPLOAD_ATTEMPTS) {
        throw error;
      }
      await new Promise((settle) => setTimeout(settle, 1_000 * attempt));
    }
  }
}

/** "84 MB", "1.2 GB": sizes as a person reads them. */
export function formatMegabytes(bytes: number): string {
  if (bytes >= 1024 ** 3) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  }
  const megabytes = bytes / 1024 ** 2;
  return `${megabytes >= 10 ? Math.round(megabytes) : megabytes.toFixed(1)} MB`;
}
