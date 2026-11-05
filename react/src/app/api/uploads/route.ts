import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { appEnv } from "@/server/env";
import { authorizeRequest, mayMutate, refusal, SIGN_IN_HINT } from "@/server/auth";
import { isSafeId, safeUploadPath } from "@/server/confine";
import { claimUploadFolder } from "@/server/access/controlled-plane";

/** Files per request and bytes per file; a corpus larger than this is a folder or a bucket, not a drop. */
const UPLOAD_MAX_FILES = 200;
const UPLOAD_MAX_FILE_BYTES = 200 * 1024 * 1024;
const UPLOAD_MAX_TOTAL_BYTES = 1024 * 1024 * 1024;

export async function POST(request: Request) {
  let authorized;
  try {
    authorized = await authorizeRequest(request.headers);
  } catch (error) {
    return refusal(error) ?? Promise.reject(error);
  }
  if (!mayMutate(authorized)) {
    return Response.json({ error: SIGN_IN_HINT }, { status: 401 });
  }
  const form = await request.formData();
  const files = form.getAll("files").filter((value): value is File => value instanceof File);
  // Each file's path inside a dropped folder, in the same order as the files.
  const paths = form.getAll("paths").map(String);
  if (paths.length > 0 && paths.length !== files.length) {
    return Response.json({ error: "Send one path per file, or none." }, { status: 400 });
  }
  if (files.length === 0) {
    return Response.json({ error: "No files uploaded" }, { status: 400 });
  }
  if (files.length > UPLOAD_MAX_FILES) {
    return Response.json({ error: `At most ${UPLOAD_MAX_FILES} files per upload; point at a folder or a bucket instead.` }, { status: 413 });
  }
  let total = 0;
  for (const file of files) {
    if (file.size > UPLOAD_MAX_FILE_BYTES) {
      return Response.json({ error: `${file.name} is larger than ${UPLOAD_MAX_FILE_BYTES / (1024 * 1024)} MB.` }, { status: 413 });
    }
    total += file.size;
  }
  if (total > UPLOAD_MAX_TOTAL_BYTES) {
    return Response.json({ error: `An upload may hold at most ${UPLOAD_MAX_TOTAL_BYTES / (1024 * 1024)} MB in all.` }, { status: 413 });
  }
  // The job id names a directory under the console's state. A second drop
  // may add to a folder this console minted; it may not name any other.
  const requested = String(form.get("jobId") ?? "").trim();
  if (requested && !isSafeId(requested)) {
    return Response.json({ error: "jobId may hold only letters, digits, dots, dashes and underscores." }, { status: 400 });
  }
  const jobId = requested || randomUUID();
  // A folder is its first uploader's: their documents, which nobody else may
  // add to (or, through a run, read).
  if (!(await claimUploadFolder(appEnv().stateRoot, jobId, authorized.viewer))) {
    return Response.json({ error: "That upload folder belongs to someone else." }, { status: 403 });
  }
  const directory = path.join(appEnv().stateRoot, "uploads", jobId);
  await mkdir(directory, { recursive: true });
  for (const [index, file] of files.entries()) {
    const target = path.join(directory, safeUploadPath(paths[index] ?? file.name));
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, Buffer.from(await file.arrayBuffer()));
  }
  return Response.json({ path: directory, count: files.length, jobId });
}
