import { TRPCError } from "@trpc/server";
import { getHTTPStatusCodeFromError } from "@trpc/server/http";
import { appEnv } from "@/server/env";
import { qualityForGraph } from "@/server/quality";
import { inspectFromDataset, inspectHtml } from "@/server/inspect-report";
import { AskHttpError, resolveAskContext } from "@/server/ask/load";
import { readRunRecord, runDirectory, runRecordExists } from "@/server/catalog";
import { ControlPlaneError, toTrpcCode } from "@/server/protocol/errors";
import { runEffect } from "@/server/trpc/init";
import { isSafeId } from "@/server/confine";

export const runtime = "nodejs";

const RUNTIME_HEADER = "x-flakegraph-runtime";

/**
 * The inspect report of one run as a standalone HTML page, served as the
 * caller with their own graph access, like every other route.
 *
 * The runtime is the one `x-flakegraph-runtime` names. A link the console
 * renders cannot send a header, so a request that names none is served by
 * the runtime the run's own record names, else the console's default.
 */
export async function GET(request: Request) {
  const runId = new URL(request.url).searchParams.get("run");
  if (!runId) {
    return Response.json({ error: "Missing run" }, { status: 400 });
  }
  if (!isSafeId(runId)) {
    return Response.json({ error: "run may hold only letters, digits, dots, dashes and underscores" }, { status: 400 });
  }
  const headers = new Headers(request.headers);
  if (!headers.get(RUNTIME_HEADER)?.trim()) {
    const directory = runDirectory(appEnv().stateRoot, runId);
    const recorded = runRecordExists(directory) ? (await readRunRecord(directory)).runtime : null;
    if (recorded) {
      headers.set(RUNTIME_HEADER, String(recorded));
    }
  }
  try {
    const ctx = await resolveAskContext(headers);
    const snapshot = await runEffect(ctx.controlPlane.getRun(runId));
    const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
    const quality = await qualityForGraph({
      graphDirectory: ctx.controlPlane.graphDirectory(snapshot),
      snapshot,
      dataset,
      repositoryRoot: appEnv().repositoryRoot,
      stateRoot: appEnv().stateRoot,
    });
    const html = inspectHtml(inspectFromDataset(snapshot, dataset, quality));
    return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
  } catch (error) {
    if (error instanceof AskHttpError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    // The status the same failure gets through tRPC: 403 for a graph the
    // caller may not read, 404 for a run the runtime does not hold.
    if (error instanceof ControlPlaneError) {
      const status = getHTTPStatusCodeFromError(new TRPCError({ code: toTrpcCode(error), message: error.message }));
      return Response.json({ error: error.message }, { status });
    }
    throw error;
  }
}
