import { askableGraphs } from "@/lib/catalog-graphs";
import { AskHttpError } from "@/server/ask/errors";
import { resolveAskContext } from "@/server/ask/load";
import { displayName } from "@/server/protocol/schema";
import { runEffect } from "@/server/trpc/init";

export const runtime = "nodejs";

/** How many of the viewer's newest runs are searched for finished graphs. */
const RUNS_SEARCHED = 500;

/**
 * The graphs the caller may ask, for a program choosing one on a person's
 * behalf: each graph the caller can read, at its newest finished version.
 * `runId` is what POST /api/ask takes.
 */
export async function GET(request: Request) {
  let ctx;
  try {
    ctx = await resolveAskContext(request.headers);
  } catch (error) {
    if (error instanceof AskHttpError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    throw error;
  }
  const runs = await runEffect(ctx.controlPlane.listRuns(RUNS_SEARCHED));
  return Response.json({
    graphs: askableGraphs(runs).map((run) => ({
      runId: run.runId,
      graphId: run.graphId,
      name: displayName(run),
      documentCount: run.documentsCompleted,
      updatedAt: run.updatedAt,
    })),
  });
}
