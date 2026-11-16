import "server-only";

import { TRPCError } from "@trpc/server";
import { ControlPlaneError } from "../protocol/errors";
import { isSuccessStatus, type GraphDataset, type RunSnapshot } from "../protocol/schema";
import { createContext, runEffect } from "../trpc/init";
import { loadWorkspace } from "../workspace";
import { AskHttpError } from "./errors";
import type { AskScope } from "./types";

export { AskHttpError } from "./errors";

/** What a console with no question pipeline answers: its models are for building graphs. */
export const NO_ASK_MODEL = "This console has no model to answer questions with. Explore the graph, or ask it from a console that has one.";

export async function resolveAskContext(headers: Headers) {
  try {
    return await createContext({ headers });
  } catch (error) {
    if (error instanceof TRPCError && error.code === "UNAUTHORIZED") {
      throw new AskHttpError(401, error.message, "unauthorized");
    }
    throw error;
  }
}

export async function loadAskRunGraph(args: {
  headers: Headers;
  runId: string;
}): Promise<{ dataset: GraphDataset; snapshot: RunSnapshot }> {
  const ctx = await resolveAskContext(args.headers);
  if (!ctx.controlPlane.capabilities.has("ask")) {
    throw new AskHttpError(409, NO_ASK_MODEL, "no_model");
  }
  let snapshot: RunSnapshot;
  try {
    snapshot = await runEffect(ctx.controlPlane.getRun(args.runId));
  } catch (error) {
    if (error instanceof ControlPlaneError && error.code === "not_found") {
      throw new AskHttpError(404, `Run ${args.runId} was not found.`, "not_found");
    }
    throw new AskHttpError(404, `Run ${args.runId} was not found.`, "not_found");
  }
  if (!isSuccessStatus(snapshot.status)) {
    throw new AskHttpError(
      409,
      `Graph ${args.runId} is ${snapshot.status}. Ask requires a finished graph.`,
      "not_ready",
    );
  }
  try {
    const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot, { purpose: "retrieval" }));
    return { dataset, snapshot };
  } catch (error) {
    throw new AskHttpError(
      409,
      error instanceof Error ? error.message : "Graph artifacts are not available.",
      "artifacts_unavailable",
    );
  }
}

export async function perspectiveScope(
  perspectiveId: string | undefined,
  graphId: string,
): Promise<AskScope | undefined> {
  const id = perspectiveId?.trim();
  if (!id) {
    return undefined;
  }
  const workspace = await loadWorkspace();
  const perspective = workspace.perspectives.find((item) => item.id === id && item.graphId === graphId);
  if (!perspective) {
    throw new AskHttpError(400, `Perspective ${id} was not found on this graph.`, "unknown_perspective");
  }
  return { search: perspective.search, communityIds: perspective.communityIds };
}
