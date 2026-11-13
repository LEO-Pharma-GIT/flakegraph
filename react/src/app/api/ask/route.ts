import { askCatalog, handleAskRequest } from "@/server/ask/stream";
import { probeAskModel } from "@/server/ask/model";
import { authorizeRequest, mayMutate, refusal, SIGN_IN_HINT } from "@/server/auth";

export const runtime = "nodejs";
export const maxDuration = 1_200;

export async function GET(request: Request) {
  await probeAskModel();
  return Response.json(askCatalog(new URL(request.url).origin));
}

export async function POST(request: Request) {
  let authorized;
  try {
    authorized = await authorizeRequest(request.headers);
  } catch (error) {
    return refusal(error) ?? Promise.reject(error);
  }
  // A question runs a model; that is spending, not reading.
  if (!mayMutate(authorized)) {
    return Response.json({ error: SIGN_IN_HINT }, { status: 401 });
  }
  await probeAskModel();
  return handleAskRequest(request);
}
