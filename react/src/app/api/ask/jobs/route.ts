import { AskJobRefused, jobOwner, startAskJob } from "@/server/ask/jobs";
import { probeAskModel } from "@/server/ask/model";
import { authorizeRequest, mayMutate, refusal, SIGN_IN_HINT } from "@/server/auth";

export const runtime = "nodejs";

/**
 * Start answering a question as a job, for a console whose ingress ends
 * requests before an answer is done. The body is `POST /api/ask`'s; poll
 * `GET /api/ask/jobs/{id}` for what it streams.
 */
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
  await probeAskModel();
  try {
    return Response.json({ jobId: await startAskJob(request, jobOwner(authorized)) }, { status: 202 });
  } catch (error) {
    if (error instanceof AskJobRefused) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    throw error;
  }
}
