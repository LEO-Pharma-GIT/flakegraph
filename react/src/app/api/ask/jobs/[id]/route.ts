import { cancelAskJob, jobOwner, readAskJob } from "@/server/ask/jobs";
import { authorizeRequest, refusal } from "@/server/auth";

export const runtime = "nodejs";

async function owner(request: Request): Promise<string | Response> {
  try {
    return jobOwner(await authorizeRequest(request.headers));
  } catch (error) {
    return refusal(error) ?? Promise.reject(error);
  }
}

/** What a question's job streamed since `?after=` characters, and whether it is done. */
export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const who = await owner(request);
  if (who instanceof Response) return who;
  const { id } = await context.params;
  const after = Number.parseInt(new URL(request.url).searchParams.get("after") ?? "0", 10) || 0;
  const progress = readAskJob(id, who, after);
  return progress ? Response.json(progress) : Response.json({ error: "No such question." }, { status: 404 });
}

/** Stop a question nobody waits for any more. */
export async function DELETE(request: Request, context: { params: Promise<{ id: string }> }) {
  const who = await owner(request);
  if (who instanceof Response) return who;
  const { id } = await context.params;
  return cancelAskJob(id, who) ? new Response(null, { status: 204 }) : Response.json({ error: "No such question." }, { status: 404 });
}
