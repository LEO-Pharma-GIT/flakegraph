export async function GET() {
  return Response.json({
    status: "ok",
    service: "flakegraph-control-plane",
    timestamp: new Date().toISOString(),
  });
}
