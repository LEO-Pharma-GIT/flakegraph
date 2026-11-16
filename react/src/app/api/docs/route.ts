import {
  API_ACCESS,
  API_AREAS,
  CONTROL_PLANE_API,
  CONTROL_PLANE_PROCEDURES,
  CONTROL_PLANE_ROUTES,
} from "@/lib/control-plane-api";

/** The catalog's entry for one route, which the summaries below quote rather than restate. */
function route(name: string, kind: "GET" | "POST") {
  const entry = CONTROL_PLANE_ROUTES.find((item) => item.name === name && item.kind === kind);
  if (!entry) {
    throw new Error(`The catalog has no ${kind} ${name}`);
  }
  return entry;
}

export async function GET(request: Request) {
  const origin = new URL(request.url).origin;
  const graphs = route(CONTROL_PLANE_API.graphsPath, "GET");
  return Response.json({
    service: "flakegraph-control-plane",
    origin,
    endpoint: `${origin}${CONTROL_PLANE_API.endpoint}`,
    ask: `${origin}${CONTROL_PLANE_API.askPath}`,
    graphs: `${origin}${CONTROL_PLANE_API.graphsPath}`,
    inspect: `${origin}${CONTROL_PLANE_API.inspectPath}`,
    uploads: `${origin}${CONTROL_PLANE_API.uploadsPath}`,
    humanGuide: `${origin}${CONTROL_PLANE_API.humanGuidePath}`,
    health: `${origin}${CONTROL_PLANE_API.healthPath}`,
    auth: {
      environmentVariable: "FLAKEGRAPH_API_KEY",
      headers: ["Authorization: Bearer $FLAKEGRAPH_API_KEY", "x-flakegraph-api-key: $FLAKEGRAPH_API_KEY"],
      runtimeHeader: "x-flakegraph-runtime: local | kubernetes | snowflake",
      note: "A presented key is served as the person who minted it, with that person's graph access, and never as a browser session: minting or revoking keys and changing the console's own settings answer 403 to a key. 401 means the key is missing or revoked.",
      delegated: "Authorization: Bearer <access token for the sign-in gate's client>. Where the gate accepts bearer tokens, the request is the person the token names, with their own graph access.",
    },
    access: API_ACCESS,
    areas: API_AREAS,
    wireFormat: CONTROL_PLANE_API.wireFormat,
    trpcHttpDocs: CONTROL_PLANE_API.trpcHttpDocs,
    sourceOfTruth: {
      procedures: CONTROL_PLANE_API.checkoutProcedures,
      payloads: CONTROL_PLANE_API.checkoutPayloads,
      ask: CONTROL_PLANE_API.checkoutAsk,
      fleetMachinePaths: CONTROL_PLANE_API.fleetGuide,
    },
    graphList: {
      path: `${origin}${CONTROL_PLANE_API.graphsPath}`,
      methods: ["GET"],
      response: graphs.returns,
      note: graphs.purpose,
    },
    askStream: {
      path: `${origin}${CONTROL_PLANE_API.askPath}`,
      methods: ["GET", "POST"],
      formats: ["ui", "ndjson"],
      events: ["status", "tool", "text", "citation", "error", "done"],
      note: "POST /api/ask streams. graphs.ask is the non-streaming JSON convenience wrapper.",
    },
    procedures: CONTROL_PLANE_PROCEDURES,
    routes: CONTROL_PLANE_ROUTES,
  });
}
