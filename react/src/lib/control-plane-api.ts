export const CONTROL_PLANE_API = {
  endpoint: "/api/trpc",
  askPath: "/api/ask",
  graphsPath: "/api/graphs",
  docsPath: "/api/docs",
  inspectPath: "/api/inspect",
  uploadsPath: "/api/uploads",
  humanGuidePath: "/?page=keys",
  healthPath: "/api/health",
  wireFormat: "tRPC HTTP with SuperJSON envelopes ({ json: … }). Queries are GET with ?input=; mutations are POST with a JSON body. Ask streaming is a separate POST /api/ask that emits AI SDK UI chunks or application/x-ndjson.",
  trpcHttpDocs: "https://trpc.io/docs/rpc",
  checkoutProcedures: "react/src/server/trpc/router.ts",
  checkoutPayloads: "react/src/server/protocol/schema.ts",
  checkoutAsk: "react/src/server/ask/",
  fleetGuide: "docs/kubernetes-fleet.md (Callers that are programs)",
} as const;

/**
 * Who an endpoint serves, as `react/src/server/auth.ts` and the procedure
 * builders in `react/src/server/trpc/init.ts` decide it.
 *
 * - `open`: nothing is checked; whoever reaches the console is answered.
 * - `read` (`publicProcedure`): anyone the console admits. That is a caller
 *   presenting an API key, a person the sign-in gate names (a browser session,
 *   or an application presenting that person's access token, which the gate
 *   verifies), the operator a single-operator deployment pins, a demo session,
 *   and an anonymous caller only where FLAKEGRAPH_APP_ANONYMOUS_READ is on.
 * - `write` (`protectedProcedure`): the same, never an anonymous caller.
 * - `named` (`identifiedProcedure`): write authority and a person's name. A
 *   key carries the name of whoever minted it; a key with no recorded owner,
 *   or a demo session that assumed nobody, carries none.
 * - `session` (`sessionProcedure`): a named person in their own session,
 *   never a request that presented an API key or an access token.
 * - `operator` (`operatorProcedure`): whoever administers the console, in a
 *   session of their own: the people and roles FLAKEGRAPH_APP_OPERATORS names
 *   on a console the gate or Snowflake serves, the pinned single operator, or
 *   a demo session.
 *
 * Every read is further limited to the graphs the caller owns or has been
 * given, and every change to the level `graph` names.
 */
export type ApiAccess = "open" | "read" | "write" | "named" | "session" | "operator";

export const API_ACCESS: Record<ApiAccess, { label: string; description: string }> = {
  open: {
    label: "Open",
    description: "Answered without any check: whoever reaches the console. On a fleet the sign-in gate still stands in front unless the caller presents a key.",
  },
  read: {
    label: "Admitted",
    description:
      "Any caller the console admits: an API key, a person the sign-in gate names (browser session or delegated access token), a pinned single operator, or a demo session. Anonymous callers only where FLAKEGRAPH_APP_ANONYMOUS_READ is on. Results cover only graphs the caller can read.",
  },
  write: {
    label: "Key or sign-in",
    description:
      "Any admitted caller except an anonymous one: an API key, a person the gate names, a pinned operator, or a demo session. Refused with 401 otherwise.",
  },
  named: {
    label: "Named person",
    description:
      "Write authority and a person's name. A key counts as the person who minted it; a key with no recorded owner, or a demo session that assumed nobody, is refused with 401.",
  },
  session: {
    label: "Browser session",
    description:
      "A named person signed in through a browser session. A request presenting an API key or an access token is refused with 403, whoever it names, so a key cannot mint or revoke keys.",
  },
  operator: {
    label: "Operator",
    description:
      "Whoever administers the console, in a session of their own: on a console behind the sign-in gate or in Snowflake, the people and roles FLAKEGRAPH_APP_OPERATORS names (nobody while it names none); the pinned single operator; a demo session. Anyone else, and any API key or access token, is refused with 403.",
  },
};

/** The graph-level standing a call needs on top of its access level. */
export type ApiGraphRole = "read" | "write" | "owner";

export const API_AREAS = [
  { id: "auth", label: "Session", description: "Who a request is served as, and the console's own session settings." },
  { id: "runs", label: "Runs", description: "Start, watch, retry and cancel ingestion runs." },
  { id: "ingestion", label: "Ingestion", description: "What the compose form checks before a run starts." },
  { id: "graphs", label: "Graphs", description: "Read, ask, share, publish and curate built graphs." },
  { id: "workspace", label: "Workspace", description: "Perspectives, reviews, watches, versions and API keys the console keeps." },
  { id: "fleet", label: "Fleet", description: "The Kubernetes fleet this console was deployed for." },
  { id: "rest", label: "REST routes", description: "Plain HTTP routes outside tRPC: streaming ask, uploads, reports, health." },
] as const;

export type ApiArea = (typeof API_AREAS)[number]["id"];

export interface ApiEndpoint {
  /** A tRPC procedure's dotted path, or a route's path under the origin. */
  name: string;
  /** A procedure's kind (queries are GET, mutations POST), or the HTTP method a route answers. */
  kind: "query" | "mutation" | "GET" | "POST";
  area: ApiArea;
  access: ApiAccess;
  /** What the caller must be on the graph the call touches, where it touches one. */
  graph?: ApiGraphRole;
  purpose: string;
  /** The fields the call takes, from its schema; `none` for no input. */
  input: string;
  returns: string;
  /** Where the endpoint is narrower than its access level, or easy to misread. */
  limits?: string;
}

const WORKSPACE = "The caller's workspace, as workspace.get returns it";
const INGESTION_REQUEST =
  "IngestionRequest (react/src/server/protocol/schema.ts): runtime, jobId, graphId, graphName, sourceKind, source, ocr, llm, embedding, output, baseConfigPath, and optional includeGlobs, cacheProvider, providerParallelism (1-64), runtimeOptions, ontology, revision, priority";
const SOURCE_CONFINED =
  "A local folder must lie under the checkout, the state root or FLAKEGRAPH_APP_SOURCE_ROOTS; an upload folder is readable only by its uploader or by readers of a graph built from it.";
const REQUEST_CONFINED = `${SOURCE_CONFINED} A run id (jobId) already in use is refused.`;
const TARGET_GRAPH = "Write access is needed only when graphId names a graph that already exists.";
const ESTIMATE = "{ objectCount, runtime, preset, usdLow, usdHigh, minutesLow, minutesHigh, note }";

/**
 * Every endpoint the console serves: each tRPC procedure in
 * `react/src/server/trpc/router.ts` and each method of each route under
 * `react/src/app/api/`. A test holds this list to the router and the route
 * files, so an endpoint cannot be added or removed without its entry.
 */
export const CONTROL_PLANE_ENDPOINTS: readonly ApiEndpoint[] = [
  // ---------------------------------------------------------------- auth
  {
    name: "auth.session",
    kind: "query",
    area: "auth",
    access: "read",
    purpose: "Who the request is served as and what this console offers it; a cheap check that a key works.",
    input: "none",
    returns:
      "{ viewer, identified, authorization (machine | gate | operator | demo | anonymous), operator (whether the caller administers the console), runtime, capabilities, availableRuntimes, samplePacks, role, suggestionMode, grants, … }",
  },
  {
    name: "auth.assume",
    kind: "mutation",
    area: "auth",
    access: "operator",
    purpose: "Assume a demo identity and shell role on a console without a sign-in gate.",
    input: "{ userName, email?, roles: string[], role: operator | analyst | staff }",
    returns: WORKSPACE,
    limits:
      "Demo consoles only: 403 where the gate's identity is trusted or FLAKEGRAPH_DEMO_IDENTITIES is off. The assumed identity is the whole console's, not one session's, so a key never assumes one. The shell role changes only what the console shows.",
  },
  {
    name: "auth.setSuggestions",
    kind: "mutation",
    area: "auth",
    access: "operator",
    purpose: "Set how the console suggests: off, on request, or auto-fill the compose form from the last run.",
    input: "{ mode: off | on-request | auto-fill }",
    returns: WORKSPACE,
    limits: "A console-wide setting, not a per-person one.",
  },

  // ---------------------------------------------------------------- runs
  {
    name: "runs.list",
    kind: "query",
    area: "runs",
    access: "read",
    purpose: "Runs of the graphs the caller can read, newest first.",
    input: "{ limit?: 1-500 } (default 100)",
    returns: "RunSnapshot[]: runId, graphId, graphName, status, stages, document counts, error, owner, access, …",
  },
  {
    name: "runs.get",
    kind: "query",
    area: "runs",
    access: "read",
    graph: "read",
    purpose: "One run's status, stages, events, document counts and error; poll it until the run ends.",
    input: "{ runId }",
    returns: "RunSnapshot",
  },
  {
    name: "runs.submit",
    kind: "mutation",
    area: "runs",
    access: "write",
    graph: "write",
    purpose: "Start an ingestion run with the payload the compose form sends. A new graph id is claimed by its submitter.",
    input: INGESTION_REQUEST,
    returns: "RunSnapshot of the new run",
    limits: `${REQUEST_CONFINED} Credentials are named as KG_* variables set on the console, never sent as values.`,
  },
  {
    name: "runs.cancel",
    kind: "mutation",
    area: "runs",
    access: "write",
    graph: "write",
    purpose: "Cancel a queued or running run.",
    input: "{ runId }",
    returns: "RunSnapshot",
  },
  {
    name: "runs.retry",
    kind: "mutation",
    area: "runs",
    access: "write",
    graph: "write",
    purpose: "Queue a run's failed documents again, leaving out files marked with runs.skipFile.",
    input: "{ runId }",
    returns: "RunSnapshot",
    limits: "Kubernetes and Snowflake runs; a local run is refused.",
  },
  {
    name: "runs.recover",
    kind: "mutation",
    area: "runs",
    access: "write",
    graph: "write",
    purpose: "Restart a stuck run's worker pool when no worker is available; task state is not touched.",
    input: "{ runId }",
    returns: "A message saying what was done",
    limits: "Kubernetes runs only.",
  },
  {
    name: "runs.skipFile",
    kind: "mutation",
    area: "runs",
    access: "write",
    graph: "write",
    purpose: "Mark a document to leave out when the run is retried.",
    input: "{ runId, fileId }",
    returns: WORKSPACE,
  },
  {
    name: "runs.ontology",
    kind: "query",
    area: "runs",
    access: "read",
    graph: "read",
    purpose: "The entity and relation types a run was built with, for a revision to keep.",
    input: "{ runId }",
    returns: "OntologySelection: entityTypes, relationTypes, relations (guided | fixed | open), folderEntity",
  },
  {
    name: "runs.documents",
    kind: "query",
    area: "runs",
    access: "read",
    graph: "read",
    purpose: "Where each of a run's documents stands in OCR and extraction, skipped files marked.",
    input: "{ runId }",
    returns: "DocumentStatus[]",
  },

  // ----------------------------------------------------------- ingestion
  {
    name: "ingestion.embeddingWidth",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "The vector width the destination's graph tables store, so a run picks an embedding model that fits them.",
    input: "none",
    returns: "number, or null where nothing fixes a width or the tables disagree",
    limits: "A console hosted in Snowflake answers; any other answers null.",
  },
  {
    name: "ingestion.documentStages",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "The stages in the console's schema that documents can be read from, upload stage first.",
    input: "none",
    returns: "{ stage, name }[], or null where the runtime lists none",
    limits: "A console hosted in Snowflake answers; any other answers null.",
  },
  {
    name: "ingestion.stageFolders",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "The folders of one document stage the caller may run on, with how many documents each holds.",
    input: "{ stage }",
    returns: "{ prefix, documents }[], or null where the runtime lists none",
    limits: "The stage must be one ingestion.documentStages offers. Another viewer's upload folders are left out.",
  },
  {
    name: "ingestion.modelChoices",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "The account's Cortex models that can extract a graph, and its embedding models of known width.",
    input: "none",
    returns: "{ llm: string[], embedding: { model, dimension }[] }, or null where the runtime lists none",
    limits: "A console hosted in Snowflake answers; any other answers null.",
  },
  {
    name: "ingestion.preflight",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    graph: "write",
    purpose: "Check an ingestion request without starting it: run id, source, providers and credentials.",
    input: INGESTION_REQUEST,
    returns: "{ ok, errors, warnings, checks }",
    limits: `${REQUEST_CONFINED} ${TARGET_GRAPH}`,
  },
  {
    name: "ingestion.sources.list",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "List the documents a source holds.",
    input: "{ source: { path | prefix | stage, … } as for the request's sourceKind, limit?: 1-5000 }",
    returns: "SourceObject[]: uri, name, sizeBytes, modifiedAt, checksum",
    limits: `Needs write authority because it names a location on this host. ${SOURCE_CONFINED}`,
  },
  {
    name: "ingestion.sources.summary",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "Count a source's documents and bytes without listing them, however large it is.",
    input: "{ source }",
    returns: "{ count, sizeBytes, complete } (complete is false when counting stopped at its ceiling)",
    limits: SOURCE_CONFINED,
  },
  {
    name: "ingestion.config.preview",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    graph: "write",
    purpose: "The run configuration a request would generate, as YAML with secrets redacted.",
    input: INGESTION_REQUEST,
    returns: "YAML text",
    limits: `${REQUEST_CONFINED} ${TARGET_GRAPH}`,
  },
  {
    name: "ingestion.estimate",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "Estimate what a run costs and how long it takes.",
    input: "{ objectCount, runtime, ocrProvider, parallelism }",
    returns: ESTIMATE,
    limits: "Changes nothing. A runId is refused with 400; ingestion.recordEstimate keeps an estimate on a run.",
  },
  {
    name: "ingestion.recordEstimate",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    graph: "write",
    purpose: "Estimate a run and keep the estimate on it, for its consumption panel to hold the actual spend against.",
    input: "{ runId, objectCount, runtime, ocrProvider, parallelism }",
    returns: ESTIMATE,
  },
  {
    name: "ingestion.scanPii",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    purpose: "Scan the first 40 documents of a source for personal data: names, and the start of text files on this host.",
    input: "{ source, sourceKey }",
    returns: "{ hits: [{ source, kind, snippet }], blocked } (blocked until the caller acknowledges the source)",
    limits: SOURCE_CONFINED,
  },
  {
    name: "ingestion.acknowledgePii",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    purpose: "Acknowledge a source's personal-data findings, for the caller only, so it is no longer blocked for them.",
    input: "{ sourceKey }",
    returns: WORKSPACE,
  },
  {
    name: "ingestion.defaultOntology",
    kind: "query",
    area: "ingestion",
    access: "write",
    purpose: "What a run extracts unless the request changes it: the fleet profile's ontology, else the base configuration's.",
    input: "{ baseConfigPath?: string | null } (optional)",
    returns: "OntologySelection plus source, naming where it came from",
    limits: "baseConfigPath must be an operator profile under configs/ or a generated configuration under the state root.",
  },
  {
    name: "ingestion.ontology",
    kind: "mutation",
    area: "ingestion",
    access: "write",
    purpose: "Propose entity and relation types for a graph described in words.",
    input: "{ intent: 8-4000 characters }",
    returns:
      "{ intent, types, relations, descriptions, warning, source: model | heuristic, modelFailure, coverage } (coverage against the checkout's sample gold file, or null)",
    limits: "Calls the Ask model where one is configured; without one, or when it fails, a word-based proposal answers.",
  },

  // -------------------------------------------------------------- graphs
  {
    name: "graphs.versions",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "A graph's published versions, oldest first.",
    input: "{ graphId }",
    returns: "GraphVersion[]: graphId, runId, number, createdAt, head",
    limits: "Empty on the local and Snowflake runtimes, which keep no versions.",
  },
  {
    name: "graphs.load",
    kind: "query",
    area: "graphs",
    access: "write",
    graph: "read",
    purpose: "Load a graph from the directory one of its runs stored it in.",
    input: "{ location, graphId }",
    returns: "GraphDataset",
    limits:
      "Needs write authority because it names a path on this host: under the state root or the checkout, and where one of the graph's runs wrote it. Without graphId it is refused. Prefer graphs.loadRun.",
  },
  {
    name: "graphs.loadRun",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "A run's graph: entities, relations, communities, evidence, documents, chunk references and known gaps.",
    input: "{ runId }",
    returns: "GraphDataset: graphId, nodes, edges, communities, evidence, documents, chunks (id and document_id, without their text), discardedWindows, rejectedRecords, …",
  },
  {
    name: "graphs.rename",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Change a graph's display name.",
    input: "{ graphId, displayName }",
    returns: "The stored name",
  },
  {
    name: "graphs.delete",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "owner",
    purpose: "Delete a graph for good: every run, what the runtime stored, and upload folders only it used.",
    input: "{ graphId }",
    returns: "Counts of what was removed, including runs and uploads",
    limits: "Refused while a run of the graph is queued or running.",
  },
  {
    name: "graphs.deletion",
    kind: "query",
    area: "graphs",
    access: "write",
    graph: "owner",
    purpose: "What graphs.delete would remove, for a confirmation to name.",
    input: "{ graphId }",
    returns: "{ graphId, graphName, runs, active, uploads, sharedWith }",
  },
  {
    name: "graphs.access",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "Who owns a graph, who it is shared with, and the caller's own role on it.",
    input: "{ graphId }",
    returns: "{ graphId, owner, role, shares: [{ principal, level, grantedBy, grantedAt }] }",
  },
  {
    name: "graphs.share",
    kind: "mutation",
    area: "graphs",
    access: "named",
    graph: "owner",
    purpose: "Share a graph with a person, or change the level they already have.",
    input: "{ graphId, principal, level: read | write }",
    returns: "The graph's access, as graphs.access returns it",
    limits: "Behind a sign-in gate the principal must be an email address.",
  },
  {
    name: "graphs.unshare",
    kind: "mutation",
    area: "graphs",
    access: "named",
    graph: "read",
    purpose: "Remove a person's access: the owner removes anyone, anyone else only themselves.",
    input: "{ graphId, principal }",
    returns: "The graph's access, or null once the caller removed themselves",
  },
  {
    name: "graphs.people",
    kind: "query",
    area: "graphs",
    access: "named",
    purpose: "People the console has seen sign in, and people on the caller's graphs, to offer when sharing.",
    input: "none",
    returns: "string[]",
  },
  {
    name: "graphs.quality",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "QA gates for a run's graph and its comparison with the gold file.",
    input: "{ runId }",
    returns: "QualityReport",
  },
  {
    name: "graphs.publishSnowflake",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Write a finished fleet graph into Snowflake tables as it is.",
    input: "{ runId, database, schema, warehouse, role?, bulkStage }",
    returns: "{ nodes, edges } written",
    limits: "Kubernetes runtime only; elsewhere 400.",
  },
  {
    name: "graphs.gold.upload",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Attach a gold file, the graph's QA contract, after checking it.",
    input: "{ graphId, json: the gold file as a string, at most 2 MB }",
    returns: "{ entities, relations } counted in the file",
  },
  {
    name: "graphs.gold.remove",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Remove a graph's uploaded gold file.",
    input: "{ graphId }",
    returns: "nothing",
  },
  {
    name: "graphs.gold.template",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "A gold-file skeleton drawn from a run's graph, to fill in.",
    input: "{ runId }",
    returns: "A gold graph: entities and relations",
  },
  {
    name: "graphs.inspect",
    kind: "query",
    area: "graphs",
    access: "read",
    graph: "read",
    purpose: "The inspect report as JSON, for evals and reports.",
    input: "{ runId }",
    returns: "{ graphId, graphName, tables, topEntities, topRelations, communitySummaries, zeroEntityFailure, gold }",
  },
  {
    name: "graphs.ask",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "read",
    purpose: "Ask a finished graph and wait for the whole answer. POST /api/ask streams the same answer.",
    input: "{ runId, question: 2-10000 characters, mode?: local | global | hybrid | drift | auto (default local), perspectiveId?, documentIds? }",
    returns: "{ summary, citations, entityIds, relationIds, communityIds, mode, scoped, queryConsumption, … }",
    limits: "409 unless the run finished. Runs a model, so it needs write authority although it changes nothing.",
  },
  {
    name: "graphs.pin",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Pin a node or a relation with a note.",
    input: "{ graphId, targetId, kind: node | edge, note }",
    returns: WORKSPACE,
  },
  {
    name: "graphs.sampleReviews",
    kind: "mutation",
    area: "graphs",
    access: "write",
    graph: "write",
    purpose: "Queue about 1% of a run's relations with confidence 0.9 or more for human review.",
    input: "{ runId }",
    returns: WORKSPACE,
  },

  // ----------------------------------------------------------- workspace
  {
    name: "workspace.get",
    kind: "query",
    area: "workspace",
    access: "read",
    purpose:
      "The caller's workspace: perspectives, versions, pins, reviews, watches, skipped files and estimates of graphs they can read, their own API keys, and the console-wide settings.",
    input: "none",
    returns: "Workspace state; key secrets are never included",
  },
  {
    name: "workspace.perspective",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Create or update a perspective, a named and scoped view of a graph.",
    input: "{ id?, graphId, name, lifecycle: draft | candidate | production, search, communityIds: string[], suggestedQuestions: string[] }",
    returns: WORKSPACE,
    limits: "An existing perspective stays on its own graph: 403 for an id from another graph.",
  },
  {
    name: "workspace.promotePerspective",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Move a perspective to draft, candidate or production.",
    input: "{ id, lifecycle: draft | candidate | production }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.review",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Decide a queued review: keep, drop, relabel, or merge one end into another entity.",
    input:
      "{ id, decision: { kind: keep } | { kind: drop } | { kind: relabel, relationType } | { kind: merge, end: source | target, into } }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.publish",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Record a new production version of a graph in the workspace; the previous production version becomes a candidate.",
    input: "{ graphId, note }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.toggleWatch",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Pause or resume a watch.",
    input: "{ id }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.applyWatch",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Record a watch's pending files as a draft incremental version. It starts no run.",
    input: "{ id }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.createWatch",
    kind: "mutation",
    area: "workspace",
    access: "write",
    graph: "write",
    purpose: "Watch a source prefix for a graph's new files.",
    input: "{ graphId, prefix }",
    returns: WORKSPACE,
  },
  {
    name: "workspace.createKey",
    kind: "mutation",
    area: "workspace",
    access: "session",
    purpose: "Mint an API key that acts as the caller.",
    input: "{ name: 1-200 characters }",
    returns: "{ secret, state }: the fg_ secret, shown only this once, and the caller's workspace",
    limits: "From the SDK keys page only: a key cannot mint another.",
  },
  {
    name: "workspace.revokeKey",
    kind: "mutation",
    area: "workspace",
    access: "session",
    purpose: "Revoke a key the caller minted; requests presenting it get 401 from then on.",
    input: "{ id }",
    returns: WORKSPACE,
    limits: "403 for a key someone else minted. From the SDK keys page only: a key cannot revoke its owner's keys, itself included.",
  },
  {
    name: "workspace.identityIncident",
    kind: "mutation",
    area: "workspace",
    access: "operator",
    purpose: "Record an identity-mismatch incident and the triage note built from it.",
    input: "{ expectedPrincipal, actualPrincipal, graphOwner, grants }",
    returns: WORKSPACE,
    limits: "One record for the whole console; the next call replaces it.",
  },
  {
    name: "workspace.markGrant",
    kind: "mutation",
    area: "workspace",
    access: "operator",
    purpose: "Mark a Snowflake grant on the setup checklist as in place or missing.",
    input: "{ object, ok }",
    returns: WORKSPACE,
    limits: "One checklist for the whole console.",
  },
  {
    name: "workspace.incident",
    kind: "query",
    area: "workspace",
    access: "read",
    purpose: "The staff incident view: recent runs of the caller's graphs, forgotten ones too, and how many are under way.",
    input: "none",
    returns: "{ banner, queueDepth, modelReady, events: [{ runId, graphName, status, error, runtime }], activeRunIds }",
  },

  // --------------------------------------------------------------- fleet
  {
    name: "fleet.profile",
    kind: "query",
    area: "fleet",
    access: "read",
    purpose: "The profile the fleet's workers mount.",
    input: "none",
    returns: "{ namespace, configMap, config, parsingEndpoint, ontology }, or null off the Kubernetes runtime",
  },
  {
    name: "fleet.cluster",
    kind: "query",
    area: "fleet",
    access: "read",
    purpose: "Nodes, workloads and warnings of the namespace this console was deployed for.",
    input: "none",
    returns: "{ context, namespace, nodesReady, nodesTotal, nodes, workloads, warnings, observedAt }, or null off the Kubernetes runtime",
    limits:
      "Not limited by graph access: it holds nodes, pods, images and served models, and no run, graph or document. fleet.nodeAssignments is where runs appear, limited to the caller's graphs.",
  },
  {
    name: "fleet.nodeAssignments",
    kind: "query",
    area: "fleet",
    access: "read",
    purpose: "The work leased on one node, limited to the caller's graphs.",
    input: "{ nodeName }",
    returns: "[{ workerId, runId, graphId, taskId, stage, scopeId, updatedAt }]",
  },

  // ---------------------------------------------------------------- rest
  {
    name: "/api/trpc/{procedure}",
    kind: "GET",
    area: "rest",
    access: "read",
    purpose: "Call a query procedure.",
    input: '?input= URL-encoded SuperJSON envelope, { "json": <input> }',
    returns: '{ "result": { "data": { "json": <output> } } }, or { "error": … } with the HTTP status',
    limits: "Each procedure's own access level applies on top.",
  },
  {
    name: "/api/trpc/{procedure}",
    kind: "POST",
    area: "rest",
    access: "read",
    purpose: "Call a mutation procedure.",
    input: '{ "json": <input> } as the JSON body',
    returns: '{ "result": { "data": { "json": <output> } } }, or { "error": … } with the HTTP status',
    limits: "Each procedure's own access level applies on top.",
  },
  {
    name: "/api/ask",
    kind: "GET",
    area: "rest",
    access: "open",
    purpose: "What POST /api/ask takes, and whether an Ask model is configured.",
    input: "none",
    returns: "{ service, path, methods, auth, formats, body, tools, model: { configured, provider, billed } }",
  },
  {
    name: "/api/ask",
    kind: "POST",
    area: "rest",
    access: "write",
    graph: "read",
    purpose: "Ask a finished graph and stream the answer as it is written.",
    input:
      "{ runId, question or messages (AI SDK UIMessage[]), mode?: local | global | hybrid | drift | auto, perspectiveId?, documentIds?, topK?: 1-40, format?: ui | ndjson } (runId and mode may be query parameters; accept: application/x-ndjson also picks NDJSON)",
    returns: "An AI SDK UI message stream, or NDJSON events: status, tool, text, citation, then one done or one error",
    limits:
      "404 for a run the caller cannot read, 409 unless it finished. Without a configured model the graph answers with its own matching text. A question may run for up to 20 minutes.",
  },
  {
    name: "/api/graphs",
    kind: "GET",
    area: "rest",
    access: "read",
    purpose: "The graphs the caller may ask, each at its newest finished version, for a program choosing one on a person's behalf.",
    input: "none (x-flakegraph-runtime picks the catalog)",
    returns: "{ graphs: [{ runId, graphId, name, documentCount, updatedAt }] }; runId is what POST /api/ask takes",
    limits: "Searches the caller's 500 newest runs.",
  },
  {
    name: "/api/inspect",
    kind: "GET",
    area: "rest",
    access: "read",
    graph: "read",
    purpose: "The inspect report as a standalone HTML page.",
    input: "?run=<runId> (x-flakegraph-runtime picks the runtime; without it, the run's own record does, else the console's default)",
    returns: "text/html; a failure is { error } with 401 unadmitted, 403 for a graph the caller cannot read, 404 for a run the runtime does not hold",
  },
  {
    name: "/api/uploads",
    kind: "POST",
    area: "rest",
    access: "write",
    purpose: "Upload documents into an upload folder, for a run's upload source to name.",
    input: "multipart/form-data: files (repeated), paths (optional, one folder-relative path per file), jobId? (add to a folder the caller started)",
    returns: "{ path, count, jobId }",
    limits: "At most 200 files, 200 MB per file and 1 GB per request. A folder is its first uploader's: 403 for anyone else.",
  },
  {
    name: "/api/docs",
    kind: "GET",
    area: "rest",
    access: "open",
    purpose: "This catalog, as JSON.",
    input: "none",
    returns: "{ endpoint, auth, access, areas, procedures, routes, … }",
  },
  {
    name: "/api/health",
    kind: "GET",
    area: "rest",
    access: "open",
    purpose: "Liveness: answers while the server runs.",
    input: "none",
    returns: '{ status: "ok", service, timestamp }',
  },
];

/** tRPC procedures, then REST routes: the two halves of the catalog. */
export const CONTROL_PLANE_PROCEDURES = CONTROL_PLANE_ENDPOINTS.filter((endpoint) => endpoint.area !== "rest");
export const CONTROL_PLANE_ROUTES = CONTROL_PLANE_ENDPOINTS.filter((endpoint) => endpoint.area === "rest");

/** Endpoints whose name, kind, area, access, input, purpose or limits contain every word of the search. */
export function filterEndpoints(endpoints: readonly ApiEndpoint[], search: string): ApiEndpoint[] {
  const words = search.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) {
    return [...endpoints];
  }
  return endpoints.filter((endpoint) => {
    const text = [
      endpoint.name,
      endpoint.kind,
      endpoint.area,
      API_ACCESS[endpoint.access].label,
      endpoint.graph ? `graph ${endpoint.graph}` : "",
      endpoint.purpose,
      endpoint.input,
      endpoint.returns,
      endpoint.limits ?? "",
    ]
      .join(" ")
      .toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

export const EXAMPLE_LANGUAGES = [
  { id: "typescript", label: "TypeScript" },
  { id: "python", label: "Python" },
  { id: "curl", label: "curl" },
] as const;

export type ExampleLanguage = (typeof EXAMPLE_LANGUAGES)[number]["id"];

export type TaskExamples = Record<ExampleLanguage, string>;

export function listRunsExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.list?input={"json":{"limit":5}}'`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "x-flakegraph-runtime": "local",
};
const input = encodeURIComponent(JSON.stringify({ json: { limit: 5 } }));
const response = await fetch(\`\${origin}/api/trpc/runs.list?input=\${input}\`, { headers });
console.log(await response.json());`,
    python: `import json, os, urllib.parse, urllib.request

origin = "${origin}"
headers = {
    "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
    "x-flakegraph-runtime": "local",
}
encoded = urllib.parse.quote(json.dumps({"json": {"limit": 5}}))
request = urllib.request.Request(
    f"{origin}/api/trpc/runs.list?input={encoded}",
    headers=headers,
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function getRunExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
# Use a runId from runs.list
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.get?input={"json":{"runId":"run_martial_arts"}}'`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "x-flakegraph-runtime": "local",
};
const input = encodeURIComponent(JSON.stringify({ json: { runId: "run_martial_arts" } }));
const response = await fetch(\`\${origin}/api/trpc/runs.get?input=\${input}\`, { headers });
console.log(await response.json());`,
    python: `import json, os, urllib.parse, urllib.request

origin = "${origin}"
headers = {
    "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
    "x-flakegraph-runtime": "local",
}
encoded = urllib.parse.quote(json.dumps({"json": {"runId": "run_martial_arts"}}))
request = urllib.request.Request(
    f"{origin}/api/trpc/runs.get?input={encoded}",
    headers=headers,
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function askGraphExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
curl -sS -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"json":{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local"}}' \\
  "${origin}/api/trpc/graphs.ask"`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "content-type": "application/json",
  "x-flakegraph-runtime": "local",
};
const response = await fetch(\`\${origin}/api/trpc/graphs.ask\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    json: { runId: "run_martial_arts", question: "Who developed judo?", mode: "local" },
  }),
});
console.log(await response.json());`,
    python: `import json, os, urllib.request

origin = "${origin}"
request = urllib.request.Request(
    f"{origin}/api/trpc/graphs.ask",
    data=json.dumps({
        "json": {
            "runId": "run_martial_arts",
            "question": "Who developed judo?",
            "mode": "local",
        }
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
        "content-type": "application/json",
        "x-flakegraph-runtime": "local",
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function askStreamExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
# NDJSON: status, tool, text, citation, done
curl -N -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "accept: application/x-ndjson" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local","format":"ndjson"}' \\
  "${origin}/api/ask"`,
    typescript: `import { DefaultChatTransport } from "ai";
import { useChat } from "@ai-sdk/react";

// Browser / useChat consumer
useChat({
  transport: new DefaultChatTransport({
    api: "${origin}/api/ask",
    body: { runId: "run_martial_arts", mode: "local", format: "ui" },
  }),
});

// Script / eval harness: NDJSON status, tool, text, citation, done
const origin = "${origin}";
const response = await fetch(\`\${origin}/api/ask\`, {
  method: "POST",
  headers: {
    Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
    "content-type": "application/json",
    accept: "application/x-ndjson",
    "x-flakegraph-runtime": "local",
  },
  body: JSON.stringify({
    runId: "run_martial_arts",
    question: "Who developed judo?",
    mode: "local",
    format: "ndjson",
  }),
});
for (const line of (await response.text()).split("\\n").filter(Boolean)) {
  console.log(JSON.parse(line));
}`,
    python: `import json, os, urllib.request

origin = "${origin}"
request = urllib.request.Request(
    f"{origin}/api/ask",
    data=json.dumps({
        "runId": "run_martial_arts",
        "question": "Who developed judo?",
        "mode": "local",
        "format": "ndjson",
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
        "content-type": "application/json",
        "accept": "application/x-ndjson",
        "x-flakegraph-runtime": "local",
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    for line in response:
        print(json.loads(line))`,
  };
}

export function askableGraphsExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  "${origin}/api/graphs"`,
    typescript: `const origin = "${origin}";
const response = await fetch(\`\${origin}/api/graphs\`, {
  headers: {
    Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
    "x-flakegraph-runtime": "local",
  },
});
const { graphs } = await response.json();
for (const graph of graphs) {
  // Pass graph.runId to POST /api/ask
  console.log(graph.runId, graph.name, graph.documentCount);
}`,
    python: `import json, os, urllib.request

origin = "${origin}"
request = urllib.request.Request(
    f"{origin}/api/graphs",
    headers={
        "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
        "x-flakegraph-runtime": "local",
    },
)
with urllib.request.urlopen(request) as response:
    for graph in json.load(response)["graphs"]:
        # Pass graph["runId"] to POST /api/ask
        print(graph["runId"], graph["name"], graph["documentCount"])`,
  };
}

export function inspectReportExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
# Without x-flakegraph-runtime the run's own record says which runtime holds it
curl -sS -o inspect.html \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  "${origin}/api/inspect?run=run_martial_arts"`,
    typescript: `import { writeFile } from "node:fs/promises";

const origin = "${origin}";
const response = await fetch(\`\${origin}/api/inspect?run=run_martial_arts\`, {
  headers: { Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\` },
});
if (!response.ok) {
  throw new Error(\`\${response.status}: \${await response.text()}\`);
}
await writeFile("inspect.html", await response.text());`,
    python: `import os, urllib.parse, urllib.request

origin = "${origin}"
query = urllib.parse.urlencode({"run": "run_martial_arts"})
request = urllib.request.Request(
    f"{origin}/api/inspect?{query}",
    headers={"Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}"},
)
with urllib.request.urlopen(request) as response, open("inspect.html", "wb") as report:
    report.write(response.read())`,
  };
}
