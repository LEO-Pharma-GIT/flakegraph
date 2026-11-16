"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/console/page-header";
import { ShowMore } from "@/components/console/show-more";
import { filterKeys } from "@/lib/paging";
import {
  API_ACCESS,
  API_AREAS,
  CONTROL_PLANE_API,
  CONTROL_PLANE_ENDPOINTS,
  EXAMPLE_LANGUAGES,
  askGraphExamples,
  askStreamExamples,
  askableGraphsExamples,
  filterEndpoints,
  getRunExamples,
  inspectReportExamples,
  listRunsExamples,
  type ApiAccess,
  type ApiEndpoint,
  type ExampleLanguage,
  type TaskExamples,
} from "@/lib/control-plane-api";

/** Keys on screen before "Show more"; a deployment that mints one per pipeline accrues dozens. */
export const KEY_PAGE_SIZE = 25;

export function SdkKeysPage() {
  const workspace = trpc.workspace.get.useQuery();
  const session = trpc.auth.session.useQuery();
  const viewerName = session.data?.viewer.userName || "you";
  // A key acts as whoever minted it, so there has to be a whoever.
  const identified = Boolean(session.data?.viewer.userName);
  const [name, setName] = useState("ci-eval");
  const [secret, setSecret] = useState<string | null>(null);
  const [revokeId, setRevokeId] = useState<string | null>(null);
  const [exampleLanguage, setExampleLanguage] = useState<ExampleLanguage>("curl");
  const create = trpc.workspace.createKey.useMutation({
    onSuccess: async (result) => {
      setSecret(result.secret);
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const revoke = trpc.workspace.revokeKey.useMutation({
    onSuccess: async () => {
      toast.success("Key revoked. Existing jobs using it will get 401.");
      setSecret(null);
      setRevokeId(null);
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const keys = useMemo(() => workspace.data?.apiKeys ?? [], [workspace.data?.apiKeys]);
  const [keySearch, setKeySearch] = useState("");
  const [keyLimit, setKeyLimit] = useState(KEY_PAGE_SIZE);
  // Revoking removes a key outright, so every key listed is live; the list
  // reads newest first, which puts a key just created at the top.
  const shownKeys = useMemo(() => filterKeys(keys, keySearch), [keys, keySearch]);
  const keyPage = shownKeys.slice(0, keyLimit);
  const origin = typeof window === "undefined" ? "http://127.0.0.1:3000" : window.location.origin;
  const docsUrl = `${origin}${CONTROL_PLANE_API.docsPath}`;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Access"
        title="SDK keys"
        description="Machine credentials for CI, eval jobs, and scripts. There is no separate SDK package to install: programs call this control plane over HTTP. This page is the human guide. GET /api/docs is the machine-readable catalog. The browser session keeps using SSO."
      />

      <Card id="api-docs" data-testid="sdk-docs">
        <CardHeader>
          <CardTitle>Where the API docs are</CardTitle>
          <CardDescription>
            Keys authenticate callers. The API they call is this control plane, not a third-party SDK site.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <ol className="list-decimal space-y-2 pl-5 text-muted-foreground">
            <li>
              <span className="text-foreground">This page.</span> Worked examples and the catalog of every endpoint below are
              the operator reference.
            </li>
            <li>
              <span className="text-foreground">JSON catalog.</span>{" "}
              <a className="font-mono text-xs text-foreground underline underline-offset-4" href={CONTROL_PLANE_API.docsPath}>
                GET {CONTROL_PLANE_API.docsPath}
              </a>{" "}
              returns the endpoint, auth headers, access levels, and every procedure and route listed below. Bookmark {docsUrl}.
            </li>
            <li>
              <span className="text-foreground">Checkout.</span> Procedure names live in{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.checkoutProcedures}</span>. Request and
              graph payloads live in{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.checkoutPayloads}</span>.
            </li>
            <li>
              <span className="text-foreground">Wire format.</span>{" "}
              <a
                className="underline underline-offset-4"
                href={CONTROL_PLANE_API.trpcHttpDocs}
                target="_blank"
                rel="noreferrer"
              >
                tRPC HTTP
              </a>{" "}
              at <span className="font-mono text-xs text-foreground">{`${origin}${CONTROL_PLANE_API.endpoint}/{procedure}`}</span>.
              Queries are GET with <span className="font-mono text-xs text-foreground">?input=</span>. Mutations are POST.
            </li>
            <li>
              <span className="text-foreground">Fleet SSO.</span> Program paths bypass the browser sign-in gate. See{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.fleetGuide}</span>.
            </li>
          </ol>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" asChild>
              <a href={CONTROL_PLANE_API.docsPath}>Open JSON catalog</a>
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                void navigator.clipboard.writeText(docsUrl);
                toast.success("Docs URL copied");
              }}
            >
              Copy docs URL
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(20rem,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Keys on this control plane</CardTitle>
            <CardDescription>The secret is shown once. Store it in a secret manager; we only keep a preview after that.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
              <label className="grid min-w-0 flex-1 gap-1.5 text-sm">
                <span className="font-medium">Key name</span>
                <Input
                  aria-label="Key name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="ci-eval"
                />
              </label>
              <Button
                className="shrink-0"
                pending={create.isPending}
                disabled={!identified}
                onClick={() => create.mutate({ name: name.trim() || "ci-eval" })}
              >
                {create.isPending ? "Creating…" : "Create a key"}
              </Button>
            </div>
            {!identified ? (
              <p className="text-sm text-muted-foreground" data-testid="sdk-key-sign-in">
                Sign in first. A key acts as the person who minted it, and nobody has said who that is.
              </p>
            ) : null}
            {secret ? (
              <Alert className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <span className="break-all font-mono text-xs">Copy now: {secret}</span>
                <Button
                  className="shrink-0"
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    void navigator.clipboard.writeText(secret);
                    toast.success("Secret copied");
                  }}
                >
                  Copy secret
                </Button>
              </Alert>
            ) : null}
            {keys.length === 0 ? (
              <p className="text-sm text-muted-foreground">No machine keys yet. Create one for CI or a script.</p>
            ) : (
              <div className="space-y-3" data-testid="sdk-key-list">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                  <Input
                    aria-label="Search keys"
                    className="h-8 sm:max-w-xs"
                    placeholder="Search by name"
                    value={keySearch}
                    onChange={(event) => {
                      setKeySearch(event.target.value);
                      setKeyLimit(KEY_PAGE_SIZE);
                    }}
                  />
                  <p className="text-xs text-muted-foreground sm:ml-auto" data-testid="sdk-key-count">
                    {keySearch.trim()
                      ? `${shownKeys.length.toLocaleString()} of ${keys.length.toLocaleString()} keys`
                      : `${keys.length.toLocaleString()} key${keys.length === 1 ? "" : "s"}`}
                  </p>
                </div>
                {shownKeys.length === 0 ? (
                  <p className="text-sm text-muted-foreground">No key matches this search.</p>
                ) : (
                  <ul className="divide-y divide-border rounded-md border border-border">
                    {keyPage.map((key) => (
                      <li key={key.id} className="flex flex-wrap items-center justify-between gap-3 px-3 py-3">
                        <div className="min-w-0">
                          <p className="font-medium">{key.name}</p>
                          <p className="truncate font-mono text-xs text-muted-foreground">
                            {key.preview} · {key.createdAt.slice(0, 10)}
                          </p>
                        </div>
                        <div className="flex shrink-0 items-center gap-1">
                          {revokeId === key.id && !revoke.isPending ? (
                            <Button size="sm" variant="ghost" onClick={() => setRevokeId(null)}>
                              Keep
                            </Button>
                          ) : null}
                          <Button
                            size="sm"
                            variant={revokeId === key.id ? "destructive" : "ghost"}
                            aria-label={`${revokeId === key.id ? "Confirm revoke" : "Revoke"} ${key.name}`}
                            pending={revoke.isPending && revoke.variables?.id === key.id}
                            disabled={revoke.isPending}
                            onClick={() => {
                              if (revokeId !== key.id) {
                                setRevokeId(key.id);
                                return;
                              }
                              revoke.mutate({ id: key.id });
                            }}
                          >
                            {revoke.isPending && revoke.variables?.id === key.id
                              ? "Revoking…"
                              : revokeId === key.id
                                ? "Confirm revoke"
                                : "Revoke"}
                          </Button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
                <ShowMore
                  shown={keyPage.length}
                  total={shownKeys.length}
                  pageSize={KEY_PAGE_SIZE}
                  noun="keys"
                  onMore={() => setKeyLimit((current) => current + KEY_PAGE_SIZE)}
                />
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>How to use a key</CardTitle>
            <CardDescription>Browsers send the session cookie. Machines send this secret instead.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>
              Create a named key, copy the <span className="font-mono text-xs">fg_…</span> secret immediately, then export it as{" "}
              <span className="font-mono text-xs">FLAKEGRAPH_API_KEY</span> in the job that calls the control plane.
            </p>
            <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
              <li>
                A key acts as the person who minted it, {viewerName} for a key made here, with that person&apos;s graph access. It
                never acts as a browser session: keys are minted and revoked only here, and a key never changes the
                console&apos;s own settings.
              </li>
              <li>
                Send <span className="font-mono text-xs text-foreground">Authorization: Bearer $FLAKEGRAPH_API_KEY</span> or{" "}
                <span className="font-mono text-xs text-foreground">x-flakegraph-api-key</span>. Browser SSO is ignored when a
                key is present.
              </li>
              <li>
                Add <span className="font-mono text-xs text-foreground">x-flakegraph-runtime: local</span> (or kubernetes /
                snowflake) so the call hits the same catalog as the sidebar.
              </li>
              <li>A 401 on this path means the key is missing or revoked. Do not bounce the operator through SSO.</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card id="api-examples">
        <CardHeader>
          <CardTitle>What you can do with a key</CardTitle>
          <CardDescription>
            Typical jobs: list graphs in CI, poll a run until it finishes, ask a completed graph, or save an inspect report for
            an eval. Replace <span className="font-mono text-xs">fg_…</span> with the secret shown once above.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="list">
            <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
              <TabsTrigger value="list">List graphs</TabsTrigger>
              <TabsTrigger value="status">Check a run</TabsTrigger>
              <TabsTrigger value="ask">Ask a graph</TabsTrigger>
              <TabsTrigger value="stream">Stream ask</TabsTrigger>
              <TabsTrigger value="graphs">Graphs to ask</TabsTrigger>
              <TabsTrigger value="inspect">Inspect report</TabsTrigger>
            </TabsList>
            <TabsContent value="list">
              <ExampleBlock
                why="CI and dashboards call this to see which graphs exist on the catalog and whether the latest job succeeded."
                examples={listRunsExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-key-example"
              />
            </TabsContent>
            <TabsContent value="status">
              <ExampleBlock
                why="Poll runs.get until status is succeeded, failed, or cancelled. runs.documents is the per-file progress the workspace uses."
                examples={getRunExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-status-example"
              />
            </TabsContent>
            <TabsContent value="ask">
              <ExampleBlock
                why="Eval jobs that only need the final JSON answer can still call graphs.ask. Prefer POST /api/ask when you want tokens and tool status."
                examples={askGraphExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-ask-example"
              />
            </TabsContent>
            <TabsContent value="stream">
              <ExampleBlock
                why="Agents, eval harnesses, and scripts should stream POST /api/ask. format=ndjson emits status, tool, text, citation, and done events."
                examples={askStreamExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-ask-stream-example"
              />
            </TabsContent>
            <TabsContent value="graphs">
              <ExampleBlock
                why="GET /api/graphs lists the graphs the caller may ask, each at its newest finished version. Its runId is what POST /api/ask takes."
                examples={askableGraphsExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-graphs-example"
              />
            </TabsContent>
            <TabsContent value="inspect">
              <ExampleBlock
                why="GET /api/inspect?run= returns the inspect report as one HTML page to keep with an eval. graphs.inspect returns the same report as JSON."
                examples={inspectReportExamples(origin)}
                language={exampleLanguage}
                onLanguageChange={setExampleLanguage}
                testId="sdk-inspect-example"
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <ApiCatalog />
    </div>
  );
}

function ExampleBlock({
  why,
  examples,
  language,
  onLanguageChange,
  testId,
}: {
  why: string;
  examples: TaskExamples;
  language: ExampleLanguage;
  onLanguageChange: (language: ExampleLanguage) => void;
  testId: string;
}) {
  const code = examples[language];
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">{why}</p>
      <div className="overflow-hidden rounded-md border border-border bg-muted/40">
        <div className="flex items-center justify-end gap-1 border-b border-border px-2 py-1">
          <div className="flex gap-1" role="group" aria-label="Example language" data-testid="sdk-example-language">
            {EXAMPLE_LANGUAGES.map((item) => (
              <Button
                key={item.id}
                size="sm"
                variant={language === item.id ? "secondary" : "ghost"}
                className="h-7 px-2 text-xs"
                aria-pressed={language === item.id}
                onClick={() => onLanguageChange(item.id)}
              >
                {item.label}
              </Button>
            ))}
          </div>
        </div>
        <pre className="overflow-x-auto p-3 font-mono text-xs" data-testid={testId}>
          {code}
        </pre>
      </div>
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          void navigator.clipboard.writeText(code);
          toast.success("Example copied");
        }}
      >
        Copy example
      </Button>
    </div>
  );
}

const ACCESS_BADGE: Record<ApiAccess, "outline" | "secondary" | "warning" | "danger"> = {
  open: "outline",
  read: "secondary",
  write: "warning",
  named: "danger",
  session: "danger",
  operator: "danger",
};

/** Every procedure and route, by area, narrowed by a search over what each one says. */
function ApiCatalog() {
  const [search, setSearch] = useState("");
  const shown = useMemo(() => filterEndpoints(CONTROL_PLANE_ENDPOINTS, search), [search]);
  const total = CONTROL_PLANE_ENDPOINTS.length;
  return (
    <Card id="api-reference">
      <CardHeader>
        <CardTitle>API catalog</CardTitle>
        <CardDescription>
          Every tRPC procedure in {CONTROL_PLANE_API.checkoutProcedures} and every route under /api, with who may call it and
          what it takes. GET {CONTROL_PLANE_API.docsPath} serves the same list.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <dl className="grid gap-3 text-sm sm:grid-cols-2" data-testid="sdk-access-levels">
          {(Object.keys(API_ACCESS) as ApiAccess[]).map((level) => (
            <div key={level} className="space-y-1">
              <dt>
                <Badge variant={ACCESS_BADGE[level]}>{API_ACCESS[level].label}</Badge>
              </dt>
              <dd className="text-muted-foreground">{API_ACCESS[level].description}</dd>
            </div>
          ))}
        </dl>
        <p className="text-sm text-muted-foreground">
          A <span className="text-foreground">graph</span> requirement is the standing the caller needs on the graph the call
          touches: read, write, or owner. Lists only ever hold graphs the caller can read.
        </p>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Input
            aria-label="Search the API catalog"
            className="h-8 sm:max-w-sm"
            placeholder="Search by name, input, or access"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <p className="text-xs text-muted-foreground sm:ml-auto" data-testid="sdk-catalog-count">
            {search.trim()
              ? `${shown.length.toLocaleString()} of ${total.toLocaleString()} endpoints`
              : `${total.toLocaleString()} endpoints`}
          </p>
        </div>
        <div className="space-y-8" data-testid="sdk-procedure-catalog">
          {shown.length === 0 ? <p className="text-sm text-muted-foreground">No endpoint matches this search.</p> : null}
          {API_AREAS.map((area) => {
            const endpoints = shown.filter((endpoint) => endpoint.area === area.id);
            if (endpoints.length === 0) {
              return null;
            }
            return (
              <section key={area.id} className="space-y-2" data-testid={`sdk-catalog-${area.id}`}>
                <div>
                  <h3 className="text-sm font-semibold">{area.label}</h3>
                  <p className="text-xs text-muted-foreground">{area.description}</p>
                </div>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{area.id === "rest" ? "Route" : "Procedure"}</TableHead>
                      <TableHead>Access</TableHead>
                      <TableHead>Input</TableHead>
                      <TableHead>What it does</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {endpoints.map((endpoint) => (
                      <EndpointRow key={`${endpoint.kind} ${endpoint.name}`} endpoint={endpoint} />
                    ))}
                  </TableBody>
                </Table>
              </section>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function EndpointRow({ endpoint }: { endpoint: ApiEndpoint }) {
  return (
    <TableRow className="align-top">
      <TableCell className="align-top">
        <div className="flex flex-col items-start gap-1">
          <span className="whitespace-nowrap font-mono text-xs">{endpoint.name}</span>
          <Badge variant={endpoint.kind === "query" || endpoint.kind === "GET" ? "secondary" : "outline"}>{endpoint.kind}</Badge>
        </div>
      </TableCell>
      <TableCell className="align-top">
        <div className="flex flex-col items-start gap-1">
          <Badge className="whitespace-nowrap" variant={ACCESS_BADGE[endpoint.access]}>{API_ACCESS[endpoint.access].label}</Badge>
          {endpoint.graph ? <span className="whitespace-nowrap text-xs text-muted-foreground">graph: {endpoint.graph}</span> : null}
        </div>
      </TableCell>
      <TableCell className="min-w-44 align-top font-mono text-xs text-muted-foreground">{endpoint.input}</TableCell>
      <TableCell className="min-w-60 align-top">
        <p>{endpoint.purpose}</p>
        <p className="mt-1 text-xs text-muted-foreground">Returns: {endpoint.returns}</p>
        {endpoint.limits ? <p className="mt-1 text-xs text-muted-foreground">{endpoint.limits}</p> : null}
      </TableCell>
    </TableRow>
  );
}
