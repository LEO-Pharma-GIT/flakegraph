"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ChevronRight, KeyRound, Plus, Search, Server, ShieldAlert, Trash2, Users } from "lucide-react";
import { trpc } from "@/components/providers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  isActiveStatus,
  isSuccessStatus,
  type Capability,
  type RuntimeMode,
  type RunSnapshot,
} from "@/server/protocol/schema";
import { useClientValue } from "@/lib/browser-storage";
import { cn, formatRelativeTime } from "@/lib/utils";
import { displayPrincipal, LEVELS } from "@/components/console/graph-sharing";
import { DeleteGraphDialog, type DeletableGraph } from "@/components/console/delete-graph-dialog";
import { catalogGraphs } from "@/lib/catalog-graphs";
import { catalogEmptyCopy, statusSentence, type CatalogScope } from "@/lib/status-sentence";
import { toast } from "sonner";

interface SidebarProps {
  runtime: RuntimeMode;
  availableRuntimes: RuntimeMode[];
  capabilities: Set<Capability>;
  page: string;
  selectedRunId: string | null;
  identified: boolean;
  // Behind a sign-in gate the identity is the gate's: nothing here assumes
  // one, and leaving means signing out of the gate.
  identityFromGate?: boolean;
  // A laptop console with no gate offers demo principals to assume; a
  // console that was not told to serve them offers nothing to sign in as.
  demoIdentities?: boolean;
  signOutUrl?: string | null;
  principal: string;
  role: "operator" | "analyst" | "staff";
  suggestionMode: "off" | "on-request" | "auto-fill";
  // Settings that are the whole console's change only for whoever
  // administers it; everyone else sees them as they stand.
  operator: boolean;
  onRuntimeChange: (runtime: RuntimeMode) => void;
  onNewGraph: () => void;
  onFleet: () => void;
  onStaff: () => void;
  onKeys: () => void;
  onSelectRun: (runId: string) => void;
  /** Runs whose graph was deleted, so a page showing one can move on. */
  onDeleted?: (runIds: string[]) => void;
  className?: string;
  onNavigate?: () => void;
  showClose?: boolean;
}

/** The identities a laptop console offers, for trying the catalog as different people. */
const DEMO_IDENTITIES: {
  label: string;
  variant: "default" | "secondary" | "outline" | "ghost";
  principal: { userName: string; roles: string[]; role: "operator" | "analyst" | "staff" };
}[] = [
  { label: "Sign in as ALICE", variant: "default", principal: { userName: "ALICE", roles: ["APP_OPERATOR"], role: "operator" } },
  { label: "Sign in as BOB", variant: "default", principal: { userName: "BOB", roles: ["APP_OPERATOR"], role: "operator" } },
  { label: "Sign in as CAROL", variant: "secondary", principal: { userName: "CAROL", roles: ["ANALYST"], role: "analyst" } },
  { label: "Staff", variant: "outline", principal: { userName: "SRE", roles: ["FLAKEGRAPH_STAFF"], role: "staff" } },
  { label: "Sign out", variant: "ghost", principal: { userName: "", roles: [], role: "operator" } },
];

const SCOPES: { value: CatalogScope; label: string; title: string }[] = [
  { value: "mine", label: "Mine", title: "Graphs you own" },
  { value: "shared", label: "Shared", title: "Graphs other people shared with you" },
];

/** Graphs the catalog loads, newest first; the server allows up to 500. */
const CATALOG_LIMIT = 500;

export function Sidebar(props: SidebarProps) {
  const [search, setSearch] = useState("");
  const [storage, setStorage] = useState("all");
  const [scope, setScope] = useState<CatalogScope>("mine");
  const [identityOpen, setIdentityOpen] = useState(false);
  // Graph ids chosen for a bulk delete.
  const [selected, setSelected] = useState<string[]>([]);
  // The graphs a Delete confirmation is open for.
  const [deleting, setDeleting] = useState<DeletableGraph[] | null>(null);
  // Where the last checkbox click landed, so shift-click can take the range.
  const lastToggled = useRef<string | null>(null);
  const utils = trpc.useUtils();
  // The list's cap: search and the filters work on what is loaded, so the
  // cap has to be said when it is reached rather than look like the whole.
  const runs = trpc.runs.list.useQuery({ limit: CATALOG_LIMIT }, { refetchInterval: 4_000 });
  // One row per graph: its versions and attempts are one thing to open or delete.
  const graphs = useMemo(() => catalogGraphs(runs.data ?? []), [runs.data]);
  const graphOf = useMemo(() => new Map(graphs.map((graph) => [graph.graphId, graph])), [graphs]);

  async function graphsDeleted(graphIds: string[]) {
    setSelected((current) => current.filter((id) => !graphIds.includes(id)));
    props.onDeleted?.(graphIds.flatMap((graphId) => graphOf.get(graphId)?.runIds ?? []));
  }
  const assume = trpc.auth.assume.useMutation({
    onSuccess: async () => {
      await utils.invalidate();
      setIdentityOpen(false);
    },
    onError: (error) => toast.error(error.message),
  });
  const setSuggestions = trpc.auth.setSuggestions.useMutation({
    onSuccess: async () => {
      await utils.auth.session.invalidate();
    },
  });
  const analyst = props.role === "analyst";
  const workspace = trpc.workspace.get.useQuery();

  const jumpKeys = useClientValue(() => (/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl+K"), "Ctrl+K");

  // Opening a graph shows the scope it belongs to, once per graph opened, so
  // its row is in view without taking the choice away afterwards.
  const [scopedFor, setScopedFor] = useState<string | null>(null);
  const opened = props.page === "run" ? runs.data?.find((run) => run.runId === props.selectedRunId) : undefined;
  if (opened && scopedFor !== opened.runId) {
    setScopedFor(opened.runId);
    const belongs: CatalogScope = (opened.access ?? "owner") === "owner" ? "mine" : "shared";
    if (belongs !== scope) {
      setScope(belongs);
    }
  }

  const filtered = useMemo(() => {
    const rows = graphs.map((graph) => graph.run);
    if (analyst) {
      const production = (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production");
      return production.flatMap((item) => {
        const run = rows.find((row) => row.graphId === item.graphId);
        if (!run) {
          return [];
        }
        const haystack = `${item.name} ${run.graphId}`.toLowerCase();
        if (search && !haystack.includes(search.toLowerCase())) {
          return [];
        }
        return [{ ...run, graphName: item.name }];
      });
    }
    return rows.filter((run) => {
      // The server lists only graphs the viewer may open; a runtime that
      // reports no standing is a laptop's, where every graph is the user's.
      const own = (run.access ?? "owner") === "owner";
      if ((scope === "mine" && !own) || (scope === "shared" && own)) {
        return false;
      }
      const haystack = `${run.graphName ?? ""} ${run.graphId} ${run.status}`.toLowerCase();
      if (search && !haystack.includes(search.toLowerCase())) {
        return false;
      }
      if (storage === "local" && run.storageKind !== "local_files") {
        return false;
      }
      if (storage === "snowflake" && run.storageKind !== "snowflake") {
        return false;
      }
      return true;
    });
  }, [analyst, graphs, search, storage, scope, workspace.data?.perspectives]);

  const empty = catalogEmptyCopy({
    loading: runs.isLoading,
    total: graphs.length,
    visible: filtered.length,
    search,
    storageFilter: storage,
    scope,
    identified: props.identified,
    analyst,
  });

  // Only an owner deletes a graph. A graph with a run under way cannot be
  // chosen for a bulk delete; its own row still offers Delete, and the
  // confirmation says why it has to wait.
  const canDelete = (run: RunSnapshot) =>
    props.capabilities.has("delete_graph") && !analyst && (run.access ?? "owner") === "owner";
  const selectable = useMemo(
    () => filtered.filter((run) => canDelete(run) && !isActiveStatus(run.status)).map((run) => run.graphId),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- canDelete reads only what is listed
    [filtered, analyst, props.capabilities],
  );
  const selectedShown = selectable.filter((id) => selected.includes(id));
  const selectedHidden = selected.length - selectedShown.length;
  const allShownSelected = selectable.length > 0 && selectedShown.length === selectable.length;
  const selectedGraphs: DeletableGraph[] = selected.flatMap((graphId) => {
    const graph = graphOf.get(graphId);
    return graph ? [{ graphId, graphName: graph.run.graphName || graphId }] : [];
  });

  function toggleRow(graphId: string, shift: boolean) {
    setSelected((current) => {
      const adding = !current.includes(graphId);
      let ids = [graphId];
      const anchor = lastToggled.current;
      if (shift && anchor && anchor !== graphId) {
        const from = selectable.indexOf(anchor);
        const to = selectable.indexOf(graphId);
        if (from >= 0 && to >= 0) {
          ids = selectable.slice(Math.min(from, to), Math.max(from, to) + 1);
        }
      }
      lastToggled.current = graphId;
      return adding ? [...current, ...ids.filter((id) => !current.includes(id))] : current.filter((id) => !ids.includes(id));
    });
  }

  const go = (action: () => void) => {
    action();
    props.onNavigate?.();
  };

  return (
    <aside className={cn("flex w-80 shrink-0 flex-col border-r border-border bg-sidebar", props.className)}>
      <div className="relative px-3 pt-3">
        <button
          type="button"
          aria-label="FlakeGraph home"
          title="Home"
          className="block w-full rounded-md text-left hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => go(props.onNewGraph)}
        >
          <Image
            src="/flakegraph-logo.png"
            alt=""
            width={1460}
            height={394}
            sizes="20rem"
            className="h-auto w-full object-contain"
            preload
          />
        </button>
        {props.showClose ? (
          <Button
            size="sm"
            variant="ghost"
            aria-label="Close navigation"
            className="absolute right-1 top-1"
            onClick={() => props.onNavigate?.()}
          >
            Close
          </Button>
        ) : null}
      </div>
      <div className="space-y-3 px-4 pb-3 pt-3">
        <button
          type="button"
          aria-label="Jump"
          className="flex h-8 w-full items-center gap-2 rounded-md border border-border bg-background px-2.5 text-left text-[13px] text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
          onClick={() => window.dispatchEvent(new Event("flakegraph:jump"))}
        >
          <Search className="size-3.5 shrink-0 opacity-70" />
          <span className="min-w-0 flex-1 truncate">Jump</span>
          <kbd className="rounded border border-border bg-muted px-1.5 py-px font-sans text-[10px] font-medium text-muted-foreground">
            {jumpKeys}
          </kbd>
        </button>
        <div className="overflow-hidden rounded-md border border-border">
          <button
            type="button"
            data-testid="identity-chip"
            aria-label="Session identity"
            className="flex w-full items-center gap-2 px-2.5 py-2 text-left hover:bg-muted/60"
            onClick={() => setIdentityOpen(true)}
          >
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                props.identified ? "bg-emerald-500" : "bg-amber-400",
              )}
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium leading-tight">
                {props.identified ? props.principal : "Unidentified"}
              </span>
              <span className="mt-0.5 block truncate text-[11px] leading-tight text-muted-foreground">
                {environmentLabel(props.runtime)}
                {props.identified ? "" : " — not signed in"}
              </span>
            </span>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
          </button>
          <div className="border-t border-border px-1.5 py-1.5">
            <p className="px-1 pb-1 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              Where the job runs
            </p>
            {analyst ? (
              <p className="px-1 text-xs text-muted-foreground">{environmentLabel(props.runtime)}</p>
            ) : (
            <Select value={props.runtime} onValueChange={(value) => props.onRuntimeChange(value as RuntimeMode)}>
              <SelectTrigger aria-label="Runtime" className="h-8 border-0 bg-transparent shadow-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {props.availableRuntimes.map((runtime) => (
                  <SelectItem key={runtime} value={runtime}>
                    {labelRuntime(runtime)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            )}
          </div>
        </div>
        <Dialog open={identityOpen} onOpenChange={setIdentityOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Session identity</DialogTitle>
              <DialogDescription>
                {props.identityFromGate
                  ? "Who the sign-in gate says you are. Graphs you own are listed under Mine."
                  : props.demoIdentities
                    ? "Assume a demo identity to see the catalog as that person. These are not a product login."
                    : "Who this console is serving. Identity comes from a sign-in gate or an API key."}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3 text-sm">
              <p>
                You are <span className="font-medium">{props.identified ? props.principal : "unidentified"}</span>{" "}
                ({props.role}) on {environmentLabel(props.runtime)}.
              </p>
              <p className="text-muted-foreground">
                {props.identified
                  ? "A graph is private to the person who built it until they share it. Mine lists graphs you own and Shared lists graphs other people shared with you, at Read or Write. Share a graph of yours from its Sharing tab."
                  : "A graph is private to the person who built it until they share it. Without a sign-in gate this console serves one person, so a session that has not signed in sees every graph here; sign in to see the catalog as someone."}
              </p>
              {props.identityFromGate ? (
                props.signOutUrl ? (
                  <Button size="sm" variant="outline" asChild>
                    <a href={props.signOutUrl}>Sign out</a>
                  </Button>
                ) : null
              ) : !props.demoIdentities ? (
                <p className="text-muted-foreground">
                  Demo identities are off. A laptop console serves them with{" "}
                  <span className="font-mono">FLAKEGRAPH_DEMO_IDENTITIES=1</span>; a shared one puts a sign-in gate in front.
                </p>
              ) : (
                <>
                  <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Assume principal</p>
                  <div className="flex flex-wrap gap-2">
                    {DEMO_IDENTITIES.map((identity) => (
                      <Button
                        key={identity.label}
                        size="sm"
                        variant={identity.variant}
                        pending={assume.isPending && assume.variables?.userName === identity.principal.userName}
                        disabled={assume.isPending}
                        onClick={() => assume.mutate(identity.principal)}
                      >
                        {identity.label}
                      </Button>
                    ))}
                  </div>
                </>
              )}
              <div className="space-y-1.5">
                <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Suggestions</p>
                <div className="flex flex-wrap gap-1.5">
                  {(["off", "on-request", "auto-fill"] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="sm"
                      variant={(setSuggestions.isPending ? setSuggestions.variables?.mode : props.suggestionMode) === mode ? "default" : "outline"}
                      pending={setSuggestions.isPending && setSuggestions.variables?.mode === mode}
                      disabled={!props.operator}
                      onClick={() => setSuggestions.mutate({ mode })}
                    >
                      {suggestionLabel(mode)}
                    </Button>
                  ))}
                </div>
                {props.operator ? null : (
                  <p className="text-xs text-muted-foreground" data-testid="suggestions-operator-only">
                    The whole console&apos;s setting; an operator changes it.
                  </p>
                )}
              </div>
              <div className="rounded-md border border-border bg-muted/30 p-3 text-sm">
                <p>
                  <span className="font-medium">Delete</span> — the owner removes a graph for good: every version and
                  run, everything stored for them, and the uploads only it used. It asks first and cannot be undone.
                </p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
        {analyst ? null : (
          <Button
            className="w-full"
            variant={props.page === "new" ? "default" : "outline"}
            aria-current={props.page === "new" ? "page" : undefined}
            onClick={() => go(props.onNewGraph)}
          >
            <Plus /> New graph
          </Button>
        )}
        <nav className="space-y-0.5">
          {props.capabilities.has("cluster") && !analyst ? (
            <>
              <NavButton active={props.page === "fleet"} onClick={() => go(props.onFleet)}>
                <Server className="size-3.5" /> Fleet
              </NavButton>
            </>
          ) : null}
          {props.role === "staff" || props.role === "operator" ? (
            <NavButton active={props.page === "keys"} onClick={() => go(props.onKeys)}>
              <KeyRound className="size-3.5" /> SDK keys
            </NavButton>
          ) : null}
          {props.role === "staff" ? (
            <NavButton active={props.page === "staff"} onClick={() => go(props.onStaff)}>
              <ShieldAlert className="size-3.5" /> Incident
            </NavButton>
          ) : null}
        </nav>
      </div>
      <Separator />
      <div className="space-y-2 px-4 py-3">
        <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          {analyst ? "Perspectives" : "Graphs"}
        </p>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2 size-3.5 text-muted-foreground" />
          <Input
            aria-label="Search graphs"
            className="h-8 pl-8 text-[13px]"
            placeholder="Search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        {analyst ? null : (
          <div className="flex gap-1.5">
            <div className="grid min-w-0 flex-1 grid-cols-2 rounded-md border border-border p-0.5" role="group" aria-label="Catalog scope">
              {SCOPES.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  aria-pressed={scope === item.value}
                  data-selected={scope === item.value}
                  title={item.title}
                  className={cn(
                    "h-7 rounded-sm border-0 text-xs font-medium appearance-none",
                    scope === item.value ? "bg-foreground text-background" : "bg-transparent text-muted-foreground hover:bg-accent",
                  )}
                  onClick={() => setScope(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <Select value={storage} onValueChange={setStorage}>
              <SelectTrigger aria-label="Storage filter" className="h-8 w-[7.75rem] shrink-0 px-2 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All storage</SelectItem>
                <SelectItem value="local">Local files</SelectItem>
                <SelectItem value="snowflake">Snowflake</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden lg:overflow-y-visible" data-testid="catalog-list">
        <div className="space-y-0.5 px-3 pb-8">
          {selectable.length > 0 || selected.length > 0 ? (
            <SelectionBar
              shown={selectable.length}
              selected={selected.length}
              selectedShown={selectedShown.length}
              selectedHidden={selectedHidden}
              allShownSelected={allShownSelected}
              onSelectAll={() =>
                setSelected((current) =>
                  allShownSelected
                    ? current.filter((id) => !selectable.includes(id))
                    : [...current, ...selectable.filter((id) => !current.includes(id))],
                )
              }
              onClear={() => setSelected([])}
              onDelete={() => setDeleting(selectedGraphs)}
            />
          ) : null}
          <DeleteGraphDialog
            open={deleting !== null}
            onOpenChange={(open) => (open ? undefined : setDeleting(null))}
            graphs={deleting ?? []}
            onDeleted={graphsDeleted}
          />
          {runs.isLoading && !runs.data ? (
            <div className="space-y-2 px-1" aria-busy="true" aria-label="Loading graphs">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : null}
          {runs.error ? (
            <p className="px-1 text-sm text-destructive" role="alert">
              {runs.error.message}
            </p>
          ) : null}
          {(runs.data?.length ?? 0) >= CATALOG_LIMIT ? (
            <p className="px-1 pb-1 text-xs text-muted-foreground" data-testid="catalog-capped">
              The {CATALOG_LIMIT} most recent graphs are listed; older ones open from their run link.
            </p>
          ) : null}
          {empty && !runs.error ? (
            <div className="px-1 text-sm text-muted-foreground" data-testid="catalog-empty">
              <p className="font-medium text-foreground">{empty.title}</p>
              <p>{empty.detail}</p>
              {scope === "mine" && (runs.data?.length ?? 0) > 0 ? (
                <Button className="mt-2" size="sm" variant="outline" onClick={() => setScope("shared")}>
                  Show shared graphs
                </Button>
              ) : null}
            </div>
          ) : null}
          {filtered.map((run) => (
            <RunRow
              key={run.graphId}
              run={run}
              runCount={graphOf.get(run.graphId)?.runIds.length ?? 1}
              selected={props.page === "run" && Boolean(graphOf.get(run.graphId)?.runIds.includes(props.selectedRunId ?? ""))}
              canDelete={canDelete(run)}
              selectable={selectable.includes(run.graphId)}
              checked={selected.includes(run.graphId)}
              onToggle={(shift) => toggleRow(run.graphId, shift)}
              onSelect={() => go(() => props.onSelectRun(run.runId))}
              onDelete={() => setDeleting([{ graphId: run.graphId, graphName: run.graphName || run.graphId }])}
            />
          ))}
        </div>
      </div>
    </aside>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex h-8 w-full items-center gap-2 rounded-md px-2.5 text-[13px] font-medium transition-colors",
        active ? "bg-accent text-foreground" : "text-muted-foreground hover:bg-muted/80 hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

/** "Shared by someone · Read" on another person's graph; "Shared with 2 people" on one's own. */
function sharingLine(run: RunSnapshot): string | null {
  if (run.access === "read" || run.access === "write") {
    const owner = displayPrincipal(run.owner);
    return `${owner ? `Shared by ${owner}` : "Shared with you"} · ${LEVELS[run.access].label}`;
  }
  const count = run.sharedWith ?? 0;
  return count > 0 ? `Shared with ${count} ${count === 1 ? "person" : "people"}` : null;
}

function RunRow({
  run,
  runCount,
  selected,
  canDelete,
  selectable,
  checked,
  onToggle,
  onSelect,
  onDelete,
}: {
  run: RunSnapshot;
  /** Runs the graph has: its versions and attempts. */
  runCount: number;
  selected: boolean;
  canDelete: boolean;
  /** Whether a bulk delete may take it: an owner's graph with nothing under way. */
  selectable: boolean;
  checked: boolean;
  onToggle: (shift: boolean) => void;
  onSelect: () => void;
  onDelete: () => void;
}) {
  const name = run.graphName || run.graphId;
  const active = isActiveStatus(run.status);
  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md border border-transparent border-l-4 px-2 py-1.5",
        selected ? "bg-accent" : "hover:bg-accent/60",
        statusAccent(run),
      )}
    >
      {canDelete ? (
        <input
          type="checkbox"
          className="mt-1 disabled:opacity-40"
          aria-label={`Select ${name}`}
          title={selectable ? undefined : "Still running - cancel it before deleting the graph"}
          disabled={!selectable}
          checked={checked}
          onClick={(event) => onToggle(event.shiftKey)}
          onChange={() => undefined}
        />
      ) : null}
      <button type="button" className="min-w-0 flex-1 text-left" onClick={onSelect}>
        <span className="block truncate text-[13px] font-medium leading-tight">{name}</span>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">{statusSentence(run, { nodeCount: asCount(run.raw?.nodeCount) })}</p>
        <p className="truncate text-[11px] text-muted-foreground">
          {run.storageKind === "snowflake" ? "Snowflake" : "Local files"} · {formatRelativeTime(run.updatedAt)}
          {versionBadge(run)}
          {runCount > 1 ? ` · ${runCount} runs` : ""}
        </p>
        {sharingLine(run) ? (
          <p className="flex min-w-0 items-center gap-1 text-[11px] text-muted-foreground" data-testid="catalog-sharing">
            <Users className="size-3 shrink-0" aria-hidden />
            <span className="truncate">{sharingLine(run)}</span>
          </p>
        ) : null}
      </button>
      {canDelete ? (
        <Button
          size="icon"
          variant="ghost"
          aria-label={`Delete ${name}`}
          title={active ? "Delete - cancel its running work first" : "Delete this graph and everything stored for it"}
          className="size-7 shrink-0"
          onClick={onDelete}
        >
          <Trash2 className="size-3.5" />
        </Button>
      ) : null}
    </div>
  );
}

/**
 * One row above the list that makes a selection visible and actionable:
 * a tri-state box selects or clears every row the filters show, the count
 * says what is held (and what a filter hides), and Delete asks first.
 */
function SelectionBar({
  shown,
  selected,
  selectedShown,
  selectedHidden,
  allShownSelected,
  onSelectAll,
  onClear,
  onDelete,
}: {
  shown: number;
  selected: number;
  selectedShown: number;
  selectedHidden: number;
  allShownSelected: boolean;
  onSelectAll: () => void;
  onClear: () => void;
  onDelete: () => void;
}) {
  const box = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (box.current) {
      box.current.indeterminate = selectedShown > 0 && !allShownSelected;
    }
  }, [selectedShown, allShownSelected]);
  return (
    <div
      className={cn(
        "mb-1.5 flex h-8 items-center gap-2 rounded-md px-2 text-xs",
        selected > 0 ? "bg-accent text-foreground" : "text-muted-foreground",
      )}
      data-testid="catalog-selection"
    >
      <input
        ref={box}
        type="checkbox"
        aria-label={allShownSelected ? "Clear the shown graphs" : "Select all shown graphs"}
        title={shown === 0 ? "Nothing shown can be deleted" : undefined}
        disabled={shown === 0}
        checked={allShownSelected}
        onChange={onSelectAll}
      />
      <span className="min-w-0 flex-1 truncate">
        {selected === 0
          ? `Select all ${shown}`
          : selectedHidden > 0
            ? `${selected} selected · ${selectedHidden} not shown`
            : `${selected} of ${shown} selected`}
      </span>
      {selected > 0 ? (
        <>
          <Button size="sm" variant="ghost" className="h-6 px-1.5 text-xs" onClick={onClear}>
            Clear
          </Button>
          <Button
            size="sm"
            variant="outline"
            aria-label="Delete selected"
            className="h-6 border-destructive/40 px-2 text-xs text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={onDelete}
          >
            <Trash2 className="mr-1 size-3" aria-hidden="true" />
            Delete {selected}
          </Button>
        </>
      ) : null}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  if (isSuccessStatus(normalized)) {
    return <Badge variant="success">succeeded</Badge>;
  }
  if (normalized === "cancelling") {
    return <Badge variant="warning">cancelling</Badge>;
  }
  if (isActiveStatus(normalized)) {
    return <Badge variant="secondary">{normalized}</Badge>;
  }
  if (normalized === ARTIFACTS_UNAVAILABLE_STATUS) {
    return <Badge variant="warning">unavailable</Badge>;
  }
  if (normalized === "interrupted") {
    return <Badge variant="warning">interrupted</Badge>;
  }
  if (normalized === "cancelled") {
    return <Badge variant="outline">cancelled</Badge>;
  }
  return <Badge variant="danger">{normalized}</Badge>;
}

function labelRuntime(runtime: RuntimeMode): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Local";
}

function environmentLabel(runtime: RuntimeMode): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Laptop · local files";
}

function suggestionLabel(mode: "off" | "on-request" | "auto-fill"): string {
  if (mode === "on-request") {
    return "On request";
  }
  if (mode === "auto-fill") {
    return "Auto-fill";
  }
  return "Off";
}

/** " · v2 · head" for a run whose graph has more than one published version. */
function versionBadge(run: RunSnapshot): string {
  const version = run.raw?.version;
  if (!version || typeof version !== "object") {
    return "";
  }
  const { number, count, head } = version as { number: number; count: number; head: boolean };
  if (!number) {
    return run.raw?.baseRunId && isActiveStatus(run.status) ? ` · new version` : "";
  }
  if (count < 2) {
    return "";
  }
  return ` · v${number}${head ? " · head" : ""}`;
}

function statusAccent(run: RunSnapshot): string {
  const normalized = run.status.toLowerCase();
  if (asCount(run.raw?.nodeCount) === 0 && isSuccessStatus(normalized)) {
    return "border-l-destructive";
  }
  if (isSuccessStatus(normalized)) {
    return "border-l-emerald-500";
  }
  if (normalized === "interrupted" || normalized === ARTIFACTS_UNAVAILABLE_STATUS || normalized === "cancelling") {
    return "border-l-amber-500";
  }
  if (isActiveStatus(normalized)) {
    return "border-l-sky-500";
  }
  if (normalized === "cancelled") {
    return "border-l-border";
  }
  return "border-l-destructive";
}

function asCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
