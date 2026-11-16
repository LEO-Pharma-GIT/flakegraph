"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { Menu } from "lucide-react";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { AppProviders, trpc } from "@/components/providers";
import { Sidebar } from "@/components/console/sidebar";
import { IngestionForm } from "@/components/console/ingestion-form";
import { RunWorkspace } from "@/components/console/run-workspace";
import { FleetView } from "@/components/console/fleet-view";
import { CommandPalette } from "@/components/console/command-palette";
import { StaffIncidentPage } from "@/components/console/operator-tools";
import { SdkKeysPage } from "@/components/console/sdk-keys";
import { AnalystHome, WelcomeBack } from "@/components/console/workspace-panels";
import { PageHeader } from "@/components/console/page-header";
import type { RuntimeMode } from "@/server/protocol/schema";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const runtimeParser = parseAsStringLiteral(["local", "kubernetes", "snowflake"] as const);
const pageParser = parseAsStringLiteral(["new", "run", "fleet", "staff", "keys"] as const);

export function ConsoleApp() {
  return (
    <AppProviders>
      <Suspense fallback={<p className="p-6 text-sm text-muted-foreground">Loading control plane…</p>}>
        <ConsoleInner />
      </Suspense>
    </AppProviders>
  );
}

function ConsoleInner() {
  const [{ runtime, page, run: runId }, setNavigation] = useQueryStates({
    runtime: runtimeParser,
    page: pageParser.withDefault("new"),
    run: parseAsString,
  });
  const setPage = useCallback((value: typeof page) => setNavigation({ page: value }), [setNavigation]);
  const navigateTo = (destination: typeof page, run: string | null) => setNavigation({ page: destination, run });
  const [navOpen, setNavOpen] = useState(false);
  const [jumpSearch, setJumpSearch] = useState("");
  const utils = trpc.useUtils();
  const session = trpc.auth.session.useQuery();
  const runs = trpc.runs.list.useQuery({ limit: 100 });
  const role = session.data?.role ?? "operator";
  const analyst = role === "analyst";
  const canCluster = Boolean(session.data?.capabilities?.includes("cluster"));
  const currentGraph = (runs.data ?? []).find((run) => run.runId === runId);
  const currentGraphName = currentGraph?.graphName || currentGraph?.graphId || null;

  useEffect(() => {
    if (!navOpen) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setNavOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navOpen]);

  useEffect(() => {
    const titles: Record<string, string> = {
      new: analyst ? "Perspectives" : "New graph",
      run: currentGraphName || "Graph",
      fleet: "Fleet",
      staff: "Staff incident",
      keys: "SDK keys",
    };
    document.title = `${titles[page] ?? "Console"} · FlakeGraph`;
  }, [analyst, currentGraphName, page]);

  useEffect(() => {
    if (!session.data) {
      return;
    }
    if (analyst && (page === "keys" || page === "staff" || page === "fleet")) {
      void setPage("new");
      return;
    }
    if (role !== "staff" && page === "staff") {
      void setPage("new");
      return;
    }
    if (!canCluster && page === "fleet") {
      void setPage("new");
    }
  }, [analyst, canCluster, page, role, session.data, setPage]);

  const onRuntimeChange = async (next: RuntimeMode) => {
    const nextPage = page === "fleet" && next === "kubernetes" ? "fleet" : "new";
    setJumpSearch("");
    await setNavigation({ runtime: next, run: null, page: nextPage });
    await utils.invalidate();
  };

  if (session.isLoading) {
    return (
      <div className="flex min-h-svh">
        <div className="w-80 border-r p-4">
          <Skeleton className="h-16 w-52" />
        </div>
        <div className="flex-1 p-6">
          <Skeleton className="h-8 w-64" />
        </div>
      </div>
    );
  }

  if (session.error) {
    return (
      <main className="flex min-h-screen items-center justify-center p-8">
        <p role="alert">Unable to load the control plane: {session.error.message}</p>
      </main>
    );
  }

  const capabilities = new Set(session.data?.capabilities ?? []);
  const effectiveRuntime = session.data?.runtime ?? runtime ?? "local";

  return (
    <div className="flex min-h-svh bg-background">
      <CommandPalette
        runId={runId}
        role={role}
        canFleet={canCluster && !analyst}
        onSelectRun={async (id) => {
          setJumpSearch("");
          await navigateTo("run", id);
        }}
        onJumpEntity={async (id, search) => {
          setJumpSearch(search);
          await navigateTo("run", id);
        }}
        onNewGraph={async () => {
          setJumpSearch("");
          await navigateTo("new", null);
        }}
        onFleet={async () => {
          await navigateTo("fleet", null);
        }}
        onStaff={async () => {
          await navigateTo("staff", null);
        }}
        onKeys={async () => {
          await navigateTo("keys", null);
        }}
      />
      {navOpen ? (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
          onClick={() => setNavOpen(false)}
        />
      ) : null}
      <Sidebar
        className={cn(navOpen ? "fixed inset-y-0 left-0 z-40 flex shadow-xl" : "hidden", "lg:relative lg:flex lg:shadow-none")}
        onNavigate={() => setNavOpen(false)}
        showClose={navOpen}
        runtime={effectiveRuntime}
        availableRuntimes={session.data?.availableRuntimes ?? ["local"]}
        capabilities={capabilities}
        page={page}
        selectedRunId={runId}
        identified={Boolean(session.data?.identified)}
        identityFromGate={Boolean(session.data?.identityFromGate)}
        demoIdentities={Boolean(session.data?.demoIdentities)}
        signOutUrl={session.data?.signOutUrl ?? null}
        principal={session.data?.viewer.userName || "unidentified"}
        role={role}
        suggestionMode={session.data?.suggestionMode ?? "on-request"}
        operator={Boolean(session.data?.operator)}
        onRuntimeChange={onRuntimeChange}
        onNewGraph={async () => {
          setJumpSearch("");
          await navigateTo("new", null);
        }}
        onFleet={async () => {
          await navigateTo("fleet", null);
        }}
        onStaff={async () => {
          await navigateTo("staff", null);
        }}
        onKeys={async () => {
          await navigateTo("keys", null);
        }}
        onSelectRun={async (id) => {
          setJumpSearch("");
          await navigateTo("run", id);
        }}
        onDeleted={async (ids) => {
          if (runId && ids.includes(runId)) {
            await navigateTo("new", null);
          }
        }}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b border-border bg-background px-3 py-2 lg:hidden">
          <Button size="icon" variant="ghost" aria-label={navOpen ? "Close navigation" : "Open navigation"} aria-expanded={navOpen} onClick={() => setNavOpen((open) => !open)}>
            <Menu />
          </Button>
          <span className="min-w-0 flex-1 truncate text-sm font-semibold">{mobileTitle(page, analyst, currentGraphName)}</span>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => window.dispatchEvent(new Event("flakegraph:jump"))}
          >
            Jump
          </Button>
        </div>
        <main className="min-w-0 flex-1">
        <div className="flex min-h-full flex-col p-4 pb-24 sm:px-6 sm:pb-24 lg:px-8 lg:pb-24">
        {role === "staff" ? (
          <Alert className="mb-4 py-1.5 text-xs">
            Staff tools act with your own access: only the graphs you own or that are shared with you.
          </Alert>
        ) : null}
        {role !== "staff" && (page === "new" || page === "run") ? (
          <WelcomeBack
            identified={Boolean(session.data?.identified)}
            suggestionMode={session.data?.suggestionMode ?? "on-request"}
            currentRunId={page === "run" ? runId : null}
            analyst={analyst}
            onOpenRun={async (id) => {
              await navigateTo("run", id);
            }}
          />
        ) : null}
        {page === "new" && !analyst ? (
          <div className="flex min-h-0 flex-1 flex-col">
          <IngestionForm
            runtime={effectiveRuntime}
            capabilities={capabilities}
            lastSuccessRuntime={session.data?.lastSuccessRuntime ?? null}
            suggestionMode={session.data?.suggestionMode ?? "on-request"}
            onSubmitted={async (id) => {
              await utils.runs.list.invalidate();
              await navigateTo("run", id);
            }}
          />
          </div>
        ) : null}
        {page === "new" && analyst ? (
          <AnalystHome
            onOpenRun={async (id) => {
              await navigateTo("run", id);
            }}
          />
        ) : null}
        {page === "run" && runId ? (
          <RunWorkspace
            key={runId}
            runId={runId}
            capabilities={capabilities}
            runtime={effectiveRuntime}
            role={role}
            suggestionMode={session.data?.suggestionMode ?? "on-request"}
            jumpSearch={jumpSearch}
            onJumpConsumed={() => setJumpSearch("")}
            onNewGraph={async () => {
              setJumpSearch("");
              await navigateTo("new", null);
            }}
            onDeleted={async () => {
              await utils.runs.list.invalidate();
              await navigateTo("new", null);
            }}
            onOpenRun={async (id) => {
              setJumpSearch("");
              await utils.runs.list.invalidate();
              await navigateTo("run", id);
            }}
          />
        ) : null}
        {page === "run" && !runId ? (
          <div className="space-y-4">
            <PageHeader
              kicker="Graph"
              title="No graph selected"
              description={
                analyst
                  ? "Open navigation to pick a production perspective."
                  : "Open navigation to pick a graph, or start a new one."
              }
            />
            {analyst ? null : (
              <Button
                onClick={async () => {
                  await navigateTo("new", null);
                }}
              >
                New graph
              </Button>
            )}
          </div>
        ) : null}
        {page === "fleet" && canCluster && !analyst ? <FleetView /> : null}
        {page === "staff" && role === "staff" ? <StaffIncidentPage /> : null}
        {page === "keys" && (role === "staff" || role === "operator") ? <SdkKeysPage /> : null}
        </div>
        </main>
      </div>
    </div>
  );
}

function mobileTitle(page: string, analyst: boolean, graphName?: string | null): string {
  if (page === "new") {
    return analyst ? "Perspectives" : "New graph";
  }
  if (page === "run") {
    return graphName || "Graph";
  }
  if (page === "fleet") {
    return "Fleet";
  }
  if (page === "staff") {
    return "Staff";
  }
  if (page === "keys") {
    return "SDK keys";
  }
  return "FlakeGraph";
}
