"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusBadge } from "@/components/console/sidebar";
import { GraphExplorer } from "@/components/console/graph-explorer";
import { GraphDataTables, type ExploreFocus } from "@/components/console/graph-data-tables";
import { ConsumptionPanel, consumptionExport } from "@/components/console/consumption-panel";
import { ExtractionGapsCard } from "@/components/console/extraction-gaps-card";
import { RejectedRecordsCard } from "@/components/console/rejected-records-card";
import { AccessBadge, GraphSharing } from "@/components/console/graph-sharing";
import { DeleteGraphDialog } from "@/components/console/delete-graph-dialog";
import { FailedDocumentsCard } from "@/components/console/failed-documents-card";
import { BenchmarkCard } from "@/components/console/benchmark-card";
import { GoldCard } from "@/components/console/gold-card";
import { PublishSnowflakeDialog, fleetSnowflakeFromProfile } from "@/components/console/publish-snowflake-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AlertTriangle, ChevronDown, FileX, Lock } from "lucide-react";
import { buildGraphML } from "@/lib/graph-geometry";
import { graphFacetOptions, graphStarterQuestions } from "@/lib/graph-facets";
import { graphNodeLabel } from "@/lib/graph-geometry";
import { downloadFile, downloadJson } from "@/lib/downloads";
import { resolvedFailures } from "@/lib/resolved-failures";
import type { BenchmarkBaseline, BenchmarkReport } from "@/server/benchmark";
import type { ConsumptionEstimate } from "@/server/estimate";
import type { FailedDocuments } from "@/server/failed-documents";
import type { ExtractionGaps } from "@/server/gaps";
import type { RejectedRecords } from "@/server/rejected";
import type { HitlItem } from "@/server/workspace";
import { GuideCard } from "@/components/console/guide-card";
import { DocumentTable, PhaseSummary, documentsNeedingAttention } from "@/components/console/document-table";
import { GraphEditor, GraphVersionsCard } from "@/components/console/graph-editor";
import { PageHeader } from "@/components/console/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { AskPanel, ReviewPanel, VersionsPanel, WatchPanel } from "@/components/console/workspace-panels";
import { ShowMore } from "@/components/console/show-more";
import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  graphCounts,
  isActiveStatus,
  isSuccessStatus,
  type Capability,
  type GraphDataset,
  type RunSnapshot,
  type StageProgress,
} from "@/server/protocol/schema";
import { readLastIngestion, writeLastIngestion } from "@/lib/last-ingestion";
import { tailLabel } from "@/lib/paging";
import { formatDocumentPhase, formatDuration, formatInstant, formatRelativeTime, titleCaseStage } from "@/lib/utils";
import type { DocumentStatus } from "@/server/documents";
import { statusSentence } from "@/lib/status-sentence";
import { FINISHED_GRAPH_QUERY } from "./graph-query";

export function RunWorkspace({
  runId,
  capabilities,
  runtime = "local",
  role = "operator",
  suggestionMode: _suggestionMode = "on-request",
  jumpSearch = "",
  onJumpConsumed,
  onDeleted,
  onNewGraph,
  onOpenRun,
}: {
  runId: string;
  capabilities: Set<Capability>;
  runtime?: "local" | "kubernetes" | "snowflake";
  role?: "operator" | "analyst" | "staff";
  suggestionMode?: "off" | "on-request" | "auto-fill";
  jumpSearch?: string;
  onJumpConsumed?: () => void;
  onDeleted?: () => Promise<void> | void;
  onNewGraph?: () => Promise<void> | void;
  /** Open another run of this catalog, such as another version of the graph. */
  onOpenRun?: (runId: string) => Promise<void> | void;
}) {
  const utils = trpc.useUtils();
  const run = trpc.runs.get.useQuery(
    { runId },
    { refetchInterval: (query) => (isActiveStatus(query.state.data?.status ?? "") ? 3_000 : false) },
  );
  const graph = trpc.graphs.loadRun.useQuery(
    { runId },
    { ...FINISHED_GRAPH_QUERY, enabled: isSuccessStatus(run.data?.status ?? "") },
  );
  const quality = trpc.graphs.quality.useQuery(
    { runId },
    { enabled: isSuccessStatus(run.data?.status ?? "") },
  );
  const session = trpc.auth.session.useQuery();
  // Cancel and retry answer with the run as the control plane now sees it;
  // that answer goes into the page at once, and the refetch behind it only
  // confirms. Otherwise the page shows the old state for as long as one
  // more status read takes, which reads as the click having done nothing.
  const cancel = trpc.runs.cancel.useMutation({
    onSuccess: async (snapshot) => {
      toast.success("Cancellation requested");
      utils.runs.get.setData({ runId }, snapshot);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const retry = trpc.runs.retry.useMutation({
    onSuccess: async (snapshot) => {
      toast.success(run.data?.status.toLowerCase() === "cancelled" ? "Run resumed" : "Retry submitted");
      utils.runs.get.setData({ runId }, snapshot);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const recover = trpc.runs.recover.useMutation({
    onSuccess: async (message) => {
      toast.success(message);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const rename = trpc.graphs.rename.useMutation({
    onSuccess: async (name) => {
      toast.success(`Renamed to ${name}`);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const documents = trpc.runs.documents.useQuery(
    { runId },
    { refetchInterval: isActiveStatus(run.data?.status ?? "") ? 5_000 : false },
  );
  const skipFile = trpc.runs.skipFile.useMutation({
    onSuccess: async () => {
      toast.success("File quarantined. Later retries skip it.");
      await documents.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  // The file whose skip is in flight, so only its button shows it.
  const skipping = skipFile.isPending ? (skipFile.variables?.fileId ?? null) : null;
  const workspace = trpc.workspace.get.useQuery();
  const pin = trpc.graphs.pin.useMutation({
    onSuccess: async () => {
      toast.success("Pinned as looks wrong. This is not a delete.");
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const [renameOpen, setRenameOpen] = useState(false);
  const [publishOpen, setPublishOpen] = useState(false);
  const fleet = trpc.fleet.profile.useQuery(undefined, { enabled: runtime === "kubernetes" });
  const fleetSnowflake = runtime === "kubernetes" ? fleetSnowflakeFromProfile(fleet.data) : null;
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [nextName, setNextName] = useState("");
  const [exploreSeed, setExploreSeed] = useState("");
  const [workspaceTab, setWorkspaceTab] = useState<string | null>(null);
  // What Explore's filters leave: the Data tab tables and the Export menu
  // read it, so a filter set on the canvas narrows both.
  const [focus, setFocus] = useState<ExploreFocus>({ nodes: [], edges: [], summary: null });
  const [jump, setJump] = useState<{ id: string; nonce: number } | null>(null);
  const [neighborhoodJump, setNeighborhoodJump] = useState<{ communityId: string; nonce: number } | null>(null);
  const [clearFiltersNonce, setClearFiltersNonce] = useState(0);

  // A jump lands in the explorer with its search; the shell is told once the
  // jump has been taken so it does not fire again on the next graph.
  const [jumpTaken, setJumpTaken] = useState("");
  if (jumpSearch && jumpTaken !== jumpSearch) {
    setJumpTaken(jumpSearch);
    setExploreSeed(jumpSearch);
    setWorkspaceTab("explore");
  } else if (!jumpSearch && jumpTaken) {
    setJumpTaken("");
  }
  useEffect(() => {
    if (jumpSearch) {
      onJumpConsumed?.();
    }
  }, [jumpSearch, onJumpConsumed]);

  // Pending covers a load that is waiting to be retried, not only one in
  // flight: until the run or an error arrives there is nothing to say.
  if (run.isPending) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
        <p className="text-sm text-muted-foreground">Loading run…</p>
      </div>
    );
  }
  if (run.error || !run.data) {
    const hidden = run.error?.data?.code === "FORBIDDEN";
    return (
      <div className="space-y-3">
        <PageHeader
          kicker="Graph"
          title={hidden ? "You don't have access to this graph" : "Run not found"}
          description={
            hidden
              ? "Graphs are private to their owner and the people they share them with. Ask the owner to share it with you."
              : "Pick a graph from the sidebar, or go back to New graph."
          }
        />
        {hidden ? (
          <Alert className="flex items-center gap-2" data-testid="no-access">
            <Lock className="size-4 shrink-0" aria-hidden />
            {run.error?.message}
          </Alert>
        ) : (
          <Alert variant="destructive">{run.error?.message ?? "Unknown run"}</Alert>
        )}
        {hidden || !onNewGraph ? null : <Button onClick={() => void onNewGraph()}>New graph</Button>}
      </div>
    );
  }

  const snapshot = run.data;
  const status = snapshot.status.toLowerCase();
  const operator = role !== "analyst";
  // What the viewer may do with this graph; a runtime that reports no
  // standing is a laptop, where everything is the user's.
  const access = snapshot.access ?? "owner";
  const canWrite = access === "owner" || access === "write";
  const canOwn = access === "owner";
  const canRevise = operator && canWrite && capabilities.has("revise") && isSuccessStatus(status);
  const nodeCount = graph.data ? graphCounts(graph.data).nodes : null;
  const sentence = statusSentence(snapshot, { nodeCount });
  const zeroEntities = quality.data?.zeroEntityFailure || nodeCount === 0;

  function cloneToCompose() {
    const last = readLastIngestion(runtime);
    const raw = snapshot.raw ?? {};
    writeLastIngestion({
      runtime,
      graphName: snapshot.graphName || last?.graphName || "",
      sourceKind: String(raw.sourceKind || last?.sourceKind || "local_path"),
      sourcePath: String(raw.sourcePath || watchPrefixFrom(snapshot) || last?.sourcePath || ""),
      ...(raw.source && typeof raw.source === "object"
        ? { source: raw.source as Record<string, unknown> }
        : last?.source
          ? { source: last.source }
          : {}),
      ocrProvider: String(raw.ocrProvider || last?.ocrProvider || "fallback"),
      llmProvider: String(raw.llmProvider || last?.llmProvider || "vllm_local"),
      embeddingProvider: String(raw.embeddingProvider || last?.embeddingProvider || "sentence_transformers"),
      savedAt: new Date().toISOString(),
    });
    if (typeof window !== "undefined") {
      sessionStorage.setItem("flakegraph.pending-clone", "1");
    }
    return onNewGraph?.();
  }

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Graph"
        title={snapshot.graphName || snapshot.graphId}
        description={
          <>
            {snapshot.graphId} · {snapshot.storageKind === "snowflake" ? snapshot.storageLocation : snapshot.outputPath} ·{" "}
            {formatRelativeTime(snapshot.updatedAt)}
            {versionLine(snapshot) ? <> · {versionLine(snapshot)}</> : null}
            {graph.data ? <> · {graphCountsLine(graph.data)}</> : null}
          </>
        }
        actions={
          <>
            {status === "failed" || zeroEntities ? null : <StatusBadge status={snapshot.status} />}
            <AccessBadge role={snapshot.access} sharedWith={snapshot.sharedWith} />
            {operator && canWrite && capabilities.has("rename") ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setNextName(snapshot.graphName || snapshot.graphId);
                  setRenameOpen(true);
                }}
              >
                Rename
              </Button>
            ) : null}
            {operator && canWrite && capabilities.has("recover") && isActiveStatus(status) ? (
              <Button size="sm" variant="outline" pending={recover.isPending} onClick={() => recover.mutate({ runId })}>
                {recover.isPending ? "Recovering…" : "Recover workers"}
              </Button>
            ) : null}
            {isSuccessStatus(status) ? (
              <ExportMenu
                snapshot={snapshot}
                dataset={graph.data}
                quality={quality.data}
                focus={focus}
                estimate={workspace.data?.estimates[runId] ?? null}
                reviewBundle={runtime !== "snowflake"}
                snowflake={
                  runtime === "kubernetes"
                    ? fleetSnowflake
                      ? { available: true, onPublish: () => setPublishOpen(true) }
                      : { available: false, onPublish: () => undefined }
                    : null
                }
              />
            ) : null}
            {operator && canOwn && capabilities.has("delete_graph") ? (
              <Button size="sm" variant="destructive" onClick={() => setDeleteOpen(true)}>
                Delete graph
              </Button>
            ) : null}
          </>
        }
      />
      <p className="text-sm" data-testid="status-sentence">
        {sentence}
        {quality.data?.failed.documents ? (
          <>
            {" · "}
            <button
              type="button"
              className="inline-flex items-center gap-1 text-red-700 underline-offset-2 hover:underline dark:text-red-300"
              data-testid="failed-documents-link"
              onClick={() => setWorkspaceTab("quality")}
            >
              <FileX className="size-3.5" aria-hidden />
              {quality.data.failed.documents.toLocaleString()}{" "}
              {quality.data.failed.documents === 1 ? "document" : "documents"} could not be read
            </button>
          </>
        ) : null}
        {quality.data?.gaps.windows ? (
          <>
            {" · "}
            <button
              type="button"
              className="inline-flex items-center gap-1 text-amber-800 underline-offset-2 hover:underline dark:text-amber-200"
              data-testid="gaps-link"
              onClick={() => setWorkspaceTab("quality")}
            >
              <AlertTriangle className="size-3.5" aria-hidden />
              {quality.data.gaps.windows.toLocaleString()} extraction {quality.data.gaps.windows === 1 ? "gap" : "gaps"} in{" "}
              {quality.data.gaps.documents.toLocaleString()} {quality.data.gaps.documents === 1 ? "document" : "documents"}
            </button>
          </>
        ) : null}
      </p>

      {guideForRun({
        status,
        sentence,
        zeroEntities: Boolean(zeroEntities),
        canRetry: canWrite && capabilities.has("retry"),
        canCancel: canWrite && capabilities.has("cancel") && isActiveStatus(status),
        onRetry: () => retry.mutate({ runId }),
        onCancel: () => cancel.mutate({ runId }),
        retrying: retry.isPending,
        cancelling: cancel.isPending,
        onExport: () => {
          downloadJson(`${snapshot.graphId}-review-bundle.json`, reviewBundle(snapshot, graph.data, quality.data, (workspace.data?.reviews ?? []).filter((item) => item.graphId === snapshot.graphId)));
          toast.success("Downloaded review bundle");
        },
        onNewGraph,
        onCloneConfig: () => cloneToCompose(),
        onDelete: canOwn && capabilities.has("delete_graph") ? () => setDeleteOpen(true) : undefined,
        onOpenQuality: () => setWorkspaceTab("quality"),
        staleQueue: isStaleQueue(snapshot),
      })}

      {snapshot.error && !sentence.includes(snapshot.error) ? (
        <Alert variant="destructive">{snapshot.error}</Alert>
      ) : null}

      {status === ARTIFACTS_UNAVAILABLE_STATUS ? (
        <Alert>
          This graph completed, but its files are not available on this host. The catalog keeps the record so the
          artifacts can still exist where they were written.
        </Alert>
      ) : null}

      {zeroEntities && isSuccessStatus(status) && !sentence.toLowerCase().includes("0 entities") ? (
        <Alert variant="destructive">
          Pipeline finished with 0 entities. This is not a succeeded graph. Check OCR pages or the schema, then retry.
        </Alert>
      ) : null}

      {isActiveStatus(status) || status === "interrupted" ? (
        <ProgressPanel
          sentence={sentence}
          stages={snapshot.stages}
          documentsCompleted={snapshot.documentsCompleted}
          documentsTotal={snapshot.documentsTotal}
          estimate={workspace.data?.estimates[runId] ?? null}
          documents={documents.data ?? []}
          onSkip={canWrite ? (fileId) => skipFile.mutate({ runId, fileId }) : undefined}
          skipping={skipping}
        />
      ) : null}

      {isSuccessStatus(status) ? (
        <Tabs value={workspaceTab ?? (zeroEntities ? "quality" : "explore")} onValueChange={setWorkspaceTab}>
          <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
            <TabsTrigger value="explore">Explore</TabsTrigger>
            {capabilities.has("ask") ? <TabsTrigger value="ask">Ask</TabsTrigger> : null}
            <TabsTrigger value="quality">Quality</TabsTrigger>
            <TabsTrigger value="review">Review</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
            {canRevise ? <TabsTrigger value="edit">Edit</TabsTrigger> : null}
            <TabsTrigger value="data">Data</TabsTrigger>
            <TabsTrigger value="details">Run details</TabsTrigger>
            {capabilities.has("share") ? <TabsTrigger value="sharing">Sharing</TabsTrigger> : null}
          </TabsList>
          <TabsContent
            value="explore"
            forceMount
            hidden={(workspaceTab ?? (zeroEntities ? "quality" : "explore")) !== "explore"}
            className={(workspaceTab ?? (zeroEntities ? "quality" : "explore")) === "explore" ? undefined : "hidden"}
          >
            {graph.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-9 w-80" />
                <Skeleton className="h-72 w-full" />
                <p className="text-sm text-muted-foreground">Loading graph…</p>
              </div>
            ) : null}
            {graph.error ? <Alert variant="destructive">{graph.error.message}</Alert> : null}
            {graph.data ? (
              <GraphExplorer
                dataset={graph.data}
                perspectives={(workspace.data?.perspectives ?? []).filter((item) => item.graphId === snapshot.graphId)}
                analyst={role === "analyst"}
                missingGold={quality.data?.gold?.missingRequired ?? []}
                goldPending={quality.isLoading}
                goldPresent={Boolean(quality.data?.gold)}
                initialSearch={exploreSeed}
                onPin={
                  canWrite
                    ? (targetId, kind) => pin.mutate({ graphId: snapshot.graphId, targetId, kind, note: "looks wrong" })
                    : undefined
                }
                onFocusChange={setFocus}
                jump={jump}
                neighborhoodJump={neighborhoodJump}
                clearFiltersNonce={clearFiltersNonce}
              />
            ) : null}
          </TabsContent>
          {capabilities.has("ask") ? (
            <TabsContent value="ask">
              <AskPanel
                runId={runId}
                perspectives={(workspace.data?.perspectives ?? []).filter(
                  (item) => item.graphId === snapshot.graphId && (operator || item.lifecycle === "production"),
                )}
                graphQuestions={graph.data ? graphStarterQuestions(graph.data.communities) : []}
                onOpenEntity={(name) => {
                  setExploreSeed(name);
                  setWorkspaceTab("explore");
                }}
              />
            </TabsContent>
          ) : null}
          <TabsContent value="quality">
            <QualityPanel
              runId={runId}
              graphId={snapshot.graphId}
              report={quality.data}
              loading={quality.isLoading}
              error={quality.error?.message}
              inspectHref={`/api/inspect?run=${runId}`}
              onChanged={() => quality.refetch()}
            />
          </TabsContent>
          <TabsContent value="review">
              <ReviewPanel
                graphId={snapshot.graphId}
                runId={runId}
                canExport={runtime !== "snowflake"}
                canReview={canWrite}
                relationTypes={graph.data ? graphFacetOptions(graph.data).relationTypes.map((item) => item.id) : []}
                entityNames={graph.data ? graph.data.nodes.map((node) => graphNodeLabel(node)) : []}
              />
          </TabsContent>
          {canRevise ? (
            <TabsContent value="edit">
              <GraphEditor
                snapshot={snapshot}
                documents={documents.data ?? []}
                runtime={runtime}
                capabilities={capabilities}
                onSubmitted={onOpenRun}
              />
            </TabsContent>
          ) : null}
          <TabsContent value="versions">
            <div className="space-y-4">
              {capabilities.has("revise") ? (
                // The fleet keeps the graph's versions itself, and a new one
                // is built from the Edit tab; the workspace's own version
                // labels and watches belong to runtimes without either.
                <GraphVersionsCard graphId={snapshot.graphId} runId={runId} onOpenRun={onOpenRun} />
              ) : (
                <>
                  <VersionsPanel graphId={snapshot.graphId} canPublish={operator && canWrite} />
                  {operator && canWrite ? (
                    <WatchPanel
                      graphId={snapshot.graphId}
                      sourcePath={watchPrefixFrom(snapshot)}
                      onProcessNow={() => void cloneToCompose()}
                    />
                  ) : null}
                </>
              )}
            </div>
          </TabsContent>
          <TabsContent value="data">
            {graph.data ? (
              <GraphDataTables
                dataset={graph.data}
                focus={focus}
                missingGold={quality.data?.gold?.missingRequired ?? []}
                selectedId={jump?.id ?? null}
                onShowInGraph={(id) => {
                  setJump((current) => ({ id, nonce: (current?.nonce ?? 0) + 1 }));
                  setWorkspaceTab("explore");
                }}
                onFocusNeighborhood={(communityId) => {
                  setNeighborhoodJump((current) => ({ communityId, nonce: (current?.nonce ?? 0) + 1 }));
                  setWorkspaceTab("explore");
                }}
                onClearFilters={() => setClearFiltersNonce((current) => current + 1)}
              />
            ) : null}
          </TabsContent>
          <TabsContent value="details">
            <RunDetails
              snapshot={snapshot}
              documents={documents.data ?? []}
              gaps={quality.data?.gaps}
              onSkip={canWrite ? (fileId) => skipFile.mutate({ runId, fileId }) : undefined}
              skipping={skipping}
              consumption={graph.data?.graphMetrics.consumption}
              estimate={workspace.data?.estimates[runId] ?? null}
            />
          </TabsContent>
          {capabilities.has("share") ? (
            <TabsContent value="sharing">
              <GraphSharing
                graphId={snapshot.graphId}
                graphName={snapshot.graphName || snapshot.graphId}
                viewer={session.data?.viewer.userName ?? ""}
                onLeft={async () => {
                  await utils.runs.list.invalidate();
                  await onDeleted?.();
                }}
              />
            </TabsContent>
          ) : null}
        </Tabs>
      ) : null}

      {!isActiveStatus(status) && !isSuccessStatus(status) && status !== ARTIFACTS_UNAVAILABLE_STATUS ? (
        <RunDetails snapshot={snapshot} />
      ) : null}

      {fleetSnowflake ? (
        <PublishSnowflakeDialog
          open={publishOpen}
          onOpenChange={setPublishOpen}
          runId={runId}
          graphName={snapshot.graphName || snapshot.graphId}
          fleet={fleetSnowflake}
        />
      ) : null}
      <Dialog open={renameOpen} onOpenChange={(open) => (rename.isPending ? undefined : setRenameOpen(open))}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename graph</DialogTitle>
            <DialogDescription>The stable graph ID does not change.</DialogDescription>
          </DialogHeader>
          <form
            className="space-y-3"
            onSubmit={(event) => {
              event.preventDefault();
              if (!nextName.trim() || rename.isPending) {
                return;
              }
              rename.mutate(
                { graphId: snapshot.graphId, displayName: nextName },
                { onSuccess: () => setRenameOpen(false) },
              );
            }}
          >
            <Input
              value={nextName}
              onChange={(event) => setNextName(event.target.value)}
              aria-label="Graph name"
              disabled={rename.isPending}
              autoFocus
            />
            <div className="flex justify-end gap-2">
              <Button variant="outline" disabled={rename.isPending} onClick={() => setRenameOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" pending={rename.isPending} disabled={!nextName.trim()}>
                {rename.isPending ? "Saving…" : "Save name"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
      <DeleteGraphDialog
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        graphs={[{ graphId: snapshot.graphId, graphName: snapshot.graphName || snapshot.graphId }]}
        onDeleted={() => onDeleted?.()}
      />
    </div>
  );
}

function guideForRun(options: {
  status: string;
  sentence: string;
  zeroEntities: boolean;
  canRetry: boolean;
  canCancel: boolean;
  onRetry: () => void;
  onCancel: () => void;
  onExport: () => void;
  onNewGraph?: () => Promise<void> | void;
  onCloneConfig?: () => Promise<void> | void;
  onDelete?: () => void;
  retrying?: boolean;
  cancelling?: boolean;
  onOpenQuality?: () => void;
  staleQueue: boolean;
}) {
  if (options.zeroEntities) {
    return (
      <GuideCard
        tone="warning"
        title="Inspect OCR before you share"
        why="The pipeline is green, but entity count is 0. A hairball would hide this."
        actions={[{ label: "Open Quality", onClick: () => options.onOpenQuality?.(), variant: "secondary" }]}
      />
    );
  }
  if (options.status === "failed" || options.status === "interrupted") {
    return (
      <GuideCard
        tone="warning"
        title={options.canRetry ? "Retry failed documents" : "This run did not finish"}
        why={options.canRetry ? options.sentence : "Clone the config, or delete this graph."}
        actions={[
          ...(options.canRetry
            ? [{ label: "Retry failed documents", onClick: options.onRetry, pending: options.retrying }]
            : []),
          { label: "New graph from this config", onClick: () => void options.onCloneConfig?.(), variant: "secondary" as const },
          ...(options.onDelete
            ? [{ label: "Delete this graph", onClick: options.onDelete, variant: "outline" as const }]
            : []),
        ]}
      />
    );
  }
  if (options.status === "cancelled") {
    return (
      <GuideCard
        title={options.canRetry ? "Resume this run where it stopped" : "This run was cancelled"}
        why={
          options.canRetry
            ? "Work that finished before the cancellation is kept; only what was stopped is queued again."
            : "Clone the config, or delete this graph."
        }
        actions={[
          ...(options.canRetry ? [{ label: "Resume run", onClick: options.onRetry, pending: options.retrying }] : []),
          { label: "New graph from this config", onClick: () => void options.onCloneConfig?.(), variant: "secondary" as const },
          ...(options.onDelete
            ? [{ label: "Delete this graph", onClick: options.onDelete, variant: "outline" as const }]
            : []),
        ]}
      />
    );
  }
  if (options.status === "cancelling") {
    return (
      <GuideCard
        title="Cancellation is in progress"
        why={options.sentence}
        actions={[]}
      />
    );
  }
  if (isActiveStatus(options.status) && options.canCancel) {
    return (
      <GuideCard
        title={options.staleQueue ? "This job has not started" : "This job is still moving"}
        why={options.sentence}
        actions={[{ label: "Cancel job", onClick: options.onCancel, variant: "outline", pending: options.cancelling }]}
      />
    );
  }
  return null;
}

function QualityPanel({
  runId,
  graphId,
  report,
  loading,
  error,
  inspectHref,
  onChanged,
}: {
  runId: string;
  graphId: string;
  onChanged: () => Promise<unknown>;
  report: {
    nodeCount: number;
    edgeCount: number;
    documentCount: number;
    documentsFailed: number;
    zeroEntityFailure: boolean;
    goldSource: "uploaded" | "beside-source" | null;
    benchmark: BenchmarkReport | null;
    benchmarkError: string | null;
    baselines: BenchmarkBaseline[];
    gaps: ExtractionGaps;
    rejected: RejectedRecords;
    failed: FailedDocuments;
    gold: {
      goldName: string;
      expectedEntities: number;
      foundEntities: number;
      expectedRelations: number;
      foundRelations: number;
      requiredTotal: number;
      matchedRequired: number;
      missingRequired: Array<{ id: string; source: string; target: string; relationType: string }>;
    } | null;
  } | undefined;
  loading: boolean;
  error?: string;
  inspectHref?: string;
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <p className="text-sm text-muted-foreground">Loading the quality report…</p>
      </div>
    );
  }
  if (error) {
    return <Alert variant="destructive">{error}</Alert>;
  }
  if (!report) {
    return <p className="text-sm text-muted-foreground">No quality report for this run.</p>;
  }
  return (
    <div className="space-y-4">
    <Card>
      <CardHeader>
        <CardTitle>Quality</CardTitle>
        <CardDescription>What the run produced, and whether it clears the bar set for this graph.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p>
          {report.documentCount} {report.documentCount === 1 ? "document" : "documents"} · {report.nodeCount} entities ·{" "}
          {report.edgeCount} relations
          {report.documentsFailed
            ? ` · ${report.documentsFailed.toLocaleString()} ${report.documentsFailed === 1 ? "document" : "documents"} could not be read`
            : ""}
        </p>
        {report.zeroEntityFailure ? (
          <Alert variant="destructive">0 entities after a green pipeline. Treat as OCR or schema failure, not Succeeded.</Alert>
        ) : null}
        {report.gaps.windows ? (
          <Alert variant="warning" data-testid="quality-gaps-note">
            {report.gaps.windows.toLocaleString()} extraction {report.gaps.windows === 1 ? "gap" : "gaps"} in{" "}
            {report.gaps.documents.toLocaleString()} {report.gaps.documents === 1 ? "document" : "documents"}: text the model
            read and nothing was kept from. Listed below.
          </Alert>
        ) : null}
        {inspectHref ? (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" asChild>
              <a href={inspectHref} target="_blank" rel="noreferrer">
                Open inspect HTML
              </a>
            </Button>
            <Button size="sm" variant="ghost" asChild>
              <a href={`${inspectHref}#gold-compare`} target="_blank" rel="noreferrer">
                Compare gold
              </a>
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
    <BenchmarkCard benchmark={report.benchmark} error={report.benchmarkError} baselines={report.baselines} />
    <FailedDocumentsCard failed={report.failed} graphId={graphId} />
    <ExtractionGapsCard gaps={report.gaps} graphId={graphId} />
    <RejectedRecordsCard rejected={report.rejected} graphId={graphId} />
    <GoldCard
      runId={runId}
      graphId={graphId}
      gold={report.gold}
      goldSource={report.goldSource}
      onChanged={onChanged}
    />
    </div>
  );
}

/** Missing gold relations on screen before "Show more"; a gold file can name hundreds. */
function ProgressPanel({
  sentence,
  stages,
  documentsCompleted,
  documentsTotal,
  estimate,
  documents,
  onSkip,
  skipping = null,
}: {
  sentence: string;
  stages: readonly StageProgress[];
  documentsCompleted: number;
  documentsTotal: number | null;
  estimate: { usdLow: number; usdHigh: number } | null;
  documents: DocumentStatus[];
  onSkip?: (fileId: string) => void;
  skipping?: string | null;
}) {
  const kept = documents.filter((item) => item.phase === "inherited").length;
  const attention = documentsNeedingAttention(documents);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Progress</CardTitle>
        <CardDescription data-testid="blocking-sentence">{sentence}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Documents {documentsCompleted}
          {documentsTotal != null ? ` / ${documentsTotal}` : ""}
          {kept > 0 ? ` · ${kept} kept from earlier versions` : ""}
          {estimate ? ` · cost so far in band ${estimate.usdLow}–${estimate.usdHigh} usd · split deserved GPU hours from idle after the run` : ""}
        </p>
        {stages.map((stage) => {
          const percent = stage.total
            ? Math.min(100, Math.round((stage.completed / stage.total) * 100))
            : stage.status === "completed" || stage.status === "succeeded"
              ? 100
              : undefined;
          return (
            <div key={stage.stage} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>{titleCaseStage(stage.stage)}</span>
                <span className="text-muted-foreground">
                  {stage.completed}
                  {stage.total != null ? ` / ${stage.total}` : ""} · {stage.status}
                </span>
              </div>
              <Progress value={percent} />
              {stage.message ? <p className="text-xs text-muted-foreground">{stage.message}</p> : null}
            </div>
          );
        })}
        {documents.length ? (
          <div className="space-y-3 border-t border-border pt-3">
            <PhaseSummary documents={documents} />
            {attention.length ? (
              <div className="space-y-1.5" data-testid="documents-needing-attention">
                <p className="text-sm font-medium text-destructive">
                  {attention.length === 1 ? "One document needs a decision" : `${attention.length} documents need a decision`}
                </p>
                <ul className="space-y-1 text-sm">
                  {attention.slice(0, ATTENTION_PREVIEW).map((item) => (
                    <li key={item.fileId} className="flex items-center justify-between gap-2">
                      <span className="min-w-0 truncate" title={item.fileId}>
                        <span className="font-medium">{item.name ?? item.fileId}</span>
                        <span className="text-muted-foreground">
                          {" "}
                          · {formatDocumentPhase(item.phase)} · {item.detail}
                        </span>
                      </span>
                      {onSkip ? (
                        <Button
                          size="sm"
                          variant="outline"
                          className="shrink-0"
                          pending={skipping === item.fileId}
                          disabled={skipping !== null}
                          onClick={() => onSkip(item.fileId)}
                        >
                          {skipping === item.fileId ? "Skipping…" : "Skip file"}
                        </Button>
                      ) : null}
                    </li>
                  ))}
                </ul>
                {attention.length > ATTENTION_PREVIEW ? (
                  <p className="text-xs text-muted-foreground">
                    {attention.length - ATTENTION_PREVIEW} more in the list below, under Needs attention.
                  </p>
                ) : null}
              </div>
            ) : null}
            <details className="group" data-testid="all-documents">
              <summary className="cursor-pointer select-none text-sm text-muted-foreground hover:text-foreground">
                All {documents.length.toLocaleString()} documents
              </summary>
              <div className="pt-3">
                <DocumentTable documents={documents} onSkip={onSkip} skipping={skipping} />
              </div>
            </details>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

/** Documents needing a decision that the Progress card lists inline; the rest are in the table. */
const ATTENTION_PREVIEW = 5;

/** The failures a retry cleared: what stopped, where, and when it was retried. */
function ResolvedFailures({ raw }: { raw: Record<string, unknown> }) {
  const { failures, total } = resolvedFailures(raw);
  if (!failures.length) {
    return null;
  }
  return (
    <Card data-testid="resolved-failures">
      <CardHeader>
        <CardTitle>Resolved failures</CardTitle>
        <CardDescription>
          {total.toLocaleString()} task failure{total === 1 ? "" : "s"} a retry cleared
          {total > failures.length ? `; the latest ${failures.length} are shown` : ""}.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Stage</TableHead>
              <TableHead>Scope</TableHead>
              <TableHead>Error</TableHead>
              <TableHead className="text-right">Failed</TableHead>
              <TableHead className="text-right">Retried</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {failures.map((failure) => (
              <TableRow key={`${failure.taskId}-${failure.retriedAt}`}>
                <TableCell className="font-medium">{titleCaseStage(failure.stage)}</TableCell>
                <TableCell className="max-w-48 truncate font-mono text-[11px]" title={failure.taskId}>
                  {failure.scopeId}
                </TableCell>
                <TableCell className="max-w-md truncate text-muted-foreground" title={failure.message}>
                  {failure.message}
                  {failure.attempts ? ` (after ${failure.attempts} attempt${failure.attempts === 1 ? "" : "s"})` : ""}
                </TableCell>
                <TableCell className="text-right tabular-nums">{formatInstant(failure.failedAt)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatInstant(failure.retriedAt)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function RunDetails({
  snapshot,
  documents = [],
  gaps,
  onSkip,
  skipping = null,
  consumption,
  estimate = null,
}: {
  snapshot: RunSnapshot;
  documents?: DocumentStatus[];
  /** The graph's extraction gaps, so each document can say how many it carries. */
  gaps?: ExtractionGaps;
  onSkip?: (fileId: string) => void;
  skipping?: string | null;
  /** The run's recorded spend, when the graph is loaded. */
  consumption?: unknown;
  estimate?: ConsumptionEstimate | null;
}) {
  const raw = snapshot.raw as Record<string, unknown>;
  const keptCount = documents.filter((item) => item.phase === "inherited").length;
  const startedAt = Date.parse(snapshot.startedAt ?? "");
  const updatedAt = Date.parse(snapshot.updatedAt ?? "");
  const duration = Number.isFinite(startedAt) && Number.isFinite(updatedAt) ? updatedAt - startedAt : null;
  const facts: Array<[string, string]> = [
    ["Run", snapshot.runId],
    ["Graph", snapshot.graphId],
    ["Status", snapshot.status],
    ["Started", formatInstant(snapshot.startedAt)],
    [isActiveStatus(snapshot.status) ? "Last update" : "Finished", formatInstant(snapshot.updatedAt)],
    [isActiveStatus(snapshot.status) ? "Running for" : "Took", formatDuration(duration)],
    [
      "Documents",
      `${
        snapshot.documentsTotal == null
          ? `${snapshot.documentsCompleted} indexed`
          : `${snapshot.documentsCompleted} of ${snapshot.documentsTotal} indexed${snapshot.documentsFailed ? `, ${snapshot.documentsFailed} failed` : ""}`
      }${keptCount ? ` · ${keptCount} kept from earlier versions` : ""}`,
    ],
    ["Storage", `${snapshot.storageKind}${snapshot.storageLocation ? ` · ${snapshot.storageLocation}` : ""}`],
  ];
  const source = [raw.sourceKind, raw.sourcePath].filter(Boolean).map(String).join(" · ");
  if (source) {
    facts.push(["Source", source]);
  }
  const providers = [
    ["OCR", raw.ocrProvider],
    ["LLM", raw.llmProvider],
    ["Embedding", raw.embeddingProvider],
  ]
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => `${label} ${String(value)}`)
    .join(" · ");
  if (providers) {
    facts.push(["Providers", providers]);
  }
  if (typeof raw.owner === "string" && raw.owner) {
    facts.push(["Owner", raw.owner]);
  }
  return (
    <div className="space-y-4" data-testid="run-details">
      <Card>
        <CardHeader>
          <CardTitle>Run</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
            {facts.map(([label, value]) => (
              <Fragment key={label}>
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="break-all">{value}</dd>
              </Fragment>
            ))}
          </dl>
          {snapshot.warnings.length ? (
            <ul className="mt-4 space-y-1 text-sm text-amber-700 dark:text-amber-400">
              {snapshot.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </CardContent>
      </Card>

      {snapshot.stages.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Stages</CardTitle>
            <CardDescription>Work per stage, and how long the workers spent on it.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Stage</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Worker time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {snapshot.stages.map((stage) => {
                  const percent = stage.total
                    ? Math.min(100, Math.round((stage.completed / stage.total) * 100))
                    : stage.status === "completed" || stage.status === "succeeded"
                      ? 100
                      : 0;
                  return (
                    <TableRow key={stage.stage}>
                      <TableCell className="font-medium">{titleCaseStage(stage.stage)}</TableCell>
                      <TableCell className="min-w-48">
                        <div className="flex items-center gap-3">
                          <Progress value={percent} className="h-2 w-28" />
                          <span className="text-muted-foreground">
                            {stage.completed}
                            {stage.total != null ? ` / ${stage.total}` : ""}
                          </span>
                        </div>
                        {stage.message ? <p className="mt-1 text-xs text-muted-foreground">{stage.message}</p> : null}
                      </TableCell>
                      <TableCell>{stage.status}</TableCell>
                      <TableCell className="text-right tabular-nums">{formatDuration(stage.elapsedMs)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      <ResolvedFailures raw={raw} />

      {documents.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Documents</CardTitle>
            <CardDescription>
              {documents.length.toLocaleString()} document{documents.length === 1 ? "" : "s"} the run discovered, and where each stands.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <PhaseSummary documents={documents} />
            <DocumentTable documents={documents} onSkip={onSkip} skipping={skipping} gaps={gapsByFile(gaps)} />
          </CardContent>
        </Card>
      ) : null}

      {consumption !== undefined ? (
        <Card data-testid="consumption">
          <CardHeader>
            <CardTitle>Consumption</CardTitle>
            <CardDescription>What the run spent on parsing, extraction and summaries, held against the estimate it was priced at.</CardDescription>
          </CardHeader>
          <CardContent>
            <ConsumptionPanel consumption={consumption} estimate={estimate} />
          </CardContent>
        </Card>
      ) : null}
      {snapshot.events.length ? <RecentEvents events={snapshot.events} /> : null}
    </div>
  );
}

/** Events on screen before "Show more"; the snapshot carries up to the last 2,000 of the run's log. */
const EVENT_PAGE_SIZE = 30;

/**
 * The tail of the run's event log, newest first. The heading says how much
 * of the log this is, so a short tail is not mistaken for a quiet run. The
 * total is what the snapshot holds; on a very long run that is itself the
 * log's tail.
 */
function RecentEvents({ events }: { events: RunSnapshot["events"] }) {
  const [limit, setLimit] = useState(EVENT_PAGE_SIZE);
  const newestFirst = useMemo(() => [...events].reverse(), [events]);
  const page = newestFirst.slice(0, limit);
  return (
    <Card data-testid="recent-events">
      <CardHeader>
        <CardTitle>Recent events</CardTitle>
        <CardDescription data-testid="recent-events-count">{tailLabel(page.length, events.length, "events")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <ol className="space-y-2 text-sm">
          {page.map((event, index) => (
            <li key={`${event.timestamp}-${index}`}>
              <span className="text-muted-foreground">{event.timestamp}</span> {event.stage} {event.status}
              {event.fileId ? ` · ${event.fileId}` : ""}
              {event.message ? ` — ${event.message}` : ""}
            </li>
          ))}
        </ol>
        <ShowMore
          shown={page.length}
          total={events.length}
          pageSize={EVENT_PAGE_SIZE}
          noun="events"
          onMore={() => setLimit((current) => current + EVENT_PAGE_SIZE)}
        />
      </CardContent>
    </Card>
  );
}

/** "Version 2 of 3 · head", from what the runtime knows of the graph's versions. */
function versionLine(snapshot: RunSnapshot): string | null {
  const version = snapshot.raw.version;
  if (!version || typeof version !== "object") {
    return null;
  }
  const { number, count, head } = version as { number: number; count: number; head: boolean };
  if (!count) {
    return null;
  }
  if (!number) {
    return `${count} published version${count === 1 ? "" : "s"} of this graph`;
  }
  return `Version ${number} of ${count}${head ? " · head" : ""}`;
}

function watchPrefixFrom(snapshot: RunSnapshot): string {
  const raw = snapshot.raw as Record<string, unknown>;
  const source = raw.source && typeof raw.source === "object" ? (raw.source as Record<string, unknown>) : {};
  return String(raw.sourcePath ?? source.path ?? "");
}

/** Everything a reviewer needs offline: the whole graph, its evidence and the quality report. */
function reviewBundle(snapshot: RunSnapshot, dataset: GraphDataset | undefined, quality: unknown, reviews: readonly HitlItem[] = []) {
  return {
    graphId: snapshot.graphId,
    graphName: snapshot.graphName,
    status: snapshot.status,
    exportedAt: new Date().toISOString(),
    counts: dataset ? graphCounts(dataset) : null,
    nodes: dataset?.nodes ?? [],
    edges: dataset?.edges ?? [],
    evidence: dataset?.evidence ?? [],
    // Every discarded window and every document that could not be read, so a
    // reviewer holding the bundle knows what the graph does not cover without
    // opening the console.
    discardedWindows: dataset?.discardedWindows ?? [],
    rejectedRecords: dataset?.rejectedRecords ?? [],
    failedDocuments: dataset?.failedDocuments ?? [],
    quality,
    reviews,
  };
}

/** Gap counts by document, for the Documents table's chip. */
function gapsByFile(gaps: ExtractionGaps | undefined): ReadonlyMap<string, number> {
  return new Map((gaps?.rows ?? []).map((row) => [row.fileId, row.windows]));
}

/**
 * One place to take the graph away. Each item says what it carries, because
 * the two graph formats follow Explore's filters while the bundle and the
 * consumption record are the whole run.
 */
function ExportMenu({
  snapshot,
  dataset,
  quality,
  focus,
  estimate,
  reviewBundle: offerBundle,
  snowflake,
}: {
  snapshot: RunSnapshot;
  dataset: GraphDataset | undefined;
  quality: unknown;
  focus: ExploreFocus;
  estimate: ConsumptionEstimate | null;
  reviewBundle: boolean;
  /** On a fleet: whether its profile names a Snowflake account, and what to open. Null off the fleet. */
  snowflake: { available: boolean; onPublish: () => void } | null;
}) {
  const graphReady = Boolean(dataset) && focus.nodes.length > 0;
  const scope = focus.summary
    ? `${focus.nodes.length.toLocaleString()} entities · ${focus.edges.length.toLocaleString()} relations, filtered by ${focus.summary}`
    : `${focus.nodes.length.toLocaleString()} entities · ${focus.edges.length.toLocaleString()} relations, the whole graph`;
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button size="sm" variant="outline" className="gap-1">
          Export
          <ChevronDown className="size-3.5" aria-hidden="true" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[18rem]">
        <DropdownMenuLabel>Graph as shown in Explore</DropdownMenuLabel>
        <DropdownMenuItem
          disabled={!graphReady}
          onSelect={() => {
            downloadJson(`${snapshot.graphId}-subgraph.json`, { nodes: focus.nodes, edges: focus.edges });
            toast.success("Exported subgraph");
          }}
        >
          <span>Subgraph as JSON</span>
          <span className="text-xs text-muted-foreground">{scope}</span>
        </DropdownMenuItem>
        <DropdownMenuItem
          disabled={!graphReady}
          onSelect={() => {
            downloadFile(`${snapshot.graphId}.graphml`, buildGraphML([...focus.nodes], [...focus.edges]), "application/graphml+xml");
            toast.success("Exported GraphML");
          }}
        >
          <span>GraphML</span>
          <span className="text-xs text-muted-foreground">Same scope, for Gephi, yEd and NetworkX</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Whole run</DropdownMenuLabel>
        {offerBundle ? (
          <DropdownMenuItem
            disabled={!dataset}
            onSelect={() => {
              downloadJson(`${snapshot.graphId}-review-bundle.json`, reviewBundle(snapshot, dataset, quality));
              toast.success("Downloaded review bundle");
            }}
          >
            <span>Review bundle</span>
            <span className="text-xs text-muted-foreground">Every entity, relation and quote plus the quality report</span>
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuItem
          disabled={!dataset?.graphMetrics.consumption}
          onSelect={() => {
            downloadJson(`${snapshot.graphId}-consumption.json`, consumptionExport(dataset?.graphMetrics.consumption, estimate));
            toast.success("Exported consumption");
          }}
        >
          <span>Consumption</span>
          <span className="text-xs text-muted-foreground">Tokens, pages and cost per call, beside the estimate</span>
        </DropdownMenuItem>
        {snowflake ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuLabel>Publish</DropdownMenuLabel>
            <DropdownMenuItem disabled={!snowflake.available || !dataset} onSelect={snowflake.onPublish}>
              <span>Snowflake…</span>
              <span className="text-xs text-muted-foreground">
                {snowflake.available
                  ? "Bulk-load this graph into a schema of the fleet's account, no re-run"
                  : "The fleet's profile names no Snowflake account"}
              </span>
            </DropdownMenuItem>
          </>
        ) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

/** "49 docs · 6,526 entities · 847 relations · 11,871 evidence", where the graph is named. */
function graphCountsLine(dataset: GraphDataset): string {
  const counts = graphCounts(dataset);
  return `${counts.documents.toLocaleString()} docs · ${counts.nodes.toLocaleString()} entities · ${counts.edges.toLocaleString()} relations · ${counts.evidence.toLocaleString()} evidence`;
}

function isStaleQueue(snapshot: RunSnapshot): boolean {
  const status = snapshot.status.toLowerCase();
  if (status !== "queued" && status !== "pending" && status !== "planning") {
    return false;
  }
  const stamp = Date.parse(snapshot.startedAt || snapshot.updatedAt || "");
  return Number.isFinite(stamp) && Date.now() - stamp > 60 * 60 * 1000;
}
