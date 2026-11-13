"use client";

import { useMemo, useState } from "react";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PageHeader } from "@/components/console/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DEFAULT_WORKLOAD_FILTER,
  FLEET_PAGE_SIZE,
  NODE_ASSIGNMENT_LIMIT,
  NODE_TABLE_THRESHOLD,
  familyCounts,
  filterWorkloads,
  fleetSummary,
  nodeSummaries,
  phaseCounts,
  phaseKind,
  workloadDetail,
  workloadNeedsAttention,
  type FamilyFilter,
  type NodeSummary,
  type PhaseFilter,
  type WorkloadFilter,
} from "@/lib/fleet-workloads";
import type { ClusterSnapshot, WorkloadStatus } from "@/server/protocol/schema";
import { cn } from "@/lib/utils";

export function FleetView() {
  const cluster = trpc.fleet.cluster.useQuery(undefined, { refetchInterval: 15_000 });
  // A refresh someone asked for, as opposed to the quiet poll behind it.
  const [refreshing, setRefreshing] = useState(false);
  const session = trpc.auth.session.useQuery();
  const selected = cluster.data;
  const grafanaUrl = session.data?.grafanaUrl ?? null;
  const [filter, setFilter] = useState<WorkloadFilter>(DEFAULT_WORKLOAD_FILTER);

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Infrastructure"
        title="Compute fleet"
        description="Registered Kubernetes nodes, colocated model servers, and FlakeGraph workers."
        actions={
          <>
            {grafanaUrl ? (
              <Button size="sm" variant="outline" asChild>
                <a href={grafanaUrl} target="_blank" rel="noreferrer">
                  Open Grafana
                </a>
              </Button>
            ) : null}
            <Button
              size="sm"
              variant="outline"
              pending={refreshing}
              onClick={() => {
                setRefreshing(true);
                void cluster.refetch().finally(() => setRefreshing(false));
              }}
            >
              {refreshing ? "Refreshing…" : "Refresh"}
            </Button>
          </>
        }
      />
      {cluster.isLoading && !selected ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-36 w-full" />
        </div>
      ) : cluster.error ? (
        <Alert variant="destructive">{cluster.error.message}</Alert>
      ) : !selected ? (
        <Alert>Connect kubectl to a Kubernetes cluster to view its registered fleet.</Alert>
      ) : (
        <>
          <p className="text-sm text-muted-foreground" data-testid="fleet-status-sentence">
            Context: {selected.context} · Namespace: {selected.namespace}
          </p>
          {selected.warnings.map((warning) => (
            <Alert key={warning}>{warning}</Alert>
          ))}
          <FleetSummaryStrip snapshot={selected} />
          <NodeOverview
            snapshot={selected}
            selectedNode={filter.node}
            onSelectNode={(node) => setFilter((current) => ({ ...current, node: current.node === node ? null : node }))}
          />
          <WorkloadTable snapshot={selected} filter={filter} onFilterChange={setFilter} />
        </>
      )}
    </div>
  );
}

function FleetSummaryStrip({ snapshot }: { snapshot: ClusterSnapshot }) {
  const summary = useMemo(() => fleetSummary(snapshot), [snapshot]);
  const attention = summary.pending + summary.failed + summary.notReady;
  const tiles: Array<{ label: string; value: string; hint: string; alarm?: boolean }> = [
    {
      label: "Nodes ready",
      value: `${summary.nodesReady} / ${summary.nodesTotal}`,
      hint: summary.nodesReady < summary.nodesTotal ? `${summary.nodesTotal - summary.nodesReady} not ready` : "all ready",
      alarm: summary.nodesReady < summary.nodesTotal,
    },
    {
      label: "GPUs",
      value: summary.gpus.toLocaleString(),
      hint: `${summary.modelServersReady} model server${summary.modelServersReady === 1 ? "" : "s"} ready`,
    },
    {
      label: "Workers running",
      value: summary.workersRunning.toLocaleString(),
      hint: `${summary.active.toLocaleString()} active pod${summary.active === 1 ? "" : "s"}`,
    },
    {
      label: "Pending / failed",
      value: `${summary.pending} / ${summary.failed}`,
      hint:
        attention === 0
          ? summary.completed > 0
            ? `${summary.completed} completed`
            : "nothing waiting"
          : summary.notReady > 0
            ? `${summary.notReady} running but not ready`
            : "needs attention",
      alarm: attention > 0,
    },
  ];
  return (
    <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Fleet summary" data-testid="fleet-summary">
      {tiles.map((tile) => (
        <div key={tile.label} className="rounded-lg border border-border bg-card px-4 py-3 shadow-sm">
          <dt className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">{tile.label}</dt>
          <dd className="mt-1 flex flex-wrap items-baseline gap-x-2">
            <span className={cn("text-2xl font-semibold tabular-nums tracking-tight", tile.alarm ? "text-destructive" : undefined)}>
              {tile.value}
            </span>
            <span className="text-xs text-muted-foreground">{tile.hint}</span>
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The nodes as cards while they fit on a screen, as rows once they do not.
 * Picking a node narrows the workload list to it; picking it again clears.
 */
function NodeOverview({
  snapshot,
  selectedNode,
  onSelectNode,
}: {
  snapshot: ClusterSnapshot;
  selectedNode: string | null;
  onSelectNode: (node: string) => void;
}) {
  const rows = useMemo(() => nodeSummaries(snapshot), [snapshot]);
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No nodes are registered in this cluster.</p>;
  }
  if (rows.length > NODE_TABLE_THRESHOLD) {
    return (
      <Card data-testid="fleet-node-table">
        <CardHeader>
          <CardTitle>Nodes ({rows.length})</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Node</TableHead>
                <TableHead>Class</TableHead>
                <TableHead>GPU</TableHead>
                <TableHead className="text-right">Workers</TableHead>
                <TableHead className="text-right">Workloads</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Ready</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow key={row.node.name} data-selected={selectedNode === row.node.name ? "true" : undefined}>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{row.node.name}</span>
                      <NodeSelectButton row={row} selected={selectedNode === row.node.name} onSelect={onSelectNode} />
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{row.node.nodeClass}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {row.node.gpuCount}
                    {row.node.gpuModel ? ` · ${row.node.gpuModel}` : ""}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{row.workersRunning}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {row.activeWorkloads}
                    {row.attention > 0 ? <span className="text-destructive"> · {row.attention} attention</span> : null}
                  </TableCell>
                  <TableCell className="max-w-xs truncate text-muted-foreground" title={row.node.model ?? undefined}>
                    {row.node.model || "none"}
                  </TableCell>
                  <TableCell>
                    <ReadyBadge ready={row.node.ready} />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    );
  }
  return (
    <div className="grid gap-3 sm:grid-cols-[repeat(auto-fit,minmax(16rem,1fr))]" data-testid="fleet-node-grid">
      {rows.map((row) => {
        const selected = selectedNode === row.node.name;
        return (
          <Card key={row.node.name} className={cn(selected ? "border-primary" : undefined)} data-selected={selected ? "true" : undefined}>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between gap-2">
                <span className="truncate">{row.node.name}</span>
                <ReadyBadge ready={row.node.ready} />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-sm text-muted-foreground">
              <p>
                {row.node.nodeClass} · {row.node.gpuCount} GPU
                {row.node.gpuModel ? ` · ${row.node.gpuModel}` : ""}
              </p>
              <p>
                {row.activeWorkloads === 0
                  ? "Idle · no workloads on this node"
                  : `${row.activeWorkloads} workload${row.activeWorkloads === 1 ? "" : "s"} · ${row.workersRunning} worker${row.workersRunning === 1 ? "" : "s"} running`}
                {row.attention > 0 ? (
                  <span className="text-destructive">
                    {" "}
                    · {row.attention} need{row.attention === 1 ? "s" : ""} attention
                  </span>
                ) : null}
              </p>
              <p className="truncate" title={row.node.model ?? undefined}>
                Model {row.node.model || "none"}
              </p>
              <NodeAssignments nodeName={row.node.name} />
              <div className="pt-1">
                <NodeSelectButton row={row} selected={selected} onSelect={onSelectNode} />
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function NodeSelectButton({ row, selected, onSelect }: { row: NodeSummary; selected: boolean; onSelect: (node: string) => void }) {
  return (
    <Button
      size="sm"
      variant={selected ? "secondary" : "ghost"}
      className="h-7 px-2 text-xs"
      aria-pressed={selected}
      aria-label={`Show workloads on ${row.node.name}`}
      onClick={() => onSelect(row.node.name)}
    >
      {selected ? "Showing workloads" : "Show workloads"}
    </Button>
  );
}

function ReadyBadge({ ready }: { ready: boolean }) {
  return <Badge variant={ready ? "success" : "danger"}>{ready ? "ready" : "not ready"}</Badge>;
}

/**
 * Every pod in the namespace, kept readable at hundreds of rows the way the
 * document list is: search by name or node, a component and a phase filter
 * with counts, problems sorted first, and a page at a time.
 */
function WorkloadTable({
  snapshot,
  filter,
  onFilterChange,
}: {
  snapshot: ClusterSnapshot;
  filter: WorkloadFilter;
  onFilterChange: (next: WorkloadFilter) => void;
}) {
  const [limit, setLimit] = useState(FLEET_PAGE_SIZE);
  const workloads = snapshot.workloads;
  const families = useMemo(() => familyCounts(workloads), [workloads]);
  const phases = useMemo(() => phaseCounts(workloads), [workloads]);
  const shown = useMemo(() => filterWorkloads(workloads, filter), [workloads, filter]);
  const page = shown.slice(0, limit);
  const completed = phases.find((item) => item.id === "succeeded")?.count ?? 0;
  const active = workloads.length - completed;

  const update = (patch: Partial<WorkloadFilter>) => {
    onFilterChange({ ...filter, ...patch });
    setLimit(FLEET_PAGE_SIZE);
  };
  const filtering =
    filter.search.trim().length > 0 || filter.family !== "all" || filter.phase !== DEFAULT_WORKLOAD_FILTER.phase || filter.node !== null;

  return (
    <Card data-testid="fleet-workloads">
      <CardHeader>
        <CardTitle>Workloads</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {workloads.length === 0 ? (
          <p className="text-sm text-muted-foreground">No workloads are running in this namespace.</p>
        ) : (
          <>
            <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
              <Input
                aria-label="Search workloads"
                className="h-8 sm:max-w-xs"
                placeholder="Search by name or node"
                value={filter.search}
                onChange={(event) => update({ search: event.target.value })}
              />
              <Select value={filter.family} onValueChange={(value) => update({ family: value as FamilyFilter })}>
                <SelectTrigger aria-label="Component filter" className="h-8 sm:w-52">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All components · {workloads.length.toLocaleString()}</SelectItem>
                  {families.map((family) => (
                    <SelectItem key={family.id} value={family.id}>
                      {family.label} · {family.count.toLocaleString()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={filter.phase} onValueChange={(value) => update({ phase: value as PhaseFilter })}>
                <SelectTrigger aria-label="Phase filter" className="h-8 sm:w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="active">Active · {active.toLocaleString()}</SelectItem>
                  <SelectItem value="all">All phases · {workloads.length.toLocaleString()}</SelectItem>
                  {phases.map((phase) => (
                    <SelectItem key={phase.id} value={phase.id}>
                      {phase.label} · {phase.count.toLocaleString()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {filter.node ? (
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-8 text-xs"
                  aria-label={`Clear node filter ${filter.node}`}
                  onClick={() => update({ node: null })}
                >
                  Node {filter.node} · clear
                </Button>
              ) : null}
              <p className="text-xs text-muted-foreground sm:ml-auto" data-testid="fleet-workloads-count">
                {filtering
                  ? `${shown.length.toLocaleString()} of ${workloads.length.toLocaleString()} workloads`
                  : completed > 0
                    ? `${active.toLocaleString()} active of ${workloads.length.toLocaleString()} workloads`
                    : `${workloads.length.toLocaleString()} workloads`}
              </p>
            </div>
            {shown.length === 0 ? (
              <p className="text-sm text-muted-foreground">No workload matches this filter.</p>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Component</TableHead>
                      <TableHead className="w-28">Phase</TableHead>
                      <TableHead>Node</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {page.map((workload) => (
                      <WorkloadRow key={workload.name} workload={workload} snapshot={snapshot} />
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
            {shown.length > page.length ? (
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>
                  Showing {page.length.toLocaleString()} of {shown.length.toLocaleString()}
                </span>
                <Button size="sm" variant="outline" className="h-7" onClick={() => setLimit((current) => current + FLEET_PAGE_SIZE)}>
                  Show {Math.min(FLEET_PAGE_SIZE, shown.length - page.length)} more
                </Button>
              </div>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function WorkloadRow({ workload, snapshot }: { workload: WorkloadStatus; snapshot: ClusterSnapshot }) {
  const kind = phaseKind(workload.phase);
  const attention = workloadNeedsAttention(workload);
  const finished = kind === "succeeded";
  const detail = workloadDetail(workload, snapshot.nodesReady, snapshot.nodesTotal);
  return (
    <TableRow data-phase={kind} className={finished ? "text-muted-foreground" : undefined}>
      <TableCell className={cn("max-w-xs truncate", finished ? undefined : "font-medium")} title={workload.name}>
        {workload.name}
      </TableCell>
      <TableCell className="whitespace-nowrap text-muted-foreground">{workload.component}</TableCell>
      <TableCell className={cn("whitespace-nowrap", attention ? "text-destructive" : undefined)}>{workload.phase}</TableCell>
      <TableCell className="whitespace-nowrap text-muted-foreground">{workload.node ?? "—"}</TableCell>
      <TableCell className={cn("max-w-md truncate", attention ? "text-destructive" : "text-muted-foreground")} title={detail}>
        {detail}
      </TableCell>
    </TableRow>
  );
}

function NodeAssignments({ nodeName }: { nodeName: string }) {
  const assignments = trpc.fleet.nodeAssignments.useQuery({ nodeName });
  const items = assignments.data ?? [];
  if (items.length === 0) {
    return null;
  }
  const shown = items.slice(0, NODE_ASSIGNMENT_LIMIT);
  return (
    <div className="pt-2 text-xs" data-testid={`node-assignments-${nodeName}`}>
      <p className="font-medium text-foreground">
        {items.length} leased task{items.length === 1 ? "" : "s"}
      </p>
      <ul>
        {shown.map((item) => (
          <li key={item.taskId} className="truncate">
            {item.stage} {item.scopeId} ({item.runId})
          </li>
        ))}
      </ul>
      {items.length > shown.length ? <p>{items.length - shown.length} more</p> : null}
    </div>
  );
}
