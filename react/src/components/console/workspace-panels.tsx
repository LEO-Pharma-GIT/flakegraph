"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Alert } from "@/components/ui/alert";
import { PageHeader } from "@/components/console/page-header";
import { ShowMore } from "@/components/console/show-more";
import { useStorageItem, writeStorage } from "@/lib/browser-storage";
import { orderVersions } from "@/lib/paging";
import { AskPanel } from "@/components/console/ask-panel";
import type { HitlItem } from "@/server/workspace";

export { AskPanel };

/** Workspace versions on screen before "Show more"; each publish and each watch ingest adds one. */
const VERSION_PAGE_SIZE = 10;

export function VersionsPanel({ graphId, canPublish = true }: { graphId: string; canPublish?: boolean }) {
  const workspace = trpc.workspace.get.useQuery();
  const publish = trpc.workspace.publish.useMutation({
    onSuccess: async () => {
      toast.success("Published a new graph version. The previous production version stays readable.");
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const [limit, setLimit] = useState(VERSION_PAGE_SIZE);
  // The production version is this graph's head; it reads first, then the
  // rest newest first. The workspace appends versions as they are recorded,
  // so position is the version number.
  const versions = orderVersions(
    (workspace.data?.versions ?? [])
      .map((item, index) => ({ ...item, number: index + 1, head: item.lifecycle === "production", runId: item.id }))
      .filter((item) => item.graphId === graphId),
  );
  const page = versions.slice(0, limit);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>Atomic head move. Older versions stay readable.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {canPublish ? (
          <Button size="sm" pending={publish.isPending} onClick={() => publish.mutate({ graphId, note: "Published from Versions" })}>
            {publish.isPending ? "Publishing…" : "Publish new version"}
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">Publishing a version is a builder action.</p>
        )}
        {versions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No published versions yet.</p>
        ) : (
          <>
            <ul className="space-y-2 text-sm" data-testid="workspace-versions">
              {page.map((item) => (
                <li key={item.id}>
                  <span className="font-medium">{item.label}</span> · {item.lifecycle} · {item.note}
                </li>
              ))}
            </ul>
            <ShowMore
              shown={page.length}
              total={versions.length}
              pageSize={VERSION_PAGE_SIZE}
              noun="versions"
              onMore={() => setLimit((current) => current + VERSION_PAGE_SIZE)}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function ReviewPanel({
  graphId,
  runId,
  canExport = true,
  canReview = true,
  relationTypes = [],
  entityNames = [],
}: {
  graphId: string;
  runId: string;
  canExport?: boolean;
  /** Whether the viewer may sample and decide; read access only looks. */
  canReview?: boolean;
  /** The graph's relation vocabulary, for a relabel to choose from. */
  relationTypes?: readonly string[];
  /** The graph's entity names, for a merge to point at. */
  entityNames?: readonly string[];
}) {
  const workspace = trpc.workspace.get.useQuery();
  const [ceiling, setCeiling] = useState(1);
  const decide = trpc.workspace.review.useMutation({
    onSuccess: async () => {
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  // The card whose decision is being saved; the others stay usable.
  const deciding = decide.isPending ? (decide.variables?.id ?? null) : null;
  const sample = trpc.graphs.sampleReviews.useMutation({
    onSuccess: async () => {
      toast.success("Sampled 1% of high-confidence edges.");
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const items = (workspace.data?.reviews ?? []).filter((item) => item.graphId === graphId);
  const pins = (workspace.data?.pins ?? []).filter((item) => item.graphId === graphId);
  const visible = items.filter((item) => (item.confidence ?? 0) <= ceiling);
  const [reviewLimit, setReviewLimit] = useState(REVIEW_PAGE_SIZE);
  const page = visible.slice(0, reviewLimit);
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Each card is a triple a person should look at. Keep or Drop it, Relabel it with the relation it should carry,
        or mark one of its entities as a duplicate to Merge. Decisions are recorded on this queue and go out with the
        review bundle; they never rewrite the stored graph.
        {canExport ? "" : " Snowflake has no export on this page, so they stay on this queue."}
        {canReview ? "" : " You have read access, so the queue is shown as it stands."}
      </p>
      <div className="flex flex-wrap items-center gap-3">
        {canReview ? (
          <Button size="sm" variant="outline" pending={sample.isPending} onClick={() => sample.mutate({ runId })}>
            {sample.isPending ? "Sampling…" : "Sample 1% high-confidence"}
          </Button>
        ) : null}
        <label className="grid min-w-56 gap-1 text-sm">
          <span>Queue ceiling ({ceiling.toFixed(2)})</span>
          <Slider aria-label="Review confidence ceiling" value={[ceiling]} max={1} step={0.05} onValueChange={(value) => setCeiling(value[0] ?? 1)} />
        </label>
      </div>
      {pins.length ? (
        <p className="text-sm text-muted-foreground">{pins.length} looks-wrong pins. Pins are not deletes.</p>
      ) : null}
      {visible.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {items.length === 0
            ? "Nothing is queued for review yet. Sample high-confidence edges here, or flag an edge that looks wrong in Explore."
            : `All ${items.length} queued triples sit above the ${ceiling.toFixed(2)} ceiling.`}
        </p>
      ) : null}
      {page.map((item) => (
        <ReviewCard
          key={item.id}
          item={item}
          relationTypes={relationTypes}
          entityNames={entityNames}
          pending={deciding === item.id}
          onDecide={canReview ? (decision) => decide.mutate({ id: item.id, decision }) : undefined}
        />
      ))}
      {visible.length > page.length ? (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            Showing {page.length} of {visible.length.toLocaleString()} queued triples
          </span>
          <Button size="sm" variant="outline" className="h-7" onClick={() => setReviewLimit((current) => current + REVIEW_PAGE_SIZE)}>
            Show {Math.min(REVIEW_PAGE_SIZE, visible.length - page.length)} more
          </Button>
        </div>
      ) : null}
    </div>
  );
}

/** Review cards shown before "Show more"; a sampled queue can hold hundreds. */
const REVIEW_PAGE_SIZE = 25;

/** Entity names offered to a merge; the list is a datalist, not a page. */
const MERGE_SUGGESTIONS = 200;

function describeDecision(decision: NonNullable<HitlItem["decision"]>): string {
  switch (decision.kind) {
    case "keep":
      return "Recorded keep.";
    case "drop":
      return "Recorded drop.";
    case "relabel":
      return `Recorded relabel to ${decision.relationType}.`;
    case "merge":
      return `Recorded merge: the ${decision.end} is a duplicate of ${decision.into}.`;
  }
}

function ReviewCard({
  item,
  relationTypes,
  entityNames,
  pending,
  onDecide,
}: {
  item: HitlItem;
  relationTypes: readonly string[];
  entityNames: readonly string[];
  pending: boolean;
  /** Absent for a viewer who may look but not decide. */
  onDecide?: (decision: NonNullable<HitlItem["decision"]>) => void;
}) {
  const [editing, setEditing] = useState<"relabel" | "merge" | null>(null);
  const [relationType, setRelationType] = useState(
    item.decision?.kind === "relabel" ? item.decision.relationType : (item.triple?.relation ?? ""),
  );
  const [end, setEnd] = useState<"source" | "target">(item.decision?.kind === "merge" ? item.decision.end : "target");
  const [into, setInto] = useState(item.decision?.kind === "merge" ? item.decision.into : "");
  const kind = item.decision?.kind ?? null;
  const listId = `merge-into-${item.id}`;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {item.triple ? (
            <>
              {item.triple.source} <span className="font-normal text-muted-foreground">—{item.triple.relation}→</span>{" "}
              {item.triple.target}
            </>
          ) : (
            item.edgeId
          )}
        </CardTitle>
        <CardDescription>
          {item.quote || "Low-confidence triple"} · confidence {(item.confidence ?? 0).toFixed(2)}
          {item.triple ? ` · ${item.edgeId}` : ""}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {onDecide ? (
          <>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant={kind === "keep" ? "default" : "outline"} disabled={pending} onClick={() => onDecide({ kind: "keep" })}>
              Keep
            </Button>
            <Button size="sm" variant={kind === "drop" ? "default" : "outline"} disabled={pending} onClick={() => onDecide({ kind: "drop" })}>
              Drop
            </Button>
            <Button
              size="sm"
              variant={kind === "relabel" ? "default" : "outline"}
              aria-expanded={editing === "relabel"}
              onClick={() => setEditing((current) => (current === "relabel" ? null : "relabel"))}
            >
              Relabel…
            </Button>
            <Button
              size="sm"
              variant={kind === "merge" ? "default" : "outline"}
              aria-expanded={editing === "merge"}
              onClick={() => setEditing((current) => (current === "merge" ? null : "merge"))}
            >
              Merge…
            </Button>
          </div>
          {editing === "relabel" ? (
            <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/30 p-2 text-sm" data-testid="relabel-editor">
              <span className="text-muted-foreground">Relation should be</span>
              <Input
                aria-label="New relation type"
                list={`relation-types-${item.id}`}
                className="h-8 w-56 font-mono text-xs"
                placeholder="e.g. STUDIED_UNDER"
                value={relationType}
                onChange={(event) => setRelationType(event.target.value.toUpperCase().replace(/[\s-]+/g, "_"))}
              />
              <datalist id={`relation-types-${item.id}`}>
                {relationTypes.map((type) => (
                  <option key={type} value={type} />
                ))}
              </datalist>
              <Button
                size="sm"
                disabled={pending || !relationType.trim() || relationType.trim() === item.triple?.relation}
                onClick={() => {
                  onDecide({ kind: "relabel", relationType: relationType.trim() });
                  setEditing(null);
                }}
              >
                Record relabel
              </Button>
            </div>
          ) : null}
          {editing === "merge" ? (
            <div className="flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/30 p-2 text-sm" data-testid="merge-editor">
              <select
                aria-label="Which entity is the duplicate"
                className="h-8 rounded-md border border-border bg-background px-2 text-sm"
                value={end}
                onChange={(event) => setEnd(event.target.value as "source" | "target")}
              >
                <option value="source">{item.triple?.source ?? "Source"}</option>
                <option value="target">{item.triple?.target ?? "Target"}</option>
              </select>
              <span className="text-muted-foreground">is the same entity as</span>
              <Input
                aria-label="Merge into"
                list={listId}
                className="h-8 w-56"
                placeholder="Entity name"
                value={into}
                onChange={(event) => setInto(event.target.value)}
              />
              <datalist id={listId}>
                {entityNames.slice(0, MERGE_SUGGESTIONS).map((name) => (
                  <option key={name} value={name} />
                ))}
              </datalist>
              <Button
                size="sm"
                disabled={pending || !into.trim()}
                onClick={() => {
                  onDecide({ kind: "merge", end, into: into.trim() });
                  setEditing(null);
                }}
              >
                Record merge
              </Button>
            </div>
          ) : null}
          </>
        ) : null}
        {pending ? (
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Spinner />
            Saving decision…
          </p>
        ) : item.decision ? (
          <p className="text-xs text-muted-foreground" data-testid="review-decision">
            {describeDecision(item.decision)} The live graph is unchanged.
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function WatchPanel({
  graphId,
  sourcePath = "",
  onProcessNow,
}: {
  graphId: string;
  sourcePath?: string;
  onProcessNow?: () => void;
}) {
  const workspace = trpc.workspace.get.useQuery();
  const [prefix, setPrefix] = useState(sourcePath);
  // The field follows the graph's own source until it is edited.
  const [prefixFrom, setPrefixFrom] = useState(sourcePath);
  if (prefixFrom !== sourcePath) {
    setPrefixFrom(sourcePath);
    setPrefix(sourcePath);
  }
  const failed = (error: { message: string }) => toast.error(error.message);
  const toggle = trpc.workspace.toggleWatch.useMutation({
    onSuccess: async (_state, variables) => {
      const watch = (workspace.data?.watches ?? []).find((item) => item.id === variables.id);
      toast.success(watch?.paused ? "Watch resumed" : "Watch paused");
      await workspace.refetch();
    },
    onError: failed,
  });
  const apply = trpc.workspace.applyWatch.useMutation({
    onSuccess: async (state) => {
      const latest = [...state.versions].reverse().find((item) => item.graphId === graphId);
      toast.success(
        latest
          ? `Recorded ${latest.label} from new checksums. Workers are not started until you submit.`
          : "Recorded an incremental version. Workers are not started until you submit.",
      );
      await workspace.refetch();
    },
    onError: failed,
  });
  const create = trpc.workspace.createWatch.useMutation({
    onSuccess: async () => {
      toast.success("Watching prefix for new checksums.");
      await workspace.refetch();
    },
    onError: failed,
  });
  const watches = (workspace.data?.watches ?? []).filter((item) => item.graphId === graphId);
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Watch new files</CardTitle>
          <CardDescription>
            Schedule is a graph property. Pause anytime. Ingest records a new version; it does not start workers until you
            process the files.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Input
            aria-label="Watch prefix"
            value={prefix}
            onChange={(event) => setPrefix(event.target.value)}
            placeholder="Path or stage prefix workers can LIST"
          />
          <Button
            size="sm"
            pending={create.isPending}
            disabled={!prefix.trim()}
            onClick={() => create.mutate({ graphId, prefix: prefix.trim() })}
          >
            {create.isPending ? "Adding watch…" : "Watch this prefix"}
          </Button>
        </CardContent>
      </Card>
      {watches.map((watch) => (
        <Card key={watch.id}>
          <CardHeader>
            <CardTitle className="text-base">{watch.prefix}</CardTitle>
            <CardDescription>
              Last window {watch.lastWindow}. {watch.pendingFiles} new files. Gold drift {watch.goldDrift}.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              pending={toggle.isPending && toggle.variables?.id === watch.id}
              onClick={() => toggle.mutate({ id: watch.id })}
            >
              {watch.paused ? "Resume watch" : "Pause watch"}
            </Button>
            <Button
              size="sm"
              pending={apply.isPending && apply.variables?.id === watch.id}
              disabled={watch.paused || watch.pendingFiles === 0}
              onClick={() => apply.mutate({ id: watch.id })}
            >
              {apply.isPending && apply.variables?.id === watch.id ? "Ingesting…" : "Ingest new files"}
            </Button>
            {onProcessNow ? (
              <Button size="sm" variant="outline" onClick={onProcessNow}>
                Process on compose
              </Button>
            ) : null}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}


export function AnalystHome({
  onOpenRun,
}: {
  onOpenRun: (runId: string) => void;
}) {
  const workspace = trpc.workspace.get.useQuery();
  const runs = trpc.runs.list.useQuery({ limit: 100 });
  const [limit, setLimit] = useState(PERSPECTIVE_PAGE_SIZE);
  const perspectives = (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production");
  const page = perspectives.slice(0, limit);
  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Analyst"
        title="Perspectives"
        description="Published production views. Builders should only publish a perspective after gold is green."
      />
      {perspectives.length === 0 ? (
        <p className="text-sm text-muted-foreground">No production perspectives yet. Ask a builder to publish one after gold.</p>
      ) : null}
      <div className="grid items-start gap-3 md:grid-cols-2">
        {page.map((item) => {
          const run = (runs.data ?? []).find((row) => row.graphId === item.graphId);
          return (
            <Card key={item.id}>
              <CardHeader>
                <CardTitle>{item.name}</CardTitle>
                <CardDescription>{item.suggestedQuestions[0] ?? "Open this neighborhood"}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  disabled={!run}
                  onClick={() => run && onOpenRun(run.runId)}
                >
                  Open {item.name}
                </Button>
                {!run ? (
                  <p className="text-sm text-muted-foreground">No succeeded run is on this catalog for that perspective yet.</p>
                ) : null}
              </CardContent>
            </Card>
          );
        })}
      </div>
      <ShowMore
        shown={page.length}
        total={perspectives.length}
        pageSize={PERSPECTIVE_PAGE_SIZE}
        noun="perspectives"
        onMore={() => setLimit((current) => current + PERSPECTIVE_PAGE_SIZE)}
      />
    </div>
  );
}

/** Perspective cards on the analyst home before "Show more"; every production graph can publish one or more. */
const PERSPECTIVE_PAGE_SIZE = 24;

export function WelcomeBack({
  identified,
  suggestionMode,
  currentRunId = null,
  analyst = false,
  onOpenRun,
}: {
  identified: boolean;
  suggestionMode: "off" | "on-request" | "auto-fill";
  currentRunId?: string | null;
  analyst?: boolean;
  onOpenRun: (runId: string) => void;
}) {
  const runs = trpc.runs.list.useQuery({ limit: 40 });
  const workspace = trpc.workspace.get.useQuery();
  const dismissed = useStorageItem("session", "flakegraph.welcome-dismissed") === "1";
  if (!identified || suggestionMode === "off" || dismissed) {
    return null;
  }
  const skipName = /forget|rename|bulk|copied|disposable|cancelling|active ocr|fleet martial|llm timeout|queued snowflake|empty share/i;
  const productionIds = new Set(
    (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production").map((item) => item.graphId),
  );
  const catalog = (runs.data ?? []).filter((run) => (analyst ? productionIds.has(run.graphId) : true));
  const active = catalog.find(
    (run) =>
      ["queued", "running", "pending", "cancelling"].includes(run.status.toLowerCase()) &&
      !skipName.test(`${run.graphName ?? ""} ${run.graphId}`),
  );
  const finished = catalog.find(
    (run) => run.status.toLowerCase() === "succeeded" && !skipName.test(`${run.graphName ?? ""} ${run.graphId}`),
  );
  const suggested = active ?? finished;
  if (!suggested || suggested.runId === currentRunId) {
    return null;
  }
  return (
    <Alert className="mb-4" data-testid="welcome-back">
      <p className="text-sm">
        Welcome back.
        {active ? ` ${active.graphName || active.graphId} is still ${active.status}.` : ""}
        {!active && finished ? ` ${finished.graphName || finished.graphId} finished and is ready to explore.` : ""}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {active ? (
          <Button size="sm" onClick={() => onOpenRun(active.runId)}>
            Resume {active.graphName || active.graphId}
          </Button>
        ) : (
          <Button size="sm" onClick={() => onOpenRun(finished!.runId)}>
            Open {finished?.graphName || finished?.graphId}
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            writeStorage("session", "flakegraph.welcome-dismissed", "1");
          }}
        >
          Dismiss
        </Button>
      </div>
    </Alert>
  );
}
