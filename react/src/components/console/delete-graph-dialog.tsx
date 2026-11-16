"use client";

import { AlertTriangle } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { displayPrincipal } from "@/components/console/graph-sharing";

export interface DeletableGraph {
  graphId: string;
  graphName: string;
}

/** Names listed in a bulk confirmation before "and N more". */
const NAME_LIMIT = 8;

function plural(count: number, one: string, many: string = `${one}s`): string {
  return `${count.toLocaleString()} ${count === 1 ? one : many}`;
}

/**
 * The one confirmation before graphs are deleted for good, for one graph or
 * many. For a single graph it says what goes - every version and attempt,
 * the upload folders only it used, and who it is shared with - and holds the
 * delete while a run is still under way. Graphs are deleted one after
 * another with the count shown; any the server refuses are named with its
 * reason and the rest still go.
 */
export function DeleteGraphDialog({
  open,
  onOpenChange,
  graphs,
  onDeleted,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  graphs: readonly DeletableGraph[];
  /** The graphs that were deleted, once the dialog is done with them. */
  onDeleted?: (graphIds: string[]) => void | Promise<void>;
}) {
  const utils = trpc.useUtils();
  const single = graphs.length === 1 ? graphs[0]! : null;
  const preview = trpc.graphs.deletion.useQuery(
    { graphId: single?.graphId ?? "" },
    { enabled: open && single !== null, retry: false },
  );
  // Each failure is named in the dialog, so the mutation stays quiet.
  const remove = trpc.graphs.delete.useMutation({ onError: () => undefined });
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [failures, setFailures] = useState<Array<{ graphName: string; message: string }>>([]);
  const running = progress !== null;
  const blocked = Boolean(single && preview.data && preview.data.active > 0);

  async function confirm() {
    setFailures([]);
    setProgress({ done: 0, total: graphs.length });
    const deleted: string[] = [];
    const failed: Array<{ graphName: string; message: string }> = [];
    for (const [index, graph] of graphs.entries()) {
      try {
        await remove.mutateAsync({ graphId: graph.graphId });
        deleted.push(graph.graphId);
      } catch (error) {
        failed.push({ graphName: graph.graphName, message: error instanceof Error ? error.message : String(error) });
      }
      setProgress({ done: index + 1, total: graphs.length });
    }
    setProgress(null);
    await utils.runs.list.invalidate();
    if (deleted.length > 0) {
      toast.success(
        deleted.length === 1
          ? `Deleted “${graphs.find((graph) => graph.graphId === deleted[0])?.graphName ?? deleted[0]}”`
          : `Deleted ${deleted.length} graphs`,
        { description: "Every run, its stored data and the uploads only it used are gone." },
      );
      await onDeleted?.(deleted);
    }
    if (failed.length > 0) {
      setFailures(failed);
    } else {
      onOpenChange(false);
    }
  }

  const title = single ? `Delete “${single.graphName}”?` : `Delete ${graphs.length} graphs?`;
  const shown = graphs.slice(0, NAME_LIMIT);

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (running) {
          return;
        }
        setFailures([]);
        onOpenChange(next);
      }}
    >
      <DialogContent aria-describedby="delete-graph-description" data-testid="delete-graph-dialog">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription id="delete-graph-description">
            {single ? "It is" : "They are"} deleted for good: this cannot be undone.
          </DialogDescription>
        </DialogHeader>

        {single ? (
          preview.isPending ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner />
              Checking what it holds…
            </p>
          ) : preview.error ? (
            <Alert variant="destructive">{preview.error.message}</Alert>
          ) : (
            <ul className="space-y-1.5 rounded-md border border-border bg-muted/30 p-3 text-sm" data-testid="delete-graph-scope">
              <li>
                {plural(preview.data.runs, "run")} - every version and attempt - with everything stored for{" "}
                {preview.data.runs === 1 ? "it" : "them"}
              </li>
              {preview.data.uploads > 0 ? (
                <li>
                  {plural(preview.data.uploads, "upload folder")} of documents only this graph was built from
                </li>
              ) : null}
              {preview.data.sharedWith.length > 0 ? (
                <li>
                  Shared with {preview.data.sharedWith.map(displayPrincipal).join(", ")}:{" "}
                  {preview.data.sharedWith.length === 1 ? "they lose" : "they all lose"} it too
                </li>
              ) : null}
            </ul>
          )
        ) : (
          <ul className="max-h-56 space-y-1 overflow-y-auto rounded-md border border-border bg-muted/30 p-3 text-sm">
            {shown.map((graph) => (
              <li key={graph.graphId} className="truncate font-medium">
                {graph.graphName}
              </li>
            ))}
            {graphs.length > shown.length ? (
              <li className="text-xs text-muted-foreground">and {graphs.length - shown.length} more</li>
            ) : null}
            <li className="pt-1 text-xs text-muted-foreground">
              Each goes with every run, its stored data, and the upload folders only it used.
            </li>
          </ul>
        )}

        {blocked ? (
          <Alert variant="destructive" className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
            It has a run under way. Cancel it from the graph&rsquo;s page, then delete the graph.
          </Alert>
        ) : null}

        {failures.length > 0 ? (
          <Alert variant="destructive" data-testid="delete-graph-failures">
            <p className="font-medium">
              {plural(failures.length, "graph")} could not be deleted:
            </p>
            <ul className="mt-1 space-y-0.5">
              {failures.map((failure) => (
                <li key={failure.graphName}>
                  {failure.graphName}: {failure.message}
                </li>
              ))}
            </ul>
          </Alert>
        ) : null}

        <div className="flex flex-wrap justify-end gap-2">
          <Button variant="outline" disabled={running} onClick={() => onOpenChange(false)}>
            {failures.length > 0 ? "Close" : "Cancel"}
          </Button>
          {failures.length === 0 ? (
            <Button
              variant="destructive"
              pending={running}
              disabled={blocked || graphs.length === 0 || (single !== null && Boolean(preview.error))}
              onClick={() => void confirm()}
            >
              {running
                ? graphs.length > 1
                  ? `Deleting ${Math.min(progress.done + 1, progress.total)} of ${progress.total}…`
                  : "Deleting…"
                : single
                  ? "Delete graph"
                  : `Delete ${graphs.length} graphs`}
            </Button>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
}
