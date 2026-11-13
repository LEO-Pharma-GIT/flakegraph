"use client";

import { useMemo, useState } from "react";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { DocumentTable } from "@/components/console/document-table";
import { IngestionForm, type RevisionTarget } from "@/components/console/ingestion-form";
import { ShowMore } from "@/components/console/show-more";
import type { DocumentStatus } from "@/server/documents";
import { folderExcluded } from "@/lib/folder-entity";
import type { Capability, GraphVersion, RunSnapshot, RuntimeMode } from "@/server/protocol/schema";
import { orderVersions } from "@/lib/paging";
import { formatInstant } from "@/lib/utils";

/** Versions on screen before "Show more"; a graph revised weekly has dozens. */
export const VERSION_PAGE_SIZE = 10;

/**
 * Edit a finished graph by building its next version.
 *
 * A graph is what its documents say, so editing means changing the document
 * set: tick the ones to leave out, add new ones through the same source
 * picker compose uses, and build. The kept documents are not processed again;
 * the workers extract only what is added, and the finalizer resolves entities
 * and communities over everything together, so the new version is one
 * consistent graph and not a patch on the old one. The old version stays.
 */
export function GraphEditor({
  snapshot,
  documents,
  runtime,
  capabilities,
  onSubmitted,
}: {
  snapshot: RunSnapshot;
  documents: DocumentStatus[];
  runtime: RuntimeMode;
  capabilities: Set<Capability>;
  onSubmitted?: (runId: string) => Promise<void> | void;
}) {
  const [removed, setRemoved] = useState<ReadonlySet<string>>(new Set());
  // A document a worker could not process is not in the graph, so there is
  // nothing to keep or remove; the rest were extracted (this run's own) or
  // kept from an earlier version.
  const held = documents.filter((item) => item.phase === "indexed" || item.phase === "inherited");
  // The base version's vocabulary: a new version is typed like the documents
  // it keeps, so the form shows it and does not offer to change it.
  const baseOntology = trpc.runs.ontology.useQuery({ runId: snapshot.runId });
  const folder = baseOntology.data?.folderEntity ?? null;
  // Which kept documents the next version ties to no folder, once someone
  // changes it here; until then the base version's own exclusions stand.
  const [untied, setUntied] = useState<ReadonlySet<string> | null>(null);
  const baseUntied = useMemo(
    () => new Set(folder ? held.filter((item) => folderExcluded(folder, item)).map((item) => item.fileId) : []),
    [folder, held],
  );
  const effectiveUntied = untied ?? baseUntied;
  const revision = useMemo<RevisionTarget>(
    () => ({
      baseRunId: snapshot.runId,
      graphId: snapshot.graphId,
      graphName: snapshot.graphName,
      dropFileIds: held.filter((item) => removed.has(item.fileId)).map((item) => item.fileId),
      keptCount: held.filter((item) => !removed.has(item.fileId)).length,
      ontology: baseOntology.data ?? null,
      folderExclusions: untied && !sameMembers(untied, baseUntied) ? [...untied].sort() : null,
    }),
    [held, removed, snapshot.graphId, snapshot.graphName, snapshot.runId, baseOntology.data, untied, baseUntied],
  );
  const toggleFolder = (fileId: string) =>
    setUntied((current) => {
      const next = new Set(current ?? baseUntied);
      if (next.has(fileId)) {
        next.delete(fileId);
      } else {
        next.add(fileId);
      }
      return next;
    });
  const toggle = (fileId: string) =>
    setRemoved((current) => {
      const next = new Set(current);
      if (next.has(fileId)) {
        next.delete(fileId);
      } else {
        next.add(fileId);
      }
      return next;
    });
  // Editing an older version branches from it: the new version is built
  // from these documents, not the head's, and becomes the head. Say so,
  // because the tab reads the same either way.
  const version = snapshot.raw.version as { number: number; count: number; head: boolean } | null | undefined;
  const behindHead = Boolean(version && version.number && version.count && !version.head);
  return (
    <div className="space-y-6" data-testid="graph-editor">
      {behindHead ? (
        <Alert variant="warning" data-testid="editing-behind-head">
          This is version {version!.number} of {version!.count}, not the head. A version built from here starts
          from these documents, not from what the head holds, and becomes the new head.
        </Alert>
      ) : null}
      <Card>
        <CardHeader>
          <CardTitle>Documents in this version</CardTitle>
          <CardDescription>
            Tick a document to leave it out of the next version. What stays is not parsed or extracted again; the
            graph is rebuilt from it together with anything you add below, so entities and neighborhoods are
            resolved across the whole corpus. This version stays readable.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DocumentTable
            documents={held}
            emptyText="This version holds no documents."
            selection={{
              removed,
              onToggle: toggle,
              onRemove: (ids) => setRemoved((current) => new Set([...current, ...ids])),
              onKeep: (ids) =>
                setRemoved((current) => {
                  const next = new Set(current);
                  for (const id of ids) {
                    next.delete(id);
                  }
                  return next;
                }),
            }}
            folder={
              folder
                ? { level: folder.level, untied: effectiveUntied, onToggle: toggleFolder }
                : undefined
            }
          />
          {removed.size ? (
            <p className="mt-3 text-sm" data-testid="removal-summary">
              {removed.size.toLocaleString()} of {held.length.toLocaleString()} document{held.length === 1 ? "" : "s"} will be left out.
            </p>
          ) : null}
          {revision.folderExclusions !== null ? (
            <p className="mt-3 text-sm" data-testid="folder-summary">
              {revision.folderExclusions.length.toLocaleString()} document{revision.folderExclusions.length === 1 ? "" : "s"} will
              be tied to no folder. The next version is rebuilt from the stored extraction; nothing is extracted again.
            </p>
          ) : null}
        </CardContent>
      </Card>
      <IngestionForm
        runtime={runtime}
        capabilities={capabilities}
        suggestionMode="off"
        revision={revision}
        onSubmitted={onSubmitted}
      />
    </div>
  );
}

/**
 * Every published version of the graph, with the one being viewed and the
 * head marked. The head comes first because it is what the fleet serves,
 * then the version on screen, so both are in view whatever the page; the
 * rest read newest first, a page at a time.
 */
export function GraphVersionsCard({
  graphId,
  runId,
  onOpenRun,
}: {
  graphId: string;
  runId: string;
  onOpenRun?: (runId: string) => void;
}) {
  const versions = trpc.graphs.versions.useQuery({ graphId });
  const [limit, setLimit] = useState(VERSION_PAGE_SIZE);
  const rows: readonly GraphVersion[] = useMemo(() => orderVersions(versions.data ?? [], runId), [versions.data, runId]);
  const page = rows.slice(0, limit);
  return (
    <Card data-testid="graph-versions">
      <CardHeader>
        <CardTitle>Versions of this graph</CardTitle>
        <CardDescription>
          Each version is one run that published a graph. The head is what the fleet reads by the graph&apos;s
          name; earlier versions stay readable by their run.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {versions.isLoading ? (
          <p className="text-sm text-muted-foreground">Reading versions…</p>
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">No version has been published for this graph yet.</p>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground" data-testid="graph-versions-count">
              {rows.length.toLocaleString()} version{rows.length === 1 ? "" : "s"}
            </p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Version</TableHead>
                  <TableHead>Run</TableHead>
                  <TableHead>Published</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {page.map((version) => (
                  <TableRow key={version.runId}>
                    <TableCell className="font-medium">
                      v{version.number}
                      {version.head ? <span className="ml-2 rounded bg-primary/10 px-1.5 py-0.5 text-xs text-primary">head</span> : null}
                      {version.runId === runId ? <span className="ml-2 text-xs text-muted-foreground">viewing</span> : null}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{version.runId}</TableCell>
                    <TableCell>{formatInstant(version.createdAt)}</TableCell>
                    <TableCell className="text-right">
                      {version.runId !== runId && onOpenRun ? (
                        <Button size="sm" variant="outline" onClick={() => onOpenRun(version.runId)}>
                          Open
                        </Button>
                      ) : null}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <ShowMore
              shown={page.length}
              total={rows.length}
              pageSize={VERSION_PAGE_SIZE}
              noun="versions"
              onMore={() => setLimit((current) => current + VERSION_PAGE_SIZE)}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function sameMembers(left: ReadonlySet<string>, right: ReadonlySet<string>): boolean {
  return left.size === right.size && [...left].every((item) => right.has(item));
}
