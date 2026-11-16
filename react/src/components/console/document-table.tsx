"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { DocumentStatus } from "@/server/documents";
import { folderAtLevel, folderName } from "@/lib/folder-entity";
import { cn, documentPhaseNeedsSkip, formatDocumentPhase, formatProvenance } from "@/lib/utils";

/** Rows shown before "Show more" - a corpus is hundreds of documents, a screen is not. */
export const DOCUMENT_PAGE_SIZE = 50;

const ATTENTION = "attention";

/** Phases in the order a reader wants them: trouble first, then the pipeline's own order. */
const PHASE_ORDER = [
  "poison",
  "ocr-failed",
  "extracted-0",
  "running",
  "queued",
  "scanned",
  "indexed",
  "inherited",
  "skipped",
  "cancelled",
];

function phaseRank(phase: string): number {
  const index = PHASE_ORDER.indexOf(phase);
  return index === -1 ? PHASE_ORDER.length : index;
}

/** Counts per phase, trouble first, for a one-line summary of a whole corpus. */
export function phaseSummary(documents: readonly DocumentStatus[]): Array<{ phase: string; count: number }> {
  const counts = new Map<string, number>();
  for (const item of documents) {
    counts.set(item.phase, (counts.get(item.phase) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([phase, count]) => ({ phase, count }))
    .sort((left, right) => phaseRank(left.phase) - phaseRank(right.phase));
}

/** The documents a person has to decide about: a worker gave up on them. */
export function documentsNeedingAttention(documents: readonly DocumentStatus[]): DocumentStatus[] {
  return documents.filter((item) => documentPhaseNeedsSkip(item.phase));
}

export function PhaseSummary({ documents }: { documents: readonly DocumentStatus[] }) {
  const summary = phaseSummary(documents);
  if (summary.length === 0) {
    return null;
  }
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Documents by phase" data-testid="phase-summary">
      {summary.map(({ phase, count }) => (
        <li
          key={phase}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs",
            documentPhaseNeedsSkip(phase)
              ? "border-destructive/40 bg-destructive/5 text-destructive"
              : "border-border bg-muted/40 text-muted-foreground",
          )}
        >
          <span className="font-medium tabular-nums text-foreground">{count.toLocaleString()}</span>
          {formatDocumentPhase(phase)}
        </li>
      ))}
    </ul>
  );
}

/**
 * A corpus-sized document list that stays readable: search by name, filter
 * by phase, trouble sorted first, and a page at a time. Optionally each row
 * carries a checkbox for leaving the document out of a revision, and, where
 * the graph ties documents to their folders, the folder it is tied to and a
 * checkbox for untying it.
 */
const HAS_GAPS = "gaps";

export function DocumentTable({
  documents,
  onSkip,
  skipping = null,
  selection,
  folder,
  gaps,
  emptyText = "No documents.",
}: {
  documents: readonly DocumentStatus[];
  onSkip?: (fileId: string) => void;
  /** The file whose skip is in flight. */
  skipping?: string | null;
  /** Extraction gaps per document, when the graph has any; shown as a chip and a filter. */
  gaps?: ReadonlyMap<string, number>;
  selection?: {
    removed: ReadonlySet<string>;
    onToggle: (fileId: string) => void;
    onRemove: (fileIds: string[]) => void;
    onKeep: (fileIds: string[]) => void;
  };
  /** The graph's folder level, and the documents the next version ties to no folder. */
  folder?: {
    level: number;
    untied: ReadonlySet<string>;
    onToggle: (fileId: string) => void;
  };
  emptyText?: string;
}) {
  const [search, setSearch] = useState("");
  const [phase, setPhase] = useState("all");
  const [limit, setLimit] = useState(DOCUMENT_PAGE_SIZE);
  const phases = useMemo(() => phaseSummary(documents), [documents]);
  const attentionCount = useMemo(() => documentsNeedingAttention(documents).length, [documents]);
  const gapCount = useMemo(() => documents.filter((item) => (gaps?.get(item.fileId) ?? 0) > 0).length, [documents, gaps]);

  const shown = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return documents
      .filter((item) => {
        if (phase === ATTENTION && !documentPhaseNeedsSkip(item.phase)) {
          return false;
        }
        if (phase === HAS_GAPS && !(gaps?.get(item.fileId) ?? 0)) {
          return false;
        }
        if (phase !== "all" && phase !== ATTENTION && phase !== HAS_GAPS && item.phase !== phase) {
          return false;
        }
        return (
          !needle ||
          `${item.name ?? ""} ${item.location ?? ""} ${item.fileId} ${item.detail}`.toLowerCase().includes(needle)
        );
      })
      .sort((left, right) => phaseRank(left.phase) - phaseRank(right.phase));
  }, [documents, gaps, phase, search]);
  const page = shown.slice(0, limit);
  const filtering = search.trim().length > 0 || phase !== "all";
  const shownIds = shown.map((item) => item.fileId);
  const removedShown = shownIds.filter((id) => selection?.removed.has(id)).length;

  if (documents.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyText}</p>;
  }

  return (
    <div className="space-y-3" data-testid="document-table">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <Input
          aria-label="Search documents"
          className="h-8 sm:max-w-xs"
          placeholder="Search by name"
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setLimit(DOCUMENT_PAGE_SIZE);
          }}
        />
        <Select
          value={phase}
          onValueChange={(value) => {
            setPhase(value);
            setLimit(DOCUMENT_PAGE_SIZE);
          }}
        >
          <SelectTrigger aria-label="Phase filter" className="h-8 sm:w-56">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All phases · {documents.length.toLocaleString()}</SelectItem>
            {attentionCount > 0 ? <SelectItem value={ATTENTION}>Needs attention · {attentionCount}</SelectItem> : null}
            {gapCount > 0 ? <SelectItem value={HAS_GAPS}>With extraction gaps · {gapCount}</SelectItem> : null}
            {phases.map(({ phase: value, count }) => (
              <SelectItem key={value} value={value}>
                {formatDocumentPhase(value)} · {count.toLocaleString()}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground sm:ml-auto" data-testid="document-table-count">
          {filtering
            ? `${shown.length.toLocaleString()} of ${documents.length.toLocaleString()} documents`
            : `${documents.length.toLocaleString()} documents`}
        </p>
      </div>
      {selection && shown.length > 0 ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span>
            {selection.removed.size > 0
              ? `${selection.removed.size.toLocaleString()} will be left out`
              : "Tick a document to leave it out of the next version."}
          </span>
          {removedShown < shown.length ? (
            <Button size="sm" variant="ghost" className="h-6 px-1.5 text-xs" onClick={() => selection.onRemove(shownIds)}>
              Remove {filtering ? "all shown" : "all"} ({shown.length.toLocaleString()})
            </Button>
          ) : null}
          {selection.removed.size > 0 ? (
            <Button size="sm" variant="ghost" className="h-6 px-1.5 text-xs" onClick={() => selection.onKeep([...selection.removed])}>
              Keep all
            </Button>
          ) : null}
        </div>
      ) : null}
      {shown.length === 0 ? (
        <p className="text-sm text-muted-foreground">No document matches this filter.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              {selection ? <TableHead className="w-20">Remove</TableHead> : null}
              <TableHead>Document</TableHead>
              {folder ? <TableHead className="w-56">Folder</TableHead> : null}
              <TableHead className="w-44">Phase</TableHead>
              <TableHead>Detail</TableHead>
              {onSkip ? <TableHead className="w-24" /> : null}
            </TableRow>
          </TableHeader>
          <TableBody>
            {page.map((item) => {
              const removed = selection?.removed.has(item.fileId) ?? false;
              return (
                <TableRow key={item.fileId} data-removed={removed ? "true" : undefined}>
                  {selection ? (
                    <TableCell>
                      <input
                        type="checkbox"
                        aria-label={`Remove ${item.name ?? item.fileId}`}
                        checked={removed}
                        onChange={() => selection.onToggle(item.fileId)}
                      />
                    </TableCell>
                  ) : null}
                  <TableCell className={cn("max-w-xs truncate", removed ? "text-muted-foreground line-through" : "font-medium")} title={item.location ?? item.fileId}>
                    {item.name ?? item.fileId}
                  </TableCell>
                  {folder ? <FolderCell item={item} folder={folder} /> : null}
                  <TableCell className={documentPhaseNeedsSkip(item.phase) ? "text-destructive" : undefined}>
                    {formatDocumentPhase(item.phase)}
                    {item.provenance && item.provenance !== "extracted" ? (
                      <Badge
                        variant="outline"
                        className="ml-2 text-[10px] font-normal"
                        data-testid="document-provenance"
                        title="How this version came by the document's extraction"
                      >
                        {formatProvenance(item.provenance)}
                      </Badge>
                    ) : null}
                  </TableCell>
                  <TableCell className="max-w-md truncate text-muted-foreground" title={item.detail}>
                    {item.detail}
                    {(gaps?.get(item.fileId) ?? 0) > 0 ? (
                      <Badge variant="warning" className="ml-2" data-testid="document-gaps">
                        {gaps?.get(item.fileId)?.toLocaleString()} {gaps?.get(item.fileId) === 1 ? "gap" : "gaps"}
                      </Badge>
                    ) : null}
                  </TableCell>
                  {onSkip ? (
                    <TableCell className="text-right">
                      {documentPhaseNeedsSkip(item.phase) ? (
                        <Button
                          size="sm"
                          variant="outline"
                          pending={skipping === item.fileId}
                          disabled={skipping !== null}
                          onClick={() => onSkip(item.fileId)}
                        >
                          {skipping === item.fileId ? "Skipping…" : "Skip file"}
                        </Button>
                      ) : null}
                    </TableCell>
                  ) : null}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
      {shown.length > page.length ? (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            Showing {page.length.toLocaleString()} of {shown.length.toLocaleString()}
          </span>
          <Button size="sm" variant="outline" className="h-7" onClick={() => setLimit((current) => current + DOCUMENT_PAGE_SIZE)}>
            Show {Math.min(DOCUMENT_PAGE_SIZE, shown.length - page.length)} more
          </Button>
        </div>
      ) : null}
    </div>
  );
}

/** The folder a document is tied to at the graph's level, with a checkbox for untying it. */
function FolderCell({
  item,
  folder,
}: {
  item: DocumentStatus;
  folder: { level: number; untied: ReadonlySet<string>; onToggle: (fileId: string) => void };
}) {
  const path = item.location ? folderAtLevel(item.location, folder.level) : null;
  const name = path === null ? null : folderName(path);
  const untied = folder.untied.has(item.fileId);
  if (name === null) {
    return (
      <TableCell className="text-xs text-muted-foreground">
        {item.location ? "No folder at this level" : "Location not recorded"}
      </TableCell>
    );
  }
  return (
    <TableCell className="max-w-56">
      <label className="flex items-center gap-2 text-xs" title={path ?? undefined}>
        <input
          type="checkbox"
          aria-label={`Exclude ${item.name ?? item.fileId} from folder ${name}`}
          checked={untied}
          onChange={() => folder.onToggle(item.fileId)}
        />
        <span className={cn("truncate", untied && "text-muted-foreground line-through")}>{name}</span>
      </label>
    </TableCell>
  );
}
