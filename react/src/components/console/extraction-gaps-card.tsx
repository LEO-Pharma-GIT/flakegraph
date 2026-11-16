"use client";

import { AlertTriangle, Download } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { RecordTable, type RecordColumn } from "@/components/console/record-table";
import { ShowMore } from "@/components/console/show-more";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { downloadJson } from "@/lib/downloads";
import { prettyLabel } from "@/lib/graph-facets";
import type { ExtractionGaps, GapDocument, GapStage, GapWindow } from "@/server/gaps";

/** Windows of one document listed before "Show more". */
export const GAP_WINDOW_PAGE_SIZE = 10;
/** Characters of a window's preview shown before it is expanded. */
const PREVIEW_CLIP = 160;

type StageFilter = "all" | GapStage;

/**
 * What the graph is known to be missing: every stretch of text the model
 * read and returned records for that validation rejected in full. One
 * document with one such window and a corpus where hundreds lost several
 * read the same way - a count line, the reasons across the run, a document
 * table that searches, sorts and pages, and the windows of one document at
 * a time.
 */
export function ExtractionGapsCard({ gaps, graphId }: { gaps: ExtractionGaps; graphId: string }) {
  const [stage, setStage] = useState<StageFilter>("all");
  const [openFileId, setOpenFileId] = useState<string | null>(null);
  const rows = useMemo(
    () =>
      gaps.rows
        .filter((row) => stage === "all" || (stage === "entities" ? row.entityWindows > 0 : row.relationWindows > 0))
        .map((row) => ({
          ...row,
          id: row.fileId,
          stageWindows: stage === "entities" ? row.entityWindows : stage === "relations" ? row.relationWindows : row.windows,
          reasonText: reasonLine(row.reasons),
        })),
    [gaps.rows, stage],
  );
  const open = openFileId ? gaps.rows.find((row) => row.fileId === openFileId) : undefined;
  const none = gaps.windows === 0;

  return (
    <Card data-testid="extraction-gaps">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Extraction gaps</CardTitle>
          {none ? (
            <Badge variant="outline">None</Badge>
          ) : (
            <Badge variant="warning" data-testid="extraction-gaps-badge">
              {gaps.windows.toLocaleString()} {gaps.windows === 1 ? "window" : "windows"} ·{" "}
              {gaps.documents.toLocaleString()} of {gaps.documentsTotal.toLocaleString()} documents
            </Badge>
          )}
        </div>
        <CardDescription>
          A gap is one pass of the model over a stretch of text where every record it returned failed validation. In
          an entity window, what the text names is missing from the graph; in a relation window, its entities stand
          but no relation between them was kept from that pass. A later pass over the same text is counted on its
          own. Rejected records below lists every record turned away, gap or not.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {none ? (
          <p className="text-muted-foreground">Every window the model returned records for kept at least one.</p>
        ) : (
          <>
            <Alert variant="warning" className="flex items-start gap-2">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
              <span>
                {gaps.records.toLocaleString()} extracted {gaps.records === 1 ? "record was" : "records were"} rejected across{" "}
                {gaps.windows.toLocaleString()} {gaps.windows === 1 ? "window" : "windows"} in{" "}
                {gaps.documents.toLocaleString()} {gaps.documents === 1 ? "document" : "documents"}. Answers and exports
                built from this graph do not cover what those records stated.
              </span>
            </Alert>
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4" data-testid="extraction-gaps-stats">
              <Stat label="Entity windows" value={gaps.entityWindows} hint="No entity from the text was kept" />
              <Stat label="Relation windows" value={gaps.relationWindows} hint="Entities stand; no relation between them was kept" />
              <Stat label="Documents affected" value={gaps.documents} of={gaps.documentsTotal} />
              <Stat label="Records rejected" value={gaps.records} />
            </dl>
            {Object.keys(gaps.reasons).length > 0 ? (
              <div className="flex flex-wrap items-center gap-1.5" data-testid="extraction-gaps-reasons">
                <span className="text-xs text-muted-foreground">Why:</span>
                {Object.entries(gaps.reasons)
                  .sort(([, left], [, right]) => right - left)
                  .map(([reason, count]) => (
                    <Badge key={reason} variant="outline" title={reasonHelp(reason)}>
                      {prettyLabel(reason)} · {count.toLocaleString()}
                    </Badge>
                  ))}
              </div>
            ) : null}
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Select value={stage} onValueChange={(value) => setStage(value as StageFilter)}>
                <SelectTrigger aria-label="Gap kind" className="h-8 sm:w-64">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All gaps · {gaps.windows.toLocaleString()}</SelectItem>
                  <SelectItem value="entities">Entity windows · {gaps.entityWindows.toLocaleString()}</SelectItem>
                  <SelectItem value="relations">Relation windows · {gaps.relationWindows.toLocaleString()}</SelectItem>
                </SelectContent>
              </Select>
              <Button
                size="sm"
                variant="outline"
                className="sm:ml-auto"
                onClick={() => downloadJson(`${graphId}-extraction-gaps.json`, gaps)}
              >
                <Download className="size-3.5" aria-hidden />
                Download as JSON
              </Button>
            </div>
            <RecordTable
              rows={rows}
              total={gaps.rows.length}
              columns={GAP_COLUMNS}
              noun={{ one: "document", many: "documents" }}
              action={{ label: "Show windows", onClick: (id) => setOpenFileId(id === openFileId ? null : id) }}
              selectedId={openFileId}
            />
            {open ? <DocumentGaps key={open.fileId} document={open} stage={stage} onClose={() => setOpenFileId(null)} /> : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}

const GAP_COLUMNS: readonly RecordColumn[] = [
  { key: "name", label: "Document", kind: "text", title: (row) => String(row.fileId ?? "") },
  { key: "stageWindows", label: "Windows", kind: "number" },
  { key: "pages", label: "Pages", kind: "text", width: "w-36" },
  { key: "records", label: "Records rejected", kind: "number" },
  { key: "reasonText", label: "Why", kind: "long" },
];

/** The windows of one document, a page at a time, each with the text that was lost. */
function DocumentGaps({ document, stage, onClose }: { document: GapDocument; stage: StageFilter; onClose: () => void }) {
  const [limit, setLimit] = useState(GAP_WINDOW_PAGE_SIZE);
  const section = useRef<HTMLElement>(null);
  // The row a reader chose may be the fiftieth; the windows open below the
  // table, so bring them to where the reader is.
  useEffect(() => {
    section.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [document.fileId]);
  const windows = document.detail.filter((window) => stage === "all" || window.stage === stage);
  const page = windows.slice(0, limit);
  return (
    <section
      ref={section}
      className="space-y-3 rounded-md border border-border p-3"
      data-testid="document-gaps"
      aria-label={`Gaps in ${document.name}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-medium">{document.name}</h4>
        <span className="text-xs text-muted-foreground">
          {document.windows.toLocaleString()} {document.windows === 1 ? "window" : "windows"} · pages {document.pages || "—"} ·{" "}
          {document.records.toLocaleString()} records rejected
        </span>
        <Button size="sm" variant="ghost" className="ml-auto h-7" onClick={onClose}>
          Close
        </Button>
      </div>
      <ol className="space-y-2">
        {page.map((window, index) => (
          <GapWindowItem key={`${window.id}-${index}`} window={window} />
        ))}
      </ol>
      <ShowMore
        shown={page.length}
        total={windows.length}
        pageSize={GAP_WINDOW_PAGE_SIZE}
        noun="windows"
        onMore={() => setLimit((current) => current + GAP_WINDOW_PAGE_SIZE)}
      />
      {document.detailOmitted > 0 ? (
        <p className="text-xs text-muted-foreground">
          {document.detailOmitted.toLocaleString()} more {document.detailOmitted === 1 ? "window is" : "windows are"} counted
          above but not listed here; the JSON download holds every one.
        </p>
      ) : null}
    </section>
  );
}

function GapWindowItem({ window }: { window: GapWindow }) {
  const [expanded, setExpanded] = useState(false);
  const clipped = window.preview.length > PREVIEW_CLIP;
  const text = expanded || !clipped ? window.preview : `${window.preview.slice(0, PREVIEW_CLIP).trimEnd()}…`;
  return (
    <li className="space-y-1 rounded-md bg-muted/40 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge variant="secondary">{window.stage === "entities" ? "Entities" : "Relations"}</Badge>
        <span className="text-muted-foreground">
          {pageLabel(window)} · {window.extracted.toLocaleString()} {window.extracted === 1 ? "record" : "records"} rejected
          {Object.keys(window.reasons).length ? ` · ${reasonLine(window.reasons)}` : ""}
        </span>
      </div>
      {window.preview ? (
        <p className="text-sm">
          {text}
          {clipped ? (
            <Button variant="link" size="sm" className="h-auto px-1 py-0 text-xs" onClick={() => setExpanded((value) => !value)}>
              {expanded ? "Less" : "More"}
            </Button>
          ) : null}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">The window&apos;s text is not in this graph version.</p>
      )}
    </li>
  );
}

function Stat({ label, value, of, hint }: { label: string; value: number; of?: number; hint?: string }) {
  return (
    <div className="rounded-md border border-border px-3 py-2" title={hint}>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-lg font-semibold tabular-nums">
        {value.toLocaleString()}
        {of !== undefined ? <span className="text-sm font-normal text-muted-foreground"> of {of.toLocaleString()}</span> : null}
      </dd>
    </div>
  );
}

export function reasonLine(reasons: Record<string, number>): string {
  return Object.entries(reasons)
    .sort(([, left], [, right]) => right - left)
    .map(([reason, count]) => `${prettyLabel(reason)} ${count.toLocaleString()}`)
    .join(" · ");
}

function pageLabel(window: GapWindow): string {
  if (window.pageStart === null) {
    return "Page unknown";
  }
  if (window.pageEnd === null || window.pageEnd === window.pageStart) {
    return `Page ${window.pageStart}`;
  }
  return `Pages ${window.pageStart}–${window.pageEnd}`;
}

/** What each rejection reason means, for the hover on its chip. */
export function reasonHelp(reason: string): string {
  return (
    REASON_HELP[reason] ??
    "A validation rule the record did not meet; the run's extraction trace names the rule."
  );
}

const REASON_HELP: Record<string, string> = {
  ungrounded_quote: "The quote the model gave for the record is not verbatim in the window's text.",
  duplicate: "The record repeats one already kept from the same window.",
  untrusted_alias: "An alias the model proposed does not appear in the text.",
  domain_or_range_violation: "The relation joins entity types the ontology does not allow for it.",
  self_loop: "The relation's two ends are the same entity.",
  open_relation_type: "The relation type is outside the ontology and the run only keeps listed types.",
  type_definition_as_description: "The description restates the type's definition instead of the text.",
  invalid_schema: "The record did not match the extraction contract.",
  llm_grounding_unconfirmed:
    "The quote was not in the text, and the lines the model then pointed at do not name both ends of the relation.",
  llm_grounding_unsupported: "The quote was not in the text, and asked again the model found no line that states it.",
  row_endpoint_unnamed:
    "The evidence is one delimited table row, and that row does not name both ends of the relation by name or alias; only the document's subject may go unnamed.",
  value_as_entity: "A value such as a limit or a result, which belongs on a relation rather than being an entity.",
  document_context_name_mismatch: "The document's subject was named differently from how its quote names it.",
  contextual_surface_as_entity: "A pointer such as 'this paper' rather than a name.",
  identifier_as_entity:
    "A code the ontology names an identifier, which is an alias of what it identifies rather than an entity.",
  location_page_name:
    "The document's own file name, a folder path, or a line of its location page, which say where the document sits rather than naming a thing.",
  invalid_endpoint: "The relation names an entity that is not among the window's entities.",
  invalid_chunk: "The record cites a part of the document outside the window.",
  invalid_entity_type: "The entity type is not in the ontology.",
  identical_endpoint_surface: "Both ends of the relation were quoted with the same words.",
  unresolved_grounded_surface: "The words the relation quoted for an end do not match that entity.",
  blank_name: "The record has no name.",
  llm_grounding_to_verifier:
    "The lines the model pointed at do not name both ends; the verifier, shown the document's opening, decided.",
  verification_adjacency:
    "The verifier found the two ends only next to each other in a list, table, or layout, with nothing linking them.",
  verification_citation: "The verifier found a citation, which names a source, not the subject of the text.",
  verification_table_cell: "The verifier found the table cell belongs to another row or column.",
  verification_not_stated: "The verifier found the quote does not state the relation.",
  verification_wrong_direction: "The verifier found the quote states the relation the other way round.",
  verification_wrong_endpoint: "The verifier found the quote states it of a different thing than one of its ends.",
  verification_composition:
    "The verifier found wording such as 'made from' or 'derived from', which does not say the whole contains it.",
  verification_contradicted: "The verifier found the quote states the opposite.",
  verification_insufficient: "The verifier found the quote does not state the relation.",
  verification_low_confidence: "The verifier supported the relation, but less surely than the run requires.",
  verification_omitted: "The verifier gave no verdict on the relation.",
  verification_error: "The verification call failed, so the relation could not be confirmed.",
  verification_endpoint_rejected: "The verifier rejected one of the relation's entities.",
  verification_generic: "The verifier found a class or category term, not one specific thing.",
  verification_heading: "The verifier found a heading or label, not a thing.",
  verification_placeholder: "The verifier found a pointer or stand-in such as 'the product' or 'N/A'.",
  verification_value: "The verifier found a value such as a number or a date, not a thing.",
  verification_out_of_scope: "The verifier found a specific thing that none of the ontology's entity types covers.",
  retyped_entity:
    "The verifier found a specific thing of another entity type; it was kept under that type and its relations were checked again.",
};
