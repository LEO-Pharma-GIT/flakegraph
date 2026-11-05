"use client";

import { Download } from "lucide-react";
import { useMemo, useState } from "react";
import { reasonHelp } from "@/components/console/extraction-gaps-card";
import { RecordTable, type RecordColumn } from "@/components/console/record-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { downloadJson } from "@/lib/downloads";
import { prettyLabel } from "@/lib/graph-facets";
import type { RejectedRecords } from "@/server/rejected";

const ALL = "all";

/**
 * Every entity and relation the model stated that validation did not keep,
 * record by record, with the reason - and the relation type rules that
 * turned the most statements away, since widening a rule that is too narrow
 * is how true statements get into the graph.
 */
export function RejectedRecordsCard({ rejected, graphId }: { rejected: RejectedRecords; graphId: string }) {
  const [reason, setReason] = useState<string>(ALL);
  const reasons = useMemo(
    () => Object.entries(rejected.reasons).sort(([, left], [, right]) => right - left),
    [rejected.reasons],
  );
  const rows = useMemo(
    () =>
      rejected.rows
        .filter((row) => reason === ALL || `${row.kind}:${row.reason}` === reason)
        .map((row) => ({ ...row, reasonLabel: prettyLabel(row.reason), pageLabel: row.page ?? "" })),
    [rejected.rows, reason],
  );
  const none = rejected.records === 0;

  return (
    <Card data-testid="rejected-records">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Rejected records</CardTitle>
          {none ? (
            <Badge variant="outline">None</Badge>
          ) : (
            <Badge variant="warning" data-testid="rejected-records-badge">
              {rejected.records.toLocaleString()} records · {rejected.documents.toLocaleString()} of{" "}
              {rejected.documentsTotal.toLocaleString()} documents
            </Badge>
          )}
        </div>
        <CardDescription>
          Every entity and relation the model stated that validation did not keep, and why - wherever it happened,
          including text that kept other records. A rejection can be right, since the model is sometimes wrong, so
          read these as what the graph may be missing. A repeat of a record the graph kept is not listed.
        </CardDescription>
      </CardHeader>
      {none ? (
        <CardContent className="text-sm text-muted-foreground">Validation kept every record the model stated.</CardContent>
      ) : (
        <CardContent className="space-y-4 text-sm">
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Stat label="Entities" value={rejected.entities} />
            <Stat label="Relations" value={rejected.relations} />
            <Stat label="Documents" value={rejected.documents} of={rejected.documentsTotal} />
          </dl>
          <div className="flex flex-wrap items-center gap-1.5" data-testid="rejected-records-reasons">
            <span className="text-xs text-muted-foreground">Why:</span>
            {reasons.map(([key, count]) => {
              const [kind, name] = key.split(":");
              return (
                <Badge key={key} variant="outline" title={reasonHelp(name ?? key)}>
                  {kind === "relation" ? "Relation" : "Entity"} · {prettyLabel(name ?? key)} · {count.toLocaleString()}
                </Badge>
              );
            })}
          </div>
          {rejected.brokenRules.length > 0 ? (
            <section className="space-y-2" data-testid="broken-type-rules" aria-label="Type rules that turned relations away">
              <h4 className="font-medium">Type rules that turned relations away</h4>
              <p className="text-xs text-muted-foreground">
                The ontology does not allow these relations between these entity types. Where the statements are true,
                widening the rule in the ontology keeps them in the next run.
              </p>
              <RecordTable
                rows={rejected.brokenRules.map((rule, index) => ({ ...rule, id: `rule-${index}` }))}
                columns={RULE_COLUMNS}
                noun={{ one: "rule", many: "rules" }}
              />
            </section>
          ) : null}
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <Select value={reason} onValueChange={setReason}>
              <SelectTrigger aria-label="Rejection reason" className="h-8 sm:w-80">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL}>All reasons · {rejected.records.toLocaleString()}</SelectItem>
                {reasons.map(([key, count]) => {
                  const [kind, name] = key.split(":");
                  return (
                    <SelectItem key={key} value={key}>
                      {kind === "relation" ? "Relation" : "Entity"} · {prettyLabel(name ?? key)} · {count.toLocaleString()}
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
            <Button
              size="sm"
              variant="outline"
              className="sm:ml-auto"
              onClick={() => downloadJson(`${graphId}-rejected-records.json`, rejected)}
            >
              <Download className="size-3.5" aria-hidden />
              Download as JSON
            </Button>
          </div>
          <RecordTable rows={rows} total={rejected.rows.length} columns={RECORD_COLUMNS} noun={{ one: "record", many: "records" }} />
          {rejected.omitted > 0 ? (
            <p className="text-xs text-muted-foreground">
              {rejected.omitted.toLocaleString()} more {rejected.omitted === 1 ? "record is" : "records are"} counted above but
              not listed here.
            </p>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
}

const RULE_COLUMNS: readonly RecordColumn[] = [
  { key: "sourceType", label: "Source type", kind: "type" },
  { key: "relationType", label: "Relation", kind: "type" },
  { key: "targetType", label: "Target type", kind: "type" },
  { key: "count", label: "Relations", kind: "number" },
  { key: "example", label: "Example", kind: "long" },
];

const RECORD_COLUMNS: readonly RecordColumn[] = [
  { key: "document", label: "Document", kind: "text", width: "w-56" },
  { key: "pageLabel", label: "Page", kind: "number" },
  { key: "reasonLabel", label: "Why", kind: "text", width: "w-48" },
  { key: "statement", label: "Statement", kind: "long" },
  { key: "quote", label: "Quote", kind: "long" },
];

function Stat({ label, value, of }: { label: string; value: number; of?: number }) {
  return (
    <div className="rounded-md border border-border px-3 py-2">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-lg font-semibold tabular-nums">
        {value.toLocaleString()}
        {of !== undefined ? <span className="text-sm font-normal text-muted-foreground"> of {of.toLocaleString()}</span> : null}
      </dd>
    </div>
  );
}
