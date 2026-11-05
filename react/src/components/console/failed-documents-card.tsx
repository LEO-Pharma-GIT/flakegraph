"use client";

import { Download, FileX } from "lucide-react";
import { useMemo } from "react";
import { RecordTable, type RecordColumn } from "@/components/console/record-table";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { downloadJson } from "@/lib/downloads";
import type { FailedDocuments } from "@/server/failed-documents";

/**
 * The source documents no parser could read. A run records each one and goes
 * on without it rather than failing, so this is where a reader learns which
 * files the graph does not cover at all, and the error each one gave - one
 * file or several hundred, the same searchable table.
 */
export function FailedDocumentsCard({ failed, graphId }: { failed: FailedDocuments; graphId: string }) {
  const rows = useMemo(
    () =>
      failed.rows.map((row) => ({
        ...row,
        id: row.fileId,
        error: row.errorMessage ? `${row.errorType}: ${row.errorMessage}` : row.errorType,
      })),
    [failed.rows],
  );
  const none = failed.documents === 0;

  return (
    <Card data-testid="failed-documents">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Documents that could not be read</CardTitle>
          {none ? (
            <Badge variant="outline">None</Badge>
          ) : (
            <Badge variant="danger" data-testid="failed-documents-badge">
              {failed.documents.toLocaleString()} of {failed.documentsTotal.toLocaleString()} documents
            </Badge>
          )}
        </div>
        <CardDescription>
          A document no parser could open — a file that is not what its name says, a damaged one, or a format the
          parser refuses. The run recorded it and went on, so the rest of the graph is complete; nothing from these
          files is in it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {none ? (
          <p className="text-muted-foreground">Every document the run was given was read.</p>
        ) : (
          <>
            <Alert variant="destructive" className="flex items-start gap-2">
              <FileX className="mt-0.5 size-4 shrink-0" aria-hidden />
              <span>
                {failed.documents.toLocaleString()} {failed.documents === 1 ? "document" : "documents"} could not be read.
                Answers and exports built from this graph do not cover {failed.documents === 1 ? "it" : "them"}; fix or
                replace the {failed.documents === 1 ? "file" : "files"} and run the graph again to include{" "}
                {failed.documents === 1 ? "it" : "them"}.
              </span>
            </Alert>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="flex flex-wrap items-center gap-1.5" data-testid="failed-documents-errors">
                <span className="text-xs text-muted-foreground">Errors:</span>
                {Object.entries(failed.byError).map(([errorType, count]) => (
                  <Badge key={errorType} variant="outline">
                    {errorType} · {count.toLocaleString()}
                  </Badge>
                ))}
              </div>
              <Button
                size="sm"
                variant="outline"
                className="sm:ml-auto"
                onClick={() => downloadJson(`${graphId}-failed-documents.json`, failed)}
              >
                <Download className="size-3.5" aria-hidden />
                Download as JSON
              </Button>
            </div>
            <RecordTable
              rows={rows}
              total={failed.rows.length}
              columns={FAILED_COLUMNS}
              noun={{ one: "document", many: "documents" }}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}

const FAILED_COLUMNS: readonly RecordColumn[] = [
  { key: "name", label: "Document", kind: "text", title: (row) => String(row.sourceUri ?? "") },
  { key: "mimeType", label: "Type", kind: "text", width: "w-40" },
  { key: "error", label: "Why", kind: "long" },
];
