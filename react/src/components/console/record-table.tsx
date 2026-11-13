"use client";

import { ArrowDown, ArrowUp, Crosshair } from "lucide-react";
import { useMemo, useState, type MouseEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { prettyLabel } from "@/lib/graph-facets";
import { cn } from "@/lib/utils";

/** Rows shown before "Show more" - the same page the document list turns. */
export const RECORD_PAGE_SIZE = 50;

/**
 * Characters of a long text cell shown before it has to be expanded. A table
 * column is a fraction of the screen, so the threshold is well below the Ask
 * panel's passage preview.
 */
export const CELL_PREVIEW = 80;

export type RecordRow = Record<string, unknown>;

export type RecordColumnKind =
  /** Plain text, clipped to the cell with the full text on hover. */
  | "text"
  /** Paragraph-sized text: one line until the row is expanded. */
  | "long"
  /** An identifier - monospace and muted so names stay the eye's anchor. */
  | "id"
  /** An entity or relation type, drawn as a chip. */
  | "type"
  /** A count or confidence, right-aligned and sorted as a number. */
  | "number";

export interface RecordColumn {
  /** The row field, and the header's stable identity. */
  key: string;
  /** Header text; the field name made readable when absent. */
  label?: string;
  kind?: RecordColumnKind;
  /** The value shown; `row[key]` (or its camelCase twin) when absent. */
  value?: (row: RecordRow) => unknown;
  /** Hover text, such as the id behind a resolved name. */
  title?: (row: RecordRow) => string | undefined;
  /** Swatch colour for a type chip. */
  color?: (text: string) => string;
  /** A Tailwind width class for the column; text columns share what is left. */
  width?: string;
}

/**
 * Columns whose content has a known shape take a fixed width so the text
 * columns get the rest; the table lays out by these widths rather than by
 * measuring every cell, which is what keeps it inside its container.
 */
const KIND_WIDTHS: Partial<Record<RecordColumnKind, string>> = {
  id: "w-48",
  type: "w-44",
  number: "w-28",
};

export function columnWidth(column: RecordColumn): string | undefined {
  return column.width ?? (column.kind ? KIND_WIDTHS[column.kind] : undefined);
}

export interface RecordSort {
  key: string;
  direction: "asc" | "desc";
}

/** The text a cell shows, searches by and sorts on. */
export function cellText(row: RecordRow, column: RecordColumn): string {
  const value = column.value ? column.value(row) : (row[column.key] ?? row[camel(column.key)]);
  return stringify(value);
}

/** Case-insensitive match against any shown column; an empty needle matches everything. */
export function matchesSearch(row: RecordRow, columns: readonly RecordColumn[], needle: string): boolean {
  const query = needle.trim().toLowerCase();
  if (!query) {
    return true;
  }
  return columns.some((column) => cellText(row, column).toLowerCase().includes(query));
}

/**
 * Compare two cell texts the way a reader would: numbers by value (so 0.95
 * sorts after 0.6 and 10 after 9), everything else alphabetically and
 * without regard to case.
 */
export function compareCells(left: string, right: string): number {
  const leftNumber = asNumber(left);
  const rightNumber = asNumber(right);
  if (leftNumber != null && rightNumber != null) {
    return leftNumber - rightNumber;
  }
  return left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" });
}

/**
 * Rows in the requested order. Empty cells sink to the bottom in either
 * direction, and ties keep the incoming order so a re-sort is stable.
 */
export function sortRecords<Row extends RecordRow>(
  rows: readonly Row[],
  columns: readonly RecordColumn[],
  sort: RecordSort | null,
): Row[] {
  const column = sort ? columns.find((item) => item.key === sort.key) : undefined;
  if (!sort || !column) {
    return [...rows];
  }
  const sign = sort.direction === "asc" ? 1 : -1;
  return rows
    .map((row, index) => ({ row, index, text: cellText(row, column) }))
    .sort((left, right) => {
      if (left.text === "" || right.text === "") {
        return left.text === right.text ? left.index - right.index : left.text === "" ? 1 : -1;
      }
      return sign * compareCells(left.text, right.text) || left.index - right.index;
    })
    .map((entry) => entry.row);
}

/** What one click on a header does: sort ascending, then flip. */
export function nextSort(current: RecordSort | null, key: string): RecordSort {
  if (current?.key === key) {
    return { key, direction: current.direction === "asc" ? "desc" : "asc" };
  }
  return { key, direction: "asc" };
}

/** The first `limit` characters of a long text, and whether anything was cut. */
export function clipText(text: string, limit: number = CELL_PREVIEW): { clipped: boolean; preview: string } {
  if (text.length <= limit) {
    return { clipped: false, preview: text };
  }
  return { clipped: true, preview: `${text.slice(0, limit).trimEnd()}…` };
}

/**
 * One line that says how much of the table is on screen: the page against
 * the rows that match, and the rows that match against everything the graph
 * has when a filter or search hides some.
 */
export function countLine(shown: number, matched: number, total: number, noun: RecordNoun): string {
  const base =
    shown < matched
      ? `Showing ${shown.toLocaleString()} of ${matched.toLocaleString()} ${noun.many}`
      : `${matched.toLocaleString()} ${matched === 1 ? noun.one : noun.many}`;
  return matched < total ? `${base} (of ${total.toLocaleString()} total)` : base;
}

/** What a row is called, for the count line and the search box. */
export interface RecordNoun {
  one: string;
  many: string;
}

/**
 * A record list that stays usable at thousands of rows: search across the
 * shown columns, sort by any header, long text folded to one line, and a
 * page at a time.
 */
export function RecordTable({
  rows,
  total = rows.length,
  columns,
  noun,
  action,
  selectedId,
  selectionId = defaultSelectionId,
  highlightIds,
}: {
  /** The rows the current graph filters leave visible. */
  rows: readonly RecordRow[];
  /** Rows the graph has before filters, for the count line. */
  total?: number;
  columns: readonly RecordColumn[];
  noun: RecordNoun;
  /**
   * What a row leads to, as a trailing button and as the row's click: the
   * table is a way of finding a record, and this is the way from it to the
   * place it lives (the canvas, a neighborhood focus).
   */
  action?: { label: string; onClick: (id: string) => void };
  selectedId?: string | null;
  /** The id a row click selects and the selection highlights; the row's own id when absent. */
  selectionId?: (row: RecordRow, index: number) => string | null;
  highlightIds?: ReadonlySet<string>;
}) {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<RecordSort | null>(null);
  const [limit, setLimit] = useState(RECORD_PAGE_SIZE);
  const matched = useMemo(
    () => sortRecords(rows.filter((row) => matchesSearch(row, columns, search)), columns, sort),
    [columns, rows, search, sort],
  );
  const page = matched.slice(0, limit);

  if (rows.length === 0) {
    return <p className="py-6 text-sm text-muted-foreground">No rows in this filter.</p>;
  }

  return (
    <div className="space-y-3" data-testid="record-table">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <Input
          aria-label={`Find in ${noun.many}`}
          className="h-8 sm:max-w-xs"
          placeholder="Find in this table"
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setLimit(RECORD_PAGE_SIZE);
          }}
        />
        <p className="text-xs text-muted-foreground sm:ml-auto" data-testid="record-table-count">
          {countLine(page.length, matched.length, Math.max(total, rows.length), noun)}
        </p>
      </div>
      {matched.length === 0 ? (
        <p className="text-sm text-muted-foreground">No row matches this search.</p>
      ) : (
        <Table className="table-fixed min-w-[40rem]">
          <TableHeader>
            <TableRow>
              {columns.map((column) => {
                const active = sort?.key === column.key ? sort.direction : null;
                return (
                  <TableHead
                    key={column.key}
                    aria-sort={active === "asc" ? "ascending" : active === "desc" ? "descending" : "none"}
                    className={cn(column.kind === "number" ? "text-right" : "", columnWidth(column))}
                  >
                    <button
                      type="button"
                      className="inline-flex items-center gap-1 uppercase tracking-wide hover:text-foreground"
                      onClick={() => {
                        setSort((current) => nextSort(current, column.key));
                        setLimit(RECORD_PAGE_SIZE);
                      }}
                    >
                      {column.label ?? prettyLabel(column.key)}
                      {active === "asc" ? (
                        <ArrowUp className="size-3" aria-hidden="true" />
                      ) : active === "desc" ? (
                        <ArrowDown className="size-3" aria-hidden="true" />
                      ) : null}
                    </button>
                  </TableHead>
                );
              })}
              {action ? <TableHead className="w-36" /> : null}
            </TableRow>
          </TableHeader>
          <TableBody>
            {page.map((row, index) => {
              const id = selectionId(row, index);
              const key = String(row.id ?? id ?? index);
              const name = cellText(row, columns[0]);
              return (
                <TableRow
                  key={key}
                  className={cn(
                    "group",
                    action && id ? "cursor-pointer" : "",
                    id && selectedId === id ? "bg-accent" : "",
                    id && highlightIds?.has(id) ? "bg-amber-50 dark:bg-amber-950/40" : "",
                  )}
                  onClick={action && id ? () => action.onClick(id) : undefined}
                >
                  {columns.map((column) => (
                    <RecordCell key={column.key} row={row} column={column} />
                  ))}
                  {action ? (
                    <TableCell className="text-right">
                      {id ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 gap-1 px-2 text-xs text-muted-foreground opacity-70 group-hover:opacity-100 hover:text-foreground"
                          aria-label={`${action.label}: ${name}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            action.onClick(id);
                          }}
                        >
                          <Crosshair className="size-3.5" aria-hidden="true" />
                          {action.label}
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
      {matched.length > page.length ? (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>
            Showing {page.length.toLocaleString()} of {matched.length.toLocaleString()}
          </span>
          <Button size="sm" variant="outline" className="h-7" onClick={() => setLimit((current) => current + RECORD_PAGE_SIZE)}>
            Show {Math.min(RECORD_PAGE_SIZE, matched.length - page.length)} more
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function RecordCell({ row, column }: { row: RecordRow; column: RecordColumn }) {
  const text = cellText(row, column);
  const title = column.title?.(row);
  switch (column.kind) {
    case "long":
      return (
        <TableCell>
          <LongText text={text} />
        </TableCell>
      );
    case "id":
      return (
        <TableCell className="truncate font-mono text-[11px] text-muted-foreground" title={title ?? text}>
          {text}
        </TableCell>
      );
    case "type":
      return (
        <TableCell className="truncate" title={title ?? text}>
          {text ? (
            <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-1.5 py-0.5 text-[11px] font-medium">
              {column.color ? (
                <span className="size-2 shrink-0 rounded-full" style={{ background: column.color(text) }} aria-hidden="true" />
              ) : null}
              {text}
            </span>
          ) : null}
        </TableCell>
      );
    case "number":
      return (
        <TableCell className="whitespace-nowrap text-right tabular-nums" title={title}>
          {text}
        </TableCell>
      );
    default:
      return (
        <TableCell className="truncate" title={title ?? text}>
          {text}
        </TableCell>
      );
  }
}

/** Paragraph-sized text on one line, with the whole of it a click away. */
function LongText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const { clipped, preview } = clipText(text);
  // The row behind the toggle selects on click; opening a quote must not
  // move the canvas.
  function toggle(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
    setExpanded((current) => !current);
  }
  if (!clipped) {
    return <span className="block truncate">{text}</span>;
  }
  return (
    <div className="flex items-baseline gap-2">
      <span className={cn("min-w-0 flex-1", expanded ? "whitespace-normal" : "truncate")} title={expanded ? undefined : text}>
        {expanded ? text : preview}
      </span>
      <button
        type="button"
        aria-expanded={expanded}
        className="shrink-0 text-xs text-muted-foreground hover:text-foreground"
        onClick={toggle}
      >
        {expanded ? "Less" : "More"}
      </button>
    </div>
  );
}

function defaultSelectionId(row: RecordRow, index: number): string {
  return String(row.id ?? index);
}

function stringify(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  return JSON.stringify(value);
}

function camel(value: string): string {
  return value.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase());
}

function asNumber(text: string): number | null {
  if (text.trim() === "") {
    return null;
  }
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}
