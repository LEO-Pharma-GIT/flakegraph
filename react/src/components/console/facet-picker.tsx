"use client";

import { useId, useMemo, useState } from "react";
import { Check, ChevronDown, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { pluralize, type FacetOption, type Noun } from "@/lib/graph-facets";
import { cn } from "@/lib/utils";

/**
 * Up to this many options a facet shows every option as a toggle chip: they
 * fit on a line or two of the card, and seeing them all is quicker than
 * opening a list. Above it chips stop being scannable (an open ontology
 * reaches hundreds of relation types), so the facet shows only what is
 * chosen and keeps the rest behind a searchable ranked list.
 */
export const INLINE_FACET_LIMIT = 8;

/**
 * Choose values of one facet (entity types, relation types, neighborhoods)
 * to narrow the graph to.
 *
 * The options arrive ranked largest first and the list keeps that order, so
 * the values that shape the graph most are the first ones seen.
 */
export function FacetPicker({
  label,
  noun,
  unit,
  options,
  value,
  onChange,
  testId,
}: {
  /** The facet's heading and the accessible name of its list. */
  label: string;
  /** What one option is, for "Choose entity types" and the footer count. */
  noun: Noun;
  /** What an option's count counts, for "12 entities". */
  unit: Noun;
  options: readonly FacetOption[];
  value: readonly string[];
  onChange: (next: string[]) => void;
  testId?: string;
}) {
  const headingId = useId();

  function toggle(id: string) {
    onChange(value.includes(id) ? value.filter((item) => item !== id) : [...value, id]);
  }

  return (
    // Content sits at the top of its column: the facet columns share a row,
    // and the tallest one must not stretch the others.
    <div className="grid content-start gap-2 text-sm" role="group" aria-labelledby={headingId} data-testid={testId}>
      <div className="flex items-baseline justify-between gap-2">
        <span id={headingId} className="text-sm font-medium leading-none">
          {label}
        </span>
        {options.length > 0 ? (
          <span className="text-xs tabular-nums text-muted-foreground">{options.length.toLocaleString()}</span>
        ) : null}
      </div>
      {options.length === 0 ? (
        <p className="text-xs text-muted-foreground">None on this graph.</p>
      ) : options.length <= INLINE_FACET_LIMIT ? (
        <InlineChips options={options} value={value} onToggle={toggle} />
      ) : (
        <PickerRow
          label={label}
          noun={noun}
          unit={unit}
          options={options}
          value={value}
          onToggle={toggle}
          onClear={() => onChange([])}
          testId={testId}
        />
      )}
    </div>
  );
}

function InlineChips({
  options,
  value,
  onToggle,
}: {
  options: readonly FacetOption[];
  value: readonly string[];
  onToggle: (id: string) => void;
}) {
  return (
    <div className="flex flex-wrap content-start items-start gap-1.5">
      {options.map((option) => {
        const selected = value.includes(option.id);
        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={selected}
            className={cn(
              "inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-xs",
              selected
                ? "border-primary bg-accent text-foreground"
                : "border-border bg-background text-muted-foreground hover:bg-muted/60",
            )}
            onClick={() => onToggle(option.id)}
          >
            {option.label}
            <span className="tabular-nums opacity-70">{option.count.toLocaleString()}</span>
          </button>
        );
      })}
    </div>
  );
}

function PickerRow({
  label,
  noun,
  unit,
  options,
  value,
  onToggle,
  onClear,
  testId,
}: {
  label: string;
  noun: Noun;
  unit: Noun;
  options: readonly FacetOption[];
  value: readonly string[];
  onToggle: (id: string) => void;
  onClear: () => void;
  testId?: string;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const byId = useMemo(() => new Map(options.map((item) => [item.id, item])), [options]);
  const chosen = value.map((id) => byId.get(id)).filter((item): item is FacetOption => Boolean(item));
  const matching = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return needle
      ? options.filter((item) => item.label.toLowerCase().includes(needle) || item.id.toLowerCase().includes(needle))
      : options;
  }, [options, search]);

  function setOpenAndReset(next: boolean) {
    setOpen(next);
    // A search left behind would hide options the next time the list opens.
    if (!next) {
      setSearch("");
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Popover open={open} onOpenChange={setOpenAndReset}>
        <PopoverTrigger asChild>
          <Button size="sm" variant="outline" className="h-7 gap-1 px-2 text-xs">
            {chosen.length ? `${chosen.length} chosen` : `Choose ${noun.many}`}
            <ChevronDown className={cn("size-3.5 transition-transform", open ? "rotate-180" : "")} aria-hidden="true" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[min(24rem,calc(100vw-2rem))]" data-testid={testId ? `${testId}-panel` : undefined}>
          <div className="border-b border-border p-2">
            <Input
              aria-label={`Find ${noun.many}`}
              className="h-8"
              placeholder={`Find ${noun.many}`}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <ul className="max-h-72 overflow-y-auto py-1" aria-label={label}>
            {matching.length === 0 ? (
              <li className="px-3 py-2 text-sm text-muted-foreground">No {noun.one} matches.</li>
            ) : (
              matching.map((option) => {
                const selected = value.includes(option.id);
                return (
                  <li key={option.id}>
                    <label
                      className={cn(
                        "flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-muted/60",
                        selected ? "bg-accent/60" : "",
                      )}
                    >
                      {/*
                        The input is the box itself, drawn with appearance-none, so a
                        click or an automated check lands on the control and not on a
                        decoration over it.
                      */}
                      <span className="relative flex size-4 shrink-0">
                        <input
                          type="checkbox"
                          className="peer size-4 appearance-none rounded-sm border border-border bg-background checked:border-primary checked:bg-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                          checked={selected}
                          onChange={() => onToggle(option.id)}
                        />
                        <Check
                          className="pointer-events-none absolute inset-0 m-auto size-3 text-primary-foreground opacity-0 peer-checked:opacity-100"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="min-w-0 flex-1 truncate" title={option.label}>
                        {option.label}
                      </span>
                      <span className="shrink-0 tabular-nums text-xs text-muted-foreground">{pluralize(option.count, unit)}</span>
                    </label>
                  </li>
                );
              })
            )}
          </ul>
          <div className="flex items-center justify-between gap-3 border-t border-border px-3 py-1.5 text-xs text-muted-foreground">
            <span>
              {search.trim() ? `${matching.length.toLocaleString()} of ` : ""}
              {pluralize(options.length, noun)} · {chosen.length} chosen
            </span>
            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs" onClick={() => setOpenAndReset(false)}>
              Done
            </Button>
          </div>
        </PopoverContent>
      </Popover>
      {chosen.map((item) => (
        <span
          key={item.id}
          className="inline-flex h-7 max-w-xs items-center gap-1 rounded-md border border-primary/40 bg-accent pl-2 pr-1 text-xs"
        >
          <span className="truncate" title={item.label}>
            {item.label}
          </span>
          <button
            type="button"
            aria-label={`Remove ${item.label}`}
            className="rounded-sm p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
            onClick={() => onToggle(item.id)}
          >
            <X className="size-3" aria-hidden="true" />
          </button>
        </span>
      ))}
      {chosen.length > 1 ? (
        <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" aria-label={`Clear ${noun.many}`} onClick={onClear}>
          Clear
        </Button>
      ) : null}
    </div>
  );
}
