"use client";

import { useEffect, useMemo, useState } from "react";
import { trpc } from "@/components/providers";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

export function CommandPalette({
  runId,
  role,
  canFleet = false,
  onSelectRun,
  onJumpEntity,
  onNewGraph,
  onFleet,
  onStaff,
  onKeys,
}: {
  runId?: string | null;
  role?: "operator" | "analyst" | "staff";
  canFleet?: boolean;
  onSelectRun: (runId: string) => void;
  onJumpEntity?: (runId: string, search: string) => void;
  onNewGraph: () => void;
  onFleet: () => void;
  onStaff: () => void;
  onKeys: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const runs = trpc.runs.list.useQuery({ limit: 100 }, { enabled: open });
  const workspace = trpc.workspace.get.useQuery(undefined, { enabled: open });
  const graph = trpc.graphs.loadRun.useQuery({ runId: runId ?? "" }, { enabled: open && Boolean(runId) });
  const analyst = role === "analyst";
  const actions = useMemo(() => {
    const items = [
      ...(analyst ? [] : [{ id: "new", label: "New graph", run: () => onNewGraph() }]),
      ...(canFleet ? [{ id: "fleet", label: "Open fleet", run: () => onFleet() }] : []),
      ...(role === "staff" ? [{ id: "staff", label: "Staff incident", run: () => onStaff() }] : []),
      ...(role === "staff" || role === "operator" ? [{ id: "keys", label: "SDK keys", run: () => onKeys() }] : []),
      ...(workspace.data?.perspectives ?? []).flatMap((item) => {
        if (analyst && item.lifecycle !== "production") {
          return [];
        }
        const match = (runs.data ?? []).find((row) => row.graphId === item.graphId);
        if (!match) {
          return [];
        }
        return [
          {
            id: `perspective:${item.id}`,
            label: `Perspective ${item.name} · ${item.lifecycle}`,
            run: () => onSelectRun(match.runId),
          },
        ];
      }),
      ...(runs.data ?? [])
        .filter((run) => {
          if (!analyst) {
            return true;
          }
          return (workspace.data?.perspectives ?? []).some(
            (item) => item.lifecycle === "production" && item.graphId === run.graphId,
          );
        })
        .map((run) => ({
        id: `run:${run.runId}`,
        label: `${run.graphName || run.graphId} · ${run.status}`,
        run: () => onSelectRun(run.runId),
      })),
      ...(graph.data?.nodes ?? []).map((node) => ({
        id: `entity:${String(node.id)}`,
        label: `Entity ${String(node.name ?? node.id)}`,
        run: () => {
          if (runId) {
            onJumpEntity?.(runId, String(node.name ?? node.id));
          }
        },
      })),
    ];
    const needle = query.trim().toLowerCase();
    return needle ? items.filter((item) => item.label.toLowerCase().includes(needle)) : items;
  }, [analyst, canFleet, graph.data?.nodes, onFleet, onJumpEntity, onKeys, onNewGraph, onSelectRun, onStaff, query, role, runId, runs.data, workspace.data?.perspectives]);
  const visible = actions.slice(0, 20);

  // The highlighted row belongs to the list it was chosen in: a new query or
  // a reopened palette starts at the top.
  const [activeList, setActiveList] = useState({ query, open });
  if (activeList.query !== query || activeList.open !== open) {
    setActiveList({ query, open });
    setActive(0);
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    const onJump = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("flakegraph:jump", onJump);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("flakegraph:jump", onJump);
    };
  }, []);

  function runAction(index: number) {
    const action = visible[index];
    if (!action) {
      return;
    }
    action.run();
    setOpen(false);
    setQuery("");
  }

  return (
    <Dialog open={open} onOpenChange={(next) => {
      setOpen(next);
      if (!next) {
        setQuery("");
      }
    }}>
      <DialogContent className="max-w-lg gap-0 overflow-hidden p-0">
        <DialogHeader className="border-b px-3.5 py-2.5">
          <DialogTitle className="text-sm">Jump</DialogTitle>
          <DialogDescription className="sr-only">
            {runId ? "Jump to a graph, entity, run, or tool." : "Jump to a graph, run, or tool."}
          </DialogDescription>
        </DialogHeader>
        <Input
          autoFocus
          id="jump-input"
          role="combobox"
          aria-label="Command palette"
          aria-expanded={open}
          aria-controls="jump-results"
          aria-activedescendant={visible[active] ? `jump-option-${active}` : undefined}
          className="h-12 rounded-none border-0 border-b shadow-none focus-visible:ring-0"
          placeholder={runId ? "Jump to a graph, entity, run, or tool…" : "Jump to a graph, run, or tool…"}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setActive((current) => Math.min(current + 1, Math.max(visible.length - 1, 0)));
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              setActive((current) => Math.max(current - 1, 0));
            }
            if (event.key === "Enter") {
              event.preventDefault();
              runAction(active);
            }
          }}
        />
        <ul role="listbox" id="jump-results" aria-label="Jump results" className="max-h-80 overflow-auto p-1.5 text-sm">
          {visible.map((action, index) => (
            <li key={action.id}>
              <button
                type="button"
                id={`jump-option-${index}`}
                role="option"
                aria-selected={index === active}
                className={cn(
                  "w-full rounded-md px-2.5 py-1.5 text-left hover:bg-accent",
                  index === active ? "bg-accent" : "",
                )}
                onMouseEnter={() => setActive(index)}
                onClick={() => runAction(index)}
              >
                {action.label}
              </button>
            </li>
          ))}
          {visible.length === 0 ? (
            <li className="px-2.5 py-6 text-center text-sm text-muted-foreground">No matches.</li>
          ) : null}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
