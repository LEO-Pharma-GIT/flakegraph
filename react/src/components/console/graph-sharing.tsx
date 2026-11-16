"use client";

import { Lock, LogOut, UserPlus, Users, X } from "lucide-react";
import { useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { toast } from "sonner";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { trpc } from "@/components/providers";
import { cn, formatRelativeTime } from "@/lib/utils";

type Level = "read" | "write";
type Role = "owner" | Level;

export const LEVELS: Record<Level, { label: string; summary: string }> = {
  read: { label: "Read", summary: "Open, explore, ask and export." },
  write: {
    label: "Write",
    summary: "Also rename, revise, retry or cancel runs, review, and edit gold and perspectives.",
  },
};

/** An address in the case people write it; a bare demo name as it is. */
export function displayPrincipal(principal: string | null | undefined): string {
  const value = (principal ?? "").trim();
  return value.includes("@") ? value.toLowerCase() : value;
}

function initials(principal: string): string {
  const name = displayPrincipal(principal).split("@")[0] ?? "";
  const parts = name.split(/[._-]+/).filter(Boolean);
  const letters = parts.length > 1 ? `${parts[0]![0]}${parts[1]![0]}` : name.slice(0, 2);
  return letters.toUpperCase() || "?";
}

/**
 * Who can use a graph, and — for its owner — the controls to change that.
 *
 * Owners add people by sign-in address at Read or Write, change a person's
 * level in place, and remove access. Everyone else sees who owns the graph,
 * their own level, and can leave. What each level allows is stated where the
 * choice is made.
 */
export function GraphSharing({
  graphId,
  graphName,
  viewer,
  onLeft,
}: {
  graphId: string;
  graphName: string;
  viewer: string;
  onLeft?: () => void | Promise<void>;
}) {
  const utils = trpc.useUtils();
  const access = trpc.graphs.access.useQuery({ graphId });
  const session = trpc.auth.session.useQuery();
  const me = viewer.trim().toUpperCase();
  const isOwner = access.data?.role === "owner";
  const people = trpc.graphs.people.useQuery(undefined, { enabled: isOwner && Boolean(me) });
  // A removal in flight, and whether the leave confirmation is open.
  const [removing, setRemoving] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);

  const refresh = async () => {
    await Promise.all([access.refetch(), utils.runs.list.invalidate(), utils.runs.get.invalidate()]);
  };
  // Failures are reported where each action was taken: inline under Add
  // people, as a toast for a level change or a removal.
  const share = trpc.graphs.share.useMutation({ onSuccess: refresh, onError: () => undefined });
  const unshare = trpc.graphs.unshare.useMutation({ onError: () => undefined });
  // The level change in flight, shown at once rather than after the refetch.
  const changing = share.isPending && share.variables ? share.variables : null;

  if (access.isPending) {
    return <p className="text-sm text-muted-foreground">Loading who can use this graph…</p>;
  }
  if (access.error || !access.data) {
    return <Alert variant="destructive">{access.error?.message ?? "Sharing is unavailable."}</Alert>;
  }

  const { owner, role, shares } = access.data;
  const mine = shares.find((item) => item.principal === me);
  const taken = new Set([...(owner ? [owner] : []), ...shares.map((item) => item.principal)]);

  /** Leave a graph shared with you: asked first, since only the owner can undo it. */
  async function leave() {
    try {
      await unshare.mutateAsync({ graphId, principal: me });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not leave the graph");
      return;
    }
    setRemoving(null);
    toast.success(`You left ${graphName}`);
    await onLeft?.();
  }

  /** Take someone's access away at once, with Undo: it is the owner's to give back. */
  async function removePerson(principal: string, level: Level) {
    const who = displayPrincipal(principal);
    setRemoving(principal);
    try {
      await unshare.mutateAsync({ graphId, principal });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : `Could not remove ${who}`);
      return;
    } finally {
      setRemoving(null);
    }
    await refresh();
    toast.success(`${who} no longer has access`, {
      action: {
        label: "Undo",
        onClick: () =>
          share.mutate(
            { graphId, principal, level },
            {
              onSuccess: () => toast.success(`${who} has ${LEVELS[level].label} access again`),
              onError: (error) => toast.error(error.message),
            },
          ),
      },
      duration: 10_000,
    });
  }

  function changeLevel(principal: string, level: Level) {
    share.mutate(
      { graphId, principal, level },
      {
        onSuccess: () => toast.success(`${displayPrincipal(principal)} now has ${LEVELS[level].label} access`),
        onError: (error) => toast.error(error.message),
      },
    );
  }

  return (
    <Card data-testid="graph-sharing">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Sharing</CardTitle>
          <AccessBadge role={role} sharedWith={shares.length} />
        </div>
        <CardDescription>
          Only the owner and the people listed here can see this graph. The owner is the only one who can share it or
          delete it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5 text-sm">
        <dl className="grid gap-2 rounded-md border border-border p-3 sm:grid-cols-2" data-testid="sharing-levels">
          {(Object.keys(LEVELS) as Level[]).map((level) => (
            <div key={level}>
              <dt className="font-medium">{LEVELS[level].label}</dt>
              <dd className="text-muted-foreground">{LEVELS[level].summary}</dd>
            </div>
          ))}
        </dl>

        {isOwner ? (
          !me ? (
            <Alert data-testid="sharing-sign-in">
              Sign in to share this graph: a share records who granted it, so it needs a name.
            </Alert>
          ) : (
            <AddPeople
              placeholder={session.data?.snowflakeHosted ? "User name, or role:NAME for a whole role" : "name@example.com"}
              suggestions={(people.data ?? []).filter((principal) => !taken.has(principal))}
              onShare={async (principal, level) => {
                const view = await share.mutateAsync({ graphId, principal, level });
                const added = view.shares.find((item) => item.principal === principal.trim().toUpperCase());
                toast.success(`Shared with ${displayPrincipal(added?.principal ?? principal)} · ${LEVELS[level].label}`);
              }}
            />
          )
        ) : (
          <Alert className="flex items-start justify-between gap-3" data-testid="sharing-mine">
            <span>
              You have <span className="font-medium">{LEVELS[(mine?.level ?? role) as Level]?.label ?? role}</span> access,
              shared by {displayPrincipal(owner) || "the owner"}. Ask them to change your level.
            </span>
            <Button size="sm" variant="outline" onClick={() => setLeaving(true)}>
              <LogOut className="size-3.5" aria-hidden />
              Leave
            </Button>
          </Alert>
        )}

        <div className="space-y-2">
          <h3 className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">People with access</h3>
          <ul className="divide-y divide-border rounded-md border border-border" data-testid="sharing-people">
            <PersonRow principal={owner ?? "Unowned"} you={owner === me} detail="Created the graph">
              <Badge variant="secondary">Owner</Badge>
            </PersonRow>
            {shares.map((item) => (
              <PersonRow
                key={item.principal}
                principal={item.principal}
                you={item.principal === me}
                detail={`Added by ${item.grantedBy === "SYSTEM" ? "an administrator" : displayPrincipal(item.grantedBy)} · ${formatRelativeTime(item.grantedAt)}`}
              >
                {isOwner ? (
                  <>
                    {changing?.principal === item.principal ? <Spinner label="Saving the new level" /> : null}
                    <Select
                      value={changing?.principal === item.principal ? changing.level : item.level}
                      disabled={removing === item.principal}
                      onValueChange={(value) => changeLevel(item.principal, value as Level)}
                    >
                      <SelectTrigger className="h-8 w-24" aria-label={`Access for ${displayPrincipal(item.principal)}`}>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="read">Read</SelectItem>
                        <SelectItem value="write">Write</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-8"
                      aria-label={`Remove ${displayPrincipal(item.principal)}`}
                      pending={removing === item.principal}
                      onClick={() => void removePerson(item.principal, item.level)}
                    >
                      <X className="size-4" aria-hidden />
                    </Button>
                  </>
                ) : (
                  <Badge variant="outline">{LEVELS[item.level].label}</Badge>
                )}
              </PersonRow>
            ))}
          </ul>
          {shares.length === 0 ? (
            <p className="flex items-center gap-1.5 text-muted-foreground" data-testid="sharing-private">
              <Lock className="size-3.5" aria-hidden />
              Not shared with anyone. Only {owner === me ? "you" : displayPrincipal(owner)} can see this graph.
            </p>
          ) : null}
        </div>
      </CardContent>

      <Dialog open={leaving} onOpenChange={(open) => (unshare.isPending ? undefined : setLeaving(open))}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Leave {graphName}?</DialogTitle>
            <DialogDescription>
              The graph disappears from your list. Only its owner can give you access again.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap justify-end gap-2">
            <Button variant="outline" disabled={unshare.isPending} onClick={() => setLeaving(false)}>
              Keep access
            </Button>
            <Button variant="destructive" pending={unshare.isPending} onClick={() => void leave()}>
              {unshare.isPending ? "Leaving…" : "Leave graph"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

function PersonRow({
  principal,
  you,
  detail,
  children,
}: {
  principal: string;
  you: boolean;
  detail: string;
  children: ReactNode;
}) {
  return (
    <li className="flex items-center gap-3 px-3 py-2" data-principal={principal}>
      <span
        aria-hidden
        className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium text-muted-foreground"
      >
        {initials(principal)}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">
          {displayPrincipal(principal)}
          {you ? <span className="font-normal text-muted-foreground"> (you)</span> : null}
        </span>
        <span className="block truncate text-xs text-muted-foreground">{detail}</span>
      </span>
      <span className="flex shrink-0 items-center gap-1.5">{children}</span>
    </li>
  );
}

/** Add a person: an address field that suggests people who use the console, a level, and Share. */
function AddPeople({
  placeholder,
  suggestions,
  onShare,
}: {
  placeholder: string;
  suggestions: readonly string[];
  onShare: (principal: string, level: Level) => Promise<void>;
}) {
  const [pending, setPending] = useState(false);
  const [value, setValue] = useState("");
  const [level, setLevel] = useState<Level>("read");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const listId = useId();
  const input = useRef<HTMLInputElement>(null);
  const matches = useMemo(() => {
    const query = value.trim().toLowerCase();
    return suggestions.filter((principal) => principal.toLowerCase().includes(query)).slice(0, 8);
  }, [suggestions, value]);
  const showList = open && matches.length > 0 && !(matches.length === 1 && matches[0]?.toLowerCase() === value.trim().toLowerCase());

  async function submit(principal = value) {
    const target = principal.trim();
    if (!target || pending) {
      return;
    }
    setError(null);
    setPending(true);
    try {
      await onShare(target, level);
      setValue("");
      setOpen(false);
      input.current?.focus();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPending(false);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown" && matches.length) {
      event.preventDefault();
      setOpen(true);
      setActive((index) => (index + 1) % matches.length);
    } else if (event.key === "ArrowUp" && matches.length) {
      event.preventDefault();
      setActive((index) => (index - 1 + matches.length) % matches.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (showList && matches[active]) {
        setValue(displayPrincipal(matches[active]));
        setOpen(false);
      } else {
        void submit();
      }
    } else if (event.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="space-y-1.5" data-testid="sharing-add">
      <label htmlFor={`${listId}-input`} className="text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
        Add people
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <div className="relative flex-1">
          <UserPlus className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input
            id={`${listId}-input`}
            ref={input}
            role="combobox"
            aria-expanded={showList}
            aria-controls={`${listId}-list`}
            aria-activedescendant={showList ? `${listId}-option-${active}` : undefined}
            aria-autocomplete="list"
            aria-label="Person to share with"
            autoComplete="off"
            placeholder={placeholder}
            className="pl-8"
            value={value}
            onChange={(event) => {
              setValue(event.target.value);
              setActive(0);
              setOpen(true);
              setError(null);
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 120)}
            onKeyDown={onKeyDown}
          />
          {showList ? (
            <ul
              id={`${listId}-list`}
              role="listbox"
              className="absolute z-20 mt-1 max-h-64 w-full overflow-y-auto rounded-md border border-border bg-popover p-1 shadow-md"
            >
              {matches.map((principal, index) => (
                <li
                  key={principal}
                  id={`${listId}-option-${index}`}
                  role="option"
                  aria-selected={index === active}
                  className={cn(
                    "flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5",
                    index === active ? "bg-accent text-accent-foreground" : "hover:bg-accent/60",
                  )}
                  onMouseDown={(event) => {
                    event.preventDefault();
                    setValue(displayPrincipal(principal));
                    setOpen(false);
                  }}
                >
                  <span className="flex size-6 items-center justify-center rounded-full bg-muted text-[10px] font-medium">
                    {initials(principal)}
                  </span>
                  {displayPrincipal(principal)}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <Select value={level} onValueChange={(next) => setLevel(next as Level)}>
          <SelectTrigger className="sm:w-28" aria-label="Access level">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="read">Read</SelectItem>
            <SelectItem value="write">Write</SelectItem>
          </SelectContent>
        </Select>
        <Button pending={pending} disabled={!value.trim()} onClick={() => void submit()}>
          {pending ? "Sharing…" : "Share"}
        </Button>
      </div>
      {error ? (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      ) : (
        <p className="text-xs text-muted-foreground">
          People you share with see this graph in their list under Shared. Suggestions are people who have signed in to
          FlakeGraph.
        </p>
      )}
    </div>
  );
}

/** Private, shared by you, or shared with you: the graph's standing at a glance. */
export function AccessBadge({ role, sharedWith }: { role: Role | null | undefined; sharedWith: number | null | undefined }) {
  if (!role) {
    return null;
  }
  if (role !== "owner") {
    return (
      <Badge variant="outline" className="gap-1" data-testid="access-badge">
        <Users className="size-3" aria-hidden />
        Shared with you · {LEVELS[role].label}
      </Badge>
    );
  }
  if (sharedWith) {
    return (
      <Badge variant="outline" className="gap-1" data-testid="access-badge">
        <Users className="size-3" aria-hidden />
        Shared with {sharedWith} {sharedWith === 1 ? "person" : "people"}
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1" data-testid="access-badge">
      <Lock className="size-3" aria-hidden />
      Private
    </Badge>
  );
}
