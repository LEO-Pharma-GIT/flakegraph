"use client";

import { useState } from "react";
import { Check, Lock, Sparkles, X } from "lucide-react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { OntologySelection, OntologyTerm } from "@/server/protocol/schema";
import { folderEntityProblem } from "@/lib/folder-entity";
import { normalizeTermName } from "@/lib/ontology-terms";
import { cn } from "@/lib/utils";


export const RELATION_MODES: Array<{ id: OntologySelection["relations"]; label: string; hint: string }> = [
  {
    id: "guided",
    label: "Guided by this list",
    hint: "The extractor prefers these predicates and may add one the text plainly states.",
  },
  { id: "fixed", label: "Only this list", hint: "Every relation is one of these; anything else is dropped." },
  {
    id: "open",
    label: "Detected automatically",
    hint: "No list. The extractor names each relation from the text (LOCATED_IN, TRAINED_UNDER…).",
  },
];


/** Why a selection cannot be built with, or null when it can. */
export function ontologyProblem(selection: OntologySelection): string | null {
  if (selection.entityTypes.length === 0) {
    return "Add at least one entity type.";
  }
  if (selection.relations !== "open" && selection.relationTypes.length === 0) {
    return "Add at least one relation type, or let relations be detected automatically.";
  }
  return folderEntityProblem(selection);
}

export function OntologyEditor({
  value,
  onChange,
  defaults,
  locked = null,
}: {
  value: OntologySelection;
  onChange: (next: OntologySelection) => void;
  /** What the runtime extracts unless changed here, and where that comes from. */
  defaults: { selection: OntologySelection; source: string } | null;
  /** When set, the vocabulary is shown but not editable, for this reason. */
  locked?: string | null;
}) {
  const [intent, setIntent] = useState("");
  const [dismissedProposal, setDismissedProposal] = useState(false);
  const propose = trpc.ingestion.ontology.useMutation({
    onError: (error) => toast.error(error.message),
    onSuccess: () => setDismissedProposal(false),
  });
  const proposal = dismissedProposal ? undefined : propose.data;
  const problem = ontologyProblem(value);
  const differsFromDefault =
    defaults !== null && JSON.stringify(defaults.selection) !== JSON.stringify(value);
  const mode = RELATION_MODES.find((item) => item.id === value.relations) ?? RELATION_MODES[0];

  function proposedTerms(names: string[]): OntologyTerm[] {
    return names.map((name) => ({ name, description: proposal?.descriptions[name] ?? "" }));
  }

  function merge(existing: readonly OntologyTerm[], added: OntologyTerm[]): OntologyTerm[] {
    const seen = new Set(existing.map((term) => term.name));
    return [...existing, ...added.filter((term) => !seen.has(term.name) && seen.add(term.name))];
  }

  return (
    <Card data-testid="ontology-editor">
      <CardHeader className="space-y-2">
        <CardTitle>What to extract</CardTitle>
        <CardDescription>
          {locked
            ? locked
            : "Entity types name what the extractor looks for; relations say how they connect. They apply to this graph only."}
        </CardDescription>
        {locked ? (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline" className="gap-1">
              <Lock className="size-3" aria-hidden="true" />
              Fixed
            </Badge>
            <span>The vocabulary of the base version.</span>
          </div>
        ) : defaults ? (
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            {differsFromDefault ? (
              <>
                <Badge variant="warning">Edited</Badge>
                <span>Changed from {defaults.source}.</span>
                <Button size="sm" variant="ghost" className="h-6 px-1.5 text-xs" onClick={() => onChange(defaults.selection)}>
                  Reset to {defaults.source}
                </Button>
              </>
            ) : (
              <>
                <Badge variant="secondary">Default</Badge>
                <span>These are the types from {defaults.source}.</span>
              </>
            )}
          </div>
        ) : null}
      </CardHeader>
      <CardContent className="space-y-6">
        {locked ? null : (
          <section className="space-y-3 rounded-lg border border-dashed border-border bg-muted/20 p-3" aria-label="Suggest types">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Sparkles className="hidden size-4 shrink-0 text-muted-foreground sm:block" aria-hidden="true" />
              <Input
                aria-label="Describe the graph you want"
                className="h-9 flex-1 bg-background"
                placeholder="Describe the graph, e.g. people, schools and techniques in these martial-arts histories"
                value={intent}
                onChange={(event) => setIntent(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && intent.trim().length >= 8 && !propose.isPending) {
                    event.preventDefault();
                    propose.mutate({ intent });
                  }
                }}
              />
              <Button
                size="sm"
                className="h-9"
                onClick={() => propose.mutate({ intent })}
                disabled={intent.trim().length < 8 || propose.isPending}
              >
                {propose.isPending ? "Suggesting…" : "Suggest types"}
              </Button>
            </div>
            {proposal ? (
              <div className="space-y-3 rounded-md border border-border bg-background p-3" data-testid="ontology-proposal">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <p className="text-sm">
                    <span className="font-medium">
                      {proposal.source === "model" ? "Suggested by the model" : "Suggested from your words"}
                    </span>
                    <span className="text-muted-foreground">
                      {proposal.source === "model"
                        ? " from your description. Types already in your lists are shown plain; new ones are marked."
                        : proposal.modelFailure
                          ? ` — the model did not answer (${proposal.modelFailure}), so these are the nouns of your description.`
                          : " — no model is configured for the console, so these are the nouns of your description."}
                    </span>
                  </p>
                  <button
                    type="button"
                    aria-label="Dismiss suggestion"
                    className="rounded-sm text-muted-foreground hover:text-foreground"
                    onClick={() => setDismissedProposal(true)}
                  >
                    <X className="size-4" aria-hidden="true" />
                  </button>
                </div>
                <ProposalRow label="Entity types" terms={proposedTerms(proposal.types)} existing={value.entityTypes} />
                <ProposalRow label="Relations" terms={proposedTerms(proposal.relations)} existing={value.relationTypes} />
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() =>
                      onChange({
                        ...value,
                        entityTypes: proposedTerms(proposal.types),
                        relationTypes: proposedTerms(proposal.relations),
                        relations: value.relations === "open" ? "guided" : value.relations,
                      })
                    }
                  >
                    Replace with suggestion
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      onChange({
                        ...value,
                        entityTypes: merge(value.entityTypes, proposedTerms(proposal.types)),
                        relationTypes: merge(value.relationTypes, proposedTerms(proposal.relations)),
                      })
                    }
                  >
                    Add to mine
                  </Button>
                </div>
              </div>
            ) : null}
          </section>
        )}

        <TermListEditor
          label="Entity types"
          testId="entity-types"
          terms={value.entityTypes}
          placeholder="Add type…"
          example="TECHNIQUE"
          locked={Boolean(locked)}
          onChange={(entityTypes) => onChange({ ...value, entityTypes })}
        />

        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <SectionLabel>Relations</SectionLabel>
            <div
              className={cn("inline-flex rounded-md border border-border bg-muted/40 p-0.5", locked ? "opacity-60" : "")}
              role="radiogroup"
              aria-label="Relation typing"
            >
              {RELATION_MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="radio"
                  aria-checked={value.relations === item.id}
                  disabled={Boolean(locked)}
                  onClick={() => onChange({ ...value, relations: item.id })}
                  className={cn(
                    "rounded-[5px] px-2.5 py-1 text-xs font-medium transition-colors disabled:cursor-default",
                    value.relations === item.id
                      ? "bg-background text-foreground shadow-sm ring-1 ring-border"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <p className="text-xs text-muted-foreground">{mode.hint}</p>
          {value.relations === "open" ? null : (
            <TermListEditor
              testId="relation-types"
              terms={value.relationTypes}
              placeholder="Add relation…"
              example="TRAINED_UNDER"
              locked={Boolean(locked)}
              onChange={(relationTypes) => onChange({ ...value, relationTypes })}
            />
          )}
        </div>

        {problem && !locked ? <Alert variant="destructive">{problem}</Alert> : null}
      </CardContent>
    </Card>
  );
}

function SectionLabel({ children, count }: { children: string; count?: number }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {children}
      {count !== undefined ? <span className="ml-1.5 font-normal normal-case tracking-normal">{count}</span> : null}
    </p>
  );
}

function ProposalRow({
  label,
  terms,
  existing,
}: {
  label: string;
  terms: OntologyTerm[];
  existing: readonly OntologyTerm[];
}) {
  const held = new Set(existing.map((term) => term.name));
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-1 text-xs text-muted-foreground">{label}</span>
      {terms.length ? (
        terms.map((term) => (
          <span
            key={term.name}
            title={term.description || undefined}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-xs",
              held.has(term.name)
                ? "border-border bg-muted/40 text-muted-foreground"
                : "border-primary/40 bg-accent text-foreground",
            )}
          >
            {term.name}
            {held.has(term.name) ? null : <span className="font-sans text-[10px] uppercase text-primary">new</span>}
          </span>
        ))
      ) : (
        <span className="text-xs text-muted-foreground">none</span>
      )}
    </div>
  );
}

function TermListEditor({
  label,
  testId,
  terms,
  placeholder,
  example,
  locked,
  onChange,
}: {
  /** Shown above the list; omitted when the surrounding section names it. */
  label?: string;
  testId: string;
  terms: readonly OntologyTerm[];
  placeholder: string;
  example: string;
  locked: boolean;
  onChange: (terms: OntologyTerm[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [description, setDescription] = useState("");
  const addLabel = testId === "entity-types" ? "Add entity type" : "Add relation type";

  function add() {
    const name = normalizeTermName(draft);
    if (!name) {
      toast.error(`A type name is letters, digits and underscores, like ${example}.`);
      return;
    }
    if (terms.some((term) => term.name === name)) {
      toast.error(`${name} is already listed.`);
      return;
    }
    onChange([...terms, { name, description: "" }]);
    setDraft("");
  }

  const current = editing ? terms.find((term) => term.name === editing) : undefined;

  return (
    <div className="space-y-2" data-testid={testId}>
      {label ? <SectionLabel count={terms.length}>{label}</SectionLabel> : null}
      <div className="flex flex-wrap items-center gap-1.5">
        {terms.length === 0 && locked ? <p className="text-xs text-muted-foreground">None.</p> : null}
        {terms.map((term) => (
          <span
            key={term.name}
            className={cn(
              "inline-flex h-7 items-center gap-1 rounded-md border pl-2 pr-1 font-mono text-xs transition-colors",
              editing === term.name ? "border-primary bg-accent" : "border-border bg-background",
              locked ? "pr-2" : "",
            )}
          >
            <button
              type="button"
              title={term.description || (locked ? undefined : "No description yet — click to add one")}
              className="disabled:cursor-default"
              disabled={locked}
              onClick={() => {
                setEditing(editing === term.name ? null : term.name);
                setDescription(term.description);
              }}
            >
              {term.name}
            </button>
            {locked ? null : (
              <button
                type="button"
                aria-label={`Remove ${term.name}`}
                className="rounded-sm p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground"
                onClick={() => {
                  onChange(terms.filter((item) => item.name !== term.name));
                  if (editing === term.name) {
                    setEditing(null);
                  }
                }}
              >
                <X className="size-3" aria-hidden="true" />
              </button>
            )}
          </span>
        ))}
        {locked ? null : (
          <span className="inline-flex items-center gap-1">
            <Input
              aria-label={addLabel}
              className={cn(
                "h-7 w-36 border-dashed bg-transparent px-2 font-mono text-xs placeholder:font-sans",
                draft ? "w-48 border-solid bg-background" : "",
              )}
              placeholder={placeholder}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  add();
                }
                if (event.key === "Escape") {
                  setDraft("");
                }
              }}
            />
            {draft.trim() ? (
              <Button size="sm" variant="outline" className="h-7 px-2" onClick={add}>
                Add
              </Button>
            ) : null}
          </span>
        )}
      </div>
      {current && !locked ? (
        <div className="flex flex-col gap-2 rounded-md border border-border bg-muted/30 p-2 sm:flex-row sm:items-center">
          <span className="px-1 font-mono text-xs">{current.name}</span>
          <Input
            aria-label={`Description of ${current.name}`}
            className="h-8 flex-1 bg-background"
            placeholder="What qualifies, in one sentence"
            value={description}
            autoFocus
            onChange={(event) => setDescription(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                onChange(terms.map((term) => (term.name === current.name ? { ...term, description: description.trim() } : term)));
                setEditing(null);
              }
              if (event.key === "Escape") {
                setEditing(null);
              }
            }}
          />
          <div className="flex gap-1">
            <Button
              size="sm"
              variant="secondary"
              className="h-8"
              onClick={() => {
                onChange(terms.map((term) => (term.name === current.name ? { ...term, description: description.trim() } : term)));
                setEditing(null);
              }}
            >
              <Check className="mr-1 size-3.5" aria-hidden="true" />
              Save
            </Button>
            <Button size="sm" variant="ghost" className="h-8" onClick={() => setEditing(null)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
