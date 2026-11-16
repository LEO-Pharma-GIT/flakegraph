"use client";

import { useRef, useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { downloadJson } from "@/lib/downloads";
import { ShowMore } from "@/components/console/show-more";

/** The format, shown as the smallest file that means something. */
const EXAMPLE = `{
  "name": "Clinical trials QA",
  "description": "What the trial protocols must yield.",
  "evaluation_scope": { "entity_coverage": "reference", "relation_coverage": "reference" },
  "documents": [{ "id": "protocol-2024.pdf", "path": "protocol-2024.pdf", "title": "Protocol" }],
  "entities": [
    { "id": "acme", "name": "Acme Pharma", "type": "SPONSOR", "aliases": ["Acme"] },
    { "id": "ab-12", "name": "AB-12", "type": "INVESTIGATIONAL_DRUG" }
  ],
  "relations": [
    { "id": "r1", "source": "acme", "target": "ab-12", "relation_type": "SPONSORS", "required": true }
  ]
}`;

export const MISSING_REQUIRED_PAGE_SIZE = 12;

export interface GoldSummary {
  goldName: string;
  expectedEntities: number;
  foundEntities: number;
  requiredTotal: number;
  matchedRequired: number;
  missingRequired: Array<{ id: string; source: string; target: string; relationType: string }>;
}

/**
 * The graph's QA contract: a gold file of the entities and relations it
 * must contain. Where it came from is said (uploaded here, or shipped beside
 * a sample pack), the format is shown rather than described, and a template
 * drawn from the graph itself saves anyone from a blank page.
 */
export function GoldCard({
  runId,
  graphId,
  gold,
  goldSource,
  onChanged,
}: {
  runId: string;
  graphId: string;
  gold: GoldSummary | null;
  goldSource: "uploaded" | "beside-source" | null;
  onChanged: () => Promise<unknown>;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [reading, setReading] = useState(false);
  const utils = trpc.useUtils();
  const upload = trpc.graphs.gold.upload.useMutation({
    onSuccess: async (result) => {
      toast.success(`Gold kept: ${result.entities} entities, ${result.relations} relations.`);
      await onChanged();
    },
    onError: (error) => toast.error(error.message),
  });
  const remove = trpc.graphs.gold.remove.useMutation({
    onSuccess: async () => {
      toast.success("Gold removed. The graph is no longer held to it.");
      await onChanged();
    },
    onError: (error) => toast.error(error.message),
  });

  async function onFile(file: File | undefined) {
    if (!file) {
      return;
    }
    setReading(true);
    try {
      upload.mutate({ graphId, json: await file.text() });
    } finally {
      setReading(false);
      if (input.current) {
        input.current.value = "";
      }
    }
  }

  const [templating, setTemplating] = useState(false);
  async function downloadTemplate() {
    setTemplating(true);
    try {
      const template = await utils.graphs.gold.template.fetch({ runId });
      downloadJson(`${graphId}-gold-template.json`, template);
      toast.success("Template downloaded. Edit it, then upload it here.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not build the template");
    } finally {
      setTemplating(false);
    }
  }

  const busy = reading || upload.isPending;

  return (
    <Card data-testid="gold-card">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Gold standard</CardTitle>
          {gold ? (
            <Badge variant={goldSource === "uploaded" ? "secondary" : "outline"}>
              {goldSource === "uploaded" ? "Uploaded" : "Shipped with the sample pack"}
            </Badge>
          ) : (
            <Badge variant="outline">None yet</Badge>
          )}
        </div>
        <CardDescription>
          A gold file names the entities and relations this graph should contain, and the graph is scored against
          it. It is a QA measure, never context for answers, and never stops the graph being shared.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {gold ? (
          <div className="space-y-2" data-testid="gold-result">
            <p>
              <span className="font-medium">{gold.goldName}</span> · {gold.foundEntities}/{gold.expectedEntities} entities present ·{" "}
              {gold.matchedRequired}/{gold.requiredTotal} required relations found.
            </p>
            {gold.missingRequired.length > 0 ? (
              <MissingRequiredList items={gold.missingRequired} />
            ) : (
              <p className="text-muted-foreground">Every required relation is present.</p>
            )}
          </div>
        ) : (
          <Alert>
            No gold file for this graph. Upload one, or download a template drawn from what the graph already holds and
            edit it down to what must be there.
          </Alert>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={input}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            aria-label="Gold file"
            onChange={(event) => void onFile(event.target.files?.[0])}
          />
          <Button size="sm" variant={gold ? "outline" : "default"} pending={busy} onClick={() => input.current?.click()}>
            {reading ? "Reading…" : upload.isPending ? "Checking against the graph…" : gold && goldSource === "uploaded" ? "Replace gold.json" : "Upload gold.json"}
          </Button>
          <Button size="sm" variant="outline" pending={templating} onClick={() => void downloadTemplate()}>
            {templating ? "Building template…" : "Download a template from this graph"}
          </Button>
          {goldSource === "uploaded" ? (
            <Button size="sm" variant="ghost" pending={remove.isPending} onClick={() => remove.mutate({ graphId })}>
              {remove.isPending ? "Removing…" : "Remove"}
            </Button>
          ) : null}
        </div>
        <details className="rounded-md border border-border" data-testid="gold-format">
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">The format</summary>
          <div className="space-y-2 border-t border-border p-3">
            <p className="text-muted-foreground">
              One JSON object with a <span className="font-mono text-xs">name</span> and a{" "}
              <span className="font-mono text-xs">description</span>. <span className="font-mono text-xs">entities</span> need an id, a name and a type;
              aliases help matching. <span className="font-mono text-xs">relations</span> point at entity ids by{" "}
              <span className="font-mono text-xs">source</span> and <span className="font-mono text-xs">target</span>;{" "}
              <span className="font-mono text-xs">required</span> defaults to true, and only required relations gate
              the graph. <span className="font-mono text-xs">documents</span> are optional; each entry needs an id and source path.
            </p>
            <p className="text-muted-foreground">
              <span className="font-mono text-xs">evaluation_scope</span> says what the file covers.{" "}
              <span className="font-mono text-xs">reference</span> is a selection: the graph is scored on recall, and
              anything else it holds is not counted against it. <span className="font-mono text-xs">exhaustive</span>{" "}
              means every entity or relation in the corpus is listed, so extra ones count as wrong and precision and F1
              are scored too. Left out, both are exhaustive.
            </p>
            <pre className="overflow-x-auto rounded-md bg-muted/50 p-3 font-mono text-xs leading-relaxed">{EXAMPLE}</pre>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}

export function MissingRequiredList({ items }: { items: GoldSummary["missingRequired"] }) {
  const [limit, setLimit] = useState(MISSING_REQUIRED_PAGE_SIZE);
  const page = items.slice(0, limit);
  return (
    <div className="space-y-2" data-testid="missing-required">
      <p className="text-destructive">
        {items.length.toLocaleString()} required relation{items.length === 1 ? " is" : "s are"} missing.
      </p>
      <ul className="list-disc space-y-1 pl-5">
        {page.map((item) => (
          <li key={item.id}>
            {item.source} —{item.relationType}→ {item.target}
          </li>
        ))}
      </ul>
      <ShowMore
        shown={page.length}
        total={items.length}
        pageSize={MISSING_REQUIRED_PAGE_SIZE}
        noun="missing relations"
        onMore={() => setLimit((current) => current + MISSING_REQUIRED_PAGE_SIZE)}
      />
    </div>
  );
}
