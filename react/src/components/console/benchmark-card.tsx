"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { useState } from "react";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ShowMore } from "@/components/console/show-more";
import type { BenchmarkBaseline, BenchmarkReport } from "@/server/benchmark";

/** Missing entities on screen before "Show more". */
const MISSING_PAGE_SIZE = 12;

/** Each gate's threshold, by the name the gold file gives it. */
const GATE_THRESHOLD: Record<string, { key: string; bound: "≥" | "≤" }> = {
  entity_recall: { key: "entity_recall", bound: "≥" },
  entity_precision: { key: "entity_precision", bound: "≥" },
  triple_recall: { key: "triple_recall", bound: "≥" },
  triple_precision: { key: "triple_precision", bound: "≥" },
  evidence_support: { key: "min_evidence_support", bound: "≥" },
  isolate_ratio: { key: "max_isolate_ratio", bound: "≤" },
  two_hop_recoverability: { key: "min_two_hop_recoverability", bound: "≥" },
};

const GATE_LABEL: Record<string, string> = {
  entity_recall: "Entity recall",
  entity_precision: "Entity precision",
  triple_recall: "Relation recall",
  triple_precision: "Relation precision",
  evidence_support: "Evidence support",
  isolate_ratio: "Isolated entities",
  two_hop_recoverability: "Two-hop recoverability",
  hard_quality_checks: "Hard quality checks",
};

const REASON_LABEL: Record<string, string> = {
  no_edge: "Both ends found, no edge",
  wrong_predicate: "Edge with another type",
  reversed: "Edge the wrong way round",
  source_endpoint_missing: "Source entity missing",
  target_endpoint_missing: "Target entity missing",
  both_endpoints_missing: "Both entities missing",
};

function score(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function humanize(value: string): string {
  const text = value.replaceAll("_", " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/**
 * The graph scored against its gold by the pipeline's own evaluator - the
 * precision, recall and F1 the local benchmarks report - with the gold's
 * acceptance gates and, for a sample pack, the scores its recorded models
 * reached on the same gold.
 */
export function BenchmarkCard({
  benchmark,
  error,
  baselines,
}: {
  benchmark: BenchmarkReport | null;
  error: string | null;
  baselines: readonly BenchmarkBaseline[];
}) {
  if (!benchmark && !error) {
    return null;
  }
  if (!benchmark) {
    return (
      <Card data-testid="benchmark-card">
        <CardHeader>
          <CardTitle>Benchmark</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="warning">The evaluator could not score this graph against its gold: {error}</Alert>
        </CardContent>
      </Card>
    );
  }
  const failed = benchmark.gates.filter((gate) => !gate.passed);
  const reference = benchmark.entities.precision === null || benchmark.triples.precision === null;
  return (
    <Card data-testid="benchmark-card">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>Benchmark</CardTitle>
          {failed.length === 0 ? (
            <Badge variant="success" className="gap-1">
              <CheckCircle2 className="size-3.5" aria-hidden />
              Meets every gate
            </Badge>
          ) : (
            <Badge variant="danger" className="gap-1">
              <XCircle className="size-3.5" aria-hidden />
              Misses {failed.length} of {benchmark.gates.length} gates
            </Badge>
          )}
        </div>
        <CardDescription>
          Scored against <span className="font-medium text-foreground">{benchmark.goldName}</span> by the pipeline&rsquo;s
          evaluator: the numbers <span className="font-mono text-xs">flakegraph inspect evaluate</span> reports.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5 text-sm">
        <div className="grid gap-3 md:grid-cols-2">
          <ScoreGroup
            title="Entities"
            coverage={benchmark.coverage.entities}
            found={benchmark.entities.found}
            expected={benchmark.entities.expected}
            precision={benchmark.entities.precision}
            recall={benchmark.entities.recall}
            f1={benchmark.entities.f1}
            testId="benchmark-entities"
          />
          <ScoreGroup
            title="Relations"
            coverage={benchmark.coverage.relations}
            found={benchmark.triples.found}
            expected={benchmark.triples.expected}
            precision={benchmark.triples.precision}
            recall={benchmark.triples.recall}
            f1={benchmark.triples.f1}
            testId="benchmark-relations"
          />
        </div>
        <dl className="grid grid-cols-3 gap-3">
          <Tile label="Evidence support" value={score(benchmark.evidenceSupport)} hint="Required relations backed by a quote from the source" />
          <Tile label="Two-hop recoverability" value={score(benchmark.twoHopRecoverability)} hint="Gold two-step paths the graph can walk" />
          <Tile label="Information retention" value={score(benchmark.informationRetention)} hint="The mean of entity and relation recall" />
        </dl>
        {reference ? (
          <p className="text-xs text-muted-foreground" data-testid="benchmark-reference-note">
            Precision and F1 read n/a where the gold lists a reference set rather than everything in the corpus: an
            entity or relation the gold does not name cannot be counted as wrong. Recall is scored in full. A gold
            file with <span className="font-mono">&quot;entity_coverage&quot;: &quot;exhaustive&quot;</span> and{" "}
            <span className="font-mono">&quot;relation_coverage&quot;: &quot;exhaustive&quot;</span> gives all three.
          </p>
        ) : null}

        <section className="space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Acceptance gates</h3>
          <ul className="grid gap-1.5 sm:grid-cols-2" data-testid="benchmark-gates">
            {benchmark.gates.map((gate) => {
              const threshold = GATE_THRESHOLD[gate.gate];
              const value = threshold ? benchmark.thresholds[threshold.key] : undefined;
              return (
                <li key={gate.gate} className="flex items-center gap-2">
                  {gate.passed ? (
                    <CheckCircle2 className="size-4 shrink-0 text-emerald-600 dark:text-emerald-400" aria-label="Met" />
                  ) : (
                    <XCircle className="size-4 shrink-0 text-destructive" aria-label="Missed" />
                  )}
                  <span>{GATE_LABEL[gate.gate] ?? humanize(gate.gate)}</span>
                  {threshold && value !== undefined ? (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {threshold.bound} {value}
                    </span>
                  ) : null}
                </li>
              );
            })}
          </ul>
        </section>

        {Object.keys(benchmark.triples.missingByReason).length > 0 ? (
          <section className="space-y-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Why relations were missed</h3>
            <div className="flex flex-wrap gap-1.5" data-testid="benchmark-miss-reasons">
              {Object.entries(benchmark.triples.missingByReason)
                .sort(([, left], [, right]) => right - left)
                .map(([reason, count]) => (
                  <Badge key={reason} variant="outline">
                    {REASON_LABEL[reason] ?? humanize(reason)} · {count.toLocaleString()}
                  </Badge>
                ))}
            </div>
          </section>
        ) : null}

        {benchmark.entities.missing.length > 0 ? <MissingEntities names={benchmark.entities.missing} /> : null}

        {baselines.length > 0 ? <Baselines benchmark={benchmark} baselines={baselines} /> : null}
      </CardContent>
    </Card>
  );
}

function ScoreGroup({
  title,
  coverage,
  found,
  expected,
  precision,
  recall,
  f1,
  testId,
}: {
  title: string;
  coverage: string;
  found: number;
  expected: number;
  precision: number | null;
  recall: number;
  f1: number | null;
  testId: string;
}) {
  return (
    <section className="space-y-2 rounded-md border border-border p-3" data-testid={testId}>
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="font-medium">{title}</h3>
        <span className="text-xs text-muted-foreground">
          {found.toLocaleString()} of {expected.toLocaleString()} found · {coverage} gold
        </span>
      </div>
      <dl className="grid grid-cols-3 gap-2">
        <Tile label="Precision" value={score(precision)} />
        <Tile label="Recall" value={score(recall)} />
        <Tile label="F1" value={score(f1)} />
      </dl>
    </section>
  );
}

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-md bg-muted/40 px-3 py-2" title={hint}>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className={value === "n/a" ? "text-lg text-muted-foreground" : "text-lg font-semibold tabular-nums"}>{value}</dd>
    </div>
  );
}

function MissingEntities({ names }: { names: readonly string[] }) {
  const [limit, setLimit] = useState(MISSING_PAGE_SIZE);
  return (
    <section className="space-y-2" data-testid="benchmark-missing-entities">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Gold entities not found ({names.length.toLocaleString()})
      </h3>
      <div className="flex flex-wrap gap-1.5">
        {names.slice(0, limit).map((name) => (
          <Badge key={name} variant="outline">
            {name}
          </Badge>
        ))}
      </div>
      <ShowMore
        shown={Math.min(limit, names.length)}
        total={names.length}
        pageSize={MISSING_PAGE_SIZE}
        noun="entities"
        onMore={() => setLimit((current) => current + MISSING_PAGE_SIZE)}
      />
    </section>
  );
}

/** This graph beside the pack's recorded results, as a table: the same measures, row by row. */
function Baselines({ benchmark, baselines }: { benchmark: BenchmarkReport; baselines: readonly BenchmarkBaseline[] }) {
  const rows = [
    {
      key: "this",
      name: "This graph",
      detail: null as string | null,
      entityRecall: benchmark.entities.recall as number | null,
      entityF1: benchmark.entities.f1,
      tripleRecall: benchmark.triples.recall as number | null,
      tripleF1: benchmark.triples.f1,
      evidence: benchmark.evidenceSupport,
      twoHop: benchmark.twoHopRecoverability,
      met: benchmark.gates.every((gate) => gate.passed) as boolean | null,
    },
    ...baselines.map((baseline) => ({
      key: baseline.resultId,
      name: baseline.model,
      detail: baseline.measuredAt ? baseline.measuredAt.slice(0, 10) : null,
      entityRecall: baseline.entities.recall,
      entityF1: baseline.entities.f1,
      tripleRecall: baseline.triples.recall,
      tripleF1: baseline.triples.f1,
      evidence: baseline.evidenceSupport,
      twoHop: baseline.twoHopRecoverability,
      met: baseline.thresholdsMet,
    })),
  ];
  return (
    <section className="space-y-2" data-testid="benchmark-baselines">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Recorded results on this gold</h3>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead className="text-right">Entity recall</TableHead>
            <TableHead className="text-right">Entity F1</TableHead>
            <TableHead className="text-right">Relation recall</TableHead>
            <TableHead className="text-right">Relation F1</TableHead>
            <TableHead className="text-right">Evidence</TableHead>
            <TableHead className="text-right">Two-hop</TableHead>
            <TableHead>Gates</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.key} className={row.key === "this" ? "bg-muted/40 font-medium" : undefined}>
              <TableCell>
                <span className="block max-w-56 truncate" title={row.name}>
                  {row.name}
                </span>
                {row.detail ? <span className="text-xs font-normal text-muted-foreground">{row.detail}</span> : null}
              </TableCell>
              {[row.entityRecall, row.entityF1, row.tripleRecall, row.tripleF1, row.evidence, row.twoHop].map((value, index) => (
                <TableCell key={index} className="text-right tabular-nums">
                  {score(value)}
                </TableCell>
              ))}
              <TableCell>{row.met === null ? "n/a" : row.met ? "Met" : "Missed"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}
