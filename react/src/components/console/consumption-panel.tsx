"use client";

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { actualUsdFromConsumption, compareEstimateToActual, type ConsumptionEstimate } from "@/server/estimate";

/** What a run reports about its own spend: the per-call events and the estimate they are held against. */
export function consumptionExport(consumption: unknown, estimate: ConsumptionEstimate | null): unknown {
  return { estimate, actual: consumption ?? null };
}

export function ConsumptionPanel({
  consumption,
  estimate,
}: {
  consumption: unknown;
  estimate: ConsumptionEstimate | null;
}) {
  if (!consumption || typeof consumption !== "object") {
    return (
      <p className="text-sm text-muted-foreground">
        This graph reports no recorded usage, so there are no tokens, pages or cost to show.
      </p>
    );
  }
  const record = consumption as {
    totals?: { billed_usd?: number; avoided_usd?: number; unpriced_calls?: number };
    events?: Array<Record<string, unknown>>;
  };
  const usd = record.totals?.billed_usd;
  const avoided = record.totals?.avoided_usd;
  const unpriced = record.totals?.unpriced_calls ?? 0;
  const comparison = compareEstimateToActual(estimate ?? undefined, actualUsdFromConsumption(consumption));
  return (
    <div className="space-y-3">
      <p className="text-sm" data-testid="estimate-vs-actual">
        {usd != null ? `Billed cost ${usd} usd` : "Usage measured, cost not priced"}
        {avoided != null ? ` · Avoided cost ${avoided} usd` : ""}
        {unpriced ? ` · ${unpriced} unpriced calls` : ""}
        {comparison ? ` · ${comparison.label}` : ""}
      </p>
      {(record.events ?? []).length === 0 ? (
        <p className="text-sm text-muted-foreground">No per-call events were recorded for this graph.</p>
      ) : (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Stage</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>Tokens</TableHead>
            <TableHead>Locality</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {(record.events ?? []).map((event, index) => (
            <TableRow key={index}>
              <TableCell>{String(event.stage ?? "")}</TableCell>
              <TableCell>{String(event.model ?? event.provider ?? "")}</TableCell>
              <TableCell>{String((event.usage as { total_tokens?: number } | undefined)?.total_tokens ?? "")}</TableCell>
              <TableCell>{String(event.locality ?? "")}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      )}
    </div>
  );
}
