export interface ConsumptionEstimate {
  objectCount: number;
  runtime: string;
  preset: "fast" | "accurate" | "custom";
  usdLow: number;
  usdHigh: number;
  minutesLow: number;
  minutesHigh: number;
  note: string;
}

export function estimateConsumption(options: {
  objectCount: number;
  runtime: string;
  ocrProvider: string;
  parallelism: number;
}): ConsumptionEstimate {
  const objectCount = Math.max(options.objectCount, 1);
  const accurate = !["builtin_text", "built_in_text"].includes(options.ocrProvider);
  const perDoc = accurate ? 0.42 : 0.18;
  const minutes = accurate ? 1.6 : 0.4;
  const runtimeFactor = options.runtime === "snowflake" ? 1.35 : options.runtime === "kubernetes" ? 1.1 : 1;
  const usdLow = round(objectCount * perDoc * 0.7 * runtimeFactor);
  const usdHigh = round(objectCount * perDoc * 1.4 * runtimeFactor);
  const minutesLow = Math.max(1, Math.round(objectCount * minutes * 0.6));
  const minutesHigh = Math.max(minutesLow + 1, Math.round(objectCount * minutes * 1.8));
  return {
    objectCount,
    runtime: options.runtime,
    preset: accurate ? "accurate" : "fast",
    usdLow,
    usdHigh,
    minutesLow,
    minutesHigh,
    note: options.runtime === "local"
      ? "Local work is priced against the hosted reference card, not a bill."
      : "Band from last similar corpus size × this runtime. Not a point estimate.",
  };
}

export function compareEstimateToActual(
  estimate: ConsumptionEstimate | undefined,
  actualUsd: number | null,
): { label: string; delta: number | null } | null {
  if (!estimate) {
    return null;
  }
  if (actualUsd == null) {
    return { label: `Estimate ${estimate.usdLow}–${estimate.usdHigh} usd · actual not recorded`, delta: null };
  }
  const mid = (estimate.usdLow + estimate.usdHigh) / 2;
  const delta = Math.round((actualUsd - mid) * 100) / 100;
  const inside = actualUsd >= estimate.usdLow && actualUsd <= estimate.usdHigh;
  return {
    label: inside
      ? `Actual ${actualUsd} usd sits inside the ${estimate.usdLow}–${estimate.usdHigh} envelope`
      : `Actual ${actualUsd} usd vs envelope ${estimate.usdLow}–${estimate.usdHigh} (${delta > 0 ? "+" : ""}${delta})`,
    delta,
  };
}

export function actualUsdFromConsumption(consumption: unknown): number | null {
  if (!consumption || typeof consumption !== "object") {
    return null;
  }
  const totals = (consumption as { totals?: { billed_usd?: number; avoided_usd?: number; unpriced_calls?: number } }).totals;
  if (!totals || totals.unpriced_calls || (totals.billed_usd === undefined && totals.avoided_usd === undefined)) return null;
  const usd = (totals.billed_usd ?? 0) + (totals.avoided_usd ?? 0);
  return typeof usd === "number" && Number.isFinite(usd) ? usd : null;
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
