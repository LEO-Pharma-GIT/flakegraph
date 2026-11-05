const STOP = new Set([
  "who",
  "what",
  "where",
  "when",
  "whom",
  "which",
  "the",
  "and",
  "for",
  "this",
  "that",
  "with",
  "from",
  "how",
  "did",
  "does",
  "are",
  "was",
  "were",
]);

export function tokenize(question: string): string[] {
  return question
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((term) => term.length > 2 && !STOP.has(term));
}

export function dedupeKeywords(keywords: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of keywords) {
    const keyword = raw.trim();
    if (!keyword) {
      continue;
    }
    const lowered = keyword.toLowerCase();
    if (seen.has(lowered)) {
      continue;
    }
    seen.add(lowered);
    out.push(keyword);
  }
  return out;
}

export function scoreHaystack(text: string, terms: string[]): number {
  if (terms.length === 0) {
    return 0;
  }
  const haystack = text.toLowerCase();
  return terms.reduce((sum, term) => {
    if (!haystack.includes(term)) {
      return sum;
    }
    const occurrences = haystack.split(term).length - 1;
    const exact = haystack === term || haystack.split(/\s+/).includes(term);
    return sum + occurrences + (exact ? 2 : 0);
  }, 0);
}

export function distanceFromScore(score: number): number {
  return 1 / (1 + Math.max(0, score));
}

export function asString(value: unknown): string {
  return value == null ? "" : String(value);
}

export function asNumber(value: unknown, fallback = 0): number {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function asStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).filter(Boolean);
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.map((item) => String(item)).filter(Boolean);
      }
    } catch {
      return value.split(",").map((item) => item.trim()).filter(Boolean);
    }
  }
  return [];
}
