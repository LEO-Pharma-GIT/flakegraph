import { MATCHED_TERM_BONUS, MIN_PLURAL_LENGTH, MIN_TOKEN_LENGTH } from "./constants";

/**
 * Words that carry no meaning a search can use: English function words and
 * question words. Kept short on purpose; BM25's inverse document frequency
 * already discounts whatever else is common in a corpus, and a word dropped
 * here can never be searched for. Two-letter words that do carry meaning in a
 * question ("EU", "UK", "IT", "US") are not on it.
 */
const STOP = new Set([
  "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "did", "do", "does", "doing",
  "for", "from", "had", "has", "have", "how", "if", "in", "into", "is", "its", "me", "my", "of", "on", "or",
  "our", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
  "was", "we", "were", "what", "when", "where", "which", "who", "whom", "whose", "why", "will", "with",
  "would", "you", "your",
]);

// A token is a run of letters and digits in any script, so "EU", "Über" and
// "2024" are all tokens and punctuation, apostrophes and symbols separate them.
const TOKEN = /[\p{L}\p{N}]+/gu;
const DIGIT = /\p{N}/u;
const NON_ASCII = /[^\x00-\x7f]/;
const COMBINING_MARKS = /\p{M}+/gu;

/**
 * The search terms of a text, in order and with repeats: case-folded,
 * diacritics folded ("Café" and "cafe" match), stopwords dropped, and English
 * plurals folded to their singular ("studies" and "study" match). Every
 * index and query in Ask goes through this one function, so whatever it folds
 * on one side it folds on the other.
 */
export function terms(text: string): string[] {
  const lower = text.toLowerCase();
  // Most text is ASCII, and folding it is a no-op; only the rest pays for
  // Unicode normalisation.
  const ascii = !NON_ASCII.test(lower);
  const tokens = (ascii ? lower : lower.normalize("NFKD").replace(COMBINING_MARKS, "")).match(TOKEN);
  if (!tokens) {
    return [];
  }
  const out: string[] = [];
  for (const token of tokens) {
    const numeric = ascii ? hasAsciiDigit(token) : DIGIT.test(token);
    if ((token.length < MIN_TOKEN_LENGTH && !numeric) || STOP.has(token)) {
      continue;
    }
    out.push(numeric ? token : singular(token));
  }
  return out;
}

const ZERO = "0".charCodeAt(0);
const NINE = "9".charCodeAt(0);

function hasAsciiDigit(token: string): boolean {
  for (let at = 0; at < token.length; at += 1) {
    const code = token.charCodeAt(at);
    if (code >= ZERO && code <= NINE) {
      return true;
    }
  }
  return false;
}

/** A light plural fold, not a stemmer: it never merges words that differ by more than an ending. */
function singular(token: string): string {
  if (token.length < MIN_PLURAL_LENGTH || !token.endsWith("s")) {
    return token;
  }
  if (token.endsWith("ies") && token.length > MIN_PLURAL_LENGTH) {
    return `${token.slice(0, -3)}y`;
  }
  if (token.endsWith("sses")) {
    return token.slice(0, -2);
  }
  // "class", "status" and "analysis" are singular already.
  if (token.endsWith("ss") || token.endsWith("us") || token.endsWith("is")) {
    return token;
  }
  return token.slice(0, -1);
}

/** A question's search terms. */
export function tokenize(question: string): string[] {
  return terms(question);
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

/**
 * How well a text matches some search terms: for each term the text holds,
 * its occurrences plus a bonus, so matching more distinct terms outweighs
 * repeating one. Terms match whole tokens, never inside a word: "art" does not
 * match "start". `terms` are what `tokenize` returns.
 */
export function scoreHaystack(text: string, queryTerms: string[]): number {
  if (queryTerms.length === 0 || !text) {
    return 0;
  }
  const counts = new Map<string, number>();
  for (const term of terms(text)) {
    counts.set(term, (counts.get(term) ?? 0) + 1);
  }
  return queryTerms.reduce((sum, term) => {
    const occurrences = counts.get(term) ?? 0;
    return occurrences > 0 ? sum + occurrences + MATCHED_TERM_BONUS : sum;
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
