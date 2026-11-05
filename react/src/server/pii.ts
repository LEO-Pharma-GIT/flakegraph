import { readFile } from "node:fs/promises";
import type { SourceObject } from "./protocol/schema";

export interface PiiHit {
  source: string;
  kind: "email" | "ssn" | "passport" | "national_id" | "card";
  snippet: string;
}

const PATTERNS: Array<{ kind: PiiHit["kind"]; regex: RegExp }> = [
  { kind: "email", regex: /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/i },
  { kind: "ssn", regex: /\b\d{3}-\d{2}-\d{4}\b/ },
  { kind: "passport", regex: /\bpassport[:\s-]*[A-Z0-9]{6,9}\b/i },
  { kind: "national_id", regex: /\b(national id|nid)[:\s-]*[A-Z0-9]{6,}\b/i },
  { kind: "card", regex: /\b(?:\d{4}[- ]?){3}\d{4}\b/ },
];

export async function scanPii(objects: readonly SourceObject[]): Promise<PiiHit[]> {
  const hits: PiiHit[] = [];
  for (const object of objects.slice(0, 40)) {
    collectHits(object.name, object.uri, hits);
    const filePath = filePathFromUri(object.uri);
    if (!filePath || !isTextSource(filePath, object.name)) {
      continue;
    }
    try {
      const text = await readFile(filePath, "utf8");
      collectHits(text.slice(0, 4000), object.uri, hits);
    } catch {
      // Object listing is enough when the file is not on this host.
    }
  }
  return uniqueHits(hits);
}

export function scanText(text: string, source: string): PiiHit[] {
  const hits: PiiHit[] = [];
  collectHits(text, source, hits);
  return uniqueHits(hits);
}

function collectHits(text: string, source: string, hits: PiiHit[]): void {
  for (const pattern of PATTERNS) {
    const match = text.match(pattern.regex);
    if (match) {
      hits.push({ source, kind: pattern.kind, snippet: match[0] });
    }
  }
}

function uniqueHits(hits: PiiHit[]): PiiHit[] {
  const seen = new Set<string>();
  return hits.filter((hit) => {
    const key = `${hit.source}:${hit.kind}:${hit.snippet}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function filePathFromUri(uri: string): string | null {
  if (uri.startsWith("file://")) {
    return decodeURIComponent(uri.replace(/^file:\/\//, "").replace(/^\/([A-Za-z]:)/, "$1"));
  }
  if (uri.startsWith("/") || uri.startsWith("data/")) {
    return uri;
  }
  return null;
}

const TEXT_SUFFIXES = new Set([".md", ".txt", ".json", ".csv", ".html", ".xml", ".yml", ".yaml"]);

function isTextSource(filePath: string, name: string): boolean {
  const ext = filePath.slice(filePath.lastIndexOf(".")).toLowerCase() || name.slice(name.lastIndexOf(".")).toLowerCase();
  return TEXT_SUFFIXES.has(ext);
}
