import { createHash, timingSafeEqual } from "node:crypto";

export function hashApiSecret(secret: string): string {
  return createHash("sha256").update(secret, "utf8").digest("hex");
}

/**
 * The console API key a request presents, if any. A bearer credential counts
 * only when it has a key's `fg_` shape: any other bearer is a token the
 * sign-in gate verified, and the request is the person the gate names.
 */
export function presentedApiSecret(headers: Headers): string | null {
  const authorization = headers.get("authorization")?.trim();
  if (authorization) {
    const match = /^(?:Bearer|Token)\s+(fg_\S+)/i.exec(authorization);
    if (match?.[1]) {
      return match[1];
    }
  }
  return headers.get("x-flakegraph-api-key")?.trim() || null;
}

/**
 * What a request presented to be admitted besides a browser's cookie: a
 * console key, or any other credential in `Authorization` - on a fleet, a
 * person's access token that an application holds and the gate verified.
 * The gate names that person exactly as it names a browser session, so this
 * header is the only thing that tells the two apart; a browser session
 * presents neither.
 */
export type PresentedCredential = "key" | "token";

export function presentedCredential(headers: Headers): PresentedCredential | null {
  if (presentedApiSecret(headers)) {
    return "key";
  }
  return headers.get("authorization")?.trim() ? "token" : null;
}

export function matchApiKey<T extends { secretHash: string }>(keys: readonly T[], secret: string): T | null {
  if (!secret.startsWith("fg_")) {
    return null;
  }
  const digest = Buffer.from(hashApiSecret(secret), "hex");
  for (const key of keys) {
    const stored = Buffer.from(key.secretHash, "hex");
    if (stored.length === digest.length && timingSafeEqual(stored, digest)) {
      return key;
    }
  }
  return null;
}

export function publicApiKeys<T extends { secretHash: string }>(keys: readonly T[]): Array<Omit<T, "secretHash">> {
  return keys.map(({ secretHash: _secretHash, ...rest }) => rest);
}
