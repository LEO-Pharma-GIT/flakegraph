import { useSyncExternalStore } from "react";

/**
 * Window storage read the way React wants an external store read.
 *
 * A component that copies a storage item into state from an effect renders
 * twice and can never be told the item changed under it. Reading it through
 * `useSyncExternalStore` renders once, with the server snapshot during
 * hydration so the markup matches, and re-renders when this module writes
 * the item or another tab does.
 */
export type StorageArea = "local" | "session";

const listeners = new Set<() => void>();

function area(kind: StorageArea): Storage | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return kind === "local" ? window.localStorage : window.sessionStorage;
  } catch {
    return null;
  }
}

export function readStorage(kind: StorageArea, key: string): string | null {
  try {
    return area(kind)?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

export function writeStorage(kind: StorageArea, key: string, value: string | null): void {
  const storage = area(kind);
  if (!storage) {
    return;
  }
  try {
    if (value === null) {
      storage.removeItem(key);
    } else {
      storage.setItem(key, value);
    }
  } catch {
    return;
  }
  for (const listener of listeners) {
    listener();
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

function serverSnapshot(): null {
  return null;
}

export function useStorageItem(kind: StorageArea, key: string): string | null {
  return useSyncExternalStore(subscribe, () => readStorage(kind, key), serverSnapshot);
}

function subscribeToNothing(): () => void {
  return () => undefined;
}

/**
 * A value only the browser can supply, such as which platform it runs on.
 *
 * `read` must return the same primitive each time it is called for React to
 * settle; the server snapshot is what the markup carries until hydration.
 */
export function useClientValue<T extends string | number | boolean | null>(read: () => T, serverValue: T): T {
  return useSyncExternalStore(subscribeToNothing, read, () => serverValue);
}
