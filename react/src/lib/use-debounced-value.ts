import { useEffect, useState } from "react";

/**
 * The value as it stood once it stopped changing for `delayMs`.
 *
 * Typing into a field that drives a remote listing should cost one request
 * per pause, not one per keystroke. Values are plain JSON data compared by
 * content, so an object rebuilt on every render does not restart the wait.
 */
export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const key = JSON.stringify(value);
  const [settled, setSettled] = useState({ key, value });
  useEffect(() => {
    if (settled.key === key) {
      return;
    }
    const timer = window.setTimeout(() => setSettled({ key, value: JSON.parse(key) as T }), delayMs);
    return () => window.clearTimeout(timer);
  }, [key, delayMs, settled.key]);
  return settled.key === key ? value : settled.value;
}
