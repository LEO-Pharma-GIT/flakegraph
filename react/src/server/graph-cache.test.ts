import { afterEach, describe, expect, it, vi } from "vitest";
import { GraphCache } from "./graph-cache";

const rows = (value: number) => value;

describe("GraphCache", () => {
  afterEach(() => vi.useRealTimers());

  it("shares one read between requests that overlap", async () => {
    const cache = new GraphCache(rows, { maxRows: 100, idleMs: 1_000 });
    let release = (_value: number) => {};
    const load = vi.fn(() => new Promise<number>((resolve) => { release = resolve; }));

    const first = cache.get("graph", "v1", load);
    const second = cache.get("graph", "v1", load);
    release(5);

    await expect(Promise.all([first, second])).resolves.toEqual([5, 5]);
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("reads again when the files change", async () => {
    const cache = new GraphCache(rows, { maxRows: 100, idleMs: 1_000 });

    await cache.get("graph", "v1", async () => 1);

    await expect(cache.get("graph", "v2", async () => 2)).resolves.toBe(2);
    await expect(cache.get("graph", "v2", async () => 3)).resolves.toBe(2);
  });

  it("does not keep a failed read", async () => {
    const cache = new GraphCache(rows, { maxRows: 100, idleMs: 1_000 });

    await expect(cache.get("graph", "v1", async () => { throw new Error("unreadable"); })).rejects.toThrow("unreadable");
    await vi.waitFor(() => expect(cache.size).toBe(0));

    await expect(cache.get("graph", "v1", async () => 4)).resolves.toBe(4);
  });

  it("drops the graphs used longest ago once the rows run over", async () => {
    const cache = new GraphCache(rows, { maxRows: 12, idleMs: 1_000 });
    await cache.get("old", "v1", async () => 6);
    await cache.get("used", "v1", async () => 3);
    await cache.get("old", "v1", async () => 0);

    await cache.get("new", "v1", async () => 5);
    await vi.waitFor(() => expect(cache.size).toBe(2));

    await expect(cache.get("old", "v1", async () => 7)).resolves.toBe(6);
    await expect(cache.get("used", "v1", async () => 8)).resolves.toBe(8);
  });

  it("keeps the graph read last even when it alone is over", async () => {
    const cache = new GraphCache(rows, { maxRows: 10, idleMs: 1_000 });

    await cache.get("huge", "v1", async () => 50);

    await expect(cache.get("huge", "v1", async () => 0)).resolves.toBe(50);
  });

  it("lets a graph go after an idle spell", async () => {
    vi.useFakeTimers();
    const cache = new GraphCache(rows, { maxRows: 100, idleMs: 1_000 });
    await cache.get("graph", "v1", async () => 1);

    await vi.advanceTimersByTimeAsync(900);
    await cache.get("graph", "v1", async () => 2);
    await vi.advanceTimersByTimeAsync(900);
    expect(cache.size).toBe(1);

    await vi.advanceTimersByTimeAsync(200);
    expect(cache.size).toBe(0);
  });
});
