/**
 * Parsed graphs, held briefly so one page open reads a graph once.
 *
 * A graph is read in full to sample and index it, and a page asks for it from
 * several places at once. Each entry holds the read's promise, so requests
 * that arrive during a read wait for it instead of starting their own. A
 * version names the files read: replacing them makes the entry miss.
 *
 * Graphs are held by rows rather than by count because the largest is hundreds
 * of times the smallest, and they leave after an idle spell so a console nobody
 * is using gives its memory back.
 */
export interface GraphCacheLimits {
  /** Rows held across all graphs; the graph read last stays whatever its size. */
  maxRows: number;
  idleMs: number;
}

interface Entry<T> {
  version: string;
  value: Promise<T>;
  rows: number;
  timer: ReturnType<typeof setTimeout> | null;
}

export class GraphCache<T> {
  /** In use order, oldest first. */
  private readonly entries = new Map<string, Entry<T>>();

  constructor(
    private readonly countRows: (value: T) => number,
    private readonly limits: GraphCacheLimits,
  ) {}

  get(key: string, version: string, load: () => Promise<T>): Promise<T> {
    const held = this.entries.get(key);
    if (held?.version === version) {
      this.entries.delete(key);
      this.entries.set(key, held);
      this.expireLater(key, held);
      return held.value;
    }
    this.forget(key);
    const entry: Entry<T> = { version, value: load(), rows: 0, timer: null };
    this.entries.set(key, entry);
    this.expireLater(key, entry);
    entry.value.then(
      (value) => {
        entry.rows = this.countRows(value);
        this.trim();
      },
      () => {
        if (this.entries.get(key) === entry) {
          this.forget(key);
        }
      },
    );
    return entry.value;
  }

  /** How many graphs are held or being read. */
  get size(): number {
    return this.entries.size;
  }

  clear(): void {
    for (const key of [...this.entries.keys()]) {
      this.forget(key);
    }
  }

  private trim(): void {
    let rows = [...this.entries.values()].reduce((total, entry) => total + entry.rows, 0);
    for (const [key, entry] of this.entries) {
      if (rows <= this.limits.maxRows || this.entries.size === 1) {
        return;
      }
      if (entry.rows > 0) {
        rows -= entry.rows;
        this.forget(key);
      }
    }
  }

  private expireLater(key: string, entry: Entry<T>): void {
    if (entry.timer) {
      clearTimeout(entry.timer);
    }
    entry.timer = setTimeout(() => {
      if (this.entries.get(key) === entry) {
        this.forget(key);
      }
    }, this.limits.idleMs);
    entry.timer.unref?.();
  }

  private forget(key: string): void {
    const entry = this.entries.get(key);
    if (entry?.timer) {
      clearTimeout(entry.timer);
    }
    this.entries.delete(key);
  }
}
