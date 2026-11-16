/**
 * A limit on calls in flight shared by every question the console answers,
 * handing free slots to the waiting questions in turn.
 *
 * Each question that reads documents one by one sends many calls at once. When
 * several such questions arrive together, their calls queue at the engines, and
 * a call's timeout runs while it waits there: under load most of a question's
 * documents then time out unread. Waiting here instead keeps the engines' queues
 * short, starts a call's clock only when it is sent, and gives every waiting
 * question the next slot in turn, so one long question cannot hold back the
 * others.
 */
export class FairGate {
  private active = 0;
  private readonly waiting = new Map<symbol, Array<() => void>>();
  private readonly turn: symbol[] = [];

  constructor(readonly capacity: number) {}

  /** Wait for a slot for `tenant`; the returned function frees it. */
  async acquire(tenant: symbol, signal?: AbortSignal): Promise<() => void> {
    signal?.throwIfAborted();
    if (this.active < this.capacity && this.waiting.size === 0) {
      this.active += 1;
      return this.releaser();
    }
    await new Promise<void>((resolve, reject) => {
      let queue = this.waiting.get(tenant);
      if (!queue) {
        queue = [];
        this.waiting.set(tenant, queue);
        this.turn.push(tenant);
      }
      const onAbort = () => {
        this.forget(tenant, admit);
        reject(signal?.reason);
      };
      const admit = () => {
        signal?.removeEventListener("abort", onAbort);
        resolve();
      };
      queue.push(admit);
      signal?.addEventListener("abort", onAbort, { once: true });
    });
    return this.releaser();
  }

  /** Calls in flight and calls waiting, for tests and status. */
  get load(): { active: number; waiting: number } {
    let waiting = 0;
    for (const queue of this.waiting.values()) {
      waiting += queue.length;
    }
    return { active: this.active, waiting };
  }

  private releaser(): () => void {
    let released = false;
    return () => {
      if (released) {
        return;
      }
      released = true;
      this.active -= 1;
      this.dispatch();
    };
  }

  private dispatch(): void {
    while (this.active < this.capacity && this.turn.length) {
      const tenant = this.turn.shift()!;
      const queue = this.waiting.get(tenant)!;
      const admit = queue.shift()!;
      if (queue.length) {
        this.turn.push(tenant);
      } else {
        this.waiting.delete(tenant);
      }
      this.active += 1;
      admit();
    }
  }

  private forget(tenant: symbol, admit: () => void): void {
    const queue = this.waiting.get(tenant);
    if (!queue) {
      return;
    }
    const index = queue.indexOf(admit);
    if (index >= 0) {
      queue.splice(index, 1);
    }
    if (!queue.length) {
      this.waiting.delete(tenant);
      const position = this.turn.indexOf(tenant);
      if (position >= 0) {
        this.turn.splice(position, 1);
      }
    }
  }
}
