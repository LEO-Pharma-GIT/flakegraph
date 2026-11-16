import { describe, expect, it } from "vitest";
import { FairGate } from "./gate";

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

describe("FairGate", () => {
  it("admits up to its capacity and hands each freed slot to the waiting questions in turn", async () => {
    const gate = new FairGate(2);
    const long = Symbol("long question");
    const short = Symbol("short question");
    const order: string[] = [];
    const held = [await gate.acquire(long), await gate.acquire(long)];
    const admitted = (label: string, tenant: symbol) =>
      gate.acquire(tenant).then((release) => {
        order.push(label);
        return release;
      });
    const [long1, long2, long3, short1] = [
      admitted("long 1", long),
      admitted("long 2", long),
      admitted("long 3", long),
      admitted("short 1", short),
    ];
    await tick();
    expect(gate.load).toEqual({ active: 2, waiting: 4 });

    // The short question waited behind three calls of the long one, yet gets the second free slot.
    for (const release of held) {
      release();
    }
    await tick();
    expect(order).toEqual(["long 1", "short 1"]);

    (await long1)();
    (await short1)();
    await tick();
    expect(order).toEqual(["long 1", "short 1", "long 2", "long 3"]);
    (await long2)();
    (await long3)();
    expect(gate.load).toEqual({ active: 0, waiting: 0 });
  });

  it("drops a waiting call whose question is cancelled", async () => {
    const gate = new FairGate(1);
    const release = await gate.acquire(Symbol("first"));
    const cancel = new AbortController();
    const waiting = gate.acquire(Symbol("second"), cancel.signal);
    cancel.abort(new Error("cancelled"));
    await expect(waiting).rejects.toThrow("cancelled");
    expect(gate.load).toEqual({ active: 1, waiting: 0 });
    release();
    expect(gate.load).toEqual({ active: 0, waiting: 0 });
  });
});
