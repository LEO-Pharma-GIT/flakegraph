import { RRF_K } from "./constants";

/** A position in a list (a chunk, a quote) and the score that ranked it. */
export interface Ranked {
  position: number;
  score: number;
}

/**
 * The best `limit` items of a stream of scored positions, kept in a binary
 * min-heap so a ranking over a hundred thousand chunks costs `limit` slots
 * rather than an array and a sort of every score.
 *
 * Equal scores rank the earlier position first, so a ranking is stable from
 * one call to the next and ties follow document order.
 */
export class TopK {
  private readonly positions: Int32Array;
  private readonly scores: Float64Array;
  private size = 0;

  constructor(private readonly limit: number) {
    this.positions = new Int32Array(Math.max(0, limit));
    this.scores = new Float64Array(Math.max(0, limit));
  }

  offer(position: number, score: number): void {
    if (this.limit <= 0) {
      return;
    }
    if (this.size < this.limit) {
      this.positions[this.size] = position;
      this.scores[this.size] = score;
      this.size += 1;
      this.siftUp(this.size - 1);
      return;
    }
    if (!this.beats(position, score, this.positions[0]!, this.scores[0]!)) {
      return;
    }
    this.positions[0] = position;
    this.scores[0] = score;
    this.siftDown(0);
  }

  /** What was kept, best first. */
  ranked(): Ranked[] {
    const order = Array.from({ length: this.size }, (_, slot) => slot);
    order.sort((left, right) =>
      this.beats(this.positions[left]!, this.scores[left]!, this.positions[right]!, this.scores[right]!) ? -1 : 1,
    );
    return order.map((slot) => ({ position: this.positions[slot]!, score: this.scores[slot]! }));
  }

  /** Whether (position, score) ranks above (otherPosition, otherScore). */
  private beats(position: number, score: number, otherPosition: number, otherScore: number): boolean {
    return score > otherScore || (score === otherScore && position < otherPosition);
  }

  private siftUp(slot: number): void {
    let child = slot;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (!this.beats(this.positions[parent]!, this.scores[parent]!, this.positions[child]!, this.scores[child]!)) {
        return;
      }
      this.swap(parent, child);
      child = parent;
    }
  }

  private siftDown(slot: number): void {
    let parent = slot;
    for (;;) {
      const left = parent * 2 + 1;
      const right = left + 1;
      let worst = parent;
      if (left < this.size && this.beats(this.positions[worst]!, this.scores[worst]!, this.positions[left]!, this.scores[left]!)) {
        worst = left;
      }
      if (right < this.size && this.beats(this.positions[worst]!, this.scores[worst]!, this.positions[right]!, this.scores[right]!)) {
        worst = right;
      }
      if (worst === parent) {
        return;
      }
      this.swap(parent, worst);
      parent = worst;
    }
  }

  private swap(left: number, right: number): void {
    const position = this.positions[left]!;
    const score = this.scores[left]!;
    this.positions[left] = this.positions[right]!;
    this.scores[left] = this.scores[right]!;
    this.positions[right] = position;
    this.scores[right] = score;
  }
}

/**
 * Reciprocal rank fusion of rankings of the same positions (Cormack et al.
 * 2009): each list adds 1 / (k + rank) for every position it holds, so an item
 * two methods both rank fairly well beats one only a single method ranks
 * first. It needs no score calibration between methods, which BM25 scores
 * and cosine similarities do not share.
 */
export function fuseRankings(rankings: readonly (readonly number[])[], limit: number, k: number = RRF_K): number[] {
  const fused = new Map<number, number>();
  for (const ranking of rankings) {
    ranking.forEach((position, rank) => {
      fused.set(position, (fused.get(position) ?? 0) + 1 / (k + rank + 1));
    });
  }
  const top = new TopK(limit);
  for (const [position, score] of fused) {
    top.offer(position, score);
  }
  return top.ranked().map((item) => item.position);
}
