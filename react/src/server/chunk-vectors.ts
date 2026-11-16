import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

/** The column the finaliser writes each chunk's embedding to. */
export const CHUNK_VECTOR_COLUMN = "embedding";

/**
 * Every chunk's embedding, unit length, in one flat array: chunk `i`'s vector
 * is `data[i * dimension .. (i + 1) * dimension)`, in the order of the
 * dataset's chunks.
 *
 * One Float32Array rather than an array per chunk: four bytes a component and
 * a single allocation, where a hundred thousand JS arrays of doubles would
 * cost twice the bytes plus an object header each. A chunk without a usable
 * vector is all zeros and marked absent.
 */
export interface ChunkVectors {
  dimension: number;
  data: Float32Array;
  present: Uint8Array;
  /** How many chunks have a vector. */
  count: number;
}

/**
 * Read the chunk embeddings of a graph directory, aligned to `chunkIds`.
 *
 * Parquet is read one row group at a time and only its id and vector columns,
 * so the transient cost is one row group's vectors as JS numbers, not the
 * whole table's. Null when the graph has no chunk vectors.
 */
export async function readChunkVectors(directory: string, chunkIds: readonly string[]): Promise<ChunkVectors | null> {
  const parquet = path.join(directory, "chunks.parquet");
  const json = path.join(directory, "chunks.json");
  const builder = new VectorBuilder(chunkIds);
  if (existsSync(parquet)) {
    const { asyncBufferFromFile, parquetMetadataAsync, parquetReadObjects, parquetSchema } = await import("hyparquet");
    const file = await asyncBufferFromFile(parquet);
    const metadata = await parquetMetadataAsync(file);
    const columns = parquetSchema(metadata).children.map((child) => child.element.name);
    if (!columns.includes(CHUNK_VECTOR_COLUMN) || !columns.includes("id")) {
      return null;
    }
    let rowStart = 0;
    for (const group of metadata.row_groups) {
      const rowEnd = rowStart + Number(group.num_rows);
      const rows = await parquetReadObjects({ file, metadata, columns: ["id", CHUNK_VECTOR_COLUMN], rowStart, rowEnd });
      for (const row of rows) {
        builder.add(row.id, row[CHUNK_VECTOR_COLUMN]);
      }
      rowStart = rowEnd;
    }
    return builder.build();
  }
  if (existsSync(json)) {
    const rows = JSON.parse(await readFile(json, "utf8")) as unknown;
    if (!Array.isArray(rows)) {
      return null;
    }
    for (const row of rows as Record<string, unknown>[]) {
      builder.add(row?.id, row?.[CHUNK_VECTOR_COLUMN]);
    }
    return builder.build();
  }
  return null;
}

/** Collects vectors into the flat store, normalising each as it arrives. */
export class VectorBuilder {
  private readonly position: Map<string, number>;
  private data: Float32Array | null = null;
  private dimension = 0;
  private readonly present: Uint8Array;
  private count = 0;

  constructor(private readonly chunkIds: readonly string[]) {
    this.position = new Map(chunkIds.map((id, index) => [id, index]));
    this.present = new Uint8Array(chunkIds.length);
  }

  /**
   * Add one chunk's vector. A vector for an unknown chunk, one of another
   * width than the first, or one with no length is skipped: a mixed store
   * cannot be compared against one query vector.
   */
  add(id: unknown, vector: unknown): void {
    const index = this.position.get(String(id ?? ""));
    if (index === undefined || !isNumberArray(vector) || vector.length === 0) {
      return;
    }
    if (this.data === null) {
      this.dimension = vector.length;
      this.data = new Float32Array(this.chunkIds.length * this.dimension);
    }
    if (vector.length !== this.dimension) {
      return;
    }
    let squared = 0;
    for (let component = 0; component < this.dimension; component += 1) {
      const value = Number(vector[component]);
      squared += value * value;
    }
    if (!(squared > 0) || !Number.isFinite(squared)) {
      return;
    }
    const scale = 1 / Math.sqrt(squared);
    const offset = index * this.dimension;
    for (let component = 0; component < this.dimension; component += 1) {
      this.data[offset + component] = Number(vector[component]) * scale;
    }
    if (this.present[index] === 0) {
      this.present[index] = 1;
      this.count += 1;
    }
  }

  build(): ChunkVectors | null {
    if (this.data === null || this.count === 0) {
      return null;
    }
    return { dimension: this.dimension, data: this.data, present: this.present, count: this.count };
  }
}

function isNumberArray(value: unknown): value is ArrayLike<number> {
  return Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));
}
