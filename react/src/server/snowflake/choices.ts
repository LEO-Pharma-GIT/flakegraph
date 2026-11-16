import path from "node:path";
import { mayReadStagePath } from "../access/snowflake-access";
import type { ModelChoices, StageChoice, StageFolder } from "../protocol/runtime";
import type { Viewer } from "../protocol/schema";
import { EMBEDDING_MODEL_DIMENSIONS, SUPPORTED_SUFFIXES } from "../providers";
import type { SqlRow } from "./client";
import { qualifiedStage, relativeStageName, UPLOAD_STAGE } from "./stage";

/**
 * Stages the worker writes to rather than reads documents from: the bulk-load
 * stages, and the stage that holds job specifications.
 */
const WORKER_STAGES = new Set(["KG_LOAD_STAGE", "KG_BULK_STAGE", "KG_SERVICE_SPECS"]);

/**
 * Cortex models that perform one fixed task (parsing, extraction, translation,
 * moderation, media understanding) rather than open completion, so they cannot
 * follow the extraction prompt. Matched by prefix.
 */
const CORTEX_TASK_MODEL_PREFIXES = [
  "arctic-extract",
  "arctic-parse",
  "arctic-sentiment",
  "arctic-text2sql",
  "arctic-transcribe",
  "arctic-translate",
  "llamaguard",
  "twelvelabs-",
  "voyage-multimodal",
];

/** At most this many folders are offered; a run can still name any prefix. */
const MAX_FOLDERS = 500;

/**
 * The stages the form offers for documents: every stage in the app's schema
 * except artifact repositories and the stages the worker writes to. The upload
 * stage comes first because that is where the app's own uploads go.
 *
 * @param rows - `SHOW STAGES` rows, keys lower-cased.
 * @param database - The app's database, used to qualify stage names.
 * @param schema - The app's schema, used to qualify stage names.
 */
export function documentStageChoices(rows: readonly SqlRow[], database: string, schema: string): StageChoice[] {
  const names = rows
    .filter((row) => !String(row.type ?? "").toUpperCase().includes("ARTIFACT"))
    .map((row) => String(row.name ?? "").trim())
    .filter((name) => name && !WORKER_STAGES.has(name.toUpperCase()));
  return [...new Set(names)]
    .sort((left, right) => Number(right === UPLOAD_STAGE) - Number(left === UPLOAD_STAGE) || left.localeCompare(right))
    .map((name) => ({ stage: qualifiedStage(name, database, schema), name }));
}

/**
 * The folders of a stage a viewer may point a run at, each with the number of
 * readable documents beneath it. A folder is offered only when the viewer may
 * read it as a whole, so a choice never fails the stage check at submission.
 *
 * @param names - Object names from `LIST`, stage namespace included.
 * @param stage - The stage the objects were listed from.
 * @param viewer - Whose access decides which folders are offered.
 */
export function readableStageFolders(names: Iterable<string>, stage: string, viewer: Viewer): StageFolder[] {
  const counts = new Map<string, number>();
  for (const name of names) {
    if (!SUPPORTED_SUFFIXES.has(path.extname(name).toLowerCase())) {
      continue;
    }
    const segments = relativeStageName(name, stage).split("/").slice(0, -1);
    for (let depth = 1; depth <= segments.length; depth += 1) {
      const prefix = segments.slice(0, depth).join("/");
      if (mayReadStagePath(viewer, prefix)) {
        counts.set(prefix, (counts.get(prefix) ?? 0) + 1);
      }
    }
  }
  return [...counts.entries()]
    .map(([prefix, documents]) => ({ prefix, documents }))
    .sort((left, right) => left.prefix.localeCompare(right.prefix))
    .slice(0, MAX_FOLDERS);
}

/**
 * Split the account's Cortex models into those that can extract a graph and
 * the text embedding models whose width is known. An embedding model of unknown
 * width is left out: offering it would let a run fail only when it writes.
 *
 * @param names - Model names from `SHOW MODELS IN SNOWFLAKE.MODELS`.
 */
export function cortexModelChoices(names: Iterable<string>): ModelChoices {
  const llm = new Set<string>();
  const embedding = new Map<string, number>();
  for (const raw of names) {
    const model = raw.trim().toLowerCase();
    if (!model) {
      continue;
    }
    const dimension = EMBEDDING_MODEL_DIMENSIONS[model];
    if (dimension) {
      embedding.set(model, dimension);
    } else if (!isEmbeddingName(model) && !CORTEX_TASK_MODEL_PREFIXES.some((prefix) => model.startsWith(prefix))) {
      llm.add(model);
    }
  }
  return {
    llm: [...llm].sort(),
    embedding: [...embedding.entries()]
      .map(([model, dimension]) => ({ model, dimension }))
      .sort((left, right) => left.dimension - right.dimension || left.model.localeCompare(right.model)),
  };
}

/** Embedding models not in the width table are still not completion models. */
function isEmbeddingName(model: string): boolean {
  return /(^|-)(embed|e5)(-|$)/.test(model) || model.startsWith("voyage-");
}
