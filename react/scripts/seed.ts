import { randomBytes } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { goldToDataset, writeDataset, type GoldGraph } from "../src/server/gold";
import type { GraphDataset } from "../src/server/protocol/schema";
import { writeRunRecord } from "../src/server/catalog";
import { makeProgressRecord } from "../src/server/progress";
import { atomicWriteJson } from "../src/server/catalog";
import { loadEnv } from "../src/server/env";
import { hashApiSecret } from "../src/server/api-keys";
import { claim, grant, updateAccess } from "../src/server/access/store";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

async function main() {
  const env = loadEnv({
    ...process.env,
    FLAKEGRAPH_REPOSITORY_ROOT: repoRoot,
    FLAKEGRAPH_APP_STATE_ROOT: process.env.FLAKEGRAPH_APP_STATE_ROOT,
  });
  const stateRoot = env.stateRoot;
  await rm(stateRoot, { recursive: true, force: true });
  await mkdir(stateRoot, { recursive: true });

  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_martial_arts",
    graphId: "graph_martial_arts",
    graphName: "Martial arts history",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/deep_learning_papers/gold.json"),
    runId: "run_deep_learning",
    graphId: "graph_deep_learning",
    graphName: "Deep learning papers",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_rename_target",
    graphId: "graph_rename_target",
    graphName: "Rename target",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_delete_me",
    graphId: "graph_delete_me",
    graphName: "Disposable graph",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_bulk_delete",
    graphId: "graph_bulk_delete",
    graphName: "Bulk delete me",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_share_demo",
    graphId: "graph_share_demo",
    graphName: "Team handbook",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_bob_notes",
    graphId: "graph_bob_notes",
    graphName: "Bob field notes",
    status: "succeeded",
    owner: "BOB",
  });
  await seedAccess(stateRoot);
  await seedThinGoldDataset(stateRoot);
  await seedGappyCorpus(stateRoot);
  await seedActiveRun(stateRoot);
  await seedUnavailableRun(stateRoot);
  await seedFailedRun(stateRoot);
  await seedKubernetes(stateRoot);
  await seedSnowflake(stateRoot);
  await seedWorkspace(stateRoot);
  await seedCancellingRun(stateRoot);
  await seedLargeCorpusRun(stateRoot);
  await seedPiiPack(stateRoot);
  console.log(`Seeded control-plane fixtures under ${stateRoot}`);
}

async function seedGoldDataset(options: {
  stateRoot: string;
  goldPath: string;
  runId: string;
  graphId: string;
  graphName: string;
  status: string;
  owner?: string | null;
}) {
  const gold = JSON.parse(await readFile(options.goldPath, "utf8")) as GoldGraph;
  const outputPath = path.join(options.stateRoot, "graphs", options.graphId);
  const base = goldToDataset(gold, options.graphId);
  // The deep-learning graph carries a handful of extraction gaps - the case
  // where one reader wants to see exactly which pages were lost - and one
  // document no parser could read, recorded while the rest of the run went on.
  // The martial-arts graph holds one relation its sources only claim.
  const dataset =
    options.graphId === "graph_deep_learning"
      ? {
          ...base,
          discardedWindows: fewGaps(options.graphId, base),
          failedDocuments: [unreadableDocument(options.graphId, "file_broken", "s3://corpus/deep-learning/broken.pdf")],
        }
      : options.graphId === "graph_martial_arts"
        ? { ...base, edges: base.edges.map((edge) => (edge.id === "rel_004" ? { ...edge, assertion: "claimed" } : edge)) }
        : base;
  await writeDataset(outputPath, { ...dataset, runReport: { ...dataset.runReport, job_id: options.runId } });
  const directory = path.join(options.stateRoot, "runs", options.runId);
  await writeRunRecord(directory, {
    runId: options.runId,
    graphId: options.graphId,
    graphName: options.graphName,
    status: options.status,
    runtime: "local",
    startedAt: "2026-09-01T10:00:00.000Z",
    updatedAt: "2026-09-01T12:00:00.000Z",
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: gold.documents?.length ?? 0,
    documentsCompleted: gold.documents?.length ?? 0,
    documentsFailed: 0,
    nodeCount: gold.entities?.length ?? 0,
    owner: options.owner ?? "ALICE",
    sourceKind: "local_path",
    sourcePath: options.goldPath.includes("deep_learning")
      ? "data/deep_learning_papers/files"
      : "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
  });
  await persistName(options.stateRoot, options.graphId, options.graphName);
}

/**
 * A martial-arts graph that found three entities and little else. Quality
 * compares it to the full gold file, so nearly every required relation is
 * missing: the list the Quality tab has to page.
 */
/**
 * BOB has shared his field notes with ALICE to read, and ALICE her martial
 * arts graph with CAROL, the analyst its production perspectives are for.
 * Every other graph is its owner's alone.
 */
async function seedAccess(stateRoot: string) {
  const at = new Date("2026-09-02T09:00:00.000Z");
  await updateAccess(stateRoot, (state) => {
    claim(state, "graph_bob_notes", "BOB");
    grant(state, "graph_bob_notes", "ALICE", "read", "BOB", at);
    claim(state, "graph_martial_arts", "ALICE");
    grant(state, "graph_martial_arts", "CAROL", "read", "ALICE", at);
  });
}

async function seedThinGoldDataset(stateRoot: string) {
  const goldPath = path.join(repoRoot, "data/martial_arts/gold.json");
  const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph;
  const kept = new Set((gold.entities ?? []).slice(0, 3).map((entity) => entity.id));
  const thin: GoldGraph = {
    ...gold,
    entities: (gold.entities ?? []).filter((entity) => kept.has(entity.id)),
    relations: (gold.relations ?? []).filter((relation) => kept.has(relation.source) && kept.has(relation.target)),
  };
  const graphId = "graph_martial_thin";
  const outputPath = path.join(stateRoot, "graphs", graphId);
  const base = goldToDataset(thin, graphId);
  // Two of its documents lost a window each, so the Documents table on Run
  // details has gap chips to show beside its phases.
  await writeDataset(outputPath, { ...base, runReport: { ...base.runReport, job_id: "run_martial_thin" }, discardedWindows: fewGaps(graphId, base, ["page-01.md", "page-02.md"]) });
  await writeRunRecord(path.join(stateRoot, "runs", "run_martial_thin"), {
    runId: "run_martial_thin",
    graphId,
    graphName: "Thin martial arts extract",
    status: "succeeded",
    runtime: "local",
    startedAt: "2026-08-20T10:00:00.000Z",
    updatedAt: "2026-08-20T10:30:00.000Z",
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: gold.documents?.length ?? 0,
    documentsCompleted: gold.documents?.length ?? 0,
    documentsFailed: 0,
    nodeCount: thin.entities?.length ?? 0,
    owner: "ALICE",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
  });
  await persistName(stateRoot, graphId, "Thin martial arts extract");
  // Enough progress events that the run's details show a tail of the log,
  // not all of it.
  const events = Array.from({ length: THIN_RUN_PAGES }, (_, index) => `page-${String(index + 1).padStart(2, "0")}.md`).flatMap(
    (fileId, index) =>
      ["ocr", "extract"].map((stage, offset) =>
        makeProgressRecord({
          timestamp: `2026-08-20T10:${String(index).padStart(2, "0")}:${offset ? "30" : "00"}.000Z`,
          stage,
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      ),
  );
  await writeFile(
    path.join(stateRoot, "runs", "run_martial_thin", "events.jsonl"),
    events.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
}

/** Three gaps in two of a graph's documents: a title page and a figure caption the model misread. */
function fewGaps(graphId: string, dataset: GraphDataset, fileIds?: string[]): Record<string, unknown>[] {
  const documents = fileIds ?? dataset.documents.slice(0, 2).map((document) => String(document.file_id ?? document.id ?? ""));
  if (documents.length < 2) {
    return [];
  }
  return [
    discardedWindow(graphId, documents[0], "entities", 1, 1, 1, { ungrounded_quote: 1 }, "An introduction to our capabilities and experience www.example.org hello@example.org"),
    discardedWindow(graphId, documents[0], "relations", 4, 5, 3, { domain_or_range_violation: 2, self_loop: 1 }, "Figure 2: Layer-wise relevance of the hidden units over training epochs, shown for the three seeds."),
    discardedWindow(graphId, documents[1], "entities", 12, 12, 2, { ungrounded_quote: 2 }, "Acknowledgements. We thank the anonymous reviewers for their comments and the compute donors for GPU time."),
  ];
}

/** A file that is not what its name says: MinerU's own refusal, as a fleet run records it. */
function unreadableDocument(graphId: string, fileId: string, sourceUri: string): Record<string, unknown> {
  return {
    id: `failed_document_${fileId}`,
    graph_id: graphId,
    document_id: fileId,
    file_id: fileId,
    source_uri: sourceUri,
    mime_type: "application/pdf",
    size_bytes: 18,
    provider: "fallback",
    error_type: "RuntimeError",
    error_message: 'mineru_api returned HTTP 400: {"detail":"Unsupported file type: txt"}',
  };
}

function discardedWindow(
  graphId: string,
  fileId: string,
  stage: "entities" | "relations",
  pageStart: number,
  pageEnd: number,
  extracted: number,
  reasons: Record<string, number>,
  preview: string,
  sequence = 0,
): Record<string, unknown> {
  return {
    id: `discarded_window_${fileId}_${stage}_${pageStart}_${sequence}`,
    graph_id: graphId,
    document_id: fileId,
    file_id: fileId,
    stage,
    window_id: `window_${fileId}_${pageStart}_${sequence}`,
    chunk_ids: [`chunk_${fileId}_${pageStart}`],
    page_start: pageStart,
    page_end: pageEnd,
    extracted_records: extracted,
    record_actions: reasons,
    preview,
  };
}

/**
 * A succeeded corpus of 600 documents where 420 carry gaps - 900 windows in
 * all, one document with 160 of them. The Quality tab has to stay readable
 * here: a count line, the reasons, a paged document table, and one
 * document's windows at a time.
 */
async function seedGappyCorpus(stateRoot: string) {
  const goldPath = path.join(repoRoot, "data/martial_arts/gold.json");
  const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph;
  const graphId = "graph_gappy_corpus";
  const runId = "run_gappy_corpus";
  const base = goldToDataset(gold, graphId);
  const total = 600;
  const documents = Array.from({ length: total }, (_, index) => {
    const name = `batch-${String(Math.floor(index / 50) + 1).padStart(2, "0")}/record-${String(index + 1).padStart(4, "0")}.pdf`;
    return { id: `doc_${index + 1}`, graph_id: graphId, file_id: `file_${index + 1}`, source_uri: `s3://corpus/${name}`, mime_type: "application/pdf" };
  });
  const windows: Record<string, unknown>[] = [];
  const reasons: Record<string, number>[] = [
    { ungrounded_quote: 1 },
    { ungrounded_quote: 2, duplicate: 1 },
    { domain_or_range_violation: 3 },
    { ungrounded_quote: 1, untrusted_alias: 4 },
    { type_definition_as_description: 2 },
  ];
  for (let index = 0; index < total; index += 1) {
    if (index % 10 >= 7) {
      continue; // three in ten documents are clean
    }
    const fileId = `file_${index + 1}`;
    const count = index === 5 ? 160 : 1 + (index % 4);
    for (let gap = 0; gap < count; gap += 1) {
      const stage = (index + gap) % 3 === 0 ? "relations" : "entities";
      const page = 1 + ((index * 7 + gap * 3) % 40);
      windows.push(
        discardedWindow(
          graphId,
          fileId,
          stage,
          page,
          page + (gap % 2),
          1 + ((index + gap) % 5),
          reasons[(index + gap) % reasons.length],
          `Batch record ${index + 1}, page ${page}: lot ${1000 + index * 3 + gap} released by QC on ${10 + (gap % 18)}-0${1 + (gap % 9)}-27; remarks: see attached certificate of analysis for the listed impurities and assay results.`,
          gap,
        ),
      );
    }
  }
  const dataset: GraphDataset = {
    ...base,
    documents,
    discardedWindows: windows,
    runReport: { ...base.runReport, job_id: runId, graph_id: graphId, files_processed: total },
  };
  const outputPath = path.join(stateRoot, "graphs", graphId);
  await writeDataset(outputPath, dataset);
  await writeRunRecord(path.join(stateRoot, "runs", runId), {
    runId,
    graphId,
    graphName: "Gappy corpus",
    status: "succeeded",
    runtime: "local",
    startedAt: "2026-09-02T08:00:00.000Z",
    updatedAt: "2026-09-02T14:00:00.000Z",
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: total,
    documentsCompleted: total,
    documentsFailed: 0,
    nodeCount: gold.entities?.length ?? 0,
    owner: "ALICE",
    sourceKind: "s3",
    sourcePath: "s3://corpus/",
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
  });
  await persistName(stateRoot, graphId, "Gappy corpus");
}

/** Documents the thin run logged: two events each, more than one page of the events card. */
const THIN_RUN_PAGES = 20;

async function seedActiveRun(stateRoot: string) {
  const runId = "run_active_ocr";
  const directory = path.join(stateRoot, "runs", runId);
  const outputPath = path.join(stateRoot, "graphs", "graph_active");
  await mkdir(outputPath, { recursive: true });
  await writeRunRecord(directory, {
    runId,
    graphId: "graph_active",
    graphName: "Active OCR run",
    status: "running",
    runtime: "local",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: 10,
    documentsCompleted: 3,
    documentsFailed: 0,
    owner: "ALICE",
  });
  const events = [
    makeProgressRecord({
      timestamp: new Date().toISOString(),
      stage: "file_source",
      status: "completed",
      fileId: null,
      message: "Discovered 10 files",
      elapsedMs: 120,
      counts: { files_seen: 10 },
    }),
    makeProgressRecord({
      timestamp: new Date().toISOString(),
      stage: "ocr",
      status: "failed",
      fileId: "doc_poison_scan",
      message: "Poison PDF quarantined",
      elapsedMs: 900,
      counts: { files_total: 10 },
    }),
  ];
  await writeFile(
    path.join(directory, "events.jsonl"),
    events.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
}

async function seedUnavailableRun(stateRoot: string) {
  await writeRunRecord(path.join(stateRoot, "runs", "run_missing_artifacts"), {
    runId: "run_missing_artifacts",
    graphId: "graph_copied_history",
    graphName: "Copied host graph",
    status: "succeeded",
    runtime: "local",
    startedAt: "2026-08-01T00:00:00.000Z",
    updatedAt: "2026-08-01T01:00:00.000Z",
    outputPath: "/not/on/this/machine/graph",
    storageKind: "local_files",
    storageLocation: "/not/on/this/machine/graph",
    owner: null,
  });
}

async function seedFailedRun(stateRoot: string) {
  const directory = path.join(stateRoot, "runs", "run_failed_llm");
  await writeRunRecord(directory, {
    runId: "run_failed_llm",
    graphId: "graph_failed_llm",
    graphName: "LLM timeout",
    status: "failed",
    runtime: "local",
    startedAt: "2026-09-02T00:00:00.000Z",
    updatedAt: "2026-09-02T00:05:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_failed_llm"),
    storageKind: "local_files",
    error: "LLM endpoint timed out",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
}

type StubNode = {
  name: string;
  ready: boolean;
  nodeClass: string;
  gpuCount: number;
  gpuModel: string | null;
  cpuCapacity: string | null;
  memoryCapacity: string | null;
  gpuPercent: string | null;
  cpuUsage: string | null;
  cpuPercent: string | null;
  memoryUsage: string | null;
  memoryPercent: string | null;
  workloadCount: number;
  workerCount: number;
  modelServerReady: boolean;
  model: string | null;
  modelImage: string | null;
};

type StubWorkload = {
  name: string;
  component: string;
  phase: string;
  node: string | null;
  ready: boolean;
  restarts: number;
  cpu: string | null;
  memory: string | null;
  image: string | null;
  model: string | null;
};

function stubNode(name: string, ready: boolean, gpuModel: string | null): StubNode {
  return {
    name,
    ready,
    nodeClass: gpuModel ? "gpu" : "cpu",
    gpuCount: gpuModel ? 1 : 0,
    gpuModel,
    cpuCapacity: "20",
    memoryCapacity: "128Gi",
    gpuPercent: null,
    cpuUsage: null,
    cpuPercent: null,
    memoryUsage: null,
    memoryPercent: null,
    workloadCount: 0,
    workerCount: 0,
    modelServerReady: false,
    model: null,
    modelImage: null,
  };
}

function stubWorkload(
  name: string,
  component: string,
  node: string | null,
  overrides: Partial<StubWorkload> = {},
): StubWorkload {
  return {
    name,
    component,
    phase: "Running",
    node,
    ready: true,
    restarts: 0,
    cpu: null,
    memory: null,
    image: null,
    model: null,
    ...overrides,
  };
}

/**
 * A fleet the size a real deployment reaches: four nodes (one not ready),
 * some sixty pods over every component family, a few in trouble and a few
 * finished Jobs. The workload list must stay readable at this size, and the
 * e2e journey uses these counts to prove its filters, order and paging.
 */
function stubFleet(): { nodes: StubNode[]; workloads: StubWorkload[] } {
  const model = "Qwen/Qwen3-8B";
  const gpuNodes = ["gpu-a", "gpu-b", "gpu-c"];
  const workloads: StubWorkload[] = [
    stubWorkload("worker-a", "worker-extraction", "gpu-a", { cpu: "2", memory: "8Gi", image: "flakegraph:mineru-oss" }),
    stubWorkload("vllm-a", "model-server", "gpu-a", { cpu: "4", memory: "24Gi", image: "vllm/vllm-openai", model }),
    stubWorkload("vllm-b", "model-serving", "gpu-b", { image: "vllm/vllm-openai", model }),
    stubWorkload("vllm-c", "model-serving", "gpu-c", { image: "vllm/vllm-openai", model }),
    stubWorkload("inference-router-0", "inference-router", "gpu-a"),
    stubWorkload("mineru-0", "document-parsing", "gpu-b"),
    stubWorkload("ocr-shim-0", "ocr-shim", "gpu-b"),
    stubWorkload("gateway-0", "gateway", "gpu-a"),
    stubWorkload("auth-proxy-0", "auth-proxy", "gpu-a"),
    stubWorkload("console-0", "control-plane", "gpu-a"),
    stubWorkload("control-plane-api-0", "control-plane", "gpu-c"),
    stubWorkload("grafana-0", "monitoring", "gpu-c"),
    stubWorkload("prometheus-0", "monitoring", "gpu-c"),
    stubWorkload("monitoring-db-0", "monitoring", "gpu-c"),
    // Spark executors carry no FlakeGraph component label and land in "other".
    ...Array.from({ length: 4 }, (_, index) => stubWorkload(`spark-exec-${index}`, "workload", gpuNodes[index % 3]!)),
    // The KEDA-scaled worker pools, spread over the ready GPU nodes.
    ...Array.from({ length: 24 }, (_, index) =>
      stubWorkload(`worker-extract-${String(index).padStart(2, "0")}`, "worker-extract", gpuNodes[index % 3]!),
    ),
    ...Array.from({ length: 12 }, (_, index) =>
      stubWorkload(`worker-prepare-${String(index).padStart(2, "0")}`, "worker-prepare", gpuNodes[index % 3]!),
    ),
    // Trouble: two workers waiting for a node, one out of memory, one crashlooping.
    stubWorkload("worker-extract-24", "worker-extract", null, { phase: "Pending", ready: false }),
    stubWorkload("worker-extract-25", "worker-extract", null, { phase: "Pending", ready: false }),
    stubWorkload("worker-extract-oom-7", "worker-extract", "gpu-b", { phase: "Failed", ready: false, restarts: 3 }),
    stubWorkload("ocr-shim-1", "ocr-shim", "gpu-c", { ready: false, restarts: 5 }),
    // Finished one-off Jobs that linger until their TTL.
    stubWorkload("bench-export-p8k2z", "workload", "gpu-a", { phase: "Succeeded", ready: false }),
    stubWorkload("vllm28-probe-x1q9r", "model-serving", "gpu-b", { phase: "Succeeded", ready: false }),
    stubWorkload("database-bootstrap-1", "database-bootstrap", "gpu-c", { phase: "Succeeded", ready: false }),
    stubWorkload("spark-driver-4f1c2", "workload", "gpu-c", { phase: "Succeeded", ready: false }),
  ];
  const nodes = [
    { ...stubNode("gpu-a", true, "NVIDIA A10"), gpuPercent: "41%", cpuUsage: "4", cpuPercent: "20%", memoryUsage: "32Gi", memoryPercent: "25%" },
    stubNode("gpu-b", true, "NVIDIA A10"),
    stubNode("gpu-c", true, "NVIDIA A10"),
    stubNode("gpu-d", false, "NVIDIA A10"),
  ].map((node) => {
    const onNode = workloads.filter((workload) => workload.node === node.name);
    const server = onNode.find((workload) => workload.model);
    return {
      ...node,
      workloadCount: onNode.length,
      workerCount: onNode.filter((workload) => workload.component.startsWith("worker")).length,
      modelServerReady: Boolean(server?.ready),
      model: server?.model ?? null,
      modelImage: server?.image ?? null,
    };
  });
  return { nodes, workloads };
}

async function seedKubernetes(stateRoot: string) {
  const directory = path.join(stateRoot, "kubernetes");
  await mkdir(directory, { recursive: true });
  const fleet = stubFleet();
  await atomicWriteJson(path.join(directory, "cluster.json"), {
    context: "lab",
    namespace: "flakegraph",
    nodesReady: fleet.nodes.filter((node) => node.ready).length,
    nodesTotal: fleet.nodes.length,
    nodes: fleet.nodes,
    workloads: fleet.workloads,
    warnings: [],
    observedAt: new Date().toISOString(),
  });
  await atomicWriteJson(path.join(directory, "assignments.json"), {
    assignments: [
      {
        workerId: "worker-a",
        runId: "run_k8s_martial",
        graphId: "graph_k8s_martial",
        taskId: "task-1",
        stage: "extract_entity_window",
        scopeId: "gpu-a",
        updatedAt: new Date().toISOString(),
      },
      // gpu-b holds more leases than a card lists, so the card has to say "N more".
      ...Array.from({ length: 23 }, (_, index) => ({
        workerId: `worker-extract-gpu-b-${String(index).padStart(2, "0")}`,
        runId: "run_k8s_martial",
        graphId: "graph_k8s_martial",
        taskId: `task-b-${index}`,
        stage: "extract_entity_window",
        scopeId: `doc-${String(index).padStart(3, "0")}`,
        updatedAt: new Date().toISOString(),
      })),
    ],
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_k8s_martial"), {
    runId: "run_k8s_martial",
    graphId: "graph_k8s_martial",
    graphName: "Fleet martial arts",
    status: "running",
    runtime: "kubernetes",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    cancellationRequestedAt: null,
    error: null,
    owner: "ALICE",
  });
  const done = path.join(stateRoot, "runs", "run_k8s_done");
  // The configuration the run was submitted with, carrying the vocabulary it
  // extracted: a revision of this run is typed like it.
  await mkdir(done, { recursive: true });
  await writeFile(
    path.join(done, "config.yaml"),
    [
      "ontology:",
      "  profile:",
      "    name: console",
      "    mode: hybrid",
      "    entity_types:",
      "      - name: PERSON",
      "        description: A named human being.",
      "      - name: SCHOOL",
      "        description: A martial-arts school or lineage.",
      "    relation_types:",
      "      - name: FOUNDED_BY",
      "        description: The source was founded by the target.",
      "",
    ].join("\n"),
    "utf8",
  );
  await writeRunRecord(done, {
    runId: "run_k8s_done",
    graphId: "graph_k8s_done",
    graphName: "Fleet judo",
    status: "succeeded",
    runtime: "kubernetes",
    configPath: path.join(done, "config.yaml"),
    startedAt: "2026-09-05T00:00:00.000Z",
    updatedAt: "2026-09-05T01:30:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 2,
    documentsCompleted: 2,
    documentsFailed: 0,
    sourceKind: "upload",
    sourcePath: path.join(stateRoot, "uploads", "judo"),
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
  await writeFile(
    path.join(done, "events.jsonl"),
    ["judo-history.md", "karate-history.md"]
      .flatMap((fileId) => [
        makeProgressRecord({
          timestamp: "2026-09-05T00:10:00.000Z",
          stage: "ocr",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
        makeProgressRecord({
          timestamp: "2026-09-05T01:00:00.000Z",
          stage: "extract",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      ])
      .map((event) => JSON.stringify(event))
      .join("\n") + "\n",
    "utf8",
  );
  // A later run that published the same graph: the stub fleet lists both as
  // versions, this one as head, so editing the older one can be exercised.
  const doneV2 = path.join(stateRoot, "runs", "run_k8s_done_v2");
  await writeRunRecord(doneV2, {
    runId: "run_k8s_done_v2",
    graphId: "graph_k8s_done",
    graphName: "Fleet judo",
    status: "succeeded",
    runtime: "kubernetes",
    startedAt: "2026-09-06T00:00:00.000Z",
    updatedAt: "2026-09-06T01:00:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 1,
    documentsCompleted: 1,
    documentsFailed: 0,
    sourceKind: "upload",
    sourcePath: path.join(stateRoot, "uploads", "judo"),
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
    baseRunId: "run_k8s_done",
  });
  await writeFile(
    path.join(doneV2, "events.jsonl"),
    [
      makeProgressRecord({
        timestamp: "2026-09-06T00:10:00.000Z",
        stage: "ocr",
        status: "completed",
        fileId: "judo-history.md",
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
      makeProgressRecord({
        timestamp: "2026-09-06T00:20:00.000Z",
        stage: "extract",
        status: "completed",
        fileId: "judo-history.md",
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
    ]
      .map((event) => JSON.stringify(event))
      .join("\n") + "\n",
    "utf8",
  );
  await writeRunRecord(path.join(stateRoot, "runs", "run_k8s_failed"), {
    runId: "run_k8s_failed",
    graphId: "graph_k8s_failed",
    graphName: "Fleet timeout",
    status: "failed",
    runtime: "kubernetes",
    startedAt: "2026-09-04T00:00:00.000Z",
    updatedAt: "2026-09-04T00:10:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    error: "Worker lease expired",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
  // A graph rebuilt every few days: more versions than one page of the
  // versions list holds. Dated before the other fixtures so the newest
  // catalog rows stay what the other journeys expect.
  for (let index = 1; index <= MANY_VERSIONS; index += 1) {
    const day = String(index).padStart(2, "0");
    await writeRunRecord(path.join(stateRoot, "runs", `run_k8s_kata_${day}`), {
      runId: `run_k8s_kata_${day}`,
      graphId: "graph_k8s_kata",
      graphName: "Revised kata",
      status: "succeeded",
      runtime: "kubernetes",
      startedAt: `2026-08-${day}T00:00:00.000Z`,
      updatedAt: `2026-08-${day}T01:00:00.000Z`,
      outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
      storageKind: "local_files",
      storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
      documentsTotal: 2,
      documentsCompleted: 2,
      documentsFailed: 0,
      sourceKind: "upload",
      sourcePath: path.join(stateRoot, "uploads", "kata"),
      ocrProvider: "fallback",
      llmProvider: "vllm_local",
      embeddingProvider: "sentence_transformers",
      owner: "ALICE",
      baseRunId: index > 1 ? `run_k8s_kata_${String(index - 1).padStart(2, "0")}` : undefined,
    });
  }
}

/** Versions of the "Revised kata" fleet graph: a page of ten and a few more. */
const MANY_VERSIONS = 14;

async function seedSnowflake(stateRoot: string) {
  const outputPath = path.join(stateRoot, "graphs", "graph_martial_arts");
  const thinPath = path.join(stateRoot, "graphs", "graph_snow_thin");
  await writeDataset(thinPath, {
    graphId: "graph_snow_thin",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents: [],
    chunks: [],
    discardedWindows: [], rejectedRecords: [],
    failedDocuments: [],
    runReport: {},
    graphMetrics: {},
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_martial"), {
    runId: "run_snow_martial",
    graphId: "graph_snow_martial",
    graphName: "Shared martial arts",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T02:00:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    nodeCount: 74,
    owner: null,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_private"), {
    runId: "run_snow_private",
    graphId: "graph_snow_private",
    graphName: "Private graph",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:00:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    owner: "BOB",
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_delete"), {
    runId: "run_snow_delete",
    graphId: "graph_snow_delete",
    graphName: "Disposable snowflake graph",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:30:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_thin"), {
    runId: "run_snow_thin",
    graphId: "graph_snow_thin",
    graphName: "Empty share candidate",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:45:00.000Z",
    outputPath: thinPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    nodeCount: 0,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_queued"), {
    runId: "run_snow_queued",
    graphId: "graph_snow_queued",
    graphName: "Queued Snowflake job",
    status: "queued",
    runtime: "snowflake",
    startedAt: "2026-09-03T03:00:00.000Z",
    updatedAt: "2026-09-03T03:00:00.000Z",
    outputPath: thinPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 0,
    documentsFailed: 0,
  });
  await atomicWriteJson(path.join(stateRoot, "snowflake", "store.json"), {
    graphs: [
      {
        graphId: "graph_snow_martial",
        owner: null,
        location: outputPath,
        name: "Shared martial arts",
      },
      {
        graphId: "graph_snow_private",
        owner: "BOB",
        location: outputPath,
        name: "Private graph",
      },
      {
        graphId: "graph_snow_delete",
        owner: null,
        location: outputPath,
        name: "Disposable snowflake graph",
      },
      {
        graphId: "graph_snow_thin",
        owner: null,
        location: thinPath,
        name: "Empty share candidate",
      },
    ],
    jobs: [],
  });
}

async function persistName(stateRoot: string, graphId: string, name: string) {
  const file = path.join(stateRoot, "graph-names.json");
  await atomicWriteJson(file, { ...(JSON.parse(await readFile(file, "utf8").catch(() => "{}")) as Record<string, string>), [graphId]: name });
}

async function seedCancellingRun(stateRoot: string) {
  await writeRunRecord(path.join(stateRoot, "runs", "run_cancelling"), {
    runId: "run_cancelling",
    graphId: "graph_cancelling",
    graphName: "Cancelling extract",
    status: "cancelling",
    runtime: "kubernetes",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 10,
    documentsCompleted: 4,
    owner: "ALICE",
  });
}

/**
 * A corpus the size FlakeGraph is built for: 300 documents in flight, a
 * few of them failed. Every document list in the console has to stay
 * readable here, not only on the ten-file packs.
 */
async function seedLargeCorpusRun(stateRoot: string) {
  const directory = path.join(stateRoot, "runs", "run_large_corpus");
  const total = 300;
  await writeRunRecord(directory, {
    runId: "run_large_corpus",
    graphId: "graph_large_corpus",
    graphName: "Large corpus",
    status: "running",
    runtime: "local",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_large_corpus"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_large_corpus"),
    documentsTotal: total,
    documentsCompleted: 120,
    documentsFailed: 6,
    owner: "ALICE",
  });
  const events = [];
  for (let index = 1; index <= total; index += 1) {
    const fileId = `paper-${String(index).padStart(3, "0")}.pdf`;
    if (index % 43 === 0) {
      events.push(
        makeProgressRecord({
          timestamp: new Date().toISOString(),
          stage: "ocr",
          status: "failed",
          fileId,
          message: "Poison PDF quarantined",
          elapsedMs: 900,
          counts: {},
        }),
      );
      continue;
    }
    events.push(
      makeProgressRecord({
        timestamp: new Date().toISOString(),
        stage: "ocr",
        status: "completed",
        fileId,
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
    );
    if (index <= 120) {
      events.push(
        makeProgressRecord({
          timestamp: new Date().toISOString(),
          stage: "extract",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      );
    }
  }
  await writeFile(path.join(directory, "events.jsonl"), events.map((event) => JSON.stringify(event)).join("\n") + "\n", "utf8");
}

async function seedPiiPack(stateRoot: string) {
  const directory = path.join(stateRoot, "pii-pack");
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "contact.md"), "Contact jane@example.com about the corpus.\n", "utf8");
}

async function seedWorkspace(stateRoot: string) {
  await atomicWriteJson(path.join(stateRoot, "workspace.json"), {
    identity: null,
    role: "operator",
    suggestionMode: "on-request",
    lastSuccessRuntime: "local",
    lastConfigDigest: { local: "local_path:data/martial_arts/files" },
    piiAcknowledged: [],
    skippedFiles: {},
    estimates: {
      run_martial_arts: {
        objectCount: 10,
        runtime: "local",
        preset: "fast",
        usdLow: 1.2,
        usdHigh: 2.4,
        minutesLow: 4,
        minutesHigh: 8,
        note: "Local work is priced against the hosted reference card, not a bill.",
      },
    },
    perspectives: [
      {
        id: "perspective_judo",
        graphId: "graph_martial_arts",
        name: "Judo",
        lifecycle: "production",
        search: "judo",
        communityIds: ["community_martial_art"],
        suggestedQuestions: ["Who developed judo?", "Which schools taught judo?"],
      },
      {
        id: "perspective_judo_draft",
        graphId: "graph_martial_arts",
        name: "Judo draft",
        lifecycle: "draft",
        search: "kano",
        communityIds: ["community_person"],
        suggestedQuestions: ["Who is Jigoro Kano?"],
      },
    ],
    // A graph the workspace has recorded more versions of than one page of
    // the Versions card holds; the last one is what is in production.
    versions: Array.from({ length: WORKSPACE_VERSIONS }, (_, index) => {
      const number = index + 1;
      const last = number === WORKSPACE_VERSIONS;
      return {
        id: `v_martial_${number}`,
        graphId: "graph_martial_arts",
        label: `v${number}`,
        lifecycle: last ? "production" : "candidate",
        createdAt: `2026-08-${String(number).padStart(2, "0")}T00:00:00.000Z`,
        note: last ? "Gold-passed martial arts extract" : `Rebuilt after watch ingest ${number}`,
      };
    }),
    pins: [],
    reviews: [
      {
        id: "review_1",
        graphId: "graph_martial_arts",
        edgeId: "rel_judo_developed_by",
        quote: "Judo was developed by Jigoro Kano.",
        confidence: 0.42,
        decision: null,
      },
    ],
    watches: [
      {
        id: "watch_martial",
        graphId: "graph_martial_arts",
        prefix: "data/martial_arts/files",
        paused: false,
        lastWindow: "2026-09-10T00:00:00.000Z",
        pendingFiles: 3,
        goldDrift: 2,
      },
    ],
    // A deployment that has minted a key per pipeline for a while: more than
    // one page of the keys list. Their secrets are random and never shown,
    // so the fixtures cannot be used to authenticate.
    apiKeys: Array.from({ length: SEEDED_KEYS }, (_, index) => {
      const number = index + 1;
      const secret = `fg_${randomBytes(18).toString("hex")}`;
      return {
        id: `key_seed_${number}`,
        name: `${["nightly-eval", "deploy-bot", "notebook"][index % 3]}-${String(number).padStart(2, "0")}`,
        preview: `${secret.slice(0, 7)}…${secret.slice(-4)}`,
        createdAt: `2026-07-${String(number).padStart(2, "0")}T00:00:00.000Z`,
        note: "This path does not use your SSO cookie. A 401 here is not a browser login bug.",
        secretHash: hashApiSecret(secret),
      };
    }),
    grants: [
      { object: "WAREHOUSE FLAKEGRAPH_WH", ok: true, sql: "GRANT USAGE ON WAREHOUSE FLAKEGRAPH_WH TO ROLE APP_OPERATOR;" },
      { object: "DATABASE FLAKEGRAPH", ok: true, sql: "GRANT USAGE ON DATABASE FLAKEGRAPH TO ROLE APP_OPERATOR;" },
      { object: "STAGE APP_STAGE", ok: true, sql: "GRANT READ, WRITE ON STAGE FLAKEGRAPH.PUBLIC.APP_STAGE TO ROLE APP_OPERATOR;" },
      { object: "CORTEX", ok: true, sql: "GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE APP_OPERATOR;" },
    ],
  });
}

/** Workspace versions of the martial arts graph: a page of ten and a couple more. */
const WORKSPACE_VERSIONS = 12;

/** Machine keys on the seeded control plane: a page of 25 and a few more. */
const SEEDED_KEYS = 30;

await main();
