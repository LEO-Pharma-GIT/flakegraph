"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { FolderOpen, Upload } from "lucide-react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { FolderEntityCard } from "@/components/console/folder-entity-card";
import { GuideCard } from "@/components/console/guide-card";
import { OntologyEditor, ontologyProblem } from "@/components/console/ontology-editor";
import { PageHeader } from "@/components/console/page-header";
import { SnowflakePreflightCard } from "@/components/console/operator-tools";
import {
  DEFAULT_PROVIDER_PARALLELISM,
  type Capability,
  type IngestionRequest,
  type OntologySelection,
  type ProviderSelection,
  type RunPriority,
  type RuntimeMode,
  type SourceKind,
  type StorageKind,
} from "@/server/protocol/schema";
import {
  CORTEX_EMBEDDING_MODELS_BY_WIDTH,
  cortexEmbeddingForWidth,
  defaultProvider,
  fleetSelection,
  providerOptions,
  providerSelection,
  selectionFromOption,
  type ProviderOption,
} from "@/server/providers";
import { useStorageItem, writeStorage } from "@/lib/browser-storage";
import {
  pickedFromDrop,
  pickedFromList,
  sortPicked,
  UPLOAD_MAX_FILE_BYTES,
  uploadBatches,
  type PickedFile,
} from "@/lib/dropped-files";
import { uploadedPath } from "@/lib/folder-entity";
import { formatMegabytes, postUploadWithRetry, UploadCancelled } from "@/lib/upload-client";
import { useLastIngestion, writeLastIngestion, type LastIngestionDraft } from "@/lib/last-ingestion";
import { useDebouncedValue } from "@/lib/use-debounced-value";
import { cn, formatBytes } from "@/lib/utils";

/**
 * The form builds a new version of an existing graph instead of a new graph:
 * the graph's identity, name and vocabulary are the base run's, the documents
 * it keeps are decided elsewhere, and this form only adds to them.
 */
export interface RevisionTarget {
  baseRunId: string;
  graphId: string;
  graphName: string | null;
  dropFileIds: string[];
  keptCount: number;
  /** What the base version extracted, when the run recorded it. */
  ontology: OntologySelection | null;
  /** The documents, by file id, the new version ties to no folder, when that changes. */
  folderExclusions: string[] | null;
}

interface IngestionFormProps {
  runtime: RuntimeMode;
  capabilities: Set<Capability>;
  lastSuccessRuntime?: string | null;
  suggestionMode?: "off" | "on-request" | "auto-fill";
  revision?: RevisionTarget | null;
  onSubmitted?: (runId: string) => Promise<void> | void;
}

/** A hosted corpus the session says is on this host. */
type SamplePack = { name: string; path: string; graphName: string; why: string; ontology: OntologySelection | null };

/** The document source on the form: a source kind, or a sample pack standing in for a local path. */
type SourceChoice = SourceKind | "sample";

const EMPTY_ONTOLOGY: OntologySelection = { entityTypes: [], relationTypes: [], relations: "guided" };

/** The credential slot a fleet's finalizer reads its Snowflake password from. */
const FLEET_SNOWFLAKE_PASSWORD_SLOT = "KG_SNOWFLAKE_PASSWORD";

export function IngestionForm({
  runtime,
  capabilities,
  lastSuccessRuntime,
  suggestionMode = "on-request",
  revision = null,
  onSubmitted,
}: IngestionFormProps) {
  const session = trpc.auth.session.useQuery();
  // Sample packs are offered only when their files are on the host the
  // runtime reads from; the fleet image carries the small one, a laptop
  // checkout both, a bare install none.
  const samplePacks: readonly SamplePack[] = session.data?.samplePacks ?? [];
  const isSamplePath = (path: string) => samplePacks.some((sample) => sample.path === path);
  // A failed start may have recorded the run, so the next attempt takes a new id.
  const [jobId, setJobId] = useState(() => cryptoRandom("run"));
  const [graphId] = useState(() => revision?.graphId ?? cryptoRandom("graph"));
  const [graphName, setGraphName] = useState(revision?.graphName ?? "");
  const [choice, setChoice] = useState<SourceChoice>(defaultSourceKind(capabilities));
  const [sourcePath, setSourcePath] = useState("");
  const [uploadPath, setUploadPath] = useState<string | null>(null);
  const [uploadJobId, setUploadJobId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  // Cancels the upload under way.
  const uploadAbort = useRef<AbortController | null>(null);
  const [uploadedCount, setUploadedCount] = useState(0);
  // Where each uploaded file sits below the upload folder, for the folder preview.
  const [uploadedPaths, setUploadedPaths] = useState<readonly string[]>([]);
  const utils = trpc.useUtils();
  const [azure, setAzure] = useState({ accountUrl: "", container: "", prefix: "" });
  const [s3, setS3] = useState({ bucket: "", prefix: "", endpointUrl: "", region: "" });
  const [stage, setStage] = useState({ stage: "", prefix: "" });
  const [outputKind, setOutputKind] = useState<StorageKind>(capabilities.has("spcs") ? "snowflake" : "local_files");
  const [ocr, setOcr] = useState<ProviderSelection>(() => defaultProvider("ocr", runtime));
  const [llm, setLlm] = useState(() => defaultProvider("llm", runtime));
  const [embedding, setEmbedding] = useState(() => defaultProvider("embedding", runtime));
  const [parallelism, setParallelism] = useState(DEFAULT_PROVIDER_PARALLELISM);
  const [priority, setPriority] = useState<RunPriority>("bulk");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [preview, setPreview] = useState<string>("");
  const [snowflake, setSnowflake] = useState({
    account: "",
    user: "",
    database: "",
    schema: "",
    warehouse: "",
    bulkStage: "",
    role: "",
    host: "",
    authenticator: "",
    credentialEnvironmentVariable: "KG_SNOWFLAKE_PASSWORD",
    credentialField: "password",
  });
  const sourceKind: SourceKind = choice === "sample" ? "local_path" : choice;

  // On the fleet the workers' own profile decides the providers: a worker
  // claims only a run whose semantic configuration hashes to its own. The
  // card shows what the fleet runs; the choice is the deployment's.
  const fleet = trpc.fleet.profile.useQuery(undefined, { enabled: runtime === "kubernetes" });
  const fleetProfile = runtime === "kubernetes" ? fleet.data ?? null : null;
  const effectiveOcr = fleetProfile ? fleetSelection(fleetProfile, "ocr", ocr) : ocr;
  const effectiveLlm = fleetProfile ? fleetSelection(fleetProfile, "llm", llm) : llm;
  const effectiveEmbedding = fleetProfile ? fleetSelection(fleetProfile, "embedding", embedding) : embedding;
  const fleetSnowflake = fleetProfile ? fleetSnowflakeAccount(fleetProfile) : null;

  // What the run extracts: the runtime's default vocabulary until it is
  // changed here - or, while a sample pack is the source, the pack's own, so
  // the run is scored by its gold in the terms the gold is written in. A
  // revision keeps the base version's. The server knows where its default
  // profile is; the form never names a server path.
  const defaultOntology = trpc.ingestion.defaultOntology.useQuery(undefined, {
    enabled: Boolean(session.data) && !revision,
  });
  const pack = choice === "sample" ? samplePacks.find((sample) => sample.path === sourcePath) ?? null : null;
  const ontologyDefaults = pack?.ontology
    ? { selection: pack.ontology, source: `the ${pack.name} pack's ontology` }
    : defaultOntology.data
      ? {
          selection: {
            entityTypes: defaultOntology.data.entityTypes,
            relationTypes: defaultOntology.data.relationTypes,
            relations: defaultOntology.data.relations,
          },
          source: defaultOntology.data.source,
        }
      : null;
  const [edited, setEdited] = useState<OntologySelection | null>(null);
  // Until someone edits the vocabulary (or clones one), it follows the
  // defaults, including when they arrive or the runtime changes. Derived
  // while rendering rather than from an effect, so the form never paints
  // an empty vocabulary first.
  const [followingDefaults, setFollowingDefaults] = useState(true);
  const [defaultsSeen, setDefaultsSeen] = useState<string | null>(null);
  const defaultsKey = ontologyDefaults ? `${runtime}:${JSON.stringify(ontologyDefaults.selection)}` : null;
  if (defaultsKey && defaultsSeen !== defaultsKey && ontologyDefaults) {
    setDefaultsSeen(defaultsKey);
    if (followingDefaults) {
      setEdited(ontologyDefaults.selection);
    }
  }
  function setOntology(next: OntologySelection) {
    setEdited(next);
    // Reset to the defaults is following them again: a later source or
    // runtime change moves the vocabulary with it.
    setFollowingDefaults(JSON.stringify(next) === JSON.stringify(ontologyDefaults?.selection ?? null));
  }
  // A revision's vocabulary is its base version's, read from that run; it
  // is shown, never edited.
  const ontology = revision ? revision.ontology : edited;
  const effectiveOntology = ontology ?? EMPTY_ONTOLOGY;

  // Each step reports its own failure where it ran, so these stay quiet.
  const quiet = { onError: () => undefined };
  const preflight = trpc.ingestion.preflight.useMutation(quiet);
  // The width the graph tables already store decides which embedding model a
  // Snowflake run can write at all.
  const storedEmbeddingWidth = trpc.ingestion.embeddingWidth.useQuery(undefined, {
    enabled: runtime === "snowflake",
    staleTime: 5 * 60_000,
  });
  // Inside Snowflake the app lists its own stages, their folders and the Cortex
  // models, so the form offers them instead of asking for names typed exactly.
  const documentStages = trpc.ingestion.documentStages.useQuery(undefined, {
    enabled: runtime === "snowflake",
    staleTime: 60_000,
  });
  const stageFolders = trpc.ingestion.stageFolders.useQuery(
    { stage: stage.stage },
    { enabled: runtime === "snowflake" && Boolean(stage.stage), staleTime: 60_000, retry: false },
  );
  const modelChoices = trpc.ingestion.modelChoices.useQuery(undefined, {
    enabled: runtime === "snowflake",
    staleTime: 10 * 60_000,
  });
  const embeddingChoices = useMemo(() => {
    const width = storedEmbeddingWidth.data;
    const all = modelChoices.data?.embedding ?? [];
    // Only a model of the stored width can be written to the graph tables.
    return typeof width === "number" ? all.filter((choice) => choice.dimension === width) : all;
  }, [modelChoices.data, storedEmbeddingWidth.data]);
  const previewConfig = trpc.ingestion.config.preview.useMutation(quiet);
  const submit = trpc.runs.submit.useMutation(quiet);
  const scanPii = trpc.ingestion.scanPii.useMutation(quiet);
  const ackPii = trpc.ingestion.acknowledgePii.useMutation({
    onSuccess: () => {
      if (request) {
        scanPii.mutate({ source: request.source, sourceKey });
      }
    },
    onError: (error) => toast.error(error.message),
  });
  // Which step of Start is running, so the button says so.
  const [starting, setStarting] = useState<"pii" | "preflight" | "submit" | null>(null);
  // What the run would read, named before it starts: local paths list at
  // once; a bucket or container lists once its fields stop changing, through
  // the pipeline's own listing. A Snowflake stage is only known to the
  // Snowflake runtime.
  const browseSource = useDebouncedValue(browsableSource(), 600);
  // No placeholder from the previous source: a count belongs to the source
  // it was read from, and Start must not open on another bucket's listing.
  const sources = trpc.ingestion.sources.summary.useQuery(
    { source: browseSource ?? { kind: "local", path: "" } },
    { enabled: browseSource !== null },
  );

  // A provider one runtime offers may be unreachable from another; Snowflake
  // workers reach only Cortex. Keep what still applies, reset what does not.
  // Adjusted while rendering rather than in an effect, so no render ever shows a
  // provider the runtime cannot reach.
  const [providersRuntime, setProvidersRuntime] = useState(runtime);
  if (providersRuntime !== runtime) {
    setProvidersRuntime(runtime);
    const keep = (kind: "ocr" | "llm" | "embedding") => (current: ProviderSelection) =>
      providerOptions(kind, runtime).some((option) => option.name === current.provider)
        ? current
        : defaultProvider(kind, runtime, storedEmbeddingWidth.data ?? null);
    setOcr(keep("ocr"));
    setLlm(keep("llm"));
    setEmbedding(keep("embedding"));
  }

  // Once the account says how wide its graph tables are, a Cortex model the
  // form picked by default is swapped for the one whose vectors fit. A model the
  // person typed themselves is left alone; preflight will say if it does not fit.
  const storedWidth = runtime === "snowflake" ? (storedEmbeddingWidth.data ?? null) : null;
  const [fittedWidth, setFittedWidth] = useState<number | null>(null);
  if (storedWidth && storedWidth !== fittedWidth) {
    setFittedWidth(storedWidth);
    setEmbedding((current) => {
      const defaultModels = Object.values(CORTEX_EMBEDDING_MODELS_BY_WIDTH);
      if (current.provider !== "snowflake_cortex" || (current.model && !defaultModels.includes(current.model))) {
        return current;
      }
      return cortexEmbeddingForWidth(storedWidth);
    });
  }

  const lastDraft = useLastIngestion(runtime);
  // A draft is copied into the fields once per runtime: when the workspace
  // asks for it, or when a run page left a clone waiting. Both are derived
  // while rendering rather than from an effect, so the form never paints its
  // empty state first.
  const cloneWaiting = useStorageItem("session", "flakegraph.pending-clone") === "1";
  const [clonedFor, setClonedFor] = useState<string | null>(null);
  if (lastDraft && (suggestionMode === "auto-fill" || cloneWaiting) && clonedFor !== runtime) {
    setClonedFor(runtime);
    applyClone(lastDraft);
  }
  useEffect(() => {
    if (cloneWaiting && (clonedFor === runtime || !lastDraft)) {
      writeStorage("session", "flakegraph.pending-clone", null);
    }
  }, [cloneWaiting, clonedFor, lastDraft, runtime]);

  const envConfirmed = useStorageItem("session", "flakegraph.runtime-ack") === runtime;

  function confirmEnvironment() {
    writeStorage("session", "flakegraph.runtime-ack", runtime);
  }


  const request = useMemo((): IngestionRequest | null => {
    // Graph artifacts live under the control plane's own state, the one place
    // it can write on a read-only image. The server fills the path in from
    // the graph id; the browser does not know, or need, where that is.
    const workspacePath = "";
    const source = sourcePayload();
    // A revision that removes documents, or changes which are tied to their
    // folder, needs no source: it keeps what it keeps and adds nothing. One
    // that changes nothing must add something.
    const dropsOnly = Boolean(
      revision && (revision.dropFileIds.length > 0 || revision.folderExclusions !== null) && !source,
    );
    if (!source && !dropsOnly) {
      return null;
    }
    // On a fleet the account and credential are the fleet's; the form names
    // only where inside that account the graph lands.
    const snowflakeTarget =
      outputKind === "snowflake"
        ? fleetSnowflake
          ? {
              ...snowflake,
              account: fleetSnowflake.account,
              user: fleetSnowflake.user,
              host: fleetSnowflake.host,
              authenticator: fleetSnowflake.authenticator,
              role: snowflake.role || null,
              credentialEnvironmentVariable: FLEET_SNOWFLAKE_PASSWORD_SLOT,
              credentialField: "password",
            }
          : {
              ...snowflake,
              role: snowflake.role || null,
              host: snowflake.host || null,
              authenticator: snowflake.authenticator || null,
              credentialEnvironmentVariable: snowflake.credentialEnvironmentVariable || null,
              credentialField: snowflake.credentialField || null,
            }
        : null;
    return {
      runtime,
      jobId,
      graphId,
      graphName: graphName.trim() || null,
      sourceKind,
      source: source ?? {},
      revision: revision
        ? {
            baseRunId: revision.baseRunId,
            dropFileIds: revision.dropFileIds,
            addDocuments: Boolean(source),
            ...(revision.folderExclusions !== null ? { folderExclusions: revision.folderExclusions } : {}),
          }
        : null,
      ocr: effectiveOcr,
      llm: effectiveLlm,
      embedding: effectiveEmbedding,
      output: { kind: outputKind, workspacePath, snowflake: snowflakeTarget },
      baseConfigPath: null,
      includeGlobs: ["**/*"],
      cacheProvider: capabilities.has("spcs") ? "snowflake" : "local",
      providerParallelism: parallelism,
      runtimeOptions: {},
      ontology: ontology && ontology.entityTypes.length ? ontology : null,
      priority,
    };

    function sourcePayload(): Record<string, unknown> | null {
      if (choice === "upload") {
        if (!uploadPath) {
          return null;
        }
        return { kind: "local", path: uploadPath };
      }
      if (choice === "local_path" || choice === "sample") {
        if (!sourcePath.trim()) {
          return null;
        }
        return { kind: "local", path: sourcePath.trim() };
      }
      if (choice === "azure_blob") {
        if (!azure.accountUrl.trim() || !azure.container.trim()) {
          return null;
        }
        return { kind: "azure_blob", ...azure };
      }
      if (choice === "s3") {
        if (!s3.bucket.trim()) {
          return null;
        }
        return { kind: "s3", ...s3 };
      }
      if (!stage.stage.trim()) {
        return null;
      }
      return { kind: "snowflake_stage", ...stage };
    }
  }, [
    runtime,
    graphName,
    revision,
    choice,
    sourceKind,
    sourcePath,
    uploadPath,
    azure,
    s3,
    stage,
    outputKind,
    effectiveOcr,
    effectiveLlm,
    effectiveEmbedding,
    parallelism,
    priority,
    ontology,
    snowflake,
    fleetSnowflake,
    jobId,
    graphId,
    capabilities,
  ]);
  // Identifies the exact request a preflight answer was given for.
  const requestKey = request ? JSON.stringify(request) : null;
  const [checkedKey, setCheckedKey] = useState<string | null>(null);
  const latestPreflightKey = useRef<string | null>(null);

  // A runtime with different capabilities changes what the form may hold:
  // the output moves to the runtime's own storage, and a local folder or the
  // sample pack is no longer a source.
  const [capabilitiesSeen, setCapabilitiesSeen] = useState(capabilities);
  if (capabilitiesSeen !== capabilities) {
    setCapabilitiesSeen(capabilities);
    setOutputKind(capabilities.has("spcs") ? "snowflake" : "local_files");
  }
  if (choice === "sample" && (!capabilities.has("local") || (session.data && samplePacks.length === 0))) {
    setChoice(defaultSourceKind(capabilities));
    setSourcePath("");
  }
  if (!capabilities.has("local") && choice === "local_path") {
    setChoice(defaultSourceKind(capabilities));
    setSourcePath("");
  }
  // A Snowflake destination needs an account to write into: the fleet's on
  // Kubernetes, the form's elsewhere. When the fleet turns out to hold none,
  // the destination falls back rather than submitting an unwritable run.
  const snowflakeAvailable =
    runtime === "snowflake" || runtime === "local" || (runtime === "kubernetes" && fleetSnowflake !== null);
  if (outputKind === "snowflake" && fleet.isFetched && !snowflakeAvailable) {
    setOutputKind("local_files");
  }

  function resolvedPath() {
    return choice === "upload" ? uploadPath ?? "" : sourcePath;
  }

  function browsableSource(): Record<string, unknown> | null {
    if (choice === "upload" || choice === "local_path" || choice === "sample") {
      return resolvedPath() ? { kind: "local", path: resolvedPath() } : null;
    }
    if (choice === "s3") {
      return s3.bucket.trim() ? { kind: "s3", ...s3 } : null;
    }
    if (choice === "azure_blob") {
      return azure.accountUrl.trim() && azure.container.trim() ? { kind: "azure_blob", ...azure } : null;
    }
    return runtime === "snowflake" && stage.stage.trim() ? { kind: "snowflake_stage", ...stage } : null;
  }

  function chooseSource(next: SourceChoice) {
    if (next === choice) {
      return;
    }
    if (choice === "sample" || next === "sample") {
      setSourcePath("");
    }
    if (next !== "upload") {
      setUploadPath(null);
      setUploadJobId(null);
      setUploadedCount(0);
    }
    setChoice(next);
    if (next === "sample") {
      const selected = samplePacks.find((sample) => sample.path === sourcePath) ?? samplePacks[0];
      if (selected) {
        applySample(selected);
      }
    }
  }

  function applySample(sample: SamplePack) {
    setChoice("sample");
    setSourcePath(sample.path);
    setGraphName((current) => {
      const fromPack = samplePacks.some((item) => item.graphName === current);
      return !current.trim() || fromPack ? sample.graphName : current;
    });
  }

  function applyClone(draft: LastIngestionDraft) {
    setGraphName(draft.graphName);
    const kind = (draft.sourceKind as SourceKind) || "local_path";
    const source = draft.source ?? {};
    const field = (name: string, fallback = "") => (typeof source[name] === "string" ? source[name] : fallback);
    setSourcePath(draft.sourcePath);
    if (kind === "upload") {
      setUploadPath(draft.sourcePath);
      setChoice("upload");
    } else if (kind === "local_path" && isSamplePath(draft.sourcePath) && capabilities.has("local")) {
      setChoice("sample");
    } else {
      setChoice(kind);
    }
    if (kind === "s3") {
      setS3({
        bucket: field("bucket"),
        prefix: field("prefix", draft.sourcePath),
        endpointUrl: field("endpointUrl"),
        region: field("region"),
      });
    } else if (kind === "azure_blob") {
      setAzure({
        accountUrl: field("accountUrl"),
        container: field("container"),
        prefix: field("prefix", draft.sourcePath),
      });
    } else if (kind === "snowflake_stage") {
      setStage({ stage: field("stage", draft.sourcePath), prefix: field("prefix") });
    }
    const width = storedEmbeddingWidth.data ?? null;
    setOcr(providerSelection("ocr", draft.ocrProvider, runtime, width));
    setLlm(providerSelection("llm", draft.llmProvider, runtime, width));
    setEmbedding(
      runtime === "snowflake" && draft.embeddingProvider === "snowflake_cortex"
        ? cortexEmbeddingForWidth(width)
        : providerSelection("embedding", draft.embeddingProvider, runtime, width),
    );
    if (draft.ontology) {
      setOntology(draft.ontology);
    }
  }

  async function onUpload(picked: readonly PickedFile[]) {
    const { upload, ignored, tooLarge } = sortPicked(picked);
    if (tooLarge.length > 0) {
      toast.error(
        `Left out ${tooLarge.length} file${tooLarge.length === 1 ? "" : "s"} over ${UPLOAD_MAX_FILE_BYTES / (1024 * 1024)} MB: ${tooLarge
          .slice(0, 3)
          .map((item) => item.path)
          .join(", ")}${tooLarge.length > 3 ? "…" : ""}`,
      );
    }
    if (upload.length === 0) {
      if (picked.length > 0 && tooLarge.length === 0) {
        toast.error("Nothing to upload: the drop held only hidden or system files.");
      }
      return;
    }
    setChoice("upload");
    const totalBytes = upload.reduce((sum, item) => sum + item.file.size, 0);
    const cancel = new AbortController();
    uploadAbort.current = cancel;
    // Bytes of finished batches, plus what each batch in flight has sent so far.
    let doneBytes = 0;
    let doneFiles = 0;
    const inFlight = new Map<number, number>();
    const report = () =>
      setUploadProgress({
        doneFiles,
        totalFiles: upload.length,
        sentBytes: doneBytes + [...inFlight.values()].reduce((sum, value) => sum + value, 0),
        totalBytes,
      });
    report();
    // Every batch lands in one upload folder: the first mints it (unless an
    // earlier drop did), and the rest go up a few at a time into it.
    let jobId = uploadJobId;
    let failure: string | null = null;
    const send = async (batch: PickedFile[], index: number) => {
      const batchBytes = batch.reduce((sum, item) => sum + item.file.size, 0);
      try {
        const answer = await postUploadWithRetry(
          "/api/uploads",
          () => {
            const body = new FormData();
            for (const item of batch) {
              body.append("files", item.file);
              body.append("paths", item.path);
            }
            if (jobId) {
              body.append("jobId", jobId);
            }
            return body;
          },
          {
            signal: cancel.signal,
            // The multipart body is a little larger than the files; never
            // count more than the batch holds.
            onProgress: (sent) => {
              inFlight.set(index, Math.min(sent, batchBytes));
              report();
            },
          },
        );
        setUploadPath(answer.path!);
        if (answer.jobId && !jobId) {
          jobId = answer.jobId;
          setUploadJobId(answer.jobId);
        }
        const count = answer.count ?? batch.length;
        setUploadedPaths((current) => [...current, ...batch.map((item) => uploadedPath(item.path))]);
        doneFiles += count;
        doneBytes += batchBytes;
        setUploadedCount((current) => current + count);
      } catch (error) {
        failure ??= error instanceof UploadCancelled ? "cancelled" : error instanceof Error ? error.message : String(error);
      } finally {
        inFlight.delete(index);
        report();
      }
    };
    try {
      const batches = uploadBatches(upload);
      let next = 0;
      if (!jobId) {
        next = 1;
        await send(batches[0]!, 0);
      }
      const lane = async () => {
        while (failure === null && next < batches.length) {
          const index = next++;
          await send(batches[index]!, index);
        }
      };
      await Promise.all(Array.from({ length: UPLOAD_LANES }, lane));
      if (failure !== null) {
        const kept = doneFiles > 0 ? ` ${doneFiles.toLocaleString()} of ${upload.length.toLocaleString()} files are in the folder.` : "";
        if (failure === "cancelled") {
          toast.message(`Upload cancelled.${kept}`);
        } else {
          toast.error(`Upload stopped: ${failure}.${kept} Drop the rest again to add them.`);
        }
        return;
      }
      const folders = new Set(upload.map((item) => item.path.split("/").slice(0, -1).join("/")).filter(Boolean)).size;
      toast.success(
        `${uploadJobId ? "Added" : "Uploaded"} ${doneFiles.toLocaleString()} file${doneFiles === 1 ? "" : "s"}` +
          (folders > 0 ? ` from ${folders} folder${folders === 1 ? "" : "s"}` : "") +
          (ignored > 0 ? ` · skipped ${ignored} hidden or system file${ignored === 1 ? "" : "s"}` : ""),
      );
    } finally {
      uploadAbort.current = null;
      setUploadProgress(null);
      // The folder's path does not change as files arrive, so its count is
      // read again rather than kept from the first batch.
      if (doneFiles > 0) {
        void utils.ingestion.sources.summary.invalidate();
        // A later drop lands in the same folder, so the request does not change
        // and the account has to be asked again to see the new files.
        if (uploadJobId && runtime === "snowflake" && request && requestKey) {
          void runSnowflakePreflight(request, requestKey);
        }
      }
    }
  }

  async function onPreview() {
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    try {
      const yaml = await previewConfig.mutateAsync(request);
      setPreview(yaml);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Unable to preview configuration");
    }
  }

  async function runSnowflakePreflight(target: IngestionRequest, key: string) {
    latestPreflightKey.current = key;
    try {
      await preflight.mutateAsync(target);
    } catch {
      // The card shows the mutation's error state; nothing passes without a result.
    } finally {
      if (latestPreflightKey.current === key) {
        setCheckedKey(key);
      }
    }
  }

  async function onPreflight() {
    if (runtime === "snowflake" && request && requestKey) {
      await runSnowflakePreflight(request, requestKey);
      return;
    }
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    try {
      setPreflightRanFor(preflightInputs);
      const result = await preflight.mutateAsync(request);
      if (result.ok) {
        toast.success("Preflight passed. The run is ready to submit.");
      } else {
        toast.error(result.errors.join("; ") || "Preflight failed");
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Preflight failed");
    }
  }

  async function onSubmit() {
    if (!request) {
      toast.error("Complete the source selection first");
      return;
    }
    if (lastSuccessRuntime && lastSuccessRuntime !== runtime && !envConfirmed) {
      toast.error(`Confirm switching from ${lastSuccessRuntime} to ${runtime} before Start.`);
      return;
    }
    try {
      if (request.revision?.addDocuments !== false) {
        setStarting("pii");
        const pii = await scanPii.mutateAsync({ source: request.source, sourceKey });
        if (pii.blocked) {
          toast.error("PII hits must be quarantined or acknowledged before embed.");
          return;
        }
        setStarting("preflight");
        setPreflightRanFor(preflightInputs);
        const result = await preflight.mutateAsync(request);
        if (requestKey) {
          latestPreflightKey.current = requestKey;
          setCheckedKey(requestKey);
        }
        if (!result.ok) {
          toast.error(result.errors.join("; ") || "Fix preflight errors before submitting");
          return;
        }
      }
      setStarting("submit");
      const snapshot = await submit.mutateAsync(request);
      writeLastIngestion({
        runtime,
        graphName: graphName.trim(),
        sourceKind,
        sourcePath: resolvedPath(),
        source: request.source,
        ocrProvider: ocr.provider,
        llmProvider: llm.provider,
        embeddingProvider: embedding.provider,
        ...(request.ontology ? { ontology: request.ontology } : {}),
        savedAt: new Date().toISOString(),
      });
      toast.success(
        revision
          ? `Building a new version of ${snapshot.graphName || snapshot.graphId}`
          : `Submitted ${snapshot.graphName || snapshot.graphId}`,
      );
      await onSubmitted?.(snapshot.runId);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Unable to start ingestion");
      setJobId(cryptoRandom("run"));
    } finally {
      setStarting(null);
    }
  }

  const sourceChoices = sourceChoiceOptions(capabilities, samplePacks.length > 0 && !revision);
  const objectCount = sources.data?.count ?? 0;
  const sourceKey = `${runtime}:${sourceKind}:${resolvedPath()}`;
  const estimate = trpc.ingestion.estimate.useQuery(
    {
      objectCount,
      runtime,
      ocrProvider: ocr.provider,
      parallelism,
    },
    { enabled: objectCount > 0, placeholderData: (previous) => previous },
  );
  const totalBytes = sources.data?.sizeBytes ?? 0;
  const sourceNoun = choice === "s3" ? "objects" : choice === "azure_blob" ? "blobs" : "files";
  const sourcePlace = choice === "s3" || choice === "azure_blob" ? "under this prefix" : "at this path";
  const envChanged = Boolean(lastSuccessRuntime && lastSuccessRuntime !== runtime);
  // Inside Snowflake the account itself is asked whether this run can start,
  // and Start waits for its answer about the form as it now stands.
  const snowflakePreflightCurrent = runtime === "snowflake" && Boolean(requestKey) && checkedKey === requestKey;
  const preflightBlocked =
    runtime === "snowflake" && (!snowflakePreflightCurrent || preflight.isPending || !preflight.data?.ok);
  // A source the console can name ahead of the run is held to what it lists:
  // Start waits for the listing, and an empty or failed one keeps it closed
  // for the same reason the run itself would stop.
  // A listing whose retry is paused (the tab lost focus between attempts)
  // has neither answered nor failed yet, so it counts as pending too.
  const browsable = browseSource !== null;
  const listingPending = browsable && (sources.isFetching || sources.isPaused) && !sources.data;
  const emptyLocalListing =
    listingPending ||
    // An upload still under way is not yet the source a run would read.
    (choice === "upload" && (!uploadPath || uploadProgress !== null)) ||
    (browsable && sources.data?.count === 0);
  const listingFailed = browsable && Boolean(sources.error);
  const envBlocked = envChanged && !envConfirmed;
  const incompleteSource = !request;
  const dropsOnly = Boolean(request?.revision && request.revision.addDocuments === false);
  // A version with nothing in it is not a graph: removing every document
  // without adding any is refused here, not after a run.
  const emptyRevision = Boolean(revision && revision.keptCount === 0 && objectCount === 0);
  // The vocabulary must be usable before a run is built with it; a revision
  // keeps its base version's and is not held to this.
  const ontologyBlocked = !revision && ontology !== null && ontologyProblem(ontology) !== null;
  // Hosted in Snowflake, the account and its defaults come with the session;
  // elsewhere the target inside the account has to be named.
  const snowflakeIncomplete =
    outputKind === "snowflake" &&
    runtime !== "snowflake" &&
    (!snowflake.database.trim() ||
      !snowflake.schema.trim() ||
      !snowflake.bulkStage.trim() ||
      (!fleetSnowflake && (!snowflake.account.trim() || !snowflake.user.trim())));
  const startBlocked =
    preflightBlocked ||
    (!dropsOnly && (emptyLocalListing || listingFailed)) ||
    envBlocked ||
    incompleteSource ||
    emptyRevision ||
    ontologyBlocked ||
    snowflakeIncomplete;

  // A preflight verdict describes the inputs it ran against; once any of them
  // changes it is no longer shown, and the next check starts fresh.
  const preflightInputs = JSON.stringify([
    embedding.provider,
    llm.provider,
    ocr.provider,
    parallelism,
    runtime,
    sourceKind,
    sourcePath,
    uploadPath,
    outputKind,
    ontology,
  ]);
  const [preflightRanFor, setPreflightRanFor] = useState<string | null>(null);
  const preflightVerdict = preflightRanFor === preflightInputs ? preflight.data : undefined;

  // Re-check the account whenever the Snowflake request changes, after the
  // typing settles. Only the answer for the latest request is kept.
  useEffect(() => {
    if (runtime !== "snowflake" || !request || !requestKey) {
      return;
    }
    const timer = setTimeout(() => void runSnowflakePreflight(request, requestKey), 700);
    return () => clearTimeout(timer);
    // The key captures every field of the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runtime, requestKey]);

  const kindFields = (
    <>
      {choice === "sample" ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {samplePacks.map((sample) => (
            <ChoiceTile
              key={sample.path}
              name={sample.name}
              title={sample.name}
              description={sample.why}
              selected={sourcePath === sample.path}
              onClick={() => applySample(sample)}
            />
          ))}
        </div>
      ) : null}
      {choice === "upload" ? (
        <FileDropzone
          progress={uploadProgress}
          onCancel={() => uploadAbort.current?.abort()}
          hasFiles={Boolean(uploadPath)}
          uploaded={uploadedCount}
          onFiles={(files) => void onUpload(files)}
        />
      ) : null}
      {choice === "local_path" ? (
        <Field label="Folder path">
          <Input
            aria-label="Folder path"
            placeholder="A folder the workers can read, e.g. /data/contracts"
            value={sourcePath}
            onChange={(event) => setSourcePath(event.target.value)}
          />
        </Field>
      ) : null}
      {choice === "azure_blob" ? (
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Account URL">
            <Input value={azure.accountUrl} onChange={(event) => setAzure({ ...azure, accountUrl: event.target.value })} />
          </Field>
          <Field label="Container">
            <Input value={azure.container} onChange={(event) => setAzure({ ...azure, container: event.target.value })} />
          </Field>
          <Field label="Prefix">
            <Input value={azure.prefix} onChange={(event) => setAzure({ ...azure, prefix: event.target.value })} />
          </Field>
        </div>
      ) : null}
      {choice === "s3" ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Bucket">
            <Input value={s3.bucket} onChange={(event) => setS3({ ...s3, bucket: event.target.value })} />
          </Field>
          <Field label="Prefix">
            <Input value={s3.prefix} onChange={(event) => setS3({ ...s3, prefix: event.target.value })} />
          </Field>
          <Field label="Endpoint">
            <Input value={s3.endpointUrl} onChange={(event) => setS3({ ...s3, endpointUrl: event.target.value })} />
          </Field>
          <Field label="Region">
            <Input value={s3.region} onChange={(event) => setS3({ ...s3, region: event.target.value })} />
          </Field>
        </div>
      ) : null}
      {choice === "snowflake_stage" ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Field label="Stage">
            <ChoiceInput
              label="Stage"
              value={stage.stage}
              placeholder="Choose a stage"
              options={(documentStages.data ?? []).map((choice) => ({ value: choice.stage, label: choice.name }))}
              onChange={(value) => setStage({ stage: value, prefix: "" })}
            />
          </Field>
          <Field label="Prefix">
            <ChoiceInput
              label="Prefix"
              value={stage.prefix}
              placeholder={
                !stage.stage ? "Choose a stage first" : stageFolders.isLoading ? "Listing folders…" : "Choose a folder"
              }
              options={(stageFolders.data ?? []).map((folder) => ({
                value: folder.prefix,
                label: folder.prefix,
                detail: `${folder.documents} document${folder.documents === 1 ? "" : "s"}`,
              }))}
              onChange={(value) => setStage({ ...stage, prefix: value })}
            />
          </Field>
        </div>
      ) : null}
    </>
  );
  const listing = (
    <>
      {browsable ? (
        listingPending ? (
          <p className="text-sm text-muted-foreground">Listing {sourceNoun}…</p>
        ) : sources.data ? (
          <p className="text-sm text-muted-foreground" data-testid="source-count">
            {sources.data.count.toLocaleString()}
            {sources.data.complete ? "" : "+"} selectable {sources.data.count === 1 ? "object" : "objects"}
            {sources.data.count === 0 ? ` ${sourcePlace}` : totalBytes > 0 ? ` · ${formatBytes(totalBytes)}` : ""}
            {sources.data.complete ? "" : " counted so far; the run reads them all"}
          </p>
        ) : null
      ) : choice === "snowflake_stage" ? (
        <p className="text-sm text-muted-foreground">Stage contents are confirmed when the job starts.</p>
      ) : choice === "upload" ? null : (
        <p className="text-sm text-muted-foreground">
          {choice === "s3"
            ? "Objects are listed once a bucket is named."
            : choice === "azure_blob"
              ? "Blobs are listed once an account URL and container are named."
              : choice === "sample"
                ? "Pick a pack to list its files."
                : "Files are listed once a folder is named."}
        </p>
      )}
      {sources.error ? <Alert variant="destructive">{sources.error.message}</Alert> : null}
    </>
  );

  return (
    <div className="flex min-h-full flex-1 flex-col gap-6">
      {revision ? null : (
        <PageHeader
          kicker="Ingestion"
          title="Build a graph"
          description={`Point FlakeGraph at documents, choose what to extract, and start. The job runs on ${runtimeLabel(runtime)} and keeps running if you close the browser.`}
        />
      )}


      <Card>
        <CardHeader>
          <CardTitle>{revision ? "Documents to add" : "Documents"}</CardTitle>
          <CardDescription>
            {revision
              ? "Only these are parsed and extracted; the documents the graph keeps are read from where the earlier run left them. A file the graph already holds is skipped."
              : "Where the documents come from. PDF, markdown, and office documents are parsed; anything else is skipped."}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2" role="group" aria-label="Document source">
            {sourceChoices.map((option) => (
              <button
                key={option.value}
                type="button"
                aria-pressed={choice === option.value}
                onClick={() => chooseSource(option.value)}
                className={cn(
                  "rounded-md border px-3 py-1.5 text-sm transition-colors",
                  choice === option.value
                    ? "border-primary bg-accent font-medium text-foreground ring-1 ring-primary"
                    : "border-border bg-background text-muted-foreground hover:bg-muted/50",
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
          {kindFields}
          {listing}
        </CardContent>
      </Card>

      {revision ? null : (
        <Card>
          <CardHeader>
            <CardTitle>Graph</CardTitle>
            <CardDescription>
              A friendly name is optional; the stable graph ID is assigned on submit. The destination is where the
              graph is written, not where the job runs.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <p className="text-xs text-muted-foreground sm:col-span-2">
              Graph ID <span className="font-mono">{graphId}</span> · minted for this Start
            </p>
            <Field label="Display name">
              <Input
                aria-label="Display name"
                value={graphName}
                onChange={(event) => setGraphName(event.target.value)}
                placeholder="Optional display name"
              />
            </Field>
            <Field label="Destination">
              <Select value={outputKind} onValueChange={(value) => setOutputKind(value as StorageKind)}>
                <SelectTrigger aria-label="Destination">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="local_files">Local files</SelectItem>
                  <SelectItem value="snowflake" disabled={!snowflakeAvailable}>
                    Snowflake
                  </SelectItem>
                </SelectContent>
              </Select>
              {runtime === "kubernetes" && fleet.isFetched && !fleetSnowflake ? (
                <span className="text-xs text-muted-foreground" data-testid="snowflake-unavailable">
                  Snowflake is not set up on this fleet: its workers hold no Snowflake account. Set{" "}
                  <span className="font-mono">snowflake.account</span> and <span className="font-mono">user</span> in
                  the fleet&apos;s processing config and the{" "}
                  <span className="font-mono">{FLEET_SNOWFLAKE_PASSWORD_SLOT}</span> key in its providers Secret.
                </span>
              ) : null}
            </Field>
          </CardContent>
        </Card>
      )}

      {outputKind === "snowflake" && runtime === "snowflake" ? (
        <Card data-testid="snowflake-destination">
          <CardHeader>
            <CardTitle>Snowflake destination</CardTitle>
            <CardDescription>
              The graph is written to this app&apos;s own database and schema. The app and its worker use
              Snowflake-managed OAuth, so no account, password, or token is stored with the run.
            </CardDescription>
          </CardHeader>
        </Card>
      ) : outputKind === "snowflake" ? (
        <Card data-testid="snowflake-destination">
          <CardHeader>
            <CardTitle>Snowflake destination</CardTitle>
            <CardDescription>
              {fleetSnowflake
                ? `The fleet writes as ${fleetSnowflake.user} into account ${fleetSnowflake.account}, with the credential its workers hold. Name where inside that account the graph lands.`
                : "Credentials stay as environment variable references; the value is read where the run executes."}
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 md:grid-cols-3">
            {(fleetSnowflake ? FLEET_SNOWFLAKE_FIELDS : ALL_SNOWFLAKE_FIELDS).map((key) => (
              <Field key={key} label={humanizeField(key)}>
                <Input
                  aria-label={humanizeField(key)}
                  value={snowflake[key]}
                  onChange={(event) => setSnowflake({ ...snowflake, [key]: event.target.value })}
                />
              </Field>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {revision && !revision.ontology ? null : (
        <OntologyEditor
          value={effectiveOntology}
          onChange={setOntology}
          defaults={revision ? null : ontologyDefaults}
          locked={
            revision
              ? "A new version is built with the vocabulary of the version it revises, so the documents it keeps and the ones it adds are typed alike. To extract different types, build a new graph."
              : null
          }
        />
      )}

      {revision && !revision.ontology?.folderEntity ? null : (
        <FolderEntityCard
          ontology={effectiveOntology}
          onChange={(folderEntity) => setOntology({ ...effectiveOntology, folderEntity })}
          paths={choice === "upload" ? uploadedPaths : []}
          locked={
            revision
              ? "A new version ties documents to their folders as the version it revises does. Untie or tie a kept document in the table above."
              : null
          }
        />
      )}

      <Card data-testid="compose-providers">
        <CardHeader>
          <CardTitle>Processing</CardTitle>
          <CardDescription>
            {fleetProfile
              ? `Set by the fleet: its workers run one parser, one model and one embedding model (profile ${fleetProfile.configMap}) and claim only runs built for them. Change the fleet's values to change these.`
              : runtime === "snowflake"
                ? "Inside Snowflake every model call goes to Cortex: Parse Document reads the files, mistral-large2 extracts the graph, and the Arctic embedding model is the one whose width fits the graph tables."
                : "The parser, the language model, and the embedding model this run uses. Adaptive layout reads native document text first and sends scans, figures, and sparse pages to MinerU."}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <ProviderFields title="OCR" options={providerOptions("ocr", runtime)} value={effectiveOcr} onChange={setOcr} locked={Boolean(fleetProfile)} />
          <ProviderFields
            title="LLM"
            options={providerOptions("llm", runtime)}
            value={effectiveLlm}
            onChange={setLlm}
            locked={Boolean(fleetProfile)}
            modelChoices={runtime === "snowflake" ? modelChoices.data?.llm : undefined}
          />
          <ProviderFields
            title="Embeddings"
            options={providerOptions("embedding", runtime)}
            value={effectiveEmbedding}
            onChange={setEmbedding}
            dimension
            locked={Boolean(fleetProfile)}
            embeddingChoices={runtime === "snowflake" ? embeddingChoices : undefined}
          />
        </CardContent>
      </Card>

      {runtime === "snowflake" ? (
        <SnowflakePreflightCard
          result={preflight.data}
          pending={preflight.isPending}
          current={snowflakePreflightCurrent}
          ready={Boolean(request)}
          error={preflight.error?.message ?? null}
          onRun={() => void onPreflight()}
        />
      ) : null}

      {envChanged && !envConfirmed ? (
        <GuideCard
          tone="warning"
          title="This Start uses a different environment"
          why={`The last job on this control plane ran on ${runtimeLabel(lastSuccessRuntime ?? "local")}. This Start will use ${runtimeLabel(runtime)}. Confirm in the bar at the bottom of this page.`}
          actions={[]}
        />
      ) : null}

      <div>
        <button
          type="button"
          aria-expanded={advancedOpen}
          className="flex w-full items-center justify-between rounded-lg border border-border bg-card px-4 py-3 text-left text-sm font-semibold hover:bg-muted/40"
          onClick={() => setAdvancedOpen((open) => !open)}
        >
          {capabilities.has("distributed") ? "Advanced: parallelism and priority" : "Advanced: parallelism"}
          <span className="text-xs font-normal text-muted-foreground" aria-hidden="true">
            {advancedOpen ? "Hide" : "Show"}
          </span>
        </button>
        {advancedOpen ? (
          <div className="mt-3 space-y-4 rounded-lg border border-border bg-card p-4">
            <Field label="Provider calls in flight">
              <Input
                type="number"
                min={1}
                max={64}
                aria-label="Provider calls in flight"
                value={parallelism}
                onChange={(event) => setParallelism(Number(event.target.value))}
              />
            </Field>
            {capabilities.has("distributed") ? (
              <Field label="Priority">
                <div className="grid gap-2 sm:grid-cols-2">
                  {RUN_PRIORITIES.map((option) => (
                    <ChoiceTile
                      key={option.value}
                      name={option.title}
                      title={option.title}
                      description={option.description}
                      selected={priority === option.value}
                      onClick={() => setPriority(option.value)}
                    />
                  ))}
                </div>
              </Field>
            ) : null}
          </div>
        ) : null}
      </div>

      {preflightVerdict && runtime !== "snowflake" ? (
        <Alert variant={preflightVerdict.ok ? "success" : "destructive"}>
          {preflightVerdict.ok
            ? "Preflight passed. The run is ready to submit."
            : preflightVerdict.errors.join(" ")}
        </Alert>
      ) : null}

      {scanPii.data?.hits.length ? (
        scanPii.data.blocked ? (
          <Alert variant="destructive">
            PII {summarizeHits(scanPii.data.hits)}. Embeddings will not be
            written until you acknowledge residual risk.
            <Button className="ml-2" size="sm" variant="secondary" pending={ackPii.isPending} onClick={() => ackPii.mutate({ sourceKey })}>
              {ackPii.isPending ? "Acknowledging…" : "Acknowledge residual risk"}
            </Button>
          </Alert>
        ) : (
          // What was found stays on screen after the acknowledgement, so the
          // person starting the run still sees what it carries.
          <Alert variant="warning" data-testid="pii-acknowledged">
            PII {summarizeHits(scanPii.data.hits)} · residual risk acknowledged
            for this source.
          </Alert>
        )
      ) : null}

      {preview ? (
        <Card>
          <CardHeader>
            <CardTitle>Effective configuration</CardTitle>
            <CardDescription>Secrets are redacted. YAML is a downloadable preview, not the first screen.</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea aria-label="YAML configuration text" readOnly value={preview} rows={16} className="font-mono text-xs" />
          </CardContent>
        </Card>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button variant="outline" onClick={() => void onPreview()} pending={previewConfig.isPending} disabled={!request}>
          {previewConfig.isPending ? "Building preview…" : "Preview configuration"}
        </Button>
        <Button
          variant="outline"
          onClick={() => void onPreflight()}
          pending={preflight.isPending && starting === null}
          disabled={!request || preflight.isPending}
        >
          {preflight.isPending && starting === null ? "Running preflight…" : "Run preflight"}
        </Button>
        <Button
          variant="outline"
          onClick={() =>
            request &&
            scanPii.mutate(
              { source: request.source, sourceKey },
              {
                // A clean scan says so; hits show as the alert above.
                onSuccess: (result) =>
                  result.hits.length === 0 ? toast.success("No PII found in this source.") : undefined,
                onError: (error) => toast.error(error.message),
              },
            )
          }
          pending={scanPii.isPending && starting === null}
          disabled={!request || scanPii.isPending}
        >
          {scanPii.isPending && starting === null ? "Scanning…" : "Scan for PII"}
        </Button>
      </div>

      <div className="sticky bottom-0 z-10 -mx-4 mt-auto flex items-center justify-between gap-3 border-t border-border bg-background/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6">
        <div className="min-w-0 flex-1 space-y-1">
          {preflightBlocked && request ? (
            snowflakePreflightCurrent && !preflight.isPending && (preflight.error || preflight.data?.ok === false) ? (
              <p className="truncate text-sm text-destructive">
                Snowflake preflight failed. Start stays blocked until it passes.
              </p>
            ) : (
              <p className="truncate text-sm text-muted-foreground">Checking the Snowflake account…</p>
            )
          ) : null}
          {envBlocked ? (
            <p className="truncate text-sm text-destructive">
              Last job used {runtimeLabel(lastSuccessRuntime ?? "local")}. Confirm {runtimeLabel(runtime)} before Start.
            </p>
          ) : null}
          {listingFailed ? (
            <p className="truncate text-sm text-destructive">
              Listing failed. Start stays disabled until the source lists {sourceNoun}.
            </p>
          ) : null}
          {ontologyBlocked && ontology ? (
            <p className="truncate text-sm text-destructive">{ontologyProblem(ontology)}</p>
          ) : null}
          {snowflakeIncomplete && !incompleteSource ? (
            <p className="truncate text-sm text-destructive">
              Name the Snowflake database, schema and bulk stage before Start.
            </p>
          ) : null}
          {revision ? (
            <p className="truncate text-sm text-muted-foreground" data-testid="revision-summary">
              Keeps {revision.keptCount} document{revision.keptCount === 1 ? "" : "s"}
              {revision.dropFileIds.length ? ` · removes ${revision.dropFileIds.length}` : ""}
              {objectCount > 0
                ? ` · adds ${objectCount}`
                : emptyRevision
                  ? " · a version needs at least one document: keep one, or add some"
                  : incompleteSource && !revision.dropFileIds.length
                    ? " · add documents or remove some to build a new version"
                    : ""}
            </p>
          ) : null}
          {revision ? null : listingPending ? (
            <p className="truncate text-sm text-muted-foreground">Listing files to estimate cost and time…</p>
          ) : emptyLocalListing || incompleteSource ? (
            <p className="truncate text-sm text-destructive">
              {choice === "upload" && !uploadPath
                ? "Drop files before Start."
                : incompleteSource
                  ? "Fill the required source fields before Start."
                  : `No ${sourceNoun} found ${sourcePlace}. Start stays disabled.`}
            </p>
          ) : null}
          {revision ? null : estimate.data && objectCount > 0 ? (
            <p className="truncate text-sm text-muted-foreground" data-testid="credit-envelope">
              Credit envelope {estimate.data.usdLow}–{estimate.data.usdHigh} usd · {estimate.data.minutesLow}–
              {estimate.data.minutesHigh} min
              <span className="hidden sm:inline">. {estimate.data.note}</span>
            </p>
          ) : !startBlocked ? (
            <p className="truncate text-sm text-muted-foreground">
              {objectCount === 0
                ? sources.isFetching || sources.isPaused
                  ? `Listing ${sourceNoun} to estimate cost and time…`
                  : browsable
                    ? `No ${sourceNoun} found ${sourcePlace}.`
                    : choice === "snowflake_stage"
                      ? "Cost is estimated after Start lists the stage."
                      : "Point at files to estimate cost and time."
                : "Estimating cost and time…"}
            </p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {envBlocked ? (
            <Button variant="secondary" onClick={confirmEnvironment}>
              Confirm environment
            </Button>
          ) : null}
          <Button
            onClick={() => void onSubmit()}
            pending={starting !== null}
            disabled={submit.isPending || preflight.isPending || scanPii.isPending || !session.data || startBlocked}
            title={
              !startBlocked
                ? undefined
                : preflightBlocked && request
                  ? "Start waits for the Snowflake account preflight to pass"
                  : envBlocked
                    ? "Confirm the environment change before Start"
                    : emptyRevision
                      ? "Keep at least one document, or add some"
                      : ontologyBlocked && ontology
                        ? (ontologyProblem(ontology) ?? undefined)
                        : snowflakeIncomplete && !incompleteSource
                          ? "Name the Snowflake database, schema and bulk stage before Start"
                          : emptyLocalListing || incompleteSource
                            ? "Fill a usable source before Start"
                            : undefined
            }
          >
            {starting === "pii"
              ? "Checking for PII…"
              : starting === "preflight"
                ? "Running preflight…"
                : starting === "submit"
                  ? "Starting…"
                  : revision
                    ? "Build new version"
                    : "Start"}
          </Button>
        </div>
      </div>
    </div>
  );
}

type SnowflakeField =
  | "account"
  | "user"
  | "database"
  | "schema"
  | "warehouse"
  | "bulkStage"
  | "role"
  | "host"
  | "authenticator"
  | "credentialEnvironmentVariable"
  | "credentialField";

const ALL_SNOWFLAKE_FIELDS: SnowflakeField[] = [
  "account",
  "user",
  "database",
  "schema",
  "warehouse",
  "bulkStage",
  "role",
  "host",
  "authenticator",
  "credentialEnvironmentVariable",
  "credentialField",
];

/** On a fleet the account, user and credential are the fleet's; only the target is named. */
const FLEET_SNOWFLAKE_FIELDS: SnowflakeField[] = ["database", "schema", "warehouse", "bulkStage", "role"];

function runtimeLabel(runtime: string): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Local";
}

/**
 * A dropdown of the given options, or a text field when there are none (the
 * runtime cannot list them, or listing failed), so a value can always be given.
 * A current value missing from the options stays selectable.
 */
function ChoiceInput({
  label,
  value,
  options,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  options: readonly { value: string; label: string; detail?: string }[];
  placeholder: string;
  onChange: (value: string) => void;
}) {
  if (options.length === 0) {
    return <Input aria-label={label} placeholder={placeholder} value={value} onChange={(event) => onChange(event.target.value)} />;
  }
  const items = value && !options.some((option) => option.value === value) ? [{ value, label: value }, ...options] : options;
  return (
    <Select value={value || undefined} onValueChange={onChange}>
      <SelectTrigger aria-label={label}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {items.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
            {"detail" in option && option.detail ? (
              <span className="ml-2 text-xs text-muted-foreground">{option.detail}</span>
            ) : null}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}

function ChoiceTile({
  name,
  title,
  description,
  selected,
  onClick,
}: {
  name: string;
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={name}
      aria-pressed={selected}
      onClick={onClick}
      className={cn(
        "flex w-full flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors",
        selected ? "border-primary bg-accent ring-2 ring-primary" : "border-border bg-background hover:bg-muted/50",
      )}
    >
      <span className="flex w-full items-center justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        {selected ? (
          <span className="rounded-full bg-primary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary-foreground">
            Selected
          </span>
        ) : null}
      </span>
      <span className="text-xs leading-snug text-muted-foreground">{description}</span>
    </button>
  );
}

/**
 * The fleet's two bands. Bulk is the default: fair share already gives a run
 * its half of the fleet beside a large one, and interactive puts a run ahead
 * of every bulk run, so it is for a run someone is waiting on.
 */
const RUN_PRIORITIES: ReadonlyArray<{ value: RunPriority; title: string; description: string }> = [
  {
    value: "bulk",
    title: "Bulk",
    description: "Shares the fleet equally with every other run.",
  },
  {
    value: "interactive",
    title: "Someone is waiting",
    description: "Claimed ahead of every bulk run. For small runs a person is waiting on.",
  },
];

/** Upload requests in flight at once for a large drop. */
const UPLOAD_LANES = 4;

/** An upload under way: files the server has, and bytes that have left the browser. */
interface UploadProgress {
  doneFiles: number;
  totalFiles: number;
  sentBytes: number;
  totalBytes: number;
}

function FileDropzone({
  onFiles,
  onCancel,
  progress = null,
  hasFiles = false,
  uploaded = 0,
}: {
  onFiles: (files: PickedFile[]) => void;
  onCancel?: () => void;
  /** The upload under way, if one is. */
  progress?: UploadProgress | null;
  hasFiles?: boolean;
  /** Files this session put in the upload folder; a restored draft's are not counted. */
  uploaded?: number;
}) {
  const [over, setOver] = useState(false);
  const [reading, setReading] = useState(false);
  const folderInput = useRef<HTMLInputElement>(null);
  const busy = reading || progress !== null;

  return (
    <div className="space-y-2">
      <label
        data-testid="file-dropzone"
        onDragEnter={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
          setOver(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          if (!event.currentTarget.contains(event.relatedTarget as Node)) {
            setOver(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          const walking = pickedFromDrop(event.dataTransfer);
          setReading(true);
          void walking
            .then((files) => onFiles(files))
            .catch(() => toast.error("Could not read the dropped folder. Try Choose a folder instead."))
            .finally(() => setReading(false));
        }}
        className={cn(
          "flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-8 text-center transition-colors",
          over ? "border-primary bg-accent" : "border-muted-foreground/40 bg-muted/40 hover:border-primary/50 hover:bg-muted/60",
          busy && "pointer-events-none opacity-60",
        )}
      >
        <input
          type="file"
          multiple
          className="sr-only"
          disabled={busy}
          aria-label="Upload documents"
          onChange={(event) => {
            if (event.target.files?.length) {
              onFiles(pickedFromList(event.target.files));
            }
            event.target.value = "";
          }}
        />
        <Upload className="size-5 text-muted-foreground" aria-hidden="true" />
        <p className="text-sm font-medium" aria-live="polite">
          {reading
            ? "Reading folders…"
            : progress
              ? `Uploading ${formatMegabytes(progress.sentBytes)} of ${formatMegabytes(progress.totalBytes)} · ${progress.doneFiles.toLocaleString()} of ${progress.totalFiles.toLocaleString()} files in the folder`
              : hasFiles
                ? uploaded > 0
                  ? `${uploaded.toLocaleString()} file${uploaded === 1 ? "" : "s"} ready · drop more to add them`
                  : "Drop more files or folders, or click to add"
                : "Drop files or folders here, or click to browse"}
        </p>
        {progress ? (
          <Progress
            className="h-1.5 w-72"
            aria-label="Upload progress"
            value={progress.totalBytes ? (progress.sentBytes / progress.totalBytes) * 100 : 0}
          />
        ) : null}
        <p className="text-xs text-muted-foreground">
          PDF, markdown, and office documents. A folder brings every file in it and its subfolders; hidden files are
          left out.
        </p>
      </label>
      <div className="flex justify-center gap-2">
        {progress && onCancel ? (
          <Button type="button" size="sm" variant="outline" onClick={onCancel}>
            Cancel upload
          </Button>
        ) : (
          <Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => folderInput.current?.click()}>
            <FolderOpen className="size-3.5" aria-hidden="true" />
            Choose a folder
          </Button>
        )}
        <input
          ref={folderInput}
          type="file"
          multiple
          className="sr-only"
          tabIndex={-1}
          aria-label="Upload a folder"
          // Outside React's input attributes; browsers support it.
          {...{ webkitdirectory: "" }}
          onChange={(event) => {
            if (event.target.files?.length) {
              onFiles(pickedFromList(event.target.files));
            }
            event.target.value = "";
          }}
        />
      </div>
    </div>
  );
}

function ProviderFields({
  title,
  options,
  value,
  onChange,
  dimension = false,
  locked = false,
  modelChoices,
  embeddingChoices,
}: {
  title: string;
  options: ProviderOption[];
  value: ProviderSelection;
  onChange: (value: ProviderSelection) => void;
  dimension?: boolean;
  locked?: boolean;
  /** Models the runtime offers; without them the model is typed. */
  modelChoices?: readonly string[];
  /** Embedding models with their widths; choosing one sets the width too. */
  embeddingChoices?: readonly { model: string; dimension: number }[];
}) {
  const fixedWidth = Boolean(embeddingChoices?.length);
  const selected = options.find((option) => option.name === value.provider);
  if (locked) {
    return (
      <div className="space-y-1" data-testid={`fleet-provider-${title.toLowerCase()}`}>
        <p className="text-sm font-medium">{title}</p>
        <p className="text-sm">{selected?.label ?? value.provider}</p>
        {value.model ? <p className="text-xs text-muted-foreground">{value.model}</p> : null}
        {dimension && value.dimension ? (
          <p className="text-xs text-muted-foreground">{value.dimension} dimensions</p>
        ) : null}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <p className="text-sm font-medium">{title}</p>
      <Select
        value={value.provider}
        onValueChange={(name) => {
          const option = options.find((item) => item.name === name);
          if (!option) {
            return;
          }
          onChange(selectionFromOption(option));
        }}
      >
        <SelectTrigger aria-label={`${title} provider`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.name} value={option.name}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {selected?.needsModel ? (
        <ChoiceInput
          label={`${title} model`}
          placeholder="Model"
          value={value.model ?? ""}
          options={
            embeddingChoices?.length
              ? embeddingChoices.map((choice) => ({ value: choice.model, label: choice.model, detail: `${choice.dimension}-wide` }))
              : (modelChoices ?? []).map((model) => ({ value: model, label: model }))
          }
          onChange={(model) =>
            onChange({
              ...value,
              model: model || null,
              ...(embeddingChoices?.length
                ? { dimension: embeddingChoices.find((choice) => choice.model === model)?.dimension ?? value.dimension }
                : {}),
            })
          }
        />
      ) : null}
      {selected?.needsEndpoint ? (
        <Input
          aria-label={`${title} endpoint`}
          placeholder="Endpoint"
          value={value.endpoint ?? ""}
          onChange={(event) => onChange({ ...value, endpoint: event.target.value || null })}
        />
      ) : null}
      {selected?.needsApiKey ? (
        <Input
          aria-label={`${title} API key environment variable`}
          placeholder="API key environment variable"
          value={value.apiKeyEnvironmentVariable ?? ""}
          onChange={(event) => onChange({ ...value, apiKeyEnvironmentVariable: event.target.value || null })}
        />
      ) : null}
      {dimension ? (
        <Input
          type="number"
          aria-label={`${title} dimension`}
          placeholder="Dimension"
          // A listed embedding model sets its own width.
          disabled={fixedWidth}
          value={value.dimension ?? ""}
          onChange={(event) => onChange({ ...value, dimension: Number(event.target.value) || null })}
        />
      ) : null}
    </div>
  );
}

/** The Snowflake account a fleet's workers write with, when its profile names one. */
function fleetSnowflakeAccount(profile: { config: Record<string, unknown> }): {
  account: string;
  user: string;
  host: string | null;
  authenticator: string | null;
} | null {
  const values = profile.config.snowflake;
  if (!values || typeof values !== "object" || Array.isArray(values)) {
    return null;
  }
  const record = values as Record<string, unknown>;
  const text = (key: string) =>
    typeof record[key] === "string" && !String(record[key]).startsWith("${") ? String(record[key]) : "";
  if (!text("account") || !text("user")) {
    return null;
  }
  return {
    account: text("account"),
    user: text("user"),
    host: text("host") || null,
    authenticator: text("authenticator") || null,
  };
}

function humanizeField(key: string): string {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (letter) => letter.toUpperCase())
    .trim();
}

function defaultSourceKind(capabilities: Set<Capability>): SourceKind {
  if (capabilities.has("snowflake_stage") && !capabilities.has("local")) {
    return "snowflake_stage";
  }
  if (capabilities.has("upload")) {
    return "upload";
  }
  if (capabilities.has("local")) {
    return "local_path";
  }
  return "upload";
}

function sourceChoiceOptions(capabilities: Set<Capability>, samplePacks: boolean) {
  const options: Array<{ value: SourceChoice; label: string }> = [];
  if (capabilities.has("upload")) {
    options.push({ value: "upload", label: "Upload" });
  }
  if (capabilities.has("local") && samplePacks) {
    options.push({ value: "sample", label: "Sample pack" });
  }
  if (capabilities.has("local")) {
    options.push({ value: "local_path", label: "Folder path" });
  }
  if (capabilities.has("azure_blob")) {
    options.push({ value: "azure_blob", label: "Azure Blob" });
  }
  if (capabilities.has("s3")) {
    options.push({ value: "s3", label: "S3-compatible bucket" });
  }
  if (capabilities.has("snowflake_stage")) {
    options.push({ value: "snowflake_stage", label: "Snowflake stage" });
  }
  return options;
}

function cryptoRandom(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`;
  }
  return `${prefix}_${Math.random().toString(16).slice(2, 14)}`;
}

/** A PII finding list a sentence can carry: the first few, then a count. */
function summarizeHits(hits: ReadonlyArray<{ kind: string; source: string }>): string {
  const shown = hits.slice(0, 5).map((hit) => `${hit.kind} in ${hit.source}`);
  const more = hits.length - shown.length;
  return more > 0 ? `${shown.join("; ")} and ${more.toLocaleString()} more` : shown.join("; ");
}
