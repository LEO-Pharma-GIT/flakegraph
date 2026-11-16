import { z } from "zod";
import { defaultOntologySelection, runOntologySelection } from "../config";
import { ontologySelectionFromProfile } from "../ontology-profile";
import { TRPCError } from "@trpc/server";
import { appEnv } from "../env";
import { availableSamplePacks } from "../sample-packs";
import { IngestionRequest, isSuccessStatus, standardSchema } from "../protocol/schema";
import { availableRuntimes } from "../runtime/resolve";
import { publicApiKeys } from "../api-keys";
import {
  identifiedProcedure,
  operatorProcedure,
  protectedProcedure,
  publicProcedure,
  router,
  runEffect,
  sessionProcedure,
  type TrpcContext,
} from "./init";
import { graphForBrowser } from "../graph";
import { qualityForGraph } from "../quality";
import { GOLD_MAX_BYTES, GoldValidationError, goldTemplate, removeUploadedGold, saveUploadedGold, validateGold } from "../gold-store";
import { inspectFromDataset } from "../inspect-report";
import { AskHttpError } from "../ask/errors";
import { completeAsk } from "../ask/complete";
import { NO_ASK_MODEL, perspectiveScope } from "../ask/load";
import { scanPii } from "../pii";
import { estimateConsumption } from "../estimate";
import { withSkippedFiles } from "../documents";
import {
  acknowledgePii,
  applyIncremental,
  assumeIdentity,
  createApiKey,
  createWatch,
  revokeApiKey,
  decideReview,
  enqueueReviews,
  loadWorkspace,
  markGrant,
  pinTarget,
  promotePerspective,
  publishVersion,
  recordEstimate,
  recordIdentityIncident,
  recordRuntimeSuccess,
  sampleHighConfidenceReviews,
  setSuggestionMode,
  skipFile,
  staffIncident,
  toggleWatch,
  upsertPerspective,
  type WorkspaceState,
} from "../workspace";
import { ontologyCoverage, type GoldGraph } from "../gold";
import { proposeOntologyForIntent } from "../ontology";
import { confineIngestionRequest, confineSource, confinedBaseConfigPath, confinedPath, SAFE_ID } from "../confine";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

/**
 * An identifier that becomes a directory name under the console's state:
 * a run, a graph, an upload. The shape is checked here so a caller gets a
 * plain validation error, and again where the directory is named.
 */
const safeId = z.string().min(1).max(200).regex(SAFE_ID, "may hold only letters, digits, dots, dashes and underscores");
const runIdInput = z.object({ runId: safeId });
const graphIdInput = z.object({ graphId: safeId });
const estimateInput = z.object({
  objectCount: z.number().int().min(0),
  runtime: z.string(),
  ocrProvider: z.string(),
  parallelism: z.number(),
});

/**
 * Write access to the graph a workspace item belongs to. An item that does
 * not exist is left to the workspace function, which reports it.
 */
async function requireItemGraph(
  ctx: TrpcContext,
  find: (state: WorkspaceState) => { graphId: string } | undefined,
): Promise<void> {
  const item = find(await loadWorkspace());
  if (item) {
    await runEffect(ctx.controlPlane.requireGraph(item.graphId, "write"));
  }
}

/** A PII acknowledgement is the person's who made it, not the whole console's. */
function piiKey(viewer: { userName: string }, sourceKey: string): string {
  return `${viewer.userName.trim().toUpperCase()}:${sourceKey}`;
}

/** Serialize every workspace response through the viewer's graph and key access. */
async function visibleWorkspace(ctx: TrpcContext, state: WorkspaceState) {
  const graphs = await runEffect(ctx.controlPlane.visibleGraphs());
  const runs = new Set((await runEffect(ctx.controlPlane.listRuns(500))).map((run) => run.runId));
  const byGraph = <T extends { graphId: string }>(items: readonly T[]) => items.filter((item) => graphs.has(item.graphId));
  const byRun = <T,>(record: Record<string, T>) =>
    Object.fromEntries(Object.entries(record).filter(([runId]) => runs.has(runId)));
  const me = ctx.viewer.userName.trim().toUpperCase();
  // What the console keeps for everyone - other people's upload paths in
  // the last config digests, their PII acknowledgements - is not sent;
  // keys are the viewer's own (and, on a console without sign-in, the
  // ones with no recorded owner).
  const { lastConfigDigest: _digests, piiAcknowledged: _acknowledged, ...shared } = state;
  return {
    ...shared,
    perspectives: byGraph(state.perspectives),
    versions: byGraph(state.versions),
    pins: byGraph(state.pins),
    reviews: byGraph(state.reviews),
    watches: byGraph(state.watches),
    skippedFiles: byRun(state.skippedFiles),
    estimates: byRun(state.estimates),
    apiKeys: publicApiKeys(
      state.apiKeys.filter((key) => (key.owner ? key.owner === me && Boolean(me) : !appEnv().requireSignIn)),
    ),
  };
}

export const appRouter = router({
  auth: router({
    session: publicProcedure.query(async ({ ctx }) => {
      const workspace = await loadWorkspace();
      return {
        viewer: ctx.viewer,
        runtime: ctx.runtimeName,
        capabilities: [...ctx.controlPlane.capabilities],
        identified: Boolean(ctx.viewer.userName),
        identityFromGate: appEnv().trustIdentityHeaders || appEnv().snowflakeHosted,
        signOutUrl: process.env.FLAKEGRAPH_APP_SIGN_OUT_URL?.trim() || null,
        grafanaUrl: process.env.FLAKEGRAPH_APP_GRAFANA_URL?.trim() || null,
        snowflakeHosted: appEnv().snowflakeHosted,
        availableRuntimes: availableRuntimes(),
        demoIdentities: appEnv().demoIdentities,
        authorization: ctx.authorization,
        operator: ctx.operator,
        samplePacks: await availableSamplePacks(appEnv().repositoryRoot),
        role: workspace.role,
        suggestionMode: workspace.suggestionMode,
        lastSuccessRuntime: workspace.lastSuccessRuntime,
        grants: workspace.grants,
      };
    }),
    // A rehearsal device for a console with no gate: only where the
    // deployment said demo identities are wanted, never where a gate speaks.
    // The identity is the whole console's, so a key never changes it.
    assume: operatorProcedure
      .input(
        z.object({
          userName: z.string().max(200),
          email: z.string().max(320).optional(),
          roles: z.array(z.string().max(200)).max(50),
          role: z.enum(["operator", "analyst", "staff"]),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        if (appEnv().trustIdentityHeaders) {
          throw new TRPCError({
            code: "FORBIDDEN",
            message: "Identity comes from the sign-in gate here; it cannot be assumed.",
          });
        }
        if (!appEnv().demoIdentities) {
          throw new TRPCError({
            code: "FORBIDDEN",
            message: "Demo identities are off. Set FLAKEGRAPH_DEMO_IDENTITIES=1 on a laptop console, or put a sign-in gate in front.",
          });
        }
        return visibleWorkspace(ctx, await assumeIdentity(input));
      }),
    setSuggestions: operatorProcedure
      .input(z.object({ mode: z.enum(["off", "on-request", "auto-fill"]) }))
      .mutation(async ({ ctx, input }) => visibleWorkspace(ctx, await setSuggestionMode(input.mode))),
  }),
  runs: router({
    list: publicProcedure
      .input(z.object({ limit: z.number().int().min(1).max(500).optional() }).optional())
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.listRuns(input?.limit ?? 100))),
    get: publicProcedure.input(runIdInput).query(({ ctx, input }) => runEffect(ctx.controlPlane.getRun(input.runId))),
    submit: protectedProcedure.input(standardSchema(IngestionRequest)).mutation(async ({ ctx, input }) => {
      const request = confineIngestionRequest(requiredInput(input));
      const snapshot = await runEffect(ctx.controlPlane.submit(request));
      await recordRuntimeSuccess(
        request.runtime,
        `${request.sourceKind}:${String(
          (request.source as Record<string, unknown>).path ??
            (request.source as Record<string, unknown>).prefix ??
            (request.source as Record<string, unknown>).stage ??
            "",
        )}`,
      );
      return snapshot;
    }),
    cancel: protectedProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.cancel(input.runId)),
    ),
    retry: protectedProcedure.input(runIdInput).mutation(async ({ ctx, input }) => {
      const workspace = await loadWorkspace();
      return runEffect(ctx.controlPlane.retry(input.runId, workspace.skippedFiles[input.runId] ?? []));
    }),
    recover: protectedProcedure.input(runIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.recover(input.runId)),
    ),
    skipFile: protectedProcedure
      .input(z.object({ runId: safeId, fileId: z.string().min(1).max(500) }))
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireRun(input.runId, "write"));
        return visibleWorkspace(ctx, await skipFile(input.runId, input.fileId));
      }),
    // The vocabulary a run was built with, for a revision to keep.
    ontology: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const configPath = snapshot.raw.config_path;
      return runOntologySelection(typeof configPath === "string" ? configPath : null);
    }),
    documents: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const statuses = await runEffect(ctx.controlPlane.documents(input.runId));
      const workspace = await loadWorkspace();
      return withSkippedFiles(statuses, workspace.skippedFiles[input.runId] ?? []);
    }),
  }),
  ingestion: router({
    // The width the destination's graph tables store, where a runtime writes
    // to fixed-width tables; null where nothing fixes one.
    embeddingWidth: protectedProcedure.query(({ ctx }) =>
      ctx.controlPlane.graphEmbeddingWidth ? runEffect(ctx.controlPlane.graphEmbeddingWidth()) : null,
    ),
    // Choices a runtime can list itself, so the form offers them instead of
    // asking for names typed exactly. Null where the runtime has none.
    documentStages: protectedProcedure.query(({ ctx }) =>
      ctx.controlPlane.documentStages ? runEffect(ctx.controlPlane.documentStages()) : null,
    ),
    stageFolders: protectedProcedure
      .input(z.object({ stage: z.string().min(1) }))
      .query(({ ctx, input }) =>
        ctx.controlPlane.stageFolders ? runEffect(ctx.controlPlane.stageFolders(input.stage)) : null,
      ),
    modelChoices: protectedProcedure.query(({ ctx }) =>
      ctx.controlPlane.modelChoices ? runEffect(ctx.controlPlane.modelChoices()) : null,
    ),
    preflight: protectedProcedure.input(standardSchema(IngestionRequest)).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.preflight(confineIngestionRequest(requiredInput(input)))),
    ),
    sources: router({
      list: protectedProcedure
        .input(
          z.object({
            source: z.record(z.string(), z.unknown()),
            limit: z.number().int().min(1).max(5_000).optional(),
          }),
        )
        .query(({ ctx, input }) => runEffect(ctx.controlPlane.listSourceObjects(confineSource(input.source), input.limit))),
      summary: protectedProcedure
        .input(z.object({ source: z.record(z.string(), z.unknown()) }))
        .query(({ ctx, input }) => runEffect(ctx.controlPlane.summarizeSource(confineSource(input.source)))),
    }),
    config: router({
      preview: protectedProcedure.input(standardSchema(IngestionRequest)).mutation(({ ctx, input }) =>
        runEffect(ctx.controlPlane.previewConfig(confineIngestionRequest(requiredInput(input)))),
      ),
    }),
    // A query changes nothing. Keeping an estimate on a run is
    // recordEstimate's, so a runId here is refused rather than ignored.
    estimate: protectedProcedure.input(estimateInput.strict()).query(({ input }) => estimateConsumption(input)),
    // The estimate a run was priced at, kept for its consumption panel to
    // hold the actual spend against.
    recordEstimate: protectedProcedure
      .input(estimateInput.extend({ runId: safeId }))
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireRun(input.runId, "write"));
        const estimate = estimateConsumption(input);
        await recordEstimate(input.runId, estimate);
        return estimate;
      }),
    scanPii: protectedProcedure
      .input(z.object({ source: z.record(z.string(), z.unknown()), sourceKey: z.string() }))
      .mutation(async ({ ctx, input }) => {
        const objects = await runEffect(ctx.controlPlane.listSourceObjects(confineSource(input.source)));
        const hits = await scanPii(objects);
        const workspace = await loadWorkspace();
        return {
          hits,
          blocked: hits.length > 0 && !workspace.piiAcknowledged.includes(piiKey(ctx.viewer, input.sourceKey)),
        };
      }),
    acknowledgePii: protectedProcedure
      .input(z.object({ sourceKey: z.string() }))
      .mutation(async ({ ctx, input }) => visibleWorkspace(ctx, await acknowledgePii(piiKey(ctx.viewer, input.sourceKey)))),
    // What a run extracts unless the form changes it: on a fleet, the profile
    // its workers mount; elsewhere, the base configuration's profile file.
    defaultOntology: protectedProcedure
      .input(z.object({ baseConfigPath: z.string().nullable().optional() }).optional())
      .query(async ({ ctx, input }) => {
        const fleet = await ctx.controlPlane.fleetProfile();
        if (fleet?.ontology) {
          return {
            ...ontologySelectionFromProfile(fleet.ontology, (fleet.config.graph ?? {}) as Record<string, unknown>),
            source: `the fleet profile (${fleet.configMap})`,
          };
        }
        return {
          ...(await defaultOntologySelection(confinedBaseConfigPath(input?.baseConfigPath ?? null))),
          source: "the default profile",
        };
      }),
    ontology: protectedProcedure
      .input(z.object({ intent: z.string().min(8).max(4_000) }))
      .mutation(async ({ input }) => {
        const gold = await loadMartialArtsGold();
        const goldTypes = [...new Set((gold?.entities ?? []).map((entity) => entity.type.toUpperCase()))];
        const proposal = await proposeOntologyForIntent(input.intent, goldTypes);
        return {
          ...proposal,
          coverage: gold ? ontologyCoverage(proposal.types, gold) : null,
        };
      }),
  }),
  graphs: router({
    versions: publicProcedure
      .input(graphIdInput)
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.versions(input.graphId))),
    // A location is a directory of artifacts under the console's state or
    // the checkout; a stored graph elsewhere is reached through its run.
    load: protectedProcedure
      .input(z.object({ location: z.string().min(1), graphId: safeId.nullable().optional() }))
      .query(({ ctx, input }) =>
        runEffect(
          ctx.controlPlane.loadGraph(
            confinedPath(input.location, {
              roots: [appEnv().stateRoot, appEnv().repositoryRoot],
              base: appEnv().repositoryRoot,
              label: "location",
            }),
            input.graphId,
          ),
        ),
      ),
    loadRun: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      return graphForBrowser(await runEffect(ctx.controlPlane.loadRunGraph(snapshot)));
    }),
    rename: protectedProcedure
      .input(z.object({ graphId: safeId, displayName: z.string().min(1) }))
      .mutation(({ ctx, input }) => runEffect(ctx.controlPlane.renameGraph(input.graphId, input.displayName))),
    // Owner only, and for good: every run and everything stored for them.
    delete: protectedProcedure.input(graphIdInput).mutation(({ ctx, input }) =>
      runEffect(ctx.controlPlane.deleteGraph(input.graphId)),
    ),
    // What a delete would remove, for the confirmation to name it.
    deletion: protectedProcedure.input(graphIdInput).query(({ ctx, input }) =>
      runEffect(ctx.controlPlane.deletionPreview(input.graphId)),
    ),
    // Who owns a graph, who it is shared with, and the viewer's own standing.
    access: publicProcedure.input(graphIdInput).query(({ ctx, input }) =>
      runEffect(ctx.controlPlane.graphAccess(input.graphId)),
    ),
    // Owner only: add a person, or change their level if they already have one.
    share: identifiedProcedure
      .input(z.object({ graphId: safeId, principal: z.string().min(1).max(200), level: z.enum(["read", "write"]) }))
      .mutation(({ ctx, input }) =>
        runEffect(ctx.controlPlane.shareGraph(input.graphId, input.principal, input.level)),
      ),
    // The owner removes anyone; anyone else may remove only themselves.
    unshare: identifiedProcedure
      .input(z.object({ graphId: safeId, principal: z.string().min(1).max(200) }))
      .mutation(({ ctx, input }) => runEffect(ctx.controlPlane.unshareGraph(input.graphId, input.principal))),
    // People the console has seen signed in, offered when sharing.
    people: identifiedProcedure.query(({ ctx }) => runEffect(ctx.controlPlane.people())),
    quality: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      return qualityForGraph({
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
        stateRoot: appEnv().stateRoot,
        graphDirectory: ctx.controlPlane.graphDirectory(snapshot),
      });
    }),
    // A finished graph can be written into Snowflake as it is; the fleet
    // runtime is the one that knows the account, so it is asked directly.
    publishSnowflake: protectedProcedure
      .input(
        z.object({
          runId: safeId,
          database: z.string().trim().min(1),
          schema: z.string().trim().min(1),
          warehouse: z.string().trim().min(1),
          role: z.string().trim().optional(),
          bulkStage: z.string().trim().min(1),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        const publish = ctx.controlPlane.snowflakePublishing;
        if (!publish) {
          throw new TRPCError({ code: "BAD_REQUEST", message: "Only a fleet graph can be published to Snowflake from here." });
        }
        await runEffect(ctx.controlPlane.requireRun(input.runId, "write"));
        try {
          return await publish(input.runId, {
            database: input.database,
            schema: input.schema,
            warehouse: input.warehouse,
            role: input.role || null,
            bulkStage: input.bulkStage,
          });
        } catch (error) {
          throw new TRPCError({ code: "BAD_REQUEST", message: error instanceof Error ? error.message : String(error) });
        }
      }),
    // A gold file is the graph's QA contract; it is kept per graph in the
    // console's state and checked before it is kept.
    gold: router({
      upload: protectedProcedure
        .input(z.object({ graphId: safeId, json: z.string().min(2).max(GOLD_MAX_BYTES) }))
        .mutation(async ({ ctx, input }) => {
          await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
          let parsed: unknown;
          try {
            parsed = JSON.parse(input.json);
          } catch {
            throw new TRPCError({ code: "BAD_REQUEST", message: "The file is not valid JSON." });
          }
          let gold: GoldGraph;
          try {
            gold = validateGold(parsed);
          } catch (error) {
            if (error instanceof GoldValidationError) {
              throw new TRPCError({ code: "BAD_REQUEST", message: error.message });
            }
            throw error;
          }
          await saveUploadedGold(appEnv().stateRoot, input.graphId, gold);
          return { entities: gold.entities?.length ?? 0, relations: gold.relations?.length ?? 0 };
        }),
      remove: protectedProcedure.input(graphIdInput).mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
        return removeUploadedGold(appEnv().stateRoot, input.graphId);
      }),
      template: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
        const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
        const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
        return goldTemplate(dataset, snapshot.graphName || snapshot.graphId);
      }),
    }),
    inspect: publicProcedure.input(runIdInput).query(async ({ ctx, input }) => {
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      const quality = await qualityForGraph({
        graphDirectory: ctx.controlPlane.graphDirectory(snapshot),
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
        stateRoot: appEnv().stateRoot,
      });
      return inspectFromDataset(snapshot, dataset, quality);
    }),
    ask: protectedProcedure
      .input(
        z.object({
          runId: safeId,
          question: z.string().min(2).max(10_000),
          mode: z.enum(["local", "global", "hybrid", "drift", "auto"]).default("local"),
          perspectiveId: z.string().optional(),
          documentIds: z.array(z.string().min(1)).optional(),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        if (!ctx.controlPlane.capabilities.has("ask")) {
          throw new TRPCError({ code: "CONFLICT", message: NO_ASK_MODEL });
        }
        const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
        if (!isSuccessStatus(snapshot.status)) {
          throw new TRPCError({
            code: "CONFLICT",
            message: `Graph ${input.runId} is ${snapshot.status}. Ask requires a finished graph.`,
          });
        }
        const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot, { purpose: "retrieval" }));
        try {
          const scope = await perspectiveScope(input.perspectiveId, snapshot.graphId);
          return await completeAsk({
            dataset,
            question: input.question,
            mode: input.mode === "auto" ? undefined : input.mode,
            scope,
            documentIds: input.documentIds,
          });
        } catch (error) {
          if (error instanceof AskHttpError) {
            throw new TRPCError({
              code: error.status === 400 ? "BAD_REQUEST" : error.status === 409 ? "CONFLICT" : "BAD_REQUEST",
              message: error.message,
            });
          }
          throw error;
        }
      }),
    pin: protectedProcedure
      .input(
        z.object({
          graphId: safeId,
          targetId: z.string(),
          kind: z.enum(["node", "edge"]),
          note: z.string(),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
        return visibleWorkspace(ctx, await pinTarget(input));
      }),
    sampleReviews: protectedProcedure.input(runIdInput).mutation(async ({ ctx, input }) => {
      await runEffect(ctx.controlPlane.requireRun(input.runId, "write"));
      const snapshot = await runEffect(ctx.controlPlane.getRun(input.runId));
      const dataset = await runEffect(ctx.controlPlane.loadRunGraph(snapshot));
      return visibleWorkspace(ctx, await enqueueReviews(sampleHighConfidenceReviews(snapshot.graphId, dataset.edges, dataset.nodes)));
    }),
  }),
  workspace: router({
    // Everything kept per graph or per run is limited to what the viewer can see.
    get: publicProcedure.query(async ({ ctx }) => {
      return visibleWorkspace(ctx, await loadWorkspace());
    }),
    perspective: protectedProcedure
      .input(
        z.object({
          id: z.string().optional(),
          graphId: z.string(),
          name: z.string(),
          lifecycle: z.enum(["draft", "candidate", "production"]),
          search: z.string(),
          communityIds: z.array(z.string()),
          suggestedQuestions: z.array(z.string()),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
        // Saving over an existing perspective keeps it on its own graph: an
        // id is not a way to move someone else's view onto yours.
        if (input.id) {
          const existing = (await loadWorkspace()).perspectives.find((item) => item.id === input.id);
          if (existing && existing.graphId !== input.graphId) {
            throw new TRPCError({ code: "FORBIDDEN", message: "That perspective belongs to another graph." });
          }
        }
        return visibleWorkspace(ctx, await upsertPerspective(input));
      }),
    promotePerspective: protectedProcedure
      .input(z.object({ id: z.string(), lifecycle: z.enum(["draft", "candidate", "production"]) }))
      .mutation(async ({ ctx, input }) => {
        await requireItemGraph(ctx, (state) => state.perspectives.find((item) => item.id === input.id));
        return visibleWorkspace(ctx, await promotePerspective(input.id, input.lifecycle));
      }),
    review: protectedProcedure
      .input(
        z.object({
          id: z.string(),
          decision: z.discriminatedUnion("kind", [
            z.object({ kind: z.literal("keep") }),
            z.object({ kind: z.literal("drop") }),
            z.object({ kind: z.literal("relabel"), relationType: z.string().trim().min(1).max(64) }),
            z.object({ kind: z.literal("merge"), end: z.enum(["source", "target"]), into: z.string().trim().min(1).max(200) }),
          ]),
        }),
      )
      .mutation(async ({ ctx, input }) => {
        await requireItemGraph(ctx, (state) => state.reviews.find((item) => item.id === input.id));
        return visibleWorkspace(ctx, await decideReview(input.id, input.decision));
      }),
    publish: protectedProcedure
      .input(z.object({ graphId: z.string(), note: z.string() }))
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
        return visibleWorkspace(ctx, await publishVersion(input.graphId, input.note));
      }),
    toggleWatch: protectedProcedure.input(z.object({ id: z.string() })).mutation(async ({ ctx, input }) => {
      await requireItemGraph(ctx, (state) => state.watches.find((item) => item.id === input.id));
      return visibleWorkspace(ctx, await toggleWatch(input.id));
    }),
    applyWatch: protectedProcedure.input(z.object({ id: z.string() })).mutation(async ({ ctx, input }) => {
      await requireItemGraph(ctx, (state) => state.watches.find((item) => item.id === input.id));
      return visibleWorkspace(ctx, await applyIncremental(input.id));
    }),
    createWatch: protectedProcedure
      .input(z.object({ graphId: z.string(), prefix: z.string().min(1) }))
      .mutation(async ({ ctx, input }) => {
        await runEffect(ctx.controlPlane.requireGraph(input.graphId, "write"));
        return visibleWorkspace(ctx, await createWatch(input));
      }),
    // A key acts as whoever minted it, so minting and revoking need a name
    // to bind it to, and that person's own session rather than a key.
    createKey: sessionProcedure
      .input(z.object({ name: z.string().min(1).max(200) }))
      .mutation(async ({ ctx, input }) => {
        const { secret, state } = await createApiKey(input.name, ctx.viewer.userName);
        return { secret, state: await visibleWorkspace(ctx, state) };
      }),
    revokeKey: sessionProcedure.input(z.object({ id: z.string().min(1) })).mutation(async ({ ctx, input }) => {
      try {
        return visibleWorkspace(ctx, await revokeApiKey(input.id, ctx.viewer.userName));
      } catch (error) {
        throw new TRPCError({ code: "FORBIDDEN", message: error instanceof Error ? error.message : String(error) });
      }
    }),
    identityIncident: operatorProcedure
      .input(
        z.object({
          expectedPrincipal: z.string(),
          actualPrincipal: z.string(),
          graphOwner: z.string(),
          grants: z.string(),
        }),
      )
      .mutation(async ({ ctx, input }) => visibleWorkspace(ctx, await recordIdentityIncident(input))),
    markGrant: operatorProcedure
      .input(z.object({ object: z.string(), ok: z.boolean() }))
      .mutation(async ({ ctx, input }) => visibleWorkspace(ctx, await markGrant(input.object, input.ok))),
    incident: publicProcedure.query(async ({ ctx }) => staffIncident(await runEffect(ctx.controlPlane.visibleGraphs()))),
  }),
  fleet: router({
    profile: publicProcedure.query(async ({ ctx }) => {
      return ctx.controlPlane.fleetProfile();
    }),
    // The fleet is the namespace this console was deployed for; a caller
    // does not get to read another one through it.
    cluster: publicProcedure.query(({ ctx }) => runEffect(ctx.controlPlane.cluster(appEnv().kubernetesNamespace))),
    nodeAssignments: publicProcedure
      .input(z.object({ nodeName: z.string().min(1).max(253) }))
      .query(({ ctx, input }) => runEffect(ctx.controlPlane.nodeAssignments(appEnv().kubernetesNamespace, input.nodeName))),
  }),
});

export type AppRouter = typeof appRouter;

function requiredInput<T>(input: T | undefined): T {
  if (input === undefined) {
    throw new TRPCError({ code: "BAD_REQUEST", message: "Invalid input" });
  }
  return input;
}

async function loadMartialArtsGold(): Promise<GoldGraph | null> {
  const file = path.join(appEnv().repositoryRoot, "data/martial_arts/gold.json");
  if (!existsSync(file)) {
    return null;
  }
  return JSON.parse(await readFile(file, "utf8")) as GoldGraph;
}
