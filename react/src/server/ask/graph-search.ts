import type { GraphDataset } from "../protocol/schema";
import {
  CHUNK_RERANK_OVERFETCH_FACTOR,
  DRIFT_PRIMER_COMMUNITIES,
  ENTITY_RERANK_OVERFETCH_FACTOR,
  GLOBAL_GROUNDING_TOP_K_MAX,
  GLOBAL_GROUNDING_TOP_K_MIN,
  GLOBAL_MAX_RELATIONS,
  LOCAL_BACKDROP_COMMUNITY_CAP,
  RERANK_COMMUNITY_SHORTLIST_FLOOR_BONUS,
  RERANK_COMMUNITY_SHORTLIST_OVERSAMPLE_FACTOR,
  TOOL_DEFAULT_TOP_K,
} from "./constants";
import { generateDriftFollowUps, mergeDriftPasses } from "./drift";
import { communityQueryText, entityQueryText, lexicalPlan, planQueryRetrieval } from "./planner";
import { emitRerankProgress, rerankCommunities, rerankEntities, rerankEvidence } from "./rerank";
import {
  blendRelations,
  communitiesForEntities,
  contributingEntities,
  evidenceForEntities,
  fuseRrf,
  incidentRelations,
  localTopK,
  maybePromoteToSuperCommunity,
  searchCommunities,
  searchEntities,
  searchEvidence,
  searchRelations,
} from "./retrieve";
import { mergeAskScope, scopeDataset } from "./scope";
import { PROGRESS_LABELS, type GraphSearchOptions, type GraphSearchResult, type LocalSearchResult, type QueryMode } from "./types";

function emit(
  onProgress: GraphSearchOptions["onProgress"],
  phase: keyof typeof PROGRESS_LABELS,
  detail?: string,
) {
  onProgress?.({ phase, message: PROGRESS_LABELS[phase], detail });
}

export async function graphSearch(
  dataset: GraphDataset,
  options: GraphSearchOptions,
): Promise<GraphSearchResult> {
  const query = options.query.trim();
  const topK = options.topK ?? TOOL_DEFAULT_TOP_K;
  const rerank = options.rerank ?? false;
  const communityLevel = options.communityLevel;
  const communityScanAll = options.communityScanAll ?? false;
  const scoped = scopeDataset(
    dataset,
    mergeAskScope(options.scope, options.documentIds?.length ? { documentIds: options.documentIds } : undefined),
  );
  emit(options.onProgress, "planning", `mode=${options.mode ?? "auto"}`);

  const plan = query ? await planQueryRetrieval(query) : lexicalPlan(query);
  const mode: QueryMode = options.mode ?? plan.mode;
  const entityQuery = entityQueryText(plan, query);
  const communityQuery = communityQueryText(plan, query);
  const ctx: SearchContext = {
    dataset: scoped,
    question: query,
    entityQuery,
    communityQuery,
    topK,
    rerank,
    communityLevel,
    communityScanAll,
    onProgress: options.onProgress,
  };

  if (mode === "local") {
    emit(options.onProgress, "searching_entities", `topK=${topK}`);
    const local = await localSearch(ctx);
    emit(options.onProgress, "building_results");
    return {
      mode,
      plan,
      entities: local.entities,
      contributingEntities: [],
      relations: local.relations,
      communities: local.communities,
      evidence: local.evidence,
    };
  }

  if (mode === "global") {
    emit(options.onProgress, "searching_communities", `topK=${topK}`);
    const groundingTopK = Math.max(
      GLOBAL_GROUNDING_TOP_K_MIN,
      Math.min(GLOBAL_GROUNDING_TOP_K_MAX, Math.ceil(topK / 2)),
    );
    const [global, evidence] = await Promise.all([
      globalSearch(ctx, topK),
      Promise.resolve().then(async () => {
        emit(options.onProgress, "searching_evidence");
        const hits = searchEvidence(scoped, entityQuery, rerank ? groundingTopK * CHUNK_RERANK_OVERFETCH_FACTOR : groundingTopK);
        return rerank
          ? rerankEvidence(query, hits, groundingTopK, emitRerankProgress(options.onProgress, "reranking_evidence", "passages"))
          : hits;
      }),
    ]);
    emit(options.onProgress, "building_results");
    return {
      mode,
      plan,
      entities: [],
      contributingEntities: global.contributingEntities,
      relations: global.relations,
      communities: global.communities,
      evidence,
    };
  }

  if (mode === "drift") {
    emit(options.onProgress, "searching_communities", `primer=${DRIFT_PRIMER_COMMUNITIES}`);
    const [primer, baseline] = await Promise.all([
      globalSearch(ctx, DRIFT_PRIMER_COMMUNITIES),
      localSearch(ctx),
    ]);
    emit(options.onProgress, "drift_follow_ups", `Generating follow-ups from ${primer.communities.length} communities`);
    const followUps = await generateDriftFollowUps(query, primer.communities);
    emit(
      options.onProgress,
      "drift_follow_ups",
      followUps.length ? followUps.map((item) => item.question).join(" | ") : "No follow-ups; using primer + baseline",
    );
    const followUpPasses = await Promise.all(
      followUps.map((item) => localSearch({ ...ctx, entityQuery: item.question, question: item.question })),
    );
    const merged = mergeDriftPasses({
      primerCommunities: primer.communities,
      passes: [baseline, ...followUpPasses],
      topK,
    });
    const contribById = new Map(primer.contributingEntities.map((entity) => [entity.id, entity]));
    for (const pass of [baseline, ...followUpPasses]) {
      for (const entity of pass.entities) {
        const prior = contribById.get(entity.id);
        if (!prior || entity.distance < prior.distance) {
          contribById.set(entity.id, entity);
        }
      }
    }
    emit(options.onProgress, "building_results");
    return {
      mode,
      plan,
      entities: merged.entities,
      contributingEntities: [...contribById.values()].sort((left, right) => left.distance - right.distance),
      relations: merged.relations,
      communities: merged.communities,
      evidence: merged.evidence,
    };
  }

  emit(options.onProgress, "searching_entities");
  const [local, global] = await Promise.all([
    localSearch(ctx),
    globalSearch(ctx, Math.min(LOCAL_BACKDROP_COMMUNITY_CAP, topK)),
  ]);
  emit(options.onProgress, "building_results");
  return {
    mode,
    plan,
    ...mergeLocalAndGlobal(local, global),
  };
}

interface SearchContext {
  dataset: GraphDataset;
  question: string;
  entityQuery: string;
  communityQuery: string;
  topK: number;
  rerank: boolean;
  communityLevel?: number;
  communityScanAll: boolean;
  onProgress?: GraphSearchOptions["onProgress"];
}

function mergeLocalAndGlobal(
  local: LocalSearchResult,
  global: { communities: LocalSearchResult["communities"]; contributingEntities: GraphSearchResult["contributingEntities"]; relations: LocalSearchResult["relations"] },
) {
  const communitiesById = new Map(global.communities.map((community) => [community.id, community]));
  for (const community of local.communities) {
    if (!communitiesById.has(community.id)) {
      communitiesById.set(community.id, community);
    }
  }
  const relationsById = new Map(local.relations.map((relation) => [relation.id, relation]));
  for (const relation of global.relations) {
    if (!relationsById.has(relation.id)) {
      relationsById.set(relation.id, relation);
    }
  }
  return {
    entities: local.entities,
    contributingEntities: global.contributingEntities,
    relations: [...relationsById.values()],
    communities: [...communitiesById.values()],
    evidence: local.evidence,
  };
}

async function localSearch(ctx: SearchContext): Promise<LocalSearchResult> {
  const { dataset, entityQuery, question, topK, rerank, communityLevel, onProgress } = ctx;
  if (!entityQuery.trim() || dataset.nodes.length === 0) {
    return { entities: [], relations: [], evidence: [], communities: [] };
  }
  const { entityTopK, relationTopK } = localTopK(topK);
  const entityFetch = rerank ? entityTopK * ENTITY_RERANK_OVERFETCH_FACTOR : entityTopK;
  const evidenceFetch = rerank ? topK * CHUNK_RERANK_OVERFETCH_FACTOR : topK;
  let entities = searchEntities(dataset, entityQuery, entityFetch);
  if (rerank) {
    entities = await rerankEntities(question, entities, entityTopK, emitRerankProgress(onProgress, "reranking_entities", "entities"));
  }
  const semanticRelations = searchRelations(dataset, entityQuery, relationTopK);
  const topology = incidentRelations(
    dataset,
    entities.map((entity) => entity.id),
  );
  const relations = blendRelations(semanticRelations, topology);
  const neighborIds = new Set(entities.map((entity) => entity.id));
  for (const relation of relations) {
    neighborIds.add(relation.sourceId);
    neighborIds.add(relation.targetId);
  }
  const entityEvidence = evidenceForEntities(dataset, [...neighborIds], entityQuery, evidenceFetch);
  const directEvidence = searchEvidence(dataset, entityQuery, evidenceFetch);
  let evidence = fuseRrf(directEvidence, entityEvidence, evidenceFetch);
  if (rerank) {
    evidence = await rerankEvidence(question, evidence, topK, emitRerankProgress(onProgress, "reranking_evidence", "passages"));
  } else {
    evidence = evidence.slice(0, topK);
  }
  return {
    entities,
    relations,
    evidence,
    communities: communitiesForEntities(
      dataset,
      entities.map((entity) => entity.id),
      Math.min(LOCAL_BACKDROP_COMMUNITY_CAP, topK),
      communityLevel ?? 0,
    ),
  };
}

async function globalSearch(ctx: SearchContext, topK: number) {
  const { dataset, communityQuery, entityQuery, question, rerank, communityLevel, communityScanAll, onProgress } = ctx;
  const applyRerank = rerank || communityScanAll;
  const shortlistK = applyRerank
    ? Math.max(topK * RERANK_COMMUNITY_SHORTLIST_OVERSAMPLE_FACTOR, topK + RERANK_COMMUNITY_SHORTLIST_FLOOR_BONUS)
    : topK;
  let communities = searchCommunities(dataset, communityQuery, shortlistK, {
    level: communityLevel ?? 0,
    scanAll: communityScanAll,
  });
  if (communities.length === 0) {
    return { communities: [], contributingEntities: [], relations: [] };
  }
  if (applyRerank) {
    communities = (
      await rerankCommunities(question, communities, emitRerankProgress(onProgress, "reranking_communities", "communities"))
    ).slice(0, topK);
  } else {
    communities = communities.slice(0, topK);
  }
  if ((communityLevel ?? 0) === 0) {
    communities = maybePromoteToSuperCommunity(dataset, communities);
  }
  if (communities.length === 0) {
    return { communities: [], contributingEntities: [], relations: [] };
  }
  const entities = contributingEntities(dataset, communities, entityQuery, topK);
  const relations = incidentRelations(
    dataset,
    entities.map((entity) => entity.id),
  )
    .sort((left, right) => right.weight - left.weight)
    .slice(0, GLOBAL_MAX_RELATIONS);
  return { communities, contributingEntities: entities, relations };
}
