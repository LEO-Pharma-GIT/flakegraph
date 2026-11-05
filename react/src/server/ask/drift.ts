import { generateText, Output } from "ai";
import { z } from "zod";
import {
  DRIFT_CHUNK_CHAR_BUDGET,
  DRIFT_FOLLOWUP_COUNT,
  DRIFT_FOLLOWUP_PRUNE_DISTANCE,
  DRIFT_FOLLOWUP_QUESTION_CHAR_MAX,
  DRIFT_MAX_RELATIONS,
  DRIFT_PRIMER_COMMUNITIES,
  DRIFT_PRIMER_SUMMARY_CHAR_LIMIT,
  RRF_K,
} from "./constants";
import { fuseRrf } from "./retrieve";
import { resolveAskModel } from "./model";
import type { CommunityHit, EntityHit, EvidenceHit, RelationHit } from "./types";
import { errorMessage } from "./errors";

export interface DriftFollowUp {
  question: string;
  rationale: string;
}

const FollowUpsSchema = z.object({
  followUps: z
    .array(
      z.object({
        question: z.string().min(5).max(DRIFT_FOLLOWUP_QUESTION_CHAR_MAX),
        rationale: z.string().max(DRIFT_FOLLOWUP_QUESTION_CHAR_MAX),
      }),
    )
    .min(1)
    .max(DRIFT_FOLLOWUP_COUNT),
});

export async function generateDriftFollowUps(
  query: string,
  primerCommunities: CommunityHit[],
): Promise<DriftFollowUp[]> {
  if (primerCommunities.length === 0) {
    return [];
  }
  const model = resolveAskModel();
  if (!model) {
    return [];
  }
  const primerBlock = primerCommunities
    .slice(0, DRIFT_PRIMER_COMMUNITIES)
    .map((community, index) => {
      const summary = community.summary.slice(0, DRIFT_PRIMER_SUMMARY_CHAR_LIMIT);
      return `#${index + 1} ${community.title}\n${summary}`;
    })
    .join("\n\n");
  try {
    const result = await generateText({
      model: model.languageModel,
      output: Output.object({ schema: FollowUpsSchema }),
      ...model.structuredOptions,
      instructions: `You plan multi-hop retrieval for a graph RAG system. Given a user question and community-level primers, generate ${DRIFT_FOLLOWUP_COUNT} targeted sub-questions whose answers together cover the user's question.

Rules:
- Each sub-question must be answerable from specific entities / passages, not abstract themes.
- Prefer questions that drill down into concrete entities named in the primers.
- Do not paraphrase the original question; decompose it.
- Keep each sub-question under ${DRIFT_FOLLOWUP_QUESTION_CHAR_MAX} characters.`,
      prompt: `User question:\n${query}\n\nPrimer communities:\n${primerBlock}`,
    });
    return result.output?.followUps.slice(0, DRIFT_FOLLOWUP_COUNT) ?? [];
  } catch (error) {
    // Without follow-ups the pass answers from the primers alone; say why.
    console.error(`DRIFT: the model did not write follow-up questions: ${errorMessage(error)}`);
    return [];
  }
}

export interface DriftPass {
  evidence: EvidenceHit[];
  entities: EntityHit[];
  relations: RelationHit[];
  communities: CommunityHit[];
}

export function mergeDriftPasses(args: {
  primerCommunities: CommunityHit[];
  passes: DriftPass[];
  topK: number;
}): {
  evidence: EvidenceHit[];
  entities: EntityHit[];
  relations: RelationHit[];
  communities: CommunityHit[];
} {
  const { primerCommunities, topK } = args;
  const passes = pruneLowConfidenceFollowUps(args.passes);
  let fused = fuseRrf(passes[0]?.evidence ?? [], [], topK);
  for (let index = 1; index < passes.length; index += 1) {
    fused = fuseRrf(fused, passes[index]?.evidence ?? [], topK);
  }
  fused = applyCharBudget(fused);

  const entitiesById = new Map<string, EntityHit>();
  for (const pass of passes) {
    for (const entity of pass.entities) {
      const prior = entitiesById.get(entity.id);
      if (!prior || entity.distance < prior.distance) {
        entitiesById.set(entity.id, entity);
      }
    }
  }

  const relationScore = new Map<string, number>();
  const relationById = new Map<string, RelationHit>();
  for (const pass of passes) {
    pass.relations.forEach((relation, rank) => {
      relationScore.set(relation.id, (relationScore.get(relation.id) ?? 0) + 1 / (RRF_K + rank + 1));
      if (!relationById.has(relation.id)) {
        relationById.set(relation.id, relation);
      }
    });
  }

  const communitiesById = new Map<string, CommunityHit>();
  for (const community of primerCommunities) {
    communitiesById.set(community.id, community);
  }
  for (const pass of passes) {
    for (const community of pass.communities) {
      if (!communitiesById.has(community.id)) {
        communitiesById.set(community.id, community);
      }
    }
  }

  return {
    evidence: fused.slice(0, topK),
    entities: [...entitiesById.values()].sort((left, right) => left.distance - right.distance),
    relations: [...relationById.values()]
      .sort((left, right) => (relationScore.get(right.id) ?? 0) - (relationScore.get(left.id) ?? 0))
      .slice(0, DRIFT_MAX_RELATIONS),
    communities: [...communitiesById.values()],
  };
}

function pruneLowConfidenceFollowUps(passes: DriftPass[]): DriftPass[] {
  if (passes.length <= 1) {
    return passes;
  }
  const [baseline, ...followUps] = passes;
  if (!baseline) {
    return passes;
  }
  const kept = followUps.filter((pass) => {
    const best = pass.evidence[0]?.distance;
    if (best === undefined || !Number.isFinite(best)) {
      return true;
    }
    return best <= DRIFT_FOLLOWUP_PRUNE_DISTANCE;
  });
  return [baseline, ...kept];
}

function applyCharBudget(hits: EvidenceHit[]): EvidenceHit[] {
  if (hits.length === 0) {
    return hits;
  }
  const kept: EvidenceHit[] = [];
  let used = 0;
  for (const hit of hits) {
    const size = hit.quote.length;
    if (kept.length > 0 && used + size > DRIFT_CHUNK_CHAR_BUDGET) {
      break;
    }
    kept.push(hit);
    used += size;
  }
  return kept;
}
