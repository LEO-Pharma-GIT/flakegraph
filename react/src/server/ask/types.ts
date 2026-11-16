export const QUERY_MODES = ["local", "global", "hybrid", "drift"] as const;
export type QueryMode = (typeof QUERY_MODES)[number];

export const SEARCH_PROGRESS_PHASES = [
  "loading_graph",
  "planning",
  "searching_entities",
  "searching_communities",
  "searching_relations",
  "searching_evidence",
  "reranking_entities",
  "reranking_evidence",
  "reranking_communities",
  "drift_follow_ups",
  "building_results",
  "answering",
] as const;
export type SearchProgressPhase = (typeof SEARCH_PROGRESS_PHASES)[number];

export interface SearchProgressUpdate {
  phase: SearchProgressPhase;
  message: string;
  detail?: string;
}

export interface AskScope {
  search?: string;
  communityIds?: string[];
  documentIds?: string[];
}

export interface AskCitation {
  quote: string;
  documentId: string;
  /** The document as a reader knows it - its filename - when the graph records one. */
  documentName?: string;
  entityId: string | null;
  entityName: string | null;
}

export interface AskQueryConsumption {
  retrievedEntities: number;
  retrievedQuotes: number;
  retrievedCommunities: number;
  retrievedRelations: number;
  locality: "graph";
  note: string;
  billed: boolean;
}

export interface EntityHit {
  id: string;
  name: string;
  type: string;
  description: string;
  score: number;
  distance: number;
}

export interface RelationHit {
  id: string;
  sourceId: string;
  targetId: string;
  sourceName: string;
  targetName: string;
  relationType: string;
  description: string;
  weight: number;
  score: number;
  distance: number;
  /** Set when every source only claims the relation; absent for a reported one. */
  claimed?: true;
}

export interface CommunityHit {
  id: string;
  title: string;
  summary: string;
  memberIds: string[];
  rating: number | null;
  score: number;
  distance: number;
  findings: { summary: string; explanation: string }[];
  level: number;
  parentCommunityId: string | null;
}

export interface EvidenceHit {
  id: string;
  quote: string;
  documentId: string;
  entityId: string | null;
  entityName: string | null;
  relationId: string | null;
  score: number;
  distance: number;
}

export interface LocalSearchResult {
  entities: EntityHit[];
  relations: RelationHit[];
  evidence: EvidenceHit[];
  communities: CommunityHit[];
}

export interface GlobalSearchResult {
  communities: CommunityHit[];
  contributingEntities: EntityHit[];
  relations: RelationHit[];
  evidence: EvidenceHit[];
}

export interface GraphSearchResult {
  mode: QueryMode;
  entities: EntityHit[];
  contributingEntities: EntityHit[];
  relations: RelationHit[];
  communities: CommunityHit[];
  evidence: EvidenceHit[];
  plan: QueryRetrievalPlan;
}

export interface QueryRetrievalPlan {
  mode: QueryMode;
  highLevelKeywords: string[];
  lowLevelKeywords: string[];
}

export interface GraphDocument {
  id: string;
  title: string;
  path: string;
  quoteCount: number;
}

export interface GraphSearchOptions {
  query: string;
  scope?: AskScope;
  topK?: number;
  mode?: QueryMode;
  /** Restrict retrieval to these document ids (hub `fileIds` / `restrictToFileIds`). */
  documentIds?: string[];
  /** LLM rerank of entity, evidence, and community shortlists. Chat default is false. */
  rerank?: boolean;
  /** Leiden community hierarchy level. Defaults to 0 when the dataset has levels. */
  communityLevel?: number;
  /** Visit every community at `communityLevel` before optional rerank (eval-style). */
  communityScanAll?: boolean;
  onProgress?: (update: SearchProgressUpdate) => void;
}

export interface AskAnswer {
  mode: QueryMode;
  question: string;
  summary: string;
  entityIds: string[];
  relationIds: string[];
  communityIds: string[];
  citations: AskCitation[];
  usedCommunityReport: boolean;
  scoped: boolean;
  queryConsumption: AskQueryConsumption;
  plan?: QueryRetrievalPlan;
}

export type AskStreamEvent =
  | { type: "status"; phase: SearchProgressPhase; message: string; detail?: string }
  | { type: "tool"; name: string; state: "start" | "done"; input?: unknown; output?: unknown }
  | { type: "text"; delta: string }
  | { type: "citation"; citation: AskCitation }
  | { type: "error"; message: string; code?: string }
  | { type: "done"; answer: AskAnswer };

export interface AskRequestBody {
  messages?: unknown[];
  runId?: string;
  question?: string;
  mode?: QueryMode | "auto";
  perspectiveId?: string;
  format?: "ui" | "ndjson";
  topK?: number;
  documentIds?: string[];
}

export const PROGRESS_LABELS: Record<SearchProgressPhase, string> = {
  loading_graph: "Loading graph",
  planning: "Planning retrieval",
  searching_entities: "Searching entities",
  searching_communities: "Searching communities",
  searching_relations: "Searching relations",
  searching_evidence: "Searching evidence",
  reranking_entities: "Reranking entities",
  reranking_evidence: "Reranking evidence",
  reranking_communities: "Reranking communities",
  drift_follow_ups: "Planning follow-up retrieval",
  building_results: "Building results",
  answering: "Writing the answer",
};
