/** Reciprocal Rank Fusion smoothing constant (Cormack et al. 2009). */
export const RRF_K = 60;

export const RELATION_BLEND_DISTANCE_WEIGHT = 0.6;
export const RELATION_BLEND_TOPOLOGY_WEIGHT = 0.4;

export const COMMUNITY_BLEND_DISTANCE_WEIGHT = 0.7;
export const COMMUNITY_BLEND_RATING_WEIGHT = 0.3;

export const GLOBAL_ENTITY_BLEND_ENTITY_WEIGHT = 0.7;
export const GLOBAL_ENTITY_BLEND_COMMUNITY_WEIGHT = 0.3;

export const GLOBAL_MAX_RELATIONS = 20;
export const GLOBAL_CONTRIBUTING_ENTITY_OVERSAMPLE_FACTOR = 3;

/** Hub community ratings are 0-10; missing ratings use a neutral 0.5 penalty. */
export const COMMUNITY_RATING_MAX = 10;

export const RERANK_COMMUNITY_SHORTLIST_OVERSAMPLE_FACTOR = 3;
export const RERANK_COMMUNITY_SHORTLIST_FLOOR_BONUS = 6;
export const CHUNK_RERANK_OVERFETCH_FACTOR = 3;
export const ENTITY_RERANK_OVERFETCH_FACTOR = 3;
export const RERANK_SCORE_MIN = 0;
export const RERANK_SCORE_MAX = 10;
export const RERANK_CHUNK_PROMPT_CHAR_LIMIT = 1_500;
export const RERANK_COMMUNITY_PROMPT_CHAR_LIMIT = 1_500;
export const RERANK_ENTITY_PROMPT_CHAR_LIMIT = 800;
export const RERANK_LLM_CALL_TIMEOUT_MS = 20_000;
export const LLM_RERANK_CONCURRENCY = 4;

export const GRAPH_SEARCH_CONTEXT_CHAR_BUDGET = 40_000;
export const TOOL_COMMUNITY_FINDINGS_SLICE = 5;
export const TOOL_ENTITY_RESULTS_SLICE = 15;
export const TOOL_RELATION_RESULTS_SLICE = 20;
export const TOOL_ENTITY_DESCRIPTION_CHAR_LIMIT = 400;
export const TOOL_RELATION_SEGMENT_CHAR_LIMIT = 200;
export const TOOL_DEFAULT_TOP_K = 20;
export const TOOL_MAX_TOP_K = 40;
export const TOOL_DOCUMENT_QUOTE_DEFAULT = 8;
export const TOOL_DOCUMENT_QUOTE_MAX = 20;

export const ENTITY_SOURCE_CHUNK_CHAR_LIMIT = 2_000;
export const ENTITY_SOURCE_CHUNK_MAX = 50;
export const EDGE_EVIDENCE_CHAR_LIMIT = 1_000;
export const EDGE_EVIDENCE_CHUNK_MAX = 10;

export const GRAPH_LOCAL_ENTITY_TOP_K_MIN = 5;
export const GRAPH_LOCAL_ENTITY_TOP_K_RATIO = 0.75;
export const GRAPH_LOCAL_RELATION_TOP_K_FLOOR = 10;
export const GRAPH_LOCAL_ENTITY_EVIDENCE_MULTIPLIER = 4;
export const LOCAL_BACKDROP_COMMUNITY_CAP = 4;

export const GLOBAL_GROUNDING_TOP_K_MIN = 3;
export const GLOBAL_GROUNDING_TOP_K_MAX = 6;

export const DRIFT_PRIMER_COMMUNITIES = 4;
export const DRIFT_FOLLOWUP_COUNT = 3;
export const DRIFT_FOLLOWUP_QUESTION_CHAR_MAX = 240;
export const DRIFT_PRIMER_SUMMARY_CHAR_LIMIT = 1_500;
export const DRIFT_CHUNK_CHAR_BUDGET = 40_000;
export const DRIFT_MAX_RELATIONS = 24;
export const DRIFT_FOLLOWUP_PRUNE_DISTANCE = 0.75;

export const ACCESSIBLE_RELATIONS_LIMIT = 200;
export const AGENT_MAX_STEPS = 8;
// How long one question may take, tool calls and reasoning included. Generous,
// because a reasoning model on a busy engine takes minutes, and a question cut
// short has no answer at all.
export const ASK_STREAM_TIMEOUT_MS = 20 * 60_000;
// The longest an NDJSON answer stays silent: while the model reads and writes,
// a status repeats at this interval, so no client or proxy between the console
// and its caller takes a long answer for a dead connection.
export const ASK_HEARTBEAT_MS = 15_000;

export const DEFAULT_AZURE_API_VERSION = "2024-12-01-preview";
export const DEFAULT_AZURE_DEPLOYMENT = "gpt-4.1-mini-2025-04-14";
export const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1";
export const DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct";
