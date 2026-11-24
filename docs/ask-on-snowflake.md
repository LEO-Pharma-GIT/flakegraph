# Ask on Snowflake

Questions over a graph run inside Snowflake once the console names a Cortex
model to answer with. This note says what each part of answering uses there,
and what an operator sets up.

## What Ask uses, by runtime

| Ask needs | Fleet and local runs | Snowflake |
|---|---|---|
| A model that calls tools | An OpenAI-compatible gateway or Azure deployment (`FLAKEGRAPH_ASK_*`) | Cortex Chat Completions (`/api/v2/cortex/v1/chat/completions`), with the service's own OAuth token (`FLAKEGRAPH_ASK_CORTEX_MODEL`) |
| The reader model for each document | `FLAKEGRAPH_ASK_READER_*`, else the Ask model | `FLAKEGRAPH_ASK_READER_CORTEX_MODEL`, else the Ask model |
| Passage search, by words and meaning | In-memory BM25 fused with the chunks' vectors | In-memory BM25 fused with a Cortex Search service over `KG_CHUNK` (`FLAKEGRAPH_CORTEX_SEARCH_SERVICE`) |
| Chunks and entities | The run's artifact files | `KG_CHUNK` (without its vector) and `KG_NODE`, loaded when a graph is opened for answering |
| An answer that takes minutes | One streamed request | A job the page polls (`POST /api/ask/jobs`, `GET /api/ask/jobs/{id}`): the ingress ends every request after about 90 seconds |

## Setting it up

1. Create the schema with `flakegraph snowflake setup` or `objects`, with a
   warehouse configured. The setup SQL creates `KG_CHUNK` with change
   tracking and, where `snowflake.cortex_search_service` names one (such as
   `KG_CHUNK_SEARCH`), a Cortex Search service over it (embedding model
   `snowflake.cortex_search_embedding_model`, lag
   `snowflake.cortex_search_target_lag`). The service is billed for as long as it
   exists, so none is created by default. One service serves every graph of
   the schema; a query filters by `GRAPH_ID`, and a reader's by `DOCUMENT_ID`.
   The schema-owning role needs `CREATE CORTEX SEARCH SERVICE` on the schema
   and the `SNOWFLAKE.CORTEX_USER` database role.
   A schema created before change tracking was part of `KG_CHUNK`'s DDL needs
   `ALTER TABLE KG_CHUNK SET CHANGE_TRACKING = TRUE` once, before the service.
2. Deploy the console with `FLAKEGRAPH_ASK_CORTEX_MODEL` (a model the account
   offers with tool calling) and `FLAKEGRAPH_CORTEX_SEARCH_SERVICE` set;
   `scripts/deploy-snowflake.ts` passes both to the service.

Without `FLAKEGRAPH_ASK_CORTEX_MODEL` the Ask tab is not offered. Without a
search service, passages are searched by their words alone.

## Limits

- Opening a graph for answering reads all of its chunks and entities into the
  console's memory. That suits graphs of thousands of documents; graphs far
  larger need the record and entity lookups done in SQL instead.
- Polled jobs live in the console's memory, which is right for its single
  instance; a restart ends the questions running.
- Cortex's structured outputs and tool calling vary by model: the answering
  and reading models must both support them.
