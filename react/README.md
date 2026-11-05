# FlakeGraph console

The operator console for FlakeGraph: a Next.js application that composes
ingestion runs, watches them, and lets you explore and ask questions of the
graphs they produce. The processing itself is the Python pipeline in the
parent repository (`flakegraph`); this package shells out to it on a laptop
and submits to the fleet queue on Kubernetes.

```text
Browser
  → Next.js (App Router, tRPC, Effect)
      → Runtime module (local | kubernetes | snowflake)
          → flakegraph worker | fleet queue + workers | Snowflake job service
```

Licensed under Apache-2.0, like the rest of the repository (see `../LICENSE`).

## Prerequisites

- [Bun](https://bun.sh) 1.3 or newer. It is the one package manager and script
  runner here; `bun.lock` is the lockfile.
- For real runs: the Python pipeline installed in the parent checkout
  (`uv sync --extra local-embeddings` at the repository root), so `uv run flakegraph` works. The
  console's `FLAKEGRAPH_CLI` names that command.
- For the default laptop providers: [Ollama](https://ollama.com) with a small
  model pulled (`ollama pull qwen3:4b` for extraction; Ask defaults to
  `qwen3:4b-instruct`, see the environment table). The embedding model
  (`Qwen/Qwen3-Embedding-0.6B`, sentence-transformers) downloads on first use.
  Plain `uv sync` suffices when you select and configure a hosted embedding
  provider instead. OCR defaults to the pipeline's built-in text extraction, which needs no
  engine; MinerU and Tesseract are opt-in.
- For the e2e suite: Playwright's bundled Chromium
  (`bunx playwright install chromium`).

## Laptop quickstart

A seeded demo with a fake worker, no GPU, no model and no account:

```bash
cd react
bun install
bun run dev:demo      # http://127.0.0.1:3100
```

`dev:demo` seeds two sample graphs plus fleet and Snowflake fixtures under a
temporary state root, starts the app with stubbed Kubernetes and Snowflake
modules, points `FLAKEGRAPH_CLI` at `scripts/fake-flakegraph.ts`, and turns on
demo identities (the sidebar's "Sign in as ALICE" buttons). Set
`FLAKEGRAPH_ASK_NO_MODEL=1` to run without any language model; otherwise the
Ask tab uses whatever `FLAKEGRAPH_ASK_*` names, or an Ollama on the loopback.

To run the real pipeline on your own documents:

```bash
cd react
FLAKEGRAPH_DEMO_IDENTITIES=1 bun run dev      # http://localhost:3000
```

Then compose a graph from the sample pack or a folder. A folder must be under
the checkout, the console's state root, or a root listed in
`FLAKEGRAPH_APP_SOURCE_ROOTS` (see Security). State is written under
`.flakegraph/app/` at the repository root unless `FLAKEGRAPH_APP_STATE_ROOT`
points elsewhere.

Without `FLAKEGRAPH_DEMO_IDENTITIES=1`, a console with no sign-in gate refuses
every request except `/api/health`, `/api/docs` and the `GET /api/ask` catalog: that is the production
posture, described next.

## Security

The console can read files, spawn the pipeline with credentials, and spend
model budget, so by default it serves nobody it cannot name. What identifies
a caller:

| Source | When | What it gives |
| --- | --- | --- |
| Sign-in gate headers | `FLAKEGRAPH_TRUST_IDENTITY_HEADERS=true` | `X-Auth-Request-Email` (or `-User`) is the viewer; `X-Auth-Request-Groups` are their roles. Nothing else is read: not `X-Forwarded-*`, not a preferred username. |
| Bearer token through the gate | `ingress.authProxy.bearerTokens.enabled` | An application acting for a person presents that person's access token; the gate verifies it and sets the same headers as for a browser session. Any bearer that is not an `fg_` key is left to the gate. |
| API key | Always | `Authorization: Bearer fg_…` or `x-flakegraph-api-key`. The request acts as the principal who minted the key. |
| Demo identities | `FLAKEGRAPH_DEMO_IDENTITIES=1` | The principal assumed in the sidebar, or nobody. For laptops and rehearsals only. |
| Snowflake context | `FLAKEGRAPH_SNOWFLAKE_HOSTED=1` | `Sf-Context-Current-User` is the viewer, inside a Snowflake container. Their roles are asked of Snowflake; no other header is read, nobody assumes an identity, and a caller the ingress did not name is refused. |

What that identity is required for:

- Every mutation (submit, cancel, delete, rename, share, reviews, keys,
  uploads, questions) and every procedure that names a path on the host
  (browsing a folder, previewing a configuration, loading a graph by
  location) requires an identified caller. Anonymous callers get 401.
- Read-only catalog queries (listing runs, opening a finished graph) are also
  closed to anonymous callers unless `FLAKEGRAPH_APP_ANONYMOUS_READ=true`.
  `FLAKEGRAPH_APP_REQUIRE_SIGN_IN=true` closes them again regardless.
- Minting or revoking an API key requires a *named* viewer in a session of
  their own; a key is bound to that name and only they can revoke it. A
  request that presents an API key or any other `Authorization` credential
  (a delegated access token) is refused with 403, so a leaked key cannot mint
  more keys or revoke its owner's.
- Settings that are the whole console's - the suggestion mode, the Snowflake
  grants checklist, the identity incident note, a demo console's assumed
  identity - change only for whoever administers the console, in a session
  of their own: on a console behind the gate or in Snowflake, the principals
  and roles `FLAKEGRAPH_APP_OPERATORS` names, and nobody while it names none;
  the pinned `FLAKEGRAPH_SINGLE_OPERATOR`; any session of a demo console. The
  sidebar's operator / analyst / staff role only changes what the console
  shows. Assuming a demo identity is refused unless demo identities are on.

What a request may name:

- Run, graph and upload ids match `[A-Za-z0-9._-]+`; they become directory
  names under the state root and nothing else. A run id already in use, by
  any runtime's catalog or by the fleet, is refused rather than overwritten.
- `source.path` (a folder of documents) must lie under the checkout, the state
  root, or a directory listed in `FLAKEGRAPH_APP_SOURCE_ROOTS`. Under the state
  root it may not name the console's own records (runs, graphs, artifacts,
  the access store and the like), and an upload folder is readable only by the
  person who uploaded it, or by someone who can read a graph built from it.
  Only the uploader may add files to an upload folder.
- A run writes only into its own graph's directory. Base configurations are
  computed by the server. An explicit base must be an operator-managed profile
  under the checkout's `configs/` or a generated configuration under the state
  root's `runs/`; uploaded YAML is not a base configuration.
- Local source folders may not contain the console state root. Globs are relative
  descendant patterns without parent components; every matched file must resolve
  inside its source directory. These checks apply to browsing, preflight and ingestion.
- Provider request options accept only data settings supported by that provider:
  LLM timeout, context window and output-token ceiling; embedding batch size
  (plus Sentence Transformers device and precision); the Azure OpenAI API
  version; OCR language, page range, timeout and provider-specific parsing
  settings. Unknown options are refused. Local OCR executables remain operator
  configuration:
  `ocr.mineru_command`, `ocr.tesseract_command` and
  `ocr.tesseract_pdf_renderer_command` in an approved profile, or the pipeline's
  `KG_MINERU_COMMAND` / `KG_TESSERACT_COMMAND` environment overrides.
- A graph id is claimed by its first submitter; an id any runtime already
  knows cannot be claimed again through another.
- A credential the request references is a name, never a value, and must be a
  `KG_*` variable set on the console. It may reach only the endpoint paired
  with that variable in an operator-managed YAML profile under `configs/`,
  the mounted fleet profile, or a complete endpoint/key environment pair.
  Model and data settings remain configurable within that destination.
  Generated run configurations do not authorize new credential destinations.
  Environment placeholders (`${…}`) belong to operator profiles; request
  values cannot interpolate the console environment. The pipeline is spawned with an
  allow-listed environment (`KG_*`, cache and SDK families, `PATH`, `HOME`,
  the coordination `DATABASE_URL`), never with the console's own variables
  such as `FLAKEGRAPH_ASK_API_KEY`.
- An upload request carries at most 200 files, 200 MB per file and 1 GB in
  all; the console splits a larger drop into as many requests as it needs, into
  one upload folder. A dropped folder keeps its subfolders: each file's path is
  reduced to plain names under the upload folder, never `..` or an absolute
  path. The dropzone itself leaves hidden files and folders out.
- The fleet procedures read only `FLAKEGRAPH_APP_KUBERNETES_NAMESPACE`.

### Graph access

A graph belongs to whoever started its first run and is private to them until
they share it. The owner shares it from the graph's Sharing tab with other
people by sign-in name, at Read (open, explore, ask, export) or Write (also
rename, revise, retry or cancel runs, review, gold, perspectives, pins,
publish). Only the owner can share, change a level, remove someone or delete
the graph; someone it was shared with can leave it. The catalog lists the
viewer's own graphs under Mine and those shared with them under Shared.
The checks run on the server for every runtime, so an API key has exactly the
access of the person who minted it.

There is no administrator override. With `FLAKEGRAPH_APP_REQUIRE_SIGN_IN=true`
a graph with no recorded owner is open to nobody. A console with demo
identities and no sign-in requirement is one person's laptop: a session that
assumed no identity sees and changes everything, while an assumed demo
identity sees what that person would. Elsewhere a caller with no name (an
anonymous reader) owns nothing and may only read graphs nobody owns.

Everything else follows the same line: run lists, documents, fleet node
assignments, the staff incident list, perspectives, reviews, pins, watches and
estimates show only the viewer's graphs, and API keys only the viewer's own.
The catalog lists one row per graph: its versions and retried runs are one
graph. Only the owner deletes a graph, after a confirmation that names what
goes, and it cannot be undone. Deletion removes the selected runtime's durable
data, the graph's console run records across runtimes, local copies, name, gold
file, workspace items, and upload folders no other graph uses. A graph with
an active catalog run in any runtime waits until that run is cancelled.
For terminal graphs stored in multiple backends, delete other backends' durable
data through their CLI before removing the console records.

File-backed state uses one console process per state root. Workspace changes
are serialized across request handlers in that process.

Ownership and grants are kept in `access/graphs.json` under the state root,
along with who created each upload folder and the people who have signed in
(offered as suggestions when sharing). A graph missing from the file takes its owner from the earliest run
record that names one. The file is read on every request; an operator may edit
it to assign owners or grants, with principals in upper case.

The header-trust requirement: `FLAKEGRAPH_TRUST_IDENTITY_HEADERS` makes the
console believe whatever `X-Auth-Request-*` headers arrive. That is sound only
when the application is unreachable except through the gate that writes them.
The Helm chart enforces this with a NetworkPolicy admitting only the ingress
controller, and its forwardAuth middleware overwrites those headers on every
request. Do not set the flag on a console that can be reached directly (a
NodePort, a port-forward, another pod), and do not add headers the gate does
not overwrite to the list in `src/server/identity.ts`.

Every response carries `X-Frame-Options: DENY`, a same-origin
Content-Security-Policy and the other headers in `next.config.ts`. Model
errors are logged by message only. The Ask feature makes one request the
operator did not configure: a probe of the Ollama address on the loopback
(`http://127.0.0.1:11434/api/tags`, once per process) when no hosted model is
named; `FLAKEGRAPH_ASK_DISABLE_OLLAMA=1` turns it off.

Hosted in Snowflake (`FLAKEGRAPH_SNOWFLAKE_HOSTED=1`), the runtime reads and
writes the account's own KG_* tables and starts a job service per run; see
[Console on App Runtime](../docs/snowflake-setup.md#console-on-app-runtime).
Anywhere else the Snowflake runtime is a rehearsal: it records runs in a JSON
store under the state root and does not touch an account.

Deploy to Snowflake App Runtime with `bun scripts/deploy-snowflake.ts -c <connection>`,
which stages this directory with the repository's configs and runs `snow app deploy`.
Do not move `app.yml` to the repository root, which would upload the Python tree.

## Environment

Every variable the console reads. Booleans accept `1`, `true`, `yes`, `on`.

### Identity and access

| Variable | Default | Role |
| --- | --- | --- |
| `FLAKEGRAPH_TRUST_IDENTITY_HEADERS` | off | Believe the sign-in gate's `X-Auth-Request-*` headers. Only behind a gate that overwrites them. |
| `FLAKEGRAPH_DEMO_IDENTITIES` | off | Serve the sidebar's demo principals and any session that assumed none. Laptops and e2e only. |
| `FLAKEGRAPH_APP_ANONYMOUS_READ` | off | Serve read-only catalog queries to callers nobody identified. |
| `FLAKEGRAPH_APP_REQUIRE_SIGN_IN` | off | Refuse anonymous callers everywhere, even where the two flags above would serve them. |
| `FLAKEGRAPH_APP_OPERATORS` | none | Who administers a console behind the gate or in Snowflake: sign-in addresses, user names and roles (gate groups, Snowflake roles), separated by commas or spaces, matched without regard to case. Only they change console-wide settings; with none named, nobody does. |
| `FLAKEGRAPH_SNOWFLAKE_HOSTED` | off | Read the `Sf-Context-Current-User` header, require sign-in, and pin the runtime to the account's own tables. Only inside a Snowflake container; `app.yml` sets it. |
| `FLAKEGRAPH_SNOWFLAKE_WAREHOUSE` | `SNOWFLAKE_QUERY_WAREHOUSE` | The warehouse a hosted console queries with and a worker bulk-loads through. |
| `FLAKEGRAPH_SNOWFLAKE_TOKEN_PATH` | `/snowflake/session/token` | The OAuth token Snowflake mounts into a hosted console. |
| `FLAKEGRAPH_APP_SIGN_OUT_URL` | none | Where "Sign out" sends a gated viewer (the gate's sign-out URL). |
| `FLAKEGRAPH_APP_SOURCE_ROOTS` | none | Extra folders a run may read documents from, separated by `:` (`;` on Windows). |

### Where things are

| Variable | Default | Role |
| --- | --- | --- |
| `FLAKEGRAPH_REPOSITORY_ROOT` | found by walking up from the working directory | The checkout that holds `configs/app-defaults.yaml`, the sample packs, and the pipeline. |
| `FLAKEGRAPH_APP_STATE_ROOT` | `<repository>/.flakegraph/app` | Catalog records, uploads, generated configurations, graph artifacts, the workspace. |
| `FLAKEGRAPH_CLI` | `uv run flakegraph` | The pipeline command, split on whitespace. |
| `FLAKEGRAPH_APP_DEFAULT_RUNTIME` | `local` | `local`, `kubernetes` or `snowflake`; the runtime a browser opens on. |
| `FLAKEGRAPH_APP_KUBERNETES_NAMESPACE` | `flakegraph` | The one namespace the fleet pages read. |
| `FLAKEGRAPH_STUB_RUNTIMES` | off | File-backed Kubernetes and Snowflake modules for demos and tests. |
| `DATABASE_URL` (or `FLAKEGRAPH_DATABASE_URL`) | none | The fleet's coordination Postgres, for listing distributed runs. Passed to the pipeline as `DATABASE_URL`. |
| `FLAKEGRAPH_APP_GRAFANA_URL` | none | Dashboards link on the fleet page. |

### Ask (questions over a finished graph)

| Variable | Default | Role |
| --- | --- | --- |
| `FLAKEGRAPH_ASK_API_KEY`, `FLAKEGRAPH_ASK_BASE_URL`, `FLAKEGRAPH_ASK_MODEL` | none | A hosted OpenAI-compatible or Azure OpenAI model. `AZURE_OPENAI_*` and `OPENAI_API_KEY`/`OPENAI_API_BASE_URL` are read as fallbacks. |
| `FLAKEGRAPH_ASK_API_VERSION` | `2024-12-01-preview` | Azure API version (`AZURE_OPENAI_API_VERSION` as fallback). |
| `FLAKEGRAPH_ASK_SECRETS_FILE` | none | One dotenv-style file to read the variables above from. Nothing is looked for by convention; the file read is logged once. |
| `FLAKEGRAPH_ASK_OLLAMA_BASE_URL` (or `OLLAMA_HOST`) | `http://127.0.0.1:11434/v1` | A local Ollama, used when no hosted model is named. Only loopback addresses are accepted without a probe. |
| `FLAKEGRAPH_ASK_OLLAMA_MODEL` | `qwen3:4b-instruct` | The Ollama model. |
| `FLAKEGRAPH_ASK_DISABLE_OLLAMA` | off | Never probe or use Ollama. |
| `FLAKEGRAPH_ASK_STRUCTURED_REASONING` | `none` | `reasoning_effort` for planning, scoring, follow-ups and type suggestions: `none` … `high`, or `inherit`. |

Ask searches the complete published graph on the server. Browser review limits
only bound explorer payloads; they do not limit entity, relation, community or
evidence retrieval for questions. Every Ask tool uses the selected document or
perspective scope. Perspective quotes come from chunks that ground the retained
entities and relations; a shared supporting chunk can contain neighboring text.

### Scripts and tests

| Variable | Default | Role |
| --- | --- | --- |
| `PORT` | `3100` | Port for `dev:demo` / the e2e server. |
| `FLAKEGRAPH_ASK_NO_MODEL` | off | `dev:demo` and e2e: no hosted model and no Ollama. |
| `FLAKEGRAPH_KEEP_STATE` | off | `dev:demo`: keep the state root instead of reseeding. |
| `PLAYWRIGHT_PORT` | `3100` | Port the e2e suite starts its server on. |
| `PLAYWRIGHT_CHANNEL` | bundled Chromium | `chrome`, `msedge`, … to run e2e against an installed browser. |
| `PLAYWRIGHT_REUSE` | off | Reuse a server already listening on the port. |
| `FAKE_FLAKEGRAPH_FAIL`, `FAKE_FLAKEGRAPH_DELAY_MS` | off, `150` | Knobs on the fake worker used by tests. |

Credentials a run references (`KG_LLM_API_KEY`, `KG_SNOWFLAKE_PASSWORD`, …) are
the pipeline's own; see the Helm chart and `../docs/` for the full set.

## Tests

```bash
bun run typecheck
bun run lint
bun run test        # Vitest: schemas, catalog, confinement, auth, runtimes
bun run test:e2e    # Playwright operator flows against the fake worker
```

The e2e server runs with `FLAKEGRAPH_STUB_RUNTIMES=1`,
`FLAKEGRAPH_DEMO_IDENTITIES=1` and the fake CLI, so it needs no GPU, cluster
or account. Ask journeys that need a language model run against whatever the
environment names and skip otherwise; `FLAKEGRAPH_ASK_NO_MODEL=1` runs the
suite the way CI does. Set `PLAYWRIGHT_PORT` and `FLAKEGRAPH_APP_STATE_ROOT`
to run two suites side by side.

## Layout

Pages never import a runtime module. tRPC procedures call the `ControlPlane`
interface (`src/server/protocol/runtime.ts`); Effect Schema
(`src/server/protocol/schema.ts`) is the source of truth for
`IngestionRequest`, `RunSnapshot`, `GraphDataset`, `Viewer` and capabilities.
`src/server/auth.ts` decides who a request is from, `src/server/confine.ts`
what it may name, and `src/server/trpc/init.ts` which procedures need which.

A graph's known gaps come from the `discarded_windows` table beside its other
artifacts (`src/server/gaps.ts` folds it per document; `extraction-gaps-card.tsx`
draws it). The Quality tab, the run header and the Documents table all read
the same summary, so a corpus of hundreds of documents with a thousand gaps
reads the same way as one document with one: a count, the reasons, a paged
table, and one document's windows at a time. The seeded demo has both
(`run_deep_learning` with three, `run_gappy_corpus` with 1,178).

A graph with a gold file (uploaded on the Quality tab, or a sample pack's
`gold.json`) is scored by the pipeline's own evaluator, `flakegraph inspect
evaluate` (`src/server/benchmark.ts`): entity and relation precision, recall
and F1, evidence support, two-hop recoverability and the gold's acceptance
gates, the numbers a benchmark run from the command line reports. The score is
cached per run under `<state root>/benchmarks/` and redone only when the
graph's files or the gold change. Precision and F1 read n/a under a
`reference` gold, which lists a selection and cannot count extra findings as
wrong. For a sample pack, the results it records in `results/*.json` sit
beside the score as baselines. Choosing a sample pack also makes its
`ontology.yaml` the run's vocabulary, so the graph is extracted in the terms
its gold is written in; a run keeps each term's aliases, endpoint types and
evidence cues from the profile its terms were offered from, whatever the form
edited.
