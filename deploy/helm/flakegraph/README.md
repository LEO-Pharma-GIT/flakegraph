# FlakeGraph Helm chart

Deploys the FlakeGraph worker pools, the optional console, the LiteLLM gateway, the
document-parsing plane and, optionally, an in-cluster vLLM serving plane with
priority-aware placement, on any Kubernetes 1.28 or newer. The full guide is
[docs/kubernetes-fleet.md](../../../docs/kubernetes-fleet.md); this file is the
chart's own contract: what it needs from the cluster, which Secrets it expects,
and which profiles ship with it.

## What the defaults assume

Nothing beyond Kubernetes itself. With the shipped `values.yaml`:

- Images come from `ghcr.io/leo-pharma-r-d-data-analytics/flakegraph` and
  `.../flakegraph-spark` at the chart's `appVersion`, published for amd64 and
  arm64 by `.github/workflows/publish.yml`. A fork sets `image.repository`.
- No CRD-backed object is rendered. KEDA autoscaling
  (`autoscaling.enabled`), a CloudNativePG cluster
  (`database.cloudNativePG.enabled`), Prometheus-operator monitoring
  (`monitoring.enabled`) and Traefik objects (`ingress.controller: traefik`)
  are all opt-in, and each refuses to render on a cluster whose API discovery
  does not serve the CRD, naming what to install or which `--api-versions` to
  pass to an offline `helm template`.
- Model serving is off. When on, it serves `Qwen/Qwen3-8B` in bf16 on one
  Ampere-or-newer GPU with no speculative drafter, no RuntimeClass, and a
  sizing block worked for an 80 GB device. Anything hardware-specific is an
  example profile, not a default.
- The gateway forwards to the in-cluster engines
  (`gateway.litellm.upstream.type: engine`). Set `type: external` with a
  `models` list to forward every alias to an OpenAI-compatible provider
  instead; the engines, the placement router and the sidecar keys are then
  not rendered.
- The console is disabled until its protected gate or isolated fixed-principal
  admission mode is configured.
- The ingress is a plain `networking.k8s.io/v1` Ingress with no gate and no
  edge compression (`ingress.controller: none`).
- Every pod runs non-root with a read-only root filesystem, the LiteLLM
  gateway included (it runs the maintainers' `litellm-non_root` build).
- Nothing phones home: vLLM usage statistics, LiteLLM telemetry and its
  startup fetch of a cost map from GitHub, and Hugging Face Hub telemetry are
  all off by default (`modelServing.server.usageStats`,
  `gateway.litellm.telemetry`, `gateway.litellm.localModelCostMap`).

## Profiles

| Values file | Shape |
| --- | --- |
| `values.yaml` (defaults) | Portable: any GPU, engines off, external nothing, plain Ingress. |
| `deploy/examples/minimal-cpu-values.yaml` | No GPU, no operators: Ollama or any OpenAI-compatible endpoint through the gateway, built-in text extraction, fixed replicas. Runs on kind. |
| `deploy/examples/external-provider-values.yaml` | No in-cluster engine: the gateway forwards to a hosted or remote provider; a managed PostgreSQL; parsing pool on CPU. |
| `deploy/examples/dgx-spark-k3s-values.yaml` | The bare-metal reference: NVFP4 checkpoint, arm64 engine image, measured drafter, unified-memory sizing, k3s RuntimeClass, Traefik, in-namespace monitoring. See [docs/reference-dgx-spark-fleet.md](../../../docs/reference-dgx-spark-fleet.md). |

CI renders all four, with and without the optional CRDs, so a change that
breaks one of them is caught before release.

## Prerequisites

| Need | Values key | What satisfies it |
| --- | --- | --- |
| PostgreSQL 14+ | `database.secretName` (a URI) | Any managed service, or `database.cloudNativePG.enabled` with the CloudNativePG operator installed. |
| S3-compatible storage (Spark finalization) | `artifactStorage.*`, `spark.enabled` | Any S3 API: a cloud bucket, MinIO, Ceph RGW. |
| GPU nodes (in-cluster serving only) | `modelServing.*` | NVIDIA device plugin or GPU Operator; `modelServing.runtimeClassName` only where GPU pods must name a RuntimeClass (k3s). |
| KEDA (queue-driven scaling) | `autoscaling.enabled` | `helm install keda kedacore/keda`. Off, pools run fixed replicas. |
| Prometheus operator (monitoring) | `monitoring.enabled`, `monitoring.release`, `monitoring.namespace` | Any kube-prometheus-stack whose Prometheus selects monitors from this namespace. In-namespace to publish Grafana behind the chart's gate. |
| DCGM exporter (GPU panels) | dashboard variable | GPU Operator's, or `deploy/spark/install-cluster.sh` installs one; the Fleet Overview dashboard reads it from whichever namespace scrapes it. |
| Ingress controller | `ingress.controller` | `none` for a plain Ingress anywhere; `traefik` or `nginx` for the sign-in gate. |

## Secrets the chart expects

The chart creates no credential. Every Secret below is the operator's to
create in the release namespace before `helm install`; the pre-install hook
(`secretsCheck.enabled`, on by default) reads each one by name and refuses the
install with the full list of what is missing, and `NOTES` prints the same list
after install. Which ones are needed depends on the features enabled.

| Secret (values key) | Keys | Needed when |
| --- | --- | --- |
| `flakegraph-postgres-app` (`database.secretName`) | `uri` (`database.secretKey`): a PostgreSQL URI | Always, unless `database.cloudNativePG.enabled`, which creates it. |
| `flakegraph-providers` (`providerSecret.name`) | `KG_LLM_API_KEY`, `KG_EMBED_API_KEY`, `KG_SNOWFLAKE_PASSWORD`, `KG_SNOWFLAKE_OAUTH_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - every key optional | Always (the Secret must exist unless `providerSecret.optional`). |
| `flakegraph-gateway-keys` (`gateway.litellm.virtualKeySecret`) | `LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`, `LITELLM_VIRTUAL_KEY_BATCH` | `gateway.enabled`. The batch key is a LiteLLM virtual key the workers call with; mint it after first start, or pre-generate one (`sk-...`) and register it. With `embeddingServing.enabled` the key must also list `embeddingServing.model.name` among its models, or every embedding call is refused. |
| `flakegraph-serving-keys` (`gateway.litellm.upstreamKeySecret`, `modelServing.sidecar.keySecret`) | `SIDECAR_KEY_INTERACTIVE`, `SIDECAR_KEY_DEV`, `SIDECAR_KEY_BATCH`, and `serving-keys.json` mapping each of those key values to its class | `gateway.litellm.upstream.type: engine` (the gateway presents them), and `modelServing.enabled` (the sidecar checks them). Not needed with an external upstream. |
| (your name) (`gateway.litellm.upstream.apiKeySecret`) | (your key) | `upstream.type: external` with a provider that wants a key. |
| `flakegraph-ocr-keys` (`documentParsing.shim.keySecret`) | `ocr-keys.json` mapping keys to classes; `KG_MINERU_API_KEY`, the workers' key | `documentParsing.enabled`. |
| (your name) (`artifactStorage.existingSecret`) | `access-key-id`, `secret-access-key` | `artifactStorage.uri` is set. |
| (your name) (`controlPlane.ask.secretName`) | `FLAKEGRAPH_ASK_API_KEY`: a virtual key on the interactive alias | Asking questions over a graph from the console. |
| (your name) (`ingress.authProxy.existingSecret`) | `client-secret`, `cookie-secret` (32 random bytes, base64) | `ingress.authProxy.enabled`. |
| (your name) (`ingress.tls.secretName`) | `tls.crt`, `tls.key` | TLS at the ingress. |
| (your name) (`modelServing.huggingFaceTokenSecret`) | `HF_TOKEN` | A gated or private checkpoint. |

The console's Role may `get` exactly these Secrets by name and nothing else;
it never lists the namespace.

Spark finalizers receive `KG_DISTRIBUTED_SPARK_PROVIDER_SECRET_REFS` as a JSON
map from environment variable names to `Secret-name:key` references. The chart
derives it from the driver's provider mappings and gateway virtual key so
executors use those same bindings. Standalone Spark installations set
`distributed.spark_provider_secret_refs` explicitly.
Credentials are never copied into literal executor environment values.

## Console admission

The console is disabled by default. Enable `controlPlane.enabled` with the
protected `ingress.authProxy` gate and its ingress-only NetworkPolicy for a
shared installation; the DGX profile configures this path. Initial admission
comes from the gate, so a signed-in viewer can mint console API keys.

For an isolated single-operator installation, leave `ingress.enabled=false`
and explicitly set `controlPlane.auth.singleOperator` to the operator's
address alongside `controlPlane.enabled=true`. Reach it with an operator-owned
local port-forward and restrict access to its ClusterIP Service to trusted
workloads. Everyone able to reach this console acts as that fixed principal;
there is no sign-in challenge. This mode accepts neither caller identity
headers nor UI identity switching and cannot be combined with ingress,
`auth.required`, or demo identities. An initial browser session already has
the named principal and can create its first API key. The application otherwise
remains closed to unidentified callers.

## Ingress controllers and the sign-in gate

`ingress.controller` selects how the gate (`ingress.authProxy`) is expressed:

- `traefik`: a forwardAuth Middleware, with an identity-strip Middleware first
  in the chain that removes every `X-Auth-Request-*` and `X-Forwarded-*`
  identity header a client sent before the gate sets the real ones; edge
  compression as a compress Middleware; console API keys routed past the gate
  on an IngressRoute. Needs `traefik.io/v1alpha1`.
- `nginx`: `auth-url` / `auth-signin` annotations, with
  `auth-response-headers` replacing `X-Auth-Request-User`, `-Email`,
  `-Preferred-Username` and `-Groups` on every forwarded request. Console
  machine keys and edge compression are not rendered (both need configuration
  snippets, which ingress-nginx disables by default).
- `none`: a plain Ingress; the chart refuses to render the gate or compression.

Behind the gate the console reads identity only from the `X-Auth-Request-*`
headers, and `controlPlane.networkPolicy` must admit only the ingress
controller, or the header is a field anyone in the cluster can set.

## Labels

`flakegraph.io/node-class` (node placement in the examples) and
`flakegraph.io/scrape` (which Service the engine ServiceMonitor selects) are
label prefixes in the Kubernetes sense: namespaced identifiers the chart owns,
not a domain it resolves or serves anything from.
