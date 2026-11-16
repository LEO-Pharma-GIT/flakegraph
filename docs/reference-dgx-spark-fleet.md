# Reference: a DGX Spark fleet on k3s

This is the bare-metal reference deployment: a handful
of NVIDIA DGX Spark machines (GB10, arm64, 128 GiB of unified memory shared
between CPU and GPU, one GPU each) joined into a k3s cluster, with every
prerequisite installed by the scripts in `deploy/spark/` and the chart
configured by `deploy/examples/dgx-spark-k3s-values.yaml`. It is a worked
example of [Deploy on Kubernetes](kubernetes-fleet.md), not a requirement:
nothing here is a chart default, and a managed cluster or a fleet of discrete
GPUs takes the general guide and skips this one.

What makes this shape different, and what the example values carry for it:

| Property | Consequence | Where |
| --- | --- | --- |
| arm64 nodes | Engine image pinned to the tag's linux/arm64 manifest; LiteLLM on the maintainers' root image, because the non-root build lacks the arm64 Prisma engine | `modelServing.image.digest`, `gateway.litellm.image` |
| Unified memory | `gpuMemoryUtilization: 0.50`, host memory limits that are a share of the node, sizing against 119.2 GiB | `modelServing.server`, `modelServing.sizing`, `modelServing.resources` |
| Blackwell (SM121) | An NVFP4 checkpoint with FlashInfer attention and Marlin MoE kernels | `modelServing.model`, `modelServing.server.*Backend` |
| Bandwidth-bound decode | A measured DFlash2 drafter, block of eight | `modelServing.server.speculative*` |
| One GPU per node | The engine claims it; the parser is handed it unclaimed through the `nvidia` RuntimeClass | `documentParsing.mineru.device` |
| k3s | The `nvidia` RuntimeClass, Traefik as the ingress controller, in-namespace monitoring | `modelServing.runtimeClassName`, `ingress.controller`, `monitoring` |

## Bring-up

`deploy/spark/` automates the cluster for this hardware; see its
[README](../deploy/spark/README.md) for what each script does and which parts
are optional.

| Script | Runs as | Does |
| --- | --- | --- |
| `stage-artifacts.sh` | operator workstation | Optional: fetches k3s, verifies its published SHA-256, copies it to a node whose network cannot download it |
| `bootstrap-node.sh` | root, on the node | Container runtime, NVIDIA runtime, k3s, node labels, an operator kubeconfig |
| `install-cluster.sh` | operator, on the node | Device plugin, DCGM exporter, KEDA, CloudNativePG, object storage, monitoring stack |
| `node-maintenance.sh` | operator, anywhere with the kubeconfig | Drains a node before a reboot and waits for it to serve again afterwards |

`bootstrap-node.sh --role server` starts embedded etcd rather than the k3s
default, because a default single-node server uses SQLite and can never gain a
second control-plane node — a decision that cannot be revisited later without
rebuilding the cluster. Further nodes take `--role agent`. It is idempotent, so
re-running it verifies a node rather than disturbing it. It writes the k3s
kubeconfig mode 0600 and gives the operator a copy of their own;
`--operator-sudo` (passwordless sudo for the operator account) is opt-in and
belongs only on nodes whose operator account is already protected as root
would be.

For high availability, put a stable DNS name or load balancer in front of
three server nodes and join the rest as agents. Server nodes remain ordinary
schedulable workers unless deliberately tainted, so control-plane redundancy
does not remove machines from the compute pool. The k3s
[requirements](https://docs.k3s.io/installation/requirements) and
[embedded-etcd HA](https://docs.k3s.io/datastore/ha-embedded) guides are the
source of truth for the topology; the script passes the flags they describe.

Every node the script joins is labelled `flakegraph.io/node-class=nvidia-spark`
(override with `--node-class`) and `nvidia.com/gpu.present=true`. The second
label is not decoration: the device plugin selects nodes with it, so a GPU node
missing it stays Ready, reports no allocatable GPU, and takes model pods
nowhere. `--node-label` only takes effect the first time a node registers; a
node already in the cluster is labelled through the API:

```bash
kubectl label node <node> flakegraph.io/node-class=nvidia-spark nvidia.com/gpu.present=true
```

All of that is per node rather than per fleet, which is easy to miss until the
fleet gains its second machine: a staged k3s binary, a site root CA, and
`/etc/rancher/k3s/registries.yaml` have to be on each new host before it
registers, or it joins Ready, accepts pods, and fails its first pull from the
registry every other node reaches. The two environmental workarounds the
scripts carry - lifting an `overlay` blacklist on a hardened image, and
installing k3s from a staged binary where GitHub's release-asset host is
intercepted - are described in the general guide's
[appendix](kubernetes-fleet.md#appendix-networks-that-intercept-tls-and-hardened-images).

`install-cluster.sh` then adds what the chart depends on but does not own, all
pinned and installed with `helm upgrade --install` so a re-run converges:

- the NVIDIA device plugin (`kube-system`) and a DCGM exporter
  (`gpu-monitoring`), which the Fleet Overview dashboard's GPU row reads;
- KEDA (`keda`), with its operator metrics on so the monitoring stack can
  scrape what it thinks the queue depth is;
- the CloudNativePG operator (`cnpg-system`);
- a single-node MinIO in the application namespace, with a generated root
  credential kept only in the cluster - the piece to replace with a managed
  endpoint when the fleet outgrows one machine;
- kube-prometheus-stack in the application namespace, configured by
  `monitoring-values.yaml`. k3s's Traefik refuses cross-namespace middleware
  references, so Grafana can only sit behind the chart's sign-in gate if it
  shares the namespace; k3s also embeds the controller manager, scheduler,
  kube-proxy and etcd without separate metrics listeners, so those scrapers
  and their rules are off. `FLAKEGRAPH_DOMAIN` is what it needs, to give
  Grafana its public URL.

Then install the chart from a copy of the example values:

```bash
mkdir -p deploy/private
cp deploy/examples/dgx-spark-k3s-values.yaml deploy/private/fleet-values.yaml
cp configs/app-defaults.yaml deploy/private/fleet-config.yaml
# point draftModelSeed.image at your registry, or set providedExternally

helm upgrade --install flakegraph deploy/helm/flakegraph \
  --namespace flakegraph --create-namespace \
  --values deploy/private/fleet-values.yaml \
  --set-file config.content=deploy/private/fleet-config.yaml \
  --set-file ontology.content=configs/ontologies/general.yaml \
  --wait
```

## Serving profile

The example serves `unsloth/Qwen3.8-27B-NVFP4` at a pinned revision with an
FP8 KV cache, FlashInfer attention, Marlin MoE (`VLLM_MARLIN_USE_ATOMIC_ADD=1`
through `modelServing.extraEnv`), chunked prefill, prefix caching, and
asynchronous scheduling. NVFP4 needs a Blackwell part (SM100/SM120); on
anything older the engine cannot load the checkpoint, which is why it is an
example and not the default. The checkpoint is quantized with
compressed-tensors; NVIDIA's own NVFP4 builds use `modelopt`, and naming the
wrong scheme is a startup failure rather than a silent fallback. It needs
`trustRemoteCode: true`, and its chat template wants the `qwen3` reasoning
parser and the `qwen3_xml` tool-call parser.

Decode on this hardware is bound by weight bandwidth rather than arithmetic,
so the profile drafts: `speculativeMethod: dflash` proposes a block of eight
tokens at a time from a separate draft model, and asynchronous scheduling stays
on with it — vLLM keeps async scheduling for every Eagle-family method. It
needs vLLM mainline rather than NVIDIA's fork, which does not register the
DFlash2 architecture. On one node the drafter decodes a code prompt at 46.1
tok/s against 27.2 for the checkpoint's own MTP head at four, and prose at 20.5
against 18.9; the gap is acceptance, roughly three fifths of drafted tokens on
code and a fifth on prose. Change the method or the
block size only behind a benchmark, and remember that the drafter is resident
for the life of the process, so its weights belong in
`modelServing.sizing.weightsGiB`.

The drafter is named as a filesystem path, so `draftModelSeed.image` names an
image whose filesystem carries it; an init container copies it into each
replica's own volume before the engine starts, writing a `.partial` name and
renaming it, so an interrupted copy is never mistaken for a complete one.
Neither the draft model nor that image is published with this repository.
Build the image with the modes stated in the Dockerfile, not inherited from the
build context: `COPY --chmod=644 <draft-dir> /dflash2`. `COPY` keeps whatever
modes the build host's umask produced, and the init container runs as the same
unprivileged user as the engine, so a context staged under a restrictive umask
yields an image whose files root can read and the copy cannot. That failure is
invisible on any replica that already holds the model and appears only on the
first node that needs seeding.

The checkpoint is a vision-language model, and the profile serves that
modality rather than refusing it: `limitMultimodalPerPrompt` admits two images
per prompt, encoder profiling is left on so the memory is reserved up front
instead of discovered mid-request, and video stays at zero. Image input is not
free — the vision tower's weights and peak activations come out of the same
device budget as the KV cache, and the sizing block models weights, KV, and
recurrent state but not the encoder's reservation. `maxNumSeqs` therefore sits
at half the figure the block implies, and the sidecar's startup check remains
the real guard.

### Sizing on unified memory

`gpuMemoryUtilization: 0.50`. On a unified-memory part the same physical pool
holds the operating system, the kubelet, workers, Spark executors, and
storage, and CUDA allocations are not fully represented in pod memory metrics.
This is not a conservative guess: `0.70` starves sshd and the cluster API and
takes a node off the network, while ICMP keeps answering. Raise it
only on a node doing nothing but inference, raise `maxNumSeqs` with it, and
confirm with the sizing command first.

The checkpoint interleaves 16 full-attention layers with 48 linear-attention
ones, and the latter hold a fixed recurrent state per sequence that vLLM pages
beside the KV cache — it says so at startup by raising the attention block
size until the attention page is at least as large as the mamba page. That
cost does not shrink with a shorter context, so it is charged per sequence,
with two slots per request under prefix caching:

```bash
flakegraph serving sizing --kv-heads 4 --head-dim 256 --attention-layers 16 \
  --recurrent-layers 48 --recurrent-state-bytes-per-layer 3207168 \
  --recurrent-state-slots-per-sequence 2 \
  --weights-gib 24.24 --device-memory-gib 119.2 --max-num-seqs 8
```

24.24 GiB is what the engine reports as "Model loading took" — the target
checkpoint and the resident drafter together; the target alone is 21.81.

With the vision encoder reserved, an engine here reports a KV cache of about
352,000 tokens at a 64k context per request, a maximum concurrency of 5.37x,
and a sequence limit of four sits just under it at that context; the example's
eight is derived at 32k. The figure to size against is the one the engine
reports after profiling, not the one the sizing block implies before it.

### The parser beside the engine

Each node has one GPU and the engine holds its `nvidia.com/gpu`, so a second
claim would never schedule. `documentParsing.mineru.device.claimGpu: false`
hands the parser the device through the `nvidia` RuntimeClass instead
(`NVIDIA_VISIBLE_DEVICES=all`), and `virtualVramGiB: 16` tells MinerU what
share of the unified pool the engine leaves it to batch against — it would
otherwise read the whole device and size its batches accordingly. Even shared,
the GPU parses a scanned paper an order of magnitude faster than the CPU.

### Capacity

Four nodes, one engine each at a sequence limit of eight, is thirty-two
concurrent sequences. The chart sizes the extraction pool from that figure: at
the default oversubscription of 1.5, six eight-slot pods hold forty-eight
windows, so every engine runs at its sequence limit with a short queue behind
it, and an interactive request goes to the head of that queue. Workers are not
pinned to the engine beside them: the endpoint picker spreads the load, so a
node whose own workers are idle still has its GPU fed. Eight two-slot
preparation pods keep sixteen documents in flight, which keeps the parsing pool
busy; the Spark driver lands on any node in the fleet.

Releasing the application does not roll the engines: pin
`modelServing.sidecar.imageTag` at the last build that touched the sidecar, so
an `image.tag` bump for the workers leaves the engines serving. A cold engine
takes about a quarter of an hour to become ready here, which is what
`node-maintenance.sh restore` waits for.

## Observability on this fleet

`monitoring-values.yaml` is the k3s profile of kube-prometheus-stack: the
control-plane scrapers k3s cannot serve are off, the node exporter skips the
`cpufreq` collector (the GB10's `cppc_cpufreq` driver stalls on agent nodes
and fills the exporter's request cap within minutes), Prometheus is bounded
to 8 GiB so it cannot eat into the unified pool as cardinality grows, and the
DCGM exporter in `gpu-monitoring` is scraped through an additional
ServiceMonitor. A GB10 has unified memory, so its GPU memory **is** the node's
memory — DCGM and `nvidia-smi` report none, and the Fleet Overview's memory
panel is the one to watch.

Grafana signs a visitor in as whoever the gate already identified, as
Viewer; `secret/flakegraph-grafana-admin` holds the credential for Grafana's
HTTP API, which nobody needs for looking. Alertmanager ships with a `null`
receiver; a Teams or Slack webhook goes under `alertmanager.config` in
`monitoring-values.yaml`.

## Maintenance

A node that needs a reboot is drained first, one at a time:
`deploy/spark/node-maintenance.sh drain <node>`, reboot, then
`node-maintenance.sh restore <node>`. The disruption budgets the chart declares
let a single engine and a single parsing replica be unavailable, so the other
three keep serving while this one warms up; a node rebooted without draining
takes its engine down mid-request and the router keeps sending it traffic for
the quarter hour it spends warming up. The drain timeout defaults to seventy
minutes, longer than the workers' 3900 s grace period, so a worker finishing
an OCR request is waited for rather than reported as a failed drain.
