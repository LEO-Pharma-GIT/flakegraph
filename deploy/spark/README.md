# Reference bring-up: a DGX Spark k3s fleet

The scripts here take a set of NVIDIA DGX Spark (GB10) machines from a fresh
operating system to a k3s cluster that satisfies every prerequisite of the
FlakeGraph chart. They are a **reference profile, not a requirement**: any
Kubernetes cluster with the prerequisites listed in
[the chart README](../helm/flakegraph/README.md#prerequisites) - a
PostgreSQL, S3-compatible storage, and optionally KEDA, the CloudNativePG
operator, a Prometheus-operator stack and GPU nodes - runs the chart just as
well, and a managed cluster already has most of them. The walkthrough is
[docs/reference-dgx-spark-fleet.md](../../docs/reference-dgx-spark-fleet.md);
the values file that matches what these scripts set up is
[deploy/examples/dgx-spark-k3s-values.yaml](../examples/dgx-spark-k3s-values.yaml).

| Script | Runs as | Does |
| --- | --- | --- |
| `bootstrap-node.sh` | root, on the node | Container runtime, NVIDIA runtime, k3s (embedded etcd on the first node), node labels, an operator kubeconfig. |
| `install-cluster.sh` | operator, anywhere with the kubeconfig | NVIDIA device plugin, DCGM exporter, KEDA, CloudNativePG operator, a single-node MinIO, kube-prometheus-stack (`monitoring-values.yaml`). |
| `node-maintenance.sh` | operator, anywhere with the kubeconfig | Drains a node before a reboot and waits for it to serve again afterwards. |
| `stage-artifacts.sh` | operator workstation | Optional: fetches k3s, verifies its checksum and copies it to a node whose network cannot download it. |

Each script is idempotent and reports what is already in place.

What is specific to this hardware and this shape, and where it is expressed:

- **arm64 and unified memory.** Every node is arm64 with 128 GiB shared
  between CPU and GPU. The example values pin the arm64 engine image, size the
  KV cache against what the node can spare, and keep the LiteLLM gateway on
  the maintainers' root image because their non-root build cannot migrate its
  database on arm64.
- **One GPU per node, shared.** The engine claims it; the parsing pool is
  handed it by the `nvidia` RuntimeClass without a claim
  (`documentParsing.mineru.device.claimGpu: false`).
- **k3s.** Its bundled Traefik terminates the ingress and expresses the gate;
  its embedded control-plane components have no separate metrics listeners,
  which `monitoring-values.yaml` accounts for; it registers the `nvidia`
  RuntimeClass GPU pods name.
- **Single tenant.** The monitoring stack and MinIO run in the application
  namespace. A shared cluster keeps its own and sets `monitoring.namespace`.

Two optional workarounds live in `bootstrap-node.sh` and `stage-artifacts.sh`
for environments that need them and are no-ops elsewhere: lifting an
`overlay` filesystem blacklist that some hardened operating-system images
carry, and installing k3s from a binary staged beside the script when the
node's network intercepts or blocks GitHub's release-asset host.

Security notes:

- `bootstrap-node.sh --operator-sudo` writes a passwordless sudoers entry for
  the invoking account. It is opt-in, off by default, and belongs only on a
  node whose operator account is already protected as root would be.
- The k3s kubeconfig is written mode 0600 (cluster-admin); the operator gets
  a 0600 copy of their own.
- `monitoring-values.yaml` configures Grafana to trust the gate's identity
  header; that is sound only with the chart's `ingress.authProxy` and
  `controlPlane.networkPolicy` on, so that nothing but the ingress controller
  can reach Grafana. Signed-in visitors are Viewers; hand out Editor
  deliberately.
