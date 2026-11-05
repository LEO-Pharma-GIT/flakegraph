#!/usr/bin/env bash
# Install the workload layer FlakeGraph expects on a freshly bootstrapped node.
#
# bootstrap-node.sh leaves a running k3s with a GPU-capable container runtime and
# nothing else. This script adds the pieces the FlakeGraph chart depends on but
# does not own: GPU scheduling and the GPU metrics exporter, queue-driven
# autoscaling, PostgreSQL for task leases, S3-compatible storage for artifacts,
# and the Prometheus/Grafana stack the chart wires its dashboards and alerts
# into. It is one way to satisfy the chart's prerequisites, written for a
# single-tenant k3s fleet; a cluster that already runs any of these keeps its
# own (see deploy/spark/README.md).
#
# It runs unprivileged against the kubeconfig k3s wrote, because none of it needs
# root — keeping the privileged surface confined to bootstrap-node.sh is the
# point of the split.
#
#   FLAKEGRAPH_DOMAIN=example.com ./install-cluster.sh
#
# The domain is the one the FlakeGraph chart's Ingress will serve; Grafana
# needs it up front to build its own URLs. monitoring-values.yaml must sit
# beside this script.
#
# Everything is installed with `helm upgrade --install`, so re-running converges
# rather than duplicating. Chart versions are pinned: an unpinned fleet drifts
# apart one node at a time and the differences only surface under load.
set -euo pipefail

namespace="${FLAKEGRAPH_NAMESPACE:-flakegraph}"
domain="${FLAKEGRAPH_DOMAIN:-}"
# Distinguishes this cluster's series once several Sparks share a store or an
# Alertmanager. The domain is unique per deployment, so it is the default.
cluster="${FLAKEGRAPH_CLUSTER:-$domain}"
helm_version="${FLAKEGRAPH_HELM_VERSION:-v3.19.0}"
bin_dir="${FLAKEGRAPH_BIN_DIR:-$HOME/.local/bin}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
monitoring_values="$script_dir/monitoring-values.yaml"

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="$bin_dir:$PATH"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }

if ! kubectl get nodes >/dev/null 2>&1; then
  echo "cannot reach the cluster with KUBECONFIG=$KUBECONFIG" >&2
  echo "run bootstrap-node.sh --role server on this node first" >&2
  exit 1
fi

# Checked before anything is installed: failing at the last step would leave
# the operator re-running the whole script for one missing variable.
if [[ -z "$domain" ]]; then
  echo "FLAKEGRAPH_DOMAIN is not set; Grafana needs the public domain for its URLs" >&2
  echo "usage: FLAKEGRAPH_DOMAIN=example.com $0" >&2
  exit 2
fi
if [[ ! -f "$monitoring_values" ]]; then
  echo "missing $monitoring_values; copy it next to this script" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
step "Helm"

if ! command -v helm >/dev/null 2>&1; then
  arch="$(uname -m)"; [[ "$arch" == "aarch64" ]] && arch=arm64 || arch=amd64
  mkdir -p "$bin_dir"
  tmp="$(mktemp -d)"
  # get.helm.sh is a plain object store rather than the GitHub release host,
  # so helm can be fetched on the node itself even where that host is
  # intercepted.
  curl -fsSL "https://get.helm.sh/helm-${helm_version}-linux-${arch}.tar.gz" \
    | tar -xz -C "$tmp"
  install -m 0755 "$tmp/linux-${arch}/helm" "$bin_dir/helm"
  rm -rf "$tmp"
fi
ok "$(helm version --short)"

# ---------------------------------------------------------------------------
step "Chart repositories"

add_repo() {
  helm repo add "$1" "$2" >/dev/null 2>&1 || true
}
add_repo nvdp https://nvidia.github.io/k8s-device-plugin
add_repo gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
add_repo kedacore https://kedacore.github.io/charts
add_repo cnpg https://cloudnative-pg.github.io/charts
add_repo minio https://charts.min.io/
helm repo update >/dev/null
ok "nvdp, gpu-helm-charts, kedacore, cnpg, minio"

# ---------------------------------------------------------------------------
step "GPU scheduling"

# Without the device plugin the GPU is invisible to the scheduler: pods that
# request nvidia.com/gpu stay Pending forever, and pods that do not request it
# run on the CPU while appearing healthy. The runtimeClass comes from k3s, which
# generates it when it detects the NVIDIA container runtime at startup.
helm upgrade --install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --version 0.17.4 \
  --set runtimeClassName=nvidia \
  --set-string nodeSelector."nvidia\.com/gpu\.present"=true \
  --wait --timeout 5m >/dev/null
ok "nvidia device plugin installed"

# The DCGM exporter serves DCGM_FI_DEV_* (utilisation, power, temperature) per
# GPU on port 9400. The Fleet Overview dashboard's GPU row reads it, through
# the additional ServiceMonitor in monitoring-values.yaml, which names this
# namespace. A cluster with the GPU Operator already has one; skip this step
# there and point the dashboard's DCGM namespace variable at its namespace.
helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace gpu-monitoring --create-namespace \
  --version "${FLAKEGRAPH_DCGM_EXPORTER_CHART_VERSION:-4.8.4}" \
  --set runtimeClassName=nvidia \
  --set-string nodeSelector."nvidia\.com/gpu\.present"=true \
  --set serviceMonitor.enabled=false \
  --wait --timeout 5m >/dev/null
ok "dcgm-exporter installed in gpu-monitoring"

# ---------------------------------------------------------------------------
step "Queue-driven autoscaling"

# FlakeGraph scales workers from the depth of its PostgreSQL task queue rather
# than from CPU, so KEDA is a hard dependency of the chart's autoscaling paths.
#
# The operator's Prometheus metrics (keda_scaler_*: what each scaler last
# read, whether it errored, whether it is active) are off by default; enabling
# them adds a `metrics` port to the keda-operator Service that the monitoring
# stack below scrapes. It is the only view of what KEDA thinks the queue depth
# is, which is the first question when workers fail to scale.
helm upgrade --install keda kedacore/keda \
  --namespace keda --create-namespace \
  --version 2.17.2 \
  --set prometheus.operator.enabled=true \
  --wait --timeout 10m >/dev/null
ok "keda installed"

# ---------------------------------------------------------------------------
step "PostgreSQL operator"

# The FlakeGraph chart can create its own CloudNativePG Cluster, but the
# operator that reconciles it is cluster-scoped and therefore installed here.
#
# Pulled from the OCI registry rather than the https repository: the index on
# cloudnative-pg.github.io resolves chart tarballs to GitHub's release-asset
# host, which some networks intercept with an untrusted certificate. ghcr.io
# serves the identical chart.
helm upgrade --install cnpg oci://ghcr.io/cloudnative-pg/charts/cloudnative-pg \
  --namespace cnpg-system --create-namespace \
  --version 0.29.1 \
  --wait --timeout 10m >/dev/null
ok "cloudnative-pg operator installed"

# ---------------------------------------------------------------------------
step "Object storage"

# Payload bytes and Parquet graph tables move through S3, not through the
# database. A single-node MinIO is right for one Spark and is the piece to
# replace with a managed endpoint when the fleet outgrows this machine.
kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f - >/dev/null

if kubectl -n "$namespace" get secret flakegraph-artifacts >/dev/null 2>&1; then
  root_user="$(kubectl -n "$namespace" get secret flakegraph-artifacts -o jsonpath='{.data.rootUser}' | base64 -d)"
  root_password="$(kubectl -n "$namespace" get secret flakegraph-artifacts -o jsonpath='{.data.rootPassword}' | base64 -d)"
else
  # Generated once and stored only in the cluster. A checked-in default would be
  # the same password on every node anyone ever racks.
  root_user=flakegraph
  root_password="$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 32)"
  ok "generated object storage credentials"
fi
# One secret serves both sides: MinIO's chart reads rootUser/rootPassword,
# the FlakeGraph chart reads access-key-id/secret-access-key (the names
# deploy/examples/dgx-spark-k3s-values.yaml relies on). Applied rather than
# created so a re-run converges.
kubectl -n "$namespace" create secret generic flakegraph-artifacts \
  --from-literal=rootUser="$root_user" \
  --from-literal=rootPassword="$root_password" \
  --from-literal=access-key-id="$root_user" \
  --from-literal=secret-access-key="$root_password" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

helm upgrade --install minio minio/minio \
  --namespace "$namespace" \
  --version 5.4.0 \
  --set mode=standalone \
  --set replicas=1 \
  --set persistence.size=200Gi \
  --set resources.requests.memory=2Gi \
  --set-string rootUser="$root_user" \
  --set-string rootPassword="$root_password" \
  --set buckets[0].name=flakegraph-artifacts \
  --set buckets[0].policy=none \
  --set buckets[0].purge=false \
  --wait --timeout 10m >/dev/null
ok "minio installed with bucket flakegraph-artifacts"

# ---------------------------------------------------------------------------
step "Monitoring"

# kube-prometheus-stack goes into the application namespace, not a monitoring
# one. k3s's Traefik forbids cross-namespace middleware references and
# ExternalName backends, so Grafana can only sit behind the application's
# shared SSO middlewares if it shares their namespace. The FlakeGraph chart
# therefore renders Grafana's Ingress, NetworkPolicy, ServiceMonitors,
# dashboards and rules itself; this step only installs the stack, configured
# by monitoring-values.yaml beside this script. (The chart's
# monitoring.namespace value supports a stack elsewhere; it then publishes
# Grafana through its own ingress.)
monitoring_version="91.2.2"
# The same detour the PostgreSQL operator takes: the community repository
# index points at GitHub's release-asset host, which some networks intercept.
monitoring_chart="oci://ghcr.io/prometheus-community/charts/kube-prometheus-stack"

if ! kubectl -n "$namespace" get secret flakegraph-grafana-admin >/dev/null 2>&1; then
  # People sign in through the SSO proxy; this credential is for Grafana's
  # HTTP API. Generated once and kept only in the cluster, like the object
  # storage credential above.
  kubectl -n "$namespace" create secret generic flakegraph-grafana-admin \
    --from-literal=admin-user=admin \
    --from-literal=admin-password="$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 32)" >/dev/null
  ok "generated grafana admin credentials"
fi

# Helm installs a chart's CRDs once and never touches them again, so an
# upgraded operator would run against stale CRDs and reject the fields it
# was upgraded for. Applying them from the chart before every install keeps
# the two in step; server-side apply with --force-conflicts is what the chart
# documents, because the Kubernetes CRD objects are too large for the
# client-side last-applied annotation.
helm show crds "$monitoring_chart" --version "$monitoring_version" \
  | kubectl apply --server-side --force-conflicts -f - >/dev/null
ok "prometheus operator CRDs applied"

# The key `grafana.ini` contains a dot, hence the escaped form in --set.
helm upgrade --install monitoring "$monitoring_chart" \
  --namespace "$namespace" \
  --version "$monitoring_version" \
  --values "$monitoring_values" \
  --set-string "grafana.grafana\.ini.server.root_url=https://grafana.$domain/" \
  --set-string "prometheus.prometheusSpec.externalLabels.cluster=$cluster" \
  --wait --timeout 15m >/dev/null
ok "kube-prometheus-stack installed"

# ---------------------------------------------------------------------------
step "Done"

cat <<SUMMARY
    Namespace:      $namespace
    Object storage: http://minio.$namespace.svc.cluster.local:9000
    Credentials:    secret/flakegraph-artifacts in $namespace
    Grafana:        https://grafana.$domain/  (once the FlakeGraph chart renders its Ingress)
    Grafana admin:  secret/flakegraph-grafana-admin in $namespace (API use; people sign in via SSO)
    Prometheus:     http://monitoring-kube-prometheus-prometheus.$namespace.svc.cluster.local:9090

    Install FlakeGraph itself with the chart in deploy/helm/flakegraph, using a
    values file derived from deploy/examples/dgx-spark-k3s-values.yaml.
SUMMARY
