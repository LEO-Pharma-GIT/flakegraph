#!/usr/bin/env bash
# Take one node out of the fleet for a reboot, and put it back afterwards.
#
#   ./node-maintenance.sh drain   <node>    before the reboot
#   ./node-maintenance.sh restore <node>    after it is back
#
# Run from a machine with the cluster's kubeconfig - the control-plane node
# itself does. A node that is simply rebooted takes its engine, its parsing
# replica, and whatever workers were on it down mid-request; the engine then
# spends a quarter of an hour warming up again while the router keeps sending
# it traffic it cannot yet take. Draining first evicts each pod gracefully,
# within the disruption budgets the chart declares, so the remaining engines
# absorb the load and callers see a slower answer rather than a failed one.
#
# One node at a time: the budgets allow a single engine and a single parsing
# replica to be unavailable, and a second drain would wait on the first.
set -euo pipefail

action="${1:-}"
node="${2:-}"
namespace="${FLAKEGRAPH_NAMESPACE:-flakegraph}"
# Longer than the longest termination grace period the chart declares: the
# workers get 3900 s to finish an OCR request they are waiting on (values.yaml
# terminationGracePeriodSeconds), so a shorter drain gives up on a pod that
# is still allowed to be running and reports failure for a node that would
# have drained. The grace periods are what actually bound each pod.
drain_timeout="${FLAKEGRAPH_DRAIN_TIMEOUT:-70m}"
# Cold engines take about fifteen minutes to become ready on this hardware.
restore_timeout="${FLAKEGRAPH_RESTORE_TIMEOUT:-30m}"

export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }
# "30m" / "2h" / "90" to seconds, the way kubectl spells durations.
seconds() {
  case "$1" in
    *h) echo $(( ${1%h} * 3600 )) ;;
    *m) echo $(( ${1%m} * 60 )) ;;
    *s) echo "${1%s}" ;;
    *)  echo "$1" ;;
  esac
}

if [[ -z "$node" || ! "$action" =~ ^(drain|restore)$ ]]; then
  echo "usage: $0 drain|restore <node>" >&2
  exit 2
fi
kubectl get node "$node" >/dev/null

case "$action" in
  drain)
    step "Cordon $node"
    # Nothing new lands here from this moment; what is here is still serving.
    kubectl cordon "$node" >/dev/null
    ok "no new pods will be scheduled"

    step "Drain $node"
    # DaemonSets (device plugin, node exporter, GPU exporter) stay - they are
    # per node by definition and come back with it. Local emptyDir data is
    # scratch: model warm-up caches and the like, rebuilt on start.
    kubectl drain "$node" \
      --ignore-daemonsets \
      --delete-emptydir-data \
      --timeout="$drain_timeout"
    ok "every evictable pod has left"

    step "Ready to reboot"
    cat <<MESSAGE
    Remaining on $node (DaemonSet pods only):
$(kubectl get pods -A --field-selector "spec.nodeName=$node" --no-headers | awk '{print "      " $2}')

    Reboot the node now. When it is back:  $0 restore $node
MESSAGE
    ;;

  restore)
    step "Wait for $node to report Ready"
    kubectl wait --for=condition=Ready "node/$node" --timeout=10m >/dev/null
    ok "kubelet is back"

    step "Uncordon $node"
    kubectl uncordon "$node" >/dev/null
    ok "pods may be scheduled here again"

    step "Wait for this node's pods to become Ready"
    # Pod Ready includes init containers and readiness gates. A pod without a
    # Ready condition is still starting, even when no container status exists.
    # The reference fleet also needs both serving StatefulSets at their desired
    # replica counts: a pinned engine/parser may not have a node assignment yet.
    deadline=$(( $(date +%s) + $(seconds "$restore_timeout") ))
    while :; do
      # A just-rebooted server may not answer yet; that is retried, not fatal.
      if not_ready=$(kubectl get pods -n "$namespace" --field-selector "spec.nodeName=$node" --request-timeout=15s \
        -o 'go-template={{range .items}}{{if and (ne .status.phase "Succeeded") (ne .status.phase "Failed")}}{{$name := .metadata.name}}{{$ready := false}}{{range .status.conditions}}{{if and (eq .type "Ready") (eq .status "True")}}{{$ready = true}}{{end}}{{end}}{{if not $ready}}{{$name}}{{"\n"}}{{end}}{{end}}{{end}}') &&
        serving_not_ready=$(kubectl get statefulsets -n "$namespace" --request-timeout=15s \
          -l 'app.kubernetes.io/name=flakegraph,app.kubernetes.io/component in (model-serving,document-parsing)' \
          -o 'go-template={{range .items}}{{.metadata.name}}|{{index .metadata.labels "app.kubernetes.io/component"}}|{{.spec.replicas}}|{{.status.readyReplicas}}|{{.metadata.generation}}|{{.status.observedGeneration}}{{"\n"}}{{end}}' \
          | awk -F '|' '
            { seen[$2] = 1 }
            $3 !~ /^[0-9]+$/ || $3 < 1 || $4 !~ /^[0-9]+$/ || $4 != $3 ||
            $5 !~ /^[0-9]+$/ || $6 !~ /^[0-9]+$/ || $6 < $5 { print $1 }
            END {
              if (!seen["model-serving"]) print "missing model-serving StatefulSet"
              if (!seen["document-parsing"]) print "missing document-parsing StatefulSet"
            }'); then
        [[ -z "$not_ready" && -z "$serving_not_ready" ]] && break
      else
        not_ready="Kubernetes API did not answer" serving_not_ready=""
      fi
      if (( $(date +%s) >= deadline )); then
        echo "still not ready after $restore_timeout:" >&2
        printf '      %s\n' "$not_ready" "$serving_not_ready" >&2
        exit 1
      fi
      sleep 20
    done
    ok "node pods and all engine/parser replicas are Ready"
    kubectl get pods -n "$namespace" --field-selector "spec.nodeName=$node" -o wide --no-headers \
      | awk '{print "      " $1 "  " $2}'
    ;;
esac
