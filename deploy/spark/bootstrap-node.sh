#!/usr/bin/env bash
# Prepare an NVIDIA DGX Spark to run FlakeGraph as a Kubernetes node.
#
# This is the reference bring-up for a bare-metal DGX Spark (GB10) fleet on
# k3s; see deploy/spark/README.md. Any cluster that already has a GPU-capable
# container runtime and the chart's prerequisites needs none of it.
#
# Every node in the fleet runs this same script. The first node takes --role
# server, which starts an etcd-backed control plane that further nodes join
# rather than a single-node cluster that would have to be rebuilt to grow.
#
# Run it with sudo, from the node itself:
#   sudo ./bootstrap-node.sh --role server
#   sudo ./bootstrap-node.sh --role agent --server https://<server-ip>:6443 --token <token>
#
# It is idempotent: re-running reports what is already in place and changes
# nothing else, so it doubles as a way to verify a node still matches the fleet.
#
# Optional, for hardened images and networks that intercept TLS: some hardened
# Ubuntu images (CIS profiles among them) blacklist the overlay filesystem,
# which every container image needs; the "Overlay filesystem exception" step
# lifts that one directive and is a no-op elsewhere. Some networks resolve
# github.com but intercept or block the host its release assets redirect to;
# a k3s binary staged beside this script by stage-artifacts.sh is used
# instead of the installer's own download when present.
set -euo pipefail

role=""
server_url=""
join_token=""
node_class="${FLAKEGRAPH_NODE_CLASS:-nvidia-spark}"
operator_sudo=0
backup_root="${FLAKEGRAPH_BACKUP_ROOT:-/etc/flakegraph/backups}"
k3s_channel="${FLAKEGRAPH_K3S_CHANNEL:-stable}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) role="${2:-}"; shift 2 ;;
    --server) server_url="${2:-}"; shift 2 ;;
    --token) join_token="${2:-}"; shift 2 ;;
    --node-class) node_class="${2:-}"; shift 2 ;;
    --operator-sudo) operator_sudo=1; shift ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$role" in
  server|agent) ;;
  "") echo "--role is required (server for the first node, agent for the rest)" >&2; exit 2 ;;
  *) echo "--role must be 'server' or 'agent'" >&2; exit 2 ;;
esac

if [[ "$role" == "agent" && ( -z "$server_url" || -z "$join_token" ) ]]; then
  echo "--role agent requires --server and --token from an existing server node" >&2
  echo "read the token there with: sudo cat /var/lib/rancher/k3s/server/node-token" >&2
  exit 2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "this script configures kernel modules and system services; run it with sudo" >&2
  exit 1
fi

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }

# ---------------------------------------------------------------------------
step "Preflight"

arch="$(uname -m)"
[[ "$arch" == "aarch64" ]] || ok "warning: expected aarch64 (GB10), found $arch"
ok "kernel $(uname -r) on $(. /etc/os-release && echo "$PRETTY_NAME")"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: install the NVIDIA driver before joining the fleet" >&2
  exit 1
fi
ok "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# The address the API server is reached at, both from other nodes and from
# the operator's own kubeconfig. Read once, before any branch may skip it.
primary_ip="$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)"

mkdir -p "$backup_root"

# ---------------------------------------------------------------------------
step "Overlay filesystem exception"

# Hardening profiles (CIS among them) disable a set of uncommon filesystems to
# shrink the kernel's attack surface. overlay is in that set, but unlike cramfs
# or hfsplus it is not uncommon on a container host: it is the mechanism every
# container image uses. The blacklist is therefore lifted here rather than
# worked around, because the alternatives are worse — the vfs graphdriver
# copies every layer in full and would turn a 12 GB CUDA image into tens of
# gigabytes per container, and fuse-overlayfs moves the same code into
# userspace without removing it. On an image with no such directive this step
# reports that and changes nothing.
#
# The directives are commented rather than deleted, and each untouched original
# is copied to the backup directory first, so the change is reversible with a
# single cp.
#
# Every file in modprobe.d is scanned, not just the obviously named one. A
# hardened image can carry the directive twice — once in an obviously named
# file and again in one that is mode 0640 and so invisible to an unprivileged
# survey. The second copy may use "install overlay /bin/true", which makes
# modprobe exit successfully while loading nothing: lifting only the first
# blacklist leaves a node that reports success and still cannot start a
# container.
stamp="$(date +%Y%m%d%H%M%S)"
lifted=0
for conf in /etc/modprobe.d/*.conf; do
  [[ -f "$conf" ]] || continue
  grep -qE '^[[:space:]]*(install|blacklist)[[:space:]]+overlay\b' "$conf" || continue
  cp -a "$conf" "$backup_root/$(basename "$conf").$stamp"
  sed -i -E 's/^([[:space:]]*(install|blacklist)[[:space:]]+overlay\b)/# \1/' "$conf"
  ok "lifted the overlay blacklist in $conf"
  lifted=$((lifted + 1))
done
if [[ "$lifted" -eq 0 ]]; then
  ok "overlay blacklist already lifted"
else
  ok "originals backed up under $backup_root"
fi

# Persist the module across reboots. Docker starts before anything would mount
# an overlay filesystem on demand, so relying on autoload leaves the daemon
# failing on every boot even once the blacklist is gone.
echo "overlay" > /etc/modules-load.d/flakegraph-overlay.conf
modprobe overlay || true
# modprobe's exit status cannot be trusted here: an "install ... /bin/true"
# directive left anywhere in modprobe.d makes it report success without loading
# anything. The filesystem appearing in /proc is the only proof that matters.
if ! grep -qw overlay /proc/filesystems; then
  echo "overlay did not register as a filesystem after modprobe." >&2
  echo "check for remaining directives: grep -r overlay /etc/modprobe.d/" >&2
  exit 1
fi
ok "overlay module loaded and set to load at boot"

# ---------------------------------------------------------------------------
step "Docker"

if ! command -v dockerd >/dev/null 2>&1; then
  echo "dockerd is not installed; install docker-ce before running this script" >&2
  exit 1
fi

systemctl reset-failed docker.service docker.socket 2>/dev/null || true
systemctl enable --now docker >/dev/null 2>&1 || systemctl start docker
sleep 2
if ! docker info >/dev/null 2>&1; then
  echo "docker failed to start; see: journalctl -u docker -n 50" >&2
  exit 1
fi
ok "docker $(docker version --format '{{.Server.Version}}') using the $(docker info --format '{{.Driver}}') driver"

# The NVIDIA container runtime is what lets a build or an ad-hoc container see
# the GB10. The toolkit ships with the DGX image; this only wires it into the
# daemon's runtime list, and leaves runc as the default so ordinary builds are
# unaffected.
if command -v nvidia-ctk >/dev/null 2>&1; then
  if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    nvidia-ctk runtime configure --runtime=docker >/dev/null
    systemctl restart docker
    sleep 2
    ok "registered the nvidia runtime with docker"
  else
    ok "nvidia runtime already registered with docker"
  fi
fi

# ---------------------------------------------------------------------------
# --operator-sudo: opt-in, and off by default, because it genuinely lowers the
# node's security posture: anyone who reaches this account reaches root without
# a second factor, so it belongs only on a node whose operator account is
# already protected as root would be (key-only SSH, no shared credential). It
# is offered because fleet work is a long tail of privileged steps — joining
# nodes, restarting k3s, rotating images — and typing a password into each one
# from a remote session is the reason people leave passwords in scripts instead.
# Remove /etc/sudoers.d/flakegraph-operator to revert.
if [[ "$operator_sudo" -eq 1 ]]; then
  step "Operator sudo"
  operator="${SUDO_USER:-root}"
  sudoers=/etc/sudoers.d/flakegraph-operator
  printf '%s ALL=(ALL) NOPASSWD:ALL\n' "$operator" > "$sudoers.new"
  chmod 0440 "$sudoers.new"
  # An invalid sudoers file locks everyone out of sudo, so it is validated
  # before it is put in place, never after.
  if visudo -cf "$sudoers.new" >/dev/null; then
    mv "$sudoers.new" "$sudoers"
    ok "$operator may now use sudo without a password (remove $sudoers to revert)"
  else
    rm -f "$sudoers.new"
    echo "generated sudoers file failed validation; left unchanged" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
step "Kubernetes (k3s)"

if systemctl is-active --quiet k3s || systemctl is-active --quiet k3s-agent; then
  ok "k3s already running; leaving the existing installation alone"
else
  export INSTALL_K3S_CHANNEL="$k3s_channel"

  # Some networks resolve github.com but intercept or block the host its
  # release assets redirect to, so the installer's own download fails. A
  # binary staged next to this script is used instead when one is there;
  # stage_dir/k3s and stage_dir/install.sh are what stage-artifacts.sh
  # produces. Without one the installer downloads as it normally would.
  stage_dir="${FLAKEGRAPH_STAGE_DIR:-$(dirname "$(readlink -f "$0")")}"
  installer="https://get.k3s.io"
  if [[ -f "$stage_dir/k3s" ]]; then
    install -m 0755 "$stage_dir/k3s" /usr/local/bin/k3s
    export INSTALL_K3S_SKIP_DOWNLOAD=true
    [[ -f "$stage_dir/install.sh" ]] && installer="$stage_dir/install.sh"
    ok "using the staged k3s binary ($("/usr/local/bin/k3s" --version | head -1))"
  fi
  # A staged installer is a local path; the fallback is a URL. cat handles the
  # first and curl the second, and both feed the same shell.
  fetch() { [[ "$installer" == http* ]] && curl -sfL "$installer" || cat "$installer"; }

  if [[ "$role" == "server" ]]; then
    # --cluster-init starts embedded etcd so additional control-plane nodes
    # can join the cluster.
    #
    # The node's own address is added as a TLS SAN so an operator's kubectl can
    # reach the API over the network rather than only from localhost.
    # INSTALL_K3S_EXEC is written verbatim into the systemd unit, and the
    # installer requires the command and flags on one line.
    #
    # The kubeconfig k3s writes is cluster-admin; 0600 keeps it root's, and the
    # operator gets a copy of their own below rather than a world-readable one.
    fetch | INSTALL_K3S_EXEC="server --cluster-init --tls-san ${primary_ip} --tls-san $(hostname -f 2>/dev/null || hostname) --write-kubeconfig-mode 0600" \
      sh -
    ok "k3s server started with embedded etcd"
  else
    fetch | INSTALL_K3S_EXEC="agent --node-label flakegraph.io/node-class=${node_class} --node-label nvidia.com/gpu.present=true" \
      K3S_URL="$server_url" K3S_TOKEN="$join_token" \
      sh -
    ok "k3s agent joined $server_url"
  fi
fi

if [[ "$role" == "server" ]]; then
  # k3s detects the NVIDIA container runtime at startup and generates a
  # containerd config that exposes it. Confirm rather than assume: a missing
  # runtime here surfaces later as GPU pods that schedule and then run on CPU.
  if grep -q 'nvidia' /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
    ok "k3s containerd exposes the nvidia runtime"
  else
    ok "warning: k3s containerd has no nvidia runtime; GPU pods will not work"
  fi

  # Label the server through the API rather than at install time. --node-label
  # only takes effect when a node first registers, so it cannot correct a node
  # that is already in the cluster; this path is idempotent and also re-applies
  # labels an operator has since removed.
  for _ in $(seq 1 30); do
    /usr/local/bin/k3s kubectl get node "$(hostname)" >/dev/null 2>&1 && break
    sleep 2
  done
  /usr/local/bin/k3s kubectl label node "$(hostname)" --overwrite \
    "flakegraph.io/node-class=${node_class}" nvidia.com/gpu.present=true >/dev/null
  ok "labelled the node flakegraph.io/node-class=${node_class}"

  # Hand the operator a kubeconfig they own. k3s writes root's copy under
  # /etc, and tools that rewrite contexts need a file they can modify. One
  # they already have is theirs, so a re-run leaves it alone.
  target_user="${SUDO_USER:-root}"
  target_home="$(getent passwd "$target_user" | cut -d: -f6)"
  if [[ -n "$target_home" && -d "$target_home" && ! -f "$target_home/.kube/config" ]]; then
    install -d -o "$target_user" -g "$target_user" -m 0700 "$target_home/.kube"
    install -o "$target_user" -g "$target_user" -m 0600 \
      /etc/rancher/k3s/k3s.yaml "$target_home/.kube/config"
    sed -i "s#https://127.0.0.1:6443#https://${primary_ip}:6443#" "$target_home/.kube/config"
    ok "wrote $target_home/.kube/config for $target_user"
  fi
fi

# ---------------------------------------------------------------------------
step "Done"

if [[ "$role" == "server" ]]; then
  cat <<SUMMARY
    Control plane:  https://${primary_ip}:6443
    Join a node:    sudo ./bootstrap-node.sh --role agent \\
                      --server https://${primary_ip}:6443 \\
                      --token \$(sudo cat /var/lib/rancher/k3s/server/node-token)

    The cluster has no GPU device plugin yet. Install the workload layer with
    deploy/spark/install-cluster.sh, which runs unprivileged against this
    kubeconfig.
SUMMARY
else
  echo "    Node joined. Verify from a server node with: kubectl get nodes -o wide"
fi
