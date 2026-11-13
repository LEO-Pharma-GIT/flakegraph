#!/usr/bin/env bash
# Stage the k3s release onto a node that cannot download it itself.
#
# Optional. Some networks resolve github.com but intercept or block the host
# its release assets redirect to (release-assets.githubusercontent.com), and
# the k3s installer's own download step then fails on every node. Run this
# from a machine that does have general internet access and it fetches,
# verifies, and copies what bootstrap-node.sh expects to find beside itself.
# A node that can reach GitHub needs none of this.
#
#   ./stage-artifacts.sh <node>
#   ssh <node> 'sudo ~/flakegraph-stage/bootstrap-node.sh --role agent ...'
#
# The checksum is verified here rather than on the node: a binary that arrives
# corrupted should fail before it is installed as the cluster's control plane.
set -euo pipefail

target="${1:-}"
version="${FLAKEGRAPH_K3S_VERSION:-v1.36.3+k3s1}"
arch="${FLAKEGRAPH_K3S_ARCH:-arm64}"
remote_dir="${FLAKEGRAPH_REMOTE_DIR:-flakegraph-stage}"

if [[ -z "$target" ]]; then
  echo "usage: $0 <ssh-host> [--local-only]" >&2
  echo "  stages k3s $version ($arch) and bootstrap-node.sh onto the node" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# GitHub wants the "+" in a release tag percent-encoded in the download path.
encoded="${version/+/%2B}"
base="https://github.com/k3s-io/k3s/releases/download/$encoded"

echo "==> fetching k3s $version ($arch)"
curl -fsSL -o "$work/k3s" "$base/k3s-${arch}"
curl -fsSL -o "$work/sha256sum-${arch}.txt" "$base/sha256sum-${arch}.txt"
curl -fsSL -o "$work/install.sh" "https://get.k3s.io"

echo "==> verifying checksum"
expected="$(awk -v want="k3s-${arch}" '$2 == want || $2 == "./"want {print $1}' "$work/sha256sum-${arch}.txt" | head -1)"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$work/k3s" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "$work/k3s" | awk '{print $1}')"
fi
if [[ -z "$expected" || "$expected" != "$actual" ]]; then
  echo "checksum mismatch for k3s-${arch}" >&2
  echo "  expected: ${expected:-<not found in sha256sum file>}" >&2
  echo "  actual:   $actual" >&2
  exit 1
fi
echo "    $actual"

if [[ "${2:-}" == "--local-only" ]]; then
  cp "$work/k3s" "$work/install.sh" "$script_dir/"
  echo "==> staged into $script_dir"
  exit 0
fi

echo "==> copying to $target:$remote_dir/"
ssh "$target" "mkdir -p ~/$remote_dir"
scp -q "$work/k3s" "$work/install.sh" "$script_dir/bootstrap-node.sh" "$target:$remote_dir/"
ssh "$target" "chmod +x ~/$remote_dir/k3s ~/$remote_dir/bootstrap-node.sh"

cat <<DONE

==> staged on $target

    Bring the node up with one of:
      ssh $target 'sudo ~/$remote_dir/bootstrap-node.sh --role server'
      ssh $target 'sudo ~/$remote_dir/bootstrap-node.sh --role agent --server https://<server-ip>:6443 --token <token>'
DONE
