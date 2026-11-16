# SPDX-License-Identifier: Apache-2.0
"""Offline shell contracts; no installer, service, or cluster command is executed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_BOOTSTRAP = Path("deploy/spark/bootstrap-node.sh")
_MAINTENANCE = Path("deploy/spark/node-maintenance.sh")


@pytest.mark.parametrize("role", ["server", "agent"])
@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_bootstrap_installer_receives_the_requested_role_environment(
    role: str,
    staged: bool,
    inherited: bool,
) -> None:
    """Only the extracted role pipeline runs, with synthetic source from curl/cat mocks."""

    source = _BOOTSTRAP.read_text(encoding="utf-8")
    fetch = next(line.strip() for line in source.splitlines() if line.strip().startswith("fetch()"))
    start = source.index('  if [[ "$role" == "server" ]]; then', source.index(fetch))
    branch = source[start : source.index("\n  fi", start) + len("\n  fi")]
    printer = r"""
printf 'exec=%s\nurl=%s\ntoken=%s\nchannel=%s\n' \
  "$INSTALL_K3S_EXEC" "${K3S_URL-unset}" "${K3S_TOKEN-unset}" "$INSTALL_K3S_CHANNEL"
"""
    environment = {"PATH": "/usr/bin:/bin", "SYNTHETIC_SOURCE": printer}
    if inherited:
        environment.update(
            INSTALL_K3S_EXEC="stale-command", K3S_URL="stale-url", K3S_TOKEN="stale-token"
        )
    preamble = r"""
set -euo pipefail
role="$1"
installer="$2"
server_url=https://synthetic.invalid:6443
join_token=synthetic-token
primary_ip=192.0.2.1
node_class=synthetic-class
export INSTALL_K3S_CHANNEL=synthetic-channel
ok() { :; }
hostname() { printf 'synthetic.invalid\n'; }
curl() { printf '%s\n' "$SYNTHETIC_SOURCE"; }
cat() { printf '%s\n' "$SYNTHETIC_SOURCE"; }
"""
    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            preamble + fetch + "\n" + branch,
            "offline-test",
            role,
            "/synthetic/install.sh" if staged else "https://synthetic.invalid/install",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert values["channel"] == "synthetic-channel"
    if role == "agent":
        assert values["exec"] == (
            "agent --node-label flakegraph.io/node-class=synthetic-class "
            "--node-label nvidia.com/gpu.present=true"
        )
        assert values["url"] == "https://synthetic.invalid:6443"
        assert values["token"] == "synthetic-token"
    else:
        assert values["exec"] == (
            "server --cluster-init --tls-san 192.0.2.1 --tls-san synthetic.invalid "
            "--write-kubeconfig-mode 0600"
        )


_READY = "engine|model-serving|2|2|3|3\nparser|document-parsing|2|2|3|3"


@pytest.mark.parametrize(
    "pod_rows,statefulsets,recovers,api_fails",
    [
        ("", _READY, True, True),
        ("", _READY, False, True),
        ("", _READY, False, False),
        ("pending-without-status", _READY, False, False),
        ("pending-without-status", _READY, True, False),
        ("", "parser|document-parsing|2|2|3|3", False, False),
        ("", "", False, False),
        ("", "engine|model-serving|2|1|3|3\nparser|document-parsing|2|2|3|3", False, False),
        (
            "",
            "engine|model-serving|2|<no value>|3|3\nparser|document-parsing|2|2|3|3",
            False,
            False,
        ),
        (
            "",
            "engine|model-serving|<no value>|2|3|3\nparser|document-parsing|2|2|3|3",
            False,
            False,
        ),
        ("", "engine|model-serving|2|2|3|2\nparser|document-parsing|2|2|3|3", False, False),
        ("", "engine|model-serving|2|1|3|3\nparser|document-parsing|2|2|3|3", True, False),
    ],
)
def test_restore_waits_for_missing_pod_conditions_and_expected_serving_replicas(
    pod_rows: str,
    statefulsets: str,
    recovers: bool,
    api_fails: bool,
) -> None:
    """The extracted wait uses mocked kubectl rows and a clock that advances without sleeping."""

    source = _MAINTENANCE.read_text(encoding="utf-8")
    start = source.index("    deadline=")
    wait = source[
        start : source.index(
            '    kubectl get pods -n "$namespace" --field-selector "spec.nodeName=$node" -o wide',
            start,
        )
    ]
    preamble = r"""
set -euo pipefail
namespace=synthetic
node=synthetic-node
restore_timeout=2s
clock=0
waits=0
seconds() { printf '2\n'; }
date() { printf '%s\n' "$clock"; }
ok() { printf '%s\n' "$1"; }
sleep() { waits=$(( waits + 1 )); clock=3; printf 'waited\n'; }
kubectl() {
  [[ "$1" == get ]] || return 99
  # An unanswered request is retried until the deadline, never fatal at once.
  [[ "$API_FAILS" == yes && ( "$RECOVERS" != yes || "$waits" -eq 0 ) ]] && return 1
  if [[ "$2" == pods ]]; then
    # Pod Ready is required even if containerStatuses is absent or empty.
    [[ "$*" == *'.status.conditions'* && "$*" == *'(eq .status "True")'* ]] || return 98
    [[ "$RECOVERS" == yes && "$waits" -gt 0 ]] || printf '%s\n' "$POD_ROWS"
  elif [[ "$2" == statefulsets ]]; then
    if [[ "$RECOVERS" == yes && "$waits" -gt 0 ]]; then
      printf '%s\n' "$READY_ROWS"
    else
      printf '%s\n' "$STATEFULSET_ROWS"
    fi
  else
    return 97
  fi
  return 0
}
"""
    result = subprocess.run(
        ["/bin/bash", "-c", preamble + wait],
        env={
            "PATH": "/usr/bin:/bin",
            "POD_ROWS": pod_rows,
            "STATEFULSET_ROWS": statefulsets,
            "READY_ROWS": _READY,
            "RECOVERS": "yes" if recovers else "no",
            "API_FAILS": "yes" if api_fails else "no",
        },
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    immediate = not pod_rows and statefulsets == _READY and not api_fails
    assert result.returncode == (0 if immediate or recovers else 1), result.stderr
    assert ("waited" in result.stdout) is (not immediate)
    assert ("all engine/parser replicas are Ready" in result.stdout) is (immediate or recovers)
    if not immediate and not recovers:
        assert "still not ready after 2s" in result.stderr
