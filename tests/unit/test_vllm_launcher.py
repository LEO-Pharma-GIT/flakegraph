"""Protect the concise quick-start launcher and keep it mirroring the chart."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

_LAUNCHER = Path("deploy/vllm/serve-qwen38.sh")
_VALUES = Path("deploy/helm/flakegraph/values.yaml")


def _launch(tmp_path: Path, **overrides: str) -> tuple[list[str], dict[str, str]]:
    """Run the launcher against a fake vllm and return its arguments and environment."""

    arguments_path = tmp_path / "arguments.txt"
    environment_path = tmp_path / "environment.txt"
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENTS_PATH"\n'
        "env | grep -E '^(VLLM_NO_USAGE_STATS|DO_NOT_TRACK|HF_HUB_DISABLE_TELEMETRY|"
        'VLLM_MARLIN_USE_ATOMIC_ADD)=\' > "$ENVIRONMENT_PATH" || true\n',
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "ARGUMENTS_PATH": str(arguments_path),
        "ENVIRONMENT_PATH": str(environment_path),
        **overrides,
    }
    subprocess.run(
        ["bash", str(_LAUNCHER)], check=True, env=environment, capture_output=True, text=True
    )
    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    exported = dict(
        line.split("=", 1) for line in environment_path.read_text(encoding="utf-8").splitlines()
    )
    return arguments, exported


def test_launcher_mirrors_the_charts_serving_defaults(tmp_path: Path) -> None:
    """The bench-top profile and the chart's serve the same engine configuration.

    The launcher claims to mirror the chart, so its defaults are read from
    values.yaml rather than repeated here: a change to one that forgets the
    other fails this test instead of drifting quietly.
    """

    serving = yaml.safe_load(_VALUES.read_text(encoding="utf-8"))["modelServing"]
    arguments, exported = _launch(tmp_path, VLLM_PORT="9000", VLLM_MAX_NUM_SEQS="2")

    assert arguments[:2] == ["serve", serving["model"]["name"]]
    assert _option(arguments, "--revision") == serving["model"]["revision"]
    assert _option(arguments, "--port") == "9000"
    assert _option(arguments, "--max-num-seqs") == "2"
    assert _option(arguments, "--max-num-batched-tokens") == str(
        serving["server"]["maxNumBatchedTokens"]
    )
    assert _option(arguments, "--max-model-len") == str(serving["server"]["maxModelLen"])
    assert (
        float(_option(arguments, "--gpu-memory-utilization"))
        == (serving["server"]["gpuMemoryUtilization"])
    )
    assert _option(arguments, "--reasoning-parser") == serving["server"]["reasoningParser"]
    assert _option(arguments, "--tool-call-parser") == serving["server"]["toolCallParser"]
    assert "--enable-auto-tool-choice" in arguments
    assert "--enable-prefix-caching" in arguments
    # Without this the engine serves FIFO and every stamped band is ignored.
    assert _option(arguments, "--scheduling-policy") == "priority"
    # The quick start stays minimal and portable: no drafter, no remote code,
    # no quantization or kernel a build might lack.
    assert "--speculative-config" not in arguments
    assert "--trust-remote-code" not in arguments
    for flag in ("--quantization", "--attention-backend", "--moe-backend", "--kv-cache-dtype"):
        assert flag not in arguments, flag
    # Nothing phones home.
    assert exported["VLLM_NO_USAGE_STATS"] == "1"
    assert exported["DO_NOT_TRACK"] == "1"
    assert exported["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert "VLLM_MARLIN_USE_ATOMIC_ADD" not in exported


def test_launcher_takes_a_hardware_profile_through_the_environment(tmp_path: Path) -> None:
    """A unified-memory or Blackwell profile is expressed without editing the script."""

    arguments, exported = _launch(
        tmp_path,
        VLLM_MODEL="unsloth/Qwen3.8-27B-NVFP4",
        VLLM_MODEL_REVISION="9e3d73c76eddb75f795cc24ccfbc5affe41c66bd",
        VLLM_GPU_MEMORY_UTILIZATION="0.50",
        VLLM_TOOL_CALL_PARSER="qwen3_xml",
        VLLM_REASONING_PARSER="",
        VLLM_EXTRA_ARGS=(
            "--quantization compressed-tensors --kv-cache-dtype fp8 --trust-remote-code"
        ),
        VLLM_USAGE_STATS="1",
        VLLM_MARLIN_USE_ATOMIC_ADD="1",
    )

    assert arguments[:2] == ["serve", "unsloth/Qwen3.8-27B-NVFP4"]
    assert _option(arguments, "--gpu-memory-utilization") == "0.50"
    assert _option(arguments, "--tool-call-parser") == "qwen3_xml"
    assert "--reasoning-parser" not in arguments
    assert _option(arguments, "--quantization") == "compressed-tensors"
    assert _option(arguments, "--kv-cache-dtype") == "fp8"
    assert "--trust-remote-code" in arguments
    assert "VLLM_NO_USAGE_STATS" not in exported
    assert exported["VLLM_MARLIN_USE_ATOMIC_ADD"] == "1"


def _option(arguments: list[str], name: str) -> str:
    """Return the value immediately following a required launcher option."""

    index = arguments.index(name)
    return arguments[index + 1]
