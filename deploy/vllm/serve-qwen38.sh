#!/usr/bin/env bash
# Launch the repository's single-GPU vLLM profile outside Kubernetes.
#
# The defaults mirror the Helm chart's own serving defaults (deploy/helm/
# flakegraph/values.yaml, modelServing.server): the same public bf16
# checkpoint at the same pinned revision, the same scheduler, batching and
# caching flags, so a bench-top check and a cluster run exercise the same
# engine configuration. Every value is an environment override, so the same
# launcher serves another model or another GPU without editing it; a
# unified-memory or Blackwell-specific profile (quantization, attention and
# MoE kernels, a drafter) lives in deploy/examples/dgx-spark-k3s-values.yaml
# and is expressed here through the VLLM_EXTRA_ARGS below.
set -euo pipefail

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed or is not on PATH" >&2
  exit 127
fi

model="${VLLM_MODEL:-Qwen/Qwen3-8B}"
revision="${VLLM_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
host="${VLLM_HOST:-0.0.0.0}"
port="${VLLM_PORT:-8000}"
# The engine's own default share of a discrete GPU. On a unified-memory part
# this competes with the operating system rather than drawing from a separate
# pool and has to drop to what the machine can spare (about half in the DGX
# Spark profile); set it lower there.
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
max_model_len="${VLLM_MAX_MODEL_LEN:-32768}"
max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
# Must stay below the max_concurrent the KV budget allows. Check a change with:
#   flakegraph serving sizing --kv-heads 8 --head-dim 128 --attention-layers 36 \
#     --kv-cache-dtype auto --weights-gib 15.3 --device-memory-gib 80 \
#     --max-num-seqs <value>
max_num_seqs="${VLLM_MAX_NUM_SEQS:-16}"
# How the engine reads the checkpoint's chat template; both belong to the
# model. Set to an empty string to omit the flag (unset takes the default).
reasoning_parser="${VLLM_REASONING_PARSER-qwen3}"
tool_call_parser="${VLLM_TOOL_CALL_PARSER-hermes}"
# Further engine flags for a hardware-specific profile, e.g.
#   VLLM_EXTRA_ARGS="--quantization compressed-tensors --kv-cache-dtype fp8"
extra_args="${VLLM_EXTRA_ARGS:-}"

# The engine reports anonymous usage statistics to its maintainers by default;
# a bench-top run is nobody's to report. Set VLLM_USAGE_STATS=1 to allow it.
if [[ "${VLLM_USAGE_STATS:-0}" != "1" ]]; then
  export VLLM_NO_USAGE_STATS=1
  export DO_NOT_TRACK=1
fi
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# Speculative decoding is deliberately absent: the right drafter and block
# size need a benchmark on the hardware in hand, and the DGX Spark example
# values carry a benchmarked choice for theirs. It does not conflict
# with --async-scheduling -- vLLM keeps async scheduling on for every
# Eagle-family method -- so the chart runs both.
args=(
  serve "$model"
  --served-model-name "$model"
  --revision "$revision"
  --host "$host"
  --port "$port"
  # Orders the waiting queue on (priority, arrival); without it every stamped
  # priority is silently ignored.
  --scheduling-policy priority
  --tensor-parallel-size 1
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-model-len "$max_model_len"
  --max-num-seqs "$max_num_seqs"
  --max-num-batched-tokens "$max_num_batched_tokens"
  --enable-chunked-prefill
  --async-scheduling
  --enable-prefix-caching
)
if [[ -n "$reasoning_parser" ]]; then
  args+=(--reasoning-parser "$reasoning_parser")
fi
if [[ -n "$tool_call_parser" ]]; then
  args+=(--tool-call-parser "$tool_call_parser" --enable-auto-tool-choice)
fi
if [[ -n "$extra_args" ]]; then
  # shellcheck disable=SC2206
  args+=($extra_args)
fi
exec vllm "${args[@]}"
