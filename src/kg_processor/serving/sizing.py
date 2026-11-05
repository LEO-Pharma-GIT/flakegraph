# SPDX-License-Identifier: Apache-2.0
"""Compute how many sequences an engine can run before KV memory binds.

``max_num_seqs`` cannot be a product constant: it depends on the model, its
quantisation, the GPU, and the context an operator expects. This module ships the
arithmetic instead, so a deployment can be checked rather than guessed.

The rule the whole QoS model rests on is ``max_num_seqs < max_concurrent``. vLLM
preempts a *running* request when it cannot allocate KV blocks, and it evicts the
lowest-priority victim — batch work, mid-flight, against policy. Sizing so the
sequence limit binds first keeps that eviction path cold.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

BYTES_PER_GIB = 1024**3

# vLLM spells its KV cache dtype with these names; the widths are what the cache
# actually costs per element.
KV_CACHE_DTYPE_BYTES: dict[str, int] = {
    "auto": 2,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
}


class ModelGeometry(BaseModel):
    """Describe the attention shape that determines KV cost per token.

    These are properties of a checkpoint, not of a cluster, so they are declared
    rather than discovered. ``kv_heads`` is the number of key/value heads after
    grouped-query attention, which is what the cache is sized on — not the number
    of query heads.

    A hybrid checkpoint interleaves full attention with linear attention, and the
    linear-attention layers are not free: each holds a fixed recurrent state per
    sequence rather than a cache that grows per token. vLLM pages that state
    alongside the KV cache and says so at startup — "Setting attention block size
    to N tokens to ensure that attention page size is >= mamba page size". Leave
    ``recurrent_layers`` at zero for a pure-attention model. Prefix caching makes
    the engine hold two slots per request instead of one, which is what
    ``recurrent_state_slots_per_sequence`` carries.
    """

    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    attention_layers: int = Field(gt=0)
    kv_cache_dtype: str = "fp8"
    weights_bytes: int = Field(gt=0)
    recurrent_layers: int = Field(default=0, ge=0)
    recurrent_state_bytes_per_layer: int = Field(default=0, ge=0)
    recurrent_state_slots_per_sequence: int = Field(default=1, ge=1)

    @field_validator("kv_cache_dtype")
    @classmethod
    def kv_cache_dtype_must_be_known(cls, value: str) -> str:
        """Reject a dtype whose width the formula cannot account for."""

        if value not in KV_CACHE_DTYPE_BYTES:
            supported = ", ".join(sorted(KV_CACHE_DTYPE_BYTES))
            raise ValueError(f"unsupported kv_cache_dtype '{value}'; supported: {supported}")
        return value

    def kv_bytes_per_token(self) -> int:
        """Return the cache cost of one token across every attention layer.

        The leading factor of two covers the separate key and value tensors.
        """

        return (
            2
            * self.kv_heads
            * self.head_dim
            * self.attention_layers
            * KV_CACHE_DTYPE_BYTES[self.kv_cache_dtype]
        )

    def recurrent_state_bytes_per_sequence(self) -> int:
        """Return the fixed per-sequence cost of the linear-attention layers.

        This does not scale with context, so it is charged once per sequence
        rather than per token. It is zero for a model without such layers.
        """

        return (
            self.recurrent_layers
            * self.recurrent_state_bytes_per_layer
            * self.recurrent_state_slots_per_sequence
        )

    def bytes_per_sequence(self, expected_context_tokens: int) -> int:
        """Return everything one sequence costs the cache at a given context."""

        return (
            expected_context_tokens * self.kv_bytes_per_token()
            + self.recurrent_state_bytes_per_sequence()
        )


class DeviceBudget(BaseModel):
    """Describe the memory an engine may spend and what it loses before the cache.

    ``overhead_bytes`` covers activations, CUDA graphs, the allocator's own
    fragmentation, and anything else resident alongside the weights. On unified
    memory parts it also has to cover what the operating system and any
    co-located process hold, which is why it is an explicit operator input.
    """

    device_memory_bytes: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0.0, le=1.0)
    overhead_bytes: int = Field(ge=0)


class SizingVerdict(BaseModel):
    """Report the computed limits and whether the configured limit is safe."""

    kv_bytes_per_token: int
    recurrent_state_bytes_per_sequence: int
    bytes_per_sequence: int
    kv_budget_bytes: int
    kv_cache_tokens: int
    expected_context_tokens: int
    max_concurrent: int
    max_num_seqs: int
    sequence_limit_binds_first: bool
    detail: str


def compute_sizing(
    geometry: ModelGeometry,
    budget: DeviceBudget,
    expected_context_tokens: int,
    max_num_seqs: int,
) -> SizingVerdict:
    """Evaluate a serving configuration against the KV budget it actually has.

    A verdict is returned rather than an exception so that callers can decide
    whether an unsafe configuration is fatal — the sidecar refuses to start, the
    CLI prints and exits non-zero.
    """

    if expected_context_tokens <= 0:
        raise ValueError("expected_context_tokens must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")

    kv_bytes_per_token = geometry.kv_bytes_per_token()
    recurrent_bytes = geometry.recurrent_state_bytes_per_sequence()
    bytes_per_sequence = geometry.bytes_per_sequence(expected_context_tokens)
    kv_budget_bytes = (
        int(budget.device_memory_bytes * budget.gpu_memory_utilization)
        - geometry.weights_bytes
        - budget.overhead_bytes
    )
    if kv_budget_bytes <= 0:
        return SizingVerdict(
            kv_bytes_per_token=kv_bytes_per_token,
            recurrent_state_bytes_per_sequence=recurrent_bytes,
            bytes_per_sequence=bytes_per_sequence,
            kv_budget_bytes=kv_budget_bytes,
            kv_cache_tokens=0,
            expected_context_tokens=expected_context_tokens,
            max_concurrent=0,
            max_num_seqs=max_num_seqs,
            sequence_limit_binds_first=False,
            detail=(
                "weights and overhead exceed the memory this utilization allows; "
                "the engine cannot allocate a KV cache at all"
            ),
        )

    kv_cache_tokens = kv_budget_bytes // kv_bytes_per_token
    # Concurrency is bounded by everything a sequence costs, not by tokens
    # alone: a hybrid model's linear-attention state is charged per sequence and
    # does not shrink with a shorter context.
    max_concurrent = kv_budget_bytes // bytes_per_sequence
    safe = max_num_seqs < max_concurrent
    if safe:
        detail = (
            f"max_num_seqs={max_num_seqs} binds before KV memory "
            f"(max_concurrent={max_concurrent} at {expected_context_tokens} tokens)"
        )
    else:
        detail = (
            f"max_num_seqs={max_num_seqs} does not bind before KV memory "
            f"(max_concurrent={max_concurrent} at {expected_context_tokens} tokens); "
            "the engine will preempt running requests under KV pressure, evicting "
            "the lowest-priority work mid-flight"
        )
    return SizingVerdict(
        kv_bytes_per_token=kv_bytes_per_token,
        recurrent_state_bytes_per_sequence=recurrent_bytes,
        bytes_per_sequence=bytes_per_sequence,
        kv_budget_bytes=kv_budget_bytes,
        kv_cache_tokens=kv_cache_tokens,
        expected_context_tokens=expected_context_tokens,
        max_concurrent=max_concurrent,
        max_num_seqs=max_num_seqs,
        sequence_limit_binds_first=safe,
        detail=detail,
    )
