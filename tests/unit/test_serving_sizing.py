from __future__ import annotations

import pytest

from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    compute_sizing,
)

# The reference checkpoint: 4 KV heads, head_dim 256, 16 full-attention layers
# and 48 linear-attention layers, fp8 cache, 21.81 GiB of weights, on a 119.2 GiB
# unified-memory device that holds about 12 GiB of activations, graphs, and
# allocator slack alongside them. Prefix caching is on, so the engine reserves
# two recurrent-state slots per request.
REFERENCE_GEOMETRY = ModelGeometry(
    kv_heads=4,
    head_dim=256,
    attention_layers=16,
    kv_cache_dtype="fp8",
    weights_bytes=int(21.81 * BYTES_PER_GIB),
    recurrent_layers=48,
    recurrent_state_bytes_per_layer=3207168,
    recurrent_state_slots_per_sequence=2,
)
REFERENCE_DEVICE_BYTES = int(119.2 * BYTES_PER_GIB)
REFERENCE_OVERHEAD_BYTES = int(12.0 * BYTES_PER_GIB)


def _budget(utilization: float) -> DeviceBudget:
    return DeviceBudget(
        device_memory_bytes=REFERENCE_DEVICE_BYTES,
        gpu_memory_utilization=utilization,
        overhead_bytes=REFERENCE_OVERHEAD_BYTES,
    )


def test_kv_cost_per_token_covers_keys_values_and_every_layer() -> None:
    assert REFERENCE_GEOMETRY.kv_bytes_per_token() == 32 * 1024


def test_a_wider_cache_dtype_doubles_the_cost_per_token() -> None:
    wide = REFERENCE_GEOMETRY.model_copy(update={"kv_cache_dtype": "bf16"})

    assert wide.kv_bytes_per_token() == 2 * REFERENCE_GEOMETRY.kv_bytes_per_token()


@pytest.mark.parametrize(
    ("utilization", "kv_budget_gib", "concurrent_at_32k", "concurrent_at_8k"),
    [
        (0.50, 25.8, 20, 48),
        (0.60, 37.7, 29, 70),
        (0.75, 55.6, 43, 103),
    ],
)
def test_the_published_sizing_table_is_what_the_formula_produces(
    utilization: float,
    kv_budget_gib: float,
    concurrent_at_32k: int,
    concurrent_at_8k: int,
) -> None:
    budget = _budget(utilization)

    at_32k = compute_sizing(REFERENCE_GEOMETRY, budget, 32768, max_num_seqs=8)
    at_8k = compute_sizing(REFERENCE_GEOMETRY, budget, 8192, max_num_seqs=8)

    assert at_32k.kv_budget_bytes / BYTES_PER_GIB == pytest.approx(kv_budget_gib, abs=0.05)
    assert at_32k.max_concurrent == concurrent_at_32k
    assert at_8k.max_concurrent == concurrent_at_8k


def test_the_shipped_default_leaves_the_sequence_limit_binding_first() -> None:
    verdict = compute_sizing(REFERENCE_GEOMETRY, _budget(0.60), 32768, max_num_seqs=16)

    assert verdict.max_concurrent == 29
    assert verdict.sequence_limit_binds_first


def test_a_sequence_limit_at_or_above_capacity_is_reported_unsafe() -> None:
    budget = _budget(0.60)

    at_capacity = compute_sizing(REFERENCE_GEOMETRY, budget, 32768, max_num_seqs=29)
    beyond = compute_sizing(REFERENCE_GEOMETRY, budget, 32768, max_num_seqs=64)

    assert not at_capacity.sequence_limit_binds_first
    assert not beyond.sequence_limit_binds_first
    assert "evicting" in beyond.detail


def test_a_budget_smaller_than_the_weights_reports_no_cache_at_all() -> None:
    verdict = compute_sizing(REFERENCE_GEOMETRY, _budget(0.20), 32768, max_num_seqs=1)

    assert verdict.kv_cache_tokens == 0
    assert not verdict.sequence_limit_binds_first
    assert "cannot allocate a KV cache" in verdict.detail


def test_an_unknown_cache_dtype_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="unsupported kv_cache_dtype"):
        ModelGeometry(
            kv_heads=4,
            head_dim=256,
            attention_layers=16,
            kv_cache_dtype="nvfp4",
            weights_bytes=1,
        )


def test_nonsensical_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected_context_tokens must be positive"):
        compute_sizing(REFERENCE_GEOMETRY, _budget(0.60), 0, max_num_seqs=8)
    with pytest.raises(ValueError, match="max_num_seqs must be positive"):
        compute_sizing(REFERENCE_GEOMETRY, _budget(0.60), 32768, max_num_seqs=0)


def test_linear_attention_state_is_charged_per_sequence_not_per_token() -> None:
    """A hybrid model pays for its recurrent layers however short the context.

    The engine pages that state beside the KV cache, so leaving it out of the
    formula overstates concurrency by the same amount at every context length —
    which is exactly the error that blesses an unsafe sequence limit.
    """

    budget = _budget(0.50)
    without = REFERENCE_GEOMETRY.model_copy(update={"recurrent_layers": 0})

    assert REFERENCE_GEOMETRY.recurrent_state_bytes_per_sequence() == 48 * 3207168 * 2
    assert without.recurrent_state_bytes_per_sequence() == 0

    for context in (8192, 32768, 65536):
        charged = compute_sizing(REFERENCE_GEOMETRY, budget, context, max_num_seqs=4)
        ignored = compute_sizing(without, budget, context, max_num_seqs=4)
        assert charged.bytes_per_sequence > ignored.bytes_per_sequence
        assert charged.max_concurrent < ignored.max_concurrent


def test_the_reference_fleet_context_admits_eleven_sequences() -> None:
    """Pin the number the deployed fleet is sized against.

    At 65536 tokens the corrected formula allows eleven concurrent sequences, so
    a limit of eleven does *not* bind first — the comparison is strict.
    """

    verdict = compute_sizing(REFERENCE_GEOMETRY, _budget(0.50), 65536, max_num_seqs=8)

    assert verdict.max_concurrent == 11
    assert verdict.sequence_limit_binds_first
    assert not compute_sizing(
        REFERENCE_GEOMETRY, _budget(0.50), 65536, max_num_seqs=11
    ).sequence_limit_binds_first
