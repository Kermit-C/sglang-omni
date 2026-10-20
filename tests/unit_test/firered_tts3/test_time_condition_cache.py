# SPDX-License-Identifier: Apache-2.0
"""CPU checks for time-table ownership and release of cached tensor storage."""

from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest
import torch

from sglang_omni.models.firered_tts3.compiled_dit import (
    GemmSignature,
    TimeCacheUnavailable,
    TimeConditionCache,
    TimeGemmTable,
    TimeGemmValue,
)
from sglang_omni.models.firered_tts3.dit import DiT


@pytest.fixture
def time_cache() -> TimeConditionCache:
    return TimeConditionCache(
        DiT(in_channels=4, out_channels=4, depth=1, num_heads=2, hidden_size=8)
    )


def make_table(cache: TimeConditionCache, batch_size: int = 1) -> TimeGemmTable:
    output = torch.ones(2 * batch_size, 8)
    signature = GemmSignature(
        input_shape=(2 * batch_size, 256),
        input_stride=(256, 1),
        weight_shape=(256, 8),
        weight_stride=(1, 256),
        bias_shape=(8,),
        bias_stride=(1,),
        output_shape=tuple(output.shape),
        output_stride=tuple(output.stride()),
        dtype=output.dtype,
        device=output.device,
    )
    return TimeGemmTable(
        model_id=id(cache.source),
        n_timesteps=1,
        steps=(
            MappingProxyType(
                {
                    "t_embedder.time_mlp.0": TimeGemmValue(
                        signature=signature, output=output
                    )
                }
            ),
        ),
    )


def test_time_cache_releases_unowned_table_and_tensor(
    time_cache: TimeConditionCache,
) -> None:
    table = make_table(time_cache)
    table_ref = weakref.ref(table)
    output_ref = weakref.ref(table.steps[0]["t_embedder.time_mlp.0"].output)

    del table
    gc.collect()

    assert table_ref() is None and output_ref() is None


def test_time_modes_retain_table_until_last_owner_releases(
    time_cache: TimeConditionCache,
) -> None:
    table = make_table(time_cache)
    table_ref = weakref.ref(table)
    output_ref = weakref.ref(table.steps[0]["t_embedder.time_mlp.0"].output)
    first = time_cache.mode(table)
    second = time_cache.mode(table)

    del table, first
    gc.collect()

    assert table_ref() is second.table and output_ref() is not None

    del second
    gc.collect()

    assert table_ref() is None and output_ref() is None


def test_time_table_release_does_not_discard_other_buckets(
    time_cache: TimeConditionCache,
) -> None:
    first = make_table(time_cache, batch_size=1)
    second = make_table(time_cache, batch_size=2)
    first_ref = weakref.ref(first)
    first_output = weakref.ref(first.steps[0]["t_embedder.time_mlp.0"].output)
    first_mode = time_cache.mode(first)
    second_mode = time_cache.mode(second)

    del first, first_mode
    gc.collect()

    assert first_ref() is None and first_output() is None
    assert second_mode.table is second


def test_time_mode_rejects_table_from_another_model(
    time_cache: TimeConditionCache,
) -> None:
    other_cache = TimeConditionCache(
        DiT(in_channels=4, out_channels=4, depth=1, num_heads=2, hidden_size=8)
    )
    table = make_table(other_cache)

    with pytest.raises(TimeCacheUnavailable, match="different model weights"):
        time_cache.mode(table)
