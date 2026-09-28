# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contract tests for the opt-in Qwen3.8 Stage-2 one-step gate."""

from types import SimpleNamespace

import pytest
import torch

from scripts.tools.qwen38_stage2_step_probe import _inner_step_parameter, _valid_observations, _validate_probe_step


def test_stage2_probe_rejects_second_step_within_a_rollout():
    args = SimpleNamespace(num_rollout=2)
    _validate_probe_step(args, 0, 0)
    _validate_probe_step(args, 1, 0)
    with pytest.raises(RuntimeError, match="one step in each"):
        _validate_probe_step(args, 0, 1)
    with pytest.raises(RuntimeError, match="one step in each"):
        _validate_probe_step(args, 2, 0)


def test_precision_aware_indexer_maps_to_actual_fp32_step_tensor():
    model_parameter = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
    bf16_shard = model_parameter.detach().view(-1)
    fp32_inner = torch.zeros(4, dtype=torch.float32)
    optimizer = SimpleNamespace(
        model_float16_groups=[[model_parameter]],
        shard_float16_groups=[[bf16_shard]],
        shard_fp32_from_float16_groups=[[None]],
        optimizer=SimpleNamespace(param_to_inner_param={bf16_shard: fp32_inner}),
    )
    assert _inner_step_parameter(optimizer, model_parameter) is fp32_inner
    assert _inner_step_parameter(optimizer, torch.nn.Parameter(torch.zeros(1))) is None

    optimizer.optimizer.param_to_inner_param.clear()
    with pytest.raises(RuntimeError, match="no mapped optimizer step tensor"):
        _inner_step_parameter(optimizer, model_parameter)


def test_fp32_indexer_maps_local_shard_through_hybrid_optimizer():
    model_parameter = torch.nn.Parameter(torch.zeros(2, 2))
    fp32_shard = model_parameter.detach().view(-1)
    fp32_inner = torch.ones(4)
    optimizer = SimpleNamespace(
        model_float16_groups=[],
        shard_float16_groups=[],
        shard_fp32_from_float16_groups=[],
        model_fp32_groups=[[model_parameter]],
        shard_fp32_groups=[[fp32_shard]],
        optimizer=SimpleNamespace(param_to_inner_param={fp32_shard: fp32_inner}),
    )
    assert _inner_step_parameter(optimizer, model_parameter) is fp32_inner


def test_stage2_observation_rejects_packed_short_documents_and_dense_route():
    long = {
        "layer": 3,
        "physical_tokens": 2304,
        "physical_doc_lengths": (2304,),
        "valid_doc_lengths": (2200,),
        "all_selected": False,
    }
    other_layer = {**long, "layer": 7}
    assert _valid_observations([long, other_layer], {3, 7})
    assert not _valid_observations([long], {3, 7})
    assert not _valid_observations([{**long, "all_selected": True}, other_layer], {3, 7})
    short_pack = {
        **long,
        "physical_tokens": 2560,
        "physical_doc_lengths": (1280, 1280),
        "valid_doc_lengths": (1200, 1200),
    }
    assert not _valid_observations([short_pack, other_layer], {3, 7})
    assert not _valid_observations([{**long, "physical_tokens": 4608}, other_layer], {3, 7})
