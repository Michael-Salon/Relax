import importlib
import sys
import types
from argparse import Namespace

import pytest
import torch


def _load_data_module(monkeypatch):
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    mpu = types.ModuleType("megatron.core.mpu")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    training = types.ModuleType("megatron.training")
    global_vars = types.ModuleType("megatron.training.global_vars")
    tracking_utils = types.ModuleType("relax.utils.tracking_utils")

    class _PackedSeqParams:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    core.mpu = mpu
    packed_seq_params.PackedSeqParams = _PackedSeqParams
    global_vars.get_args = lambda: None

    modules = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.mpu": mpu,
        "megatron.core.packed_seq_params": packed_seq_params,
        "megatron.training": training,
        "megatron.training.global_vars": global_vars,
        "relax.utils.tracking_utils": tracking_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("relax.backends.megatron.data", None)
    return importlib.import_module("relax.backends.megatron.data")


def test_vpp_microbatch_rounding_uses_ceil_multiple(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    rounded = data_module._round_up_to_microbatch_group(torch.tensor([1, 2, 3, 5]), microbatch_group_size=4)

    assert rounded.tolist() == [4, 4, 4, 8]


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("loss_coeff", [None, 0.0, 0.01])
def test_vl_packed_stage2_preserves_true_lengths_without_changing_cu(
    monkeypatch, cp_size: int, loss_coeff: float | None
):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=loss_coeff))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)

    def slice_with_cp(tokens, pad_value, *_args, **_kwargs):
        if cp_size == 1:
            return tokens
        chunk = (tokens.size(0) + 3) // 4
        padded = torch.nn.functional.pad(tokens, (0, 4 * chunk - tokens.size(0)), value=pad_value)
        return torch.cat((padded[:chunk], padded[3 * chunk :]))

    monkeypatch.setattr(data_module, "slice_with_cp", slice_with_cp)
    batch = {
        "tokens": [torch.arange(5), torch.arange(2)],
        "loss_masks": [torch.ones(5, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "total_lengths": [5, 2],
        "response_lengths": [5, 2],
    }
    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)
    params = packed["vlm_packed_seq_params"] if cp_size == 2 else packed["packed_seq_params"]

    if cp_size == 2:
        assert params.cu_seqlens_q.tolist() == [0, 8, 16]
        assert params.cu_seqlens_q_padded.tolist() == [0, 8, 16]
        expected_lengths = [5, 2]
        expected_starts = (0, 8, 16)
    else:
        assert params.cu_seqlens_q.tolist() == [0, 5, 7, 8]
        expected_lengths = [5, 2, 0]
        expected_starts = (0, 5, 7, 8)
    if loss_coeff is not None and loss_coeff > 0:
        assert params.qsa_stage2_valid_lengths.tolist() == expected_lengths
        assert params.qsa_stage2_layout_cpu == (expected_starts, tuple(expected_lengths))
    else:
        assert not hasattr(params, "qsa_stage2_valid_lengths")
        assert not hasattr(params, "qsa_stage2_layout_cpu")


def test_vl_cp_packing_keeps_two_image_samples_and_loss_masks_separate(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=0.0))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)

    def slice_with_cp(values, pad_value, *_args, **_kwargs):
        chunk = (values.size(0) + 3) // 4
        padded = torch.nn.functional.pad(values, (0, 4 * chunk - values.size(0)), value=pad_value)
        return torch.cat((padded[:chunk], padded[3 * chunk :]))

    monkeypatch.setattr(data_module, "slice_with_cp", slice_with_cp)
    lengths = [92, 99]
    tokens = [torch.full((length,), 10 + i, dtype=torch.long) for i, length in enumerate(lengths)]
    loss_masks = [torch.arange(length) >= length - (i + 2) for i, length in enumerate(lengths)]
    batch = {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": lengths,
        "response_lengths": lengths,
        "multimodal_train_inputs": [
            {
                "pixel_values": torch.full((4, 3), 1.0),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            },
            {
                "pixel_values": torch.full((9, 3), 2.0),
                "image_grid_thw": torch.tensor([[1, 3, 3]]),
            },
        ],
    }

    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, is_vl_model=True)

    assert packed["padded_total_lengths"] == [96, 104]
    assert packed["vlm_packed_seq_params"].cu_seqlens_q.tolist() == [0, 96, 200]
    assert packed["unsplit_attention_mask"].sum(dim=1).tolist() == lengths
    assert packed["multimodal_num_items"]["pixel_values"] == [4, 9]
    assert packed["multimodal_num_items"]["image_grid_thw"] == [1, 1]
    assert packed["multimodal_train_inputs"]["image_grid_thw"].tolist() == [[1, 2, 2], [1, 3, 3]]
    assert packed["multimodal_train_inputs"]["pixel_values"][:4].eq(1).all()
    assert packed["multimodal_train_inputs"]["pixel_values"][4:].eq(2).all()
    assert packed["unsplit_tokens"][0, : lengths[0]].eq(10).all()
    assert packed["unsplit_tokens"][1, : lengths[1]].eq(11).all()
    assert packed["full_loss_masks"].sum().item() == sum(i + 2 for i in range(2))


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("loss_coeff", [0.0, 0.01])
def test_allgather_packed_stage2_layout_keeps_last_document_padding(monkeypatch, cp_size: int, loss_coeff: float):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(data_module, "get_args", lambda: Namespace(qsa_indexer_loss_coeff=loss_coeff))
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 2, raising=False)
    batch = {
        "tokens": [torch.arange(5), torch.arange(2)],
        "loss_masks": [torch.ones(5, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "total_lengths": [5, 2],
        "response_lengths": [5, 2],
    }

    if loss_coeff > 0 and cp_size > 1:
        with pytest.raises(NotImplementedError, match="per-document zigzag"):
            data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, allgather_cp=True)
        return

    packed = data_module.get_batch(iter([(batch, None)]), ["tokens"], pad_multiplier=4, allgather_cp=True)
    params = packed["packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 5, 7]
    assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8 * cp_size]
    if loss_coeff > 0:
        assert params.qsa_stage2_layout_cpu == ((0, 5, 8), (5, 2))
    else:
        assert not hasattr(params, "qsa_stage2_layout_cpu")


def test_rollout_minibatch_plan_derives_from_global_batch(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=None,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 2
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_prefers_explicit_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    plan = data_module.build_rollout_minibatch_plan(args, dp_size=2)

    assert plan.num_rollout_minis == 3
    assert plan.mini_rollout_batch_size == 4
    assert plan.mini_global_samples == 32
    assert plan.mini_local_sample_request == 16


def test_rollout_minibatch_plan_rejects_non_divisible_prompt_groups(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    args = Namespace(
        rollout_batch_size=9,
        n_samples_per_prompt=8,
        global_batch_size=None,
        num_steps_per_rollout=3,
    )

    with pytest.raises(ValueError, match="mini_rollout_batch_size must be divisible"):
        data_module.build_rollout_minibatch_plan(args, dp_size=2)


def test_log_rollout_data_creates_collective_stats_on_training_device(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    loss_masks = [torch.tensor([0, 1]), torch.tensor([0, 1, 1])]
    requested_devices = []
    original_tensor = torch.tensor

    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_group",
        lambda with_context_parallel=True: object(),
        raising=False,
    )
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: "training-device")

    def capture_tensor(*args, **kwargs):
        requested_devices.append(kwargs.get("device"))
        kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(data_module.torch, "tensor", capture_tensor)
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "gather_log_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(data_module, "maybe_padded_total_lengths", lambda *_args, **_kwargs: None)

    args = Namespace(
        qkv_format="thd",
        is_vl_model=False,
        uses_unsplit_forward=False,
        dynamic_context_parallel=False,
        use_opd=False,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        ci_test=False,
        log_multi_turn=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "total_lengths": [4, 6],
        "response_lengths": [2, 3],
        "loss_masks": loss_masks,
    }

    data_module.log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert requested_devices == ["training-device"]


def test_concat_rollout_batches_preserves_order_and_scalar_metadata(monkeypatch):
    data_module = _load_data_module(monkeypatch)

    merged = data_module.concat_rollout_batches(
        [
            {
                "tokens": ["a", "b"],
                "total_lengths": [1, 2],
                "scores": torch.tensor([[1], [2]]),
                "weight_version": 7,
            },
            {
                "tokens": ["c"],
                "total_lengths": [3],
                "scores": torch.tensor([[3]]),
                "weight_version": 7,
            },
        ]
    )

    assert merged["tokens"] == ["a", "b", "c"]
    assert merged["total_lengths"] == [1, 2, 3]
    assert torch.equal(merged["scores"], torch.tensor([[1], [2], [3]]))
    assert merged["weight_version"] == 7


def test_get_data_iterator_uses_rollout_mini_boundaries_with_balance_data(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=32,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {
        "total_lengths": list(range(32)),
        data_module.ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY: [16, 16],
    }

    data_iterators, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [4, 4]
    iterator = data_iterators[0]
    first_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    second_step = [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(4)]
    assert first_step[0] == [0, 1, 2, 3]
    assert first_step[-1] == [12, 13, 14, 15]
    assert second_step[0] == [16, 17, 18, 19]
    assert second_step[-1] == [28, 29, 30, 31]


def test_get_data_iterator_balance_data_without_boundaries_uses_regular_steps(monkeypatch):
    data_module = _load_data_module(monkeypatch)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 2,
        raising=False,
    )
    monkeypatch.setattr(data_module.mpu, "get_data_parallel_group", lambda: object(), raising=False)
    monkeypatch.setattr(data_module.mpu, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.device_utils, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(data_module.dist, "all_reduce", lambda tensor, op=None, group=None: None)

    args = Namespace(
        balance_data=True,
        global_batch_size=16,
        micro_batch_size=4,
        use_dynamic_batch_size=False,
    )
    rollout_data = {"total_lengths": list(range(16))}

    _, num_microbatches = data_module.get_data_iterator(args, object(), rollout_data)

    assert num_microbatches == [2, 2]
