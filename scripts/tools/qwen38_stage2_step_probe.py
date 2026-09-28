# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Audit one optimizer step per Stage-2 rollout, for up to two rollouts."""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _mapped_fp32_step_parameter(inner: Any, source: torch.Tensor) -> torch.Tensor:
    step_param = getattr(inner, "param_to_inner_param", {}).get(source)
    if step_param is None:
        raise RuntimeError("QSA indexer has no mapped optimizer step tensor")
    if step_param.dtype != torch.float32 or step_param.numel() == 0:
        raise RuntimeError("QSA indexer has no local FP32 optimizer step tensor")
    return step_param


def _inner_step_parameter(optimizer: Any, model_parameter: torch.nn.Parameter) -> torch.Tensor | None:
    """Find the local FP32 step shard; a DP/CP bucket may live on a peer
    rank."""
    for part in getattr(optimizer, "chained_optimizers", [optimizer]):
        model_groups = getattr(part, "model_float16_groups", ())
        fp32_groups = getattr(part, "shard_fp32_from_float16_groups", ())
        bf16_groups = getattr(part, "shard_float16_groups", ())
        for model_group, fp32_group, bf16_group in zip(model_groups, fp32_groups, bf16_groups):
            for model_param, fp32_param, bf16_param in zip(model_group, fp32_group, bf16_group):
                if model_param is model_parameter:
                    inner = getattr(part, "optimizer", None)
                    # Precision-aware DistributedOptimizer intentionally has no FP32 main
                    # shard. HybridDeviceOptimizer makes its FP32 copy from the BF16 shard.
                    source = fp32_param if fp32_param is not None else bf16_param
                    return _mapped_fp32_step_parameter(inner, source)
        for model_group, shard_group in zip(
            getattr(part, "model_fp32_groups", ()), getattr(part, "shard_fp32_groups", ())
        ):
            for model_param, shard_param in zip(model_group, shard_group):
                if model_param is model_parameter:
                    return _mapped_fp32_step_parameter(getattr(part, "optimizer", None), shard_param)
    return None


def _gradient_status(parameter: torch.nn.Parameter, *, split_rows: int | None = None) -> tuple[bool, bool]:
    """Check the already reduced Megatron main gradient once on this gate
    step."""
    grad = getattr(parameter, "main_grad", None)
    if grad is None or grad.shape != parameter.shape or not torch.isfinite(grad).all().item():
        return False, False
    if split_rows is None:
        return bool(torch.count_nonzero(grad).item()), True
    q_present = bool(torch.count_nonzero(grad[:split_rows]).item())
    k_present = bool(torch.count_nonzero(grad[split_rows:]).item())
    return q_present and k_present, True


def _valid_observations(observed: list[dict[str, Any]], expected_layers: set[int]) -> bool:
    """Require a true long packed document through every QSA indexer, not just
    a long pack."""
    if {batch["layer"] for batch in observed} != expected_layers:
        return False
    return all(
        not batch["all_selected"]
        and 2051 < batch["physical_tokens"] <= 4096
        and max(batch["valid_doc_lengths"], default=0) > 2051
        and max(batch["physical_doc_lengths"], default=0) > 2051
        for batch in observed
    )


def _validate_probe_step(args: Any, rollout_id: int, step_id: int) -> None:
    if args.num_rollout not in (1, 2) or rollout_id not in range(args.num_rollout) or step_id != 0:
        raise RuntimeError(
            f"Stage-2 probe requires one step in each of one or two rollouts; "
            f"got num_rollout={args.num_rollout}, rollout_id={rollout_id}, step_id={step_id}"
        )


def install_probe(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: list[Any],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Wrap each bounded Stage-2 Megatron optimizer step with checks."""
    del opt_param_scheduler
    _validate_probe_step(args, rollout_id, step_id)
    if (
        args.qsa_indexer_loss_coeff != 0.7
        or os.environ.get("MCORE_QSA_SPARSE_BACKEND") != "id_sparse"
        or args.main_grads_dtype != torch.bfloat16
        or args.accumulate_allreduce_grads_in_fp32
    ):
        raise RuntimeError("Stage-2 step probe requires coeff0.7, id_sparse, and BF16 main grads")

    indexers = []
    backbone = []
    vision = []
    for chunk in model:
        for name, module in chunk.named_modules():
            if name.endswith("self_attention.indexer"):
                parameter = module.index_qk_proj.weight
                indexers.append((name, module, parameter, module.n_heads * module.head_dim))
        for name, parameter in chunk.named_parameters():
            if name.endswith("self_attention.linear_qkv.weight"):
                backbone.append((name, parameter))
            if name.endswith("vision_model.patch_embed.proj.weight"):
                vision.append((name, parameter))
    if len(indexers) != 2 or not backbone or len(vision) != 1:
        raise RuntimeError(
            f"Expected two QSA indexers, backbone QKV, and one vision patch embed; "
            f"got {len(indexers)}, {len(backbone)}, {len(vision)}"
        )
    if any(not parameter.requires_grad for _, _, parameter, _ in indexers):
        raise RuntimeError("Stage-2 indexer is frozen")

    from megatron.core import parallel_state as mpu

    # Dense DistributedOptimizer shards over its intra DP/CP group. A rank can
    # legitimately own neither indexer; check the whole shard group before the
    # forward so a missing mapping fails on all peers at the same point.
    shard_group = mpu.get_data_parallel_group(
        with_context_parallel=True, with_gtp_remat=False, partial_data_parallel=True
    )
    step_tensors = {}
    local_mapping_errors = []
    for name, _, parameter, _ in indexers:
        try:
            tensor = _inner_step_parameter(optimizer, parameter)
        except RuntimeError as exc:
            local_mapping_errors.append(f"{name}: {exc}")
        else:
            if tensor is not None:
                step_tensors[name] = tensor
    local_shard_status = torch.tensor(
        [
            value
            for name, _, _, _ in indexers
            for value in (int(name in step_tensors), int(bool(local_mapping_errors)))
        ],
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    shard_status = local_shard_status.clone()
    dist.all_reduce(shard_status, op=dist.ReduceOp.SUM, group=shard_group)
    if any(shard_status[2 * i].item() == 0 or shard_status[2 * i + 1].item() for i in range(len(indexers))):
        raise RuntimeError(
            f"QSA indexer FP32 optimizer shards unavailable; local_errors={local_mapping_errors}, "
            f"group_status={shard_status.tolist()}"
        )
    observed_batches: list[dict[str, Any]] = []

    def observe_selection(module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any], selection: Any) -> None:
        packed = kwargs.get("packed_seq_params", inputs[2] if len(inputs) > 2 else None)
        layout = getattr(packed, "qsa_stage2_layout_cpu", None)
        ids = getattr(selection, "selected_ids", None)
        if layout is None or ids is None:
            raise RuntimeError("Stage-2 training forward did not use packed selected-ID QSA")
        physical_cu, valid_lengths = layout
        observed_batches.append(
            {
                "layer": module.layer_number,
                "physical_tokens": ids.shape[0],
                "physical_doc_lengths": tuple(b - a for a, b in zip(physical_cu[:-1], physical_cu[1:])),
                "valid_doc_lengths": valid_lengths,
                "all_selected": selection.all_selected,
            }
        )

    hook_handles = [module.register_forward_hook(observe_selection, with_kwargs=True) for _, module, _, _ in indexers]
    original_step = optimizer.step
    torch.cuda.reset_peak_memory_stats()

    def probed_step(*step_args: Any, **step_kwargs: Any) -> tuple[Any, Any, Any]:
        indexer_gradients = {}
        for name, _, parameter, q_rows in indexers:
            nonzero, finite = _gradient_status(parameter, split_rows=q_rows)
            grad = getattr(parameter, "main_grad", None)
            indexer_gradients[name] = {
                "q_and_k_nonzero": nonzero,
                "finite": finite,
                "q_abs_sum": float(grad[:q_rows].float().abs().sum().item()) if finite else None,
                "k_abs_sum": float(grad[q_rows:].float().abs().sum().item()) if finite else None,
            }
        backbone_gradients = {name: _gradient_status(parameter) for name, parameter in backbone}
        vision_grad = getattr(vision[0][1], "main_grad", None)
        vision_nonzero, vision_finite = _gradient_status(vision[0][1])
        # A CP rank without image tokens may have no local vision gradient.
        # Another rank in its CP group must still supply a finite contribution.
        if vision_grad is None:
            vision_finite = True
        from megatron.core import parallel_state as mpu

        cp_group = mpu.get_context_parallel_group()
        vision_any = torch.tensor(int(vision_nonzero), device=torch.device("cuda", torch.cuda.current_device()))
        dist.all_reduce(vision_any, op=dist.ReduceOp.MAX, group=cp_group)

        before = {name: tensor.detach().clone() for name, tensor in step_tensors.items()}
        try:
            result = original_step(*step_args, **step_kwargs)
        finally:
            optimizer.step = original_step
            for handle in hook_handles:
                handle.remove()

        updated, grad_norm, _ = result
        changed = {
            name: int(torch.count_nonzero(tensor.detach() != before[name]).item())
            for name, tensor in step_tensors.items()
        }
        # The distributed optimizer assigns DP/CP bucket ranges to ranks. All
        # shard peers participate, including ranks without this parameter.
        local_step_status = torch.tensor(
            [
                value
                for name, _, _, _ in indexers
                for value in (int(name in step_tensors), int(changed.get(name, 0) > 0))
            ],
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        shard_step_status = local_step_status.clone()
        dist.all_reduce(shard_step_status, op=dist.ReduceOp.SUM, group=shard_group)
        shard_owners = {name: int(shard_step_status[2 * i].item()) for i, (name, _, _, _) in enumerate(indexers)}
        shard_changed = {name: int(shard_step_status[2 * i + 1].item()) for i, (name, _, _, _) in enumerate(indexers)}
        norm_finite = grad_norm is not None and math.isfinite(float(grad_norm))
        logger.info(
            "QWEN38_STAGE2_STEP_PROBE rank=%d tp_rank=%d cp_rank=%d rollout=%d step=%d indexer_gradients=%s "
            "indexer_fp32_local_owner=%s indexer_fp32_changed=%s indexer_shard_owners=%s "
            "indexer_shard_changed=%s backbone_gradients=%s vision_local_nonzero=%s "
            "vision_local_finite=%s vision_cp_any=%s optimizer_step_returned_success=%s "
            "grad_norm_finite=%s grad_norm_positive=%s observed_batches=%s "
            "train_step_peak_allocated_bytes=%d train_step_peak_reserved_bytes=%d",
            dist.get_rank(group=dist.group.WORLD),
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_context_parallel_rank(),
            rollout_id,
            step_id,
            indexer_gradients,
            {name: name in step_tensors for name, _, _, _ in indexers},
            changed,
            shard_owners,
            shard_changed,
            backbone_gradients,
            vision_nonzero,
            vision_finite,
            bool(vision_any.item()),
            bool(updated),
            norm_finite,
            norm_finite and float(grad_norm) > 0,
            observed_batches,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
        )
        if (
            not all(status["q_and_k_nonzero"] and status["finite"] for status in indexer_gradients.values())
            or not all(nonzero and finite for nonzero, finite in backbone_gradients.values())
            or not vision_finite
            or not bool(vision_any.item())
            or not updated
            or not norm_finite
            or float(grad_norm) <= 0
            or not all(shard_owners[name] > 0 and shard_changed[name] > 0 for name, _, _, _ in indexers)
            or not _valid_observations(observed_batches, {module.layer_number for _, module, _, _ in indexers})
        ):
            raise RuntimeError(f"QWEN38_STAGE2_STEP_PROBE failed acceptance at rollout {rollout_id}, step {step_id}")
        return result

    optimizer.step = probed_step
