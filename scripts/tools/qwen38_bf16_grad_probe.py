# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-step, opt-in BF16 main-gradient diagnostic for Qwen3.8 VL GRPO."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _step_parameters(optimizer: Any) -> list[torch.Tensor]:
    """Use HybridDeviceOptimizer's real CPU/GPU step tensors when available."""
    result: list[torch.Tensor] = []
    for part in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner = getattr(part, "optimizer", None)
        if hasattr(inner, "cpu_optimizers"):
            step_optimizers = list(inner.cpu_optimizers)
            if inner.gpu_optimizer is not None:
                step_optimizers.append(inner.gpu_optimizer)
            for step_optimizer in step_optimizers:
                for group in step_optimizer.param_groups:
                    result.extend(group["params"])
        else:
            result.extend(part.get_parameters())
    return result


def install_probe(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: list[Any],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Inspect the actual backward and optimizer step without changing default
    runs."""
    del opt_param_scheduler
    if rollout_id != 0 or step_id != 0:
        return
    if args.main_grads_dtype != torch.bfloat16 or args.accumulate_allreduce_grads_in_fp32:
        raise RuntimeError("BF16 grad probe requires BF16 main grads and BF16 reduction")

    # With CPU offload, DistributedOptimizer.get_parameters() returns the BF16
    # shard while HybridDeviceOptimizer updates a separate FP32 CPU copy.
    # Sample the actual tensors given to the inner step optimizers.
    sampled = [
        (param, param.detach().reshape(-1)[:64].clone())
        for param in _step_parameters(optimizer)[:512]
        if param.numel() > 0
    ]
    if not sampled:
        raise RuntimeError("BF16 grad probe found no optimizer parameters")

    torch.cuda.reset_peak_memory_stats()
    original_step = optimizer.step

    def probed_step(*step_args: Any, **step_kwargs: Any) -> tuple[Any, Any, Any]:
        dtype_counts: Counter[str] = Counter()
        grad_missing = 0
        grad_dtype_mismatch = 0
        indexer_total = 0
        indexer_trainable = 0
        for chunk in model:
            for name, param in chunk.named_parameters():
                if "self_attention.indexer." in name:
                    indexer_total += 1
                    indexer_trainable += int(param.requires_grad)
                main_grad = getattr(param, "main_grad", None)
                if param.requires_grad and main_grad is None:
                    grad_missing += 1
                elif param.requires_grad:
                    dtype_counts[str(main_grad.dtype)] += 1
                    grad_dtype_mismatch += int(main_grad.dtype != param.dtype)

        try:
            result = original_step(*step_args, **step_kwargs)
        finally:
            optimizer.step = original_step

        updated, grad_norm, _ = result
        changed = sum(not torch.equal(param.detach().reshape(-1)[:64], before) for param, before in sampled)
        sampled_dtypes = dict(Counter(str(param.dtype) for param, _ in sampled))
        finite_norm = grad_norm is not None and math.isfinite(float(grad_norm))
        positive_norm = finite_norm and float(grad_norm) > 0
        peak = torch.cuda.max_memory_allocated()
        logger.info(
            "QWEN38_BF16_GRAD_PROBE rollout=%d step=%d main_grad_dtypes=%s grad_missing=%d "
            "grad_dtype_mismatch=%d indexer_total=%d indexer_trainable=%d optimizer_step_returned_success=%s "
            "grad_norm_finite=%s grad_norm_positive=%s sampled_step_param_dtypes=%s "
            "sampled_step_params_changed=%d/%d "
            "train_step_peak_allocated_bytes=%d",
            rollout_id,
            step_id,
            dict(dtype_counts),
            grad_missing,
            grad_dtype_mismatch,
            indexer_total,
            indexer_trainable,
            bool(updated),
            finite_norm,
            positive_norm,
            sampled_dtypes,
            changed,
            len(sampled),
            peak,
        )
        # A zero changed sample is inconclusive, so it cannot satisfy the
        # one-step update gate. Report it and investigate the sampled params.
        if (
            not dtype_counts.get("torch.bfloat16")
            or grad_missing
            or grad_dtype_mismatch
            or indexer_total == 0
            or indexer_trainable
            or not updated
            or not finite_norm
            or not positive_norm
            or changed == 0
        ):
            raise RuntimeError("QWEN38_BF16_GRAD_PROBE failed its one-step acceptance gate")
        return result

    optimizer.step = probed_step
