# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rank-local parameter audit around one selected 48L SFT optimizer step."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_STATE: dict[str, Any] = {}
_SAMPLE_COUNT = 257
_FROZEN_TOKENS = ("per_layer_embedding.", "self_attention.indexer.", "vision_model.", "vision_projection.")


def _parameters(model: list[torch.nn.Module]) -> list[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    result: list[tuple[str, torch.nn.Parameter]] = []
    for chunk_index, chunk in enumerate(model):
        for name, parameter in chunk.named_parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append((f"chunk{chunk_index}.{name}", parameter))
    return result


def _sampled_tensor(parameter: torch.Tensor) -> torch.Tensor:
    flat = parameter.detach().reshape(-1)
    if not flat.numel():
        return flat
    count = min(_SAMPLE_COUNT, flat.numel())
    indices = torch.arange(count, dtype=torch.long, device=flat.device)
    if count > 1:
        indices = indices * (flat.numel() - 1) // (count - 1)
    return flat.index_select(0, indices)


def _sample(parameter: torch.Tensor, *, as_float32: bool = False) -> bytes:
    sampled = _sampled_tensor(parameter)
    if not sampled.numel():
        return b""
    if as_float32:
        sampled = sampled.float()
    return sampled.contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _optimizer_owned_parameters(optimizer: Any) -> tuple[dict[int, tuple[Any, int, int]], bool]:
    owned: dict[int, tuple[Any, int, int]] = {}
    found = False
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            child_owned, child_found = _optimizer_owned_parameters(child)
            owned.update(child_owned)
            found = found or child_found
    mapping = getattr(optimizer, "model_param_group_index_map", None)
    if mapping is not None:
        owned.update({id(parameter): (optimizer, *position) for parameter, position in mapping.items()})
        found = True
    return owned, found


def _write_record(rank: int, phase: str, payload: dict[str, Any]) -> None:
    root = os.environ.get("QWEN48_TRAIN_AUDIT_DIR")
    if not root:
        raise RuntimeError("QWEN48_TRAIN_AUDIT_DIR is required for the training gate")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"rank_{rank:03d}_{phase}.json"
    target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _gradient_hook(name: str, parameter: torch.nn.Parameter):
    def capture(gradient: torch.Tensor) -> None:
        # TE fused wgrad writes the real accumulation to main_grad and returns
        # an uninitialized dummy tensor solely to trigger autograd hooks.
        fused = bool(getattr(parameter, "grad_added_to_main_grad", False))
        source = "main_grad" if fused else "autograd"
        actual = getattr(parameter, "main_grad", None) if fused else gradient
        if actual is None:
            raise RuntimeError(f"{name} has fused wgrad but no main_grad")
        sampled = _sampled_tensor(actual)
        if not sampled.numel():
            return
        values = sampled.float()
        result = _STATE["gradients"].setdefault(name, {"source": source, "finite_gpu": [], "max_abs_gpu": []})
        if result["source"] != source:
            raise RuntimeError(f"gradient source changed for {name}: {result['source']} -> {source}")
        result["finite_gpu"].append(torch.isfinite(values).all())
        result["max_abs_gpu"].append(torch.nan_to_num(values.abs(), nan=0.0, posinf=0.0).max())

    return capture


def audit_qwen38_48l_train_gate(
    args: Any,
    rollout_id: int,
    step_id: int,
    model: list[torch.nn.Module],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Record frozen membership and compare parameters around one optimizer
    step."""
    del opt_param_scheduler
    start_rollout = int(os.environ.get("QWEN48_AUDIT_START_ROLLOUT", "0"))
    if start_rollout < 0:
        raise ValueError("QWEN48_AUDIT_START_ROLLOUT must be nonnegative")
    if rollout_id not in (start_rollout, start_rollout + 1):
        return
    if step_id != 0:
        raise RuntimeError(f"48L audit requires one optimizer step per rollout; got step_id={step_id}")
    rank = dist.get_rank(group=dist.group.WORLD)
    named = _parameters(model)
    frozen = [(name, parameter) for name, parameter in named if any(token in name for token in _FROZEN_TOKENS)]
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    mtp_enabled = bool(getattr(args, "enable_mtp_training", False))
    mtp = [(name, parameter) for name, parameter in named if ".mtp." in name]
    mtp_expected_trainable = [
        (name, parameter) for name, parameter in mtp if not any(token in name for token in _FROZEN_TOKENS)
    ]
    if not trainable:
        raise RuntimeError(f"rank {rank} has no trainable parameters")

    if rollout_id == start_rollout:
        if _STATE:
            raise RuntimeError("stale 48L audit state")
        unexpected = [name for name, parameter in frozen if parameter.requires_grad]
        if unexpected:
            raise RuntimeError(f"expected frozen parameters are trainable: {unexpected[:8]}")
        if mtp_enabled:
            frozen_mtp = [name for name, parameter in mtp_expected_trainable if not parameter.requires_grad]
            if frozen_mtp:
                raise RuntimeError(f"MTP parameters unexpectedly frozen: {frozen_mtp[:8]}")
        owned, found_mapping = _optimizer_owned_parameters(optimizer)
        if not found_mapping:
            raise RuntimeError("optimizer exposes no model parameter membership mapping")
        frozen_in_optimizer = [name for name, parameter in frozen if id(parameter) in owned]
        if frozen_in_optimizer:
            raise RuntimeError(f"frozen parameters in optimizer: {frozen_in_optimizer[:8]}")
        # The distributed optimizer's mapping contains only this rank's DP/CP
        # shards. A trainable model parameter can be owned by a peer rank.
        local_trainable = [(name, parameter) for name, parameter in trainable if id(parameter) in owned]
        if not local_trainable:
            raise RuntimeError(f"rank {rank} owns no trainable optimizer shards")
        norms = [(name, parameter) for name, parameter in local_trainable if "norm" in name.lower()]
        local_mtp = [(name, parameter) for name, parameter in mtp_expected_trainable if id(parameter) in owned]
        probes = list(dict(norms[:4] + local_trainable[:12] + (local_mtp[:4] if mtp_enabled else [])).items())
        local_mtp_names = {name for name, _ in local_mtp}
        mtp_probe_names = [name for name, _ in probes if name in local_mtp_names]
        samples = {name: (_sample(parameter), "frozen_model") for name, parameter in frozen}
        probe_shards: dict[str, tuple[Any, torch.Tensor]] = {}
        for name, parameter in probes:
            child, group_index, group_order = owned[id(parameter)]
            shard = child.optimizer.param_groups[group_index]["params"][group_order]
            probe_shards[name] = (child, shard)
            # TE FusedAdam creates the FP32 master from this BF16 shard at
            # its first step; compare the post-step master against this value.
            samples[name] = (_sample(shard, as_float32=True), "owned_shard_fp32_initial")
        gradient_probes = list(
            dict(probes[:4] + [(name, parameter) for name, parameter in probes if name in mtp_probe_names]).items()
        )
        handles = [parameter.register_hook(_gradient_hook(name, parameter)) for name, parameter in gradient_probes]
        _STATE.update(
            {
                "samples": samples,
                "probe_shards": probe_shards,
                "handles": handles,
                "gradients": {},
                "probe_names": [x[0] for x in probes],
                "mtp_enabled": mtp_enabled,
                "mtp_parameters": len(mtp),
                "mtp_expected_trainable_parameters": len(mtp_expected_trainable),
                "mtp_trainable_parameters": sum(parameter.requires_grad for _, parameter in mtp),
                "mtp_frozen_parameters": sum(not parameter.requires_grad for _, parameter in mtp),
                "mtp_owned_trainable_parameters": len(local_mtp),
                "mtp_probe_names": mtp_probe_names,
            }
        )
        payload = {
            "rank": rank,
            "phase": "before",
            "parameters": len(named),
            "trainable_parameters": len(trainable),
            "trainable_numel": sum(p.numel() for _, p in trainable),
            "frozen_parameters": len(frozen),
            "frozen_numel": sum(p.numel() for _, p in frozen),
            "frozen_in_optimizer": len(frozen_in_optimizer),
            "optimizer_membership_mapping_found": found_mapping,
            "local_owned_trainable_parameters": len(local_trainable),
            "trainable_parameters_owned_by_peer": len(trainable) - len(local_trainable),
            "probe_names": _STATE["probe_names"],
            "mtp_enabled": mtp_enabled,
            "mtp_loss_normalization": getattr(args, "mtp_loss_normalization", None),
            "mtp_detach_paths": list(getattr(args, "mtp_detach_paths", ())),
            "moe_aux_loss_coeff": getattr(args, "moe_aux_loss_coeff", None),
            "mtp_parameters": len(mtp),
            "mtp_expected_trainable_parameters": len(mtp_expected_trainable),
            "mtp_trainable_parameters": sum(parameter.requires_grad for _, parameter in mtp),
            "mtp_frozen_parameters": sum(not parameter.requires_grad for _, parameter in mtp),
            "mtp_owned_trainable_parameters": len(local_mtp),
            "mtp_probe_names": mtp_probe_names,
            "probe_sources": {name: samples[name][1] for name in _STATE["probe_names"]},
            "frozen_examples": [name for name, _ in frozen[:8]],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        _write_record(rank, "before", payload)
        logger.info("[QWEN38_48L_AUDIT] %s", json.dumps(payload, sort_keys=True))
        return

    if not _STATE:
        raise RuntimeError(f"rollout {rollout_id} reached without rollout {start_rollout} audit state")
    for handle in _STATE["handles"]:
        handle.remove()
    current = dict(named)
    changed_frozen: list[str] = []
    changed_probes: list[str] = []
    changed_master_probes: list[str] = []
    master_state_missing: list[str] = []
    nonfinite_master_probes: list[str] = []
    for name, (before, source) in _STATE["samples"].items():
        parameter = current.get(name)
        if parameter is None:
            raise RuntimeError(f"parameter disappeared after optimizer step: {name}")
        if source == "frozen_model":
            if _sample(parameter) != before:
                changed_frozen.append(name)
            continue
        child, shard = _STATE["probe_shards"][name]
        state = child.optimizer.state.get(shard, {})
        if "master_param" in state:
            selected = child.optimizer.get_unscaled_state(shard, "master_param")
            if not bool(torch.isfinite(_sampled_tensor(selected).float()).all().item()):
                nonfinite_master_probes.append(name)
            if _sample(selected, as_float32=True) != before:
                changed_master_probes.append(name)
                changed_probes.append(name)
        else:
            master_state_missing.append(name)
            if _sample(shard, as_float32=True) != before:
                changed_probes.append(name)
    if changed_frozen:
        raise RuntimeError(f"frozen parameter changed after optimizer step: {changed_frozen[:8]}")
    if nonfinite_master_probes:
        raise RuntimeError(
            f"non-finite sampled FP32 master after optimizer step on rank {rank}: {nonfinite_master_probes[:8]}"
        )
    gradients = {
        name: {
            "source": item["source"],
            "calls": len(item["finite_gpu"]),
            "finite": bool(torch.stack(item["finite_gpu"]).all().item()),
            "max_abs": float(torch.stack(item["max_abs_gpu"]).max().item()),
        }
        for name, item in _STATE["gradients"].items()
        if item["finite_gpu"]
    }
    if any(not item["finite"] for item in gradients.values()):
        raise RuntimeError(f"non-finite sampled gradient after optimizer step on rank {rank}: {gradients}")
    payload = {
        "rank": rank,
        "phase": "after_one_optimizer_step",
        "changed_trainable_probe_count": len(changed_probes),
        "changed_trainable_probe_examples": changed_probes[:8],
        "changed_fp32_master_probe_count": len(changed_master_probes),
        "changed_mtp_fp32_master_probe_count": sum(
            name in changed_master_probes for name in _STATE["mtp_probe_names"]
        ),
        "changed_mtp_fp32_master_probe_names": [
            name for name in _STATE["mtp_probe_names"] if name in changed_master_probes
        ],
        "finite_nonzero_mtp_gradient_probe_count": sum(
            gradients.get(name, {}).get("finite", False) and gradients.get(name, {}).get("max_abs", 0) > 0
            for name in _STATE["mtp_probe_names"]
        ),
        "fp32_master_state_missing_probe_count": len(master_state_missing),
        "finite_nonzero_gradient_probe_count": sum(
            item["finite"] and item["max_abs"] > 0 for item in gradients.values()
        ),
        "changed_frozen_count": 0,
        "gradients": gradients,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    _write_record(rank, "after", payload)
    logger.info("[QWEN38_48L_AUDIT] %s", json.dumps(payload, sort_keys=True))
    _STATE.clear()
