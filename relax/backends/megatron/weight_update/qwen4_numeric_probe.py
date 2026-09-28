# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Isolated, opt-in BF16 value samples for Qwen4 online weight sync.

This module is a probe: it samples actual HF export tensors, not model
metadata. Producer and consumer copies follow the same sampling contract.
"""

from __future__ import annotations

import hashlib

import torch


VISION = "model.visual.blocks.0.norm1.weight"
QSA = "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight"
BACKBONE = "model.language_model.layers.3.self_attn.q_norm.weight"
PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
DIRECT_KEYS = frozenset((VISION, QSA, BACKBONE))


def wanted(name: str, tp_rank: int | None) -> bool:
    return name in DIRECT_KEYS or (name == PLE and tp_rank == 0)


def fingerprint(name: str, tensor: torch.Tensor, tp_rank: int | None) -> dict | None:
    """Hash all small norms, or 68 deterministic cells of TP0's PLE shard 0.

    The caller must pass the exact source shard or the corresponding
    destination slice. No implicit cast is done: dtype differences remain
    visible.
    """
    if not wanted(name, tp_rank):
        return None
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"probe expected BF16 for {name}, got {tensor.dtype}")
    if name == PLE:
        if tensor.ndim != 2 or tuple(tensor.shape) != (125015, 160):
            raise ValueError(f"unexpected PLE shard shape: {tuple(tensor.shape)}")
        rows_data = [(i * 125014) // 16 for i in range(17)]
        rows = torch.tensor(rows_data, device=tensor.device)
        cols = torch.tensor([0, 53, 106, 159], device=tensor.device)
        sample = tensor.index_select(0, rows).index_select(1, cols)
        selection = "17_even_rows_x_4_fixed_cols"
    else:
        sample = tensor
        selection = "all"
    cpu = sample.detach().contiguous().to("cpu")
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "selection": selection,
        "sample_count": cpu.numel(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "distinct_bf16_values": len({raw[i : i + 2] for i in range(0, len(raw), 2)}),
    }
