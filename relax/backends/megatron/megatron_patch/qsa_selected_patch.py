# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in tensor-core QSA backend from the existing compact selected block
IDs."""

import os

import torch
from megatron.core.transformer.experimental_attention_variant.qsa import QSACoreAttention, QSASelection

from relax.backends.megatron.megatron_patch.qsa_selected_triton import qsa_sparse_attention_from_block_ids
from relax.utils.logging_utils import get_logger


if os.environ.get("MCORE_QSA_SPARSE_BACKEND") != "id_sparse":
    raise RuntimeError("QWEN48_QSA_SELECTED_TRITON requires MCORE_QSA_SPARSE_BACKEND=id_sparse")

_original_id_sparse_forward = QSACoreAttention._id_sparse_forward


def _selected_forward(
    self: QSACoreAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selection: QSASelection,
    is_thd: bool,
) -> torch.Tensor:
    if selection.selected_ids is None:
        raise RuntimeError("Selected QSA Triton backend requires compact block IDs")
    if is_thd:
        q, k, v = query, key, value
        sequence_length, heads, width = q.shape
        batch_size = 1
    else:
        sequence_length, batch_size, heads, width = query.shape
        q = query.permute(1, 0, 2, 3).reshape(batch_size * sequence_length, heads, width)
        k = key.permute(1, 0, 2, 3).reshape(batch_size * sequence_length, key.shape[-2], width)
        v = value.permute(1, 0, 2, 3).reshape(batch_size * sequence_length, value.shape[-2], width)
    # The selected-ID gather kernel wins for narrow head dimensions in local A/B tests.
    if width < 128:
        return _original_id_sparse_forward(self, query, key, value, selection, is_thd)
    out = qsa_sparse_attention_from_block_ids(
        q,
        k,
        v,
        selection.selected_ids.view(batch_size * sequence_length, -1),
        selection.positions.reshape(-1),
        self.softmax_scale,
        selection.compress_ratio,
    )
    if is_thd:
        return out.reshape(sequence_length, heads * width)
    return out.reshape(batch_size, sequence_length, heads * width).transpose(0, 1).contiguous()


if os.environ.get("QWEN48_QSA_SELECTED_TRITON") == "1":
    QSACoreAttention._id_sparse_forward = _selected_forward
    get_logger(__name__).info("Selected-index QSA Triton patch enabled")
