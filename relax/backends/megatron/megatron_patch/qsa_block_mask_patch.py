# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in coarse QSA BlockMask construction from the selected R=4 bitset.

The original QSA mask_mod still decides every token pair inside partial blocks.
This patch only replaces the expensive token-pair scan used to discover which
128-token FlexAttention blocks are empty, partial, or full.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from megatron.core.transformer.experimental_attention_variant import qsa
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_BLOCK_SIZE = 128


@triton.jit
def _classify_blocks(
    doc_ids,
    positions,
    selected_bits,
    flags,
    sequence_length: tl.constexpr,
    blocks_per_row: tl.constexpr,
    bytes_per_row: tl.constexpr,
    block_size: tl.constexpr,
    ratio: tl.constexpr,
):
    pair = tl.program_id(0)
    batch = pair // (blocks_per_row * blocks_per_row)
    query_block = pair // blocks_per_row % blocks_per_row
    key_block = pair % blocks_per_row
    query_start = query_block * block_size
    key_start = key_block * block_size
    query_end = tl.minimum(query_start + block_size, sequence_length) - 1
    key_end = tl.minimum(key_start + block_size, sequence_length) - 1
    result = 0

    # Document IDs are monotone in packed THD order. A mixed boundary block
    # is kept as partial whenever its document ranges and causal ranges overlap.
    if key_start <= query_end:
        q_first_doc = tl.load(doc_ids + batch * sequence_length + query_start)
        q_last_doc = tl.load(doc_ids + batch * sequence_length + query_end)
        k_first_doc = tl.load(doc_ids + batch * sequence_length + key_start)
        k_last_doc = tl.load(doc_ids + batch * sequence_length + key_end)
        if q_first_doc <= k_last_doc and k_first_doc <= q_last_doc:
            if q_first_doc != q_last_doc or k_first_doc != k_last_doc:
                result = 1
            elif q_first_doc == k_first_doc:
                key_first_position = tl.load(positions + batch * sequence_length + key_start)
                key_last_position = tl.load(positions + batch * sequence_length + key_end)
                first_group = key_first_position // ratio
                last_group = key_last_position // ratio
                query_offsets = tl.arange(0, block_size)
                group_offsets = tl.arange(0, 64)
                query_indices = query_start + query_offsets
                group_ids = first_group + group_offsets
                valid_queries = query_indices <= query_end
                valid_groups = group_ids <= last_group
                query_positions = tl.load(
                    positions + batch * sequence_length + query_indices,
                    mask=valid_queries,
                    other=0,
                )
                bit_offsets = (batch * sequence_length + query_indices[:, None]) * bytes_per_row + group_ids[
                    None, :
                ] // 8
                bit_bytes = tl.load(
                    selected_bits + bit_offsets,
                    mask=valid_queries[:, None] & valid_groups[None, :],
                    other=0,
                ).to(tl.int32)
                selected = ((bit_bytes >> (group_ids[None, :] % 8)) & 1) != 0
                # A group at a physical tile edge may contain only some of its
                # four tokens. The first/last covered positions preserve exact
                # empty/full classification without a token-pair scan.
                first_key = tl.maximum(key_first_position, group_ids * ratio)
                last_key = tl.minimum(key_last_position, group_ids * ratio + ratio - 1)
                tail_start = ((query_positions + 1) // ratio) * ratio
                admitted = selected | (first_key[None, :] >= tail_start[:, None])
                any_pair = admitted & (first_key[None, :] <= query_positions[:, None])
                full_pair = admitted & (last_key[None, :] <= query_positions[:, None])
                valid = valid_queries[:, None] & valid_groups[None, :]
                has_pair = tl.sum(tl.sum((any_pair & valid).to(tl.int32), 0), 0) > 0
                missing_pair = tl.sum(tl.sum((~full_pair & valid).to(tl.int32), 0), 0) > 0
                result = tl.where(has_pair, tl.where(missing_pair, 1, 2), 0)
                # create_block_mask treats an incomplete Q or KV tile as partial,
                # even when every in-range token pair is admitted.
                if query_end - query_start + 1 < block_size or key_end - key_start + 1 < block_size:
                    result = tl.minimum(result, 1)
    tl.store(flags + pair, result)


@triton.jit
def _pack_block_indices(
    flags,
    partial_counts,
    partial_indices,
    full_counts,
    full_indices,
    blocks_per_row: tl.constexpr,
    scan_width: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, scan_width)
    valid = columns < blocks_per_row
    block_flags = tl.load(flags + row * blocks_per_row + columns, mask=valid, other=0)
    partial = (block_flags == 1) & valid
    full = (block_flags == 2) & valid
    partial_offsets = tl.cumsum(partial.to(tl.int32), 0) - 1
    full_offsets = tl.cumsum(full.to(tl.int32), 0) - 1
    partial_count = tl.sum(partial.to(tl.int32), 0)
    full_count = tl.sum(full.to(tl.int32), 0)
    tl.store(partial_counts + row, partial_count)
    tl.store(full_counts + row, full_count)
    tl.store(partial_indices + row * blocks_per_row + partial_offsets, columns, mask=partial)
    tl.store(full_indices + row * blocks_per_row + full_offsets, columns, mask=full)


def _build_sparse_block_mask(selection: qsa.QSASelection, mask_mod, sequence_length: int) -> BlockMask:
    """Return a full-width BlockMask; partial blocks keep the original
    mask_mod."""
    batch_size, selected_length = selection.positions.shape
    if selected_length != sequence_length:
        raise ValueError("QSA selection length differs from the attention length")
    blocks_per_row = triton.cdiv(sequence_length, _BLOCK_SIZE)
    flags = torch.empty(
        (batch_size, blocks_per_row, blocks_per_row), device=selection.positions.device, dtype=torch.uint8
    )
    _classify_blocks[(batch_size * blocks_per_row * blocks_per_row,)](
        selection.doc_ids,
        selection.positions,
        selection.selected_bits,
        flags,
        sequence_length,
        blocks_per_row,
        selection.bits_per_row,
        _BLOCK_SIZE,
        selection.compress_ratio,
        num_warps=8,
    )
    shape = (batch_size, 1, blocks_per_row)
    index_shape = (*shape, blocks_per_row)
    partial_counts = torch.empty(shape, device=flags.device, dtype=torch.int32)
    full_counts = torch.empty_like(partial_counts)
    # _transpose_ordered reads only the first count entries from each row.
    partial_indices = torch.zeros(index_shape, device=flags.device, dtype=torch.int32)
    full_indices = torch.zeros_like(partial_indices)
    _pack_block_indices[(batch_size * blocks_per_row,)](
        flags,
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        blocks_per_row,
        triton.next_power_of_2(blocks_per_row),
    )
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(sequence_length, sequence_length),
        compute_q_blocks=True,
    )


_original_flex_forward = qsa.QSACoreAttention._flex_forward


def _sparse_flex_forward(
    self: qsa.QSACoreAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    selection: qsa.QSASelection,
    is_thd: bool,
) -> Tensor:
    if (
        not query.is_cuda
        or selection.selected_bits is None
        or selection.compress_ratio != 4
        or selection.bits_per_row <= 0
        or not qsa.HAVE_FLEX_ATTENTION
    ):
        return _original_flex_forward(self, query, key, value, selection, is_thd)
    if is_thd:
        q = query.unsqueeze(0).transpose(1, 2)
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
    else:
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
    batch_size, heads, sequence_length, width = q.shape
    mask_mod = qsa._qsa_mask_mod_factory(selection)
    block_mask = _build_sparse_block_mask(selection, mask_mod, sequence_length)
    try:
        out = qsa._compiled_flex_attention()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            block_mask=block_mask,
            scale=self.softmax_scale,
            enable_gqa=k.shape[1] != heads,
        )
    except torch._dynamo.exc.TorchDynamoException:
        if sequence_length > 4096:
            raise
        return self._dense_masked_forward(query, key, value, selection, is_thd)
    if is_thd:
        return out.transpose(1, 2).squeeze(0).reshape(sequence_length, heads * width)
    return out.permute(2, 0, 1, 3).reshape(sequence_length, batch_size, heads * width)


if os.environ.get("QWEN48_QSA_SPARSE_BLOCKMASK") == "1":
    qsa.QSACoreAttention._flex_forward = _sparse_flex_forward
    logger.info("Sparse QSA BlockMask patch enabled")
