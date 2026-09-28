# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from ModelScope mcore-bridge 7baa28c (Apache-2.0),
# qsa_block_sparse_attn.py, itself derived from radixark/miles.
#
"""Tensor Core QSA forward/backward from compact selected block IDs.

The ModelScope kernels use ``tl.dot`` on key tiles. This adaptation stores
document-relative selected blocks in packed bits and derives key tile lists
from the same IDs. Each kernel also admits the incomplete causal tail and
checks document-relative positions, including misaligned packed starts. BF16
dot products accumulate in FP32. Local measurements are recorded in the Qwen3.8
accelerator milestone; upstream short-context timings are not used as evidence
for Relax's packed 128K run.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


TILE_BUILD_ROWS = 4096


@triton.jit
def _qsa_bs_fwd_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(
        Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    m_i = tl.full((BQ,), float("-inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)

    # Only the key tiles this query tile actually selected, so the kernel beats a dense
    # sweep by the coverage ratio instead of merely matching it. The list is the union
    # over the tile's queries; per-query exactness comes from the mask below.
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-sequence block grid: after the packed-indexer fix a sequence's blocks start
        # at its own first token, which is not a multiple of BLK in a packed batch.
        # Looked up per TOKEN, not per 4-token group: a sequence whose start is not a
        # multiple of BLK has its per-sequence blocks straddling the global groups, so a
        # group-granular lookup would assign some tokens to the wrong block.
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        bits = tl.load(
            SEL + offs_q[:, None] * stride_st + blk // 8,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        selected = ((bits.to(tl.int32) >> (blk % 8)) & 1) != 0
        tail_start = tok_base[:, None] + ((hi[:, None] - tok_base[:, None] + 1) // BLK) * BLK
        ok = (
            (selected | (offs_k[None, :] >= tail_start))
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
        )

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale
        s = tl.where(ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_use[:, None])
        p = tl.where(ok, p, 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_use))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        OUT + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :],
        out,
        mask=q_mask[:, None],
    )
    lse = tl.where(m_i == float("-inf"), float("-inf"), m_i + tl.log(l_safe))
    tl.store(LSE + pid_h * T + offs_q, lse, mask=q_mask)


@triton.jit
def _qsa_bs_dq_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    DO,
    DQ,
    DELTA,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0)
    do = tl.load(
        DO + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    )
    lse = tl.load(LSE + pid_h * T + offs_q, mask=q_mask, other=0.0)
    delta = tl.load(DELTA + pid_h * T + offs_q, mask=q_mask, other=0.0)
    lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
    alive = lse != float("-inf")

    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    dq = tl.zeros((BQ, D), tl.float32)
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-sequence block grid: after the packed-indexer fix a sequence's blocks start
        # at its own first token, which is not a multiple of BLK in a packed batch.
        # Looked up per TOKEN, not per 4-token group: a sequence whose start is not a
        # multiple of BLK has its per-sequence blocks straddling the global groups, so a
        # group-granular lookup would assign some tokens to the wrong block.
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        bits = tl.load(
            SEL + offs_q[:, None] * stride_st + blk // 8,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        selected = ((bits.to(tl.int32) >> (blk % 8)) & 1) != 0
        tail_start = tok_base[:, None] + ((hi[:, None] - tok_base[:, None] + 1) // BLK) * BLK
        ok = (
            (selected | (offs_k[None, :] >= tail_start))
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & alive[:, None]
        )

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )
        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )

        s = tl.dot(q, tl.trans(k_tile)) * scale
        p = tl.exp(s - lse_safe[:, None])
        p = tl.where(ok, p, 0.0)

        dp = tl.dot(do, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

        dq += tl.dot(ds, k_tile)

    tl.store(DQ + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], dq, mask=q_mask[:, None])


@triton.jit
def _qsa_bs_dkdv_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    QLIST,
    QCNT,
    LSE,
    DO,
    DELTA,
    DK,
    DV,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_ql,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    """dK/dV keyed on the KEY tile, so nothing needs atomics.

    Launched per KV head, not per query head: with GQA the ``GROUP`` query
    heads sharing a KV head all contribute to the same dK/dV, so they have to
    be summed here. Writing them from separate programs would have them
    overwrite each other (the gather kernel got away with it only because it
    used atomic_add).
    """
    pid_k = tl.program_id(0)
    kv_head = tl.program_id(1)

    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    k_in = offs_k < T

    k_tile = tl.load(
        K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    v_tile = tl.load(
        V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)

    n_q = tl.load(QCNT + pid_k)
    for i in range(0, n_q):
        qt = tl.load(QLIST + pid_k * stride_ql + i)
        offs_q = qt * BQ + tl.arange(0, BQ)
        q_mask = offs_q < T
        lo = tl.load(LO + offs_q, mask=q_mask, other=0)
        hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
        blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
        tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        bits = tl.load(
            SEL + offs_q[:, None] * stride_st + blk // 8,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        selected = ((bits.to(tl.int32) >> (blk % 8)) & 1) != 0
        tail_start = tok_base[:, None] + ((hi[:, None] - tok_base[:, None] + 1) // BLK) * BLK
        ok = (
            (selected | (offs_k[None, :] >= tail_start))
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & q_mask[:, None]
        )

        for gh in range(0, GROUP):
            qh = kv_head * GROUP + gh
            q = tl.load(
                Q + offs_q[:, None] * stride_qt + qh * stride_qh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO + offs_q[:, None] * stride_ot + qh * stride_oh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(LSE + qh * T + offs_q, mask=q_mask, other=0.0)
            delta = tl.load(DELTA + qh * T + offs_q, mask=q_mask, other=0.0)
            okh = ok & (lse[:, None] != float("-inf"))

            lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
            sc = tl.dot(q, tl.trans(k_tile)) * scale
            p = tl.exp(sc - lse_safe[:, None])
            p = tl.where(okh, p, 0.0)

            dp = tl.dot(do, tl.trans(v_tile))
            ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

            dk += tl.dot(tl.trans(ds), q)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

    tl.store(DK + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], dk, mask=k_in[:, None])
    tl.store(DV + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], dv, mask=k_in[:, None])


def pack_selected_bits(ids: Tensor, num_tokens: int, block_size: int) -> Tensor:
    """Pack document-relative complete-block IDs with bounded int32
    workspace."""
    num_bytes = max(1, triton.cdiv(triton.cdiv(num_tokens, block_size), 8))
    if ids.shape[0] * num_bytes > 1 << 30:
        raise RuntimeError("QSA packed route bitmap would exceed 1 GiB")
    packed = torch.empty((ids.shape[0], num_bytes), dtype=torch.uint8, device=ids.device)
    for begin in range(0, ids.shape[0], 256):
        end = min(begin + 256, ids.shape[0])
        chunk = ids[begin:end]
        valid = chunk >= 0
        safe = chunk.clamp_min(0).long()
        words = torch.zeros((end - begin, num_bytes), dtype=torch.int32, device=ids.device)
        words.scatter_add_(1, safe // 8, ((1 << (safe % 8)) * valid).to(torch.int32))
        packed[begin:end] = words.to(torch.uint8)
    return packed


def build_tile_map(ids: Tensor, positions: Tensor, block_size: int, bk: int) -> Tensor:
    """Mark every key tile touched by selected blocks and the incomplete causal
    tail."""
    total, _ = ids.shape
    bq = 64
    tile = torch.zeros((triton.cdiv(total, bq), triton.cdiv(total, bk)), dtype=torch.bool, device=ids.device)
    for begin in range(0, total, TILE_BUILD_ROWS):
        end = min(begin + TILE_BUILD_ROWS, total)
        query = torch.arange(begin, end, device=ids.device)
        doc_start = query - positions[begin:end].long()
        selected = ids[begin:end].long()
        start = doc_start[:, None] + selected * block_size
        stop = start + block_size - 1
        valid = (selected >= 0) & (start >= 0) & (stop < total)
        row = (query // bq)[:, None].expand_as(selected)
        tile[row[valid], (start // bk)[valid]] = True
        tile[row[valid], (stop // bk)[valid]] = True
        tail_start = doc_start + ((positions[begin:end].long() + 1) // block_size) * block_size
        tail_valid = tail_start <= query
        tile[(query // bq)[tail_valid], (tail_start // bk)[tail_valid]] = True
        tile[(query // bq)[tail_valid], (query // bk)[tail_valid]] = True
    return tile


def compact_tile_map(tile: Tensor) -> tuple[Tensor, Tensor]:
    count = tile.sum(dim=1).to(torch.int32)
    order = torch.argsort((~tile).to(torch.int8), dim=1, stable=True)
    return order.contiguous().to(torch.int32), count


class _QSABlockSparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sel, lo, hi, blk_base, tok_base, tile_fwd, tile_bwd, scale, block_size):
        T, Hq, D = q.shape
        S, Hkv, _ = k.shape
        assert Hq % Hkv == 0
        group = Hq // Hkv
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        selc = sel.contiguous()
        klist, kcnt = compact_tile_map(tile_fwd)
        o = torch.empty(T, Hq, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
        BQ, BK = 64, 64
        grid = (triton.cdiv(T, BQ), Hq)
        _qsa_bs_fwd_kernel[grid](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
            klist.stride(0),
            o.stride(0),
            o.stride(1),
            T,
            selc.shape[1] * 8,
            scale,
            GROUP=group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=block_size,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(qc, kc, vc, selc, lo, hi, blk_base, tok_base, tile_bwd, o, lse)
        ctx.scale = scale
        ctx.group = group
        ctx.block_size = block_size
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        qc, kc, vc, selc, lo, hi, blk_base, tok_base, tile_bwd, o, lse = ctx.saved_tensors
        T, Hq, D = qc.shape
        do = grad_out.contiguous().to(qc.dtype)
        # delta once, in torch: both backward kernels need it and neither should redo it
        delta = (do.float() * o.float()).sum(-1).transpose(0, 1).contiguous()
        dq = torch.empty(T, Hq, D, device=qc.device, dtype=torch.float32)
        dk = torch.zeros(kc.shape, device=kc.device, dtype=torch.float32)
        dv = torch.zeros(vc.shape, device=vc.device, dtype=torch.float32)

        BQ, BK = 64, 32
        klist, kcnt = compact_tile_map(tile_bwd)
        qlist, qcnt = compact_tile_map(tile_bwd.t().contiguous())
        common = (
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
        )
        _qsa_bs_dq_kernel[(triton.cdiv(T, BQ), Hq)](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            do,
            dq,
            delta,
            *common,
            klist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1] * 8,
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        _qsa_bs_dkdv_kernel[(triton.cdiv(T, BK), kc.shape[1])](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            qlist,
            qcnt,
            lse,
            do,
            delta,
            dk,
            dv,
            *common,
            qlist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1] * 8,
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        return dq.to(qc.dtype), dk.to(kc.dtype), dv.to(vc.dtype), None, None, None, None, None, None, None, None, None


def qsa_block_sparse_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    tile_fwd: Tensor,
    tile_bwd: Tensor,
    scale: float,
    block_size: int = 4,
) -> Tensor:
    """Run tensor-core attention from packed selection bits and bounded tile
    maps."""
    return _QSABlockSparseAttn.apply(
        q, k, v, sel_blocks, lo, hi, blk_base, tok_base, tile_fwd, tile_bwd, scale, block_size
    )


def qsa_sparse_attention_from_block_ids(
    q: Tensor, k: Tensor, v: Tensor, block_ids: Tensor, positions: Tensor, scale: float, block_size: int = 4
) -> Tensor:
    """Use the existing indexer's complete block IDs and document-relative
    positions.

    The ModelScope tensor-core kernels are retained, but their uint8 [T,T/R]
    bitmap is replaced with packed bits and tile routes built from the same
    IDs. This preserves packed samples whose starts are not aligned to
    ``block_size``.
    """
    total = q.shape[0]
    if total != k.shape[0] or total != v.shape[0] or block_ids.shape[0] != total:
        raise ValueError("QSA tensor and route lengths must match")
    if total > 2**31 - 1 or q.shape[-1] & (q.shape[-1] - 1):
        raise ValueError("QSA kernel requires int32 indexing and a power-of-two head width")
    if block_size < 1 or block_size > 32 or 32 % block_size:
        raise ValueError("QSA selected kernel requires a block size dividing 32")
    if block_ids.dtype != torch.int32 or positions.dtype != torch.int32:
        raise ValueError("QSA routes and positions must be int32")
    if not all(item.is_cuda for item in (q, k, v, block_ids, positions)):
        raise ValueError("QSA tensor-core candidate requires CUDA tensors")
    ids = block_ids.contiguous()
    pos = positions.contiguous().view(-1)
    query = torch.arange(total, device=q.device, dtype=torch.int32)
    doc_start = query - pos
    packed = pack_selected_bits(ids, total, block_size)
    tile_fwd = build_tile_map(ids, pos, block_size, 64)
    tile_bwd = build_tile_map(ids, pos, block_size, 32)
    zeros = torch.zeros_like(query)
    return qsa_block_sparse_attention_triton(
        q,
        k,
        v,
        packed,
        doc_start,
        query,
        zeros,
        doc_start,
        tile_fwd,
        tile_bwd,
        scale,
        block_size,
    )
