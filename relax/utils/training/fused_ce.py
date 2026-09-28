# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in fused CE paths that preserve native heads and main_grad.

Set ``RELAX_LOGPROB_CE_IMPL=te`` for the main log-probability path and/or
``RELAX_MTP_CE_IMPL=te`` for chunked MTP loss. Both default to ``native``.
"""

import os
from functools import lru_cache

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def ce_implementation(branch: str) -> str:
    value = os.environ.get(f"RELAX_{branch}_CE_IMPL", "native")
    if value not in ("native", "te"):
        raise ValueError(f"Unsupported RELAX_{branch}_CE_IMPL={value!r}")
    return value


@lru_cache(maxsize=1)
def _te_ce():
    from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

    logger.info("[fused-ce] TE Triton CE enabled; FP32 loss with vocab-parallel support")
    return parallel_cross_entropy


def te_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, tp_group=None) -> torch.Tensor:
    """Unreduced fused CE for local or vocabulary-parallel logits.

    Transformer Engine expects a three-dimensional input, so flatten arbitrary
    leading token dimensions to ``[1, tokens, vocab/TP]`` and restore the label
    shape afterwards. Targets remain global vocabulary IDs; TE uses
    ``tp_group`` to combine the per-rank online-softmax statistics.
    """
    if logits.ndim < 2 or labels.shape != logits.shape[:-1]:
        raise ValueError("TE CE requires labels to match every logits dimension except the vocabulary dimension")
    # Singleton B=1 transposes can be contiguous while retaining label stride
    # (1, full_sequence_length). TE uses target.stride(-1) to walk all rows,
    # so normalize to [1, rows, vocab] / [1, rows] explicitly.
    loss = _te_ce()(
        logits.reshape(1, -1, logits.shape[-1]).contiguous(),
        labels.contiguous().reshape(1, -1),
        reduce_loss=False,
        dist_process_group=tp_group,
        is_cg_capturable=True,
    )
    return loss.reshape(labels.shape)
