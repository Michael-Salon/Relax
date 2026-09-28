# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Avoid repeated packed GDN sequence-length validation on unchanged
tensors."""

import torch


def patch_gdn_cu_seqlens_validation(gdn_class: type) -> bool:
    """Cache a successful validation for one tensor/version and layout.

    The cache retains its tensor, so a later tensor cannot reuse its identity.
    Tensor version invalidates the result after an in-place edit. Errors are
    never cached. Validation still runs once for each fresh microbatch tensor.
    """
    if getattr(gdn_class, "_relax_cu_seqlens_memoized", False):
        return False
    original = getattr(gdn_class, "_resolve_cu_seqlens", None)
    if original is None:
        return False

    last: tuple[torch.Tensor, int, int, int] | None = None

    def memoized(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1):
        nonlocal last
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual
        if not isinstance(cu_seqlens, torch.Tensor):
            return original(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=cp_size)
        try:
            version = cu_seqlens._version
        except RuntimeError:
            # Inference-mode tensors do not expose version counters.
            return original(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=cp_size)

        cached = last
        if cached is not None and cached[0] is cu_seqlens and cached[1:] == (version, total_seq_len, cp_size):
            return cu_seqlens

        resolved = original(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size=cp_size)
        last = (cu_seqlens, version, total_seq_len, cp_size)
        return resolved

    gdn_class._resolve_cu_seqlens = memoized
    gdn_class._relax_cu_seqlens_memoized = True
    return True


def install_gdn_cu_seqlens_cache() -> bool:
    """Patch the shared GDN base, covering both GDN variants when available."""
    try:
        from megatron.core.ssm.gated_delta_net.common import _GDNBase
    except ImportError:
        return False
    return patch_gdn_cu_seqlens_validation(_GDNBase)
