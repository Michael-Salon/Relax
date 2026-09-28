# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Runtime Megatron monkey-patches applied for their import side effects. Mirrors
# the slime ``megatron_utils/megatron_patch`` package: importing this subpackage
# swaps patched implementations into ``megatron.core`` before any training step.
import os

from . import (
    chunked_grad_coalesce_patch,  # noqa: F401
    chunked_mtp_loss_patch,  # noqa: F401
)


if os.environ.get("QWEN48_PLE_STREAMED") == "1":
    from . import ple_stream_patch  # noqa: F401

if os.environ.get("QWEN48_QSA_SPARSE_BLOCKMASK") == "1":
    from . import qsa_block_mask_patch  # noqa: F401

if os.environ.get("QWEN48_QSA_SELECTED_TRITON") == "1":
    from . import qsa_selected_patch  # noqa: F401
