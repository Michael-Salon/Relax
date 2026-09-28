# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Distributed parity test for the context-parallel gather/split utilities
(``reconstruct_tensor_cp`` / ``split_tensor_cp``) added to the candidate MCore
pin.

The Qwen4Exp CP port (PLE causal conv + QSA indexer block selection) rests on these two
helpers, so they are the highest-risk new code and the cheapest to verify in isolation.
The check runs on a 2-process (CP=2, TP=1, sequence-parallel off) gloo/CPU group:

1. ``reconstruct_tensor_cp`` gathers every rank's zigzag-local slice into the full
   sequential sequence on every rank — parity with the CP=1 full (so a module that ran on
   the full sequence under CP=1 produces the same output under CP>1).
2. ``split_tensor_cp`` is the round-trip inverse: ``split(reconstruct(x)) == x``.
3. The local autograd graph survives the all-gather: a loss on the split output backprops
   a finite gradient into the local input (the local slot is re-pinned, not detached).

Run::

    torchrun --nproc_per_node=2 \\
        tests/backends/megatron/test_cp_reconstruct_split.py

The module pins ``import megatron`` to the candidate MCore submodule (see the path setup
below), so no ``PYTHONPATH`` is needed and a host venv's editable ``megatron-core`` cannot
shadow the CP port under test.
"""

import os
import sys

import torch
import torch.distributed as dist


# This test verifies code in the candidate MCore submodule
# (``deps/Megatron-Bridge-qwen4exp/3rdparty/Megatron-LM``), which carries the CP port. A
# host venv may ship an older ``megatron-core`` installed editable from a different source
# tree; its meta-path finder shadows ``PYTHONPATH`` and would hide the CP port. Pin the
# import to the submodule's megatron so the test exercises the code it is meant to verify,
# regardless of how the host venv is provisioned.
_HERE = os.path.dirname(os.path.abspath(__file__))
# The candidate MCore submodule lives at
# ``deps/Megatron-Bridge-qwen4exp/3rdparty/Megatron-LM`` — a sibling of the Relax repo
# root, not inside it — so walk up from this file until that path is found (robust to
# where the test lives) rather than hard-coding a relative depth.
_SUBMODULE_ROOT = None
_candidate = _HERE
for _ in range(8):
    _trial = os.path.join(_candidate, "deps", "Megatron-Bridge-qwen4exp", "3rdparty", "Megatron-LM")
    if os.path.isdir(os.path.join(_trial, "megatron")):
        _SUBMODULE_ROOT = os.path.normpath(_trial)
        break
    _parent = os.path.dirname(_candidate)
    if _parent == _candidate:
        break
    _candidate = _parent
if _SUBMODULE_ROOT is None:
    raise RuntimeError(
        "Could not locate the candidate MCore submodule "
        "(deps/Megatron-Bridge-qwen4exp/3rdparty/Megatron-LM) by walking up from " + _HERE
    )
# Pin the submodule's megatron ahead of any host-installed megatron-core so the
# test exercises the CP port under test. Remove the editable ``megatron-core``
# meta-path finder and its path-hook placeholder (either would redirect
# ``import megatron`` to the editable source tree, bypassing sys.path order),
# then prepend the submodule root so it shadows any pip-installed
# ``megatron-core`` that also lives in site-packages. Do NOT drop other sys.path
# dirs merely because they ship a ``megatron/`` namespace: with the editable
# redirect gone and the submodule at index 0 the submodule wins for ``megatron``,
# while dropping those dirs (e.g. site-packages) would also hide unrelated
# runtime deps such as ``packaging`` that the ``megatron.core`` import chain
# transitively needs (dist_checkpointing.strategies.torch imports packaging).
sys.meta_path = [f for f in sys.meta_path if "__editable___megatron_core" not in getattr(f, "__module__", "")]
sys.path = [p for p in sys.path if not p.startswith("__editable__.megatron_core")]
sys.path.insert(0, _SUBMODULE_ROOT)
for _name in list(sys.modules):
    if _name == "megatron" or _name.startswith("megatron."):
        del sys.modules[_name]


def _init():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=world,
        expert_tensor_parallel_size=1,
    )
    return parallel_state


def main():
    ps = _init()
    from megatron.core.ssm.mamba_context_parallel import (
        _redo_attention_load_balancing,
        reconstruct_tensor_cp,
        split_tensor_cp,
    )

    rank = ps.get_context_parallel_rank()
    cp_size = ps.get_context_parallel_world_size()
    s_local, b, d = 4, 1, 3
    s_full = s_local * cp_size

    # Build a known global "full" (sequential) identically on every rank, then derive each
    # rank's zigzag-local slice the way the model's CP split would. This lets reconstruct be
    # checked against the *true* full (parity), not merely a round-trip.
    torch.manual_seed(123)
    full = torch.randn(s_full, b, d)
    load_balanced = _redo_attention_load_balancing(full, cp_size, None)
    local_x = load_balanced[rank * s_local : (rank + 1) * s_local].clone().requires_grad_(True)

    # 1. reconstruct -> every rank holds the full sequential sequence (parity vs CP=1 full).
    gathered = reconstruct_tensor_cp(local_x, None, dim=0)
    assert gathered.shape[0] == s_full, f"gathered seq {gathered.shape[0]} != {s_full}"
    max_err = (gathered - full).abs().max().item()
    assert max_err < 1e-5, f"reconstruct parity failed on rank {rank}: max_err={max_err}"
    if rank == 0:
        print(f"[reconstruct] parity OK (max_err={max_err:.2e})")

    # 2. split is the round-trip inverse.
    back = split_tensor_cp(gathered, None, dim=0)
    assert back.shape[0] == s_local, f"split seq {back.shape[0]} != {s_local}"
    rt_err = (back - local_x).abs().max().item()
    assert rt_err < 1e-5, f"round-trip failed on rank {rank}: rt_err={rt_err}"
    if rank == 0:
        print(f"[split round-trip] OK (rt_err={rt_err:.2e})")

    # 3. backward reaches the local tensor (autograd preserved through the all-gather).
    loss = back.pow(2).sum()
    loss.backward()
    assert local_x.grad is not None, f"rank {rank}: local_x.grad is None"
    assert torch.isfinite(local_x.grad).all(), f"rank {rank}: local_x.grad not finite"
    if rank == 0:
        print(f"[backward] local_x.grad finite OK (max={local_x.grad.abs().max().item():.2e})")

    if rank == 0:
        print("ALL CHECKS PASSED")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
