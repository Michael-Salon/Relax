# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[3] / "relax/backends/megatron/megatron_patch/gdn_cu_seqlens_cache.py"
_SPEC = importlib.util.spec_from_file_location("gdn_cu_seqlens_cache", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
patch_gdn_cu_seqlens_validation = _MODULE.patch_gdn_cu_seqlens_validation


def test_gdn_cu_seqlens_validation_reuses_only_unchanged_success():
    class GDN:
        calls = 0

        def _resolve_cu_seqlens(self, padded, actual, total_seq_len, name, cp_size=1):
            self.calls += 1
            seq = padded if padded is not None else actual
            if seq[-1].item() != total_seq_len or bool(((seq[1:] - seq[:-1]) % cp_size).any()):
                raise ValueError(name)
            return seq

    assert patch_gdn_cu_seqlens_validation(GDN)
    assert not patch_gdn_cu_seqlens_validation(GDN)
    first, second = GDN(), GDN()
    seq = torch.tensor([0, 4, 8])
    assert first._resolve_cu_seqlens(None, seq, 8, "q", cp_size=2) is seq
    assert second._resolve_cu_seqlens(None, seq, 8, "kv", cp_size=2) is seq
    assert first.calls + second.calls == 1

    # A changed expected length is checked again and a failure is not cached.
    with pytest.raises(ValueError):
        first._resolve_cu_seqlens(None, seq, 7, "q", cp_size=2)
    with pytest.raises(ValueError):
        first._resolve_cu_seqlens(None, seq, 7, "q", cp_size=2)
    assert first.calls + second.calls == 3

    # CP divisibility and in-place changes both invalidate the prior success.
    with pytest.raises(ValueError):
        first._resolve_cu_seqlens(None, seq, 8, "q", cp_size=3)
    seq[1] = 3
    with pytest.raises(ValueError):
        first._resolve_cu_seqlens(None, seq, 8, "q", cp_size=2)
    seq[1] = 4
    assert first._resolve_cu_seqlens(None, seq, 8, "q", cp_size=2) is seq
    assert first.calls + second.calls == 6

    # A new microbatch tensor with the same contents still validates once.
    next_seq = seq.clone()
    assert first._resolve_cu_seqlens(None, next_seq, 8, "q", cp_size=2) is next_seq
    assert first.calls + second.calls == 7

    class GDN2(GDN):
        pass

    assert GDN2()._resolve_cu_seqlens(None, next_seq, 8, "q", cp_size=2) is next_seq
    assert first.calls + second.calls == 7

    # The padded tensor is the validated/cache-selected object when both exist.
    other_actual = torch.tensor([0, 3, 8])
    assert first._resolve_cu_seqlens(next_seq, other_actual, 8, "q", cp_size=2) is next_seq
    assert first.calls + second.calls == 7


def test_gdn_cu_seqlens_cache_bypasses_inference_tensors():
    class GDN:
        calls = 0

        def _resolve_cu_seqlens(self, padded, actual, total_seq_len, name, cp_size=1):
            self.calls += 1
            return padded if padded is not None else actual

    patch_gdn_cu_seqlens_validation(GDN)
    layer = GDN()
    with torch.inference_mode():
        seq = torch.tensor([0, 4])
        assert layer._resolve_cu_seqlens(None, seq, 4, "q") is seq
        assert layer._resolve_cu_seqlens(None, seq, 4, "kv") is seq
    assert layer.calls == 2
