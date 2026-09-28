# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the chunked-MTP-loss patch
(megatron_patch/chunked_mtp_loss_patch.py).

The patch computes each MTP prediction depth's head + cross-entropy in sequence
chunks so the full ``[S, V/TP]`` per-depth logits never materialize (fixes the
tail-stage OOM). Because ``compute_language_model_loss`` reduces per token, chunking
the sequence and concatenating the per-token ``[b, chunk]`` losses must reconstruct
the full ``[b, s]`` loss BIT-FOR-BIT. A trainable output-head weight can still see
floating-point accumulation-order differences in its gradient across chunks; these
tests intentionally assert the loss and hidden-gradient input contract.

These tests drive the patched ``process_mtp_loss`` against the ORIGINAL upstream one
with mocked head (a plain Linear) + stub per-token CE, and assert the per-depth loss
tensor fed to ``MTPLossAutoScaler.apply`` is identical. Pure CPU, no distributed init.
"""

from __future__ import annotations

import types
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from megatron.core.transformer import multi_token_prediction as _mtp_mod

    from relax.backends.megatron.megatron_patch import chunked_mtp_loss_patch as _patch
except Exception as exc:  # pragma: no cover
    pytest.skip(f"megatron/relax unavailable: {exc}", allow_module_level=True)


S, B, H, V, LAYERS = 16, 2, 8, 32, 2


@pytest.fixture(scope="module", autouse=True)
def _single_threaded_cpu() -> Iterator[None]:
    """Run bitwise CPU comparisons with one thread, then restore the previous
    setting."""
    previous_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_num_threads)


class _FakeHead(nn.Module):
    """Stand-in for ColumnParallelLinear output_layer.

    forward returns (logits, None) matching the (logits, bias) contract. The
    patch calls ``type(output_layer).forward(output_layer, x, weight=,
    runtime_gather_output=)``, so this must accept those kwargs.
    """

    def __init__(self, hidden, vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden, dtype=torch.float32))

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = weight if weight is not None else self.weight
        return input_ @ w.t(), None  # [.., vocab]


def _stub_clml(labels, logits):
    """Per-token CE: labels [b, s], logits [s, b, V] -> loss [b, s]."""
    s, b, v = logits.shape
    lg = logits.transpose(0, 1).reshape(b * s, v)
    lb = labels.reshape(b * s)
    return F.cross_entropy(lg, lb, reduction="none").reshape(b, s)


def _make_config(calculate_per_token_loss=False, fuse_linear=False):
    return types.SimpleNamespace(
        mtp_num_layers=LAYERS,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=calculate_per_token_loss,
        cross_entropy_loss_fusion=fuse_linear,
        cross_entropy_fusion_impl="linear" if fuse_linear else "native",
        mtp_detach_heads=False,
    )


def _run_and_capture(fn, *, head, config, monkeypatch, chunk_size=None, scale_logits_fn=None, is_training=False):
    """Run a process_mtp_loss impl, spying on MTPLossAutoScaler.apply to
    capture the per-depth loss tensors it would send to backward.

    Returns (returned_hidden, [losses]).
    """
    torch.manual_seed(0)
    hidden = torch.randn(S * (1 + LAYERS), B, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, S))
    loss_mask = (torch.rand(B, S) > 0.3).to(torch.float32)

    captured = []

    class _SpyAutoScaler:
        @staticmethod
        def apply(hs, loss):
            captured.append(loss.detach().clone())
            return hs

    # Patch MTPLossAutoScaler in BOTH the source module and the patch module's binding.
    monkeypatch.setattr(_mtp_mod, "MTPLossAutoScaler", _SpyAutoScaler)
    monkeypatch.setattr(_patch, "MTPLossAutoScaler", _SpyAutoScaler, raising=False)
    if chunk_size is not None:
        monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: chunk_size)
    # force gate ON for the patched fn (no get_args() in-process)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)

    ret = fn(
        hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        is_training,
        _stub_clml,
        config,
        cp_group=None,
        packed_seq_params=None,
        scale_logits_fn=scale_logits_fn,
    )
    return ret, captured


@pytest.mark.parametrize("chunk_size", [S, 2, 3, 7])
def test_chunked_mtp_matches_original(chunk_size, monkeypatch):
    """Chunked head+CE == full head+CE, bit-identical (fp32), for every chunk
    size."""
    head = _FakeHead(H, V)
    cfg = _make_config()

    # original (unchunked): the captured upstream impl. Force chunk irrelevant.
    _, ref = _run_and_capture(_patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch)
    # patched (chunked) at this chunk size.
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=chunk_size
    )

    assert len(got) == len(ref) == LAYERS
    for d, (g, r) in enumerate(zip(got, ref)):
        assert torch.equal(g, r), f"depth {d} chunk={chunk_size}: max|Δ|={(g - r).abs().max()}"


def test_scale_logits_fn_per_chunk(monkeypatch):
    """A per-token scalar scale_logits_fn is chunk-invariant."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    sfn = lambda x: 0.5 * x  # noqa: E731
    _, ref = _run_and_capture(
        _patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch, scale_logits_fn=sfn
    )
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3, scale_logits_fn=sfn
    )
    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_te_ce_runs_after_acceptance_metric(monkeypatch):
    """Acceptance observes logits before TE CE consumes them in place."""
    events = []

    def acceptance(logits, *_args, **_kwargs):
        events.append(("acceptance", logits.detach().clone()))
        return torch.tensor(1), torch.tensor(2)

    def destructive_te_ce(logits, labels, _group):
        events.append(("ce", logits.detach().clone()))
        with torch.no_grad():
            logits.zero_()
        return logits.new_zeros(labels.shape)

    monkeypatch.setattr(_patch, "_COMPUTE_MTP_ACCEPTANCE_COUNTS", acceptance)
    monkeypatch.setattr(_patch, "ce_implementation", lambda branch: "te")
    monkeypatch.setattr(_patch, "te_cross_entropy", destructive_te_ce)

    hidden = torch.randn(5, B, H)
    labels = torch.randint(0, V, (B, 5))
    loss, correct, total = _patch._chunked_head_ce(
        hidden,
        labels,
        _FakeHead(H, V),
        None,
        None,
        _stub_clml,
        None,
        chunk_size=2,
        tp_group=object(),
        loss_mask=torch.ones_like(labels),
    )

    assert [kind for kind, _ in events] == ["acceptance", "ce"] * 3
    for acceptance_event, ce_event in zip(events[::2], events[1::2]):
        torch.testing.assert_close(acceptance_event[1], ce_event[1])
        assert torch.count_nonzero(acceptance_event[1]) > 0
    assert loss.shape == labels.shape
    assert correct.item() == 3
    assert total.item() == 6


def test_calculate_per_token_loss_branch(monkeypatch):
    """The per-token-loss normalization branch is preserved under chunking."""
    head = _FakeHead(H, V)
    cfg = _make_config(calculate_per_token_loss=True)
    _, ref = _run_and_capture(_patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch)
    _, got = _run_and_capture(_patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3)
    for g, r in zip(got, ref):
        assert torch.equal(g, r)


def test_main_update_tokens_uses_one_denominator_across_microbatches(monkeypatch):
    """Unequal MTP masks must give every valid NLL the same update-wide
    coefficient."""
    monkeypatch.setattr(_patch, "_main_update_token_normalization_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)

    def roll(tensor, **_kwargs):
        shifted = torch.roll(tensor, shifts=-1, dims=-1)
        return shifted, shifted.sum()

    monkeypatch.setattr(_patch, "roll_tensor", roll)
    config = _make_config(calculate_per_token_loss=True)
    config.mtp_num_layers = 1
    config.mtp_loss_scaling_factor = 0.1
    nlls = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True),
        torch.tensor([[5.0, 6.0, 7.0, 8.0]], requires_grad=True),
    ]
    masks = [torch.tensor([[1.0, 1.0, 1.0, 0.0]]), torch.tensor([[1.0, 1.0, 1.0, 1.0]])]
    mtp_input_masks = [torch.tensor([[1, 0, 0, 0]], dtype=torch.bool), torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)]
    captured = []

    class SpyAutoScaler:
        @staticmethod
        def apply(hidden, loss):
            captured.append(loss)
            return hidden

    monkeypatch.setattr(_patch, "MTPLossAutoScaler", SpyAutoScaler)
    for nll, mask, mtp_input_mask in zip(nlls, masks, mtp_input_masks):
        monkeypatch.setattr(_patch, "_chunked_head_ce", lambda *_args, nll=nll, **_kwargs: (nll, None, None))
        _patch.process_mtp_loss(
            torch.zeros(8, 1, 2),
            torch.zeros(1, 4, dtype=torch.long),
            mask,
            _FakeHead(2, 8),
            None,
            None,
            False,
            _stub_clml,
            config,
            mtp_input_mask=mtp_input_mask,
        )

    main_update_tokens = sum(mask.sum() for mask in masks)
    mtp_masks = [
        torch.roll(mask, -1, dims=-1) & torch.roll(mtp_input_mask, -1, dims=-1)
        for mask, mtp_input_mask in zip((mask.bool() for mask in masks), mtp_input_masks)
    ]
    assert sum(mask.sum() for mask in mtp_masks) < main_update_tokens
    aligned_loss = sum(value.sum() for value in captured) / main_update_tokens
    expected = 0.1 * sum((nll * mask).sum() for nll, mask in zip(nlls, mtp_masks)) / main_update_tokens
    torch.testing.assert_close(aligned_loss, expected)
    aligned_loss.backward()
    for nll, mask in zip(nlls, mtp_masks):
        torch.testing.assert_close(nll.grad, 0.1 * mask / main_update_tokens)


def test_main_update_tokens_rejects_fused_delegate(monkeypatch):
    monkeypatch.setattr(_patch, "_main_update_token_normalization_enabled", lambda: True)
    config = _make_config(calculate_per_token_loss=True, fuse_linear=True)
    config.mtp_num_layers = 1
    with pytest.raises(ValueError, match="native chunked CE"):
        _patch.process_mtp_loss(
            torch.zeros(8, 1, 2),
            torch.zeros(1, 4, dtype=torch.long),
            torch.ones(1, 4),
            _FakeHead(2, 8),
            None,
            None,
            False,
            _stub_clml,
            config,
        )


@pytest.mark.parametrize("detach_lm_head", [False, True])
def test_chunked_mtp_respects_lm_head_detach(monkeypatch, detach_lm_head):
    """The chunked path honors the independent LM-head gradient setting."""
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: 3)
    config = _make_config()
    config.mtp_detach_lm_head = detach_lm_head
    head = _FakeHead(H, V)
    hidden = torch.randn(S * (1 + LAYERS), B, H, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)

    output = _patch.process_mtp_loss(
        hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        False,
        _stub_clml,
        config,
    )
    output.sum().backward()

    assert (head.weight.grad is None) is detach_lm_head


def test_is_training_logging_matches(monkeypatch):
    """is_training path: both the per-depth GRADIENT tensor
    (MTPLossAutoScaler.apply) AND the LOGGED per-depth loss
    (save_loss_to_tracker's first arg) must match upstream.

    Spies on save_loss_to_tracker so a divergence in mtp_loss_for_log (e.g.
    dropping the mtp_loss_scale factor from the log-only value) is caught — the
    gradient path alone would still match and hide it (regression: chunked
    patch logged the UNSCALED loss).
    """
    head = _FakeHead(H, V)
    cfg = _make_config()

    logged = {"ref": [], "got": []}

    def _spy_tracker(dst):
        return staticmethod(lambda loss, *a, **k: dst.append(loss.detach().clone()))

    monkeypatch.setattr(_mtp_mod.parallel_state, "get_data_parallel_group", lambda **k: None)
    monkeypatch.setattr(_patch.parallel_state, "get_data_parallel_group", lambda **k: None, raising=False)

    tracker_method_name = _patch._MTP_TRACKER_METHOD_NAME
    monkeypatch.setattr(
        _mtp_mod.MTPLossLoggingHelper,
        tracker_method_name,
        _spy_tracker(logged["ref"]),
    )
    monkeypatch.setattr(
        _patch.MTPLossLoggingHelper,
        tracker_method_name,
        _spy_tracker(logged["ref"]),
    )
    _, ref = _run_and_capture(
        _patch._ORIG_PROCESS_MTP_LOSS, head=head, config=cfg, monkeypatch=monkeypatch, is_training=True
    )
    monkeypatch.setattr(
        _mtp_mod.MTPLossLoggingHelper,
        tracker_method_name,
        _spy_tracker(logged["got"]),
    )
    monkeypatch.setattr(
        _patch.MTPLossLoggingHelper,
        tracker_method_name,
        _spy_tracker(logged["got"]),
    )
    _, got = _run_and_capture(
        _patch.process_mtp_loss, head=head, config=cfg, monkeypatch=monkeypatch, chunk_size=3, is_training=True
    )
    for g, r in zip(got, ref):
        assert torch.equal(g, r), "gradient (MTPLossAutoScaler.apply) tensor mismatch"
    assert len(logged["got"]) == len(logged["ref"]) == LAYERS
    for d, (g, r) in enumerate(zip(logged["got"], logged["ref"])):
        assert torch.equal(g, r), f"logged mtp_loss_for_log mismatch at depth {d}: {g} vs {r}"


def test_token_aware_tracker_api_receives_scaled_sum_and_count(monkeypatch):
    """New Megatron tracker APIs receive the raw scaled sum and token count."""
    seen = {}

    def tracker(loss_sum, num_tokens, layer_number, num_layers, **kwargs):
        seen.update(
            loss_sum=loss_sum,
            num_tokens=num_tokens,
            layer_number=layer_number,
            num_layers=num_layers,
            kwargs=kwargs,
        )

    monkeypatch.setattr(
        _patch,
        "_MTP_TRACKER_PARAMETERS",
        {"loss_sum", "num_tokens", "layer_number", "num_layers", "correct", "total"},
    )
    monkeypatch.setattr(_patch, "_MTP_TRACKER_METHOD_NAME", "save_loss_to_tracker")
    monkeypatch.setattr(
        _patch.MTPLossLoggingHelper,
        "save_loss_to_tracker",
        staticmethod(tracker),
        raising=False,
    )
    monkeypatch.setattr(_patch.parallel_state, "get_data_parallel_group", lambda **_kwargs: None)
    config = _make_config()
    loss = torch.tensor([[1.0, 2.0]])
    num_tokens = torch.tensor(2.0)
    correct = torch.tensor(1.0)
    total = torch.tensor(2.0)

    _patch._save_mtp_loss_to_tracker(loss, num_tokens, 0, config, correct, total)

    expected_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
    assert torch.equal(seen["loss_sum"], expected_scale * loss.sum())
    assert seen["num_tokens"] is num_tokens
    assert seen["layer_number"] == 0
    assert seen["num_layers"] == config.mtp_num_layers
    assert seen["kwargs"]["correct"] is correct
    assert seen["kwargs"]["total"] is total


def test_metrics_tracker_api_receives_normalized_loss_and_acceptance_counts(monkeypatch):
    """Renamed MCore metrics API receives normalized loss and acceptance
    counts."""
    seen = {}

    def tracker(loss, correct, total, layer_number, num_layers, **kwargs):
        seen.update(
            loss=loss,
            correct=correct,
            total=total,
            layer_number=layer_number,
            num_layers=num_layers,
            kwargs=kwargs,
        )

    monkeypatch.setattr(_patch, "_MTP_TRACKER_METHOD_NAME", "save_metrics_to_tracker")
    monkeypatch.setattr(_patch.MTPLossLoggingHelper, "save_metrics_to_tracker", staticmethod(tracker), raising=False)
    monkeypatch.setattr(_patch.parallel_state, "get_data_parallel_group", lambda **_kwargs: None)
    config = _make_config()
    loss = torch.tensor([[1.0, 2.0]])
    num_tokens = torch.tensor(2.0)
    correct = torch.tensor(1.0)
    total = torch.tensor(2.0)
    metric_avg_group = object()

    _patch._save_mtp_loss_to_tracker(
        loss,
        num_tokens,
        0,
        config,
        correct,
        total,
        metric_avg_group=metric_avg_group,
    )

    assert torch.equal(seen["loss"], loss.sum() / num_tokens)
    assert seen["correct"] is correct
    assert seen["total"] is total
    assert seen["layer_number"] == 0
    assert seen["num_layers"] == config.mtp_num_layers
    assert seen["kwargs"]["avg_group"] is metric_avg_group


def test_main_hidden_states_is_preserved_on_chunked_path(monkeypatch):
    """Hybrid models return their separate main hidden tensor after MTP
    loss."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    main_hidden = torch.randn(S, B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)

    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: 3)

    got = _patch.process_mtp_loss(
        hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
        main_hidden_states=main_hidden,
    )

    assert torch.equal(got, main_hidden)


def test_mtp_input_mask_shape_is_validated(monkeypatch):
    """The chunked path preserves the upstream mask-shape contract."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)
    bad_mtp_input_mask = torch.ones(B, S - 1, dtype=torch.bool)

    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)

    with pytest.raises(AssertionError, match="must match loss_mask shape"):
        _patch.process_mtp_loss(
            hidden,
            labels,
            loss_mask,
            head,
            None,
            None,
            False,
            _stub_clml,
            cfg,
            mtp_input_mask=bad_mtp_input_mask,
        )


def test_input_ids_derive_labels_and_boundary_mask(monkeypatch):
    """The chunked RL path derives next-token labels and masks the boundary."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    torch.manual_seed(0)
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    input_ids = torch.randint(0, V, (B, S))
    loss_mask = (torch.rand(B, S) > 0.3).to(torch.float32)

    captured = {"derived": [], "explicit": []}

    class _SpyAutoScaler:
        target = "derived"

        @staticmethod
        def apply(hidden_states, loss):
            captured[_SpyAutoScaler.target].append(loss.detach().clone())
            return hidden_states

    monkeypatch.setattr(_patch, "MTPLossAutoScaler", _SpyAutoScaler, raising=False)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: 3)
    monkeypatch.setattr(
        _patch,
        "_ORIG_PROCESS_MTP_LOSS",
        lambda *_args, **_kwargs: pytest.fail("labels=None must stay on the chunked path when input_ids are present"),
    )

    _patch.process_mtp_loss(
        hidden,
        None,
        loss_mask.clone(),
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
        input_ids=input_ids,
    )

    labels, _ = _patch._roll_tensor_without_sum(input_ids, shifts=-1, dims=-1)
    explicit_mask, _ = _patch._roll_tensor_without_sum(loss_mask, shifts=-1, dims=-1)
    _SpyAutoScaler.target = "explicit"
    _patch.process_mtp_loss(
        hidden,
        labels,
        explicit_mask,
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    assert len(captured["derived"]) == len(captured["explicit"]) == LAYERS
    for derived, explicit in zip(captured["derived"], captured["explicit"], strict=True):
        assert torch.equal(derived, explicit)


def test_old_mcore_labels_none_delegates_without_chunking(monkeypatch):
    """Old MCore returns its untouched hidden tensor when labels are absent."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    hidden = torch.randn(S, B, H)
    seen = {}

    def original(*args, **kwargs):
        seen["hidden"] = args[0]
        seen["kwargs"] = kwargs
        return args[0]

    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", original)
    monkeypatch.setattr(
        _patch,
        "_ORIG_PROCESS_MTP_LOSS_PARAMETERS",
        {
            "hidden_states",
            "labels",
            "loss_mask",
            "output_layer",
            "output_weight",
            "runtime_gather_output",
            "is_training",
            "compute_language_model_loss",
            "config",
        },
    )

    got = _patch.process_mtp_loss(
        hidden,
        None,
        None,
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    assert got is hidden
    assert seen["hidden"] is hidden
    assert seen["kwargs"] == {}


def test_fuse_linear_delegates(monkeypatch):
    """fuse_linear_cross_entropy=True delegates to the original (never
    chunks)."""
    head = _FakeHead(H, V)
    cfg = _make_config(fuse_linear=True)
    called = {"orig": 0, "args": None}

    # Pure spy (does NOT call the real upstream body — the fused Blackwell path can't
    # run with a plain mock head). We only assert the patch DELEGATED without chunking.
    def spy_orig(*a, **k):
        called["orig"] += 1
        called["args"] = (a, k)
        return a[0]  # return hidden_states unchanged

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", spy_orig)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)
    _patch.process_mtp_loss(hidden, labels, loss_mask, head, None, None, False, _stub_clml, cfg)
    assert called["orig"] == 1, "fuse_linear=True must delegate to the original, not chunk"


def test_gate_off_delegates(monkeypatch):
    """When the chunked-MTP gate is off, the patched fn delegates to the
    original."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    called = {"orig": 0}
    orig = _patch._ORIG_PROCESS_MTP_LOSS

    def spy_orig(*a, **k):
        called["orig"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", spy_orig)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: False)
    monkeypatch.setattr(_mtp_mod, "MTPLossAutoScaler", types.SimpleNamespace(apply=staticmethod(lambda hs, loss: hs)))
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    loss_mask = torch.ones(B, S)
    _patch.process_mtp_loss(hidden, labels, loss_mask, head, None, None, False, _stub_clml, cfg)
    assert called["orig"] == 1


def test_new_megatron_kwargs_are_accepted_and_version_filtered(monkeypatch):
    """Newer Megatron callers pass tp_group/input_ids; old originals must still
    work."""
    head = _FakeHead(H, V)
    cfg = _make_config()
    called = {}

    def old_style_orig(*args, cp_group=None, packed_seq_params=None, scale_logits_fn=None):
        called.update(
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            scale_logits_fn=scale_logits_fn,
        )
        return args[0]

    monkeypatch.setattr(_patch, "_ORIG_PROCESS_MTP_LOSS", old_style_orig)
    monkeypatch.setattr(
        _patch,
        "_ORIG_PROCESS_MTP_LOSS_PARAMETERS",
        {"cp_group", "packed_seq_params", "scale_logits_fn"},
    )
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: False)
    hidden = torch.randn(S * (1 + LAYERS), B, H)
    labels = torch.randint(0, V, (B, S))
    result = _patch.process_mtp_loss(
        hidden,
        labels,
        torch.ones(B, S),
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
        tp_group=object(),
        input_ids=torch.ones(B, S, dtype=torch.long),
    )
    assert result is hidden
    assert called == {"cp_group": None, "packed_seq_params": None, "scale_logits_fn": None}


def test_sequence_parallel_gathers_hidden_once(monkeypatch):
    """With sequence_parallel=True on the head, the patch must gather the SP-
    scattered hidden to the full sequence ONCE (so chunk boundaries align with
    the full-sequence labels), run the head with SP disabled per chunk, then
    restore the flag — mirroring relax model.py _bypass_output_layer.

    Regression for the on-cluster IndexError ('shapes [2048], [1024]'): before
    the fix the patch chunked the SP-scattered input and let the
    ColumnParallelLinear all-gather each chunk (×TP), so logits seq (chunk×TP)
    no longer matched the label slice (chunk). Here the head's own forward
    would NOT gather (it's a plain matmul), so we assert the behavioral
    contract: (1) gather_from_sequence_parallel_region is invoked exactly once
    per depth, (2) sequence_parallel is False during the head calls, (3) it is
    restored to True afterwards, and (4) the per-depth loss equals the
    unchunked reference.
    """
    torch.manual_seed(0)
    head = _FakeHead(H, V)
    head.sequence_parallel = True
    head.tp_group = object()  # truthy sentinel so tp_group resolution skips parallel_state

    cfg = _make_config()
    full_hidden = torch.randn(S * (1 + LAYERS), B, H, dtype=torch.float32)
    labels = torch.randint(0, V, (B, S))
    loss_mask = (torch.rand(B, S) > 0.3).to(torch.float32)

    # Reference: unchunked upstream on the FULL hidden with SP off (no gather, matmul only).
    ref_head = _FakeHead(H, V)
    ref_head.load_state_dict(head.state_dict())
    captured_ref = []
    monkeypatch.setattr(
        _mtp_mod,
        "MTPLossAutoScaler",
        types.SimpleNamespace(apply=staticmethod(lambda hs, loss: captured_ref.append(loss.detach().clone()) or hs)),
    )
    _patch._ORIG_PROCESS_MTP_LOSS(
        full_hidden,
        labels,
        loss_mask,
        ref_head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    # Stub the SP gather as identity (the test feeds full-seq hidden directly) and count calls
    # + observe the sequence_parallel flag at head-call time.
    gather_calls = {"n": 0}
    sp_during_head = []

    def _fake_gather(x, tensor_parallel_output_grad=False, group=None):
        gather_calls["n"] += 1
        return x

    monkeypatch.setattr(
        "megatron.core.tensor_parallel.mappings.gather_from_sequence_parallel_region", _fake_gather, raising=True
    )

    orig_fake_forward = _FakeHead.forward

    def _spy_forward(self, input_, weight=None, runtime_gather_output=None):
        sp_during_head.append(self.sequence_parallel)
        return orig_fake_forward(self, input_, weight=weight, runtime_gather_output=runtime_gather_output)

    monkeypatch.setattr(_FakeHead, "forward", _spy_forward)

    captured_got = []
    monkeypatch.setattr(
        _mtp_mod,
        "MTPLossAutoScaler",
        types.SimpleNamespace(apply=staticmethod(lambda hs, loss: captured_got.append(loss.detach().clone()) or hs)),
    )
    monkeypatch.setattr(_patch, "MTPLossAutoScaler", _mtp_mod.MTPLossAutoScaler, raising=False)
    monkeypatch.setattr(_patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(_patch, "_resolve_chunk_size", lambda: 3)
    _patch.process_mtp_loss(
        full_hidden,
        labels,
        loss_mask,
        head,
        None,
        None,
        False,
        _stub_clml,
        cfg,
    )

    assert gather_calls["n"] == LAYERS, f"expected one SP gather per depth, got {gather_calls['n']}"
    assert sp_during_head and all(sp is False for sp in sp_during_head), "head must run with sequence_parallel=False"
    assert head.sequence_parallel is True, "sequence_parallel must be restored to True after chunking"
    assert len(captured_got) == len(captured_ref) == LAYERS
    for d, (g, r) in enumerate(zip(captured_got, captured_ref)):
        assert torch.equal(g, r), f"SP depth {d}: chunked != unchunked (max|Δ|={(g - r).abs().max()})"


def test_patch_replaces_model_bindings():
    """The patch swaps every MCore model-level process_mtp_loss binding."""
    import megatron.core.models.gpt.gpt_model as gpt_mod

    assert _mtp_mod.process_mtp_loss is _patch.process_mtp_loss
    assert gpt_mod.process_mtp_loss is _patch.process_mtp_loss

    try:
        import megatron.core.models.hybrid.hybrid_model as hybrid_mod
    except ImportError:
        return
    assert hybrid_mod.process_mtp_loss is _patch.process_mtp_loss
