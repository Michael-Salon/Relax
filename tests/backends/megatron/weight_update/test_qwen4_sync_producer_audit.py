# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import ast
import importlib.util
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


SOURCE = Path(__file__).parents[4] / "relax/backends/megatron/weight_update/qwen4_sync_producer_audit.py"
UPDATE_SOURCE = SOURCE.with_name("update_weight_from_tensor.py")
SPEC = importlib.util.spec_from_file_location("qwen4_sync_producer_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Qwen4SyncProducerAudit = MODULE.Qwen4SyncProducerAudit


class FakeTensor:
    shape = (3, 7)
    dtype = "torch.bfloat16"


def test_qwen4_sync_producer_audit_publishes_only_complete_round(tmp_path):
    audit = Qwen4SyncProducerAudit(str(tmp_path), weight_version=3, source_rank=5, sglang_tp_rank=5)
    audit.record_bucket([("model.visual.patch_embed.proj.weight", FakeTensor())])
    audit.record_bucket([("model.layers.0.input_layernorm.weight", FakeTensor())])

    assert list(tmp_path.iterdir()) == []
    path = audit.complete()
    payload = json.loads(path.read_text())
    assert payload["completed"] is True
    assert payload["source_rank"] == payload["sglang_tp_rank"] == 5
    assert payload["weight_version"] == "3"
    assert payload["bucket_count"] == 2
    assert payload["entry_count"] == 2
    assert [entry["bucket"] for entry in payload["entries"]] == [1, 2]
    assert payload["entries"][0]["shape"] == [3, 7]
    assert payload["entries"][0]["dtype"] == "torch.bfloat16"
    assert not list(tmp_path.glob("*.tmp"))


def test_qwen4_sync_producer_audit_keeps_same_version_attempts_distinct(tmp_path):
    first = Qwen4SyncProducerAudit(str(tmp_path), weight_version=1, source_rank=0, sglang_tp_rank=0)
    second = Qwen4SyncProducerAudit(str(tmp_path), weight_version=1, source_rank=0, sglang_tp_rank=0)
    first.record_bucket([("a", FakeTensor())])
    second.record_bucket([("b", FakeTensor())])

    first_path = first.complete()
    second_path = second.complete()
    assert first_path != second_path
    assert {json.loads(path.read_text())["entries"][0]["name"] for path in (first_path, second_path)} == {"a", "b"}


def _run_update_with_mock_transport(monkeypatch, *, colocated_engines, remote_engines, member, complete_error=None):
    """Execute the real update_weights body with its GPU and Ray edges
    mocked."""
    tree = ast.parse(UPDATE_SOURCE.read_text())
    updater_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UpdateWeightFromTensor"
    )
    method = next(
        node for node in updater_class.body if isinstance(node, ast.FunctionDef) and node.name == "update_weights"
    )
    method.decorator_list = []
    namespace = {}
    audit = MagicMock()
    audit_class = MagicMock(return_value=audit)
    events = []

    def complete():
        events.append("complete")
        if complete_error is not None:
            raise complete_error

    audit.complete.side_effect = complete
    fake_dist = SimpleNamespace(
        group=SimpleNamespace(WORLD=object()),
        get_rank=MagicMock(return_value=0),
        barrier=MagicMock(side_effect=lambda **_kwargs: events.append("barrier")),
    )
    namespace.update(
        os=os,
        dist=fake_dist,
        ray=SimpleNamespace(get=lambda value: value),
        Qwen4SyncProducerAudit=audit_class,
        get_gloo_group=lambda: None,
        nullcontext=nullcontext,
        device_utils=SimpleNamespace(maybe_backend_barrier_on_weight_chunk=lambda **_kwargs: None),
        logger=MagicMock(),
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(UPDATE_SOURCE), "exec"),
        namespace,
    )
    updater = SimpleNamespace(
        lora_enabled=False,
        lora_adapter_mode=False,
        lora_merge_mode=False,
        weight_version=0,
        _ipc_engine=object() if member else None,
        _ipc_gather_group=object() if member else None,
        rollout_engines=[MagicMock() for _ in range(colocated_engines)],
        distributed_rollout_engines=[MagicMock() for _ in range(remote_engines)],
        args=SimpleNamespace(offload_rollout=True),
        quantization_config=None,
        weights_getter=lambda: {},
        _hf_weight_iterator=SimpleNamespace(get_hf_weight_chunks=lambda _weights: [[("a", FakeTensor())]]),
        _send_hf_params=lambda _weights: ([object()], None),
    )
    monkeypatch.setenv("RELAX_QWEN4_SYNC_AUDIT_DIR", "/tmp/qwen4-audit-test")
    namespace["update_weights"](updater)
    return audit_class, audit, namespace["logger"], events


@pytest.mark.parametrize(
    ("colocated_engines", "remote_engines", "member"),
    [(1, 0, False), (0, 1, False), (2, 0, True), (1, 1, True)],
)
def test_qwen4_sync_producer_audit_skips_non_single_ipc_routes(monkeypatch, colocated_engines, remote_engines, member):
    audit_class, _, _, _ = _run_update_with_mock_transport(
        monkeypatch, colocated_engines=colocated_engines, remote_engines=remote_engines, member=member
    )
    audit_class.assert_not_called()


def test_qwen4_sync_producer_audit_completes_after_final_barrier(monkeypatch):
    audit_class, audit, _, events = _run_update_with_mock_transport(
        monkeypatch, colocated_engines=1, remote_engines=0, member=True
    )
    audit_class.assert_called_once()
    audit.record_bucket.assert_called_once()
    audit.complete.assert_called_once()
    assert events == ["barrier", "barrier", "barrier", "complete"]


def test_qwen4_sync_producer_audit_io_failure_does_not_abort_sync(monkeypatch):
    audit_class, audit, logger, _ = _run_update_with_mock_transport(
        monkeypatch, colocated_engines=1, remote_engines=0, member=True, complete_error=OSError("disk full")
    )
    audit_class.assert_called_once()
    audit.complete.assert_called_once()
    logger.exception.assert_called_once()
