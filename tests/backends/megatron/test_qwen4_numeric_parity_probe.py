# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Counterexamples for the isolated Qwen4 online value probe."""

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
PROBE = runpy.run_path(str(ROOT / "relax/backends/megatron/weight_update/qwen4_numeric_probe.py"))
CHECKER = runpy.run_path(str(ROOT / "scripts/tools/check_qwen4_numeric_parity.py"))


def test_qwen4_numeric_probe_detects_changed_bf16_values() -> None:
    fingerprint = PROBE["fingerprint"]
    vision = torch.linspace(-1, 1, 1152).to(torch.bfloat16)
    original = fingerprint(PROBE["VISION"], vision, 0)
    changed = vision.clone()
    changed[-1] = 3
    assert original["sha256"] != fingerprint(PROBE["VISION"], changed, 0)["sha256"]

    ple = torch.ones((125015, 160), dtype=torch.bfloat16)
    ple[0, 0] = 2
    ple[-1, -1] = 3
    shard_sample = fingerprint(PROBE["PLE"], ple, 0)
    assert shard_sample["sample_count"] == 68
    assert shard_sample["distinct_bf16_values"] == 3
    assert fingerprint(PROBE["PLE"], ple, 1) is None
    changed_ple = ple.clone()
    changed_ple[-1, -1] = 4
    assert shard_sample["sha256"] != fingerprint(PROBE["PLE"], changed_ple, 0)["sha256"]


def _write_manifests(root: Path) -> tuple[Path, Path]:
    producer, consumer = root / "producer", root / "consumer"
    producer.mkdir()
    consumer.mkdir()
    target_names = CHECKER["EXPECTED_TARGETS"]
    shapes = CHECKER["EXPECTED_SHAPES"]
    for version in ("1", "2"):
        for rank in range(8):
            names = CHECKER["DIRECT"] | ({CHECKER["PLE"]} if rank == 0 else set())
            entries = [
                {"name": name, "shape": shapes[name], "dtype": "torch.bfloat16", "status": "loaded"} for name in names
            ]
            entries.extend({"name": f"filler{i}"} for i in range(12939 - len(entries)))
            samples = {
                name: {
                    "name": name,
                    "shape": shapes[name],
                    "dtype": "torch.bfloat16",
                    "selection": "all" if name != CHECKER["PLE"] else "17_even_rows_x_4_fixed_cols",
                    "sample_count": shapes[name][0] if name != CHECKER["PLE"] else 68,
                    "sha256": f"{version}-{rank}-{name}",
                    "distinct_bf16_values": 3,
                }
                for name in names
            }
            base = {"weight_version": version, "bucket_count": 92, "entries": entries}
            producer_payload = dict(base)
            producer_payload.update(completed=True, source_rank=rank, sglang_tp_rank=rank, numeric_samples=samples)
            (producer / f"{version}-{rank}.json").write_text(json.dumps(producer_payload))
            consumer_payload = dict(base)
            consumer_payload.update(
                tp_rank=rank,
                last_bucket_success=True,
                unknown_names=[],
                unhandled_names=[],
                numeric_samples={
                    name: {"target_parameter": target_names[name], "received": value, "stored": copy.copy(value)}
                    for name, value in samples.items()
                },
            )
            (consumer / f"{version}-{rank}.json").write_text(json.dumps(consumer_payload))
    return producer, consumer


def test_qwen4_numeric_checker_rejects_tamper_and_wrong_parameter(tmp_path: Path) -> None:
    producer, consumer = _write_manifests(tmp_path)
    check = CHECKER["check"]
    assert check(producer, consumer, {"1", "2"}, set(range(8)))["checked_triplets"] == 50

    path = consumer / "2-0.json"
    valid = json.loads(path.read_text())
    corrupt = copy.deepcopy(valid)
    corrupt["numeric_samples"][CHECKER["PLE"]]["stored"]["sha256"] = "wrong"
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="value mismatch"):
        check(producer, consumer, {"1", "2"}, set(range(8)))

    corrupt = copy.deepcopy(valid)
    corrupt["numeric_samples"][CHECKER["VISION"]]["target_parameter"] = "unrelated.weight"
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="wrong stored parameter"):
        check(producer, consumer, {"1", "2"}, set(range(8)))

    path.unlink()
    with pytest.raises(ValueError, match="expected 16 manifests"):
        check(producer, consumer, {"1", "2"}, set(range(8)))
