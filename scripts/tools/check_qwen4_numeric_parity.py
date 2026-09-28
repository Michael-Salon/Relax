#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict checker for the isolated four-key Qwen4 online sync value probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VISION = "model.visual.blocks.0.norm1.weight"
QSA = "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight"
BACKBONE = "model.language_model.layers.3.self_attn.q_norm.weight"
PLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
DIRECT = {VISION, QSA, BACKBONE}
EXPECTED_SHAPES = {VISION: [1152], QSA: [128], BACKBONE: [256], PLE: [125015, 160]}
EXPECTED_TARGETS = {
    VISION: "visual.blocks.0.norm1.weight",
    QSA: "model.layers.3.indexer.q_layernorm.weight",
    BACKBONE: "model.layers.3.q_norm.weight",
    PLE: "model.layers.1.ple.ple_embedding.ngram_embedding.weight",
}


def manifests(directory: Path, role: str, versions: set[str], ranks: set[int]) -> dict:
    files = list(directory.glob("*.json"))
    expected_count = len(versions) * len(ranks)
    if len(files) != expected_count:
        raise ValueError(f"{role}: expected {expected_count} manifests, found {len(files)}")
    by_pair = {}
    for path in files:
        data = json.loads(path.read_text())
        version = str(data["weight_version"])
        rank = data["sglang_tp_rank"] if role == "producer" else data["tp_rank"]
        pair = (version, rank)
        if version not in versions or rank not in ranks or pair in by_pair:
            raise ValueError(f"{role}: unexpected or duplicate version/rank {pair}")
        if role == "producer" and (data.get("completed") is not True or data["source_rank"] != rank):
            raise ValueError(f"{role}: incomplete or wrong source rank {pair}")
        if role == "consumer" and (
            not data.get("last_bucket_success") or data.get("unknown_names") or data.get("unhandled_names")
        ):
            raise ValueError(f"{role}: incomplete consumer load {pair}")
        if data.get("bucket_count") != 92 or len(data["entries"]) != 12939:
            raise ValueError(f"{role}: missing buckets or keys {pair}")
        by_pair[pair] = data
    return by_pair


def check(producer_dir: Path, consumer_dir: Path, versions: set[str], ranks: set[int]) -> dict:
    producer = manifests(producer_dir, "producer", versions, ranks)
    consumer = manifests(consumer_dir, "consumer", versions, ranks)
    checked = 0
    changing = []
    for pair in sorted(producer):
        p, c = producer[pair], consumer[pair]
        version, rank = pair
        names = DIRECT | ({PLE} if rank == 0 else set())
        ps = p.get("numeric_samples")
        cs = c.get("numeric_samples")
        if not isinstance(ps, dict) or not isinstance(cs, dict) or set(ps) != names or set(cs) != names:
            raise ValueError(f"{pair}: numeric sample set is missing or unexpected")
        loaded = {e["name"]: e for e in c["entries"] if e["name"] in names}
        if set(loaded) != names or any(e["status"] != "loaded" for e in loaded.values()):
            raise ValueError(f"{pair}: sampled source names were not loaded")
        for name in sorted(names):
            if cs[name].get("target_parameter") != EXPECTED_TARGETS[name]:
                raise ValueError(f"{pair}: wrong stored parameter for {name}")
            a, b, d = ps[name], cs[name]["received"], cs[name]["stored"]
            fields = ("name", "shape", "dtype", "selection", "sample_count", "sha256", "distinct_bf16_values")
            if any(a[field] != b[field] or a[field] != d[field] for field in fields):
                raise ValueError(f"{pair}: source/received/stored value mismatch for {name}")
            if name in (VISION, PLE) and a["distinct_bf16_values"] < 2:
                raise ValueError(f"{pair}: degenerate constant-value sample for {name}")
            if (
                list(a["shape"]) != EXPECTED_SHAPES[name]
                or list(a["shape"]) != loaded[name]["shape"]
                or a["dtype"] != "torch.bfloat16"
                or a["dtype"] != loaded[name]["dtype"]
            ):
                raise ValueError(f"{pair}: numeric/metadata shape or dtype mismatch for {name}")
            checked += 1
    if len(versions) >= 2:
        first, last = sorted(versions)[0], sorted(versions)[-1]
        for rank in sorted(ranks):
            for name in sorted(DIRECT | ({PLE} if rank == 0 else set())):
                if (
                    producer[first, rank]["numeric_samples"][name]["sha256"]
                    != producer[last, rank]["numeric_samples"][name]["sha256"]
                ):
                    changing.append((rank, name))
    return {
        "checked_triplets": checked,
        "versions": sorted(versions),
        "ranks": sorted(ranks),
        "changed_samples": changing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("producer_dir", type=Path)
    parser.add_argument("consumer_dir", type=Path)
    parser.add_argument("--versions", nargs="+", default=["1", "2"])
    parser.add_argument("--ranks", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    print(json.dumps(check(args.producer_dir, args.consumer_dir, set(args.versions), set(args.ranks)), sort_keys=True))


if __name__ == "__main__":
    main()
