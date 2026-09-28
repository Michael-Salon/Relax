# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in manifest of HF tensors sent by a colocated Qwen4 weight sync.

The caller logs audit errors and lets training continue. A missing completed
manifest fails external sync acceptance instead of claiming partial coverage.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from .qwen4_numeric_probe import fingerprint


class Qwen4SyncProducerAudit:
    def __init__(self, directory: str, weight_version: int, source_rank: int, sglang_tp_rank: int | None) -> None:
        self.directory = Path(directory)
        self.weight_version = str(weight_version)
        self.source_rank = source_rank
        self.sglang_tp_rank = sglang_tp_rank
        self.session_id = uuid.uuid4().hex
        self.entries: list[dict] = []
        self.bucket_count = 0
        self.numeric_enabled = os.environ.get("QWEN4_NUMERIC_PARITY") == "1"
        self.numeric_samples: dict[str, dict] = {}

    def record_bucket(self, named_tensors: list[tuple]) -> None:
        """Capture metadata, plus selected value samples when explicitly
        enabled."""
        self.bucket_count += 1
        for name, tensor in named_tensors:
            self.entries.append(
                {
                    "bucket": self.bucket_count,
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
            if self.numeric_enabled:
                sample = fingerprint(name, tensor, self.sglang_tp_rank)
                if sample is not None:
                    if name in self.numeric_samples:
                        raise ValueError(f"duplicate numeric probe source key: {name}")
                    self.numeric_samples[name] = sample

    def complete(self) -> Path:
        """Publish only after all engine replies and the final actor
        barrier."""
        payload = {
            "schema": 1,
            "scope": "colocated_tensor_hf_export",
            "completed": True,
            "source_rank": self.source_rank,
            "sglang_tp_rank": self.sglang_tp_rank,
            "pid": os.getpid(),
            "session_id": self.session_id,
            "weight_version": self.weight_version,
            "bucket_count": self.bucket_count,
            "entry_count": len(self.entries),
            "entries": self.entries,
        }
        if self.numeric_enabled:
            payload["numeric_samples"] = self.numeric_samples
        version_digest = hashlib.sha256(self.weight_version.encode()).hexdigest()[:16]
        filename = f"qwen4-producer-{version_digest}-rank{self.source_rank}-{self.session_id}.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / filename
        temporary = self.directory / (filename + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
        return destination
