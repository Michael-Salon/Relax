# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for the Qwen3.8 VL SFT data preflight."""

import pytest
import torch

from relax.engine.sft.dataset.streaming import ProcessedSample
from scripts.tools.qwen38_vl_sft_data_preflight import _check_processed


def _processed(*, grid_rows: int = 2, assistant_loss: bool = True) -> ProcessedSample:
    pixels = torch.zeros(grid_rows * 4, 1536)
    return ProcessedSample(
        tokens=torch.tensor([1, 99, 99, 2, 99, 99, 3]),
        loss_mask=torch.tensor([0, 0, 0, 0, 0, 0, int(assistant_loss)]),
        total_length=7,
        multimodal_train_inputs={
            "image_grid_thw": torch.tensor([[1, 2, 2]] * grid_rows),
            "pixel_values": pixels,
        },
        source_idx=0,
    )


def test_qwen38_vl_sft_data_preflight_accepts_expanded_two_image_sample() -> None:
    assert _check_processed(_processed(), image_count=2, image_token_id=99, capacity=7) == 4


def test_qwen38_vl_sft_data_preflight_rejects_missing_image_grid() -> None:
    with pytest.raises(ValueError, match="image_grid_thw"):
        _check_processed(_processed(grid_rows=1), image_count=2, image_token_id=99, capacity=7)


def test_qwen38_vl_sft_data_preflight_rejects_image_supervision() -> None:
    processed = _processed()
    processed.loss_mask[1] = 1
    with pytest.raises(ValueError, match="image tokens"):
        _check_processed(processed, image_count=2, image_token_id=99, capacity=7)


def test_qwen38_vl_sft_data_preflight_rejects_unsupervised_or_oversize_sample() -> None:
    with pytest.raises(ValueError, match="no assistant loss"):
        _check_processed(_processed(assistant_loss=False), image_count=2, image_token_id=99, capacity=7)
    with pytest.raises(ValueError, match="exceeds SFT capacity"):
        _check_processed(_processed(), image_count=2, image_token_id=99, capacity=6)
