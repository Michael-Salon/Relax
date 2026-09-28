# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU preflight through Relax's real SFT processor."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from relax.engine.sft.dataset.streaming import ProcessedSample, SFTStreamingDataset
from relax.utils.data.processor_pool import ProcessorPool
from relax.utils.multimodal.config import MultimodalConfig


def _check_processed(
    processed: ProcessedSample,
    *,
    image_count: int,
    image_token_id: int,
    capacity: int,
) -> int:
    tokens = processed.tokens
    loss_mask = processed.loss_mask
    if tokens.ndim != 1 or loss_mask.shape != tokens.shape or processed.total_length != tokens.numel():
        raise ValueError("token, loss mask, and total length disagree")
    if tokens.numel() > capacity:
        raise ValueError(f"expanded length {tokens.numel()} exceeds SFT capacity {capacity}")
    if not bool(loss_mask.any()):
        raise ValueError("no assistant loss tokens")

    image_positions = tokens == image_token_id
    mm_inputs = processed.multimodal_train_inputs
    if image_count == 0:
        if mm_inputs is not None or bool(image_positions.any()):
            raise ValueError("text-only row unexpectedly has image input")
        return 0
    if not isinstance(mm_inputs, dict):
        raise ValueError("image row has no processor multimodal inputs")
    grid = mm_inputs.get("image_grid_thw")
    pixels = mm_inputs.get("pixel_values")
    if not isinstance(grid, torch.Tensor) or grid.ndim != 2 or tuple(grid.shape) != (image_count, 3):
        raise ValueError(f"image_grid_thw must have shape ({image_count}, 3)")
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 2 or pixels.numel() == 0:
        raise ValueError("pixel_values must be a nonempty 2D tensor")
    patch_count = int(grid.to(torch.int64).prod(dim=1).sum())
    if bool((grid <= 0).any()) or pixels.shape[0] != patch_count:
        raise ValueError("image grid patch count disagrees with pixel_values")
    image_tokens = int(image_positions.sum())
    if image_tokens < image_count or bool(loss_mask[image_positions].any()):
        raise ValueError("image tokens are missing or included in assistant loss")
    return image_tokens


def validate_sft_data(
    *,
    model_dir: Path,
    prompt_data: Path,
    max_tokens_per_gpu: int,
    cp_size: int,
    image_max_token_num: int,
    apply_chat_template_kwargs: dict[str, Any],
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Reject rows that cannot train safely."""
    if max_tokens_per_gpu < 1 or cp_size < 1 or image_max_token_num < 1:
        raise ValueError("token cap, CP size, and image cap must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")
    capacity = max_tokens_per_gpu * cp_size
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if not isinstance(image_token_id, int) or image_token_id == tokenizer.unk_token_id:
        raise ValueError("model tokenizer has no <|image_pad|> token")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if image_token_id != getattr(config, "image_token_id", None):
        raise ValueError("tokenizer image token id disagrees with model config")
    pool = ProcessorPool(
        str(model_dir),
        pool_size=1,
        trust_remote_code=True,
        multimodal_config=MultimodalConfig(image_max_token_num=image_max_token_num),
    )
    dataset: SFTStreamingDataset | None = None
    try:
        dataset = SFTStreamingDataset(
            str(prompt_data),
            tokenizer=tokenizer,
            processor_pool=pool,
            capacity=capacity,
            prompt_key="messages",
            multimodal_keys={"image": "images"},
            pad_token_ids=(image_token_id,),
            oversize_strategy="keep",
            invalid_multimodal_strategy="error",
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            loss_ignore_empty_think=True,
            prefetch_max_cached=0,
        )
        count = min(len(dataset), max_rows) if max_rows is not None else len(dataset)
        if count == 0:
            raise ValueError("SFT dataset is empty")
        image_histogram: Counter[int] = Counter()
        max_expanded = max_loss_tokens = max_image_tokens = 0
        for row_index in range(count):
            try:
                sample = dataset.get_canonical_sample(row_index)
                if sample.videos or sample.audios:
                    raise ValueError("this image-only gate does not accept video or audio")
                images = sample.images or []
                for image_index, image in enumerate(images):
                    if not isinstance(image, str) or not Path(image).is_absolute() or not Path(image).is_file():
                        raise ValueError(
                            f"image position {image_index} is not an existing absolute local file: {image!r}"
                        )
                processed = dataset._process_one(row_index)
                if processed is None:
                    raise ValueError("SFT processor dropped this row")
                image_tokens = _check_processed(
                    processed,
                    image_count=len(images),
                    image_token_id=image_token_id,
                    capacity=capacity,
                )
            except Exception as exc:
                raise ValueError(f"row {row_index}: {exc}") from exc
            image_histogram[len(images)] += 1
            max_expanded = max(max_expanded, processed.total_length)
            max_loss_tokens = max(max_loss_tokens, int(processed.loss_mask.sum()))
            max_image_tokens = max(max_image_tokens, image_tokens)
        return {
            "status": "passed",
            "rows_checked": count,
            "total_rows": len(dataset),
            "capacity": capacity,
            "image_max_token_num": image_max_token_num,
            "ignore_empty_think": True,
            "image_count_histogram": dict(sorted(image_histogram.items())),
            "total_images": sum(n * n_rows for n, n_rows in image_histogram.items()),
            "max_expanded_tokens": max_expanded,
            "max_assistant_loss_tokens": max_loss_tokens,
            "max_image_pad_tokens": max_image_tokens,
        }
    finally:
        if dataset is not None:
            dataset.stop()
        pool.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--max-tokens-per-gpu", type=int, required=True)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--image-max-token-num", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--apply-chat-template-kwargs",
        type=json.loads,
        default={"enable_thinking": False},
    )
    args = parser.parse_args()
    template_kwargs = args.apply_chat_template_kwargs
    if not isinstance(template_kwargs, dict):
        parser.error("--apply-chat-template-kwargs must be a JSON object")
    try:
        summary = validate_sft_data(
            model_dir=args.model_dir,
            prompt_data=args.prompt_data,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            cp_size=args.cp_size,
            image_max_token_num=args.image_max_token_num,
            apply_chat_template_kwargs=template_kwargs,
            max_rows=args.max_rows,
        )
    except Exception as exc:
        sys.stderr.write(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False) + "\n")
        raise SystemExit(1) from exc
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
