# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Create and validate a genuine-image prompt set for the bounded QSA Stage-2
RL gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relax.utils.data.data_utils import check_sample_length, process_raw_sample
from relax.utils.data.processing_utils import adapt_processor_kwargs, load_processor, load_tokenizer


def _rows(image: Path, repeat: int) -> list[dict[str, object]]:
    return [
        {
            "prompt": f"<image>{question}" + " color" * repeat,
            "label": "blue",
            "images": [str(image)],
        }
        for question in (
            "What is the dominant color? Answer with one word.",
            "Describe the main color in one word.",
        )
    ]


def validate_prompt_data(
    prompt_data: Path,
    model_dir: Path,
    *,
    max_prompt: int = 3072,
    max_response: int = 64,
    max_context: int = 4096,
) -> list[int]:
    """Measure the same chat-template and processor prompt IDs as Relax
    rollout."""
    tokenizer = load_tokenizer(str(model_dir), trust_remote_code=True)
    processor = load_processor(str(model_dir), trust_remote_code=True)
    if processor is None:
        raise ValueError(f"No multimodal processor loaded from {model_dir}")
    rows = [json.loads(line) for line in prompt_data.read_text().splitlines() if line.strip()]
    if len(rows) != 2:
        raise ValueError(f"Expected exactly two image prompts for rollout_batch_size=2, got {len(rows)}")

    lengths = []
    for row_number, row in enumerate(rows):
        images = row.get("images")
        prompt = row.get("prompt")
        if (
            not isinstance(prompt, str)
            or prompt.count("<image>") != 1
            or not isinstance(images, list)
            or len(images) != 1
            or not isinstance(images[0], str)
            or not Path(images[0]).is_absolute()
            or not Path(images[0]).is_file()
            or not row.get("label")
        ):
            raise ValueError(f"Row {row_number} lacks one real <image>, absolute image file, or label")
        sample = process_raw_sample(
            row,
            tokenizer,
            processor,
            prompt_key="prompt",
            label_key="label",
            multimodal_keys={"image": "images"},
            apply_chat_template=True,
        )
        if not check_sample_length(sample, tokenizer, processor, max_prompt):
            raise ValueError(f"Row {row_number} would be filtered by the Relax rollout Dataset")
        media = sample.multimodal_inputs
        if media is None or len(media.get("images") or []) != 1:
            raise ValueError(f"Row {row_number} lost its image in Relax processing")
        processor_output = processor(
            text=sample.prompt,
            **adapt_processor_kwargs(
                processor,
                media,
                {"use_audio_in_video": False, "return_mm_token_type_ids": False},
            ),
        )
        length = len(processor_output["input_ids"][0])
        if (
            "pixel_values" not in processor_output
            or "image_grid_thw" not in processor_output
            or len(processor_output["pixel_values"]) == 0
            or len(processor_output["image_grid_thw"]) != 1
        ):
            raise ValueError(f"Row {row_number} did not produce image pixels and grid")
        if not 2051 < length <= max_prompt or length + max_response > max_context:
            raise ValueError(
                f"Row {row_number} processor prompt length {length} cannot fit Stage-2 "
                f"(need 2051 < length <= {max_prompt}, length + {max_response} <= {max_context})"
            )
        lengths.append(length)
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--create-from-image", type=Path)
    parser.add_argument("--repeat-color", type=int, default=2052)
    parser.add_argument("--max-prompt", type=int, default=3072)
    parser.add_argument("--max-response", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    if args.create_from_image is not None:
        image = args.create_from_image.resolve(strict=True)
        if args.prompt_data.exists():
            parser.error(f"Refusing to overwrite {args.prompt_data}")
        args.prompt_data.parent.mkdir(parents=True, exist_ok=True)
        with args.prompt_data.open("w", encoding="utf-8") as stream:
            for row in _rows(image, args.repeat_color):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = validate_prompt_data(
        args.prompt_data,
        args.model_dir,
        max_prompt=args.max_prompt,
        max_response=args.max_response,
        max_context=args.max_context,
    )
    sys.stdout.write(
        json.dumps({"prompt_lengths": lengths, "max_total_tokens": max(lengths) + args.max_response}) + "\n"
    )


if __name__ == "__main__":
    main()
