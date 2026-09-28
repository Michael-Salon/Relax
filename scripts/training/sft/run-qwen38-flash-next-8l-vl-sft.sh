#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next 8L VL (multimodal) SFT bring-up on the reduced VL checkpoint.
# Uses the Qwen4ExpVLBridge (language + vision tower) via --megatron-to-hf-mode bridge.
# The multimodal data carries an `images` field; --multimodal-keys maps it to the
# `image` media. CP_SIZE>1 exercises the VL+CP path (THD packed + vision CP).
#
# Usage:
#   MODEL_DIR=/path/to/models/Qwen3.8 VL_MODEL_NAME=Qwen3.8-Flash-Next-8L-VL \
#   DATA_DIR=/path/to/vl-sft-data SAVE_DIR=/path/to/checkpoints \
#   bash scripts/training/sft/run-qwen38-flash-next-8l-vl-sft.sh

set -ex
set -o pipefail

export RAY_NO_WAIT="${RAY_NO_WAIT-1}"

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-6}"
MOE_ROUTER_LOAD_BALANCING_TYPE="${MOE_ROUTER_LOAD_BALANCING_TYPE:-aux_loss}"
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-8l.sh"

: "${MODEL_DIR:?Set MODEL_DIR to the directory containing the Qwen3.8 checkpoints}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing the VL SFT JSONL file}"
if [ "${NO_SAVE:-0}" != "1" ]; then
    : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root, or NO_SAVE=1 for a smoke test}"
fi
if [ "${RESUME_TRAINING:-0}" = "1" ] && [ "${NO_SAVE:-0}" = "1" ]; then
    echo "RESUME_TRAINING=1 requires checkpoint saving to be enabled." >&2
    exit 1
fi

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-flash-next}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-8l-vl-sft}"
MODEL_NAME="${VL_MODEL_NAME:-Qwen3.8-Flash-Next-8L-VL}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/vl_sft.jsonl}"
ACTOR_GPUS="${ACTOR_GPUS:-8}"
RUN_DIR="${SAVE_DIR:-}/${EXP_NAME}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
)
if [ -n "${NUM_EPOCH:-}" ]; then
    # Epoch runs must not inherit the two-rollout smoke cap.
    CKPT_ARGS+=(--num-epoch "${NUM_EPOCH}")
    DEFAULT_SAVE_INTERVAL=500
    DEFAULT_CKPT_KEEP=2
else
    CKPT_ARGS+=(--num-rollout "${NUM_ROLLOUT:-2}")
    DEFAULT_SAVE_INTERVAL=1
    DEFAULT_CKPT_KEEP=3
fi
if [ "${NO_SAVE:-0}" != "1" ]; then
    if [ "${RESUME_TRAINING:-0}" = "1" ]; then
        if [ ! -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ]; then
            echo "RESUME_TRAINING=1 but no checkpoint tracker exists at ${RUN_DIR}" >&2
            exit 1
        fi
        CKPT_ARGS+=(--load "${RUN_DIR}")
    elif [ -e "${RUN_DIR}" ] && {
        [ ! -d "${RUN_DIR}" ] || [ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]
    }; then
        echo "Fresh run refused: ${RUN_DIR} already exists and is not an empty directory." >&2
        exit 1
    fi
    CKPT_ARGS+=(
        --save "${RUN_DIR}"
        --save-interval "${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-${DEFAULT_CKPT_KEEP}}"
    )
fi
if [ "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    if [ "${RESUME_TRAINING:-0}" != "1" ]; then
        echo "USE_CHECKPOINT_OPT_PARAM_SCHEDULER=1 requires RESUME_TRAINING=1." >&2
        exit 1
    fi
    CKPT_ARGS+=(--use-checkpoint-opt-param-scheduler)
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --sft-ignore-empty-think
    # multimodal: dataset `images` field -> `image` media; cap image tokens (ms-swift parity)
    --multimodal-keys '{"image": "images"}'
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-256}"
    --sft-invalid-multimodal-strategy error
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --global-batch-size "${GLOBAL_BATCH_SIZE:-1}"
    --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
    --seq-length "${SEQ_LENGTH:-512}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-1024}"
    --per-rank-fetch
)
if [ "${ENABLE_MTP:-0}" = "1" ]; then
    python3 - "${MODEL_DIR}/${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
config = json.loads((model_dir / "config.json").read_text())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
mtp_keys = [key for key in index["weight_map"] if key.startswith("mtp.")]
if config["text_config"].get("mtp_num_hidden_layers") != 1 or len(mtp_keys) != 31:
    raise SystemExit("ENABLE_MTP=1 requires the validated 8L VL checkpoint with all 31 official MTP tensors")
PY
    SFT_ARGS+=(
        --enable-mtp-training
        --mtp-num-layers 1
        --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR:-0.1}"
    )
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-8}"
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size "${CP_SIZE:-1}"
    --expert-model-parallel-size "${EP_SIZE:-8}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"
)
if [ "${FULL_RECOMPUTE:-0}" = "1" ]; then
    PARALLEL_ARGS+=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
fi

TRAINABLE_ARGS=(
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list
    'self_attention\.indexer\.'
)

if [ "${FREEZE_PLE:-0}" = "1" ]; then
    TRAINABLE_ARGS+=('(^|\.)per_layer_embedding\.')
    TRAIN_ENV_EXTRA="${TRAIN_ENV_EXTRA:-},\"QWEN38_EXPECT_FROZEN_PLE\":\"1\""
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style "${LR_DECAY_STYLE:-constant}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --use-precision-aware-optimizer
    --no-store-param-remainders
)
if [ -n "${MIN_LR:-}" ]; then
    OPTIMIZER_ARGS+=(--min-lr "${MIN_LR}")
fi
if [ -n "${LR_WARMUP_FRACTION:-}" ]; then
    OPTIMIZER_ARGS+=(--lr-warmup-fraction "${LR_WARMUP_FRACTION}")
fi

if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
    OPTIMIZER_ARGS+=(
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
    )
fi

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --train-env-vars "{\"NCCL_NVLS_ENABLE\":\"0\",\"TORCH_DISTRIBUTED_DEBUG\":\"INFO\"${TRAIN_ENV_EXTRA:-}}"
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
)

if [ -n "${CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH:-}" ]; then
    MISC_ARGS+=(--custom-megatron-before-train-step-hook-path "${CUSTOM_BEFORE_TRAIN_STEP_HOOK_PATH}")
fi

mkdir -p "${SCRIPT_DIR}/log"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "{\"sft\": [1, 0], \"actor\": [1, ${ACTOR_GPUS}]}" \
    --sft-max-in-flight-steps 2 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
