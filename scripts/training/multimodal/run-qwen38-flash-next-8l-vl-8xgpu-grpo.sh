#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next 8L VL GRPO smoke on one 8-GPU colocated node.
# The prompt JSONL has prompt, label, and images fields; prompt contains <image>.
#
# Usage:
#   MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints \
#   bash scripts/training/multimodal/run-qwen38-flash-next-8l-vl-8xgpu-grpo.sh
# Bounded Stage-2 one- or two-step gate (opt-in; generate data on shared storage first):
#   PYTHONPATH=. python3 scripts/tools/qwen38_stage2_rl_prompt_gate.py \
#     --model-dir "$MODEL_DIR/Qwen3.8-Flash-Next-8L-VL" \
#     --prompt-data "$DATA_DIR/vl_rl_stage2_8l.jsonl" --create-from-image "$DATA_DIR/test.png"
#   MODEL_DIR=/models DATA_DIR=/data STAGE2_RL_GATE=1 NO_SAVE=1 NUM_ROLLOUT=1 \
#     bash scripts/training/multimodal/run-qwen38-flash-next-8l-vl-8xgpu-grpo.sh

set -ex
set -o pipefail

export RAY_NO_WAIT="${RAY_NO_WAIT-1}"

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-8l.sh"

: "${MODEL_DIR:?Set MODEL_DIR to the 8L VL model root}"
: "${DATA_DIR:?Set DATA_DIR to the VL GRPO prompt root}"
: "${STAGE2_RL_GATE:=0}"
if [[ "${STAGE2_RL_GATE}" != 0 && "${STAGE2_RL_GATE}" != 1 ]]; then
    echo "STAGE2_RL_GATE must be 0 or 1" >&2
    exit 2
fi
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    : "${BF16_MAIN_GRADS:=1}"
    : "${NO_SAVE:=1}"
    : "${NUM_ROLLOUT:=1}"
else
    : "${BF16_MAIN_GRADS:=0}"
    : "${NO_SAVE:=0}"
fi
: "${BF16_GRAD_PROBE:=0}"
if [[ "${BF16_MAIN_GRADS}" != 0 && "${BF16_MAIN_GRADS}" != 1 ]]; then
    echo "BF16_MAIN_GRADS must be 0 or 1" >&2
    exit 2
fi
if [[ "${NO_SAVE}" != 0 && "${NO_SAVE}" != 1 ]]; then
    echo "NO_SAVE must be 0 or 1" >&2
    exit 2
fi
if [[ "${BF16_GRAD_PROBE}" != 0 && "${BF16_GRAD_PROBE}" != 1 ]]; then
    echo "BF16_GRAD_PROBE must be 0 or 1" >&2
    exit 2
fi
if [[ "${BF16_GRAD_PROBE}" == 1 && "${BF16_MAIN_GRADS}" != 1 ]]; then
    echo "BF16_GRAD_PROBE requires BF16_MAIN_GRADS=1" >&2
    exit 2
fi
if [[ "${STAGE2_RL_GATE}" == 1 && (
    "${NO_SAVE}" != 1 || ! "${NUM_ROLLOUT}" =~ ^[12]$ || "${BF16_MAIN_GRADS}" != 1 || "${BF16_GRAD_PROBE}" != 0
) ]]; then
    echo "STAGE2_RL_GATE requires NO_SAVE=1, NUM_ROLLOUT=1 or 2, BF16_MAIN_GRADS=1, BF16_GRAD_PROBE=0" >&2
    exit 2
fi
if [[ "${BF16_MAIN_GRADS}" == 1 && ( "${NO_SAVE}" != 1 || ( "${STAGE2_RL_GATE}" == 0 && "${NUM_ROLLOUT:-2}" != 1 ) ) ]]; then
    echo "BF16_MAIN_GRADS requires NO_SAVE=1 and NUM_ROLLOUT=1 for this isolated experiment" >&2
    exit 2
fi
if [[ "${NO_SAVE}" == 1 && "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == 1 ]]; then
    echo "NO_SAVE=1 cannot override a checkpoint optimizer scheduler" >&2
    exit 2
fi
if [[ "${NO_SAVE}" == 0 ]]; then
    : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
fi

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/qwen38-flash-next-vl}"
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    EXP_NAME="${EXP_NAME:-qwen38-flash-next-8l-vl-stage2-grpo-t4e8c2}"
else
    EXP_NAME="${EXP_NAME:-qwen38-flash-next-8l-vl-grpo-8xgpu}"
fi
MODEL_NAME="${VL_MODEL_NAME:-Qwen3.8-Flash-Next-8L-VL}"
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/vl_rl_stage2_8l.jsonl}"
    ROLLOUT_MAX_PROMPT_LEN=3072
    ROLLOUT_MAX_CONTEXT_LEN=4096
    # CP2 x 2048 = 4096 physical tokens per microbatch. The processor gate below
    # requires each prompt >2051 tokens, so two samples cannot fit in one pack.
    TOKEN_BUDGET=2048
else
    PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/vl_rl_smoke.jsonl}"
    ROLLOUT_MAX_PROMPT_LEN=1024
    ROLLOUT_MAX_CONTEXT_LEN=1152
    TOKEN_BUDGET=1024
fi
TP_SIZE="${TP_SIZE:-$((2 + 2 * STAGE2_RL_GATE))}"
CP_SIZE="${CP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-$((2 + 6 * STAGE2_RL_GATE))}"
if [[ "${STAGE2_RL_GATE}" == 1 && (
    "${TP_SIZE}" != 4 || "${CP_SIZE}" != 2 || "${EP_SIZE}" != 8
) ]]; then
    echo "STAGE2_RL_GATE requires TP4/EP8/CP2 on 8 GPUs" >&2
    exit 2
fi

if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
    PYTHONPATH="${RELAX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" python3 "${RELAX_ROOT}/scripts/tools/qwen38_stage2_rl_prompt_gate.py" \
        --model-dir "${MODEL_DIR}/${MODEL_NAME}" \
        --prompt-data "${PROMPT_DATA}" \
        --max-prompt 3072 --max-response 64 --max-context 4096
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
)
if [[ "${NO_SAVE}" == 0 ]]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-2}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    )
fi
if [ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    CKPT_ARGS+=(--override-opt-param-scheduler)
fi

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024
    --rm-type random
    --num-rollout "${NUM_ROLLOUT:-2}"
    --rollout-batch-size 2
    --n-samples-per-prompt 4
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-response-len 64
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --rollout-temperature 0.8
    --global-batch-size 8
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --micro-batch-size 1
    --max-tokens-per-gpu "${TOKEN_BUDGET}"
    --log-probs-max-tokens-per-gpu "${TOKEN_BUDGET}"
    --no-rope-fusion
)

if [ "${CP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS+=(--calculate-per-token-loss)
fi
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    PARALLEL_ARGS+=(--use-dynamic-batch-size)
fi

TRAINABLE_ARGS=()
if [[ "${STAGE2_RL_GATE}" == 0 ]]; then
    TRAINABLE_ARGS+=(--freeze-params-name-list 'self_attention\.indexer\.')
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)
if [[ "${BF16_MAIN_GRADS}" == 1 ]]; then
    OPTIMIZER_ARGS+=(--main-grads-dtype bf16)
fi
if [[ "${BF16_GRAD_PROBE}" == 1 ]]; then
    OPTIMIZER_ARGS+=(--custom-megatron-before-train-step-hook-path scripts.tools.qwen38_bf16_grad_probe.install_probe)
fi
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    OPTIMIZER_ARGS+=(--custom-megatron-before-train-step-hook-path scripts.tools.qwen38_stage2_step_probe.install_probe)
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-ep-size 8
    --sglang-mem-fraction-static 0.5
    --sglang-disable-cuda-graph
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
)
if [[ "${BF16_MAIN_GRADS}" == 1 ]]; then
    MISC_ARGS+=(--grad-reduce-in-bf16)
else
    MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
MISC_ARGS+=(--selective-offload --use-health-check)
if [[ "${STAGE2_RL_GATE}" == 1 ]]; then
    MISC_ARGS+=(
        --qsa-indexer-loss-coeff 0.7
        --train-env-vars '{"MCORE_QSA_SPARSE_BACKEND":"id_sparse"}'
    )
fi

mkdir -p "${SCRIPT_DIR}/log"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
