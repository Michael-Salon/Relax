#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 48L TP8/PP8/CP2/EP16 PLE CPU offload precision gate from a new HF start.
# Do not source scripts/entrypoint/ray-job.sh on a shared cluster: it performs
# cluster-wide cleanup. Supply a reviewed RUNTIME_ENV_JSON for execution.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../models/qwen38-flash-next.sh"

# The shared model template defaults to auxiliary router loss 0.001; the
# delivered AutoModel run explicitly disables it for this packed Qwen path.
aux_coeff_found=0
for ((i = 0; i < ${#MODEL_ARGS[@]}; i++)); do
    if [[ "${MODEL_ARGS[i]}" == --moe-aux-loss-coeff ]]; then
        MODEL_ARGS[i + 1]=0
        ((aux_coeff_found += 1))
    fi
done
[[ "${aux_coeff_found}" == 1 ]] || { echo "Expected one MoE aux coefficient in model template" >&2; exit 2; }

: "${MODEL_DIR:?Set MODEL_DIR to the checkpoint parent directory}"
: "${PROMPT_DATA:?Set the original delivery train JSONL with real images}"
: "${QWEN48_PRECISION_CONFIG:?Set the pinned precision YAML path}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the current Ray Jobs HTTP endpoint}"
: "${SAVE_DIR:?Set a persistent checkpoint parent directory}"
: "${EXP_NAME:?Set a unique experiment name or the same name when resuming}"

MODEL_PATH="${MODEL_DIR}/${MODEL_NAME:-Qwen3.8-Flash-Next}"
[[ -f "${MODEL_PATH}/config.json" && -f "${PROMPT_DATA}" && -f "${QWEN48_PRECISION_CONFIG}" ]] || {
    echo "Model config, training data, or precision YAML is missing" >&2
    exit 2
}

TP_SIZE=8
PP_SIZE=8
CP_SIZE=2
EP_SIZE=16
GBS="${GLOBAL_BATCH_SIZE:-256}"
MBS="${MICRO_BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-131072}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((SEQ_LENGTH / CP_SIZE))}"
NUM_STEPS="${NUM_STEPS:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
DP_SIZE=1
if [[ "${SEQ_LENGTH}" != 32768 && "${SEQ_LENGTH}" != 65536 && "${SEQ_LENGTH}" != 131072 ]]; then
    echo "Length gate supports only 32K, 64K, or 128K" >&2
    exit 2
fi
if (( MAX_TOKENS_PER_GPU != SEQ_LENGTH / CP_SIZE )); then
    echo "Set max tokens per GPU to sequence length divided by CP size" >&2
    exit 2
fi
if (( GBS < PP_SIZE || MBS != 1 || NUM_STEPS < 3 || SAVE_INTERVAL < 1 || EVAL_INTERVAL < 1 )); then
    echo "Long run requires GBS>=PP_SIZE, MBS=1, NUM_STEPS>=3 and positive save/eval intervals" >&2
    exit 2
fi
[[ -d "${SAVE_DIR}" && -w "${SAVE_DIR}" ]] || { echo "SAVE_DIR must exist and be writable" >&2; exit 2; }
RUN_DIR="${SAVE_DIR}/${EXP_NAME}"
[[ "${EXP_NAME}" == qwen38-48l-ple-offload-* ]] || {
    echo "PLE offload gate requires an isolated qwen38-48l-ple-offload-* experiment name" >&2
    exit 2
}
if [[ "${QWEN48_RESUME:-0}" == "1" ]]; then
    tracker="${RUN_DIR}/latest_checkpointed_iteration.txt"
    [[ -f "${tracker}" ]] || { echo "Resume requires an existing checkpoint tracker" >&2; exit 2; }
    checkpoint_iteration="$(<"${tracker}")"
    [[ "${checkpoint_iteration}" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint tracker" >&2; exit 2; }
    printf -v iteration_dir '%s/iter_%07d' "${RUN_DIR}" "${checkpoint_iteration}"
    [[ -f "${iteration_dir}/.metadata" && -f "${iteration_dir}/metadata.json" ]] || {
        echo "Resume requires a complete distributed checkpoint iteration" >&2
        exit 2
    }
    (( NUM_STEPS > checkpoint_iteration + 2 )) || {
        echo "Resume requires at least two optimizer steps after the checkpoint" >&2
        exit 2
    }
    audit_start_rollout="$((checkpoint_iteration + 1))"
elif [[ -e "${RUN_DIR}" ]]; then
    echo "Existing PLE offload checkpoint requires QWEN48_RESUME=1" >&2
    exit 2
else
    audit_start_rollout=1
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --load "${RUN_DIR}"
    --save "${RUN_DIR}"
    --save-interval "${SAVE_INTERVAL}"
    --max-actor-ckpt-to-keep 2
    --num-rollout "${NUM_STEPS}"
)
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --eval-size "${EVAL_SIZE:-512}"
    --eval-interval "${EVAL_INTERVAL}"
    --input-key messages
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false}'
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-128}"
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len "${SEQ_LENGTH}"
    --global-batch-size "${GBS}"
    --micro-batch-size "${MBS}"
    --seq-length "${SEQ_LENGTH}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --per-rank-fetch
    --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-8}"
    --sft-train-data-prefetch
    --custom-megatron-before-train-step-hook-path scripts.tools.qwen38_ple_offload_train_gate_audit.audit_qwen38_48l_train_gate
)
TRAINABLE_ARGS=(
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list 'self_attention\.indexer\.'
)
MTP_ARGS=(
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
)
PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --decoder-first-pipeline-num-layers 2
    --decoder-last-pipeline-num-layers 4
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)
if [[ "${QWEN48_NO_OFFLOAD_TRAIN:-0}" == "1" ]]; then
    PARALLEL_ARGS+=(--no-offload-train)
fi
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-2e-5}"
    --lr-decay-style cosine
    --min-lr 2e-6
    --lr-warmup-fraction 0.05
    --lr-decay-iters 2023
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
)
MISC_ARGS=(
    --custom-config-path "${QWEN48_PRECISION_CONFIG}"
    --seed 1234
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
)

echo "48L long run: TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE} DP=${DP_SIZE} GBS=${GBS} MBS=${MBS} seq=${SEQ_LENGTH} max_tokens_per_gpu=${MAX_TOKENS_PER_GPU} steps=${NUM_STEPS} save_interval=${SAVE_INTERVAL} eval_interval=${EVAL_INTERVAL}"
echo "Trainable: language backbone, MoE, PLE projection/norm/conv and MTP. Frozen: vision tower/aligner, PLE N-gram table, QSA indexer. MTP scale=0.1; MoE aux=0."
if [[ "${QWEN48_EXECUTE:-0}" != "1" ]]; then
    echo "QWEN48_PREFLIGHT_ONLY; no Ray job submitted"
    exit 0
fi

: "${RUNTIME_ENV_JSON:?Set reviewed runtime env JSON; do not source ray-job.sh on a shared cluster}"
: "${SUBMISSION_ID:?Set a unique submission ID for this gate}"
: "${QWEN48_TRAIN_AUDIT_DIR:?Set the rank-local audit output directory}"
: "${QWEN48_GPU_CLEAR_VERIFIED:?Set to 1 after stopping this experiment keepalive and verifying Ray GPU use is zero}"
[[ "${QWEN48_GPU_CLEAR_VERIFIED}" == "1" ]] || exit 2
python3 - "${RUNTIME_ENV_JSON}" "${QWEN48_TRAIN_AUDIT_DIR}" "${audit_start_rollout}" <<'PY'
import json
import sys

runtime_env = json.loads(sys.argv[1])
audit_dir = sys.argv[2]
audit_start_rollout = sys.argv[3]
if runtime_env.get("env_vars", {}).get("QWEN48_TRAIN_AUDIT_DIR") != audit_dir:
    raise SystemExit("runtime env must propagate the exact QWEN48_TRAIN_AUDIT_DIR to every rank")
if runtime_env.get("env_vars", {}).get("QWEN48_AUDIT_START_ROLLOUT") != audit_start_rollout:
    raise SystemExit(f"precision gate must audit rollout {audit_start_rollout}")
if runtime_env.get("env_vars", {}).get("QWEN38_AUTOMODEL_CHAT_TEMPLATE_COMPAT") != "1":
    raise SystemExit("precision gate must use the delivered AutoModel chat rendering")
if runtime_env.get("env_vars", {}).get("QWEN38_PRECISION_METRICS") != "1":
    raise SystemExit("precision gate must record additive main and MTP loss statistics")
if runtime_env.get("env_vars", {}).get("QWEN48_PLE_CPU_OFFLOAD") != "1":
    raise SystemExit("PLE offload gate requires QWEN48_PLE_CPU_OFFLOAD=1")
if runtime_env.get("env_vars", {}).get("QWEN48_PLE_STREAMED") != "1":
    raise SystemExit("PLE offload gate requires QWEN48_PLE_STREAMED=1")
env_vars = runtime_env.get("env_vars", {})
if env_vars.get("QWEN48_QSA_SELECTED_TRITON") == "1":
    if env_vars.get("MCORE_QSA_SPARSE_BACKEND") != "id_sparse":
        raise SystemExit("selected QSA Triton requires MCORE_QSA_SPARSE_BACKEND=id_sparse")
    if env_vars.get("QWEN48_QSA_SPARSE_BLOCKMASK") == "1":
        raise SystemExit("selected QSA Triton and Flex BlockMask are separate A/B variants")
print(
    "PLE fused:", env_vars.get("QWEN48_PLE_FUSED", "0"),
    "QSA selected Triton:", env_vars.get("QWEN48_QSA_SELECTED_TRITON", "0"),
)
PY

ray job submit --no-wait --address="${RAY_ADDRESS}" --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft":[1,0],"actor":[1,128]}' \
    --actor-num-nodes 16 \
    --actor-num-gpus-per-node 8 \
    --sft-max-in-flight-steps 2 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${MISC_ARGS[@]}"
