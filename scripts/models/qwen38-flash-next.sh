# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Qwen3.8-Flash-Next published 48-layer text shape. Experimental GDN/QSA/PLE
# details are finalized by the pinned Qwen4Exp Megatron Bridge provider.

MODEL_ARGS=(
    --num-layers 48
    --hidden-size 2560
    --ffn-hidden-size 640
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 2
    --kv-channels 256
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --apply-layernorm-1p
    --position-embedding-type rope
    --rotary-percent 0.25
    --rotary-base 10000000
    --disable-bias-linear
    --qk-layernorm
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --num-experts 512
    --moe-ffn-hidden-size 640
    --moe-router-topk 10
    --moe-router-load-balancing-type global_aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-shared-expert-intermediate-size 640
)
