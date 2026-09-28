#!/usr/bin/env bash
# Apply the Qwen3.8 dependency patches to disposable, pinned upstream checkouts.
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 MEGATRON_LM_DIR MEGATRON_BRIDGE_DIR TRANSFORMERS_DIR" >&2
    exit 2
fi

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mcore_dir="$1"
bridge_dir="$2"
transformers_dir="$3"

apply_one() {
    local name="$1" repo="$2" base="$3" patch="$4" expected_sha="$5" expected_tree="$6"
    local actual_head actual_sha actual_tree
    actual_head="$(git -C "$repo" rev-parse HEAD)"
    if [[ "$actual_head" != "$base" ]]; then
        echo "$name: expected clean upstream base $base, got $actual_head" >&2
        return 1
    fi
    if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
        echo "$name: checkout has existing changes" >&2
        return 1
    fi
    actual_sha="$(sha256sum "$patch_dir/$patch")"
    actual_sha="${actual_sha%% *}"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
        echo "$name: patch SHA256 mismatch" >&2
        return 1
    fi
    git -C "$repo" apply --check "$patch_dir/$patch"
    git -C "$repo" apply --index "$patch_dir/$patch"
    actual_tree="$(git -C "$repo" write-tree)"
    if [[ "$actual_tree" != "$expected_tree" ]]; then
        echo "$name: result tree mismatch: $actual_tree != $expected_tree" >&2
        return 1
    fi
    echo "$name: applied $patch to $base; tree=$actual_tree"
}

apply_one "Megatron-LM" "$mcore_dir" \
    b26f370df24169e64164aede07507bdda64ef4ec megatron-lm.patch \
    62c04b3ec3ca1c71d58885193a3794bb4fb5ad4d624c88ff4b8ffd50c712bce7 \
    6af6158a5e8e521160d2cfd4b5876213c852e78e

# Bridge deliberately keeps the upstream .gitmodules and Megatron-LM gitlink.
# The patched MCore above is supplied through PYTHONPATH instead of a fork submodule.
apply_one "Megatron-Bridge" "$bridge_dir" \
    8ec5c9cdc2bf1af3247a2dbe73696534d6055814 megatron-bridge.patch \
    872dffd668a1f4099c54f4f95d84c86ea15fde4bfe6266784e802e644976c3ae \
    97ae5165268f9fbcf0093075d95808475a57d24e

transformers_head="$(git -C "$transformers_dir" rev-parse HEAD)"
if [[ "$transformers_head" == fc5c5bde8e656dad91cbf34e61940d984b1c7b91 ]]; then
    if [[ -n "$(git -C "$transformers_dir" status --porcelain)" ]]; then
        echo "Transformers: official target checkout has existing changes" >&2
        exit 1
    fi
    echo "Transformers: official Qwen4Exp commit already pinned; patch not needed"
else
    apply_one "Transformers" "$transformers_dir" \
        f0e42efb2ca057209aa0a2350908578a01283018 transformers-optional.patch \
        785762a987e09b6d38bfacb52bbcf249ce77b86eb3964c8723f30625140f525d \
        973abc409d6395677f760cbd2ba425a056e25e30
fi
