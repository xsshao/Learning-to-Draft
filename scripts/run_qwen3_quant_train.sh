#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/run_qwen3_quant_train.sh <size|depth> [extra train args...]

Environment:
  MODEL_SIZE       Optional. Defaults to 14B. Used for Qwen/Qwen3-\${MODEL_SIZE}.
  BASE_MODEL_PATH  Optional override. Defaults to Qwen/Qwen3-\${MODEL_SIZE}.
  EA_MODEL_PATH    Optional override. Defaults to AngelSlim/Qwen3-\${MODEL_SIZE}_eagle3.
  QUANTIZATION     Optional. fp16, bf16, int8, or int4. Defaults to bf16.
  CHECKPOINT_VARIANT Optional. Defaults to QUANTIZATION so policies do not overwrite.
  SAVE_PATH        Optional. Defaults to ./checkpoints/quant_transfer.
  DEVICE_MAP       Optional. Defaults to auto.

Examples:
  QUANTIZATION=bf16 bash scripts/run_qwen3_quant_train.sh size
  QUANTIZATION=int8 bash scripts/run_qwen3_quant_train.sh depth
  MODEL_SIZE=32B QUANTIZATION=int4 bash scripts/run_qwen3_quant_train.sh size
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    size|depth) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown policy: $1" >&2; usage; exit 1 ;;
esac
policy="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="${MODEL_SIZE:-14B}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-${MODEL_SIZE}}"
export EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-${MODEL_SIZE}_eagle3}"
export QUANTIZATION="${QUANTIZATION:-bf16}"
export CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-${QUANTIZATION}}"
export SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/checkpoints/quant_transfer}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"

has_option() {
    local wanted="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "${wanted}" ]]; then
            return 0
        fi
    done
    return 1
}

extra_args=("$@")
if [[ "${policy}" == "depth" ]] && ! has_option --rl_token_model_path "${extra_args[@]}"; then
    model_name="${BASE_MODEL_PATH%/}"
    model_name="${model_name##*/}"
    default_size_policy="${SAVE_PATH}/${model_name}/${CHECKPOINT_VARIANT}/size/final.zip"
    if [[ -f "${default_size_policy}" ]]; then
        extra_args=(--rl_token_model_path "${default_size_policy}" "${extra_args[@]}")
    else
        echo "Warning: size policy not found at ${default_size_policy}; depth training will use fixed 60-token observations." >&2
    fi
fi

bash "${REPO_ROOT}/scripts/run_qwen3_32b_train.sh" "${policy}" "${extra_args[@]}"
