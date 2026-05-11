#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_qwen3_32b_train.sh <size|depth> [extra train args...]

Environment:
  EA_MODEL_PATH   Optional override for the Qwen3 EAGLE draft model.
  BASE_MODEL_PATH Optional. Defaults to Qwen/Qwen3-32B.
  DATA_DIR        Optional. Defaults to ./eagle/data.
  DATASET_TRAIN   Optional. Defaults to humaneval.
  SAVE_PATH       Optional. Defaults to ./checkpoints.
  GPU_INDEX       Optional. Defaults to 0.
  QUANTIZATION    Optional. fp16, bf16, int8, or int4. Defaults to int8,
                  matching this wrapper's original behavior.
  DEVICE_MAP      Optional. Defaults to auto.
  CHECKPOINT_VARIANT Optional checkpoint subdirectory. Use this to keep
                  bf16/int8/int4 policies separate.
  BNB_4BIT_COMPUTE_DTYPE Optional. Defaults to bf16.
  BNB_4BIT_QUANT_TYPE    Optional. Defaults to nf4.
  BNB_4BIT_USE_DOUBLE_QUANT Optional. Defaults to 1.
  AUTO_RESUME     Optional. Defaults to 1. When enabled, resumes from the
                  newest step checkpoint for the requested policy.
  CHECKPOINT_FREQ Optional. Defaults to 5000 timesteps.
  HF_HOME         Optional. Defaults to <repo>/.cache/huggingface.
  HF_HUB_CACHE    Optional. Defaults to $HF_HOME/hub.
  TRANSFORMERS_CACHE Optional. Defaults to $HF_HUB_CACHE.

This wrapper keeps the repo's original train_size.sh/train_depth.sh flow and
adds the model IDs/flags needed for a Qwen3 run.

Default draft model:
  AngelSlim/Qwen3-32B_eagle3
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

policy="$1"
shift
extra_args=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-32B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-32B_eagle3}"
DATA_DIR="${DATA_DIR:-./eagle/data}"
DATASET_TRAIN="${DATASET_TRAIN:-humaneval}"
SAVE_PATH="${SAVE_PATH:-./checkpoints}"
GPU_INDEX="${GPU_INDEX:-0}"
QUANTIZATION="${QUANTIZATION:-int8}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-}"
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bf16}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE

case "${QUANTIZATION}" in
    fp16|bf16|int8|int4) ;;
    *) echo "QUANTIZATION must be one of: fp16, bf16, int8, int4" >&2; exit 1 ;;
esac

mkdir -p "${HF_HUB_CACHE}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

cd "${REPO_ROOT}"

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

latest_step_checkpoint() {
    local checkpoint_dir="$1"
    local latest_path=""
    local latest_step=-1
    local file filename step

    shopt -s nullglob
    for file in "${checkpoint_dir}"/step_*.zip; do
        filename="${file##*/}"
        step="${filename#step_}"
        step="${step%.zip}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
            latest_step="${step}"
            latest_path="${file}"
        fi
    done
    shopt -u nullglob

    if [[ -n "${latest_path}" ]]; then
        printf '%s\n' "${latest_path}"
    fi
}

model_name="${BASE_MODEL_PATH%/}"
model_name="${model_name##*/}"
if [[ -n "${CHECKPOINT_VARIANT}" ]]; then
    checkpoint_dir="${SAVE_PATH}/${model_name}/${CHECKPOINT_VARIANT}/${policy}"
else
    checkpoint_dir="${SAVE_PATH}/${model_name}/${policy}"
fi
resume_checkpoint=""

if [[ "${AUTO_RESUME}" != "0" ]] && ! has_option --rl_checkpoint_path "${extra_args[@]}"; then
    resume_checkpoint="$(latest_step_checkpoint "${checkpoint_dir}")"
fi

common_args=(
    --gpu "${GPU_INDEX}"
    --base_model_path "${BASE_MODEL_PATH}"
    --ea_model_path "${EA_MODEL_PATH}"
    --data_dir "${DATA_DIR}"
    --dataset_train "${DATASET_TRAIN}"
    --save_path "${SAVE_PATH}"
    --quantization "${QUANTIZATION}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
    common_args+=(--device_map "${DEVICE_MAP}")
fi

if [[ -n "${CHECKPOINT_VARIANT}" ]]; then
    common_args+=(--checkpoint_variant "${CHECKPOINT_VARIANT}")
fi

if [[ "${QUANTIZATION}" == "int4" ]]; then
    common_args+=(
        --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}"
        --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE}"
    )
    if [[ "${BNB_4BIT_USE_DOUBLE_QUANT}" == "0" || "${BNB_4BIT_USE_DOUBLE_QUANT}" == "false" ]]; then
        common_args+=(--no_bnb_4bit_use_double_quant)
    fi
fi

if [[ -n "${resume_checkpoint}" ]]; then
    common_args+=(--rl_checkpoint_path "${resume_checkpoint}")
fi

if ! has_option --eval_freq "${extra_args[@]}"; then
    common_args+=(--eval_freq "${CHECKPOINT_FREQ}")
fi

echo "Qwen3 training configuration:"
echo "  Base model: ${BASE_MODEL_PATH}"
echo "  EA model:   ${EA_MODEL_PATH}"
echo "  Data dir:   ${DATA_DIR}"
echo "  Dataset:    ${DATASET_TRAIN}"
echo "  Save path:  ${SAVE_PATH}"
echo "  Quantization: ${QUANTIZATION}"
echo "  Device map: ${DEVICE_MAP:-none}"
echo "  Checkpoint variant: ${CHECKPOINT_VARIANT:-none}"
echo "  Auto resume: ${AUTO_RESUME}"
if [[ -n "${resume_checkpoint}" ]]; then
    echo "  Resume checkpoint: ${resume_checkpoint}"
else
    echo "  Resume checkpoint: none"
fi
echo "  Checkpoint freq: ${CHECKPOINT_FREQ}"
echo "  HF cache:   ${HF_HUB_CACHE}"

case "${policy}" in
    size)
        bash train_size.sh "${common_args[@]}" "${extra_args[@]}"
        ;;
    depth)
        bash train_depth.sh "${common_args[@]}" "${extra_args[@]}"
        ;;
    *)
        usage
        exit 1
        ;;
esac
