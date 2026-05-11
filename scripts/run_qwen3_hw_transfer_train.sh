#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/run_qwen3_hw_transfer_train.sh <size|depth> [extra train args...]

Environment:
  MODEL_SIZE       Optional. Defaults to 14B. Used for Qwen/Qwen3-\${MODEL_SIZE}.
  BASE_MODEL_PATH  Optional override. Defaults to Qwen/Qwen3-\${MODEL_SIZE}.
  EA_MODEL_PATH    Optional override. Defaults to AngelSlim/Qwen3-\${MODEL_SIZE}_eagle3.
  QUANTIZATION     Optional. Defaults to bf16 for the full-precision study.
  CHECKPOINT_VARIANT Optional. Defaults to bf16_hwproxy.
  SAVE_PATH        Optional. Defaults to ./checkpoints/hardware_transfer.
  DEVICE_MAP       Optional. Defaults to auto.
  DATASET_TRAIN    Optional. Defaults to humaneval.
  REWARD_MODE      Optional. Defaults to randomized_proxy_cost.
  HARDWARE_PROFILE Optional. Defaults to balanced.
  HARDWARE_COST_COEFFS Optional coefficient overrides.

Examples:
  bash scripts/run_qwen3_hw_transfer_train.sh size
  bash scripts/run_qwen3_hw_transfer_train.sh depth
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
export CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-bf16_hwproxy}"
export SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/checkpoints/hardware_transfer}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
DATA_DIR="${DATA_DIR:-./eagle/data}"
DATASET_TRAIN="${DATASET_TRAIN:-humaneval}"
GPU_INDEX="${GPU_INDEX:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"
REWARD_MODE="${REWARD_MODE:-randomized_proxy_cost}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-balanced}"
HARDWARE_COST_COEFFS="${HARDWARE_COST_COEFFS:-}"

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
checkpoint_dir="${SAVE_PATH}/${model_name}/${CHECKPOINT_VARIANT}/${policy}"
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
    --checkpoint_variant "${CHECKPOINT_VARIANT}"
    --eval_freq "${CHECKPOINT_FREQ}"
    --reward_mode "${REWARD_MODE}"
    --hardware_profile "${HARDWARE_PROFILE}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
    common_args+=(--device_map "${DEVICE_MAP}")
fi
if [[ -n "${HARDWARE_COST_COEFFS}" ]]; then
    common_args+=(--hardware_cost_coeffs "${HARDWARE_COST_COEFFS}")
fi
if [[ -n "${resume_checkpoint}" ]]; then
    common_args+=(--rl_checkpoint_path "${resume_checkpoint}")
fi

echo "Qwen3 hardware-transfer training configuration:"
echo "  Policy: ${policy}"
echo "  Base model: ${BASE_MODEL_PATH}"
echo "  EA model: ${EA_MODEL_PATH}"
echo "  Quantization: ${QUANTIZATION}"
echo "  Save path: ${SAVE_PATH}"
echo "  Checkpoint variant: ${CHECKPOINT_VARIANT}"
echo "  Reward mode: ${REWARD_MODE}"
echo "  Hardware profile: ${HARDWARE_PROFILE}"
echo "  Cost overrides: ${HARDWARE_COST_COEFFS:-none}"
echo "  Resume checkpoint: ${resume_checkpoint:-none}"

case "${policy}" in
    size)
        bash "${REPO_ROOT}/train_size_hw_transfer.sh" "${common_args[@]}" "${extra_args[@]}"
        ;;
    depth)
        bash "${REPO_ROOT}/train_depth_hw_transfer.sh" "${common_args[@]}" "${extra_args[@]}"
        ;;
esac
