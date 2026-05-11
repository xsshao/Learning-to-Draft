#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/run_vicuna_train.sh <size|depth> [extra train args...]

This wrapper keeps the original repo training flow intact and only swaps in
Vicuna model IDs plus optional 8-bit loading for single-GPU runs.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

policy="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-lmsys/vicuna-13b-v1.3}"
EA_MODEL_PATH="${EA_MODEL_PATH:-yuhuili/EAGLE3-Vicuna1.3-13B}"
DATA_DIR="${DATA_DIR:-./eagle/data}"
DATASET_TRAIN="${DATASET_TRAIN:-humaneval}"
SAVE_PATH="${SAVE_PATH:-./checkpoints}"
GPU_INDEX="${GPU_INDEX:-0}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

cd "${REPO_ROOT}"

case "${policy}" in
    size)
        bash train_size.sh \
            --gpu "${GPU_INDEX}" \
            --base_model_path "${BASE_MODEL_PATH}" \
            --ea_model_path "${EA_MODEL_PATH}" \
            --data_dir "${DATA_DIR}" \
            --dataset_train "${DATASET_TRAIN}" \
            --save_path "${SAVE_PATH}" \
            --load_in_8bit \
            --device_map auto \
            "$@"
        ;;
    depth)
        bash train_depth.sh \
            --gpu "${GPU_INDEX}" \
            --base_model_path "${BASE_MODEL_PATH}" \
            --ea_model_path "${EA_MODEL_PATH}" \
            --data_dir "${DATA_DIR}" \
            --dataset_train "${DATASET_TRAIN}" \
            --save_path "${SAVE_PATH}" \
            --load_in_8bit \
            --device_map auto \
            "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac
