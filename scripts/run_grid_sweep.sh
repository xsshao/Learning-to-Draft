#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configurable via environment variables (with defaults for Llama-3.1-8B)
BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
EA_MODEL_PATH="${EA_MODEL_PATH:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/grid_sweep}"
DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEPTHS="${DEPTHS:-3 5 8 12}"
SIZES="${SIZES:-20 40 60 100 150 200}"
DEVICE="${DEVICE:-cuda}"
if [[ -z "${QUANTIZATION:-}" ]]; then
    if [[ "${LOAD_IN_8BIT:-1}" != "0" ]]; then
        QUANTIZATION="int8"
    else
        QUANTIZATION="fp16"
    fi
fi
DEVICE_MAP="${DEVICE_MAP:-auto}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

case "${QUANTIZATION}" in
    fp16|bf16|int8|int4) ;;
    *) echo "QUANTIZATION must be one of: fp16, bf16, int8, int4" >&2; exit 1 ;;
esac

read -r -a DATASET_ARRAY <<< "${DATASETS}"

CMD=(
    python "${REPO_ROOT}/evaluate/benchmark_grid_sweep.py"
    --base_model_path "${BASE_MODEL_PATH}"
    --ea_model_path "${EA_MODEL_PATH}"
    --data_dir "${REPO_ROOT}/eagle/data"
    --dataset_names "${DATASET_ARRAY[@]}"
    --num_samples "${NUM_SAMPLES}"
    --temperature "${TEMPERATURE}"
    --depths ${DEPTHS}
    --sizes ${SIZES}
    --output_dir "${RESULTS_DIR}"
    --device "${DEVICE}"
    --quantization "${QUANTIZATION}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
    CMD+=(--device_map "${DEVICE_MAP}")
fi

# SLURM array support: map SLURM_ARRAY_TASK_ID (1-based) to dataset index (0-based)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    dataset_idx=$((SLURM_ARRAY_TASK_ID - 1))
    if (( dataset_idx < 0 || dataset_idx >= ${#DATASET_ARRAY[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#DATASET_ARRAY[@]} datasets." >&2
        exit 1
    fi
    selected="${DATASET_ARRAY[dataset_idx]}"
    RESULTS_DIR="${RESULTS_DIR}/${selected}"
    mkdir -p "${RESULTS_DIR}"
    # Override output_dir and add dataset_index
    CMD=()
    CMD=(
        python "${REPO_ROOT}/evaluate/benchmark_grid_sweep.py"
        --base_model_path "${BASE_MODEL_PATH}"
        --ea_model_path "${EA_MODEL_PATH}"
        --data_dir "${REPO_ROOT}/eagle/data"
        --dataset_names "${DATASET_ARRAY[@]}"
        --dataset_index "${dataset_idx}"
        --num_samples "${NUM_SAMPLES}"
        --temperature "${TEMPERATURE}"
        --depths ${DEPTHS}
        --sizes ${SIZES}
        --output_dir "${RESULTS_DIR}"
        --device "${DEVICE}"
        --quantization "${QUANTIZATION}"
    )
    if [[ -n "${DEVICE_MAP}" ]]; then
        CMD+=(--device_map "${DEVICE_MAP}")
    fi
    echo "SLURM array task ${SLURM_ARRAY_TASK_ID}: running dataset '${selected}'"
fi

mkdir -p "${RESULTS_DIR}"
cd "${REPO_ROOT}"

echo "Grid sweep configuration:"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  EA model:     ${EA_MODEL_PATH}"
echo "  Datasets:     ${DATASETS}"
echo "  Num samples:  ${NUM_SAMPLES}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Depths:       ${DEPTHS}"
echo "  Sizes:        ${SIZES}"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Quantization: ${QUANTIZATION}"
echo "  Device map:   ${DEVICE_MAP}"

echo ""
echo "Command:"
printf '%q ' "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "Grid sweep complete!"
