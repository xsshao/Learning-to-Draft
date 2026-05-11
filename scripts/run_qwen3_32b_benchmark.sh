#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/Qwen3-32B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-32B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-32B_eagle3}"
SIZE_MODEL_PATH="${SIZE_MODEL_PATH:-${REPO_ROOT}/checkpoints/Qwen3-32B/size/final.zip}"
DEPTH_MODEL_PATH="${DEPTH_MODEL_PATH:-${REPO_ROOT}/checkpoints/Qwen3-32B/depth/final.zip}"
DATASETS="${DATASETS:-alpaca bbh gsm8k humaneval leetcode math mmlu mt_bench qa sum theoremqa}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda}"
if [[ -z "${QUANTIZATION:-}" ]]; then
    if [[ "${LOAD_IN_8BIT:-1}" != "0" ]]; then
        QUANTIZATION="int8"
    else
        QUANTIZATION="fp16"
    fi
fi
DEVICE_MAP="${DEVICE_MAP:-auto}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

case "${QUANTIZATION}" in
    fp16|bf16|int8|int4) ;;
    *) echo "QUANTIZATION must be one of: fp16, bf16, int8, int4" >&2; exit 1 ;;
esac

read -r -a DATASET_ARRAY <<< "${DATASETS}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    task_index=$((SLURM_ARRAY_TASK_ID - 1))
    if (( task_index < 0 || task_index >= ${#DATASET_ARRAY[@]} )); then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range for ${#DATASET_ARRAY[@]} datasets." >&2
        exit 1
    fi

    selected_dataset="${DATASET_ARRAY[task_index]}"
    DATASETS="${selected_dataset}"
    RESULTS_DIR="${RESULTS_DIR}/${selected_dataset}"
fi

mkdir -p "${RESULTS_DIR}"
cd "${REPO_ROOT}"

echo "Qwen3-32B benchmark configuration:"
echo "  Base model:   ${BASE_MODEL_PATH}"
echo "  EA model:     ${EA_MODEL_PATH}"
echo "  Size model:   ${SIZE_MODEL_PATH}"
echo "  Depth model:  ${DEPTH_MODEL_PATH}"
echo "  Datasets:     ${DATASETS}"
echo "  Num samples:  ${NUM_SAMPLES}"
echo "  Temperature:  ${TEMPERATURE}"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Quantization: ${QUANTIZATION}"
echo "  Device map:   ${DEVICE_MAP}"
echo "  Warmup runs:  ${WARMUP_RUNS}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "  Array task:   ${SLURM_ARRAY_TASK_ID}/${#DATASET_ARRAY[@]}"
fi

benchmark_args=(
    --base_model "${BASE_MODEL_PATH}"
    --ea_model "${EA_MODEL_PATH}"
    --size_model "${SIZE_MODEL_PATH}"
    --depth_model "${DEPTH_MODEL_PATH}"
    --datasets "${DATASETS}"
    --num_samples "${NUM_SAMPLES}"
    --temperature "${TEMPERATURE}"
    --output_dir "${RESULTS_DIR}"
    --device "${DEVICE}"
    --quantization "${QUANTIZATION}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
    benchmark_args+=(--device_map "${DEVICE_MAP}")
fi

benchmark_args+=(--warmup_runs "${WARMUP_RUNS}")

bash "${REPO_ROOT}/evaluate/run_benchmark.sh" \
    "${benchmark_args[@]}" \
    "$@"

shopt -s nullglob
result_files=("${RESULTS_DIR}"/benchmark_results_*.json)
shopt -u nullglob

if [[ ${#result_files[@]} -eq 0 ]]; then
    echo "No benchmark results JSON found in ${RESULTS_DIR}" >&2
    exit 1
fi

latest_json="$(ls -t "${result_files[@]}" | head -n 1)"
echo "Analyzing ${latest_json}"

if ! python "${REPO_ROOT}/evaluate/analyze_results.py" \
    --results_file "${latest_json}" \
    --plot \
    --plot_dir "${RESULTS_DIR}"; then
    echo "Warning: benchmark completed, but post-run analysis failed for ${latest_json}" >&2
fi
