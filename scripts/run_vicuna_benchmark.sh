#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/vicuna-13b-v1.3}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-lmsys/vicuna-13b-v1.3}"
EA_MODEL_PATH="${EA_MODEL_PATH:-yuhuili/EAGLE3-Vicuna1.3-13B}"
SIZE_MODEL_PATH="${SIZE_MODEL_PATH:-${REPO_ROOT}/checkpoints/vicuna-13b-v1.3/size/final.zip}"
DEPTH_MODEL_PATH="${DEPTH_MODEL_PATH:-${REPO_ROOT}/checkpoints/vicuna-13b-v1.3/depth/final.zip}"
DATASETS="${DATASETS:-humaneval alpaca mt_bench gsm8k qa}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DEVICE="${DEVICE:-cuda}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

mkdir -p "${RESULTS_DIR}"
cd "${REPO_ROOT}"

bash "${REPO_ROOT}/evaluate/run_benchmark.sh" \
    --base_model "${BASE_MODEL_PATH}" \
    --ea_model "${EA_MODEL_PATH}" \
    --size_model "${SIZE_MODEL_PATH}" \
    --depth_model "${DEPTH_MODEL_PATH}" \
    --datasets "${DATASETS}" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${RESULTS_DIR}" \
    --device "${DEVICE}" \
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

python "${REPO_ROOT}/evaluate/analyze_results.py" \
    --results_file "${latest_json}" \
    --plot \
    --plot_dir "${RESULTS_DIR}"

