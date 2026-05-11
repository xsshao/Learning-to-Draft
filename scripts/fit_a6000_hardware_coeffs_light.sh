#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GRID_RESULTS="${GRID_RESULTS:-${REPO_ROOT}/evaluate/results/grid_sweep/Qwen3-14B/a6000_bf16_light/**/*.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/evaluate/results/hardware_transfer/Qwen3-14B/bf16/rtxa6000/a6000_light_fitted_coeffs.json}"
DATASETS="${DATASETS:-humaneval}"
GPU_NAME_CONTAINS="${GPU_NAME_CONTAINS:-A6000}"
TARGET_MODE="${TARGET_MODE:-per_cycle}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1e-6}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

read -r -a DATASET_ARRAY <<< "${DATASETS}"

python "${REPO_ROOT}/evaluate/fit_hardware_coeffs.py" \
    --grid_results "${GRID_RESULTS}" \
    --datasets "${DATASET_ARRAY[@]}" \
    --gpu_name_contains "${GPU_NAME_CONTAINS}" \
    --target_mode "${TARGET_MODE}" \
    --ridge_alpha "${RIDGE_ALPHA}" \
    --output_json "${OUTPUT_JSON}" \
    "$@"
