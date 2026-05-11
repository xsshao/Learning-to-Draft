#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/run_qwen3_quant_transfer_benchmark.sh [extra benchmark args...]

Runs one source LTD policy across one or more evaluation precisions.

Environment:
  MODEL_SIZE       Optional. Defaults to 14B.
  BASE_MODEL_PATH  Optional. Defaults to Qwen/Qwen3-\${MODEL_SIZE}.
  EA_MODEL_PATH    Optional. Defaults to AngelSlim/Qwen3-\${MODEL_SIZE}_eagle3.
  CHECKPOINT_ROOT  Optional. Defaults to ./checkpoints/quant_transfer.
  POLICY_VARIANT   Optional. Defaults to bf16.
  EVAL_QUANTIZATIONS Optional. Defaults to "bf16 int8 int4".
  DATASETS         Optional. Defaults to "humaneval alpaca gsm8k mt_bench".
  NUM_SAMPLES      Optional. Defaults to 50.
  RESULTS_ROOT     Optional. Defaults to ./evaluate/results/quant_transfer.
  REQUIRE_POLICIES Optional. Defaults to 1. Set 0 to allow baseline-only runs.
  RUN_ANALYSIS     Optional. Defaults to 1.

Example:
  POLICY_VARIANT=bf16 EVAL_QUANTIZATIONS="bf16 int8 int4" \\
    bash scripts/run_qwen3_quant_transfer_benchmark.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="${MODEL_SIZE:-14B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-${MODEL_SIZE}}"
EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-${MODEL_SIZE}_eagle3}"
MODEL_NAME="${BASE_MODEL_PATH%/}"
MODEL_NAME="${MODEL_NAME##*/}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/quant_transfer}"
POLICY_VARIANT="${POLICY_VARIANT:-bf16}"
EVAL_QUANTIZATIONS="${EVAL_QUANTIZATIONS:-bf16 int8 int4}"
DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/evaluate/results/quant_transfer}"
REQUIRE_POLICIES="${REQUIRE_POLICIES:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

SIZE_MODEL_PATH="${SIZE_MODEL_PATH:-${CHECKPOINT_ROOT}/${MODEL_NAME}/${POLICY_VARIANT}/size/final.zip}"
DEPTH_MODEL_PATH="${DEPTH_MODEL_PATH:-${CHECKPOINT_ROOT}/${MODEL_NAME}/${POLICY_VARIANT}/depth/final.zip}"

if [[ "${REQUIRE_POLICIES}" != "0" ]]; then
    if [[ ! -f "${SIZE_MODEL_PATH}" ]]; then
        echo "Missing size policy: ${SIZE_MODEL_PATH}" >&2
        exit 1
    fi
    if [[ ! -f "${DEPTH_MODEL_PATH}" ]]; then
        echo "Missing depth policy: ${DEPTH_MODEL_PATH}" >&2
        exit 1
    fi
fi

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

read -r -a QUANT_ARRAY <<< "${EVAL_QUANTIZATIONS}"

for eval_quantization in "${QUANT_ARRAY[@]}"; do
    case "${eval_quantization}" in
        fp16|bf16|int8|int4) ;;
        *) echo "Unknown eval quantization: ${eval_quantization}" >&2; exit 1 ;;
    esac

    output_dir="${RESULTS_ROOT}/${MODEL_NAME}/${POLICY_VARIANT}_policy/${eval_quantization}"
    mkdir -p "${output_dir}"

    echo ""
    echo "Quantization transfer benchmark:"
    echo "  Model:             ${BASE_MODEL_PATH}"
    echo "  EA model:          ${EA_MODEL_PATH}"
    echo "  Policy variant:    ${POLICY_VARIANT}"
    echo "  Eval quantization: ${eval_quantization}"
    echo "  Results:           ${output_dir}"

    bash "${REPO_ROOT}/evaluate/run_benchmark.sh" \
        --base_model "${BASE_MODEL_PATH}" \
        --ea_model "${EA_MODEL_PATH}" \
        --size_model "${SIZE_MODEL_PATH}" \
        --depth_model "${DEPTH_MODEL_PATH}" \
        --datasets "${DATASETS}" \
        --num_samples "${NUM_SAMPLES}" \
        --temperature "${TEMPERATURE}" \
        --output_dir "${output_dir}" \
        --device "${DEVICE}" \
        --quantization "${eval_quantization}" \
        --device_map "${DEVICE_MAP}" \
        --warmup_runs "${WARMUP_RUNS}" \
        "$@"
done

if [[ "${RUN_ANALYSIS}" != "0" ]]; then
    python "${REPO_ROOT}/evaluate/analyze_quant_transfer.py" \
        --results_root "${RESULTS_ROOT}/${MODEL_NAME}/${POLICY_VARIANT}_policy"
fi
