#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/run_qwen3_14b_hardware_transfer_benchmark.sh [extra benchmark args...]

Benchmarks Qwen3-14B LTD policies trained under different GPU/runtime settings
on the current benchmark GPU.

Environment:
  BASE_MODEL_PATH      Optional. Defaults to Qwen/Qwen3-14B.
  EA_MODEL_PATH        Optional. Defaults to AngelSlim/Qwen3-14B_eagle3.
  POLICY_LABELS        Optional. Defaults to "h100_bf16 a5090_bf16".
  POLICY_LABEL         Optional. If set, benchmark only this policy label.
  EVAL_QUANTIZATIONS   Optional. Defaults to bf16.
  DATASETS             Optional. Defaults to "humaneval alpaca gsm8k mt_bench".
  NUM_SAMPLES          Optional. Defaults to 50.
  RESULTS_ROOT         Optional. Defaults to ./evaluate/results/hardware_transfer.
  BENCHMARK_GPU_TAG    Optional requested GPU label for output paths, e.g. rtxa6000.
  DEVICE_MAP           Optional. Defaults to auto.
  RUN_ANALYSIS         Optional. Defaults to 1.

Known policy labels:
  h100_bf16    checkpoints/quant_transfer/Qwen3-14B/bf16/{size,depth}/final.zip
  a5090_bf16   checkpoints/Qwen3-14B/{size,depth}/final.zip

Example:
  BENCHMARK_GPU_TAG=rtxa6000 EVAL_QUANTIZATIONS=bf16 \\
    bash scripts/run_qwen3_14b_hardware_transfer_benchmark.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-14B}"
EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-14B_eagle3}"
MODEL_NAME="${BASE_MODEL_PATH%/}"
MODEL_NAME="${MODEL_NAME##*/}"
DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-cuda}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/evaluate/results/hardware_transfer}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
REQUIRE_POLICIES="${REQUIRE_POLICIES:-1}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-qwen3_14b_hardware_transfer}"
BENCHMARK_GPU_TAG="${BENCHMARK_GPU_TAG:-unknown_gpu}"
EVAL_QUANTIZATIONS="${EVAL_QUANTIZATIONS:-${EVAL_QUANTIZATION:-bf16}}"

if [[ -n "${POLICY_LABEL:-}" ]]; then
    POLICY_LABELS="${POLICY_LABEL}"
else
    POLICY_LABELS="${POLICY_LABELS:-h100_bf16 a5090_bf16}"
fi

sanitize_tag() {
    local raw="$1"
    local clean
    clean="$(printf '%s' "${raw}" | tr ' /:' '___' | tr -cs 'A-Za-z0-9_.=-' '_')"
    clean="${clean##_}"
    clean="${clean%%_}"
    printf '%s' "${clean:-unknown}"
}

resolve_policy_paths() {
    local label="$1"
    case "${label}" in
        h100_bf16|h100)
            POLICY_TRAIN_GPU="h100"
            SIZE_MODEL_PATH="${H100_BF16_SIZE_MODEL_PATH:-${REPO_ROOT}/checkpoints/quant_transfer/${MODEL_NAME}/bf16/size/final.zip}"
            DEPTH_MODEL_PATH="${H100_BF16_DEPTH_MODEL_PATH:-${REPO_ROOT}/checkpoints/quant_transfer/${MODEL_NAME}/bf16/depth/final.zip}"
            ;;
        a5090_bf16|a5090)
            POLICY_TRAIN_GPU="a5090"
            SIZE_MODEL_PATH="${A5090_SIZE_MODEL_PATH:-${REPO_ROOT}/checkpoints/${MODEL_NAME}/size/final.zip}"
            DEPTH_MODEL_PATH="${A5090_DEPTH_MODEL_PATH:-${REPO_ROOT}/checkpoints/${MODEL_NAME}/depth/final.zip}"
            ;;
        *)
            echo "Unknown POLICY_LABEL='${label}'. Known labels: h100_bf16, a5090_bf16." >&2
            exit 1
            ;;
    esac
}

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/ensure_ltd_env.sh"

read -r -a POLICY_ARRAY <<< "${POLICY_LABELS}"
read -r -a QUANT_ARRAY <<< "${EVAL_QUANTIZATIONS}"
gpu_tag_safe="$(sanitize_tag "${BENCHMARK_GPU_TAG}")"

for eval_quantization in "${QUANT_ARRAY[@]}"; do
    case "${eval_quantization}" in
        fp16|bf16|int8|int4) ;;
        *) echo "Unknown eval quantization: ${eval_quantization}" >&2; exit 1 ;;
    esac

    for policy_label in "${POLICY_ARRAY[@]}"; do
        resolve_policy_paths "${policy_label}"

        if [[ "${REQUIRE_POLICIES}" != "0" ]]; then
            if [[ ! -f "${SIZE_MODEL_PATH}" ]]; then
                echo "Missing size policy for ${policy_label}: ${SIZE_MODEL_PATH}" >&2
                exit 1
            fi
            if [[ ! -f "${DEPTH_MODEL_PATH}" ]]; then
                echo "Missing depth policy for ${policy_label}: ${DEPTH_MODEL_PATH}" >&2
                exit 1
            fi
        fi

        output_dir="${RESULTS_ROOT}/${MODEL_NAME}/${eval_quantization}/${gpu_tag_safe}/${policy_label}"
        mkdir -p "${output_dir}"

        echo ""
        echo "Qwen3-14B hardware-transfer benchmark:"
        echo "  Policy label:      ${policy_label}"
        echo "  Policy train GPU:  ${POLICY_TRAIN_GPU}"
        echo "  Eval quantization: ${eval_quantization}"
        echo "  Benchmark GPU tag: ${BENCHMARK_GPU_TAG}"
        echo "  Size policy:       ${SIZE_MODEL_PATH}"
        echo "  Depth policy:      ${DEPTH_MODEL_PATH}"
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
            --experiment_label "${EXPERIMENT_LABEL}" \
            --policy_label "${policy_label}" \
            --policy_train_gpu "${POLICY_TRAIN_GPU}" \
            --benchmark_gpu_tag "${BENCHMARK_GPU_TAG}" \
            "$@"
    done

    if [[ "${RUN_ANALYSIS}" != "0" ]]; then
        analysis_root="${RESULTS_ROOT}/${MODEL_NAME}/${eval_quantization}/${gpu_tag_safe}"
        python "${REPO_ROOT}/evaluate/analyze_quant_transfer.py" \
            --results_root "${analysis_root}" \
            --output_csv "${analysis_root}/summary.csv"
    fi
done
