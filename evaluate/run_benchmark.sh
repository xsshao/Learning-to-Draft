#!/bin/bash
# Quick start benchmark script for RL-enhanced speculative decoding

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/benchmark_rl_inference.py"

# Default values
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
EA_MODEL="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
SIZE_MODEL_PATH=""
DEPTH_MODEL_PATH=""
DATA_DIR="${REPO_ROOT}/eagle/data"
DATASET_NAMES="humaneval"
NUM_SAMPLES=20
TEMPERATURE=0.0
OUTPUT_DIR="${SCRIPT_DIR}/results"
DEVICE="cuda"
LOAD_IN_8BIT=0
QUANTIZATION="fp16"
DEVICE_MAP=""
WARMUP_RUNS=3
EXPERIMENT_LABEL=""
POLICY_LABEL=""
POLICY_TRAIN_GPU=""
BENCHMARK_GPU_TAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --ea_model)
            EA_MODEL="$2"
            shift 2
            ;;
        --size_model)
            SIZE_MODEL_PATH="$2"
            shift 2
            ;;
        --depth_model)
            DEPTH_MODEL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --datasets)
            DATASET_NAMES="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --load_in_8bit)
            LOAD_IN_8BIT=1
            QUANTIZATION="int8"
            shift 1
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --device_map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        --warmup_runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --experiment_label)
            EXPERIMENT_LABEL="$2"
            shift 2
            ;;
        --policy_label)
            POLICY_LABEL="$2"
            shift 2
            ;;
        --policy_train_gpu)
            POLICY_TRAIN_GPU="$2"
            shift 2
            ;;
        --benchmark_gpu_tag)
            BENCHMARK_GPU_TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --base_model MODEL      Base model path or HF model ID"
            echo "  --ea_model MODEL        EA model path or HF model ID"
            echo "  --size_model PATH       Path to size RL model checkpoint (.zip)"
            echo "  --depth_model PATH      Path to depth RL model checkpoint (.zip)"
            echo "  --data_dir DIR          Dataset root directory (default: ${REPO_ROOT}/eagle/data)"
            echo "  --datasets NAMES        Dataset names space-separated (default: humaneval)"
            echo "  --num_samples N         Number of samples per dataset (default: 20)"
            echo "  --temperature FLOAT     Sampling temperature (default: 0.0)"
            echo "  --output_dir DIR        Output directory for results (default: ${SCRIPT_DIR}/results)"
            echo "  --device DEVICE         Device to run on (default: cuda)"
            echo "  --load_in_8bit          Deprecated alias for --quantization int8"
            echo "  --quantization MODE     fp16, bf16, int8, or int4 (default: fp16)"
            echo "  --device_map VALUE      Optional Hugging Face device_map value"
            echo "  --warmup_runs N        Untimed warmup passes on first prompt (default: 3, 0=off)"
            echo "  --experiment_label TEXT Optional label stored in result config"
            echo "  --policy_label TEXT     Optional policy label stored in result config"
            echo "  --policy_train_gpu TEXT Optional policy-training GPU label stored in result config"
            echo "  --benchmark_gpu_tag TEXT Optional requested benchmark GPU label stored in result config"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case "$QUANTIZATION" in
    fp16|bf16|int8|int4) ;;
    *) echo "--quantization must be one of: fp16, bf16, int8, int4" >&2; exit 1 ;;
esac

# Print configuration
echo "=================================="
echo "🚀 RL-Enhanced Inference Benchmark"
echo "=================================="
echo "Base Model:        $BASE_MODEL"
echo "EA Model:          $EA_MODEL"
echo "Size Model:        ${SIZE_MODEL_PATH:-'(not provided)'}"
echo "Depth Model:       ${DEPTH_MODEL_PATH:-'(not provided)'}"
echo "Datasets:          $DATASET_NAMES"
echo "Samples per DS:    $NUM_SAMPLES"
echo "Output Directory:  $OUTPUT_DIR"
echo "Quantization:      $QUANTIZATION"
echo "8-bit Alias Used:  $LOAD_IN_8BIT"
echo "Device Map:        ${DEVICE_MAP:-'(default)'}"
echo "Warmup runs:       $WARMUP_RUNS"
echo "Experiment label:  ${EXPERIMENT_LABEL:-'(none)'}"
echo "Policy label:      ${POLICY_LABEL:-'(none)'}"
echo "Policy train GPU:  ${POLICY_TRAIN_GPU:-'(none)'}"
echo "Benchmark GPU tag: ${BENCHMARK_GPU_TAG:-'(none)'}"
echo "=================================="

mkdir -p "$OUTPUT_DIR"

# Build command
read -r -a DATASET_ARGS <<< "$DATASET_NAMES"
CMD=(
    python "$PYTHON_SCRIPT"
    --base_model_path "$BASE_MODEL"
    --ea_model_path "$EA_MODEL"
    --data_dir "$DATA_DIR"
    --dataset_names "${DATASET_ARGS[@]}"
    --num_samples "$NUM_SAMPLES"
    --temperature "$TEMPERATURE"
    --output_dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --quantization "$QUANTIZATION"
)

if [ -n "$SIZE_MODEL_PATH" ]; then
    CMD+=(--size_model_path "$SIZE_MODEL_PATH")
fi

if [ -n "$DEPTH_MODEL_PATH" ]; then
    CMD+=(--depth_model_path "$DEPTH_MODEL_PATH")
fi

if [ -n "$DEVICE_MAP" ]; then
    CMD+=(--device_map "$DEVICE_MAP")
fi

CMD+=(--warmup_runs "$WARMUP_RUNS")

if [ -n "$EXPERIMENT_LABEL" ]; then
    CMD+=(--experiment_label "$EXPERIMENT_LABEL")
fi

if [ -n "$POLICY_LABEL" ]; then
    CMD+=(--policy_label "$POLICY_LABEL")
fi

if [ -n "$POLICY_TRAIN_GPU" ]; then
    CMD+=(--policy_train_gpu "$POLICY_TRAIN_GPU")
fi

if [ -n "$BENCHMARK_GPU_TAG" ]; then
    CMD+=(--benchmark_gpu_tag "$BENCHMARK_GPU_TAG")
fi

# Run benchmark
echo ""
echo "Starting benchmark..."
printf '%q ' "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "✅ Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
