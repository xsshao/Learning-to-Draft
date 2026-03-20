#!/bin/bash
# Quick start benchmark script for RL-enhanced speculative decoding

set -e

# Default values
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
EA_MODEL="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
SIZE_MODEL_PATH=""
DEPTH_MODEL_PATH=""
DATA_DIR="./eagle/data"
DATASET_NAMES="humaneval"
NUM_SAMPLES=20
TEMPERATURE=0.0
OUTPUT_DIR="./evaluate/results"
DEVICE="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size_model)
            SIZE_MODEL_PATH="$2"
            shift 2
            ;;
        --depth_model)
            DEPTH_MODEL_PATH="$2"
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --size_model PATH       Path to size RL model checkpoint (.zip)"
            echo "  --depth_model PATH      Path to depth RL model checkpoint (.zip)"
            echo "  --datasets NAMES        Dataset names space-separated (default: humaneval)"
            echo "  --num_samples N         Number of samples per dataset (default: 20)"
            echo "  --output_dir DIR        Output directory for results (default: ./evaluate/results)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
echo "=================================="

# Build command
CMD="python evaluate/benchmark_rl_inference.py"
CMD="$CMD --base_model_path $BASE_MODEL"
CMD="$CMD --ea_model_path $EA_MODEL"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --dataset_names $DATASET_NAMES"
CMD="$CMD --num_samples $NUM_SAMPLES"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --device $DEVICE"

if [ -n "$SIZE_MODEL_PATH" ]; then
    CMD="$CMD --size_model_path $SIZE_MODEL_PATH"
fi

if [ -n "$DEPTH_MODEL_PATH" ]; then
    CMD="$CMD --depth_model_path $DEPTH_MODEL_PATH"
fi

# Run benchmark
echo ""
echo "Starting benchmark..."
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "✅ Benchmark complete!"
echo "Results saved to: $OUTPUT_DIR"
