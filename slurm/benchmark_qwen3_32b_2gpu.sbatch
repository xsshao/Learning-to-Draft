#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_32b_benchmark_2gpu
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H100:2
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.err
#
# Phase A: Qwen3-32B with HuggingFace device_map=auto across 2 GPUs (pipeline sharding).
# Compare to single-GPU: sbatch slurm/benchmark_qwen3_32b.sbatch
#
# Optional overrides: export RESULTS_DIR, NUM_SAMPLES, WARMUP_RUNS, DATASETS before sbatch

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/Qwen3-32B_2gpu}"
export WARMUP_RUNS="${WARMUP_RUNS:-3}"
# 8-bit + device_map=auto: model shards across both GPUs (set in run_qwen3_32b_benchmark.sh)

bash "${REPO_ROOT}/scripts/run_qwen3_32b_benchmark.sh" "$@"
