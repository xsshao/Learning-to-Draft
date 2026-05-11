#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_14b_hwproxy_h100_eval
#SBATCH --account=parasol
#SBATCH --partition=secondary-cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=1-1
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export POLICY_LABELS="${POLICY_LABELS:-bf16_hwproxy}"
export BENCHMARK_GPU_TAG="${BENCHMARK_GPU_TAG:-h100}"
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-${BENCHMARK_GPU_TAG}}"
export EVAL_QUANTIZATIONS="${EVAL_QUANTIZATIONS:-bf16}"
export DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
export NUM_SAMPLES="${NUM_SAMPLES:-50}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
export PYTHONUNBUFFERED=1

echo "Benchmarking hardware-transfer policy '${POLICY_LABELS}' on requested GPU tag '${BENCHMARK_GPU_TAG}'"
bash "${REPO_ROOT}/scripts/run_qwen3_hw_transfer_benchmark.sh" "$@"
