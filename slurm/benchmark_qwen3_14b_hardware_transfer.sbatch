#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_14b_hw_transfer
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --array=1-2
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

POLICY_LABELS="${POLICY_LABELS:-h100_bf16 a5090_bf16}"
read -r -a POLICY_ARRAY <<< "${POLICY_LABELS}"

task_index=$((SLURM_ARRAY_TASK_ID - 1))
if (( task_index < 0 || task_index >= ${#POLICY_ARRAY[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#POLICY_ARRAY[@]} policies." >&2
    exit 1
fi

export POLICY_LABEL="${POLICY_ARRAY[task_index]}"
export BENCHMARK_GPU_TAG="${BENCHMARK_GPU_TAG:-rtxa6000}"
export EVAL_QUANTIZATIONS="${EVAL_QUANTIZATIONS:-bf16}"
export DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
export NUM_SAMPLES="${NUM_SAMPLES:-50}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
export PYTHONUNBUFFERED=1

echo "Benchmarking policy '${POLICY_LABEL}' on requested GPU tag '${BENCHMARK_GPU_TAG}'"
bash "${REPO_ROOT}/scripts/run_qwen3_14b_hardware_transfer_benchmark.sh" "$@"
