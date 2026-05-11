#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_14b_quant_transfer
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=1-3
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

SOURCE_POLICY_VARIANTS="${SOURCE_POLICY_VARIANTS:-bf16 int8 int4}"
read -r -a SOURCE_ARRAY <<< "${SOURCE_POLICY_VARIANTS}"

task_index=$((SLURM_ARRAY_TASK_ID - 1))
if (( task_index < 0 || task_index >= ${#SOURCE_ARRAY[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#SOURCE_ARRAY[@]} source variants." >&2
    exit 1
fi

export MODEL_SIZE="${MODEL_SIZE:-14B}"
export POLICY_VARIANT="${SOURCE_ARRAY[task_index]}"
export EVAL_QUANTIZATIONS="${EVAL_QUANTIZATIONS:-bf16 int8 int4}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/quant_transfer}"
export RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/evaluate/results/quant_transfer}"
export DATASETS="${DATASETS:-humaneval alpaca gsm8k mt_bench}"
export NUM_SAMPLES="${NUM_SAMPLES:-50}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export PYTHONUNBUFFERED=1

echo "Benchmarking ${POLICY_VARIANT} policy across ${EVAL_QUANTIZATIONS}"
bash "${REPO_ROOT}/scripts/run_qwen3_quant_transfer_benchmark.sh" "$@"
