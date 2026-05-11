#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_14b_quant_train
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

POLICY="${POLICY:-size}"
QUANTIZATIONS="${QUANTIZATIONS:-bf16 int8 int4}"
read -r -a QUANT_ARRAY <<< "${QUANTIZATIONS}"

task_index=$((SLURM_ARRAY_TASK_ID - 1))
if (( task_index < 0 || task_index >= ${#QUANT_ARRAY[@]} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for ${#QUANT_ARRAY[@]} quantizations." >&2
    exit 1
fi

export MODEL_SIZE="${MODEL_SIZE:-14B}"
export QUANTIZATION="${QUANT_ARRAY[task_index]}"
export CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-${QUANTIZATION}}"
export SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/checkpoints/quant_transfer}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"
export PYTHONUNBUFFERED=1

echo "Training Qwen3-${MODEL_SIZE} ${POLICY} policy under ${QUANTIZATION}"
bash "${REPO_ROOT}/scripts/run_qwen3_quant_train.sh" "${POLICY}" "$@"
