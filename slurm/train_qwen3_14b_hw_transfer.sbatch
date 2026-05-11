#!/usr/bin/env bash
#SBATCH --job-name=ltd_qwen3_14b_hwproxy_train
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --array=1-1
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export MODEL_SIZE="${MODEL_SIZE:-14B}"
export QUANTIZATION="${QUANTIZATION:-bf16}"
export CHECKPOINT_VARIANT="${CHECKPOINT_VARIANT:-bf16_hwproxy}"
export SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/checkpoints/hardware_transfer}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export REWARD_MODE="${REWARD_MODE:-randomized_proxy_cost}"
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-balanced}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"
export PYTHONUNBUFFERED=1

POLICY="${POLICY:-size}"
echo "Training Qwen3-${MODEL_SIZE} ${POLICY} hardware-transfer policy under ${QUANTIZATION}"
bash "${REPO_ROOT}/scripts/run_qwen3_hw_transfer_train.sh" "${POLICY}" "$@"
