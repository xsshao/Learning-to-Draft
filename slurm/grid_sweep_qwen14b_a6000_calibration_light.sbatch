#!/usr/bin/env bash
#SBATCH --job-name=grid_qwen14b_a6000_light
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-14B}"
export EA_MODEL_PATH="${EA_MODEL_PATH:-AngelSlim/Qwen3-14B_eagle3}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/grid_sweep/Qwen3-14B/a6000_bf16_light}"
export DATASETS="${DATASETS:-humaneval}"
export NUM_SAMPLES="${NUM_SAMPLES:-10}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export QUANTIZATION="${QUANTIZATION:-bf16}"
export DEVICE_MAP="${DEVICE_MAP:-auto}"
export DEPTHS="${DEPTHS:-1 3 8 12}"
export SIZES="${SIZES:-10 60 220}"

bash "${REPO_ROOT}/scripts/run_grid_sweep.sh"
