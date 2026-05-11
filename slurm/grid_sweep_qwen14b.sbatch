#!/usr/bin/env bash
#SBATCH --job-name=grid_sweep_qwen14b
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:H100:1
#SBATCH --exclude=ccc0424
#SBATCH --array=1-4
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export BASE_MODEL_PATH="Qwen/Qwen3-14B"
export EA_MODEL_PATH="AngelSlim/Qwen3-14B_eagle3"
export RESULTS_DIR="${REPO_ROOT}/evaluate/results/grid_sweep/Qwen3-14B"
export DATASETS="humaneval alpaca gsm8k mt_bench"
export NUM_SAMPLES=50
export TEMPERATURE=0.0
export QUANTIZATION="${QUANTIZATION:-int8}"
export DEVICE_MAP=auto

bash "${REPO_ROOT}/scripts/run_grid_sweep.sh"
