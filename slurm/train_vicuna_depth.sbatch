#!/usr/bin/env bash
#SBATCH --job-name=ltd_vicuna_depth
#SBATCH --account=parasol
#SBATCH --partition=parasol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:RTXA5000:1
#SBATCH --output=/u/sx24/LTD_reproduce/slurm/logs/%x_%j.out
#SBATCH --error=/u/sx24/LTD_reproduce/slurm/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/u/sx24/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

bash scripts/run_vicuna_train.sh depth "$@"
