#!/usr/bin/env bash
#SBATCH --job-name=ltd_vicuna_benchmark
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:RTXA6000:1
#SBATCH --exclude=ccc0424
#SBATCH --output=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.out
#SBATCH --error=/u/sx24/scratch/LTD_reproduce/slurm/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="/u/sx24/scratch/LTD_reproduce"
mkdir -p "${REPO_ROOT}/slurm/logs"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

bash "${REPO_ROOT}/scripts/run_vicuna_benchmark.sh" "$@"
