#!/usr/bin/env bash
#SBATCH --job-name=grid_sweep_llama8b
#SBATCH --account=parasol
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
export EA_MODEL_PATH="${EA_MODEL_PATH:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/evaluate/results/grid_sweep/Llama-3.1-8B-Instruct}"
export DATASETS="humaneval alpaca gsm8k mt_bench"
export NUM_SAMPLES=50
export TEMPERATURE=0.0
export LOAD_IN_8BIT=1
export DEVICE_MAP=auto

# meta-llama/* hub IDs are gated. Slurm does not inherit your shell env. Options:
#   A) sbatch --export=ALL,HF_TOKEN="$HF_TOKEN" ...  (after: huggingface-cli login)
#   B) Token file from `huggingface-cli login`: ~/.cache/huggingface/token
#   C) Set BASE_MODEL_PATH to a local weight directory (absolute path) to skip this.
_needs_hf=1
if [[ "${BASE_MODEL_PATH}" == /* || "${BASE_MODEL_PATH}" == ./* || "${BASE_MODEL_PATH}" == ../* ]]; then
  _needs_hf=0
fi
if [[ "${_needs_hf}" -eq 1 ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    for _tok in "${HF_HOME:-$HOME/.cache/huggingface}/token" "${HOME}/.huggingface/token"; do
      if [[ -f "${_tok}" ]]; then
        export HF_TOKEN="$(tr -d '[:space:]' < "${_tok}")"
        break
      fi
    done
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: Need Hugging Face credentials for hub model: ${BASE_MODEL_PATH}" >&2
    echo "  huggingface-cli login   OR   sbatch --export=ALL,HF_TOKEN=\\\$HF_TOKEN ..." >&2
    echo "  Or: BASE_MODEL_PATH=/path/to/local/Llama-3.1-8B-Instruct sbatch ..." >&2
    exit 1
  fi
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

bash "${REPO_ROOT}/scripts/run_grid_sweep.sh"
