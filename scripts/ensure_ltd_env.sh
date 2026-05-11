#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${LTD_VENV_DIR:-$HOME/.venvs/ltd_vicuna}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4 >/dev/null 2>&1 || true
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/.cache/matplotlib}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p "${PIP_CACHE_DIR}"

python -m pip install --upgrade pip setuptools wheel

if ! python - <<'PY'
mods = ["torch", "transformers", "stable_baselines3", "wandb", "gymnasium", "bitsandbytes", "fastchat", "pandas", "matplotlib"]
missing = []
for name in mods:
    try:
        __import__(name)
    except Exception:
        missing.append(name)
print("missing_modules=" + ",".join(missing))
raise SystemExit(1 if missing else 0)
PY
then
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
    tmp_req="$(mktemp)"
    grep -v '^torch$' "${REPO_ROOT}/requirements.txt" > "${tmp_req}"
    python -m pip install -r "${tmp_req}"
    python -m pip install bitsandbytes
    rm -f "${tmp_req}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
