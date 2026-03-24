#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
gpu=0
base_model_path="meta-llama/Llama-3.1-8B-Instruct"
ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
rl_token_model_path=""
rl_checkpoint_path=""
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints"

# RL Hyperparameters (paper Appendix A.3)
total_timesteps=1000000
n_steps=2048
batch_size=256
n_epochs=20
lr=1e-3
ent_coef=0.01
gamma=0.999
pi_arch="1024"
vf_arch="1024 256"

# ── Argument Parsing ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash train_depth.sh [OPTIONS]

Options:
  --gpu                  GPU index (default: $gpu)
  --base_model_path      Base LLM path or HF model ID (default: $base_model_path)
  --ea_model_path        Eagle3 model path or HF model ID (default: $ea_model_path)
  --rl_token_model_path  Size policy checkpoint zip for obs (default: use 60 tokens)
  --rl_checkpoint_path   Resume from checkpoint zip (default: start from scratch)
  --data_dir             Data directory (default: $data_dir)
  --dataset_train        Training dataset name (default: $dataset_train)
  --save_path            Checkpoint save directory (default: $save_path)
  --total_timesteps      Total RL timesteps (default: $total_timesteps)
  --n_steps              Rollout buffer size (default: $n_steps)
  --batch_size           Minibatch size (default: $batch_size)
  --n_epochs             PPO epochs per update (default: $n_epochs)
  --lr                   Learning rate (default: $lr)
  --ent_coef             Entropy coefficient (default: $ent_coef)
  --gamma                Discount factor (default: $gamma)
  --pi_arch              Policy network hidden sizes, space-separated (default: "$pi_arch")
  --vf_arch              Value network hidden sizes, space-separated (default: "$vf_arch")
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)                gpu="$2";                  shift 2 ;;
        --base_model_path)    base_model_path="$2";      shift 2 ;;
        --ea_model_path)      ea_model_path="$2";        shift 2 ;;
        --rl_token_model_path) rl_token_model_path="$2"; shift 2 ;;
        --rl_checkpoint_path) rl_checkpoint_path="$2";  shift 2 ;;
        --data_dir)           data_dir="$2";             shift 2 ;;
        --dataset_train)      dataset_train="$2";        shift 2 ;;
        --save_path)          save_path="$2";            shift 2 ;;
        --total_timesteps)    total_timesteps="$2";      shift 2 ;;
        --n_steps)            n_steps="$2";              shift 2 ;;
        --batch_size)         batch_size="$2";           shift 2 ;;
        --n_epochs)           n_epochs="$2";             shift 2 ;;
        --lr)                 lr="$2";                   shift 2 ;;
        --ent_coef)           ent_coef="$2";             shift 2 ;;
        --gamma)              gamma="$2";                shift 2 ;;
        --pi_arch)            pi_arch="$2";              shift 2 ;;
        --vf_arch)            vf_arch="$2";              shift 2 ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$gpu

# ── Launch ────────────────────────────────────────────────────────────────────
# shellcheck disable=SC2086
python3 -m rl.rl_depth \
    --base_model_path "${base_model_path}" \
    --ea_model_path "${ea_model_path}" \
    --rl_token_model_path "${rl_token_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir "${data_dir}" \
    --dataset_train "${dataset_train}" \
    --save_path "${save_path}" \
    --total_timesteps ${total_timesteps} \
    --n_steps ${n_steps} \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --ent_coef ${ent_coef} \
    --gamma ${gamma} \
    --pi_arch ${pi_arch} \
    --vf_arch ${vf_arch}
