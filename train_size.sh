#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
gpu=0
base_model_path="meta-llama/Llama-3.1-8B-Instruct"
ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
rl_checkpoint_path=""
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints"
load_in_8bit=false
quantization="fp16"
device_map=""
checkpoint_variant=""
bnb_4bit_compute_dtype="bf16"
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=true
eval_freq=10000

# RL Hyperparameters (paper Appendix A.3)
total_timesteps=100000
n_steps=2048
batch_size=256
n_epochs=20
lr=1e-3
ent_coef=0.01
gamma=0.9
pi_arch="1024 256"
vf_arch="1024 256"

# ── Argument Parsing ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash train_size.sh [OPTIONS]

Options:
  --gpu                  GPU index (default: $gpu)
  --base_model_path      Base LLM path or HF model ID (default: $base_model_path)
  --ea_model_path        Eagle3 model path or HF model ID (default: $ea_model_path)
  --rl_checkpoint_path   Resume from checkpoint zip (default: start from scratch)
  --data_dir             Data directory (default: $data_dir)
  --dataset_train        Training dataset name (default: $dataset_train)
  --save_path            Checkpoint save directory (default: $save_path)
  --load_in_8bit         Load the base model in 8-bit to reduce GPU memory usage
  --quantization         Precision/quantization mode: fp16, bf16, int8, int4 (default: $quantization)
  --device_map           Hugging Face device_map value (example: auto)
  --checkpoint_variant   Optional checkpoint subdirectory (example: bf16)
  --bnb_4bit_compute_dtype fp16 or bf16 for 4-bit compute (default: $bnb_4bit_compute_dtype)
  --bnb_4bit_quant_type  fp4 or nf4 for 4-bit quantization (default: $bnb_4bit_quant_type)
  --bnb_4bit_use_double_quant / --no_bnb_4bit_use_double_quant
                         Enable/disable nested 4-bit quantization (default: $bnb_4bit_use_double_quant)
  --eval_freq            Checkpoint save frequency in timesteps (default: $eval_freq)
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
        --gpu)               gpu="$2";                  shift 2 ;;
        --base_model_path)   base_model_path="$2";      shift 2 ;;
        --ea_model_path)     ea_model_path="$2";        shift 2 ;;
        --rl_checkpoint_path) rl_checkpoint_path="$2"; shift 2 ;;
        --data_dir)          data_dir="$2";             shift 2 ;;
        --dataset_train)     dataset_train="$2";        shift 2 ;;
        --save_path)         save_path="$2";            shift 2 ;;
        --load_in_8bit)      load_in_8bit=true;         shift 1 ;;
        --quantization)      quantization="$2";         shift 2 ;;
        --device_map)        device_map="$2";           shift 2 ;;
        --checkpoint_variant) checkpoint_variant="$2";  shift 2 ;;
        --bnb_4bit_compute_dtype) bnb_4bit_compute_dtype="$2"; shift 2 ;;
        --bnb_4bit_quant_type) bnb_4bit_quant_type="$2"; shift 2 ;;
        --bnb_4bit_use_double_quant) bnb_4bit_use_double_quant=true; shift 1 ;;
        --no_bnb_4bit_use_double_quant) bnb_4bit_use_double_quant=false; shift 1 ;;
        --eval_freq)         eval_freq="$2";            shift 2 ;;
        --total_timesteps)   total_timesteps="$2";      shift 2 ;;
        --n_steps)           n_steps="$2";              shift 2 ;;
        --batch_size)        batch_size="$2";           shift 2 ;;
        --n_epochs)          n_epochs="$2";             shift 2 ;;
        --lr)                lr="$2";                   shift 2 ;;
        --ent_coef)          ent_coef="$2";             shift 2 ;;
        --gamma)             gamma="$2";                shift 2 ;;
        --pi_arch)           pi_arch="$2";              shift 2 ;;
        --vf_arch)           vf_arch="$2";              shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$gpu

if [[ "${load_in_8bit}" == "true" ]]; then
    quantization="int8"
fi

case "${quantization}" in
    fp16|bf16|int8|int4) ;;
    *) echo "--quantization must be one of: fp16, bf16, int8, int4" >&2; exit 1 ;;
esac

set -- --quantization "${quantization}" "$@"

if [[ -n "${device_map}" ]]; then
    set -- --device_map "${device_map}" "$@"
fi

if [[ -n "${checkpoint_variant}" ]]; then
    set -- --checkpoint_variant "${checkpoint_variant}" "$@"
fi

if [[ "${quantization}" == "int4" ]]; then
    set -- \
        --bnb_4bit_compute_dtype "${bnb_4bit_compute_dtype}" \
        --bnb_4bit_quant_type "${bnb_4bit_quant_type}" \
        "$@"
    if [[ "${bnb_4bit_use_double_quant}" == "false" ]]; then
        set -- --no_bnb_4bit_use_double_quant "$@"
    fi
fi

# shellcheck disable=SC2086
python3 -m rl.rl_total \
    --base_model_path "${base_model_path}" \
    --ea_model_path "${ea_model_path}" \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir "${data_dir}" \
    --dataset_train "${dataset_train}" \
    --save_path "${save_path}" \
    --eval_freq ${eval_freq} \
    --total_timesteps ${total_timesteps} \
    --n_steps ${n_steps} \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --ent_coef ${ent_coef} \
    --gamma ${gamma} \
    --pi_arch ${pi_arch} \
    --vf_arch ${vf_arch} \
    "$@"
