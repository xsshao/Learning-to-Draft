export CUDA_VISIBLE_DEVICES=$1

# Model Paths
base_model_path="meta-llama/Llama-3.1-8B-Instruct"
ea_model_path="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
rl_checkpoint_path=""  # Leave empty to start from scratch

# Data and Save Directories
data_dir="./eagle/data"
dataset_train="humaneval"
save_path="./checkpoints"

# RL Hyperparameters (paper Appendix A.3)
total_timesteps=100000
n_steps=2048        # rollout buffer size
batch_size=256      # minibatch size
n_epochs=20         # PPO epochs per update
lr=1e-3
ent_coef=0.01
gamma=0.9

python3 -m rl.rl_total \
    --base_model_path ${base_model_path} \
    --ea_model_path ${ea_model_path} \
    --rl_checkpoint_path "${rl_checkpoint_path}" \
    --data_dir ${data_dir} \
    --dataset_train ${dataset_train} \
    --save_path ${save_path} \
    --total_timesteps ${total_timesteps} \
    --n_steps ${n_steps} \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --ent_coef ${ent_coef} \
    --gamma ${gamma} \
    --pi_arch 1024 256 \
    --vf_arch 1024 256
