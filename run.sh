base_model=(
    meta-llama/Llama-3.1-8B-Instruct
    deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    Qwen/Qwen3-14B
    lmsys/vicuna-13b-v1.3
)

ea_model=(
    yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
    yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B
    AngelSlim/Qwen3-14B_eagle3
    yuhuili/EAGLE3-Vicuna1.3-13B
)

len=${#base_model[@]}

for ((i=0; i<$len; i++)); do
    echo "Running benchmark for base model: ${base_model[i]} with EA model: ${ea_model[i]}"
    python evaluate/benchmark_rl_inference.py \
        --base_model_path ${base_model[i]} \
        --ea_model_path ${ea_model[i]} || true
done

echo "All benchmarks completed. Committing results to git..."
git add .
git commit -m "Update benchmark results" || true
git push || true

echo "Stopping this instance..."
vastai stop instance $CONTAINER_ID