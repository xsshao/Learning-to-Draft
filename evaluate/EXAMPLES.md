# Benchmark 使用示例

## 快速开始（30分钟）

### 1. 最简单的baseline对比
```bash
# 仅评估Baseline和Eagle3（无RL）
python evaluate/benchmark_rl_inference.py \
    --num_samples 10 \
    --dataset_names humaneval
```

输出示例：
```
Dataset: humaneval
------
Baseline            : Tokens  125.3, Time   3.234s, Throughput  38.72 tok/s
Eagle3              : Tokens  125.3, Time   0.456s, Throughput 275.00 tok/s
```

---

## 完整评估（包含RL）

### 2. 单个数据集 + RL模型
```bash
# 单个数据集，包含RL改进
bash evaluate/run_benchmark.sh \
    --size_model ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --datasets humaneval \
    --num_samples 50
```

输出关键部分：
```
Dataset: humaneval
----
Baseline            : Tokens  125.3, Time   3.234s, Throughput   38.72 tok/s
Eagle3              : Tokens  125.3, Time   0.456s, Throughput  275.00 tok/s, Acceptance   5.42
Eagle3+RL           : Tokens  125.3, Time   0.389s, Throughput  322.11 tok/s, Acceptance   5.15
  → Speedup vs Baseline: 8.32x
  → Improvement over Eagle3: 17.1%
```

### 3. 多个数据集对比
```bash
# 对标论文的多数据集评估（需要约2小时）
bash evaluate/run_benchmark.sh \
    --size_model ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --datasets "humaneval alpaca mt_bench gsm8k qa" \
    --num_samples 100 \
    --output_dir ./evaluate/results_multi_dataset
```

### 4. 只评估Eagle + RL（不要baseline）
```python
# 直接调用Python（跳过baseline计算）
python -c """
import torch
from pathlib import Path
from stable_baselines3 import PPO
from eagle.model.ea_model import EaModel

model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    torch_dtype=torch.float16,
    device_map='auto'
).to('cuda')

# 加载RL模型
size_policy = PPO.load('./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip')
depth_policy = PPO.load('./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip')

print('✓ Models loaded successfully')
"""
```

---

## 高级用法

### 5. 不同温度下的性能评估
```bash
# 在温度=0.0（贪心）和温度=1.0（采样）下对比
for temp in 0.0 1.0; do
  python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --dataset_names humaneval \
    --num_samples 50 \
    --temperature $temp \
    --output_dir ./evaluate/results_temp_${temp}
done
```

### 6. 增量评估（累积结果）
```bash
# 逐个数据集添加到同一结果文件
for dataset in humaneval alpaca gsm8k; do
  python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --dataset_names $dataset \
    --num_samples 100 \
    --output_dir ./evaluate/results_incremental
done
```

---

## 结果分析

### 查看详细统计
```bash
# 自动分析结果并生成对比表
python evaluate/analyze_results.py \
    --results_file ./evaluate/results/benchmark_results.json
```

输出示例：
```
====================================================================================
Dataset: humaneval
====================================================================================
Method              Tokens       Time (s)    Throughput  Acceptance  
------------------------------------------------------------------------------------
baseline            125.30       3.2340      38.72       —           
eagle3              125.30       0.4560      275.00      5.42        
  → Speedup vs Baseline: 7.10x
eagle3_rl           125.30       0.3890      322.11      5.15        
  → Speedup vs Baseline: 8.32x

...
====================================================================================
```

---

## 自定义脚本示例

### 加载模型并手动测试
```python
import torch
from evaluate.benchmark_rl_inference import eagle3_decoding, eagle3_rl_decoding
from eagle.model.ea_model import EaModel
from stable_baselines3 import PPO

# 1. 加载基础模型
model = EaModel.from_pretrained(
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to('cuda')

# 2. 加载RL策略
size_policy = PPO.load('./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip')
depth_policy = PPO.load('./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip')

# 3. 准备输入
tokenizer = model.get_tokenizer()
prompt = "What is 2+2?"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

# 4. 运行推理
result_eagle3 = eagle3_decoding(model, input_ids)
result_rl = eagle3_rl_decoding(model, input_ids, size_policy, depth_policy)

# 5. 对比
print(f"Eagle3 throughput: {result_eagle3['throughput']:.2f} tok/s")
print(f"RL throughput: {result_rl['throughput']:.2f} tok/s")
print(f"Speedup: {result_rl['throughput'] / result_eagle3['throughput']:.2f}x")
```

---

## 常见问题

### Q: Benchmark太慢了，怎么加速？
**A:** 减少样本数：
```bash
python evaluate/benchmark_rl_inference.py --num_samples 5  # 默认20，改为5
```

### Q: 显存不足怎么办？
**A:** 使用较小的模型或启用int8量化：
```bash
# 在 benchmark_rl_inference.py 中修改：
# torch_dtype=torch.float16 → torch.int8
# load_in_8bit=True
```

### Q: 想跑完整的论文复现
**A:** 按照论文使用相同的数据集和参数：
```bash
python evaluate/benchmark_rl_inference.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --ea_model_path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --dataset_names humaneval alpaca mt_bench gsm8k qa \
    --num_samples 200 \
    --temperature 0.0
```

---

## 输出目录结构

```
evaluate/
├── results/
│   ├── benchmark_results.json          # 详细结果（JSON格式）
│   └── benchmark_results_temp_0.0/     # 温度=0.0的结果
├── results_multi_dataset/              # 多数据集结果
└── results_temp_*/                     # 不同温度的结果
```

---

## 参考指标解释

| 指标 | 说明 | 单位 |
|------|------|------|
| **Throughput** | 每秒生成的token数 | tok/s |
| **Acceptance Length** | 平均每个验证步接受的token数 | tokens |
| **Cycles** | draft-verify循环数 | 次 |
| **Speedup** | 相对于baseline的加速比 | 倍数 |

## 论文对标

本脚本支持复现论文中的评估：
- 数据集：HumanEval, Alpaca, MT-Bench, GSM8K, QA
- 模型：Llama-3.1-8B, Vicuna-13B, DeepSeek-8B, Qwen3系列
- 温度：0.0（贪心）和1.0（采样）

详见论文 Table 1 和 Appendix A.3
