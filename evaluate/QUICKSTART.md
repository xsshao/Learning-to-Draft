# Evaluate 目录使用指南

## 📁 目录结构

```
evaluate/
├── __init__.py                          # 模块初始化
├── benchmark_rl_inference.py            # 核心benchmark脚本
├── analyze_results.py                   # 结果分析工具
├── run_benchmark.sh                     # 快速启动脚本
├── README.md                            # 详细文档
├── EXAMPLES.md                          # 使用示例
├── config_examples.json                 # 配置模板
├── QUICKSTART.md                        # 本文件
└── results/                             # 结果输出目录（自动创建）
    └── benchmark_results.json           # 详细结果文件
```

---

## 🚀 快速开始（5分钟）

### 1️⃣ 最简单的用法（仅Eagle3，无RL）

```bash
cd /workspace/Learning-to-Draft
python evaluate/benchmark_rl_inference.py --num_samples 10
```

**输出示例：**
```
Dataset: humaneval
  Results for humaneval:
    baseline            : Tokens  125.3, Time   3.234s, Throughput  38.72 tok/s
    eagle3              : Tokens  125.3, Time   0.456s, Throughput 275.00 tok/s, Acceptance  5.42
```

### 2️⃣ 加入RL模型（对标论文）

```bash
python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --num_samples 50
```

**输出示例：**
```
Dataset: humaneval
    baseline            : Tokens  125.3, Time   3.234s, Throughput  38.72 tok/s
    eagle3              : Tokens  125.3, Time   0.456s, Throughput 275.00 tok/s, Acceptance  5.42
    eagle3_rl           : Tokens  125.3, Time   0.389s, Throughput 322.11 tok/s, Acceptance  5.15
```

### 3️⃣ 查看详细结果

```bash
python evaluate/analyze_results.py --results_file ./evaluate/results/benchmark_results.json
```

---

## 📋 核心文件说明

### `benchmark_rl_inference.py` (主体脚本)

**核心功能：**
- 加载基础模型、EA模型、RL策略权重
- 实现三种推理模式
- 在多个数据集上运行benchmark
- 输出详细指标

**关键函数：**

| 函数 | 说明 |
|------|------|
| `baseline_decoding()` | 纯自回归，逐token生成 |
| `eagle3_decoding()` | 静态Eagle3 (tree-based) |
| `eagle3_rl_decoding()` | 动态Eagle3+RL策略 |

**参数说明：**

```
--base_model_path          基础LLM路径
--ea_model_path            EA/Draft模型路径
--size_model_path          Size策略权重 (.zip)
--depth_model_path         Depth策略权重 (.zip)
--dataset_names            评估数据集（多个）
--num_samples              每个数据集样本数
--temperature              采样温度（0=贪心）
--output_dir               结果保存目录
--device                   cuda 或 cpu
```

---

### `run_benchmark.sh` (快速启动脚本)

简化的bash脚本，自动构建Python命令。

**用法：**
```bash
bash evaluate/run_benchmark.sh \
    --size_model ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --datasets "humaneval alpaca" \
    --num_samples 100
```

---

### `analyze_results.py` (结果分析)

解析benchmark输出的JSON，生成汇总表格。

**用法：**
```bash
python evaluate/analyze_results.py --results_file ./evaluate/results/benchmark_results.json
```

**输出：**
```
Dataset: humaneval
Method              Tokens       Time (s)    Throughput   Acceptance  
─────────────────────────────────────────────────────────────────────
baseline            125.30       3.2340      38.72        —           
eagle3              125.30       0.4560      275.00       5.42        
  → Speedup vs Baseline: 7.10x
eagle3_rl           125.30       0.3890      322.11       5.15        
  → Speedup vs Baseline: 8.32x
```

---

## 📊 推理模式对比

| 特性 | Baseline | Eagle3 | Eagle3+RL |
|------|----------|--------|-----------|
| draft策略 | 无 | 静态树 | 动态+树 |
| verify规模 | N/A | 固定60 | 动态（策略决定） |
| draft深度 | N/A | 固定8 | 动态（策略决定） |
| 吞吐 | ~40 tok/s | ~275 tok/s | ~320 tok/s |
| 接受长度 | N/A | 5~7 | 4~6 |

---

## 🔧 常见用法场景

### 场景A: 快速测试（3分钟）
```bash
python evaluate/benchmark_rl_inference.py --num_samples 5 --dataset_names humaneval
```

### 场景B: 单数据集详细评估（20分钟）
```bash
python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --num_samples 100 \
    --dataset_names alpaca
```

### 场景C: 多数据集对标论文（2小时）
```bash
python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --dataset_names humaneval alpaca mt_bench gsm8k qa \
    --num_samples 200
```

### 场景D: 温度稳定性测试
```bash
for temp in 0.0 0.5 1.0; do
  python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --temperature $temp \
    --dataset_names humaneval \
    --num_samples 50 \
    --output_dir ./evaluate/results_temp_${temp}
done
```

---

## 📈 输出指标说明

### Baseline 的指标
- **tokens**: 生成的token数
- **time**: 总耗时（秒）
- **throughput**: token/秒

### Eagle3 / Eagle3+RL 的指标
- **tokens_generated**: 生成的token数
- **elapsed_time**: 总耗时（秒）
- **throughput**: token/秒
- **avg_acceptance_len**: 平均接受长度
- **cycles_per_sec**: 每秒cycle数

---

## 🔄 集成到RL训练流程

### 完整工作流：

1. **训练 Size 策略** (1小时)
   ```bash
   bash train_size.sh <gpu_id>
   ```

2. **训练 Depth 策略** (2小时)
   ```bash
   bash train_depth.sh <gpu_id>
   ```

3. **评估性能** (1小时)
   ```bash
   python evaluate/benchmark_rl_inference.py \
       --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
       --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
       --dataset_names humaneval alpaca mt_bench \
       --num_samples 100
   ```

4. **分析结果**
   ```bash
   python evaluate/analyze_results.py
   ```

---

## 🐛 故障排除

### ❌ "模型加载失败"
→ 检查路径和GPU内存
```bash
nvidia-smi  # 查看GPU状态
```

### ❌ "RL模型加载失败"
→ 确认checkpoint是有效的PPO文件且格式正确
```bash
unzip -t ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip
```

### ❌ "数据集找不到"
→ 检查数据目录结构
```bash
ls -la ./eagle/data/humaneval/question.jsonl
```

### ❌ "结果文件很大"
→ 减少 `--num_samples` 以生成更小的结果文件

---

## 📞 反馈和改进

如果遇到问题：

1. 检查 README.md 中的详细说明
2. 参考 EXAMPLES.md 中的使用示例
3. 查看 config_examples.json 中的配置模板

---

## 参考文献

- Paper: [Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning](https://arxiv.org/abs/2603.01639)
- RL Training: [rl/rl_total.py](../rl/rl_total.py), [rl/rl_depth.py](../rl/rl_depth.py)
- Original Eagle: [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)

---

**版本**: 0.1.0  
**最后更新**: 2026-03-19
