# RL-Enhanced Speculative Decoding Evaluation

这个目录包含用于评估RL增强推理性能的工具和脚本。

## 文件说明

- **benchmark_rl_inference.py**: 主要的benchmark脚本，支持对比多种模式
  - Baseline：纯自回归解码
  - Eagle3：静态参数配置
  - Eagle3+RL：加载RL策略进行动态调整

## 快速开始

### 1. 准备模型权重

确保已有以下检查点：
- 基础模型：`meta-llama/Llama-3.1-8B-Instruct`
- EA模型：`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`
- Size RL模型：`./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip`（from `train_size.sh`）
- Depth RL模型：`./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip`（from `train_depth.sh`）

### 2. 运行benchmark

最简单的用法（使用默认参数）：
```bash
python evaluate/benchmark_rl_inference.py
```

完整参数示例：
```bash
python evaluate/benchmark_rl_inference.py \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --ea_model_path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --data_dir ./eagle/data \
    --dataset_names humaneval alpaca mt_bench gsm8k qa \
    --num_samples 100 \
    --temperature 0.0 \
    --device cuda \
    --output_dir ./evaluate/results
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_model_path` | 基础LLM路径 | `meta-llama/Llama-3.1-8B-Instruct` |
| `--ea_model_path` | EA/Draft模型路径 | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |
| `--size_model_path` | Size策略RL模型 (.zip) | `""` (可选) |
| `--depth_model_path` | Depth策略RL模型 (.zip) | `""` (可选) |
| `--data_dir` | 数据目录 | `./eagle/data` |
| `--dataset_names` | 评估数据集 | `["humaneval"]` |
| `--num_samples` | 每个数据集的样本数 | `20` |
| `--temperature` | 采样温度（0=贪心） | `0.0` |
| `--batch_size` | 批大小 | `1` |
| `--output_dir` | 结果保存目录 | `./evaluate/results` |
| `--device` | 计算设备 | `cuda` 或 `cpu` |

## 输出说明

### 控制台输出

每个数据集的结果格式：
```
baseline            : Tokens  128.5, Time   2.345s, Throughput  54.78 tok/s
eagle3              : Tokens  128.5, Time   0.234s, Throughput 549.15 tok/s, Acceptance  6.42
eagle3_rl           : Tokens  128.5, Time   0.189s, Throughput 680.42 tok/s, Acceptance  5.89
```

### 结果文件

`./evaluate/results/benchmark_results.json` 包含完整的详细结果：
```json
{
  "humaneval": {
    "baseline": [
      {"tokens": 128, "time": 2.345, "throughput": 54.6},
      ...
    ],
    "eagle3": [
      {"tokens_generated": 128, "elapsed_time": 0.234, "throughput": 547.0, 
       "num_cycles": 20, "avg_acceptance_len": 6.42, "cycles_per_sec": 85.5},
      ...
    ],
    "eagle3_rl": [...]
  },
  ...
}
```

## 代码架构

### 关键函数

1. **baseline_decoding()**: 纯自回归，逐token生成
2. **eagle3_decoding()**: Eagle3静态配置，使用tree-based draft+verify
3. **eagle3_rl_decoding()**: Eagle3+RL，加载策略进行动态调整

### 关键指标

| 指标 | 说明 |
|------|------|
| **tokens_generated** | 总生成token数 |
| **elapsed_time** | 总耗时（秒） |
| **throughput** | 吞吐量（token/秒） |
| **num_cycles** | draft-verify周期数 |
| **avg_acceptance_len** | 平均接受长度 |
| **cycles_per_sec** | 每秒周期数 |

## 常见场景

### 场景1: 只评估Eagle3（无RL）
```bash
python evaluate/benchmark_rl_inference.py \
    --dataset_names humaneval \
    --num_samples 50
```

### 场景2: 评估RL改进
```bash
python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --dataset_names alpaca gsm8k \
    --num_samples 100
```

### 场景3: 多数据集对比
```bash
python evaluate/benchmark_rl_inference.py \
    --size_model_path ./checkpoints/ppo_speculative_decoder_controller_rebuttal.zip \
    --depth_model_path ./checkpoints/ppo_speculative_decoder_controller_v1_single_action.zip \
    --dataset_names humaneval alpaca mt_bench gsm8k qa sum \
    --num_samples 200 \
    --output_dir ./evaluate/results_full
```

## 故障排除

### 模型加载失败
- 确认模型路径正确且文件存在
- 检查GPU内存是否充足（需要16GB用于float16）

### RL模型加载失败
- 确认 `.zip` 文件是有效的PPO checkpoint
- 如果RL路径不存在，脚本会自动跳过该模式

### 数据加载失败
- 检查 `--data_dir` 目录结构：
  ```
  data_dir/
    humaneval/
      question.jsonl
    alpaca/
      question.jsonl
    ...
  ```

## 参考

- RL训练脚本：[rl/rl_total.py](../rl/rl_total.py), [rl/rl_depth.py](../rl/rl_depth.py)
- 论文：[Learning To Draft: Adaptive Speculative Decoding with Reinforcement Learning](https://arxiv.org/abs/2603.01639)
