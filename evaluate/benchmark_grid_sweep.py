"""
Grid sweep benchmark: run Eagle3 static decoding over a grid of
(draft_depth, verification_size) configurations to map the throughput
landscape that LTD's RL policies adaptively navigate.

Usage:
  python evaluate/benchmark_grid_sweep.py \
      --base_model_path meta-llama/Llama-3.1-8B-Instruct \
      --ea_model_path yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
      --data_dir ./eagle/data \
      --dataset_names humaneval alpaca gsm8k mt_bench \
      --num_samples 50 \
      --depths 3 5 8 12 \
      --sizes 20 40 60 100 150 200

  # SLURM array mode: run only one dataset per task
  python evaluate/benchmark_grid_sweep.py ... --dataset_index 0
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluate.benchmark_rl_inference import (
    to_jsonable,
    load_and_sample_data,
    baseline_decoding,
    eagle3_decoding,
    _input_device,
)
from eagle.model.ea_model import EaModel
from eagle.model.quantization import (
    add_quantization_args,
    apply_quantization_config,
    quantization_metadata,
    should_move_model_to_device,
)


def run_grid_sweep(model, samples, depths, sizes, temperature):
    """Run eagle3_decoding for every (depth, size) pair on all samples."""
    grid_results = {}
    total_configs = len(depths) * len(sizes)
    input_dev = _input_device(model)

    for cfg_idx, (depth, size) in enumerate(product(depths, sizes), 1):
        key = f"d{depth}_s{size}"
        print(f"  [{cfg_idx}/{total_configs}] depth={depth}, size={size}")
        config_samples = []
        for sample in samples:
            input_ids = sample["input_ids"].to(input_dev)
            res = eagle3_decoding(
                model, input_ids,
                draft_depth=depth,
                verification_size=size,
                temperature=temperature,
            )
            config_samples.append(res)
        grid_results[key] = config_samples
        avg_tp = np.mean([s["throughput"] for s in config_samples])
        avg_acc = np.mean([s["avg_acceptance_len"] for s in config_samples])
        print(f"         throughput={avg_tp:.1f} tok/s, acc_len={avg_acc:.2f}")

    return grid_results


def main():
    parser = argparse.ArgumentParser(description="Grid sweep over (depth, size) for Eagle3 static decoding")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--ea_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./eagle/data")
    parser.add_argument("--dataset_names", nargs="+",
                        default=["humaneval", "alpaca", "gsm8k", "mt_bench"])
    parser.add_argument("--dataset_index", type=int, default=-1,
                        help="If >= 0, run only this dataset index (for SLURM array jobs)")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--depths", nargs="+", type=int, default=[3, 5, 8, 12])
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 40, 60, 100, 150, 200])
    parser.add_argument("--output_dir", type=str, default="./evaluate/results/grid_sweep")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default="")
    add_quantization_args(parser)
    args = parser.parse_args()

    if args.dataset_index >= 0:
        if args.dataset_index >= len(args.dataset_names):
            print(f"dataset_index={args.dataset_index} out of range for {len(args.dataset_names)} datasets")
            sys.exit(1)
        args.dataset_names = [args.dataset_names[args.dataset_index]]

    os.makedirs(args.output_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else "N/A"

    print("=" * 80)
    print("Grid Sweep Benchmark: Eagle3 static (depth x size)")
    print("=" * 80)
    print(f"  Base model:  {args.base_model_path}")
    print(f"  EA model:    {args.ea_model_path}")
    print(f"  Depths:      {args.depths}")
    print(f"  Sizes:       {args.sizes}")
    print(f"  Grid points: {len(args.depths) * len(args.sizes)}")
    print(f"  Datasets:    {args.dataset_names}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  GPU:         {gpu_name}")
    print("=" * 80)

    model_kwargs = dict(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        low_cpu_mem_usage=True,
        depth=max(args.depths),
        top_k=10,
        total_token=max(args.sizes),
        use_eagle3=True,
        use_dyn_len=False,
    )
    quantization = apply_quantization_config(model_kwargs, args)
    quant_meta = quantization_metadata(args, quantization)
    print(f"  Quantization: {quantization}")
    print(f"  Effective device_map: {quant_meta['device_map'] or '(none)'}")

    print("\nLoading model...")
    model = EaModel.from_pretrained(**model_kwargs)
    if should_move_model_to_device(quantization, args.device_map):
        model = model.to(args.device)
    model.eval()
    input_dev = _input_device(model)
    print(f"  Input device: {input_dev}")

    tokenizer = model.get_tokenizer()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
            "{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>\n' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        )

    all_results = {
        "config": {
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "depths": args.depths,
            "sizes": args.sizes,
            "dataset_names": args.dataset_names,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "gpu_name": gpu_name,
            **quant_meta,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    for dataset_name in args.dataset_names:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        samples = load_and_sample_data(args.data_dir, dataset_name, args.num_samples, tokenizer)
        if not samples:
            print(f"  Skipping {dataset_name} (no samples)")
            continue

        dataset_results = {}

        print(f"\n  Running baseline autoregressive decoding...")
        baseline_samples = []
        for sample in samples:
            input_ids = sample["input_ids"].to(input_dev)
            tokens, elapsed = baseline_decoding(model, input_ids, temperature=args.temperature)
            baseline_samples.append({
                "tokens": tokens,
                "time": elapsed,
                "throughput": tokens / elapsed if elapsed > 0 else 0,
            })
        dataset_results["baseline"] = baseline_samples
        avg_baseline_tp = np.mean([s["throughput"] for s in baseline_samples])
        print(f"         baseline throughput={avg_baseline_tp:.1f} tok/s")

        print(f"\n  Running grid sweep ({len(args.depths)}x{len(args.sizes)} = {len(args.depths)*len(args.sizes)} configs)...")
        grid = run_grid_sweep(model, samples, args.depths, args.sizes, args.temperature)
        dataset_results.update(grid)

        all_results[dataset_name] = dataset_results

    model_short = args.base_model_path.split("/")[-1]
    ds_tag = args.dataset_names[0] if len(args.dataset_names) == 1 else "multi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"grid_sweep_{model_short}_{ds_tag}_{timestamp}.json",
    )
    with open(output_file, "w") as f:
        json.dump(to_jsonable(all_results), f, indent=2)

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 80)
    print("Summary: avg throughput (tok/s) per (depth, size)")
    print("=" * 80)
    for dataset_name in args.dataset_names:
        if dataset_name not in all_results or dataset_name == "config":
            continue
        dr = all_results[dataset_name]
        baseline_tp = np.mean([s["throughput"] for s in dr["baseline"]])
        print(f"\n  {dataset_name}  (baseline={baseline_tp:.1f} tok/s)")
        header = "depth\\size | " + " | ".join(f"{s:>6}" for s in args.sizes)
        print(f"  {header}")
        print(f"  {'-' * len(header)}")
        for depth in args.depths:
            row_vals = []
            for size in args.sizes:
                key = f"d{depth}_s{size}"
                if key in dr:
                    tp = np.mean([s["throughput"] for s in dr[key]])
                    speedup = tp / baseline_tp if baseline_tp > 0 else 0
                    row_vals.append(f"{speedup:.2f}x")
                else:
                    row_vals.append("  N/A ")
            print(f"  {depth:>9} | " + " | ".join(f"{v:>6}" for v in row_vals))


if __name__ == "__main__":
    main()
