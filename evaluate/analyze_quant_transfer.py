"""
Summarize quantization-transfer benchmark results.

Usage:
  python evaluate/analyze_quant_transfer.py \
    --results_root evaluate/results/quant_transfer/Qwen3-14B \
    --output_csv evaluate/results/quant_transfer/Qwen3-14B/summary.csv \
    --plot
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

QUANT_ORDER = {
    "bf16": 0,
    "fp16": 1,
    "int8": 2,
    "int4": 3,
}

POLICY_COLORS = {
    "bf16": "#4C72B0",
    "fp16": "#8172B2",
    "int8": "#55A868",
    "int4": "#C44E52",
}


def _method_stats(samples, method):
    if not samples:
        return {}
    if method == "baseline":
        return {
            "throughput": mean(s.get("throughput", 0.0) for s in samples),
            "time": mean(s.get("time", 0.0) for s in samples),
            "acceptance": "",
            "size_tokens": "",
            "depth_stop": "",
        }
    return {
        "throughput": mean(s.get("throughput", 0.0) for s in samples),
        "time": mean(s.get("elapsed_time", 0.0) for s in samples),
        "acceptance": mean(s.get("avg_acceptance_len", 0.0) for s in samples),
        "size_tokens": (
            mean(s.get("avg_size_tokens", 0.0) for s in samples)
            if any("avg_size_tokens" in s for s in samples)
            else ""
        ),
        "depth_stop": (
            mean(s.get("avg_depth_stop", 0.0) for s in samples)
            if any("avg_depth_stop" in s for s in samples)
            else ""
        ),
    }


def _policy_variant(config):
    if config.get("policy_label"):
        return config["policy_label"]
    size_model_path = config.get("size_model_path", "")
    if size_model_path:
        parts = Path(size_model_path).parts
        if len(parts) >= 4 and parts[-2] == "size":
            return parts[-3]
    return "unknown"


def summarize_file(path):
    with path.open() as f:
        results = json.load(f)

    config = results.get("config", {})
    eval_quantization = config.get("quantization")
    if not eval_quantization:
        eval_quantization = "int8" if config.get("load_in_8bit") else "fp16"

    rows = []
    for dataset, dataset_results in results.items():
        if dataset == "config":
            continue

        stats = {
            method: _method_stats(samples, method)
            for method, samples in dataset_results.items()
            if samples
        }
        baseline_tp = stats.get("baseline", {}).get("throughput", 0.0)
        eagle_tp = stats.get("eagle3", {}).get("throughput", 0.0)
        rl_tp = stats.get("eagle3_rl", {}).get("throughput", 0.0)
        rl = stats.get("eagle3_rl", {})
        rows.append(
            {
                "result_file": str(path),
                "experiment_label": config.get("experiment_label", ""),
                "base_model": config.get("base_model_path", ""),
                "source_policy": _policy_variant(config),
                "policy_train_gpu": config.get("policy_train_gpu", ""),
                "benchmark_gpu_tag": config.get("benchmark_gpu_tag", ""),
                "actual_gpu_name": config.get("gpu_name", ""),
                "eval_quantization": eval_quantization,
                "dataset": dataset,
                "baseline_tp": baseline_tp,
                "eagle3_tp": eagle_tp,
                "eagle3_rl_tp": rl_tp,
                "rl_vs_eagle3": rl_tp / eagle_tp if eagle_tp else 0.0,
                "rl_vs_baseline": rl_tp / baseline_tp if baseline_tp else 0.0,
                "rl_acceptance": rl.get("acceptance", ""),
                "rl_size_tokens": rl.get("size_tokens", ""),
                "rl_depth_stop": rl.get("depth_stop", ""),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze quantization-transfer benchmark results")
    parser.add_argument("--results_root", type=str, default="./evaluate/results/quant_transfer")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--plot", action="store_true", help="Generate quant-transfer summary charts")
    parser.add_argument("--plot_path", type=str, default="", help="Path for the summary chart PNG")
    parser.add_argument(
        "--dataset_plot_path",
        type=str,
        default="",
        help="Path for the per-dataset transfer chart PNG",
    )
    args = parser.parse_args()

    paths = sorted(Path(args.results_root).glob("**/benchmark_results_*.json"))
    if not paths:
        raise SystemExit(f"No benchmark_results_*.json files found under {args.results_root}")

    rows = []
    for path in paths:
        rows.extend(summarize_file(path))

    headers = [
        "experiment_label",
        "source_policy",
        "policy_train_gpu",
        "benchmark_gpu_tag",
        "actual_gpu_name",
        "eval_quantization",
        "dataset",
        "baseline_tp",
        "eagle3_tp",
        "eagle3_rl_tp",
        "rl_vs_eagle3",
        "rl_vs_baseline",
        "rl_acceptance",
        "rl_size_tokens",
        "rl_depth_stop",
        "result_file",
    ]

    print("source_policy train_gpu bench_tag actual_gpu                 eval_quant dataset      eagle3_rl  rl/eagle3  rl/base  accept  size  depth")
    print("-" * 140)
    for row in rows:
        print(
            f"{row['source_policy']:<13} "
            f"{row['policy_train_gpu'] or '-':<9} "
            f"{row['benchmark_gpu_tag'] or '-':<9} "
            f"{(row['actual_gpu_name'] or '-')[:26]:<26} "
            f"{row['eval_quantization']:<10} "
            f"{row['dataset']:<12} "
            f"{row['eagle3_rl_tp']:>9.2f} "
            f"{row['rl_vs_eagle3']:>9.3f} "
            f"{row['rl_vs_baseline']:>8.3f} "
            f"{_fmt(row['rl_acceptance']):>7} "
            f"{_fmt(row['rl_size_tokens']):>5} "
            f"{_fmt(row['rl_depth_stop']):>6}"
        )

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})
        print(f"\nSaved CSV summary to {args.output_csv}")

    if args.plot:
        results_root = Path(args.results_root)
        plot_path = Path(args.plot_path) if args.plot_path else results_root / "quant_transfer_summary_charts.png"
        dataset_plot_path = (
            Path(args.dataset_plot_path)
            if args.dataset_plot_path
            else results_root / "quant_transfer_by_dataset_charts.png"
        )
        plot_quant_transfer(rows, plot_path, dataset_plot_path)

    _print_pairwise_transfer(rows)


def _fmt(value):
    if value == "":
        return "-"
    return f"{value:.2f}"


def _print_pairwise_transfer(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (
            row.get("benchmark_gpu_tag", ""),
            row.get("actual_gpu_name", ""),
            row.get("eval_quantization", ""),
            row.get("dataset", ""),
        )
        groups[key][row.get("source_policy", "")] = row

    pairs = []
    for key, policy_rows in groups.items():
        if "h100_bf16" not in policy_rows or "a5090_bf16" not in policy_rows:
            continue
        h100 = policy_rows["h100_bf16"]
        a5090 = policy_rows["a5090_bf16"]
        a5090_tp = a5090.get("eagle3_rl_tp", 0.0)
        h100_tp = h100.get("eagle3_rl_tp", 0.0)
        pairs.append((key, h100, a5090, h100_tp / a5090_tp if a5090_tp else 0.0))

    if not pairs:
        return

    print("\nPairwise transfer: h100_bf16 policy / a5090_bf16 policy")
    print("bench_tag actual_gpu                 eval_quant dataset      h100_rl   a5090_rl  ratio")
    print("-" * 96)
    for key, h100, a5090, ratio in sorted(pairs):
        benchmark_gpu_tag, actual_gpu_name, eval_quantization, dataset = key
        print(
            f"{benchmark_gpu_tag or '-':<9} "
            f"{(actual_gpu_name or '-')[:26]:<26} "
            f"{eval_quantization:<10} "
            f"{dataset:<12} "
            f"{h100['eagle3_rl_tp']:>8.2f} "
            f"{a5090['eagle3_rl_tp']:>9.2f} "
            f"{ratio:>6.3f}"
        )


def _sort_quant(values):
    return sorted(values, key=lambda value: (QUANT_ORDER.get(value, 99), value))


def _display_quant(value):
    return value.upper() if value in {"bf16", "fp16", "int8", "int4"} else value


def _display_policy(value):
    if value in {"bf16", "fp16", "int8", "int4"}:
        return f"{_display_quant(value)} policy"
    return value


def _to_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def _mean_metric(rows, row_key, row_value, col_key, col_value, metric):
    vals = [
        _to_float(row.get(metric))
        for row in rows
        if row.get(row_key) == row_value and row.get(col_key) == col_value
    ]
    vals = [value for value in vals if value is not None]
    return mean(vals) if vals else None


def _mean_for_filter(rows, metric, **filters):
    vals = []
    for row in rows:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        value = _to_float(row.get(metric))
        if value is not None:
            vals.append(value)
    return mean(vals) if vals else None


def _matrix(rows, policies, eval_quantizations, metric):
    return [
        [
            _mean_metric(rows, "source_policy", policy, "eval_quantization", eval_quantization, metric)
            for eval_quantization in eval_quantizations
        ]
        for policy in policies
    ]


def _annotated_heatmap(ax, values, policies, eval_quantizations, title, fmt, cmap):
    import math
    import numpy as np

    arr = np.array(
        [[math.nan if value is None else value for value in row] for row in values],
        dtype=float,
    )
    masked = np.ma.masked_invalid(arr)
    image = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(eval_quantizations)))
    ax.set_xticklabels([_display_quant(value) for value in eval_quantizations])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([_display_policy(value) for value in policies])
    ax.set_xlabel("Evaluation Quantization")
    ax.set_ylabel("Trained LTD Policy")

    for y, row in enumerate(values):
        for x, value in enumerate(row):
            if value is None:
                label = "-"
            elif fmt.endswith("x"):
                label = f"{value:.2f}x"
            elif fmt == ".3f":
                label = f"{value:.3f}"
            elif fmt == ".2f":
                label = f"{value:.2f}"
            else:
                label = f"{value:.1f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold")

    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_quant_transfer(rows, summary_path, dataset_plot_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        print("No data to plot.")
        return

    policies = _sort_quant({row["source_policy"] for row in rows})
    eval_quantizations = _sort_quant({row["eval_quantization"] for row in rows})
    datasets = sorted({row["dataset"] for row in rows})

    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Qwen3-14B Quantization Transfer Summary", fontsize=16, fontweight="bold", y=0.98)

    heatmaps = [
        ("Eagle3+RL Throughput (tok/s)", "eagle3_rl_tp", ".1f", "Blues"),
        ("Speedup vs Baseline", "rl_vs_baseline", "x", "Greens"),
        ("Eagle3+RL / Eagle3", "rl_vs_eagle3", ".3f", "Oranges"),
        ("Average Acceptance Length", "rl_acceptance", ".2f", "Purples"),
    ]
    for ax, (title, metric, fmt, cmap) in zip(axes.flat, heatmaps):
        values = _matrix(rows, policies, eval_quantizations, metric)
        _annotated_heatmap(ax, values, policies, eval_quantizations, title, fmt, cmap)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary chart to {summary_path}")

    if dataset_plot_path:
        dataset_plot_path = Path(dataset_plot_path)
        dataset_plot_path.parent.mkdir(parents=True, exist_ok=True)
        _plot_dataset_transfer(rows, policies, eval_quantizations, datasets, dataset_plot_path)


def _plot_dataset_transfer(rows, policies, eval_quantizations, datasets, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        len(eval_quantizations),
        len(datasets),
        figsize=(4.4 * len(datasets), 3.8 * len(eval_quantizations)),
        sharey="row",
    )
    if len(eval_quantizations) == 1:
        axes = np.array([axes])
    if len(datasets) == 1:
        axes = axes.reshape(len(eval_quantizations), 1)

    fig.suptitle(
        "Qwen3-14B Quantization Transfer by Dataset",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    x = np.arange(len(policies))
    colors = [POLICY_COLORS.get(policy, "#777777") for policy in policies]

    for row_idx, eval_quantization in enumerate(eval_quantizations):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            vals = [
                _mean_for_filter(
                    rows,
                    "eagle3_rl_tp",
                    source_policy=policy,
                    eval_quantization=eval_quantization,
                    dataset=dataset,
                )
                or 0.0
                for policy in policies
            ]
            bars = ax.bar(x, vals, color=colors, width=0.62, label="Eagle3+RL")
            baseline = _mean_for_filter(rows, "baseline_tp", eval_quantization=eval_quantization, dataset=dataset)
            eagle3 = _mean_for_filter(rows, "eagle3_tp", eval_quantization=eval_quantization, dataset=dataset)
            if baseline:
                ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.2, label="Baseline")
            if eagle3:
                ax.axhline(eagle3, color="#4C72B0", linestyle=":", linewidth=1.8, label="Eagle3")

            for bar, value in zip(bars, vals):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.015,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            if row_idx == 0:
                ax.set_title(dataset, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{_display_quant(eval_quantization)} eval\ntok/s")
            ax.set_xticks(x)
            ax.set_xticklabels([_display_quant(policy) for policy in policies], rotation=20, ha="right")
            ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved by-dataset chart to {save_path}")


if __name__ == "__main__":
    main()
