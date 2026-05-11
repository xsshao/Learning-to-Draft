"""
Simple analysis and visualization of benchmark results.
Usage: python evaluate/analyze_results.py --results_file ./evaluate/results/benchmark_results.json
       python evaluate/analyze_results.py --results_file ... --plot --plot_dir ./evaluate/results
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _collect_stats(results: dict) -> dict:
    """Return {dataset_name: {method_name: stat_dict}} (skips 'config' key)."""
    all_stats = {}
    for dataset_name, dataset_results in results.items():
        if dataset_name == "config":
            continue
        stats = {}
        for method_name, samples in dataset_results.items():
            if not samples:
                continue
            df = pd.DataFrame(samples)
            if method_name == "baseline":
                stats[method_name] = {
                    "avg_tokens": df["tokens"].mean(),
                    "avg_time": df["time"].mean(),
                    "avg_throughput": df["throughput"].mean(),
                    "acceptance_len": None,
                    "avg_size_tokens": None,
                    "avg_depth_stop": None,
                }
            else:
                stats[method_name] = {
                    "avg_tokens": df["tokens_generated"].mean(),
                    "avg_time": df["elapsed_time"].mean(),
                    "avg_throughput": df["throughput"].mean(),
                    "acceptance_len": df["avg_acceptance_len"].mean() if "avg_acceptance_len" in df.columns else None,
                    "avg_size_tokens": df["avg_size_tokens"].mean() if "avg_size_tokens" in df.columns else None,
                    "avg_depth_stop": df["avg_depth_stop"].mean() if "avg_depth_stop" in df.columns else None,
                }
        all_stats[dataset_name] = stats
    return all_stats


# ── public API ────────────────────────────────────────────────────────────────

def print_results(results: dict):
    """Print analysis of benchmark results from an in-memory dict."""

    print("=" * 110)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 110)

    config = results.get("config")
    if config:
        print("\nRun Configuration:")
        print(f"  Base model:    {config.get('base_model_path', '—')}")
        print(f"  EA model:      {config.get('ea_model_path', '—')}")
        if config.get("size_model_path"):
            print(f"  Size policy:   {config['size_model_path']}")
        if config.get("depth_model_path"):
            print(f"  Depth policy:  {config['depth_model_path']}")
        print(f"  Datasets:      {', '.join(config.get('dataset_names', []))}")
        print(f"  Num samples:   {config.get('num_samples', '—')}")
        print(f"  Batch size:    {config.get('batch_size', '—')}")
        print(f"  Temperature:   {config.get('temperature', '—')}")
        print(f"  Device:        {config.get('device', '—')}")
        if config.get("quantization"):
            print(f"  Quantization:  {config['quantization']}")
        elif "load_in_8bit" in config:
            print(f"  Quantization:  {'int8' if config.get('load_in_8bit') else 'fp16'}")
        if config.get("timestamp"):
            print(f"  Timestamp:     {config['timestamp']}")

    all_stats = _collect_stats(results)

    for dataset_name, stats in all_stats.items():
        print(f"\n{'Dataset: ' + dataset_name:^110}")
        print("-" * 110)
        print(f"{'Method':<20} {'Tokens':<10} {'Time (s)':<12} {'Throughput':<14} {'Acceptance':<12} {'SizeTok':<10} {'DepthStop':<12} {'Speedup'}")
        print("-" * 110)

        baseline_throughput = None
        for method_name, stat in stats.items():
            tokens = stat["avg_tokens"]
            time_s = stat["avg_time"]
            throughput = stat["avg_throughput"]
            acceptance = stat["acceptance_len"]
            avg_size_tokens = stat["avg_size_tokens"]
            avg_depth_stop = stat["avg_depth_stop"]

            if method_name == "baseline":
                baseline_throughput = throughput
                print(f"{method_name:<20} {tokens:<10.1f} {time_s:<12.4f} {throughput:<14.2f} {'—':<12} {'—':<10} {'—':<12} {'—'}")
            else:
                acceptance_str = f"{acceptance:.2f}" if acceptance is not None else "—"
                size_str = f"{avg_size_tokens:.1f}" if avg_size_tokens is not None else "—"
                depth_str = f"{avg_depth_stop:.2f}" if avg_depth_stop is not None else "—"
                speedup_str = f"{throughput / baseline_throughput:.2f}x" if baseline_throughput else "—"
                print(f"{method_name:<20} {tokens:<10.1f} {time_s:<12.4f} {throughput:<14.2f} {acceptance_str:<12} {size_str:<10} {depth_str:<12} {speedup_str}")

        print()

    print("=" * 110)
    print("Analysis complete!")


def plot_results(results: dict, save_path: Optional[str] = None):
    """
    Generate benchmark comparison charts.

    Parameters
    ----------
    results  : the same dict passed to print_results
    save_path: if given, save the figure to this file path (PNG/PDF/…);
               otherwise display interactively.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    METHOD_LABELS = {
        "baseline": "Baseline",
        "eagle3": "Eagle3",
        "eagle3_rl": "Eagle3+RL",
    }
    METHOD_COLORS = {
        "baseline": "#888888",
        "eagle3": "#4C72B0",
        "eagle3_rl": "#DD8452",
    }

    all_stats = _collect_stats(results)
    if not all_stats:
        print("No data to plot.")
        return

    datasets = list(all_stats.keys())
    # collect which methods actually appear across all datasets
    methods = []
    for m in ["baseline", "eagle3", "eagle3_rl"]:
        if any(m in stats for stats in all_stats.values()):
            methods.append(m)

    has_rl_stats = any(
        all_stats[d].get("eagle3_rl", {}).get("avg_size_tokens") is not None
        for d in datasets
    )

    n_plots = 4 if has_rl_stats else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    fig.suptitle("Speculative Decoding Benchmark", fontsize=14, fontweight="bold", y=1.01)

    x = np.arange(len(datasets))
    n_methods = len(methods)
    bar_w = 0.7 / n_methods

    # ── (1) Throughput ────────────────────────────────────────────────────────
    ax = axes[0]
    for i, method in enumerate(methods):
        vals = [all_stats[d].get(method, {}).get("avg_throughput", 0) for d in datasets]
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, None))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Throughput (tok/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("tok/s")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)

    # ── (2) Speedup vs Baseline ───────────────────────────────────────────────
    ax = axes[1]
    spec_methods = [m for m in methods if m != "baseline"]
    n_spec = len(spec_methods)
    bar_w2 = 0.7 / max(n_spec, 1)
    for i, method in enumerate(spec_methods):
        speedups = []
        for d in datasets:
            base_tp = all_stats[d].get("baseline", {}).get("avg_throughput", 0)
            method_tp = all_stats[d].get(method, {}).get("avg_throughput", 0)
            speedups.append(method_tp / base_tp if base_tp else 0)
        offset = (i - (n_spec - 1) / 2) * bar_w2
        bars = ax.bar(x + offset, speedups, bar_w2, label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, None))
        for bar, v in zip(bars, speedups):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}x", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1×)")
    ax.set_title("Speedup vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Speedup")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)

    # ── (3) Acceptance Length ────────────────────────────────────────────────
    ax = axes[2]
    for i, method in enumerate(spec_methods):
        vals = [all_stats[d].get(method, {}).get("acceptance_len") or 0 for d in datasets]
        offset = (i - (n_spec - 1) / 2) * bar_w2
        bars = ax.bar(x + offset, vals, bar_w2, label=METHOD_LABELS.get(method, method),
                      color=METHOD_COLORS.get(method, None))
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Avg Acceptance Length")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("tokens")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)

    # ── (4) Eagle3-RL: draft size & depth stop ────────────────────────────────
    if has_rl_stats:
        ax = axes[3]
        size_vals = [all_stats[d].get("eagle3_rl", {}).get("avg_size_tokens") or 0 for d in datasets]
        depth_vals = [all_stats[d].get("eagle3_rl", {}).get("avg_depth_stop") or 0 for d in datasets]

        ax2 = ax.twinx()
        b1 = ax.bar(x - bar_w / 2, size_vals, bar_w, label="Avg Draft Tokens",
                    color=METHOD_COLORS["eagle3_rl"], alpha=0.85)
        b2 = ax2.bar(x + bar_w / 2, depth_vals, bar_w, label="Avg Depth Stop",
                     color="#55A868", alpha=0.85)
        for bar, v in zip(b1, size_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(b2, depth_vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_title("Eagle3+RL Policy Stats")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Draft Tokens")
        ax2.set_ylabel("Depth Stop")
        lines = [b1, b2]
        labels = ["Avg Draft Tokens", "Avg Depth Stop"]
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        ax2.legend(lines, labels, fontsize=8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def analyze_results(results_file: str, plot: bool = False, plot_dir: Optional[str] = None):
    """Load and analyze benchmark results from a JSON file."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded benchmark results from: {results_file}")
    print_results(results)

    if plot:
        stem = Path(results_file).stem
        out_dir = Path(plot_dir) if plot_dir else Path(results_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(out_dir / f"{stem}_charts.png")
        plot_results(results, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--results_file", type=str, default="",
                        help="Path to benchmark results JSON file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save charts alongside the results file")
    parser.add_argument("--plot_dir", type=str, default="",
                        help="Directory to save charts (default: same as results_file)")

    args = parser.parse_args()

    if not args.results_file:
        path = Path(__file__).resolve().parent / "results"
        jsons = sorted(path.glob("*.json"), reverse=True)
        if jsons:
            args.results_file = str(jsons[0])
        else:
            print(f"No results file found in {path}")
            return

    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return

    analyze_results(
        args.results_file,
        plot=args.plot,
        plot_dir=args.plot_dir if args.plot_dir else None,
    )


if __name__ == "__main__":
    main()
