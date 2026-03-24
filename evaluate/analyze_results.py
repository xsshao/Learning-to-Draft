"""
Simple analysis and visualization of benchmark results.
Usage: python evaluate/analyze_results.py --results_file ./evaluate/results/benchmark_results.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path


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
        if config.get("timestamp"):
            print(f"  Timestamp:     {config['timestamp']}")

    for dataset_name, dataset_results in results.items():
        if dataset_name == "config":
            continue
        print(f"\n{'Dataset: ' + dataset_name:^110}")
        print("-" * 110)

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


def analyze_results(results_file: str):
    """Load and analyze benchmark results from a JSON file."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"Loaded benchmark results from: {results_file}")
    print_results(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results_file",
        type=str,
        default='',
        help="Path to benchmark results JSON file"
    )

    args = parser.parse_args()

    if not args.results_file:
        # use the last file in the results directory
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

    analyze_results(args.results_file)


if __name__ == "__main__":
    main()
