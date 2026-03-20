"""
Simple analysis and visualization of benchmark results.
Usage: python evaluate/analyze_results.py --results_file ./evaluate/results/benchmark_results.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def analyze_results(results_file: str):
    """Load and analyze benchmark results."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("=" * 100)
    print("BENCHMARK RESULTS ANALYSIS")
    print("=" * 100)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{'Dataset: ' + dataset_name:^100}")
        print("-" * 100)
        
        # Collect stats for each method
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
            }
            else:
                stats[method_name] = {
                    "avg_tokens": df["tokens_generated"].mean(),
                    "avg_time": df["elapsed_time"].mean(),
                    "avg_throughput": df["throughput"].mean(),
                    "acceptance_len": df.get("avg_acceptance_len", pd.Series()).mean(),
                }
        
        # Print table
        print(f"{'Method':<20} {'Tokens':<12} {'Time (s)':<12} {'Throughput':<12} {'Acceptance':<12}")
        print("-" * 100)
        
        baseline_throughput = None
        for method_name, stat in stats.items():
            tokens = stat["avg_tokens"]
            time_s = stat["avg_time"]
            throughput = stat["avg_throughput"]
            acceptance = stat["acceptance_len"]
            
            if method_name == "baseline":
                baseline_throughput = throughput
                print(f"{method_name:<20} {tokens:<12.1f} {time_s:<12.4f} {throughput:<12.2f} {'—':<12}")
            else:
                acceptance_str = f"{acceptance:.2f}" if acceptance else "—"
                speedup = (throughput / baseline_throughput) if baseline_throughput else 1.0
                print(f"{method_name:<20} {tokens:<12.1f} {time_s:<12.4f} {throughput:<12.2f} {acceptance_str:<12}")
                print(f"  → Speedup vs Baseline: {speedup:.2f}x")
        
        print()
    
    print("=" * 100)
    print("Analysis complete!")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--results_file",
        type=str,
        default="./evaluate/results/benchmark_results.json",
        help="Path to benchmark results JSON file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    analyze_results(args.results_file)


if __name__ == "__main__":
    main()
