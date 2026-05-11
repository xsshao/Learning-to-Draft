"""Plot hardware-transfer benchmark summaries.

Usage:
  python evaluate/plot_hw_transfer_analysis.py \
    --summary_csv evaluate/results/hardware_transfer/Qwen3-14B/bf16/rtxa6000/summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


POLICY_ORDER = ["a5090_bf16", "bf16_hwproxy", "h100_bf16"]
POLICY_LABELS = {
    "a5090_bf16": "A5090 policy",
    "bf16_hwproxy": "HW-proxy policy",
    "h100_bf16": "H100 policy",
}
POLICY_COLORS = {
    "a5090_bf16": "#8C6D31",
    "bf16_hwproxy": "#1F77B4",
    "h100_bf16": "#2CA02C",
}
DATASET_ORDER = ["humaneval", "alpaca", "gsm8k", "mt_bench"]


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ordered(values: set[str], preferred: list[str]) -> list[str]:
    known = [value for value in preferred if value in values]
    extra = sorted(values.difference(known))
    return known + extra


def _mean(rows: list[dict[str, str]], metric: str, **filters: str) -> float:
    vals = [
        _float(row, metric)
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    return mean(vals) if vals else 0.0


def plot_policy_comparison(rows: list[dict[str, str]], save_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    policies = _ordered({row["source_policy"] for row in rows}, POLICY_ORDER)
    datasets = _ordered({row["dataset"] for row in rows}, DATASET_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Qwen3-14B Hardware Transfer on RTX A6000", fontsize=16, fontweight="bold")

    x = np.arange(len(datasets))
    bar_w = 0.72 / max(len(policies), 1)

    ax = axes[0, 0]
    for idx, policy in enumerate(policies):
        vals = [_mean(rows, "eagle3_rl_tp", source_policy=policy, dataset=dataset) for dataset in datasets]
        offset = (idx - (len(policies) - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy, "#777777"),
        )
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=8)

    baseline = [_mean(rows, "baseline_tp", source_policy="bf16_hwproxy", dataset=dataset) for dataset in datasets]
    eagle3 = [_mean(rows, "eagle3_tp", source_policy="bf16_hwproxy", dataset=dataset) for dataset in datasets]
    ax.plot(x, baseline, color="#666666", linestyle="--", marker="o", linewidth=1.5, label="Baseline")
    ax.plot(x, eagle3, color="#D62728", linestyle=":", marker="o", linewidth=2.0, label="Static Eagle3")
    ax.set_title("Eagle3+RL Throughput")
    ax.set_ylabel("tokens/s")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=9, ncol=2)

    ax = axes[0, 1]
    for idx, policy in enumerate(policies):
        vals = [_mean(rows, "rl_vs_eagle3", source_policy=policy, dataset=dataset) for dataset in datasets]
        offset = (idx - (len(policies) - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy, "#777777"),
        )
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title("RL Throughput / Static Eagle3")
    ax.set_ylabel("ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylim(0, max(1.08, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1, 0]
    for idx, policy in enumerate(policies):
        vals = [_mean(rows, "rl_vs_baseline", source_policy=policy, dataset=dataset) for dataset in datasets]
        offset = (idx - (len(policies) - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            bar_w,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy, "#777777"),
        )
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f}x", ha="center", fontsize=8)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.2)
    ax.set_title("RL Speedup vs Baseline")
    ax.set_ylabel("speedup")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.35)

    ax = axes[1, 1]
    policy_x = np.arange(len(policies))
    width = 0.24
    metrics = [
        ("rl_acceptance", "acceptance", "#9467BD"),
        ("rl_depth_stop", "depth stop", "#FF7F0E"),
    ]
    for idx, (metric, label, color) in enumerate(metrics):
        vals = [_mean(rows, metric, source_policy=policy) for policy in policies]
        bars = ax.bar(policy_x + (idx - 0.5) * width, vals, width, label=label, color=color)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06, f"{value:.2f}", ha="center", fontsize=8)

    ax2 = ax.twinx()
    size_vals = [_mean(rows, "rl_size_tokens", source_policy=policy) for policy in policies]
    ax2.plot(policy_x, size_vals, color="#17BECF", marker="D", linewidth=2.2, label="draft tokens")
    for xpos, value in zip(policy_x, size_vals):
        ax2.text(xpos, value + 4, f"{value:.0f}", ha="center", fontsize=8, color="#0B6F78")

    ax.set_title("Policy Behavior")
    ax.set_ylabel("tokens / depth")
    ax2.set_ylabel("draft tokens")
    ax.set_xticks(policy_x)
    ax.set_xticklabels([POLICY_LABELS.get(policy, policy) for policy in policies], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, fontsize=9, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_policy"]].append(row)

    print("Policy summary across datasets")
    print("policy         rl tok/s  rl/eagle3  rl/base  accept  size  depth")
    print("-" * 72)
    for policy in _ordered(set(grouped), POLICY_ORDER):
        policy_rows = grouped[policy]
        print(
            f"{policy:<13} "
            f"{mean(_float(row, 'eagle3_rl_tp') for row in policy_rows):>8.2f} "
            f"{mean(_float(row, 'rl_vs_eagle3') for row in policy_rows):>10.3f} "
            f"{mean(_float(row, 'rl_vs_baseline') for row in policy_rows):>8.3f} "
            f"{mean(_float(row, 'rl_acceptance') for row in policy_rows):>7.2f} "
            f"{mean(_float(row, 'rl_size_tokens') for row in policy_rows):>5.1f} "
            f"{mean(_float(row, 'rl_depth_stop') for row in policy_rows):>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hardware-transfer benchmark summaries")
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("evaluate/results/hardware_transfer/Qwen3-14B/bf16/rtxa6000/summary.csv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.summary_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.summary_csv}")

    output = args.output or args.summary_csv.with_name("hardware_transfer_policy_comparison.png")
    plot_policy_comparison(rows, output)
    print_summary(rows)
    print(f"\nSaved policy comparison chart to {output}")


if __name__ == "__main__":
    main()
