"""
Fit hardware proxy-cost coefficients from fixed-grid benchmark results.

The intended input is evaluate/benchmark_grid_sweep.py output. Each static
(depth, size) configuration gives enough aggregate structure to fit:

  elapsed_time_per_cycle ~= c_draft  * drafted_tokens
                         + c_verify * verify_tokens
                         + c_depth  * depth_steps
                         + c_kv     * accepted_tokens
                         + c_launch * launch_steps
                         + c_fixed

By default, coefficients are normalized by baseline autoregressive token
latency so the output is in the same rough "target-token equivalent" units used
by rl/hardware_reward.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


COEFF_KEYS = ("draft", "verify", "depth", "kv", "launch", "fixed")
GRID_KEY_RE = re.compile(r"^d(?P<depth>\d+)_s(?P<size>\d+)$")


def _expand_paths(patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted({path for path in paths if path.is_file()})


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _baseline_token_times(dataset_results: Dict[str, Any]) -> List[float]:
    times = []
    for row in dataset_results.get("baseline", []):
        tokens = _float_or_none(row.get("tokens"))
        elapsed = _float_or_none(row.get("time"))
        if tokens and elapsed and tokens > 0 and elapsed > 0:
            times.append(elapsed / tokens)
    return times


def _row_from_grid_sample(
    sample: Dict[str, Any],
    depth: int,
    size: int,
    target_mode: str,
) -> Optional[Tuple[np.ndarray, float, float]]:
    elapsed = _float_or_none(sample.get("elapsed_time"))
    cycles = _float_or_none(sample.get("num_cycles"))
    avg_accept = _float_or_none(sample.get("avg_acceptance_len"))

    if elapsed is None or cycles is None or avg_accept is None:
        return None
    if elapsed <= 0 or cycles <= 0 or avg_accept <= 0:
        return None

    verify_tokens = size + 1
    launch_steps = depth + 2

    if target_mode == "total":
        features = np.array(
            [
                size * cycles,
                verify_tokens * cycles,
                depth * cycles,
                avg_accept * cycles,
                launch_steps * cycles,
                cycles,
            ],
            dtype=np.float64,
        )
        target = elapsed
        weight = 1.0
    else:
        features = np.array(
            [size, verify_tokens, depth, avg_accept, launch_steps, 1.0],
            dtype=np.float64,
        )
        target = elapsed / cycles
        # Samples with more cycles provide a less noisy per-cycle estimate.
        weight = cycles

    return features, target, weight


def rows_from_grid_results(
    paths: Sequence[Path],
    datasets: Sequence[str],
    gpu_name_contains: str,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], Dict[str, Any]]:
    rows = []
    targets = []
    weights = []
    baseline_times = []
    stats = {
        "files_seen": 0,
        "files_used": 0,
        "datasets_used": {},
        "rows": 0,
    }

    wanted_datasets = set(datasets)
    gpu_filter = gpu_name_contains.lower().strip()

    for path in paths:
        stats["files_seen"] += 1
        with path.open() as f:
            data = json.load(f)

        gpu_name = str(data.get("config", {}).get("gpu_name", ""))
        if gpu_filter and gpu_filter not in gpu_name.lower():
            continue

        file_used = False
        for dataset_name, dataset_results in data.items():
            if dataset_name == "config":
                continue
            if wanted_datasets and dataset_name not in wanted_datasets:
                continue
            if not isinstance(dataset_results, dict):
                continue

            baseline_times.extend(_baseline_token_times(dataset_results))
            dataset_row_count = 0

            for key, samples in dataset_results.items():
                match = GRID_KEY_RE.match(key)
                if not match or not isinstance(samples, list):
                    continue
                depth = int(match.group("depth"))
                size = int(match.group("size"))
                for sample in samples:
                    row = _row_from_grid_sample(sample, depth, size, target_mode)
                    if row is None:
                        continue
                    features, target, weight = row
                    rows.append(features)
                    targets.append(target)
                    weights.append(weight)
                    dataset_row_count += 1

            if dataset_row_count:
                file_used = True
                stats["datasets_used"][dataset_name] = (
                    stats["datasets_used"].get(dataset_name, 0) + dataset_row_count
                )

        if file_used:
            stats["files_used"] += 1

    if not rows:
        raise ValueError("No calibration rows found. Did you pass grid_sweep JSON files?")

    X = np.vstack(rows)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    stats["rows"] = int(len(y))
    return X, y, w, baseline_times, stats


def rows_from_table(paths: Sequence[Path], target_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    targets = []
    weights = []

    def add_record(record: Dict[str, Any]) -> None:
        elapsed = _float_or_none(record.get("elapsed_time", record.get("time")))
        if elapsed is None or elapsed <= 0:
            return

        values = []
        for key in COEFF_KEYS[:-1]:
            value = _float_or_none(record.get(key))
            if value is None:
                value = _float_or_none(record.get(f"{key}_tokens"))
            if value is None:
                value = _float_or_none(record.get(f"{key}_steps"))
            if value is None:
                return
            values.append(value)

        fixed = _float_or_none(record.get("fixed"))
        values.append(1.0 if fixed is None else fixed)

        cycles = _float_or_none(record.get("num_cycles"))
        if target_mode == "per_cycle" and cycles and cycles > 0:
            target = elapsed / cycles
            feature_scale = 1.0 / cycles
            features = np.asarray(values, dtype=np.float64) * feature_scale
            features[-1] = 1.0
            weight = cycles
        else:
            target = elapsed
            features = np.asarray(values, dtype=np.float64)
            weight = 1.0

        rows.append(features)
        targets.append(target)
        weights.append(weight)

    for path in paths:
        if path.suffix == ".jsonl":
            with path.open() as f:
                for line in f:
                    if line.strip():
                        add_record(json.loads(line))
        elif path.suffix == ".csv":
            with path.open(newline="") as f:
                for record in csv.DictReader(f):
                    add_record(record)
        else:
            raise ValueError(f"Unsupported table file type: {path}")

    if not rows:
        raise ValueError("No calibration rows found in table inputs.")

    return (
        np.vstack(rows),
        np.asarray(targets, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def fit_weighted_ridge(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
    nonnegative: bool,
) -> np.ndarray:
    if np.any(weights <= 0):
        raise ValueError("All weights must be positive.")

    sqrt_w = np.sqrt(weights)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    if not nonnegative:
        lhs = Xw.T @ Xw + ridge_alpha * np.eye(X.shape[1])
        rhs = Xw.T @ yw
        return np.linalg.solve(lhs, rhs)

    # Exact enough for six coefficients: enumerate active sets and solve ridge
    # least squares on each, keeping the best feasible non-negative solution.
    best_beta = None
    best_obj = float("inf")
    n_features = X.shape[1]
    for mask in range(1, 1 << n_features):
        active = [idx for idx in range(n_features) if mask & (1 << idx)]
        Xa = Xw[:, active]
        lhs = Xa.T @ Xa + ridge_alpha * np.eye(len(active))
        rhs = Xa.T @ yw
        try:
            beta_active = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_active = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.any(beta_active < -1e-12):
            continue

        beta = np.zeros(n_features, dtype=np.float64)
        beta[active] = np.maximum(beta_active, 0.0)
        residual = Xw @ beta - yw
        obj = float(residual @ residual + ridge_alpha * (beta @ beta))
        if obj < best_obj:
            best_obj = obj
            best_beta = beta

    if best_beta is None:
        return np.zeros(n_features, dtype=np.float64)
    return best_beta


def regression_metrics(X: np.ndarray, y: np.ndarray, weights: np.ndarray, beta: np.ndarray) -> Dict[str, float]:
    pred = X @ beta
    residual = pred - y
    rmse = math.sqrt(float(np.average(residual**2, weights=weights)))
    mae = float(np.average(np.abs(residual), weights=weights))
    y_mean = float(np.average(y, weights=weights))
    ss_res = float(np.sum(weights * residual**2))
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "weighted_rmse_seconds": rmse,
        "weighted_mae_seconds": mae,
        "weighted_r2": r2,
        "target_mean_seconds": y_mean,
    }


def coeff_dict(beta: np.ndarray) -> Dict[str, float]:
    return {key: float(value) for key, value in zip(COEFF_KEYS, beta)}


def coeff_string(coeffs: Dict[str, float]) -> str:
    return ",".join(f"{key}={coeffs[key]:.8g}" for key in COEFF_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit hardware proxy coefficients from calibration results.")
    parser.add_argument(
        "--grid_results",
        nargs="*",
        default=[],
        help="Grid-sweep JSON files or globs from evaluate/benchmark_grid_sweep.py.",
    )
    parser.add_argument(
        "--table",
        nargs="*",
        default=[],
        help="Optional CSV/JSONL calibration rows with elapsed_time and coefficient feature columns.",
    )
    parser.add_argument("--datasets", nargs="*", default=[], help="Optional dataset filter for grid results.")
    parser.add_argument(
        "--gpu_name_contains",
        default="A6000",
        help="Only use grid files whose config.gpu_name contains this string. Empty disables filtering.",
    )
    parser.add_argument(
        "--target_mode",
        choices=["per_cycle", "total"],
        default="per_cycle",
        help="Fit per-cycle latency or total sample latency.",
    )
    parser.add_argument("--ridge_alpha", type=float, default=1e-8)
    parser.add_argument("--allow_negative", action="store_true", help="Use unconstrained ridge regression.")
    parser.add_argument(
        "--baseline_token_time",
        type=float,
        default=0.0,
        help="Seconds per baseline target-model token. If omitted, derive from grid baseline rows.",
    )
    parser.add_argument(
        "--no_normalize_by_baseline",
        action="store_true",
        help="Leave output coefficients in seconds instead of baseline-token equivalent units.",
    )
    parser.add_argument("--output_json", default="", help="Optional path to write fit metadata and coefficients.")
    args = parser.parse_args()

    paths = _expand_paths(args.grid_results)
    table_paths = _expand_paths(args.table)

    Xs = []
    ys = []
    ws = []
    baseline_times: List[float] = []
    stats: Dict[str, Any] = {}

    if paths:
        X, y, w, grid_baseline_times, stats = rows_from_grid_results(
            paths,
            datasets=args.datasets,
            gpu_name_contains=args.gpu_name_contains,
            target_mode=args.target_mode,
        )
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        baseline_times.extend(grid_baseline_times)

    if table_paths:
        X, y, w = rows_from_table(table_paths, target_mode=args.target_mode)
        Xs.append(X)
        ys.append(y)
        ws.append(w)

    if not Xs:
        raise SystemExit("Pass at least one --grid_results or --table path/glob.")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    weights = np.concatenate(ws)

    beta_seconds = fit_weighted_ridge(
        X,
        y,
        weights=weights,
        ridge_alpha=args.ridge_alpha,
        nonnegative=not args.allow_negative,
    )
    metrics = regression_metrics(X, y, weights, beta_seconds)

    baseline_token_time = args.baseline_token_time
    if baseline_token_time <= 0 and baseline_times:
        baseline_token_time = float(np.median(baseline_times))

    if args.no_normalize_by_baseline or baseline_token_time <= 0:
        beta_reward = beta_seconds
        coefficient_units = "seconds"
    else:
        beta_reward = beta_seconds / baseline_token_time
        coefficient_units = "baseline_token_equivalents"

    seconds_coeffs = coeff_dict(beta_seconds)
    reward_coeffs = coeff_dict(beta_reward)

    output = {
        "coefficient_units": coefficient_units,
        "hardware_cost_coeffs": reward_coeffs,
        "hardware_cost_coeffs_string": coeff_string(reward_coeffs),
        "seconds_coeffs": seconds_coeffs,
        "baseline_token_time_seconds": baseline_token_time,
        "num_rows": int(len(y)),
        "feature_order": list(COEFF_KEYS),
        "target_mode": args.target_mode,
        "ridge_alpha": args.ridge_alpha,
        "nonnegative": not args.allow_negative,
        "metrics": metrics,
        "grid_stats": stats,
    }

    print("Fitted hardware coefficients")
    print("============================")
    print(f"Rows: {len(y)}")
    print(f"Units: {coefficient_units}")
    if baseline_token_time > 0:
        print(f"Baseline token time: {baseline_token_time:.8f} sec/token")
    print(f"Weighted R^2: {metrics['weighted_r2']:.4f}")
    print(f"Weighted RMSE: {metrics['weighted_rmse_seconds']:.8f} sec")
    print("")
    print("Seconds coefficients:")
    for key in COEFF_KEYS:
        print(f"  {key:>6}: {seconds_coeffs[key]:.10g}")
    print("")
    print("Reward-profile coefficients:")
    for key in COEFF_KEYS:
        print(f"  {key:>6}: {reward_coeffs[key]:.10g}")
    print("")
    print("Use this with the hardware-transfer benchmark/training:")
    print(f"  --hardware_cost_coeffs '{output['hardware_cost_coeffs_string']}'")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
