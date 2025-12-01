"""
Run a configurable benchmark suite across environments and algorithms.

This script wraps the helpers in runtime_experiments.py to evaluate:
 - Classic Bernoulli bandits (epsilon-greedy, UCB1, Thompson sampling)
 - Contextual bandits (kNN-UCB, GP-UCB, OSVP-PL) on logistic, smooth GP, and GP-grid envs.
"""

import argparse
import csv
from typing import Dict, List, Any, Tuple

import numpy as np
from joblib import Parallel, delayed

from online_bandit_env import BernoulliBandit, ContextualBandit, SmoothGPBandit, GPContextualBandit
from runtime_experiments import (
    benchmark,
    contextual_benchmark,
    parse_means,
    parse_theta,
)


def _run_bandit_env_task(payload: Tuple[Dict[str, Any], int, int, int]) -> List[Dict]:
    cfg, env_seed, horizon, runs = payload
    means = np.array(cfg["means"], dtype=float)
    results = benchmark(means, horizon, runs, epsilon=0.1, seed=env_seed)
    rows = []
    for algo, metrics in results.items():
        rows.append(
            {
                "env": cfg["name"],
                "env_type": "bandit",
                "algorithm": algo,
                "horizon": horizon,
                "runs": runs,
                "avg_reward_step": metrics["avg_reward"] / float(horizon),
                "exp_cum_regret": metrics["avg_final_regret"],
                "regret_std": metrics["std_final_regret"],
                "regret_auc": metrics["avg_regret_auc"],
                "runtime_ms": metrics["avg_runtime_ms"],
            }
        )
    return rows


def _run_contextual_env_task(
    payload: Tuple[Dict[str, Any], int, int, int]
) -> List[Dict]:
    cfg, env_seed, horizon, runs = payload
    theta = parse_theta("", cfg["n_arms"], cfg["context_dim"], env_seed)
    results = contextual_benchmark(
        theta=theta,
        env_type=cfg["env_type"],
        horizon=horizon,
        runs=runs,
        seed=env_seed,
        knn_k=cfg.get("knn_k", 10),
        knn_alpha=cfg.get("knn_alpha", 1.0),
        gp_length_scale=cfg.get("gp_length_scale", 1.0),
        gp_noise=cfg.get("gp_noise", 1e-2),
        gp_delta=cfg.get("gp_delta", 0.1),
        gp_buffer=cfg.get("gp_buffer", 300),
        lambda_asvp=cfg.get("lambda_asvp", 0.1),
        smoothgp_kernel=cfg.get("smoothgp_kernel", "matern"),
        smoothgp_length=cfg.get("smoothgp_length", 0.4),
        smoothgp_nu=cfg.get("smoothgp_nu", 3.0),
        smoothgp_noise=cfg.get("smoothgp_noise", 0.5),
        smoothgp_grid=cfg.get("smoothgp_grid", 200),
        gpctx_kernel=cfg.get("gpctx_kernel", "matern"),
        gpctx_length=cfg.get("gpctx_length", 0.5),
        gpctx_nu=cfg.get("gpctx_nu", 1.5),
        gpctx_noise=cfg.get("gpctx_noise", 0.1),
        gpctx_grid=cfg.get("gpctx_grid", 400),
        osvp_update_k=cfg.get("osvp_update_k", 1),
        osvp_maxiter=cfg.get("osvp_maxiter", 20),
        osvp_adaptive_lambda=cfg.get("osvp_adaptive_lambda", False),
    )
    rows: List[Dict] = []
    for algo, metrics in results.items():
        rows.append(
            {
                "env": cfg["name"],
                "env_type": cfg["env_type"],
                "algorithm": algo,
                "horizon": horizon,
                "runs": runs,
                "avg_reward_step": metrics["avg_reward"] / float(horizon),
                "exp_cum_regret": metrics["avg_final_regret"],
                "regret_std": metrics["std_final_regret"],
                "regret_auc": metrics["avg_regret_auc"],
                "runtime_ms": metrics["avg_runtime_ms"],
            }
        )
    return rows


def _run_tasks_in_pool(
    tasks: List[Tuple],
    func,
    workers: int,
) -> List[Dict]:
    rows: List[Dict] = []
    if not tasks:
        return rows
    if workers == 1:
        for t in tasks:
            rows.extend(func(t))
        return rows

    n_jobs = workers if workers > 0 else -1
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(func)(t) for t in tasks
    )
    for res in results:
        rows.extend(res)
    return rows


def _parse_int_list(raw: str, default: List[int]) -> List[int]:
    if not raw:
        return default
    return [int(x) for x in raw.split(",") if x.strip()]


def run_bandit_suite(horizon: int, runs: int, seed: int, workers: int) -> List[Dict]:
    envs = [
        {"name": "bernoulli_gap", "means": [0.1, 0.2, 0.6, 0.9]},
        {"name": "bernoulli_close", "means": [0.45, 0.5, 0.55]},
    ]
    tasks = [
        (cfg, seed + idx * 37, horizon, runs) for idx, cfg in enumerate(envs)
    ]
    return _run_tasks_in_pool(tasks, _run_bandit_env_task, workers)


def run_contextual_suite(
    horizon: int,
    runs: int,
    seed: int,
    workers: int,
    n_arms_list: List[int],
    context_dims: List[int],
    osvp_update_k: int,
    osvp_maxiter: int,
    osvp_adaptive_lambda: bool,
) -> List[Dict]:
    # Base configs; n_arms/context_dim will be overridden per grid combo.
    base_envs = [
        {
            "name": "logistic",
            "env_type": "logistic",
            "knn_k": 10,
            "knn_alpha": 1.0,
            "gp_length_scale": 1.0,
            "gp_noise": 1e-2,
            "gp_delta": 0.1,
            "gp_buffer": 300,
            "lambda_asvp": 0.1,
            "osvp_update_k": osvp_update_k,
            "osvp_maxiter": osvp_maxiter,
            "osvp_adaptive_lambda": osvp_adaptive_lambda,
        },
        {
            "name": "smoothgp",
            "env_type": "smoothgp",
            "knn_k": 10,
            "knn_alpha": 1.0,
            "gp_length_scale": 1.0,
            "gp_noise": 1e-2,
            "gp_delta": 0.1,
            "gp_buffer": 300,
            "lambda_asvp": 0.1,
            "smoothgp_kernel": "matern",
            "smoothgp_length": 0.4,
            "smoothgp_nu": 3.0,
            "smoothgp_noise": 0.3,
            "smoothgp_grid": 150,
            "osvp_update_k": osvp_update_k,
            "osvp_maxiter": osvp_maxiter,
            "osvp_adaptive_lambda": osvp_adaptive_lambda,
        },
        {
            "name": "gpctx",
            "env_type": "gpctx",
            "knn_k": 10,
            "knn_alpha": 1.0,
            "gp_length_scale": 1.0,
            "gp_noise": 1e-2,
            "gp_delta": 0.1,
            "gp_buffer": 300,
            "lambda_asvp": 0.1,
            "gpctx_kernel": "matern",
            "gpctx_length": 0.5,
            "gpctx_nu": 1.5,
            "gpctx_noise": 0.1,
            "gpctx_grid": 200,
            "osvp_update_k": osvp_update_k,
            "osvp_maxiter": osvp_maxiter,
            "osvp_adaptive_lambda": osvp_adaptive_lambda,
        },
    ]

    envs = []
    for base in base_envs:
        for n_arms in n_arms_list:
            for d in context_dims:
                cfg = dict(base)
                cfg["n_arms"] = n_arms
                cfg["context_dim"] = d
                cfg["name"] = f"{base['name']}_a{n_arms}_d{d}"
                envs.append(cfg)

    tasks = [
        (cfg, seed + idx * 53, horizon, runs) for idx, cfg in enumerate(envs)
    ]
    return _run_tasks_in_pool(tasks, _run_contextual_env_task, workers)


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No results.")
        return
    header_cols = [
        ("env", 16),
        ("env_type", 10),
        ("algorithm", 18),
        ("avg_reward_step", 16),
        ("exp_cum_regret", 16),
        ("regret_std", 12),
        ("regret_auc", 14),
        ("runtime_ms", 12),
    ]
    header = "".join([f"{name:<{w}}" for name, w in header_cols])
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['env']:<16}"
            f"{row['env_type']:<10}"
            f"{row['algorithm']:<18}"
            f"{row['avg_reward_step']:>16.4f}"
            f"{row['exp_cum_regret']:>16.3f}"
            f"{row['regret_std']:>12.3f}"
            f"{row['regret_auc']:>14.1f}"
            f"{row['runtime_ms']:>12.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run benchmark suite over multiple bandit environments.")
    parser.add_argument(
        "--suite",
        choices=["bandit", "contextual", "all"],
        default="all",
        help="Which suite to run.",
    )
    parser.add_argument("--bandit-horizon", type=int, default=500, help="Horizon for Bernoulli bandits.")
    parser.add_argument("--bandit-runs", type=int, default=20, help="Runs for Bernoulli bandits.")
    parser.add_argument("--context-horizon", type=int, default=300, help="Horizon for contextual bandits.")
    parser.add_argument("--context-runs", type=int, default=10, help="Runs for contextual bandits.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--output", type=str, default="", help="Optional CSV output path.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (<=0 for auto, 1 for sequential).",
    )
    parser.add_argument(
        "--context-n-arms",
        type=str,
        default="3",
        help="Comma-separated list of arm counts for contextual env grid (e.g., '3,5').",
    )
    parser.add_argument(
        "--context-dims",
        type=str,
        default="3",
        help="Comma-separated list of context dimensions for contextual env grid (e.g., '3,5').",
    )
    parser.add_argument(
        "--osvp-update-k",
        type=int,
        default=1,
        help="Re-optimize OSVP-PL every k steps (contextual).",
    )
    parser.add_argument(
        "--osvp-maxiter",
        type=int,
        default=20,
        help="Max iterations for OSVP-PL L-BFGS (contextual).",
    )
    parser.add_argument(
        "--osvp-adaptive-lambda",
        action="store_true",
        help="Use adaptive lambda sqrt(log(log(18*T))) in OSVP-PL.",
    )
    args = parser.parse_args()

    n_arms_list = _parse_int_list(args.context_n_arms, default=[3])
    context_dims = _parse_int_list(args.context_dims, default=[3])

    all_rows: List[Dict] = []
    if args.suite in ("bandit", "all"):
        all_rows.extend(run_bandit_suite(args.bandit_horizon, args.bandit_runs, args.seed, args.workers))
    if args.suite in ("contextual", "all"):
        all_rows.extend(
            run_contextual_suite(
                args.context_horizon,
                args.context_runs,
                args.seed + 1000,
                args.workers,
                n_arms_list,
                context_dims,
                args.osvp_update_k,
                args.osvp_maxiter,
                args.osvp_adaptive_lambda,
            )
        )

    print_table(all_rows)
    if args.output:
        write_csv(all_rows, args.output)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
