"""
Lightweight benchmark for three classic non-parametric bandit algorithms.

Runs epsilon-greedy, UCB1, and Thompson sampling on a Bernoulli multi-armed
bandit and reports average reward, final regret, and runtime.
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError as exc:  # pragma: no cover - defensive
    GaussianProcessRegressor = None
    WhiteKernel = None
    RBF = None
    Matern = None

from online_bandit_env import (
    BernoulliBandit,
    ContextualBandit,
    SmoothGPBandit,
    GPContextualBandit,
)


@dataclass
class BanditResult:
    total_reward: float
    regrets: List[float]
    runtime: float




def epsilon_greedy(env: BernoulliBandit, horizon: int, epsilon: float) -> BanditResult:
    counts = np.zeros(env.n_arms, dtype=int)
    reward_sums = np.zeros(env.n_arms, dtype=float)
    total_reward = 0.0
    regrets: List[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        if t < env.n_arms:
            arm = t  # pull each arm once to seed averages
        elif env.rng.rand() < epsilon:
            arm = env.rng.randint(env.n_arms)
        else:
            averages = np.divide(
                reward_sums,
                counts,
                out=np.zeros_like(reward_sums),
                where=counts > 0,
            )
            arm = int(env.rng.choice(np.flatnonzero(averages == averages.max())))

        reward = env.pull(arm)
        counts[arm] += 1
        reward_sums[arm] += reward
        total_reward += reward
        regrets.append((t + 1) * env.best_mean - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def ucb1(env: BernoulliBandit, horizon: int) -> BanditResult:
    counts = np.zeros(env.n_arms, dtype=int)
    reward_sums = np.zeros(env.n_arms, dtype=float)
    total_reward = 0.0
    regrets: List[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        if t < env.n_arms:
            arm = t
        else:
            averages = reward_sums / counts
            confidence = np.sqrt(2.0 * np.log(t + 1) / counts)
            arm = int(np.argmax(averages + confidence))

        reward = env.pull(arm)
        counts[arm] += 1
        reward_sums[arm] += reward
        total_reward += reward
        regrets.append((t + 1) * env.best_mean - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def thompson_sampling(env: BernoulliBandit, horizon: int) -> BanditResult:
    alpha = np.ones(env.n_arms, dtype=float)
    beta = np.ones(env.n_arms, dtype=float)
    total_reward = 0.0
    regrets: List[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        samples = env.rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        reward = env.pull(arm)
        alpha[arm] += reward
        beta[arm] += 1.0 - reward
        total_reward += reward
        regrets.append((t + 1) * env.best_mean - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def knn_ucb(
    env: ContextualBandit, horizon: int, k: int, alpha: float
) -> BanditResult:
    data_x: List[List[np.ndarray]] = [[] for _ in range(env.n_arms)]
    data_r: List[List[float]] = [[] for _ in range(env.n_arms)]
    total_reward = 0.0
    cum_best = 0.0
    regrets: List[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        context = env.sample_context()
        ucb_values = []
        for arm in range(env.n_arms):
            if not data_x[arm]:
                ucb_values.append(float("inf"))
                continue
            X = np.stack(data_x[arm])
            rewards = np.asarray(data_r[arm], dtype=float)
            dists = np.linalg.norm(X - context, axis=1)
            nn = min(k, len(dists))
            idxs = np.argpartition(dists, nn - 1)[:nn]
            mean_est = float(rewards[idxs].mean())
            bonus = alpha * np.sqrt(np.log(t + 2) / max(1, len(idxs)))
            ucb_values.append(mean_est + bonus)

        arm = int(np.argmax(ucb_values))
        reward = env.pull(arm, context)
        best = env.best_mean(context)

        data_x[arm].append(context)
        data_r[arm].append(reward)

        total_reward += reward
        cum_best += best
        regrets.append(cum_best - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def gp_ucb(
    env: ContextualBandit,
    horizon: int,
    length_scale: float,
    noise: float,
    delta: float,
    buffer: int,
) -> BanditResult:
    if GaussianProcessRegressor is None:  # pragma: no cover - import guard
        raise ImportError("scikit-learn is required for GP-UCB.")

    kernels = [
        RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise)
        for _ in range(env.n_arms)
    ]
    models = [
        GaussianProcessRegressor(kernel=kernels[a], normalize_y=True, random_state=env.rng)
        for a in range(env.n_arms)
    ]
    X: List[np.ndarray] = [np.empty((0, env.d)) for _ in range(env.n_arms)]
    y: List[np.ndarray] = [np.empty((0, 1)) for _ in range(env.n_arms)]

    total_reward = 0.0
    cum_best = 0.0
    regrets: List[float] = []
    start = time.perf_counter()

    for t in range(horizon):
        context = env.sample_context().reshape(1, -1)
        beta_t = np.sqrt(2.0 * np.log((t + 2) ** 2 * np.pi**2 / (6.0 * delta)))

        ucb_values = []
        for arm in range(env.n_arms):
            if X[arm].shape[0] == 0:
                ucb_values.append(float("inf"))
                continue

            mu, std = models[arm].predict(context, return_std=True)
            ucb_values.append(float(mu[0]) + beta_t * float(std[0]))

        arm = int(np.argmax(ucb_values))
        reward = env.pull(arm, context.ravel())
        best = env.best_mean(context.ravel())

        # Update buffers
        if X[arm].shape[0] == 0:
            X[arm] = context
            y[arm] = np.array([[reward]])
        else:
            X[arm] = np.vstack([X[arm], context])[-buffer:]
            y[arm] = np.vstack([y[arm], [[reward]]])[-buffer:]
        models[arm].fit(X[arm], y[arm])

        total_reward += reward
        cum_best += best
        regrets.append(cum_best - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def osvp_pl(
    env: ContextualBandit,
    horizon: int,
    lambda_reg: float,
    update_every_k: int = 1,
    maxiter: int = 20,
    lambda_adaptive: bool = False,
) -> BanditResult:
    """Fully online ASVP-PL: re-fit at each step on logged data and act with the new policy."""
    rng = env.rng

    d = env.d
    n_arms = env.n_arms
    theta_curr = np.zeros((n_arms, d), dtype=float)

    contexts: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    behav_prop: List[float] = []

    start = time.perf_counter()
    total_reward = 0.0
    cum_best = 0.0
    regrets: List[float] = []

    for _ in range(horizon):
        ctx = env.sample_context()
        logits = theta_curr @ ctx
        pi = softmax(logits)
        act = rng.choice(n_arms, p=pi)
        behav_p = float(pi[act])
        rew = env.pull(act, ctx)

        contexts.append(ctx)
        actions.append(act)
        rewards.append(rew)
        behav_prop.append(behav_p)

        X = np.stack(contexts)
        a = np.asarray(actions, dtype=int)
        r = np.asarray(rewards, dtype=float)
        b = np.asarray(behav_prop, dtype=float)

        def objective(theta_flat: np.ndarray) -> float:
            theta = theta_flat.reshape(n_arms, d)
            logits = X @ theta.T  # (N, n_arms)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            chosen = np.clip(probs[np.arange(len(X)), a], 1e-8, 1.0)
            weights = chosen / np.clip(b, 1e-8, 1.0)
            est = np.mean(r * weights)
            variance = np.var(r * weights)
            lam = lambda_reg
            if lambda_adaptive and len(X) > 1:
                lam = np.sqrt(np.log(np.log(max(2.0, 18.0 * len(X)))))
            return -(est - lam * np.sqrt(variance / max(1, len(X))))


        if (_ + 1) % max(1, update_every_k) == 0:
            result = minimize(
                objective,
                theta_curr.ravel(),
                method="L-BFGS-B",
                options={"maxiter": maxiter, "disp": False},
            )
            theta_curr = result.x.reshape(n_arms, d)

        total_reward += rew
        cum_best += env.best_mean(ctx)
        regrets.append(cum_best - total_reward)

    return BanditResult(total_reward, regrets, time.perf_counter() - start)


def osvp_pl_no_var(
    env: ContextualBandit,
    horizon: int,
    update_every_k: int = 1,
    maxiter: int = 20,
) -> BanditResult:
    """OSVP-PL variant without variance penalization (lambda fixed to 0)."""
    return osvp_pl(env, horizon, lambda_reg=0.0, update_every_k=update_every_k, maxiter=maxiter, lambda_adaptive=False)


def build_algorithms(epsilon: float) -> Dict[str, Callable[[BernoulliBandit, int], BanditResult]]:
    return {
        "epsilon-greedy": lambda env, horizon: epsilon_greedy(env, horizon, epsilon),
        "ucb1": ucb1,
        "thompson-sampling": thompson_sampling,
    }


def build_contextual_algorithms(
    knn_k: int,
    knn_alpha: float,
    gp_length_scale: float,
    gp_noise: float,
    gp_delta: float,
    gp_buffer: int,
    lambda_asvp: float,
    osvp_update_k: int,
    osvp_maxiter: int,
    osvp_adaptive_lambda: bool,
) -> Dict[str, Callable[[ContextualBandit, int], BanditResult]]:
    algos: Dict[str, Callable[[ContextualBandit, int], BanditResult]] = {
        "knn-ucb": lambda env, horizon: knn_ucb(env, horizon, knn_k, knn_alpha),
        "osvp-pl": lambda env, horizon: osvp_pl(
            env, horizon, lambda_asvp, osvp_update_k, osvp_maxiter, osvp_adaptive_lambda
        ),
        "osvp-pl-novar": lambda env, horizon: osvp_pl_no_var(env, horizon, osvp_update_k, osvp_maxiter),
    }
    if GaussianProcessRegressor is not None:
        algos["gp-ucb"] = lambda env, horizon: gp_ucb(
            env, horizon, gp_length_scale, gp_noise, gp_delta, gp_buffer
        )
    return algos


def benchmark(
    arm_means: np.ndarray, horizon: int, runs: int, epsilon: float, seed: int
) -> Dict[str, Dict[str, Any]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if runs <= 0:
        raise ValueError("runs must be positive.")
    if arm_means.size == 0:
        raise ValueError("provide at least one arm mean.")

    algos = build_algorithms(epsilon)
    collected: Dict[str, Dict[str, List[float]]] = {
        name: {"reward": [], "final_regret": [], "runtime": [], "regret_curve": []}
        for name in algos
    }

    for run in range(runs):
        base_seed = seed + run * 997  # prime multiplier to reduce collisions
        for idx, (name, algo) in enumerate(algos.items()):
            env_seed = base_seed + idx * 13
            rng = np.random.RandomState(env_seed)
            env = BernoulliBandit(arm_means, rng)
            result = algo(env, horizon)
            collected[name]["reward"].append(result.total_reward)
            collected[name]["final_regret"].append(result.regrets[-1])
            collected[name]["runtime"].append(result.runtime * 1000.0)
            collected[name]["regret_curve"].append(result.regrets)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for name, data in collected.items():
        aggregated[name] = {
            "avg_reward": float(np.mean(data["reward"])),
            "avg_final_regret": float(np.mean(data["final_regret"])),
            "std_final_regret": float(np.std(data["final_regret"])),
            "avg_regret_auc": float(
                np.mean([np.sum(curve) for curve in data["regret_curve"]])
            ),
            "avg_runtime_ms": float(np.mean(data["runtime"])),
        }
        aggregated[name]["avg_regret_curve"] = np.mean(
            np.stack(data["regret_curve"]), axis=0
        )

    return aggregated


def contextual_benchmark(
    theta: np.ndarray,
    env_type: str,
    horizon: int,
    runs: int,
    seed: int,
    knn_k: int,
    knn_alpha: float,
    gp_length_scale: float,
    gp_noise: float,
    gp_delta: float,
    gp_buffer: int,
    lambda_asvp: float,
    smoothgp_kernel: str,
    smoothgp_length: float,
    smoothgp_nu: float,
    smoothgp_noise: float,
    smoothgp_grid: int,
    gpctx_kernel: str,
    gpctx_length: float,
    gpctx_nu: float,
    gpctx_noise: float,
    gpctx_grid: int,
    osvp_update_k: int,
    osvp_maxiter: int,
    osvp_adaptive_lambda: bool,
) -> Dict[str, Dict[str, Any]]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if runs <= 0:
        raise ValueError("runs must be positive.")
    if theta.size == 0:
        raise ValueError("provide contextual parameters.")

    algos = build_contextual_algorithms(
        knn_k,
        knn_alpha,
        gp_length_scale,
        gp_noise,
        gp_delta,
        gp_buffer,
        lambda_asvp,
        osvp_update_k,
        osvp_maxiter,
        osvp_adaptive_lambda,
    )
    collected: Dict[str, Dict[str, List[float]]] = {
        name: {"reward": [], "final_regret": [], "runtime": [], "regret_curve": []}
        for name in algos
    }

    def make_env(rng: np.random.RandomState):
        if env_type == "logistic":
            return ContextualBandit(theta, rng)
        elif env_type == "smoothgp":
            return SmoothGPBandit(
                d=theta.shape[1],
                n_arms=theta.shape[0],
                rng=rng,
                kernel_type=smoothgp_kernel,
                length_scale=smoothgp_length,
                nu=smoothgp_nu,
                noise=smoothgp_noise,
                grid_size=smoothgp_grid,
            )
        elif env_type == "gpctx":
            return GPContextualBandit(
                d=theta.shape[1],
                n_arms=theta.shape[0],
                rng=rng,
                kernel_type=gpctx_kernel,
                length_scale=gpctx_length,
                nu=gpctx_nu,
                noise=gpctx_noise,
                grid_size=gpctx_grid,
            )
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

    for run in range(runs):
        base_seed = seed + run * 1013
        for idx, (name, algo) in enumerate(algos.items()):
            env_seed = base_seed + idx * 17
            rng = np.random.RandomState(env_seed)
            env = make_env(rng)
            result = algo(env, horizon)
            collected[name]["reward"].append(result.total_reward)
            collected[name]["final_regret"].append(result.regrets[-1])
            collected[name]["runtime"].append(result.runtime * 1000.0)
            collected[name]["regret_curve"].append(result.regrets)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for name, data in collected.items():
        aggregated[name] = {
            "avg_reward": float(np.mean(data["reward"])),
            "avg_final_regret": float(np.mean(data["final_regret"])),
            "std_final_regret": float(np.std(data["final_regret"])),
            "avg_regret_auc": float(
                np.mean([np.sum(curve) for curve in data["regret_curve"]])
            ),
            "avg_runtime_ms": float(np.mean(data["runtime"])),
        }
        aggregated[name]["avg_regret_curve"] = np.mean(
            np.stack(data["regret_curve"]), axis=0
        )

    return aggregated


def parse_means(raw: str, n_arms: int, seed: int) -> np.ndarray:
    if n_arms <= 0:
        raise ValueError("n_arms must be positive.")
    if raw:
        values = [float(x) for x in raw.split(",") if x.strip()]
        return np.clip(np.array(values, dtype=float), 1e-3, 1 - 1e-3)
    rng = np.random.RandomState(seed)
    return rng.uniform(0.05, 0.95, size=n_arms)


def parse_theta(raw: str, n_arms: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if raw:
        values = [float(x) for x in raw.split(",") if x.strip()]
        expected = n_arms * dim
        if len(values) != expected:
            raise ValueError(f"Expected {expected} values for theta, got {len(values)}.")
        arr = np.array(values, dtype=float).reshape(n_arms, dim)
    else:
        arr = rng.normal(scale=1.0, size=(n_arms, dim))
    return arr


def print_summary(
    results: Dict[str, Dict[str, float]],
    arm_means: np.ndarray,
    horizon: int,
    runs: int,
    epsilon: float,
) -> None:
    print(f"Arms: {arm_means.tolist()} | horizon={horizon} | runs={runs} | epsilon={epsilon}")
    header = (
        f"{'algorithm':<18}"
        f"{'avg reward/step':>16}"
        f"{'exp cum regret':>16}"
        f"{'regret std':>12}"
        f"{'regret auc':>14}"
        f"{'runtime (ms)':>14}"
    )
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        avg_per_step = metrics["avg_reward"] / float(horizon)
        print(
            f"{name:<18}"
            f"{avg_per_step:>16.4f}"
            f"{metrics['avg_final_regret']:>16.3f}"
            f"{metrics['std_final_regret']:>12.3f}"
            f"{metrics['avg_regret_auc']:>14.1f}"
            f"{metrics['avg_runtime_ms']:>14.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark classic bandits (epsilon-greedy, UCB1, Thompson) or contextual nonparametric "
            "bandits (kNN-UCB, GP-UCB)."
        )
    )
    parser.add_argument("--mode", choices=["bandit", "contextual"], default="bandit", help="Benchmark mode.")
    parser.add_argument("--horizon", type=int, default=5000, help="Number of pulls per run.")
    parser.add_argument("--runs", type=int, default=20, help="Number of independent runs.")
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Exploration rate for epsilon-greedy."
    )
    parser.add_argument(
        "--means",
        type=str,
        default="",
        help="Comma-separated list of arm means; leave empty to sample randomly.",
    )
    parser.add_argument("--n-arms", type=int, default=5, help="Number of arms.")
    parser.add_argument(
        "--context-dim", type=int, default=5, help="Context dimension for contextual mode."
    )
    parser.add_argument(
        "--context-env",
        type=str,
        default="logistic",
        choices=["logistic", "smoothgp", "gpctx"],
        help="Contextual environment type.",
    )
    parser.add_argument(
        "--theta",
        type=str,
        default="",
        help="Comma-separated flattened theta for contextual mode (size n_arms*context_dim).",
    )
    parser.add_argument("--knn-k", type=int, default=10, help="k for kNN-UCB.")
    parser.add_argument("--knn-alpha", type=float, default=1.0, help="Bonus scale for kNN-UCB.")
    parser.add_argument(
        "--gp-length-scale", type=float, default=1.0, help="RBF length-scale for GP-UCB."
    )
    parser.add_argument("--gp-noise", type=float, default=1e-2, help="Noise term for GP-UCB.")
    parser.add_argument("--gp-delta", type=float, default=0.1, help="Confidence parameter for GP-UCB.")
    parser.add_argument(
        "--gp-buffer",
        type=int,
        default=300,
        help="Max stored samples per arm for GP-UCB to cap training time/memory.",
    )
    parser.add_argument(
        "--asvp-lambda",
        type=float,
        default=0.1,
        help="Variance penalty coefficient lambda for OSVP-PL.",
    )
    parser.add_argument(
        "--smoothgp-kernel",
        type=str,
        default="matern",
        choices=["matern", "rbf"],
        help="Kernel for smooth GP bandit environment.",
    )
    parser.add_argument(
        "--smoothgp-length-scale",
        type=float,
        default=0.4,
        help="Length-scale for smooth GP bandit environment.",
    )
    parser.add_argument(
        "--smoothgp-nu",
        type=float,
        default=3.0,
        help="Nu parameter for Matern kernel in smooth GP bandit.",
    )
    parser.add_argument(
        "--smoothgp-noise",
        type=float,
        default=0.5,
        help="Observation noise for smooth GP bandit.",
    )
    parser.add_argument(
        "--smoothgp-grid",
        type=int,
        default=200,
        help="Grid size for smooth GP bandit interpolation.",
    )
    parser.add_argument(
        "--gpctx-kernel",
        type=str,
        default="matern",
        choices=["matern", "rbf"],
        help="Kernel for GP contextual bandit environment.",
    )
    parser.add_argument(
        "--gpctx-length-scale",
        type=float,
        default=0.5,
        help="Length-scale for GP contextual bandit environment.",
    )
    parser.add_argument(
        "--gpctx-nu",
        type=float,
        default=1.5,
        help="Nu parameter for Matérn kernel in GP contextual bandit.",
    )
    parser.add_argument(
        "--gpctx-noise",
        type=float,
        default=0.1,
        help="Observation noise for GP contextual bandit.",
    )
    parser.add_argument(
        "--gpctx-grid",
        type=int,
        default=400,
        help="Grid size for GP contextual bandit interpolation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducibility.")
    parser.add_argument(
        "--osvp-update-k",
        type=int,
        default=1,
        help="Re-optimize OSVP-PL every k steps (>=1).",
    )
    parser.add_argument(
        "--osvp-maxiter",
        type=int,
        default=20,
        help="Max L-BFGS iterations for OSVP-PL updates.",
    )
    parser.add_argument(
        "--osvp-adaptive-lambda",
        action="store_true",
        help="Use adaptive lambda sqrt(log(log(18*T))) in OSVP-PL.",
    )
    args = parser.parse_args()

    if args.mode == "bandit":
        arm_means = parse_means(args.means, args.n_arms, args.seed)
        results = benchmark(arm_means, args.horizon, args.runs, args.epsilon, args.seed)
        print_summary(results, arm_means, args.horizon, args.runs, args.epsilon)
    else:
        theta = parse_theta(args.theta, args.n_arms, args.context_dim, args.seed)
        results = contextual_benchmark(
            theta,
            args.context_env,
            args.horizon,
            args.runs,
            args.seed,
            args.knn_k,
            args.knn_alpha,
            args.gp_length_scale,
            args.gp_noise,
            args.gp_delta,
            args.gp_buffer,
            args.asvp_lambda,
            args.smoothgp_kernel,
            args.smoothgp_length_scale,
            args.smoothgp_nu,
            args.smoothgp_noise,
            args.smoothgp_grid,
            args.gpctx_kernel,
            args.gpctx_length_scale,
            args.gpctx_nu,
            args.gpctx_noise,
            args.gpctx_grid,
            args.osvp_update_k,
            args.osvp_maxiter,
            args.osvp_adaptive_lambda,
        )
        print(
            f"Contextual theta shape: {theta.shape}, horizon={args.horizon}, runs={args.runs}, "
            f"kNN-k={args.knn_k}, GP buffer={args.gp_buffer}"
        )
        header = (
            f"{'algorithm':<18}"
            f"{'avg reward/step':>16}"
            f"{'exp cum regret':>16}"
            f"{'regret std':>12}"
            f"{'regret auc':>14}"
            f"{'runtime (ms)':>14}"
        )
        print(header)
        print("-" * len(header))
        for name, metrics in results.items():
            avg_per_step = metrics["avg_reward"] / float(args.horizon)
            print(
                f"{name:<18}"
                f"{avg_per_step:>16.4f}"
                f"{metrics['avg_final_regret']:>16.3f}"
                f"{metrics['std_final_regret']:>12.3f}"
                f"{metrics['avg_regret_auc']:>14.1f}"
                f"{metrics['avg_runtime_ms']:>14.2f}"
            )


if __name__ == "__main__":
    main()
