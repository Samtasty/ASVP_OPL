import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed

# Add base directory to system path
base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

# Import experiment function
from experiments import experiment

def ensure_dir(directory):
    """Ensure that the specified directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)




def a_posteriori_experiments(estimator, dataset_name, args):
    """
    Run experiments for a single lambda value (args.lambda_) and return stats.
    """
    args.lambda_ = args.lambda_
    args.env_name = dataset_name
    args.estimator = estimator

    def run_experiment(seed):
        args.seed = seed
        return experiment(args=args)

    histories = Parallel(n_jobs=-1)(
        delayed(run_experiment)(seed) for seed in range(60, 80, 2)
    )

    losses = np.array([h.online_loss for h in histories])
    mean_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)

    baseline = np.array([h.losses_baseline for h in histories])
    mean_baseline = np.mean(baseline, axis=0)
    std_baseline = np.std(baseline, axis=0)

    skyline = np.array([h.losses_skyline for h in histories])
    mean_skyline = np.mean(skyline, axis=0)
    std_skyline = np.std(skyline, axis=0)

    return mean_losses, std_losses, mean_baseline[-1], std_baseline[-1], mean_skyline[-1], std_skyline[-1]


def compare_estimators(results, dataset_name, args, estimators=None):
    """
    Run a_posteriori_experiments for each estimator, plot them, and append performances.
    """
    all_estimators = ["is", "sis", "mis", "mixt", "multi_mixt", "r_mixt", "sr_mixt"]
    estimators = estimators or all_estimators

    losses_dict, stds_dict = {}, {}
    baseline_perf = baseline_std = skyline_perf = skyline_std = None

    for idx, est in enumerate(estimators):
        mean_losses, std_losses, bp, bs, sp, ss = a_posteriori_experiments(est, dataset_name, args)
        losses_dict[est] = mean_losses
        stds_dict[est] = std_losses
        if idx == 0:
            baseline_perf, baseline_std = bp, bs
            skyline_perf, skyline_std = sp, ss

    rollouts = np.arange(1, len(next(iter(losses_dict.values()))) + 1)

    fig, ax = plt.subplots(figsize=(14, 9), dpi=100, constrained_layout=True)
    ax.set_title(dataset_name)
    ax.set_xlabel("Rollouts $m$")
    ax.set_ylabel("Loss")

    cmap = plt.get_cmap("tab10")
    colors = {est: cmap(i) for i, est in enumerate(estimators)}
    for est in estimators:
        ax.plot(rollouts, losses_dict[est], "o-", label=est.upper(), color=colors[est])
        ax.fill_between(rollouts,
                        losses_dict[est] - stds_dict[est],
                        losses_dict[est] + stds_dict[est],
                        alpha=0.25, color=colors[est])
    ax.legend(loc="best")
    figures_dir = os.path.join("figures", args.policy_type, dataset_name)
    ensure_dir(figures_dir)

    # Build filename with requested flags
    fname_parts = [args.split, str(args.n_0), str(args.M), dataset_name]

    if getattr(args, "variance_penalty", False):
        fname_parts.append("variance_penalty")

    # Add lambda (format to scientific) and adaptive flag if set
    lambda_tag = f"lambda_{args.lambda_:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    fname_parts.append(lambda_tag)

    if getattr(args, "adaptive_lambda", False):
        fname_parts.append("adaptive_lambda")

    fname = "_".join(fname_parts) + ".pdf"
    plt.savefig(os.path.join(figures_dir, fname), bbox_inches="tight")
    plt.close()

    results["dataset"].append(dataset_name)
    results["Baseline"].append(f"$%.3f \pm %.3f$" % (baseline_perf, baseline_std))
    for est in estimators:
        perf, std = losses_dict[est][-1], stds_dict[est][-1]
        col = est.upper() if est != "r_mixt" else "RDM-Mixt"
        results[col].append(f"$%.3f \pm %.3f$" % (perf, std))
    results["Skyline"].append(f"$%.3f \pm %.3f$" % (skyline_perf, skyline_std))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scripts for the evaluation of methods")
    parser.add_argument("--n_0", type=int, default=100, help="initial number of samples")
    parser.add_argument("--policy_type", default="discrete",
                        choices=["discrete", "continuous", "multiclass"], help="policy type")
    parser.add_argument("--estimators", nargs='+', default=None,
                        choices=["is","sis","mis","mixt","multi_mixt","r_mixt","sr_mixt"],
                        help="Which estimators to run; if omitted, all will be tested")
    parser.add_argument("--datasets", nargs='+', default=None,
                         help="Which datasets to run; if omitted, defaults based on policy_type")
    parser.add_argument("--contextual_modelling", default="linear",
                        choices=["linear","polynomial"], help="contextual modelling type")
    parser.add_argument("--data", default="geometrical",
                        choices=["geometrical","other"], help="data type")
    parser.add_argument("--M", type=int, default=5, help="number of splits")
    parser.add_argument("--validation", action="store_true", help="Enable validation flag")
    parser.add_argument("--pdf_type", default=None, help="PDF type")
    parser.add_argument("--epsilon", type=float, default=0, help="epsilon value")
    parser.add_argument("--clipping_parameter", type=float,
                        default=1e-3, help="clipping parameter")
    parser.add_argument(
        "--lambda_", type=float, default=1e-4,
        help="Lambda value for experiments (default: 1e-4)"
    )
    parser.add_argument(
        "--adaptive_clipping",
        action="store_true",
        help="Enable adaptive clipping based on total sample size.",
    )

    parser.add_argument(
        "--adaptive_lambda",
        action="store_true",
        help="Enable adaptive lambda based on the number of rounds.",
    )

    parser.add_argument("--variance_penalty", action="store_true",
                        help="Enable variance penalty (default: False)")
    parser.add_argument("--split", default="doubling",
                        choices=["doubling","uniform","gaussian","linear_split"], help="split type")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument("--env_name", default="yeast",
                        choices=[
                            "yeast","pricing","tmc2007","advertising","toy_env","scene","discrete_synthetic",
                            "synthetic_50_5","synthetic_20_5","synthetic_50_3","synthetic_20_3",
                            "iris","wine","breast_cancer"
                        ], help="environment name")
    parser.add_argument("--display", action="store_true", help="Enable display flag")
    parser.add_argument("--kl_reg", type=float, default=0.01, help="KL regularization parameter")
    parser.add_argument("--kl_penalty", action="store_true", help="Enable KL penalty (default: False)")

    args = parser.parse_args()

    # Prepare results container
    results = defaultdict(list)

    # Determine datasets
    if args.datasets is None:
        if args.policy_type == "continuous":
            datasets = ["pricing","advertising","toy_env"]
        elif args.policy_type == "discrete":
            datasets = ["yeast","tmc2007","scene","discrete_synthetic"]
        else:
            datasets = [
                "synthetic_50_5","synthetic_20_5","synthetic_50_3",
                "synthetic_20_3","iris","wine","breast_cancer"
            ]
    else:
        datasets = args.datasets

    # Run experiments
    results_list = Parallel(n_jobs=-1)(
        delayed(compare_estimators)(defaultdict(list), ds, args, estimators=args.estimators)
        for ds in datasets
    )
    for res in results_list:
        for key, vals in res.items():
            results[key].extend(vals)

    # Save results DataFrame
    ensure_dir("dataframes")
    df = pd.DataFrame(results)
    # Build filename including optional flags
    fname_parts = [args.split, str(args.n_0), str(args.M), args.policy_type]
    if getattr(args, "variance_penalty", False):
        fname_parts.append("variance_penalty")
    if getattr(args, "adaptive_lambda", False):
        fname_parts.append("adaptive_lambda")
    tex_fname = "_".join(fname_parts) + ".tex"

    df.to_latex(os.path.join("dataframes", tex_fname), index=False, column_format="r", escape=False)
    print("-"*80)
    print(df.drop(columns=['Baseline','Skyline']))
    print("-"*80)
