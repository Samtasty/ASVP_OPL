import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize  # Add this import

import os
import sys
import csv


# Get the current working directory
base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)


from policy.base_policy import Policy
from policy.continuous_policy import ContinuousPolicy
from policy.discrete_policy import DiscretePolicy
from utils.utils import display_experiment, dataset_split, online_evaluation
from policy.models.contextual_modelling import ContextualModel
from utils.loader import (
    get_env_by_name,
    uniform_split,
    gaussian_split,
    doubling_split,
    split_method,
    get_estimator_by_name,
    get_policy_from_type,
    verify_settings,
    assessing_list_samples,
)

from utils.logger import LossHistory
from utils.utils import make_baselines_skylines
from sklearn.linear_model import LogisticRegression
import argparse


def experiment(args):

    settings = {
        "policy_type": args.policy_type,
        "contextual_modelling": args.contextual_modelling,
        "estimator": args.estimator,
        "data": args.data,
        "n_0": args.n_0,
        "M": args.M,
        "validation": args.validation,
        "pdf_type": args.pdf_type,
        "epsilon": args.epsilon,
        "clipping_parameter": args.clipping_parameter,
        "lambda_": args.lambda_,
        "variance_penalty": args.variance_penalty,
        "split": args.split,
        "seed": args.seed,
        "kl_penalty": args.kl_penalty,
        "kl_reg": args.kl_reg,
        "env_name": args.env_name,
        "display": args.display,
        "adaptive_clipping": args.adaptive_clipping,
        "adaptive_lambda":args.adaptive_lambda
    }
    verify_settings(settings)

    # %%
    lambda_grid = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    env = get_env_by_name(settings["env_name"], settings["seed"])

    total_samples = settings["n_0"] * (2 ** (settings["M"] - 1))
    list_of_samples = split_method(
        settings["split"], total_samples, settings["M"], settings["n_0"], env
    )

    print("Split sizes:", list_of_samples)
    print("Total requested:", sum(list_of_samples))

    display_experiment(
        settings["seed"],
        env,
        settings["estimator"],
        settings["clipping_parameter"],
        settings["variance_penalty"],
        settings["lambda_"],
        settings["kl_penalty"],
        settings["kl_reg"],
    )

    # %%
    # Model setting
    contextual_model = ContextualModel(
        settings["contextual_modelling"], settings["seed"]
    )
    pi_0_params = contextual_model.create_start_parameter(env)
    # create the policy that instantiate the estimator
    policy_0 = get_policy_from_type(settings["policy_type"])(
        parameter=pi_0_params,
        contextual_model=contextual_model,
        random_seed=settings["seed"],
        type=settings["pdf_type"],
        epsilon=settings["epsilon"],
        sigma=env.logging_scale,
        log_sigma=env.start_sigma,
    )

    # create the logging policy associated to the environment
    behavior_policy = env.LoggingPolicy(env, settings["seed"])

    ## Add the logging policy to the list of policies used in the environment
    env.add_policy(behavior_policy)

    # define the estimator
    estimator = get_estimator_by_name(settings["estimator"])(
        policy_0,
        settings["clipping_parameter"],
        settings["lambda_"],
        settings["variance_penalty"],
        kl_penalty=settings["kl_penalty"],
        kl_reg=settings["kl_reg"],
        seed=settings["seed"],
        adaptive_clipping=settings["adaptive_clipping"],
        adaptive_lambda=settings['adaptive_lambda']

    )
    loss_history = LossHistory(settings["estimator"])

    # compute the optimal theta
    if settings["policy_type"] == "continuous":
        optimal_theta, pistar_determinist = env.get_optimal_parameter(
            settings["contextual_modelling"]
        )
        optimal_loss = online_evaluation(
            optimal_theta, contextual_model, env, settings["seed"]
        )
    if settings["policy_type"] == "discrete":
        pi0, pistar = make_baselines_skylines(env.X_train, env.y_train)
        optimal_loss = env.online_evaluation_star(pistar)

    elif settings["policy_type"] == "multiclass":

        pistar = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=10000,
            random_state=42,
        )

        pistar.fit(env.X_train, env.y_train)
        optimal_loss = env.online_evaluation_star(pistar)

    # %%
    # Logging data
    list_of_samples = assessing_list_samples(env, list_of_samples)

    for i, j in enumerate(list_of_samples):
        # Defensive: check if enough data remains for this round
        # If not, break the loop (skip last partial/empty iteration)
        if hasattr(env, "start") and hasattr(env, "n_samples"):
            if env.start >= env.n_samples:
                print(f"[DEBUG] Skipping iteration {i} due to no data left.")
                break


        # sample data from the environment with the last policy
        env.update_round(j)
        # Print average reward for this round (for diagnosis)
        if hasattr(env, 'list_of_losses') and len(env.list_of_losses) > 0:
            round_losses = env.list_of_losses[-1]
            avg_reward = 1.0 - np.mean(round_losses) / round_losses.shape[-1] if round_losses.size > 0 else float('nan')
        
        cumulated_losses = np.sum(env.list_of_losses[-1])

        optimized_theta, loss_crm = estimator.optimize(
            env, model=contextual_model, type=settings["pdf_type"], seed=i
        )


        optimized_theta_val = optimized_theta.copy()

        # create a new policy with the optimized theta
        estimator.policy.parameter = optimized_theta_val
        new_policy = get_policy_from_type(settings["policy_type"])(
            parameter=optimized_theta_val,
            contextual_model=contextual_model,
            random_seed=i,
            type=settings["pdf_type"],
            epsilon=settings["epsilon"],
            sigma=env.logging_scale,
            log_sigma=env.start_sigma,
        )

        # Debug: show context_modelling output for first 5 test samples

        # add the new policy to the list of policies used in the environment
        if settings["estimator"] == "crm":
            env.add_policy(behavior_policy)
        else:
            env.add_policy(new_policy)


        # Evaluate and print both training and test (online) loss for overfitting diagnosis
        # Online loss (test set)
        online_loss = new_policy.online_evaluation(env)
        loss_crm = loss_crm  # Use the optimized loss value
        
        
        if np.isnan(loss_crm):

            print(f"Warning: loss_crm is NaN at iteration {i}")
            continue
        regret = online_loss - optimal_loss
        loss_history.update(
            optimized_theta_val,
            online_loss,
            regret,
            loss_crm,
            cumulated_losses,
            0.0,
            0.0,
            j,
        )

        if settings["display"]:

            loss_history.show_last()

    return loss_history



if __name__ == "__main__":

    env_name_list = [
        "yeast",
        "pricing",
        "tmc2007",
        "advertising",
        "yeast",
        "pricing",
        "tmc2007",
        "advertising",
        "toy_env",
        "scene",
        "discrete_synthetic",
        "synthetic_50_5",
        "synthetic_20_5",
        "synthetic_50_3",
        "synthetic_20_3",
        "iris",
        "wine",
        "breast_cancer",
    ]

    parser = argparse.ArgumentParser(
        description="Run scripts for the evaluation of methods"
    )
    parser.add_argument(
        "--n_0",
        nargs="?",
        type=int,
        default=100,
        help="initial number of samples",
    )
    parser.add_argument(
        "--policy_type",
        nargs="?",
        default="discrete",
        choices=["discrete", "continuous", "multiclass"],
        help="policy type",
    )
    parser.add_argument(
        "--estimator",
        nargs="?",
        default="osvp-pl",
        choices=[ "mixt", "crm", "scrm", "osvp-pl", "multi_mixt", "r_mixt", "sr_mixt"],
        help="estimator type",
    )
    parser.add_argument(
        "--contextual_modelling",
        nargs="?",
        default="linear",
        choices=["linear", "polynomial"],
        help="contextual modelling type",
    )
    parser.add_argument(
        "--data",
        nargs="?",
        default="geometrical",
        choices=["geometrical", "other"],
        help="data type",
    )
    parser.add_argument("--M", nargs="?", type=int, default=5, help="number of splits")
    parser.add_argument(
        "--validation",
        nargs="?",
        type=bool,
        default=False,
        help="validation flag",
    )
    parser.add_argument("--pdf_type", nargs="?", default=None, help="PDF type")
    parser.add_argument(
        "--epsilon", nargs="?", type=float, default=0, help="epsilon value"
    )
    parser.add_argument(
        "--clipping_parameter",
        nargs="?",
        type=float,
        default=1e-4,
        help="clipping parameter",
    )
    parser.add_argument(
        "--lambda_", nargs="?", type=float, default=1e-4, help="lambda value"
    )
    parser.add_argument(
        "--variance_penalty",
        nargs="?",
        type=bool,
        default=False,
        help="variance penalty flag",
    )
    parser.add_argument(
        "--split",
        nargs="?",
        default="doubling",
        type=str,
        choices=["doubling", "uniform", "gaussian", "linear_split"],
        help="split type",
    )
    parser.add_argument("--seed", nargs="?", type=int, default=42, help="seed value")
    parser.add_argument(
        "--env_name",
        nargs="?",
        default="yeast",
        help="environment name",
    )
    parser.add_argument(
        "--display", nargs="?", type=bool, default=False, help="display flag"
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

    parser.add_argument(
        "--kl_penalty",
        action="store_true",  # Use store_false to disable KL penalty
        help="Disable KL penalty (default: True).",
    )
    parser.add_argument(
        "--kl_reg",
        nargs="?",
        type=float,
        default=0.01,
        help="KL regularization parameter",
    )

    args = parser.parse_args()
    history = experiment(args)
    
        
