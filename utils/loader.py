
from environment.env_multiclass import Multiclass
from environment.continuous_envs import CONTINUOUS_ENV_REGISTRY
from environment.discrete_envs import DISCRETE_ENV_REGISTRY

from estimator.is_estimator import ImportanceSamplingEstimator
from estimator.mis_estimator import MultipleImportanceSamplingEstimator
from estimator.mixture_estimator import MixtureEstimator
from estimator.sis_estimator import SequentialImportanceSamplingEstimator
from estimator.multi_mixt_estimator import MultiMixtureEstimator
from estimator.random_mixture_estimator import RandomMixtureEstimator
from estimator.sequential_mixture_estimator import SequentialMixtureEstimator


from policy.continuous_policy import ContinuousPolicy
from policy.discrete_policy import DiscretePolicy
from policy.multiclass_policy import MulticlassPolicy

import numpy as np
from scipy.stats import norm


env_dico = {}
env_dico.update(CONTINUOUS_ENV_REGISTRY)
env_dico.update(DISCRETE_ENV_REGISTRY)

# Manually add the datasets that are handled by the generic Multiclass class
for name in ["iris", "wine", "breast_cancer"]:
    env_dico[name] = Multiclass

def get_env_by_name(name, random_seed=42):
    # If the name follows the synthetic pattern, it will be handled by the Multiclass class
    if name.startswith("synthetic_"):
        return Multiclass(name=name, random_seed=random_seed)
    
    # Otherwise, look it up in the dynamically built dictionary.
    if name not in env_dico:
        raise ValueError(f"Environment '{name}' not found. Check registries and spelling.")
    
    return env_dico[name](name=name, random_seed=random_seed)



def uniform_split(total_samples, M, n_0):
    return [total_samples // M] * M


def gaussian_split(total_samples, M, n_0):
    # Generate M points from a Gaussian distribution

    x = np.linspace(-3, 3, M + 1)  # M+1 points to create M intervals

    cdf_values = norm.cdf(x)  # Get the CDF values for these points
    cdf_values = cdf_values - cdf_values[0]  # Normalize to start from 0
    cdf_values = cdf_values / cdf_values[-1]  # Normalize to end at 1
    split_points = np.diff(cdf_values)  # Get the differences to create intervals
    return (split_points * total_samples).astype(int)


def doubling_split(total_samples, M, n_0):
    return [n_0 * 2**i for i in range(M)]


def linear_split(total_samples, M, n_0):
    # Start with arithmetic sequence 1, 2, 3, ..., M
    seq = np.arange(1, M + 1)

    # Scale to sum exactly total_samples
    scaled_seq = seq * (total_samples / seq.sum())

    # Take the integer floor of the scaled sequence
    int_seq = np.floor(scaled_seq).astype(int)

    # Adjust for rounding errors to match exactly total_samples
    diff = total_samples - int_seq.sum()

    # Add the difference to the largest elements incrementally
    for i in range(diff):
        int_seq[-(i + 1)] += 1

    # Ensure the length of the array is M
    if len(int_seq) < M:
        int_seq = np.append(int_seq, [0] * (M - len(int_seq)))

    # Ensure the sum of the array is total_samples
    if int_seq.sum() != total_samples:
        diff = total_samples - int_seq.sum()
        int_seq[-1] += diff

    return int_seq


def find_max_x(y, M):
    x = 0
    while (x + 1) * (2 ** (M - 1)) < y:

        x += 1
    return x

def split_method(method, total_samples, M, n_0, dataset):
    """
    Determines the number of samples for each of the M rollouts.
    - Discrete envs (tmc2007/yeast/scene/discrete_synthetic): ignore n_0, use all data.
    - Others: keep existing behavior.
    """
    samples_to_split = getattr(dataset, 'n_samples', total_samples)
    is_discrete = getattr(dataset, 'name', None) in DISCRETE_ENV_REGISTRY

    if is_discrete:
        # Use all available samples, ignore n_0
        if M <= 0:
            return []
        if method == "uniform":
            base = samples_to_split // M
            arr = np.full(M, base, dtype=int)
            arr[: samples_to_split - base * M] += 1
            return arr.tolist()

        # Build weights per split, then convert to integer counts that sum exactly to samples_to_split
        if method == "gaussian":
            x = np.linspace(-3, 3, M + 1)
            cdf = norm.cdf(x)
            widths = np.diff((cdf - cdf[0]) / (cdf[-1] - cdf[0]))
            weights = widths
        elif method == "doubling":
            weights = 2.0 ** np.arange(M, dtype=float)
        elif method in ("linear_split", "linear"):
            weights = np.arange(1, M + 1, dtype=float)
        else:
            raise ValueError(f"Unknown split method: '{method}'")

        raw = weights / weights.sum() * float(samples_to_split)
        ints = np.floor(raw).astype(int)
        diff = int(samples_to_split - ints.sum())
        if diff > 0:
            # give extra ones to largest fractional parts
            frac = raw - ints
            order = np.argsort(-frac)
            ints[order[:diff]] += 1
        elif diff < 0:
            # remove ones from smallest fractional parts (rare)
            frac = raw - ints
            order = np.argsort(frac)
            for k in range(-diff):
                idx = order[k % M]
                if ints[idx] > 0:
                    ints[idx] -= 1
        return ints.tolist()

    # Non-discrete envs: keep existing logic (and safety truncation)
    if method == "uniform":
        list_samples = uniform_split(samples_to_split, M, n_0)
    elif method == "gaussian":
        list_samples = gaussian_split(samples_to_split, M, n_0)
    elif method == "doubling":
        list_samples = doubling_split(samples_to_split, M, n_0)
    elif method == "linear_split":
        list_samples = linear_split(samples_to_split, M, n_0)
    else:
        raise ValueError(f"Unknown split method: '{method}'")

    if sum(list_samples) > samples_to_split:
        cumulative_sum = np.cumsum(list_samples)
        cut_off_index = np.where(cumulative_sum > samples_to_split)[0][0]
        list_samples = list_samples[:cut_off_index + 1]
        if cut_off_index > 0:
            remaining_samples = samples_to_split - cumulative_sum[cut_off_index - 1]
            list_samples[-1] = remaining_samples
        else:
            list_samples[-1] = samples_to_split
        if len(list_samples) > 1 and list_samples[-1] < list_samples[-2]:
            list_samples = list_samples[:-1]

    return [int(s) for s in list_samples if s > 0]

dic_estimator = {
    "osvp-pl": MultipleImportanceSamplingEstimator,
    "crm": ImportanceSamplingEstimator,
    "scrm": SequentialImportanceSamplingEstimator,
    "mixt": MixtureEstimator,
    "multi_mixt": MultiMixtureEstimator,
    "r_mixt": RandomMixtureEstimator,
    "sr_mixt": SequentialMixtureEstimator,}


def get_estimator_by_name(name):
    return dic_estimator[name]


policy_dic = {
    "continuous": ContinuousPolicy,
    "discrete": DiscretePolicy,
    "multiclass": MulticlassPolicy,
}


def get_policy_from_type(name):
    return policy_dic[name]


def verify_settings(settings):
    if settings["env_name"] in ["pricing", "advertising", "toy_env"]:
        if settings["policy_type"] != "continuous":
            print(
                f"INFO: Environment '{settings['env_name']}' is a continuous environment. "
                f"Overriding policy_type to 'continuous'."
            )
            settings["policy_type"] = "continuous"

    # If the environment is 'toy_env', we can allow 'discrete'

    multiclass_envs = [
        "synthetic_50_5",
        "synthetic_20_5",
        "synthetic_50_3",
        "synthetic_20_3",
        "iris",
        "wine",
        "breast_cancer",
    ]

    if settings["env_name"].startswith("synthetic_") or settings["env_name"] in multiclass_envs:
        if settings["policy_type"] != "multiclass":
            print(
                f"INFO: Environment '{settings['env_name']}' is a multiclass environment. "
                f"Overriding policy_type to 'multiclass'."
            )
            settings["policy_type"] = "multiclass"


    if settings["policy_type"] == "continuous":
        settings["pdf_type"] = "normal"

    if settings["policy_type"] == "discrete":
        settings["pdf_type"] = "discrete"
    if settings["policy_type"] == "multiclass":
        settings["pdf_type"] = "multiclass"

def assessing_list_samples(env, list_samples):
    if env.name == "pricing" or env.name == "advertising" or env.name == "toy_env":
        return list_samples
    else:
        if env.X_train is None:
            raise ValueError(
                "env.X_train is None. Ensure that the data is properly loaded."
            )

        if list_samples is None:
            raise ValueError(
                "list_samples is None. Ensure that the list of samples is properly generated."
            )
        return list_samples
