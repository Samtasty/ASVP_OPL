import numpy as np
from estimator.base_estimator import Estimator
from policy.base_policy import Policy
from scipy.stats import norm
import ipdb

import os
import sys
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.optimize import minimize  # Use scipy for optimization

from policy.base_policy import Policy
from policy.continuous_policy import ContinuousPolicy
from policy.discrete_policy import DiscretePolicy
from policy.multiclass_policy import MulticlassPolicy
import ipdb



def make_frequency_list(list):
    return [list[i] / sum(list) for i in range(len(list))]



class SequentialMixtureEstimator(Estimator):
    """Mixture estimator that samples existing context-action pairs according
    to a mixture of past policies, then performs importance-weighted loss.

    The mixture weights \alpha_i are proportional to the number of samples
    each past policy generated. We first draw a batch of (x,a) by:
      1) drawing counts per policy via multinomial on \alpha,
      2) sampling with replacement from that policy's logged (x,a) history.
    Then we compute:
      - logged_propensity = \sum_i \alpha_i \pi_i(a|x)
      - new_propensity    = \pi_{current}(a|x)
    and form importance weights new_propensity / logged_propensity.
    Optionally add a KL divergence penalty between the current policy logits and
    the mixture logits.
    """

    def __init__(
        self,
        policy: Policy,
        clipping_parameter: float = 1e-6,
        lambda_: float = 1e-4,
        variance_penalty: bool = True,
        kl_reg: float = 0.01,
        kl_penalty: bool = True,
        seed=42,
    adaptive_clipping: bool = False,
    adaptive_lambda=False
    ):
        super().__init__(
            policy, clipping_parameter, lambda_, variance_penalty, seed=seed, adaptive_clipping=adaptive_clipping,adaptive_lambda=adaptive_lambda
        )
        self.kl_reg = kl_reg
        self.kl_penalty = kl_penalty

    def find_proper_N(self,list_len_samples,freq_len_samples, confidence=0.95):
        """
        Compute the largest N so that, by Bonferroni‐corrected Normal approx.,
        P[ any multinomial count > its historical size ] ≤ 1 - confidence.
        """
        n = np.array(list_len_samples, dtype=float)
        M = n.sum()
        alphas = np.array(freq_len_samples, dtype=float)
        T = len(n)
        if T == 1:
            return M
        else:

            # Bonferroni‐corrected two‐sided z
            delta = 1 - confidence
            z = norm.ppf(1 - delta/(2*T))

            # Solve per‐category quadratic for sqrt(N)
            N_candidates = []
            for n_i, alpha_i in zip(n, alphas):
                # a s^2 + b s - n_i = 0
                a = alpha_i
                b = z * np.sqrt(alpha_i * (1 - alpha_i))
                # positive root:
                s = (-b + np.sqrt(b*b + 4*a*n_i)) / (2*a)
                N_candidates.append(s**2)
                

            # floor of min candidate
            N = int(np.floor(min(N_candidates)))
            return N


    def kl_divergence(
        self, p: np.ndarray, q: np.ndarray, propensity_type: str
    ) -> float:
        """Compute D_KL(p || q)."""
        if propensity_type == "normal":
            return np.mean(p * np.log(p / q))
        else:
            probs = np.exp(p)
            return np.mean(probs * (p - q))

    def loss_function(
        self,
        policy,
        features_list,
        targets_list,
        actions_list,
        logged_propensities_matrix,
        losses_list,
        random_subset_list
    ) -> float:
        """
        Importance-sampling loss under current policy, using sampled mixture history.

        1) Draw a mixture batch of size `num_mixture_samples` from history.
        2) Compute new propensities under `policy`.
        3) Form weights = new prop / logged_mixture_prop.
        4) Compute weighted loss = weights * observed_losses (lookup from losses_list).
        5) Optionally add KL penalty between new and mixture propensities.
        """
        # 1) sample mixture history
        # Number of samples for each round
        list_len_samples = [len(features_list[i]) for i in range(len(features_list))]
        total_samples = sum(list_len_samples)
        freq_len_samples = np.array([a / total_samples for a in list_len_samples])

        loss_per_round = []
        kl_per_round = []

        # Iterate over each historical round i
        for i, n_samples in enumerate(list_len_samples):
            predictions = policy.get_propensity(
                actions_list[i], features_list[i], targets_list[i]
            )

            # handle two possible shapes for random_subset_list:
            # - simple case: random_subset_list[i] is a 1D array of indices (RandomMixtureEstimator)
            # - sequential case: random_subset_list[i] is a list/array of subsets for mixtures j>=i
            is_sequential_structure = (
                isinstance(random_subset_list[i], (list, tuple, np.ndarray))
                and len(random_subset_list) == len(list_len_samples)
                and hasattr(random_subset_list[i], "__len__")
                and len(random_subset_list[i]) >= 1
                and (len(random_subset_list[i]) == (len(list_len_samples) - i) or isinstance(random_subset_list[i][0], np.ndarray))
            )

            if is_sequential_structure and len(random_subset_list[i]) == (len(list_len_samples) - i):
                # nested/sequential subsets: random_subset_list[i][k] corresponds to mixture j = i + k
                for k, subset in enumerate(random_subset_list[i]):
                    j = i + k
                    if subset is None or len(subset) == 0:
                        continue
                    predictions_subset = predictions[subset]
                    mixt_propens_subset = (
                        logged_propensities_matrix[i, subset, : j + 1] @ make_frequency_list(list_len_samples[: j + 1])
                    )
                    round_loss = losses_list[i][subset] * self.transform_weights(
                        predictions_subset, mixt_propens_subset, policy.propensity_type
                    )
                    loss_per_round.append(round_loss)
            else:
                # simple case: random_subset_list[i] is an index array for round i
                subset = np.asarray(random_subset_list[i])
                if subset.size > 0:
                    predictions_subset = predictions[subset]
                    mixt_propens_subset = (
                        logged_propensities_matrix[i, subset, :] @ freq_len_samples
                    )
                    round_loss = losses_list[i][subset] * self.transform_weights(
                        predictions_subset, mixt_propens_subset, policy.propensity_type
                    )
                    loss_per_round.append(round_loss)

            # compute mixture propensities for KL (over full round)
            mixt_propens = (
                logged_propensities_matrix[i, :n_samples, :] @ freq_len_samples
            )
            if self.kl_penalty:
                kl_div = self.kl_divergence(
                    predictions, mixt_propens, policy.propensity_type
                )
                kl_per_round.append(kl_div)

        # after collecting all round losses, flatten and return
        if len(loss_per_round) == 0:
            # defensive: nothing sampled -> return large penalty
            return float(np.inf)

        # concatenate (each element may be 1D arrays)
        try:
            estimate = np.concatenate([np.ravel(x) for x in loss_per_round])
        except Exception:
            # fallback: attempt to flatten nested lists
            flattened = []
            for x in loss_per_round:
                flattened.extend(np.ravel(x).tolist())
            estimate = np.array(flattened)

        # check for NaNs
        if np.isnan(estimate).any():
            # make it explicit for debugging
            ipdb.set_trace()
            raise ValueError("NaN detected in estimate in SequentialMixtureEstimator.loss_function")

        if self.kl_penalty and len(kl_per_round) > 0:
            kl_total = np.mean(np.array(kl_per_round))
            return float(self.objective_function(estimate) + self.kl_reg * kl_total)
        else:
            return float(self.objective_function(estimate))

    def optimize(self, env, model, type, max_iter=1000, tol=1e-4, seed=42):

        initial_shape = self.policy.parameter.shape
        initial_theta = self.policy.parameter.flatten()

        # Set RNG once, fixed
        rng = np.random.RandomState(seed)

        # Pre-sample your mixture history once, outside the loss function
        # (This assumes your Estimator has a method like `_sample_mixture_batch`)

        list_len_samples = [len(env.list_of_features[i]) for i in range(len(env.list_of_features))]

        # Compute frequency of samples per round (as a proportion) and convert to JAX array
        total_samples = sum(list_len_samples)
        freq_len_samples = np.array([a / total_samples for a in list_len_samples])

       

        random_subset_list=np.zeros((len(list_len_samples), len(list_len_samples)), dtype=int)
        available_samples = np.array(list_len_samples, dtype=int).copy()

        if len(list_len_samples) == 1:
            # Handle the single-round case
            freq_len_samples = make_frequency_list(list_len_samples)
            find_proper_N_value = self.find_proper_N(available_samples, freq_len_samples)
            num_mixture_samples = rng.multinomial(find_proper_N_value, freq_len_samples)
            random_subset_list[0, :1] = num_mixture_samples
        else:
            # Handle the multi-round case
            for j in reversed(range(len(list_len_samples))):
                
                freq_len_samples = make_frequency_list(list_len_samples[:j+1])
                find_proper_N_value = self.find_proper_N(available_samples[:j+1], freq_len_samples)
                num_mixture_samples = rng.multinomial(find_proper_N_value, freq_len_samples)
                
                # Check if the last element is 0
                random_subset_list[j, :j+1] = num_mixture_samples
                available_samples = available_samples[:-1] - num_mixture_samples[:-1]
        index_subset=[]
    
        for j in range(len(list_len_samples)):
            
            m=random_subset_list[j:,j]
            c=np.arange(list_len_samples[j])
            index_subset.append(split_array_into_subsets(c, m, rng_seed=seed))




        

        def _loss(flattened_theta):
            theta = flattened_theta.reshape(initial_shape)

            if type in ["normal", "log_normal"]:
                policy = ContinuousPolicy(
                    theta,
                    type=type,
                    sigma=env.logging_scale,
                    log_sigma=env.start_sigma,
                    contextual_model=model,
                    random_seed=seed,
                )
            elif type == "discrete":
                policy = DiscretePolicy(
                    theta, contextual_model=model, random_seed=seed, epsilon=0
                )
            elif type == "multiclass":
                policy = MulticlassPolicy(
                    theta, contextual_model=model, random_seed=seed, epsilon=0
                )

            loss= self.loss_function(
                policy,
                env.list_of_features,
                env.list_of_targets,
                env.list_of_actions,
                env.matrix_of_propensities,
                env.list_of_losses,
                index_subset
            )
            return loss

        result = minimize(
            fun=_loss,
            x0=initial_theta,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False},
            tol=tol,
            bounds=[(-2, 2) for _ in range(len(initial_theta))],
        )

        optimized_theta = result.x.reshape(initial_shape)
        self.list_of_parameters.append(optimized_theta)
        fun_val = result.fun

        return optimized_theta, fun_val




def split_array_into_subsets(arr, subset_sizes, rng_seed=42):
    """
    Splits an array into subsets of specified sizes.

    Parameters:
        arr (numpy.ndarray): The array to split.
        subset_sizes (list): A list of integers specifying the sizes of each subset.
        rng_seed (int): Seed for the random number generator.

    Returns:
        list: A list of numpy arrays, each representing a subset.
    """
    rng = np.random.RandomState(rng_seed)

    # Ensure we don't ask for more elements than available
    total = sum(subset_sizes)
    if total > len(arr):
        raise ValueError("Sum of subset sizes exceeds array length.")

    # 1) Draw `total` unique indices without replacement
    indices = rng.choice(len(arr), size=total, replace=False)

    # 2) Split those indices into chunks of the specified sizes
    subsets = []
    start = 0
    for size in subset_sizes:
        end = start + size
        chunk_idx = indices[start:end]
        subsets.append(arr[chunk_idx])
        start = end

    return subsets
