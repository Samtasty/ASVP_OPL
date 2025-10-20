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



class RandomMixtureEstimator(Estimator):
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

    def find_proper_N(self,list_len_samples, confidence=0.95):
        """
        Compute the largest N so that, by Bonferroni‐corrected Normal approx.,
        P[ any multinomial count > its historical size ] ≤ 1 - confidence.
        """
        n = np.array(list_len_samples, dtype=float)
        M = n.sum()
        alphas = n / M
        T = len(n)

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
        3) Form weights = new_prop / logged_mixture_prop.
        4) Compute weighted loss = weights * observed_losses (lookup from losses_list).
        5) Optionally add KL penalty between new and mixture propensities.
        """
        # 1) sample mixture history
        # Number of samples for each round
        list_len_samples = [len(features_list[i]) for i in range(len(features_list))]

        # Compute frequency of samples per round (as a proportion) and convert to JAX array
        total_samples = sum(list_len_samples)
        freq_len_samples = np.array([a / total_samples for a in list_len_samples])


        loss_per_round = []
        kl_per_round = []
        for i, n_samples in enumerate(list_len_samples):

            # for each policy get a random subset of sample indices
            # if samples_per_mixture[i] > n_samples:
            #     ipdb.set_trace()
            

            # Get the policy's predicted propensities for the current round
            predictions = policy.get_propensity(
                actions_list[i], features_list[i], targets_list[i]
            )
            predictions_subset = predictions[random_subset_list[i]]
            # Compute the mixture of logged propensities: logged_propensities_matrix[i, :n_samples, :]
            # is assumed to have shape (n_samples, num_rounds) and is multiplied (dot-product)
            # with the frequency vector (shape: (num_rounds,))


            mixt_propens_subset = (
                logged_propensities_matrix[i, random_subset_list[i], :] @ freq_len_samples
            )

            # Compute the weighted loss for the current round
            round_loss = losses_list[i][random_subset_list[i]] * self.transform_weights(
                predictions_subset, mixt_propens_subset, policy.propensity_type
            )
            loss_per_round.append(round_loss)

            # If KL regularization is enabled, compute KL divergence between predictions and mixture
            mixt_propens = (
                logged_propensities_matrix[i, :n_samples, :] @ freq_len_samples
            )
            if self.kl_penalty:
                kl_div = self.kl_divergence(
                    predictions, mixt_propens, policy.propensity_type
                )
                kl_per_round.append(kl_div)

        # Concatenate losses from all rounds into a single estimate array
        estimate = np.concatenate(loss_per_round)

        if self.kl_penalty:
            # Convert the list of scalar KL divergences into a JAX array and sum them
            kl_total = np.mean(np.array(kl_per_round))
            return self.objective_function(estimate) + self.kl_reg * kl_total
        else:
            return self.objective_function(estimate)

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

        find_proper_N = self.find_proper_N(list_len_samples)

        # Sample from multinomial distribution
        num_mixture_samples = find_proper_N
        samples_per_mixture = rng.multinomial(
            num_mixture_samples, freq_len_samples
        )
        random_subset_list = []
        for i, num_samples_i in enumerate(samples_per_mixture):
            # Sample indices with replacement
            a=len(env.list_of_features[i])
            idx_subset = rng.choice(
                a, min(num_samples_i,a), replace=False
            )
            random_subset_list.append(idx_subset)
#%%

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
                random_subset_list
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


# %%
