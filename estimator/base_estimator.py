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


class Estimator(object):
    """Base class for all estimators

    Args:
        object (class): Base class for all estimators
    """

    def __init__(self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True,seed=42, adaptive_clipping=False, adaptive_lambda=False):
        """Initializes the Estimator.

        Parameters
        ----------
        settings : dict
            Estimator configuration settings.
        verbose : bool
            Whether to enable verbose output.
        args, kwargs : optional
            Additional arguments.

        """
        self.rng = np.random.RandomState(seed)
        self.clipping_parameter = clipping_parameter
        self.policy = policy
        self.list_of_parameters = []
        self.lambda_ = lambda_
        self.variance_penalty = variance_penalty  # Regular ERM or CRM
        self.adaptive_clipping = adaptive_clipping
        self.adaptive_lambda= adaptive_lambda

    @abstractmethod
    def loss_function(self, policy, features, targets, actions, propensities, losses):
        pass

    def transform_weights(self, propensities, logged_propensities, propensity_type):
        if propensity_type == "normal":
            if self.clipping_parameter:
                return propensities / (logged_propensities + self.clipping_parameter * propensities)
            else:
                return propensities / logged_propensities
        elif propensity_type == "logarithmic":
            if self.clipping_parameter:
                a = propensities - (logged_propensities + self.clipping_parameter * propensities)
                b = np.clip(a, -50, 50)
                return np.exp(b)
            else:
                a = propensities - logged_propensities
                b = np.clip(a, -50, 50)
                return np.exp(b)

    def objective_function(self, estimate):
        # Defensive: handle empty input
        if estimate is None or len(estimate) == 0:
            return 0.0

        if self.variance_penalty:
            if len(estimate) == 1:
                return estimate[0]
            else:
                a = np.mean(estimate)
                b = np.mean(estimate**2)
                return np.mean(estimate) + self.lambda_ * np.sqrt(b - a**2)/np.sqrt(len(estimate))

        else:
            return np.mean(estimate)

    def optimize(self, env, model, type, max_iter=1000, tol=1e-4, seed=42):

        if self.adaptive_clipping:
            list_len_samples = [len(features) for features in env.list_of_features]
            total_samples = sum(list_len_samples)
            if total_samples > 0:
                self.clipping_parameter = 1.0 / total_samples

        # Adaptive lambda for MIS estimator
        if self.adaptive_lambda and (self.__class__.__name__ == "MultipleImportanceSamplingEstimator" or self.__class__.__name__ == "MixtureEstimator"):
            list_len_samples = [len(features) for features in env.list_of_features]
            total_samples = sum(list_len_samples)
            if total_samples > 1:
                self.lambda_ = np.sqrt(np.log(np.log(18 * total_samples)))

        # Adaptive lambda for SIS estimator (based on last round only)
        if self.adaptive_lambda and self.__class__.__name__ == "SequentialImportanceSamplingEstimator":
            if hasattr(env, 'list_of_features') and len(env.list_of_features) > 0:
                n_last = len(env.list_of_features[-1])
                if n_last > 1:
                    self.lambda_ = np.sqrt(np.log(n_last * 18))

        initial_shape = self.policy.parameter.shape
        initial_theta = self.policy.parameter.flatten()



        def _loss(flattened_theta):
            theta = flattened_theta.reshape(initial_shape)

            if type in ["normal", "log_normal"]:
                policy = ContinuousPolicy(
                    theta, type=type, sigma=env.logging_scale,
                    log_sigma=env.start_sigma, contextual_model=model, random_seed=seed
                )
            elif type == "discrete":
                policy = DiscretePolicy(theta, contextual_model=model, random_seed=seed, epsilon=0)
            elif type == "multiclass":
                policy = MulticlassPolicy(theta, contextual_model=model, random_seed=seed, epsilon=0)

            loss = self.loss_function(
                policy,
                env.list_of_features,
                env.list_of_targets,
                env.list_of_actions,
                env.matrix_of_propensities,
                env.list_of_losses,
            )
            return loss


        # Tighter bounds for discrete policy parameters (to prevent saturation)

        bounds = [(-2, 2) for _ in range(len(initial_theta))]


        result = minimize(
            fun=_loss,
            x0=initial_theta,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False},
            tol=tol, bounds=bounds,
        )

        optimized_theta = result.x.reshape(initial_shape)  # Final reshape



        self.list_of_parameters.append(optimized_theta)
        fun_val = result.fun


        return optimized_theta, fun_val
