import jax
import numpy as np
import ipdb
from estimator.base_estimator import Estimator

class SequentialImportanceSamplingEstimator(Estimator):
    """Estimator for sequential importance sampling with optional KL divergence regularization.

    In the sequential setting, only the most recent round is used. The estimator:
      - Retrieves the logging policy's predicted propensities for the last round.
      - Extracts the corresponding logged propensities.
      - Computes the weighted loss for the last round.
      - Optionally adds a KL divergence penalty between the predicted and logged propensities.

    Args:
        policy (Policy): The logging policy used for data collection.
        clipping_parameter (float): Parameter for clipping weights.
        lambda_ (float): Regularization parameter.
        variance_penalty (bool): Whether to include a variance penalty.
        kl_reg (float): Regularization strength for the KL divergence term.
        kl_penalty (bool): If True, the KL divergence term is computed and added.
    """
    
    def __init__(self, policy, clipping_parameter=1e-6, lambda_=1e-4, 
                 variance_penalty=True, kl_reg=0.01, kl_penalty=True,seed=42, adaptive_clipping=False,adaptive_lambda=False):
        super().__init__(policy, clipping_parameter, lambda_, variance_penalty,seed=seed, adaptive_clipping=adaptive_clipping,adaptive_lambda=adaptive_lambda)
        self.kl_reg = kl_reg
        self.kl_penalty = kl_penalty

    def kl_divergence(self, p, q,propensity_type):

        """Compute the KL divergence between two distributions p and q.
        
        D_KL(p || q) = sum_i p_i * log(p_i / q_i)
        """

        if propensity_type == "normal":

            return np.mean(p * np.log(p / q),axis=0)
        elif propensity_type == "logarithmic":
            probs=np.exp(p)
            return np.mean(probs * (p-q),axis=0)


    def loss_function(
        self,
        policy,
        features_list,
        targets_list,
        actions_list,
        logged_propensities,
        losses_list,
    ):
        """
        Compute the loss function for the sequential importance sampling estimator with optional KL regularization.
        
        This function uses only the last round (most recent context, targets, actions, logged propensities, and losses).
        It retrieves the policy's predicted propensities for the last round, then computes a weighted loss using
        your transformation function. If KL regularization is enabled, it adds a penalty based on the divergence between
        the predicted and logged propensities.
        
        Args:
            policy (Policy): The logging policy.
            features_list (list): List of feature arrays from previous rounds.
            targets_list (list): List of target arrays from previous rounds.
            actions_list (list): List of action arrays from previous rounds.
            logged_propensities (Array): Logged propensities (e.g., shape [num_rounds, num_samples, num_rounds]).
            losses_list (list): List of loss arrays from previous rounds.
        
        Returns:
            float: The sequential importance sampling estimator objective.
        """
        # Determine the number of samples in each round
        list_len_samples = [len(features) for features in features_list]

        if self.adaptive_clipping:
            total_samples = list_len_samples[-1]
            if total_samples > 0:
                self.clipping_parameter = 1.0 / total_samples
        
        # Use only the last round's data
        predictions = policy.get_propensity(
            actions_list[-1], features_list[-1], targets_list[-1]
        )
        logged_propens = logged_propensities[-1, : list_len_samples[-1], -1]
        
        # Compute the weighted loss for the last round
        estimate = losses_list[-1] * self.transform_weights(
            predictions, logged_propens, policy.propensity_type
        )
        
        if self.kl_penalty:
            kl_div = self.kl_divergence(predictions, logged_propens,policy.propensity_type)
            return self.objective_function(estimate) + self.kl_reg * kl_div
        else:
            return self.objective_function(estimate)
