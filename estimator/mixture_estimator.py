import numpy as np
import jax

from estimator.base_estimator import Estimator
from policy.base_policy import Policy

class MixtureEstimator(Estimator):
    """Estimator for the mixture estimator with optional KL divergence regularization.

    The estimator computes a weighted loss per round by mixing the logged propensities.
    If KL regularization is enabled, it additionally penalizes discrepancies between
    the policy's predicted propensities and the weighted (mixture) logged propensities.

    Args:
        policy (Policy): The logging policy.
        clipping_parameter (float): Parameter for clipping weights.
        lambda_ (float): Regularization parameter.
        variance_penalty (bool): Whether to include a variance penalty.
        kl_reg (float): Regularization strength for the KL term.
        kl_penalty (bool): If True, the KL divergence term is included.
    """
    
    def __init__(self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True,
                 kl_reg=0.01, kl_penalty=True,seed=42,adaptive_clipping=False,adaptive_lambda=False):
        super().__init__(policy, clipping_parameter, lambda_, variance_penalty,seed=seed,adaptive_clipping=adaptive_clipping,adaptive_lambda=adaptive_lambda)
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
        logged_propensities_matrix,
        losses_list,
    ):
        """
        Compute the loss function for the mixture estimator with optional KL divergence.
        
        For each round:
          - The policy’s predicted propensities are retrieved.
          - A mixture of the logged propensities is computed using the relative frequency 
            of samples from all rounds.
          - A weighted loss is computed by combining the observed losses and the transformed
            weights (using your helper method transform_weights).
          - Optionally, the KL divergence between the predictions and the mixture is computed.
        
        The final objective is computed by concatenating the per-round losses and, if enabled,
        adding a KL penalty (weighted by kl_reg).
        
        Args:
            policy (Policy): The logging policy.
            features_list (list): List of feature arrays (one per round).
            targets_list (list): List of target arrays (one per round).
            actions_list (list): List of action arrays (one per round).
            logged_propensities_matrix (Array): Logged propensities from previous rounds.
            losses_list (list): List of loss arrays (one per round).
        
        Returns:
            float: The final mixture estimator objective.
        """
        # Number of samples for each round
        list_len_samples = [len(features_list[i]) for i in range(len(features_list))]
        
        # Compute frequency of samples per round (as a proportion) and convert to JAX array
        total_samples = sum(list_len_samples)
        freq_len_samples = np.array([a / total_samples for a in list_len_samples])
        
        loss_per_round = []
        kl_per_round = []
        for i, n_samples in enumerate(list_len_samples):
            # Get the policy's predicted propensities for the current round
            predictions = policy.get_propensity(
                actions_list[i], features_list[i], targets_list[i]
            )
            # Compute the mixture of logged propensities: logged_propensities_matrix[i, :n_samples, :]
            # is assumed to have shape (n_samples, num_rounds) and is multiplied (dot-product)
            # with the frequency vector (shape: (num_rounds,))
            mixt_propens = logged_propensities_matrix[i, :n_samples, :] @ freq_len_samples
            
            # Compute the weighted loss for the current round
            round_loss = losses_list[i] * self.transform_weights(predictions, mixt_propens, policy.propensity_type)
            loss_per_round.append(round_loss)
            
            # If KL regularization is enabled, compute KL divergence between predictions and mixture
            if self.kl_penalty:
                kl_div = self.kl_divergence(predictions, mixt_propens,policy.propensity_type)
                kl_per_round.append(kl_div)
        
        # Concatenate losses from all rounds into a single estimate array
        estimate = np.concatenate(loss_per_round)
        
        if self.kl_penalty:
            # Convert the list of scalar KL divergences into a JAX array and sum them
            kl_total = np.mean(np.array(kl_per_round))
            return self.objective_function(estimate) +self.kl_reg * kl_total
        else:
            return self.objective_function(estimate)
