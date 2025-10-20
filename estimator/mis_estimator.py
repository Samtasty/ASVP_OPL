import numpy as np


from estimator.base_estimator import Estimator
from policy.base_policy import Policy


class MultipleImportanceSamplingEstimator(Estimator):
    """Estimator for the classical Multiple importance sampling method

    Args:
        BaseEstimator (object): Base class for all estimators
    """

    def __init__(self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True, kl_reg=0.01, kl_penalty=True,seed=42, adaptive_clipping=False,adaptive_lambda=False):
        super().__init__(policy, clipping_parameter, lambda_, variance_penalty,seed=seed, adaptive_clipping=adaptive_clipping,adaptive_lambda=adaptive_lambda)
        self.kl_penalty = kl_penalty  # Ensure this is set correctly
        self.kl_reg = kl_reg  # Ensure this is set correctly

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
        """return the loss function for the mixture estimator

        Args:
            theta (Array):parameter of the policy to optimize
            features_list (list): list of the past context sampled from the previous round
            actions_list (list): list of the past actions sampled from the previous round
            logged_propensities_matrix (Array): matrix of the logged propensities for all the previous rounds
            losses_list (list): list of the past losses sampled from the previous round

        Returns:
            float: the mixture estimator objective
        """
        list_len_samples = [len(features_list[i]) for i in range(len(features_list))]
        
        if self.adaptive_clipping:
            total_samples = sum(list_len_samples)
            if total_samples > 0:
                self.clipping_parameter = 1.0 / total_samples

        loss_per_round = []
        kl_per_round = []
        for i, j in enumerate(list_len_samples):
            propensities = policy.get_propensity(
                actions_list[i], features_list[i], targets_list[i]
            )

            if np.isnan(propensities).any():
                
                print(f"Warning: propensities is NaN at iteration {len(list_len_samples)}")
            mis_propens = logged_propensities_matrix[i, :j, i]
            if np.isnan(mis_propens).any():
                
                print(f"Warning: propensities is NaN at iteration {len(list_len_samples)}")
            loss_per_round.append(
                losses_list[i]*self.transform_weights(propensities, mis_propens,policy.propensity_type )
            )


   
                # KL divergence for the current round
            if self.kl_penalty:
                kl_div = self.kl_divergence(
                    propensities, mis_propens,policy.propensity_type
                )

   
                kl_per_round.append(kl_div)
                
            

        estimate = np.concatenate(loss_per_round)

  
  

        if self.kl_penalty:  # Check if KL penalty is enabled
            kl_div = np.mean(np.array(kl_per_round))
            return self.objective_function(estimate) + self.kl_reg * kl_div
        else:
            return self.objective_function(estimate)  # Ensure compatibility with scipy

