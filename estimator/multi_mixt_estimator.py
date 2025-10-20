import numpy as np
from estimator.base_estimator import Estimator
from policy.base_policy import Policy


def make_freq(list):
    return [list[i] / sum(list) for i in range(len(list))]


class MultiMixtureEstimator(Estimator):
    def __init__(
        self,
        policy,
        clipping_parameter=1e-6,
        lambda_=1e-4,
        variance_penalty=True,
        kl_reg=0.01,
        kl_penalty=True,
        seed=42,adaptive_clipping=False,adaptive_lambda=False
    ):
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

        freq_len_samples = [a / sum(list_len_samples) for a in list_len_samples]

        loss_per_round = []
        kl_per_round = []

        predictions = [
            policy.get_propensity(actions_list[m], features_list[m], targets_list[m])
            for m in range(len(list_len_samples))
        ]
        for i, j in enumerate(list_len_samples):
            loss_per_per = []
            for m in range(i + 1):
                mixt_propens = logged_propensities_matrix[
                    m, : list_len_samples[m], : i + 1
                ] @ make_freq(list_len_samples[: i + 1])

                loss_per_per.append(
                    losses_list[m]
                    * self.transform_weights(
                        predictions[m], mixt_propens, policy.propensity_type
                    )
                )

            if self.kl_penalty:
                kl_div = self.kl_divergence(predictions[i], mixt_propens,policy.propensity_type)
                kl_per_round.append(kl_div)

            loss_per_round.append(loss_per_per)

        estimate = np.concatenate(
            [np.concatenate(loss_per_round[j]) for j in range(len(loss_per_round))]
        )

        if self.kl_penalty:
            # Convert the list of scalar KL divergences into a numpy array and sum them
            kl_total = np.mean(np.array(kl_per_round))
            return self.objective_function(estimate) + self.kl_reg * kl_total
        else:
            return self.objective_function(estimate)
