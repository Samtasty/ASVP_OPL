from policy.base_policy import Policy

import numpy as np

import jax


from scipy.special import expit as jexpit


class DiscretePolicy(Policy):
    """A class representing a discrete policy.

    Args:
        parameter (ndarray): Policy parameters.
        random_seed (int): Seed for random number generation.
        epsilon (float, optional): Exploration-exploitation trade-off.
        type (str, optional): Distribution Law of the policy, in a discrete setting the distribution isn't defined analytically .
        contextual_model (object, optional): Model for contextual parameters, also None in a discrete setting.
        propensity_type (str, optional):  Are we using a log trick to treat the propensity ('normal' or 'logarithmic')
    """

    def __init__(
        self,
        parameter,
        random_seed,
        epsilon=0,
        sigma=None,
        log_sigma=None,
        type='discrete',
        contextual_model=None,
        propensity_type="logarithmic",
    ):
        super().__init__()
        self.parameter = parameter
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.epsilon = epsilon
        self.type = type
        self.contextual_model = contextual_model
        self.propensity_type = propensity_type
        self.start_sigma=None
        self.sigma=None


    def create_start_parameter(self, env):
        d = env.dimension
        beta = np.array(np.zeros(d))
        self.parameter = beta

    def context_modelling(self, features):
        """Model the context to get the parameter

        Args:
            features (array): array of feature vector
        """
        parameter = self.parameter.copy()
        wx = self.contextual_model._linear_modelling(parameter, features)
        return wx

    def get_propensity(self, actions, features, targets):
        # Return propensity of provided actions under this policy
        if features is None or features.size == 0:
            return np.array([])
        wx = self.context_modelling(features)  # (n, k)
        p1 = (1.0 - self.epsilon) * jexpit(wx) + self.epsilon * 0.5  # P(a=1|x)
        # Ensure actions is (n, k)
        if actions.ndim == 1:
            actions = actions.reshape(wx.shape[0], wx.shape[1])
        prop_per_label = np.where(actions == 1, p1, 1.0 - p1)
        prop_per_label = np.clip(prop_per_label, 1e-12, 1.0)
        log_prop = np.log(prop_per_label).sum(axis=1)
        return log_prop if self.propensity_type == "logarithmic" else np.exp(log_prop)


    def get_actions_and_propensities(self, features, targets):
        # targets unused for sampling; kept for signature compatibility
        if features is None or features.size == 0:
            return np.array([]), np.array([])
        wx = self.context_modelling(features)  # (n, k)
        p1 = (1.0 - self.epsilon) * jexpit(wx) + self.epsilon * 0.5
        actions = (self.rng.uniform(size=wx.shape) < p1).astype(int)
        prop_per_label = np.where(actions == 1, p1, 1.0 - p1)
        prop_per_label = np.clip(prop_per_label, 1e-12, 1.0)
        log_prop = np.log(prop_per_label).sum(axis=1)
        return (actions, log_prop) if self.propensity_type == "logarithmic" else (actions, np.exp(log_prop))


    def online_evaluation(self, env):
        # Expected Hamming loss (deterministic, comparable to LoggingPolicy)
        contexts, targets = env.X_test, env.y_test  # targets shape (n, k)
        wx = self.context_modelling(contexts)
        p1 = (1.0 - self.epsilon) * jexpit(wx) + self.epsilon * 0.5  # P(a=1|x)
        loss_matrix = np.where(targets == 1, 1.0 - p1, p1)
        return float(loss_matrix.mean())


        