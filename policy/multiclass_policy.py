from policy.base_policy import Policy

import numpy as np

import jax


from scipy.special import softmax



class MulticlassPolicy(Policy):
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
        epsilon=None,
        contextual_model=None,
        propensity_type="normal",
        sigma=None,
        type="multiclass",
        log_sigma=None,
    ):
        super().__init__()
        self.parameter = parameter
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.epsilon = 0.0 if epsilon is None else float(epsilon)
        self.type = type
        self.contextual_model = contextual_model
        self.propensity_type = propensity_type

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
        return self.contextual_model._linear_modelling(parameter, features)
    
    def _class_probs(self, features):
        logits = self.context_modelling(features)
        # numeric stability
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = softmax(logits, axis=1)
        if self.epsilon > 0.0:
            k = probs.shape[1]
            probs = (1.0 - self.epsilon) * probs + self.epsilon * (1.0 / k)
        return probs

    def get_propensity(self, actions, features, targets):
        probs = self._class_probs(features)
        actions = np.asarray(actions, dtype=int)
        return probs[np.arange(actions.shape[0]), actions]

    def get_actions_and_propensities(self, features, targets):
        probs = self._class_probs(features)
        n, k = probs.shape
        actions = np.array([self.rng.choice(k, p=p) for p in probs], dtype=int)
        propensities = probs[np.arange(n), actions]
        return actions, propensities

    def online_evaluation(self, env):
        """Perform online evaluation of the policy.

        Args:
            env (Environment): Environment object with test data.

        Returns:
            float: Average propensity across the test data.
        """

        contexts, targets = env.X_test, env.y_test
        logits = self.context_modelling(contexts)
        actions = logits.argmax(axis=1)
        losses = env.sample_losses(actions, contexts, targets)

        return np.mean(losses)
