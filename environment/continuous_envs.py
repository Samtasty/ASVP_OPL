import os
import sys
import numpy as np
import jax.numpy as jnp
from sklearn import datasets
from sklearn.linear_model import RidgeCV
from scipy.stats import norm

# Add base directory to system path
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from environment.env_base import Environment

# --- The Environment Registry ---
# This dictionary will hold a mapping from name to class for all environments in this file.
CONTINUOUS_ENV_REGISTRY = {}

def register_env(name):
    """A decorator to register a new continuous environment class."""
    def decorator(cls):
        CONTINUOUS_ENV_REGISTRY[name] = cls
        return cls
    return decorator

@register_env("advertising")
class Advertising(Environment):

    def __init__(self, name, **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            potentials_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Advertising, self).__init__(**kw)
        self.name = name
        self.dimension = (2,1)
        # self.n_samples = n_samples
        self.start_mean = 2.0
        self.start_std = 0.3
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean**2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma**2 / 2
        self.mus = [3, 1, 0.1]
        self.targets_sigma = 0.5
        self.evaluation_offline = False
        self.test_data = self.sample_data(10000, 0)
        self.logging_scale = 0.3
        self.parameter_scale = 1
        self.start=0
        self.param_dimension= (3,1)  

    def sample_contexts_targets(self, n_samples):

        X, y = datasets.make_moons(
            n_samples=n_samples, noise=0.05, random_state=self.rng
        )
        v = self._get_targets(y)
        return X, v

    def _get_targets(self, y):
        """
        Args
            y (np.array): group labels

        """
        n_samples = y.shape[0]
        groups = [
            self.rng.normal(loc=mu, scale=self.targets_sigma, size=n_samples)
            for mu in self.mus
        ]
        targets = np.ones_like(y, dtype=np.float64)
        for y_value, group in zip(np.unique(y), groups):
            targets[y == y_value] = group[y == y_value]

        return np.abs(targets)

    # def sample_logging_actions(self, n_samples):
    #     actions=self.rng.lognormal(
    #         mean=self.start_mu, sigma=self.start_sigma, size=n_samples )
        
    #     return actions

    def sample_logging_actions(self, n_samples):
        actions=self.rng.normal(
            loc=self.start_mean, scale=self.start_std, size=n_samples )
        
        return actions

    def generate_data(self, n_samples):
        """
        Sets up experiments and generates data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        tuple of np.ndarray
            - Actions sampled by the logging policy.
            - Contextual features.
            - Loss values.
            - Propensity scores for the actions.
            - Target values.
        """

        contexts, targets = self.sample_contexts_targets(n_samples)

        actions = self.sample_logging_actions(n_samples)
        losses = self.get_losses_from_actions(targets, actions)
        propensities = self.logging_policy(actions, self.start_mean, self.start_std)
        return actions, contexts, losses, propensities, targets

    @staticmethod
    # def logging_policy(action, mu, sigma):
    #     """Log-normal distribution PDF policy

    #     Args:
    #         action (np.array)
    #         mu (np.array): parameter of log normal pdf
    #         sigma (np.array): parameter of log normal pdf
    #     """
    #     return jnp.exp(-((jnp.log(action) - mu) ** 2) / (2 * sigma**2)) / (
    #         action * sigma * jnp.sqrt(2 * jnp.pi)
    #     )
    def logging_policy(action, mu, sigma):
        """Normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of normal pdf
            sigma (np.array): parameter of normal pdf
        """
        return jnp.exp(-((action - mu) ** 2) / (2 * sigma**2)) / (
            sigma * jnp.sqrt(2 * jnp.pi)
        )

    def get_logging_data(self, n_samples):
        """_summary_

        Args:
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """

        actions, contexts, losses, propensities, _ = self.generate_data(n_samples)
        return actions, contexts, losses, propensities

    def sample_data(self, n_samples, index):

        _, contexts, _, _, targets = self.generate_data(n_samples)
        return contexts, targets
        
    @staticmethod
    def get_losses_from_actions(targets, actions):
        return -np.maximum(
            np.where(
                actions < targets,
                actions / targets,
                -0.5 * actions + 1 + 0.5 * targets,
            ),
            -0.1,
        )

    def sample_losses(self, actions, contexts, targets):
        return self.get_losses_from_actions(targets, actions)

    def get_optimal_parameter(self, contextual_modelling):
        features, targets = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

        if contextual_modelling == "linear":
            embedding = features
        elif contextual_modelling == "polynomial":
            quadra_features = np.einsum("ij,ih->ijh", features, features).reshape(
                features.shape[0], -1
            )
            embedding = np.hstack([features, quadra_features])
        else:
            return
        pistar_determinist.fit(embedding, targets)
        return (
            np.concatenate(
                [np.array([pistar_determinist.intercept_]).reshape(1,1), pistar_determinist.coef_.reshape(2,1)], axis=0
            ),
            pistar_determinist,
        )

    class LoggingPolicy:
        """Logging policy for the environment
        """
        def __init__(self, env,random_seed):

            

            self.env = env
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = 'normal'

        def get_propensity(self, actions, features, targets):
            propensities = self.env.logging_policy(actions, self.env.start_mean, self.env.start_std)
            return propensities
        
        def get_actions_and_propensities(self, features, targets):
            n_samples=len(targets)
            actions = self.env.sample_logging_actions(n_samples)
            
            propensities = self.get_propensity(actions, features, targets)
            return actions, propensities
            
        def online_evaluation(self,env,random_seed):

            rng = np.random.RandomState(random_seed)
            contexts, targets = env.test_data
            size = contexts.shape[0]
            losses = []

            for i in range(10):
                sampled_actions = self.env.sample_logging_actions(len(targets))
                losses += [env.get_losses_from_actions(targets, sampled_actions)]

            losses_array = np.stack(losses, axis=0)
            return losses_array.mean()








@register_env("pricing")
class Pricing(Environment):
    """Parent class for Data"""

    def __init__(self, name, mode="quadratic", **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            targets_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Pricing, self).__init__(**kw)
        self.name = name
        self.dimension = (10,1)
        self.l = 3
        self.start_std = 0.5
        self.start_mean = 1.5
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean**2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma**2 / 2
        self.mode = mode
        self.a, self.b = self.get_functions(self.mode)
        self.test_data = self.sample_data(10000, 0)
        self.parameter_scale = 0.01
        self.param_dimension= (11,1)
        self.start_sigma= None
        self.logging_scale = 0.5

        
        

    def get_functions(self, mode):
        """
        Returns reward functions based on the specified mode.

        Parameters
        ----------
        mode : str
            Mode of the environment (e.g., "quadratic").

        Returns
        -------
        tuple of callables
            Functions `a` and `b` defining the reward structure.
        """

        a = lambda z: 2 * z**2
        b = lambda z: 0.6 * z
        return a, b

    def sample_contexts_targets(self, n_samples):

        X = self.rng.uniform(low=1, high=2, size=(n_samples, self.dimension[0]))
        v = self._get_targets(X)
        return X, v

    def sample_logging_actions(self, targets):
        p = self.rng.normal(loc=targets, scale=self.start_std)
        return p

    def generate_data(self, n_samples):
        """
        Sets up experiments and generates data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        tuple of np.ndarray
            - Actions sampled by the logging policy.
            - Contextual features.
            - Loss values.
            - Propensity scores for the actions.
            - Target values.
        """

        contexts, targets = self.sample_contexts_targets(n_samples)
        p = self.sample_logging_actions(targets)
        losses = self.get_losses_from_actions(targets, p)
        propensities = norm(loc=targets, scale=self.start_std).pdf(p)

        return p, contexts, losses, propensities, targets


    def sample_data(self, n_samples, index):

        _, contexts, _, _, targets = self.generate_data(n_samples)
        return contexts, targets

    def _get_targets(self, z):
        """
        Computes target values from context features.

        Parameters
        ----------
        z : np.ndarray
            Contextual features.

        Returns
        -------
        np.ndarray
            Target values computed as the mean of the first `l` dimensions.
        """

        return np.mean(z[:, : self.l], axis=1)

    def get_losses_from_actions(self, z_bar, actions):
        """
        Computes losses based on actions and target values.

        Parameters
        ----------
        z_bar : np.ndarray
            Target values.
        actions : np.ndarray
            Actions taken by the policy.

        Returns
        -------
        np.ndarray
            Loss values for each action-target pair.
        """

        epsilon_noise = self.rng.normal(loc=np.zeros_like(z_bar), scale=1)
        losses = -(actions * (self.a(z_bar) - self.b(z_bar) * actions) + epsilon_noise)
        return np.minimum(losses, np.zeros_like(losses))

    def sample_losses(self, actions, contexts, targets):
        """
        Computes losses based on actions and target values.
        """
        return self.get_losses_from_actions(targets, actions)

    def get_optimal_parameter(self, contextual_modelling):
        """
        Computes the optimal parameter for a given contextual modeling method.

        Parameters
        ----------
        contextual_modelling : str
            The type of contextual modeling to use (e.g., "linear", 
            "polynomial").

        Returns
        -------
        tuple
            - Optimal parameters as a numpy array.
            - Fitted RidgeCV model.
        """

        z, z_bar = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        optimal_prices = self.a(z_bar) / (2 * self.b(z_bar))
        if contextual_modelling == "linear":
            embedding = z
        elif contextual_modelling == "polynomial":
            quadra_z = np.einsum("ij,ih->ijh", z, z).reshape(z.shape[0], -1)
            embedding = np.hstack([z, quadra_z])
        else:
            return
        pistar_determinist.fit(embedding, optimal_prices)
        return (
            np.concatenate(
                [np.array([pistar_determinist.intercept_]), pistar_determinist.coef_]
            ).reshape(-1, 1),
            pistar_determinist,
        )

    class LoggingPolicy:
        """Logging Policy of the environment
        """
        def __init__(self, env,random_seed):
            self.env = env
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = 'normal'
            
          

        def get_propensity(self, actions, features, targets):
        

                return norm(loc=targets, scale=self.env.start_std).pdf(actions)

        def get_actions_and_propensities(self, features, targets):
            
            actions = self.env.sample_logging_actions(targets)
            propensities = self.get_propensity(actions, features, targets)
            return actions, propensities
        
        def online_evaluation(self, env):
            
            contexts, targets = env.test_data
            size = contexts.shape[0]
            losses = []

            for i in range(10):
                sampled_actions = env.rng.normal(loc=targets, scale=env.logging_scale)
                losses += [env.get_losses_from_actions(targets, sampled_actions)]

            losses_array = np.stack(losses, axis=0)
            return losses_array.mean()



@register_env("toy_env")
class ToyEnvironment(Environment):
    """
    A toy continuous-action environment that keeps the same method/function
    signatures as the 'Pricing' example you provided, but does not focus on
    'prices' or 'targets' in the same sense. The 'targets' here can be interpreted
    as the environment’s ideal or near-optimal actions for each context.
    
    The environment uses a reward function of the form:
        reward(x, a) = (x dot w_env) * a - a^2 + noise
    so the "best action" for each x is about (x dot w_env) / 2.
    We treat that as a 'target' to keep method names consistent.
    """
    def __init__(self, name="toy_env", mode="toy", **kw):
        """
        Attributes mirroring the original structure:
            name (str): name of the dataset/environment
            dimension (tuple): shape of the context
            l (int): how many features go into the 'target' or part of the environment
            start_std (float): std dev for the logging policy
            start_mean (float): an example mean (not used directly in the same way)
            start_sigma (float): log-normal param (unused here, but kept for signature)
            start_mu (float): log-normal param (unused here, but kept for signature)
            mode (str): environment mode (we just store it)
            w_env (np.ndarray): environment's internal weight vector to compute reward
            test_data: sample data for evaluation
            logging_scale (float): scale for alternative random draws in online eval
            parameter_scale (float): dummy param to mirror the original
            param_dimension (tuple): dummy param to mirror the original
        """
        super(ToyEnvironment, self).__init__(**kw)
        self.name = name
        self.dimension = (10, 1)   # 10 features, for example
        self.l = 3                # first 3 features matter for "targets"
        self.start_std = 0.5
        self.start_mean = 1.5
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean**2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma**2 / 2
        self.mode = mode

        # Let's define an internal weight vector for reward calculations
        # so the "best action" depends on x dot w_env
        self.w_env = np.array([0.5, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3, 0.4, -0.4, 0.5])  # Fixed weight vector

        # Prepare some test data
        self.test_data = self.sample_data(10000, 0)

        # Additional attributes to match original structure
        self.parameter_scale = 0.01
        self.param_dimension = (11, 1)
        self.logging_scale = 0.5

    def get_functions(self, mode):
        """
        In your original code, this returned some reward function parameters (a, b).
        Here, we'll keep the signature but do nothing with it, or just return placeholders.
        """
        a = lambda z: z   # dummy
        b = lambda z: z   # dummy
        return a, b

    def sample_contexts_targets(self, n_samples):
        """
        Mirrors 'sample_contexts_targets' from the original. We interpret 'targets' as
        'best actions' or near-optimal actions for each context.
        """
        # Create random contexts
        X = self.rng.normal(loc=0.0, scale=1.0, size=(n_samples, self.dimension[0]))

        # We'll define "targets" = (x dot w_env) / 2, i.e. the environment's near-optimal action
        # You can also incorporate only the first self.l features if you wish.
        # For consistency with your original snippet, let's define:
        #   'z_bar' or 'targets' as the average of the first l features, but also
        #   let's add a factor to demonstrate that we have a distinct "best" action dimension.
        # In a simpler approach, let's do:
        #   targets = x dot w_env / 2
        # to give each sample a single "target" scalar.
        targets = np.dot(X, self.w_env) / 2.0
        
        return X, targets

    def _get_targets(self, X):
        """
        This was in the original code to create 'z_bar'. We'll do the same logic as above,
        but separated out so the original call structure remains consistent.
        """
        return np.dot(X, self.w_env) / 2.0

    def sample_logging_actions(self, targets):
        """
        In your Pricing code, the logging policy draws from Normal(target, self.start_std).
        We'll replicate that logic here:
            action ~ N( target, start_std )
        so the logging policy is suboptimal if the environment's actual optimum differs
        from 'target'. (In reality, it's the same if we set target= (x dot w_env)/2, but
        you can add a shift if you want to be suboptimal.)
        """
        # If you want a suboptimal logging policy, you could shift or scale 'targets'.
        # For instance:
        #   p = self.rng.normal(loc=targets * 1.3, scale=self.start_std)
        #   return p
        # But let's keep it consistent with the original snippet:
        p = self.rng.normal(loc=targets, scale=self.start_std)
        return p

    def get_losses_from_actions(self, targets, actions):
        """
        Convert the environment's reward to a 'loss' perspective, as in your snippet.
        
        We'll define:
            reward(x, a) = (x dot w_env)*a - a^2 + noise
        Then:
            loss = -reward
        If we want to replicate your snippet's final clamp with `np.minimum(losses, 0)`,
        we can do that too, but it's optional.
        """
        # For the environment's reward, we need the 'x dot w_env' factor. But we only
        # have 'targets' here, which we defined as (x dot w_env)/2. So:
        #   x dot w_env = 2 * targets
        # let's re-infer that in the reward formula.
        x_dot_w = 2 * targets  # because targets = (x dot w_env)/2
        noise = self.rng.normal(loc=0.0, scale=1.0, size=len(targets))
        
        reward = (x_dot_w * actions) - (actions**2) + noise
        losses = -reward

        # Optionally replicate the snippet's capping at 0:
        #   losses = np.minimum(losses, np.zeros_like(losses))
        # We'll do that for consistency:
        losses = np.minimum(losses, 0.0)
        return losses

    def sample_losses(self, actions, contexts, targets):
        """
        Just calls `get_losses_from_actions`, so your code can remain consistent.
        """
        return self.get_losses_from_actions(targets, actions)

    def generate_data(self, n_samples):
        """
        Core method to produce the (action, context, loss, propensity, target).
        Replicates your snippet's structure:
            1) sample contexts & targets
            2) sample logging actions
            3) compute losses
            4) compute propensities
        """
        contexts, targets = self.sample_contexts_targets(n_samples)
        p = self.sample_logging_actions(targets)
        losses = self.get_losses_from_actions(targets, p)

        # The snippet uses a normal with loc=targets, scale=start_std to compute pdf
        propensities = norm(loc=targets, scale=self.start_std).pdf(p)

        return p, contexts, losses, propensities, targets

    def sample_data(self, n_samples, index):
        """
        Just like in your snippet, returns (contexts, targets) for test or any usage.
        'index' is unused, but we keep it for signature compatibility.
        """
        _, contexts, _, _, targets = self.generate_data(n_samples)
        return contexts, targets

    def get_optimal_parameter(self, contextual_modelling):
        """
        In your snippet, you used a RidgeCV to regress the 'optimal price' from contexts.
        We'll do something analogous: We know the environment's best action is a* = (x dot w_env)/2,
        so we can do a RidgeCV to approximate that from X.
        
        The method returns:
            (best_params, fitted_ridge_model)
        """
        z, z_bar = self.sample_data(10000, 0)  # z are contexts, z_bar the 'targets'

        # The "true" optimum is exactly z_bar, but let's mimic the snippet's approach:
        #   we'll define a polynomial or linear embedding, then do Ridge to predict 'z_bar'.
        if contextual_modelling == "linear":
            embedding = z
        elif contextual_modelling == "polynomial":
            quadra_z = np.einsum("ij,ik->ijk", z, z).reshape(z.shape[0], -1)
            embedding = np.hstack([z, quadra_z])
        else:
            # fallback to linear
            embedding = z

        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        pistar_determinist.fit(embedding, z_bar)

        # Combine intercept + coefficients
        best_params = np.concatenate(
            [np.array([pistar_determinist.intercept_]), pistar_determinist.coef_]
        ).reshape(-1, 1)

        return best_params, pistar_determinist


    # ---------------------------------------------------------------------
    # Logging Policy Class with same methods
    # ---------------------------------------------------------------------
    class LoggingPolicy:
        """
        Logging policy with the same structure as your snippet:
        - get_propensity
        - get_actions_and_propensities
        - online_evaluation
        """

        def __init__(self, env, random_seed=123):
            self.env = env
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = "uniform"

        def get_propensity(self, actions, features, targets):
            """
            For a uniform distribution, the propensity is constant within the range.
            Let's assume the range is [min_action, max_action].
            """
            min_action = -1.0  # Define the range of actions
            max_action = 1.0
            propensities = np.ones_like(actions) / (max_action - min_action)
            return propensities

        def get_actions_and_propensities(self, features, targets):
            """
            Sample from a uniform distribution within a defined range.
            """
            min_action = -1.0  # Define the range of actions
            max_action = 1.0
            actions = self.rng.uniform(low=min_action, high=max_action, size=len(targets))
            propensities = self.get_propensity(actions, features, targets)
            return actions, propensities

        def online_evaluation(self, env):
            """
            Mirrors your snippet’s method, though it's just a placeholder for demonstration.
            We'll do 10 draws from some distribution and average the losses.
            """
            contexts, targets = env.test_data
            n = contexts.shape[0]
            losses_list = []

            for i in range(10):
                # Sample actions uniformly within the range
                min_action = -1.0
                max_action = 1.0
                sampled_actions = env.rng.uniform(low=min_action, high=max_action, size=n)
                losses = env.get_losses_from_actions(targets, sampled_actions)
                losses_list.append(losses)

            losses_array = np.stack(losses_list, axis=0)
            return losses_array.mean()
