import os
import sys
import numpy as np
from sklearn.datasets import load_svmlight_file, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, add_dummy_feature
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

# Add base directory to system path
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from environment.env_base import Environment
from utils.utils import make_baselines_skylines # Assuming this is where this helper lives

# --- The Discrete Environment Registry ---
DISCRETE_ENV_REGISTRY = {}

def register_env(name):
    """A decorator to register a new discrete environment class."""
    def decorator(cls):
        DISCRETE_ENV_REGISTRY[name] = cls
        return cls
    return decorator

# --- Helper for loading tmc2007, yeast, scene ---
def load_multilabel_dataset(dataset_name, test_size=0.25, seed=0, reduce_dim=None, scale=False, add_intercept=True):
    # Always resolve data path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    train_path = os.path.join(project_root, 'data', dataset_name, f'{dataset_name}_train.svm')
    test_path = os.path.join(project_root, 'data', dataset_name, f'{dataset_name}_test.svm')
    X_train, y_train_ = load_svmlight_file(train_path, multilabel=True)
    X_test, y_test_ = load_svmlight_file(test_path, multilabel=True)

    if reduce_dim:
        proj = GaussianRandomProjection(n_components=reduce_dim, random_state=seed)
        X_train = proj.fit_transform(X_train)
        X_test = proj.transform(X_test)
    try:
        X_train = np.array(X_train.todense())
        X_test = np.array(X_test.todense())
    except AttributeError:
        pass

    onehot = MultiLabelBinarizer()
    y_train = onehot.fit_transform(y_train_).astype(int)
    y_test = onehot.transform(y_test_).astype(int)

    X_all = np.vstack([X_train, X_test])
    if add_intercept:
        X_all = add_dummy_feature(X_all)
    y_all = np.vstack([y_train, y_test])

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print(f"{dataset_name} | X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train, X_test, y_test

# --- Base Class for tmc2007, yeast, scene ---
class BaseMultiLabelEnv(Environment):
    """Base class for shared logic between yeast, scene, tmc2007."""
    def __init__(self, name, mode="quadratic", test_size=0.25, reduce_dim=None, load_data=True, **kw):
        super(BaseMultiLabelEnv, self).__init__(**kw)
        self.name = name
        self.test_size = test_size
        self.start = 0
        if load_data:
            self.X_train, self.y_train, self.X_test, self.y_test = load_multilabel_dataset(
                self.name, self.test_size, seed=self.random_seed, reduce_dim=reduce_dim
            )
            # Feature normalization for all discrete envs
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            self.n_samples = self.X_train.shape[0]
            self.dimension = (self.X_train.shape[1], self.y_train.shape[1])
        self.l = 3
        self.start = 0
        self.mode = mode

        self.logging_scale = 0.5
        self.parameter_scale = 0.01
        self.start_sigma = None

    def sample_contexts_targets(self, n_samples):
        if self.start >= self.n_samples:
            return np.array([]), np.array([])
        b = min(self.start + n_samples, self.n_samples)
        X = self.X_train[self.start : b]
        y = self.y_train[self.start : b]
        self.start = b  # advance here only
        return X, y

    def sample_losses(self, actions, contexts, targets):
        # Defensive: handle empty actions
        if actions is None or actions.size == 0:
            return np.array([])
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)
        k = actions.shape[1]
        rewards = (1 - np.logical_xor(actions, targets)).sum(axis=1).reshape((-1,))
        return k - rewards
    

    def online_evaluation_star(self, pi_star):
        contexts, targets = self.X_test, self.y_test
        predictions = pi_star.predict_proba(contexts)
        predictions = np.array([_[:, 1] for _ in predictions]).T
        idx = np.where(targets == 0)
        fp = predictions[idx].sum()
        idx = np.where(targets == 1)
        fn = (1 - predictions[idx]).sum()
        return (fn + fp) / (targets.shape[0] * targets.shape[1])
    
    def update_round(self, n_samples):
        X, y = self.sample_contexts_targets(n_samples)
        if X is None or X.size == 0:
            return  # nothing to add

        # increment only when we actually got data
        self.number_of_rounds += 1
        self.list_of_features.append(X)
        self.list_of_targets.append(y)

        last_policy = self.list_of_policies[-1]
        actions, log_p_curr = last_policy.get_actions_and_propensities(X, y)  # log p(a|x) summed over labels
        self.list_of_actions.append(actions)
        self.list_of_losses.append(self.sample_losses(actions, X, y))

        lengths = np.array([len(f) for f in self.list_of_features], dtype=int)
        T = int(lengths.max())
        pad = -np.inf  # log(0)

        if self.matrix_of_propensities is None:
            self.matrix_of_propensities = np.full(
                (self.number_of_rounds, T, self.number_of_rounds), pad, dtype=float
            )
        else:
            a, b, c = self.matrix_of_propensities.shape
            new_mat = np.full(
                (self.number_of_rounds, T, self.number_of_rounds), pad, dtype=float
            )
            new_mat[:a, :b, :c] = self.matrix_of_propensities
            self.matrix_of_propensities = new_mat

        t = self.number_of_rounds - 1
        # fill current policy on current round
        self.matrix_of_propensities[t, :len(X), t] = log_p_curr

        # cross-propensities
        for s in range(self.number_of_rounds - 1):
            # previous rounds under last policy
            lp_prev = last_policy.get_propensity(
                self.list_of_actions[s], self.list_of_features[s], self.list_of_targets[s]
            )
            self.matrix_of_propensities[s, :lengths[s], t] = lp_prev
            # current round under previous policy s
            lp_curr_under_s = self.list_of_policies[s].get_propensity(actions, X, y)
            self.matrix_of_propensities[t, :len(X), s] = lp_curr_under_s




    class LoggingPolicy:
        def __init__(self, env, random_seed):
            self.env = env
            self.pi0, _ = make_baselines_skylines(
                env.X_train,
                env.y_train,
                bonus=None,
                mlp=False,
                n_jobs=4,
                skip_skyline=False,
            )
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = "logarithmic"

        def get_propensity(self, actions, features, targets):
            # Defensive: if features is empty, return empty array
            if features is None or features.size == 0:
                return np.array([])
            """Returns the probability of selecting an action given the state

            Args:
                features (array): array of feature vector
                actions (array): array of actions
            """
            # Implement the logic to select an action based on the state

            sampling_probas = np.array(
                [_[:, 1] for _ in self.pi0.predict_proba(features)]
            ).T
            zero_chosen = np.where(actions == 0)
            propensities = np.array(sampling_probas)
            propensities[zero_chosen] = 1 - sampling_probas[zero_chosen]

            # Avoid log(0) by clipping propensities
            propensities = np.clip(propensities, 1e-12, 1.0)
            log_propensities = np.log(propensities).sum(axis=1)

            if self.propensity_type == "logarithmic":
                return log_propensities
            else:
                propensities = np.exp(log_propensities)
                return propensities

        def get_actions_and_propensities(self, features, targets):
            """Return actions and associated propensities given features and targets

            Args:
                features (_type_): _description_

                targets (_type_): _description_
            """
            # Implement the logic to update the policy based on the experience

            n = features.shape[0]
            k = targets.shape[1]
            probas = np.array([_[:, 1] for _ in self.pi0.predict_proba(features)]).T
            actions = (self.rng.uniform(size=(n, k)) < probas).astype(int)
            zero_chosen = np.where(actions == 0)
            propensities = np.array(probas)
            propensities[zero_chosen] = 1 - probas[zero_chosen]

            # Avoid log(0) by clipping propensities
            propensities = np.clip(propensities, 1e-12, 1.0)
            log_propensities = np.log(propensities).sum(axis=1)

            if self.propensity_type == "logarithmic":
                return actions, log_propensities
            else:
                propensities = np.exp(log_propensities)     
                return actions, propensities
        def online_evaluation(self, env):
            """Evaluate the Logging Policy

            Args:
                env (Environment): Environment object with test data.

            Returns:
                _type_: the baseline evaluation
            """
            contexts, targets = env.X_test, env.y_test
            predictions = self.pi0.predict_proba(contexts)
            predictions = np.array([_[:, 1] for _ in predictions]).T
            idx = np.where(targets == 0)
            fp = predictions[idx].sum()
            idx = np.where(targets == 1)
            fn = (1 - predictions[idx]).sum()
            return (fn + fp) / (targets.shape[0] * targets.shape[1])    
    
    # ... (Add LoggingPolicy and other methods if they are shared) ...

# --- Specific Environment Definitions ---

@register_env("tmc2007")
class tmc2007(BaseMultiLabelEnv):
    def __init__(self, name="tmc2007", test_size=0.25, **kw):
        super(tmc2007, self).__init__(name, test_size, reduce_dim=100, **kw)

@register_env("yeast")
class yeast(BaseMultiLabelEnv):
    def __init__(self, name="yeast", test_size=0.25, **kw):
        super(yeast, self).__init__(name, test_size, reduce_dim=30, **kw)

@register_env("scene")
class scene(BaseMultiLabelEnv):
    def __init__(self, name="scene", test_size=0.25, **kw):
        super(scene, self).__init__(name, test_size, reduce_dim=20, **kw)

@register_env("discrete_synthetic")
class discrete_synthetic(BaseMultiLabelEnv):
    """Synthetic environment for discrete action spaces. Inherits from BaseMultiLabelEnv."""
    def __init__(self, name="discrete_synthetic", test_size=0.25, **kw):
        # Skip file loading in parent, generate synthetic data instead
        super(discrete_synthetic, self).__init__(name=name, test_size=test_size, load_data=False, **kw)

        # Generate data instead of loading it from a file.
        X, y = make_multilabel_classification(
            n_samples=10000,
            n_features=20,
            n_classes=5,
            n_labels=2,
            random_state=40
        )
        # Feature normalization
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)

        # Set the instance attributes, just like the other classes
        self.n_samples = self.X_train.shape[0]
        self.dimension = (self.X_train.shape[1], self.y_train.shape[1])
