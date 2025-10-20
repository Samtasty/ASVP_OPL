import os
import sys

# Libraries
import os
import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.datasets import make_classification,load_iris,load_wine,load_breast_cancer


# Get the current working directory
base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

import numpy as np

#
from environment.env_base import Environment


class Multiclass(Environment):
    
    """Parent class for Data"""

    def __init__(self, name, mode="quadratic",test_size=0.25, **kw):
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
        super(Multiclass, self).__init__(**kw)
        
        self.name = name
        self.test_size = test_size
        self.l = 3
        self.start = 0
        self.mode = mode

        self.logging_scale = 0.5
        self.parameter_scale = 0.01
        self.X_train, self.y_train, self.X_test, self.y_test= load_dataset(
            self.name,seed=self.random_seed, test_size=self.test_size)
        self.X_all, self.y_all = np.vstack([self.X_train, self.X_test]), np.r_[self.y_train, self.y_test]

        self.n_classes = len(np.unique(self.y_all))
        self.dimension = (self.X_train.shape[1], self.n_classes)

        self.param_dimension = self.dimension
        self.n_samples = self.X_train.shape[0]
        self.start_sigma = None
        self.log_sigma=None

    



    def sample_contexts_targets(self, n_samples):

        b = min(self.start + n_samples, self.n_samples)

        X = self.X_train[self.start : b]
        y = self.y_train[self.start : b]

        return X, y


    def sample_losses(self, actions, contexts, targets):
        """
        Computes losses based on actions and target values.

        Parameters
        ----------
        actions : np.ndarray
            Actions taken by the policy.
        contexts : np.ndarray
            Contextual features corresponding to the actions.
        targets : np.ndarray
            Target values to compare against actions.

        Returns
        -------
        np.ndarray
            Loss values for each action-target pair.
        """


        targs=targets
        losses=(1-(actions==targs)).reshape(-1,)
       
       
        return losses
    
    def online_evaluation_star(self, pi_star):
        contexts, targets = self.X_test, self.y_test
        predictions = pi_star.predict(contexts)
        losses=(1-(targets==predictions)).reshape(-1,)
        return np.mean(losses)
        



    class LoggingPolicy:
        def __init__(self, env, random_seed):
            self.env = env
            self.rng = np.random.RandomState(random_seed)
            self.pi0 = make_pi0(
                env.X_train,
                env.y_train,subset_size=0.1)
            
            self.propensity_type = 'normal'

        def get_propensity(self, actions, features, targets):
            """Returns the probability of selecting an action given the state

            Args:
                features (array): array of feature vector
                actions (array): array of actions
            """
            # Implement the logic to select an action based on the state
            n= features.shape[0]

            propensities_array = self.pi0.predict_proba(features)

            propensities=propensities_array[list(range(n)), actions]

            return propensities

        def get_actions_and_propensities(self, features, targets):
            n = features.shape[0]
            probabilities = self.pi0.predict_proba(features)
            k = probabilities.shape[1]
            # use policy-local RNG for reproducible sampling
            actions = np.array([self.rng.choice(k, p=p) for p in probabilities], dtype=int)
            propensities = probabilities[np.arange(n), actions]
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
            losses=(1-(targets==predictions)).reshape(-1,)
            
            return np.mean(losses)




def make_pi0(X, y,subset_size):
    """Create the logging policy

    Args:
        X (array): array of feature vector
        y (array): array of actions

    Returns:
        pi0: the logging policy
    """
    n = X.shape[0]
    n_0=int(n*subset_size)
    
    X_subset=X[:n_0,:]
    y_subset=y[:n_0]
    pi0 = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=10000,
        random_state=42,
    )
    pi0.fit(X_subset, y_subset)
    return pi0



def load_dataset(
    dataset_name,
    seed,
    test_size=0.25,
):
    """Load the dataset

    Args:
        dataset_name (str): name of the dataset
        test_size (float): size of the test set
        seed (int): random seed
        add_intercept (bool): add intercept to the data
        scale (bool): scale the data
        reduce_dim (int): reduce the dimension of the data

    Returns:
        X_train: training data
        y_train: training labels
        X_test: test data
        y_test: test labels
    """

    # Create a synthetic multi-categorical dataset

    if dataset_name == "iris":
        data = load_iris()
        X = data.data
        y = data.target
    if dataset_name == "wine":
        data = load_wine()
        X = data.data
        y = data.target
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target 

    if dataset_name.startswith("synthetic_"):
    
    
        # Assuming dataset_name is in the format 'synthetic_n_features_n_classes_n_informative_flip_y_class_sep'
        print('datsaset_name',dataset_name)
        parts = dataset_name.split('_')[1:]
        
        # Default values
        flip_y = 0.01  # Default noise
        class_sep = 1.0 # Default separation

        if len(parts) < 2:
            raise ValueError("Synthetic dataset name must have at least 2 parts: n_features, n_classes")
        
        n_features = int(parts[0])
        n_classes = int(parts[1])
        
        # Calculate the minimum informative features required by scikit-learn.
        # The default for n_clusters_per_class is 2. The condition is:
        # n_classes * n_clusters_per_class <= 2**n_informative
        min_required_informative = int(np.ceil(np.log2(n_classes * 2)))

        # Check if the user provided a value for n_informative.
        if len(parts) > 2:
            user_provided_n_informative = int(parts[2])
            # Use the user's value, but correct it upwards if it's too small.
            n_informative = max(user_provided_n_informative, min_required_informative)
            if n_informative != user_provided_n_informative:
                print(f"Warning: Provided n_informative ({user_provided_n_informative}) was too small. "
                      f"Corrected to the minimum required value: {n_informative}.")
        else:
            # If not provided, default to half the features, ensuring it meets the minimum.
            default_informative = n_features // 2
            n_informative = max(default_informative, min_required_informative)

        if len(parts) > 3:
            flip_y = float(parts[3])
        if len(parts) > 4:
            class_sep = float(parts[4])

        X, y = make_classification(
            n_samples=10000, 
            n_features=n_features, 
            n_classes=n_classes,
            n_informative=n_informative, 
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=seed,
            n_redundant=0
        )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, y_train, X_test, y_test
