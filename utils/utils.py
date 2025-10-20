import numpy as np
import os
import sys

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier



def display_experiment(random_seed, dataset, name,clipping_parameter,variance_penalty,variance_reg,kl_penalty,kl_reg):

    if variance_penalty:


        print(
            '***', 'EXPERIMENT', name,
            'Random seed: %i' % random_seed,
            'Dataset: %s' % dataset.name,
            'Clipping parameter: %s' % clipping_parameter,
            'Variance penalty: %s' % variance_penalty,
            'Variance regularization: %s' % variance_reg,

            '***',
            file=sys.stderr
        )
    if kl_penalty: 

        print(
            '***', 'EXPERIMENT', name,
            'Random seed: %i' % random_seed,
            'Dataset: %s' % dataset.name,
            'Clipping parameter: %s' % clipping_parameter,
            'KL penalty: %s' % kl_penalty,
            'KL regularization: %s' % kl_reg,

            '***',
            file=sys.stderr
        )
    elif not kl_penalty and not variance_penalty:
        print(
            '***', 'EXPERIMENT', name,
            'Random seed: %i' % random_seed,
            'Dataset: %s' % dataset.name,
            'Clipping parameter: %s' % clipping_parameter,
            'No penalty',
            '***',
            file=sys.stderr
        )

        
        
def get_logging_data(n_samples, dataset):

    actions, contexts, losses, propensities, potentials = dataset.sample_logged_data(n_samples)
    logging_data = actions, contexts, losses, propensities

    return logging_data


def dataset_split(contexts, actions, losses, propensities, random_seed, ratio=0.25):
    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(contexts.shape[0])
    contexts, actions, losses, propensities = contexts[idx], actions[idx], losses[idx], \
                                              propensities[idx]

    size = int(contexts.shape[0] * ratio)
    contexts_train, contexts_valid = contexts[:size, :], contexts[size:, :]
    actions_train, actions_valid = actions[:size], actions[size:]
    losses_train, losses_valid = losses[:size], losses[size:]
    propensities_train, propensities_valid = propensities[:size], propensities[size:]
    #     potentials_train, potentials_valid = potentials[:size], potentials[size:]

    logged_train = actions_train, contexts_train, losses_train, propensities_train
    logged_valid = actions_valid, contexts_valid, losses_valid, propensities_valid

    return logged_train, logged_valid

def update_past_data(data, samples):
    return np.hstack([data, samples])

def online_evaluation(optimized_param, contextual_modelling, dataset, random_seed):

    
    rng = np.random.RandomState(random_seed)
    contexts, potentials = dataset.test_data
    contextual_param = contextual_modelling.get_parameter(optimized_param, contexts)
    contextual_param = contextual_param.reshape(-1,)
    size = contexts.shape[0]
    losses = []

    for i in range(10):
        sampled_actions = rng.normal(contextual_param, dataset.logging_scale, size)
        losses += [dataset.get_losses_from_actions(potentials, sampled_actions)]

    losses_array = np.stack(losses, axis=0)
    return np.mean(losses_array)

def skyline_evaluation(pi_star_determinist, dataset):

    contexts, potentials = dataset.test_data
    predictions = pi_star_determinist.predict(contexts)

    losses = dataset.get_losses_from_actions(potentials, predictions)

    return np.mean(losses)


# def _check_configuration(datasetname, policy_type):
#     if dataset_name == '' and 
#         raise ValueError('Please provide a dataset name and policy type')
#     elif :
    



class OneClassRobustLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        try:
            LogisticRegression.fit(self, X, y, sample_weight)
            return self
        except ValueError as exc:
            print("WARN: training set has only positive examples")
            self.coef_ = np.zeros(((1, X.shape[1])))
            return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 2))
        return LogisticRegression.predict_proba(self, X)
    


def make_baselines_skylines(
    X_train,
    y_train,
    bonus: float = None,
    mlp=False,
    n_jobs=4,
    skip_skyline = False):
    """
        Creates baseline and skyline models for policy evaluation.

        Parameters
        ----------
        X_train : np.ndarray
            Training data for contexts.
        y_train : np.ndarray
            Training data for targets.
        bonus : float, optional
            Adjustment factor for baseline model coefficients (default is None).
        mlp : bool, optional
            Whether to use a multi-layer perceptron for modeling (default is 
            False).
        n_jobs : int, optional
            Number of parallel jobs for model training (default is 4).
        skip_skyline : bool, optional
            Whether to skip the skyline model creation (default is False).

        Returns
        -------
        tuple
            - Baseline model (`pi0`).
            - Skyline model (`pistar`), or None if `skip_skyline` is True.
        """







    if mlp:
        base_clf = MLPClassifier(
            hidden_layer_sizes=(
                500,
                100,
                40,
                10,
            )
        )
        pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
    else:
        base_clf = LogisticRegressionCV(max_iter=10000, n_jobs=6)
        pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
    if skip_skyline:
        pistar = None
    else:
        try:
            pistar.fit(X_train, y_train)
        except ValueError as exc:
            base_clf = OneClassRobustLogisticRegression()
            pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
            pistar.fit(X_train, y_train)

    n_0 = int(len(X_train) * 0.1)
    # print('learning pi0 on', n_0, 'data points')
    X_0 = X_train[-n_0:, :]
    y_0 = y_train[-n_0:, :]
    pi0 = MultiOutputClassifier(OneClassRobustLogisticRegression(), n_jobs=n_jobs)
    pi0.fit(X_0, y_0)

    # making sure every class has non-zero proba and pi0 is not too good
    if bonus is None:
        bonus = 4
    for i in range(y_train.shape[1]):
        pi0.estimators_[i].coef_ = pi0.estimators_[i].coef_ + bonus

    return pi0, pistar
