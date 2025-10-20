import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import jax.numpy as jnp

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone


class Environment(object):
    """
    General abstract class for policy learning environments.

    Notes
    -----
    This class is intended as a base for specific environment implementations. 
    It requires subclassing and implementing abstract methods.
    """


    __metaclass__ = ABCMeta

    def __init__(self, random_seed=42):
        """
        Initializes the environment.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility (default is 42).

        Attributes
        ----------
        rng : numpy.RandomState
            Random number generator for reproducibility.
        list_of_features : list
            Stores feature data across rounds.
        list_of_targets : list
            Stores target data across rounds.
        list_of_actions : list
            Stores action data across rounds.
        list_of_losses : list
            Stores observed losses across rounds.
        list_of_policies : list
            Stores policies learned in the environment.
        matrix_of_propensities : ndarray or None
            Stores propensities of actions across rounds. Shape varies with the 
            number of rounds and samples.
        number_of_rounds : int
            Tracks the number of rounds conducted in the environment.
        """

        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.list_of_features = []
        self.list_of_targets = []
        self.list_of_actions = []
        self.list_of_losses = []
        self.list_of_policies = []
        self.matrix_of_propensities = None
        self.number_of_rounds = 0
        self.start = 0

 
    @abstractmethod
    def sample_logging_actions(self, n_samples):
        """
        Samples logging actions for the environment.

        Parameters
        ----------
        n_samples : int
            Number of actions to generate.
        """

        pass

    @abstractmethod
    def sample_contexts_targets(self, n_samples):
        """
        Samples context-target pairs for the environment.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        """

        pass

    @abstractmethod
    def add_policy(self, policy):
        """Add a new policy to the environment

        Args:
            policy (Policy): policy to add
        """
        self.list_of_policies.append(policy)

    @abstractmethod
    def sample_losses(self, actions, contexts, targets):
        """Estimates causal effect on data.

        Parameters
        ----------
        actions : array-like, shape (n_samples,)
            Treatment values, binary.

        contexts : array-like, shape (n_samples, n_features_covariates)
            Covariates (potential confounders).
        targets : array-like, shape (n_samples,)
            Outcome values, continuous.

        """
        pass

    @abstractmethod
    def update_round(self, n_samples):
        """samples and add new round to the environment using the last policy

        Args:
            n_samples (int): number of samples to generate
        """
        self.number_of_rounds += 1
        new_features, new_targets = self.sample_contexts_targets(n_samples)
        self.list_of_features.append(new_features)
        self.list_of_targets.append(new_targets)
        last_policy = self.list_of_policies[-1]
        new_actions, propensities = last_policy.get_actions_and_propensities(
            self.list_of_features[-1], self.list_of_targets[-1]
        )
        self.list_of_actions.append(new_actions)
        new_losses = self.sample_losses(new_actions, new_features, new_targets)
        self.list_of_losses.append(new_losses)
        list_len_samples = jnp.array(
            [len(self.list_of_features[i]) for i in range(len(self.list_of_features))]
        )
        max_nb_sample = max(list_len_samples)
        if self.matrix_of_propensities is None:
            self.matrix_of_propensities = np.zeros(
                (self.number_of_rounds, max_nb_sample, self.number_of_rounds)
            )
            self.matrix_of_propensities[0, :max_nb_sample, 0] = propensities
        else:

            a, b, c = self.matrix_of_propensities.shape
            matrix_of_propensities = np.zeros(
                (self.number_of_rounds, max_nb_sample, self.number_of_rounds)
            )
            matrix_of_propensities[:a, :b, :c] = self.matrix_of_propensities.copy()
            for i in range(self.number_of_rounds):

                #compute the propensities of the actions-contexts pairs of the previous rounds with the last policy
                matrix_of_propensities[
                    i, : list_len_samples[i], self.number_of_rounds - 1
                ] = self.list_of_policies[-1].get_propensity(
                    self.list_of_actions[i],
                    self.list_of_features[i],
                    self.list_of_targets[i],
                )

                #compute the propensities of the actions-contexts pairs of the last round with the previous policies
                matrix_of_propensities[
                    self.number_of_rounds - 1, : list_len_samples[-1], i
                ] = self.list_of_policies[i].get_propensity(
                    self.list_of_actions[-1],
                    self.list_of_features[-1],
                    self.list_of_targets[-1],
                )

            self.matrix_of_propensities = matrix_of_propensities

        self.start=self.start+n_samples
