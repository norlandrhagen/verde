"""
Base class for cross-validators
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from ..coordinates import block_split


class BaseBlockCrossValidator(BaseCrossValidator, metaclass=ABCMeta):
    """
    Base class for spatially blocked cross-validators.

    Instead of splitting the data randomly or in folds, divide the data into spatial
    blocks and split the blocks between train and test. See [Roberts2017]_.
    """

    is_spatial = True

    def __init__(self, n_splits, spacing=None, shape=None, iterations=1):
        self.n_splits = n_splits
        self.spacing = spacing
        self.shape = shape
        self.iterations = iterations

    def get_n_splits(self):
        """
        """
        return self.n_splits

    @abstractmethod
    def _get_cv(self):
        """
        """
        pass

    def split(self, coordinates):
        """
        """
        _, labels = block_split(
            coordinates,
            spacing=self.spacing,
            shape=self.shape,
            region=None,
            adjust="spacing",
        )
        block_ids = np.unique(labels)
        shuffle = self._get_cv().split(block_ids)
        for _ in range(self.n_splits):
            trains, tests, balance = [], [], []
            for _ in range(self.iterations):
                train_id, test_id = next(shuffle)
                trains.append(np.where(np.isin(labels, block_ids[train_id]))[0])
                tests.append(np.where(np.isin(labels, block_ids[test_id]))[0])
                balance.append(
                    abs(trains[-1].size / tests[-1].size - train_id.size / test_id.size)
                )
            best = np.argmin(balance)
            yield trains[best], tests[best]
