"""
Test the model selection and cross-validation classes and functions.
"""
import pytest
import numpy as np

from ..model_selection import BlockShuffleSplit, train_test_split


def test_train_test_split_missing_block_size():
    "Check that an error is raised when missing block size in method=block"
    with pytest.raises(ValueError):
        train_test_split(
            coordinates=(np.arange(5), np.arange(5)), data=np.arange(5), method="block"
        )
